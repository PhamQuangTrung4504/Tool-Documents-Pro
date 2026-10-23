#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tiến trình OCR thử nghiệm, dùng lại OCREngine nguyên trạng.

Module này chỉ được khởi chạy khi TDP_OCR_PROCESS=1. IPC chỉ mang chuỗi JSON
qua multiprocessing.Connection; dữ liệu ảnh đi qua tệp .npy tạm allow_pickle=False.
"""
from __future__ import annotations

import json
import hashlib
import os
import time
from dataclasses import fields, is_dataclass
from enum import Enum
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, Dict

import numpy as np


def _json_value(value: Any) -> Any:
    if is_dataclass(value):
        return {field.name: _json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("OCRResult chỉ hỗ trợ khóa metadata dạng chuỗi qua IPC.")
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Không thể tuần tự hóa kiểu OCRResult qua IPC: {type(value).__name__}")


def ocr_result_fingerprint(result: Any) -> str:
    payload = _json_value(result)
    if isinstance(payload, dict):
        payload.pop("elapsed_seconds", None)
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _restore_result(payload: Dict[str, Any]):
    # Imports happen in the calling thread/process, never in the UI module import path.
    from app.core.models import DocumentBlock, OCRBox, OCRResult, OCRTable, ParallelHeader, TableCell

    def box(data):
        return OCRBox(**data)

    tables = []
    for data in payload.get("tables", []):
        data = dict(data)
        data["row_boxes"] = [[box(item) for item in row] for row in data.get("row_boxes", [])]
        data["cells"] = [TableCell(**{
            **cell,
            "boxes": [box(item) for item in cell.get("boxes", [])],
        }) for cell in data.get("cells", [])]
        tables.append(OCRTable(**data))
    blocks = [DocumentBlock(**item) for item in payload.get("document_blocks", [])]
    header_data = payload.get("parallel_header")
    header = None
    if header_data is not None:
        header = ParallelHeader(
            left_lines=header_data.get("left_lines", []),
            right_lines=header_data.get("right_lines", []),
            left_boxes=[box(item) for item in header_data.get("left_boxes", [])],
            right_boxes=[box(item) for item in header_data.get("right_boxes", [])],
        )
    values = dict(payload)
    values["boxes"] = [box(item) for item in payload.get("boxes", [])]
    values["tables"] = tables
    values["document_blocks"] = blocks
    values["parallel_header"] = header
    return OCRResult(**values)


def ocr_process_main(connection: Connection, cancel_event) -> None:
    """Vòng đời tiến trình bền vững; mỗi yêu cầu dùng cùng singleton model."""
    engine = None
    try:
        connection.send(json.dumps({"type": "ready", "pid": os.getpid()}))
        while True:
            try:
                request = json.loads(connection.recv())
            except (EOFError, OSError):
                return
            if not isinstance(request, dict) or request.get("type") not in {"run", "shutdown"}:
                connection.send(json.dumps({"type": "error", "job_id": None,
                                            "error": "Yêu cầu IPC không hợp lệ."}))
                continue
            if request["type"] == "shutdown":
                return
            job_id = request.get("job_id")
            image_path = Path(request.get("image_path", ""))
            if not isinstance(job_id, str) or not image_path.is_file():
                connection.send(json.dumps({"type": "error", "job_id": job_id,
                                            "error": "Yêu cầu OCR thiếu mã job hoặc ảnh hợp lệ."}))
                continue
            cancel_event.clear()
            started = time.perf_counter()
            try:
                if engine is None:
                    # Windows spawn khởi chạy interpreter mới: thứ tự nạp native
                    # phải được thiết lập lại ngay trong tiến trình worker.
                    from app.native_runtime import bootstrap_torch_before_paddle
                    bootstrap_torch_before_paddle()
                    import cv2
                    if os.environ.get("TDP_UI_PERF") == "1" and os.environ.get("TDP_OPENCV_THREADS"):
                        try:
                            requested_threads = int(os.environ["TDP_OPENCV_THREADS"])
                            if not 1 <= requested_threads <= 256:
                                raise ValueError
                            cv2.setNumThreads(requested_threads)
                        except ValueError:
                            pass
                    from app.core.ocr_engine import OCREngine
                    engine = OCREngine(**request.get("engine_config", {}))
                image = np.load(image_path, allow_pickle=False)
                connection.send(json.dumps({"type": "stage", "job_id": job_id,
                                            "stage": "model_loading", "transition": "start",
                                            "timestamp_ns": time.perf_counter_ns()}))
                engine._ensure_loaded()
                connection.send(json.dumps({"type": "stage", "job_id": job_id,
                                            "stage": "model_loading", "transition": "end",
                                            "timestamp_ns": time.perf_counter_ns()}))

                def send(event: Dict[str, Any]) -> None:
                    connection.send(json.dumps({**event, "job_id": job_id}, ensure_ascii=False))

                result = engine.recognize(
                    image, preprocess=True, denoise=True, cancel_event=cancel_event,
                    progress_callback=lambda message: send({"type": "progress", "message": message}),
                    progress_detail_callback=lambda done, total: send({
                        "type": "progress_detail", "done": int(done), "total": int(total),
                    }),
                    stage_callback=lambda stage, transition, timestamp_ns: send({
                        "type": "stage", "stage": stage, "transition": transition,
                        "timestamp_ns": int(timestamp_ns),
                    }),
                )
                result_message = json.dumps({"type": "result", "job_id": job_id,
                                             "elapsed_seconds": time.perf_counter() - started,
                                             "result_sha256": ocr_result_fingerprint(result),
                                             "result": _json_value(result)}, ensure_ascii=False)
                if len(result_message.encode("utf-8")) > 64 * 1024 * 1024:
                    raise ValueError("Kết quả OCR vượt giới hạn IPC 64 MiB.")
                connection.send(result_message)
            except BaseException as exc:
                connection.send(json.dumps({"type": "error", "job_id": job_id,
                                            "cancelled": type(exc).__name__ == "OCRCancelledError",
                                            "error_type": type(exc).__name__,
                                            "error": str(exc)[:2000]}, ensure_ascii=False))
    finally:
        try:
            connection.close()
        except OSError:
            pass


def restore_ocr_result(payload: Dict[str, Any]):
    return _restore_result(payload)

