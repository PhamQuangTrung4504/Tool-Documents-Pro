#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Adapter QThread cho nhánh OCR tiến trình thử nghiệm, mặc định không dùng."""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import tempfile
import threading
import time
import uuid
import ctypes
from ctypes import wintypes
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from loguru import logger

from app.core.ocr_engine import OCRCancelledError
from app.core.ocr_process_worker import ocr_process_main, restore_ocr_result


class OCRProcessClient:
    """Quản lý một worker process dùng lại qua nhiều job tuần tự."""

    def __init__(self, worker_target=ocr_process_main, job_timeout_seconds: float = 900.0,
                 cancel_grace_seconds: float = 60.0) -> None:
        self._context = mp.get_context("spawn")
        self._worker_target = worker_target
        self._job_timeout_seconds = max(1.0, float(job_timeout_seconds))
        self._cancel_grace_seconds = max(1.0, float(cancel_grace_seconds))
        self._process = None
        self._connection = None
        self._cancel_event = None
        self._lock = threading.Lock()
        self._active_job_id: Optional[str] = None
        self._engine_config: Optional[dict] = None
        self.worker_ready_ms: Optional[float] = None

    def _ensure_started(self, engine_config: dict) -> None:
        if self._process is not None and self._process.is_alive() and self._engine_config != engine_config:
            try:
                if self._connection is not None:
                    self._connection.send(json.dumps({"type": "shutdown"}))
            except (OSError, EOFError, BrokenPipeError):
                pass
            self._process.join(timeout=2.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1.0)
            self._close_handles()
        if self._process is not None and self._process.is_alive():
            return
        self._close_handles()
        parent, child = self._context.Pipe(duplex=True)
        cancel_event = self._context.Event()
        process = self._context.Process(
            target=self._worker_target, args=(child, cancel_event),
            name="ToolDocumentsPro-OCR", daemon=False,
        )
        start = time.perf_counter()
        process.start()
        child.close()
        self._process, self._connection, self._cancel_event = process, parent, cancel_event
        if not parent.poll(60.0):
            self._discard_process()
            raise TimeoutError("Tiến trình OCR không sẵn sàng trong 60 giây.")
        ready = json.loads(parent.recv())
        if ready.get("type") != "ready" or int(ready.get("pid", -1)) != process.pid:
            self._discard_process()
            raise RuntimeError("Handshake tiến trình OCR không hợp lệ.")
        self._engine_config = dict(engine_config)
        self.worker_ready_ms = (time.perf_counter() - start) * 1000
        logger.info("TDP_OCR_PROCESS worker_ready_ms={:.1f} pid={}", self.worker_ready_ms, process.pid)

    @property
    def is_running(self) -> bool:
        process = self._process
        if process is None:
            return False
        try:
            return process.is_alive()
        except (ValueError, OSError):
            # shutdown() có thể đóng handle trên luồng nền đồng thời với lần
            # thăm dò trạng thái từ GUI.
            return False

    @staticmethod
    def _process_memory_bytes(process) -> Optional[int]:
        if os.name != "nt" or process is None or process.pid is None:
            return None
        class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                        ("PrivateUsage", ctypes.c_size_t)]
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, process.pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return None
        try:
            counters = PROCESS_MEMORY_COUNTERS_EX()
            counters.cb = ctypes.sizeof(counters)
            ok = ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
            return int(counters.PrivateUsage) if ok else None
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)

    def run_job(self, image: np.ndarray, generation: int, worker, engine_config: dict) -> None:
        if not self._lock.acquire(blocking=False):
            worker.result_ready.emit(RuntimeError("Tiến trình OCR đang xử lý job khác."))
            return
        image_path = None
        job_id = f"{generation}-{uuid.uuid4().hex}"
        try:
            self._active_job_id = job_id
            self._ensure_started(engine_config)
            if np.asarray(image).nbytes > 128 * 1024 * 1024:
                raise ValueError("Ảnh vượt giới hạn thử nghiệm IPC 128 MiB.")
            with tempfile.NamedTemporaryFile(prefix="tdp-ocr-", suffix=".npy", delete=False) as handle:
                image_path = Path(handle.name)
            np.save(image_path, np.asarray(image), allow_pickle=False)
            self._cancel_event.clear()
            self._connection.send(json.dumps({
                "type": "run", "job_id": job_id, "image_path": str(image_path),
                "engine_config": engine_config,
            }))
            terminal = False
            job_started = time.monotonic()
            timeout_requested_at = None
            while not terminal:
                if worker._cancel_event.is_set():
                    self._cancel_event.set()
                if timeout_requested_at is None and time.monotonic() - job_started > self._job_timeout_seconds:
                    timeout_requested_at = time.monotonic()
                    self._cancel_event.set()
                    worker.progress.emit("Đã quá thời hạn thử nghiệm; đang yêu cầu dừng ở ranh giới an toàn …")
                elif timeout_requested_at is not None and time.monotonic() - timeout_requested_at > self._cancel_grace_seconds:
                    raise TimeoutError("OCR subprocess không dừng sau khi hết thời hạn và đã bị đóng.")
                if self._connection.poll(0.1):
                    message = json.loads(self._connection.recv())
                    if message.get("job_id") != job_id:
                        continue
                    kind = message.get("type")
                    if kind == "progress":
                        worker.progress.emit(str(message.get("message", "")))
                    elif kind == "progress_detail":
                        worker.progress_detail.emit(int(message.get("done", 0)), int(message.get("total", 0)))
                    elif kind == "stage":
                        worker.stage_event.emit(generation, str(message["stage"]),
                                                str(message["transition"]), int(message["timestamp_ns"]))
                    elif kind == "result":
                        result = restore_ocr_result(message["result"])
                        private_bytes = self._process_memory_bytes(self._process)
                        logger.info("TDP_OCR_PROCESS result job_id={} elapsed_s={:.3f} child_private_mb={}",
                                    job_id, float(message.get("elapsed_seconds", 0.0)),
                                    round(private_bytes / (1024 * 1024), 1) if private_bytes is not None else "unknown")
                        logger.info("TDP_OCR_PROCESS result generation={} boxes={} tables={} result_sha256={}",
                                    generation, len(result.boxes), len(result.tables), message.get("result_sha256", "unknown"))
                        worker.result_ready.emit(result)
                        terminal = True
                    elif kind == "error":
                        if timeout_requested_at is not None:
                            raise TimeoutError("OCR subprocess hết thời hạn xử lý.")
                        if message.get("cancelled"):
                            error = OCRCancelledError(message.get("error", "OCR đã bị hủy."))
                        else:
                            error = RuntimeError(f"{message.get('error_type', 'OCRProcessError')}: {message.get('error', '')}")
                        worker.result_ready.emit(error)
                        terminal = True
                    else:
                        raise RuntimeError("Tiến trình OCR gửi loại thông điệp không hợp lệ.")
                elif self._process is None or not self._process.is_alive():
                    raise RuntimeError("Tiến trình OCR đã dừng bất thường.")
        except TimeoutError as exc:
            worker.result_ready.emit(exc)
            self._discard_process()
        except (EOFError, OSError, BrokenPipeError) as exc:
            logger.warning("TDP_OCR_PROCESS IPC disconnected job_id={} error_type={}",
                           job_id, type(exc).__name__)
            process = self._process
            try:
                crashed = process is None or not process.is_alive()
            except (ValueError, OSError):
                crashed = True
            message = ("Tiến trình OCR đã dừng bất thường; kết nối IPC bị đóng."
                       if crashed else "Kết nối IPC với tiến trình OCR bị đóng hoặc mất.")
            worker.result_ready.emit(RuntimeError(message))
            self._discard_process()
        except BaseException as exc:
            worker.result_ready.emit(exc)
            self._discard_process()
        finally:
            self._active_job_id = None
            if image_path is not None:
                try:
                    image_path.unlink(missing_ok=True)
                except OSError:
                    pass
            self._lock.release()

    def request_cancel(self) -> None:
        if self._cancel_event is not None and self._active_job_id is not None:
            self._cancel_event.set()

    def shutdown(self, timeout: float = 2.0) -> None:
        """Dừng tiến trình; gọi từ luồng nền, không từ GUI thread."""
        with self._lock:
            process = self._process
            connection = self._connection
            if process is None:
                return
            if process.is_alive() and connection is not None:
                try:
                    if self._active_job_id is not None and self._cancel_event is not None:
                        self._cancel_event.set()
                    connection.send(json.dumps({"type": "shutdown"}))
                except (OSError, EOFError, BrokenPipeError):
                    pass
            process.join(timeout=max(0.0, timeout))
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
            self._close_handles()

    def _discard_process(self) -> None:
        process = self._process
        if process is not None:
            try:
                if process.is_alive():
                    process.terminate()
                # join() cũng cần được gọi cho tiến trình đã chết tự nhiên để
                # đồng bộ trạng thái và thu hồi handle trước khi đóng đối tượng.
                process.join(timeout=1.0)
            except (ValueError, OSError):
                pass
        self._close_handles()

    def _close_handles(self) -> None:
        if self._connection is not None:
            try:
                self._connection.close()
            except OSError:
                pass
        if self._process is not None:
            try:
                self._process.close()
            except (ValueError, OSError):
                pass
        self._connection = self._process = self._cancel_event = None
        self._engine_config = None


class OCRProcessWorker(QThread):
    """Giao diện signal tương thích OCRWorker để giữ nguyên MainWindow."""

    result_ready = pyqtSignal(object)
    progress = pyqtSignal(str)
    progress_detail = pyqtSignal(int, int)
    stage_event = pyqtSignal(int, str, str, object)

    def __init__(self, client: OCRProcessClient, image: np.ndarray, engine_config: dict,
                 parent=None, generation: int = 0) -> None:
        super().__init__(parent)
        self._client = client
        self._image = image
        self._generation = generation
        self._engine_config = dict(engine_config)
        self._cancel_event = threading.Event()

    def request_cancel(self) -> None:
        self._cancel_event.set()
        self._client.request_cancel()

    def run(self) -> None:
        self.progress.emit("Đang chạy OCR trong tiến trình thử nghiệm …")
        self._client.run_job(self._image, self._generation, self, self._engine_config)

