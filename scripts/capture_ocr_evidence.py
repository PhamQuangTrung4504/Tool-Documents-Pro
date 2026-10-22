#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Capture source-linked OCR crops, recognizer candidates, and style evidence.

This tool is opt-in and writes only to an external output directory. It calls
the production OCR pipeline with an observer callback; the callback receives
copies/views of intermediate images but cannot replace OCR predictions.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import cv2

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("._-") or "document"


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


class EvidenceWriter:
    """Persist observer events and source crops without logging recognized text."""

    def __init__(self, run_dir: Path, source_path: Path, run_id: str) -> None:
        self.run_dir = run_dir
        self.source_path = source_path
        self.run_id = run_id
        self.records: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self._event_number = 0
        self.source_image: Optional[Any] = None
        self._crop_artifacts: dict[int, dict[str, Any]] = {}
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def _save_image(self, relative_path: str, image: Any) -> Optional[str]:
        if image is None or getattr(image, "size", 0) == 0:
            return None
        path = self.run_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), image):
            raise OSError(f"Could not write diagnostic image {relative_path}")
        return relative_path.replace("\\", "/")

    def _save_style_regions(self, event: dict[str, Any]) -> None:
        if self.source_image is None:
            return
        block = int(event.get("block_index", 0))
        target_artifacts = []
        comparisons = []
        measurements = event.get("bold_evidence", {}).get("measurements", [])
        for measurement in measurements:
            target_index = int(measurement["box_index"])
            target_dir = f"style/block_{block:04d}/box_{target_index:04d}"
            target_artifacts.append({
                "box_index": target_index,
                "bbox_image": self._save_image(f"{target_dir}/bbox.png", self._crop(measurement["bbox"])),
                "measurement_image": self._save_image(
                    f"{target_dir}/measurement_region.png", self._crop(measurement["measurement_bbox"])
                ),
            })
            for peer in measurement.get("peers", []):
                peer_index = int(peer["box_index"])
                peer_dir = f"style/block_{block:04d}/box_{target_index:04d}/peer_{peer_index:04d}"
                comparisons.append({
                    "target_box_index": target_index,
                    "peer_box_index": peer_index,
                    "bbox_image": self._save_image(f"{peer_dir}/bbox.png", self._crop(peer["bbox"])),
                    "measurement_image": self._save_image(
                        f"{peer_dir}/measurement_region.png", self._crop(peer["measurement_bbox"])
                    ),
                })
        event["target_artifacts"] = target_artifacts
        event["comparison_artifacts"] = comparisons

    def _crop(self, bbox: list[int]) -> Any:
        left, top, right, bottom = [int(value) for value in bbox]
        height, width = self.source_image.shape[:2]
        left, right = max(0, left), min(width, right)
        top, bottom = max(0, top), min(height, bottom)
        return self.source_image[top:bottom, left:right].copy()

    def observe(self, event: dict[str, Any], image: Any = None) -> None:
        """Observer contract used by OCREngine; failures are retained in report."""
        record = dict(event)
        try:
            self._event_number += 1
            event_name = str(record.get("event", "event"))
            crop_id = record.get("crop_id")
            subcrop = record.get("subcrop_index")
            if event_name == "source_image" and image is not None:
                self.source_image = image.copy()
            if event_name == "style_analysis":
                self._save_style_regions(record)
            elif image is not None:
                suffix = f"crop_{int(crop_id):04d}" if crop_id is not None else "page"
                if subcrop is not None:
                    suffix += f"_sub_{int(subcrop):03d}"
                filename = f"images/{self._event_number:06d}_{_safe_name(event_name)}_{suffix}.png"
                record["image_file"] = self._save_image(filename, image)
                if crop_id is not None:
                    artifacts = self._crop_artifacts.setdefault(int(crop_id), {})
                    if event_name == "vietocr_recognizer_input":
                        if subcrop is None:
                            artifacts.setdefault("vietocr_inputs", []).append(record["image_file"])
                        else:
                            artifacts.setdefault("vietocr_subcrop_inputs", {})[str(subcrop)] = record["image_file"]
                    elif event_name in {"source_rectified_crop", "preprocessed_rectified_crop",
                                        "vietocr_border_cleaned_crop", "paddle_polygon_crop_preview"}:
                        artifacts[event_name] = record["image_file"]
            if event_name in {"recognition_candidate", "recognition_subcandidate"} and crop_id is not None:
                record["crop_artifact_files"] = self._crop_artifacts.get(int(crop_id), {}).copy()
                if subcrop is not None:
                    record["recognizer_input_image"] = record["crop_artifact_files"].get(
                        "vietocr_subcrop_inputs", {}
                    ).get(str(subcrop))
                record["detector_page_image"] = next((
                    prior.get("image_file") for prior in reversed(self.records)
                    if prior.get("event") == "paddle_detector_input"
                ), None)
            record["event_number"] = self._event_number
            record["run_id"] = self.run_id
            record["source_file"] = self.source_path.name
            self.records.append(record)
        except Exception as exc:
            self.errors.append(f"event={event.get('event', 'unknown')}: {type(exc).__name__}: {exc}")

    def write_report(self, result: Any = None, error: Optional[BaseException] = None) -> None:
        report: dict[str, Any] = {
            "schema_version": 1,
            "run_id": self.run_id,
            "source_file": self.source_path.name,
            "source_path": str(self.source_path.resolve()),
            "status": "success" if error is None else "failed",
            "events": self.records,
            "observer_errors": self.errors,
        }
        if result is not None:
            report["ocr_result_snapshot"] = {
                "full_text": result.full_text,
                "raw_full_text": result.raw_full_text,
                "elapsed_seconds": result.elapsed_seconds,
                "average_confidence": result.average_confidence,
                "boxes": [
                    {"index": index, "bbox": box.bbox, "text": box.text,
                     "raw_text": box.raw_text, "confidence": box.confidence,
                     "recognition_metadata": box.metadata}
                    for index, box in enumerate(result.boxes)
                ],
                "tables": [
                    {"bbox": table.bbox, "rows": table.num_rows,
                     "columns": table.num_cols, "grid": table.grid}
                    for table in result.tables
                ],
                "document_blocks": [
                    {"kind": block.kind, "order": block.order, "text": block.text,
                     "box_indices": block.box_indices, "bbox": block.bbox,
                     "metadata": block.metadata}
                    for block in result.document_blocks
                ],
            }
        if error is not None:
            report["error"] = {"type": type(error).__name__, "message": str(error)}
        (self.run_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Xuất crop OCR, ứng viên recognizer và bằng chứng style vào ZIP ngoài repository."
    )
    parser.add_argument("--images", nargs="+", required=True, type=Path,
                        help="Một hoặc nhiều ảnh nguồn; ví dụ truyền ba đường dẫn trong một lệnh.")
    parser.add_argument("--output-dir", type=Path,
                        help="Thư mục ngoài repository. Mặc định tạo trong thư mục TEMP hệ thống.")
    parser.add_argument("--language", default="vi", help="Mã ngôn ngữ OCR (mặc định: vi).")
    parser.add_argument("--no-preprocess", action="store_true",
                        help="Tắt preprocessing; mặc định khớp luồng production.")
    parser.add_argument("--no-denoise", action="store_true",
                        help="Tắt denoise khi preprocessing đang bật.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = args.output_dir or (Path(tempfile.gettempdir()) / f"tool_documents_pro_ocr_evidence_{timestamp}")
    output_root = output_root.expanduser().resolve()
    try:
        output_root.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError:
        pass
    else:
        print("ERROR: output-dir phải nằm ngoài repository.", file=sys.stderr)
        return 2
    output_root.mkdir(parents=True, exist_ok=True)

    from app.core.ocr_engine import OCREngine

    engine = OCREngine(lang=args.language)
    run_summaries = []
    used_names: dict[str, int] = {}
    for image_path in args.images:
        image_path = image_path.expanduser().resolve()
        base = _safe_name(image_path.stem)
        used_names[base] = used_names.get(base, 0) + 1
        label = f"{base}_{used_names[base]:02d}"
        run_id = f"{timestamp}_{label}"
        writer = EvidenceWriter(output_root / label, image_path, run_id)
        try:
            result = engine.recognize(
                image_path,
                preprocess=not args.no_preprocess,
                denoise=not args.no_denoise,
                include_diagnostics=False,
                diagnostic_observer=writer.observe,
            )
            writer.write_report(result=result)
            run_summaries.append({"source_file": image_path.name, "run_id": run_id,
                                  "status": "success", "report": f"{label}/report.json"})
        except Exception as exc:
            writer.write_report(error=exc)
            run_summaries.append({"source_file": image_path.name, "run_id": run_id,
                                  "status": "failed", "error": type(exc).__name__,
                                  "report": f"{label}/report.json"})
            print(f"FAILED: {image_path.name}: {type(exc).__name__}: {exc}", file=sys.stderr)

    index = {
        "schema_version": 1,
        "created_utc": timestamp,
        "note": "Crop IDs identify detections within one run only; use run_id with crop_id.",
        "paddleocr_input_limit": (
            "PaddleOCR predict does not expose its internal recognition crop/tensor. "
            "The bundle contains the exact detector page input, polygon, candidate and an explicitly labeled crop preview."
        ),
        "runs": run_summaries,
    }
    (output_root / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    archive = output_root.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(output_root.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(output_root.parent))
    print(f"Evidence ZIP: {archive}")
    print(f"Run status: {sum(run['status'] == 'success' for run in run_summaries)} success, "
          f"{sum(run['status'] == 'failed' for run in run_summaries)} failed")
    return 0 if all(run["status"] == "success" for run in run_summaries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
