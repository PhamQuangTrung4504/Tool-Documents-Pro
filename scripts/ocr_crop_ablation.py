#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Thử nghiệm crop OCR độc lập, không thay đổi kết quả của luồng sản xuất."""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
VARIANT_NAMES = ("current_crop", "skip_border_cleaning", "safe_padding", "aspect_preserving_resize")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-") or "document"


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if image is None or image.size == 0 or not cv2.imwrite(str(path), image):
        raise OSError(f"Không lưu được ảnh crop: {path.name}")


def build_crop_variants(raw_crop: np.ndarray, cleaned_crop: np.ndarray,
                        resize_scale: float = 1.5) -> dict[str, np.ndarray]:
    """Tạo bốn biến thể; không sửa ảnh đầu vào hay gọi recognizer."""
    return _build_crop_variants_timed(raw_crop, cleaned_crop, resize_scale)[0]


def _build_crop_variants_timed(raw_crop: np.ndarray, cleaned_crop: np.ndarray,
                               resize_scale: float) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    from app.core.vision_preprocess import adaptive_safe_crop_padding, pad_small_crop_for_vietocr

    variants: dict[str, np.ndarray] = {}
    timings: dict[str, float] = {}
    transforms = {
        "current_crop": lambda: pad_small_crop_for_vietocr(cleaned_crop),
        "skip_border_cleaning": lambda: pad_small_crop_for_vietocr(raw_crop),
        "safe_padding": lambda: pad_small_crop_for_vietocr(adaptive_safe_crop_padding(cleaned_crop)),
        "aspect_preserving_resize": lambda: _resize_aspect_preserving(cleaned_crop, resize_scale),
    }
    for name, transform in transforms.items():
        started = time.perf_counter()
        variants[name] = transform()
        timings[name] = time.perf_counter() - started
    return variants, timings


def _resize_aspect_preserving(crop: np.ndarray, scale: float) -> np.ndarray:
    height, width = crop.shape[:2]
    return cv2.resize(
        crop, (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_CUBIC if scale >= 1 else cv2.INTER_AREA,
    )


class AblationObserver:
    """Thu ảnh trung gian từ observer production, không sửa OCRResult."""

    def __init__(self) -> None:
        self.crops: dict[int, dict[str, Any]] = {}

    def observe(self, event: dict[str, Any], image: np.ndarray | None = None) -> None:
        crop_id = event.get("crop_id")
        if crop_id is None:
            return
        crop = self.crops.setdefault(int(crop_id), {"events": [], "images": {}})
        crop["events"].append(dict(event))
        name = event.get("event")
        if image is not None and name in {
            "preprocessed_rectified_crop", "vietocr_border_cleaned_crop",
            "vietocr_recognizer_input", "vietocr_subcrop_before_padding",
        }:
            crop["images"].setdefault(name, []).append(image.copy())


def _predict(model: Any, image_bgr: np.ndarray) -> tuple[str, float | None]:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = model.predict(Image.fromarray(rgb), return_prob=True)
    if isinstance(result, tuple):
        text = str(result[0] or "")
        try:
            score = float(result[1]) if result[1] is not None else None
        except (TypeError, ValueError):
            score = None
        return text, score
    return str(result), None


def _bbox_iou(left: list[float], right: list[float]) -> float:
    x1, y1, x2, y2 = left
    a1, b1, a2, b2 = right
    intersection = max(0.0, min(x2, a2) - max(x1, a1)) * max(0.0, min(y2, b2) - max(y1, b1))
    area_left = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_right = max(0.0, a2 - a1) * max(0.0, b2 - b1)
    union = area_left + area_right - intersection
    return intersection / union if union else 0.0


def _verified_region_label(bbox: Any, labels: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str]:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None, "bbox_unavailable"
    matches = [label for label in labels if _bbox_iou(bbox, label["bbox_source"]) >= 0.8]
    if len(matches) == 1:
        return matches[0], "matched"
    return None, "ambiguous_or_unmatched" if matches else "unmatched"


def run_image(engine: Any, image_path: Path, output_dir: Path, resize_scale: float,
              region_labels: list[dict[str, Any]] | None = None,
              preprocess: bool = True, denoise: bool = True) -> dict[str, Any]:
    observer = AblationObserver()
    started = time.perf_counter()
    production_result = engine.recognize(image_path, preprocess=preprocess, denoise=denoise,
                                         diagnostic_observer=observer.observe)
    production_seconds = time.perf_counter() - started
    model = getattr(engine, "_vietocr", None)
    report: dict[str, Any] = {
        "source_image": str(image_path.resolve()),
        "production_ocr_seconds": production_seconds,
        "production_result_unchanged": True,
        "ground_truth_status": "region_labels_available" if region_labels else "unlabeled",
        "note": "Khác biệt chuỗi không xác định độ chính xác; chỉ tính chỉ số khi nhãn vùng đã xác minh và khớp duy nhất bằng hình học.",
        "regions": [],
    }
    if model is None:
        report["status"] = "vietocr_unavailable"
        return report

    for crop_id in sorted(observer.crops):
        capture = observer.crops[crop_id]
        images = capture["images"]
        raw_list = images.get("preprocessed_rectified_crop", [])
        clean_list = images.get("vietocr_border_cleaned_crop", [])
        exact_inputs = images.get("vietocr_recognizer_input", [])
        events = capture["events"]
        primary = next((e for e in reversed(events) if e.get("event") == "recognition_candidate"
                        and str(e.get("recognizer", "")).startswith("vietocr:")), None)
        split_events = [e for e in events if e.get("event") == "recognition_subcandidate"]
        if not raw_list or not clean_list or not exact_inputs or primary is None:
            continue
        # Dòng bị tách thành sub-crop cần ablation riêng cho từng sub-crop; không trộn
        # chúng với crop gốc vì khi đó phép so sánh không còn cùng vùng ảnh.
        if split_events or len(exact_inputs) != 1:
            report["regions"].append({
                "crop_id": crop_id, "status": "skipped_split_region",
                "reason": "Luồng production đã tách dòng thành nhiều sub-crop; cần ablation riêng để bảo toàn cùng vùng.",
                "polygon_source": primary.get("polygon_source"),
                "bbox_source": primary.get("bbox_source"),
            })
            continue

        region_dir = output_dir / "regions" / f"crop_{crop_id:04d}"
        raw_crop, clean_crop = raw_list[0], clean_list[0]
        production_input = exact_inputs[0]
        _write_image(region_dir / "source_rectified_before_cleaning.png", raw_crop)
        _write_image(region_dir / "after_border_cleaning.png", clean_crop)
        variants, transform_times = _build_crop_variants_timed(raw_crop, clean_crop, resize_scale)
        variant_rows = []
        for variant_name, variant_image in variants.items():
            _write_image(region_dir / f"{variant_name}.png", variant_image)
            t0 = time.perf_counter()
            text, score = _predict(model, variant_image)
            inference_seconds = time.perf_counter() - t0
            variant_rows.append({
                "name": variant_name,
                "text": text,
                "confidence": score,
                "confidence_is_cross_model_comparable": False,
                "inference_seconds": inference_seconds,
                "transform_seconds": transform_times[variant_name],
                "image": f"regions/crop_{crop_id:04d}/{variant_name}.png",
                "dimensions": [int(variant_image.shape[1]), int(variant_image.shape[0])],
            })
        production_input_text = str(primary.get("text", ""))
        current_text = next((row["text"] for row in variant_rows if row["name"] == "current_crop"), "")
        region_report = {
            "crop_id": crop_id,
            "status": "completed",
            "polygon_source": primary.get("polygon_source"),
            "bbox_source": primary.get("bbox_source"),
            "production_vietocr_text": production_input_text,
            "repeated_current_crop_text": current_text,
            "repeated_current_matches_production": current_text == production_input_text,
            "current_crop_matches_observed_input_pixels": bool(
                variants["current_crop"].shape == production_input.shape
                and np.array_equal(variants["current_crop"], production_input)
            ),
            "variants_differ_from_current": [row["name"] for row in variant_rows
                                               if row["text"] != current_text],
            "variants": variant_rows,
        }
        label, label_status = _verified_region_label(
            primary.get("bbox_source"), region_labels or []
        )
        region_report["ground_truth_status"] = label_status if region_labels else "unlabeled"
        if label is not None:
            from app.core.recognition_diagnostics import compute_cer, compute_wer
            region_report["ground_truth"] = label["text"]
            region_report["metrics"] = {
                row["name"]: {"cer": compute_cer(label["text"], row["text"]),
                              "wer": compute_wer(label["text"], row["text"])}
                for row in variant_rows
            }
        report["regions"].append(region_report)
    completed = [r for r in report["regions"] if r["status"] == "completed"]
    report["status"] = "success"
    report["summary"] = {
        "regions_captured": len(observer.crops),
        "regions_compared": len(completed),
        "regions_skipped": len(report["regions"]) - len(completed),
        "regions_with_text_differences": sum(bool(r["variants_differ_from_current"]) for r in completed),
        "variant_inference_seconds": {
            row["name"]: sum(v["inference_seconds"] for region in completed for v in region["variants"]
                             if v["name"] == row["name"])
            for row in (completed[0]["variants"] if completed else [])
        },
        "variant_transform_seconds": {
            row["name"]: sum(v["transform_seconds"] for region in completed for v in region["variants"]
                             if v["name"] == row["name"])
            for row in (completed[0]["variants"] if completed else [])
        },
        "labeled_region_count": sum(r.get("ground_truth_status") == "matched" for r in completed),
        "macro_metrics_by_variant": {
            name: {
                metric: sum(r["metrics"][name][metric] for r in completed if name in r.get("metrics", {})) /
                        sum(1 for r in completed if name in r.get("metrics", {}))
                for metric in ("cer", "wer")
            }
            for name in VARIANT_NAMES
            if any(name in r.get("metrics", {}) for r in completed)
        },
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", nargs="+", required=True, type=Path, help="Đường dẫn ảnh bất kỳ.")
    parser.add_argument("--output-dir", type=Path, help="Thư mục ngoài repository; mặc định là TEMP.")
    parser.add_argument("--resize-scale", type=float, default=1.5,
                        help="Hệ số phóng/thu đồng đều cho biến thể giữ tỷ lệ (mặc định 1.5).")
    parser.add_argument("--region-ground-truth", type=Path,
                        help="JSON nhãn vùng đã xác minh; cần bbox_source, text, verified, reviewed_by, reviewed_at, source_note.")
    parser.add_argument("--no-preprocess", action="store_true")
    parser.add_argument("--no-denoise", action="store_true")
    args = parser.parse_args()
    if args.resize_scale <= 0:
        parser.error("--resize-scale phải lớn hơn 0")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = (args.output_dir or Path(tempfile.gettempdir()) / f"tdp_crop_ablation_{stamp}").expanduser().resolve()
    try:
        root.relative_to(ROOT.resolve())
    except ValueError:
        pass
    else:
        parser.error("--output-dir phải nằm ngoài repository")
    root.mkdir(parents=True, exist_ok=True)

    labels_by_image: dict[str, list[dict[str, Any]]] = {}
    if args.region_ground_truth:
        annotation = json.loads(args.region_ground_truth.expanduser().read_text(encoding="utf-8"))
        label_rows = annotation.get("regions") if isinstance(annotation, dict) else None
        if not isinstance(label_rows, list):
            parser.error("Tệp nhãn phải chứa danh sách regions")
        for index, label in enumerate(label_rows):
            required = ("image", "bbox_source", "text", "verified", "reviewed_by", "reviewed_at", "source_note")
            if (not isinstance(label, dict) or any(key not in label for key in required)
                    or label.get("verified") is not True or not isinstance(label.get("text"), str)
                    or any(not str(label.get(key, "")).strip() for key in ("reviewed_by", "reviewed_at", "source_note"))
                    or not isinstance(label.get("bbox_source"), list) or len(label["bbox_source"]) != 4):
                parser.error(f"Nhãn vùng {index} thiếu nội dung hoặc nguồn xác minh hợp lệ")
            label_path = Path(label["image"]).expanduser().resolve()
            labels_by_image.setdefault(str(label_path), []).append(label)

    from app.core.ocr_engine import OCREngine
    engine = OCREngine(lang="vi")
    runs = []
    for image_path in args.images:
        image_path = image_path.expanduser().resolve()
        if not image_path.is_file():
            runs.append({"image": str(image_path), "status": "failed", "error": "image_not_found"})
            continue
        destination = root / _safe_name(image_path.stem)
        destination.mkdir(parents=True, exist_ok=True)
        try:
            result = run_image(engine, image_path, destination, args.resize_scale,
                               labels_by_image.get(str(image_path), []),
                               preprocess=not args.no_preprocess, denoise=not args.no_denoise)
            result["source_file"] = image_path.name
            (destination / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            runs.append({"source_file": image_path.name, "status": result["status"],
                         "report": str((destination / "report.json").relative_to(root))})
            summary = result.get("summary", {})
            print(f"{image_path.name}: {summary.get('regions_compared', 0)} crop đã so sánh, "
                  f"{summary.get('regions_with_text_differences', 0)} crop có khác biệt")
        except Exception as exc:
            runs.append({"source_file": image_path.name, "status": "failed",
                         "error": type(exc).__name__, "message": str(exc)})
            print(f"LỖI {image_path.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
    index = {"schema_version": 1, "created_utc": stamp, "ground_truth_status": "unlabeled",
             "note": "Biến thể khác nhau không đồng nghĩa với nhận dạng chính xác hơn.", "runs": runs}
    (root / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    archive = root.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(root.parent))
    print(f"Thư mục kết quả: {root}")
    print(f"Gói ZIP: {archive}")
    return 0 if all(row["status"] == "success" for row in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
