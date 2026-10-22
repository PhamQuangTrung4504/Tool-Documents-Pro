#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Thử nghiệm tái nhận dạng có chọn lọc; không thay OCRResult production."""

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

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ocr_crop_ablation import (  # noqa: E402
    AblationObserver,
    _build_crop_variants_timed,
    _predict,
    _safe_name,
    _verified_region_label,
    _write_image,
)

RERUN_VARIANTS = ("skip_border_cleaning", "safe_padding", "aspect_preserving_resize")
MAX_RERUNS_PER_REGION = 3


def assess_region_risk(capture: dict[str, Any], low_confidence: float = 0.35) -> dict[str, Any]:
    """Tạo tín hiệu ưu tiên kiểm tra; không dự đoán candidate nào đúng."""
    from app.core.recognition_diagnostics import analyze_crop_quality

    events = capture.get("events", [])
    primary = next((event for event in reversed(events)
                    if event.get("event") == "recognition_candidate"
                    and str(event.get("recognizer", "")).startswith("vietocr:")), None)
    paddle = next((event for event in reversed(events)
                   if event.get("event") == "recognition_candidate"
                   and event.get("recognizer") == "paddleocr"), None)
    images = capture.get("images", {})
    raw_list = images.get("preprocessed_rectified_crop", [])
    clean_list = images.get("vietocr_border_cleaned_crop", [])
    if primary is None or not raw_list or not clean_list:
        return {"selected": False, "reasons": ["missing_vietocr_crop_or_candidate"],
                "primary": primary, "paddle": paddle, "crop_quality": None}

    raw = raw_list[0]
    quality = analyze_crop_quality(raw)
    confidence = primary.get("confidence")
    reasons: list[str] = []
    mapping = primary.get("paddle_candidate_mapping") or {}
    paddle_text = str(paddle.get("text", "")) if paddle else ""
    primary_text = str(primary.get("text", ""))
    if paddle and mapping.get("status") == "matched" and paddle_text.strip() != primary_text.strip():
        reasons.append("verified_same_region_model_disagreement")
    elif paddle and mapping.get("status") != "matched":
        reasons.append("paddle_geometry_mapping_not_verified")
    if any(marker in primary_text for marker in ("?", "�", "�")):
        reasons.append("uncertain_or_replacement_character")
    if quality.has_right_char_clipping:
        reasons.append("possible_right_edge_clipping")
    if quality.has_top_diacritic_clipping:
        reasons.append("possible_top_diacritic_clipping")
    if quality.has_vertical_border_intrusion:
        reasons.append("border_ink_at_vertical_edge")
    if quality.height < 20:
        reasons.append("small_crop_height")
    if quality.aspect_ratio < 1.15:
        reasons.append("narrow_crop")
    if confidence is None:
        reasons.append("primary_confidence_unknown")
    elif float(confidence) <= low_confidence:
        reasons.append("low_primary_confidence_within_vietocr_scale")
    if not primary_text.strip():
        reasons.append("empty_primary_text")
    return {
        "selected": bool(reasons), "reasons": reasons, "primary": primary, "paddle": paddle,
        "crop_quality": vars(quality), "primary_confidence": confidence,
        "paddle_mapping": mapping,
    }


def run_selective(engine: Any, image_path: Path, output_dir: Path, *, max_regions: int = 12,
                  max_extra_calls: int = 36, max_extra_seconds: float = 30.0,
                  resize_scale: float = 1.5,
                  region_labels: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Chạy production OCR rồi tái nhận dạng các vùng ưu tiên trong ngân sách."""
    observer = AblationObserver()
    production_started = time.perf_counter()
    result = engine.recognize(image_path, diagnostic_observer=observer.observe)
    production_seconds = time.perf_counter() - production_started
    model = getattr(engine, "_vietocr", None)
    report: dict[str, Any] = {
        "source_image": str(image_path.resolve()), "status": "success",
        "production_ocr_seconds": production_seconds,
        "production_result_unchanged": True,
        "ground_truth_status": "available_if_region_label_matches" if region_labels else "unlabeled",
        "automatic_candidate_selection": False, "selected_candidate": None,
        "limits": {"max_regions": max_regions, "max_extra_calls": max_extra_calls,
                   "max_extra_seconds": max_extra_seconds,
                   "max_reruns_per_region": MAX_RERUNS_PER_REGION},
        "regions": [],
    }
    if model is None:
        report["status"] = "vietocr_unavailable"
        return report

    assessed = []
    for crop_id, capture in observer.crops.items():
        risk = assess_region_risk(capture)
        if risk["selected"]:
            assessed.append((crop_id, capture, risk))
    assessed.sort(key=lambda row: (-len(row[2]["reasons"]), row[0]))
    selected = assessed[:max_regions]
    deadline = time.perf_counter() + max(0.0, max_extra_seconds)
    calls_used = 0
    extra_started = time.perf_counter()
    stop_reason = None

    for crop_id, capture, risk in selected:
        if calls_used >= max_extra_calls:
            stop_reason = "extra_call_budget_exhausted"
            break
        events, images = capture["events"], capture["images"]
        primary = risk["primary"]
        raw_list = images.get("preprocessed_rectified_crop", [])
        clean_list = images.get("vietocr_border_cleaned_crop", [])
        exact_inputs = images.get("vietocr_recognizer_input", [])
        split = any(event.get("event") == "recognition_subcandidate" for event in events)
        region: dict[str, Any] = {
            "crop_id": crop_id, "risk_reasons": risk["reasons"],
            "risk_evidence": {key: value for key, value in risk.items()
                              if key not in {"primary", "paddle"}},
            "bbox_source": primary.get("bbox_source"),
            "polygon_source": primary.get("polygon_source"),
            "production_candidates": [
                {"recognizer": primary.get("recognizer"), "text": primary.get("text"),
                 "confidence": primary.get("confidence"), "provenance": "production_primary"},
                *([{"recognizer": risk["paddle"].get("recognizer"),
                    "text": risk["paddle"].get("text"),
                    "confidence": risk["paddle"].get("confidence"),
                    "mapping": risk["paddle"].get("paddle_candidate_mapping"),
                    "provenance": "production_paddle_candidate"}] if risk["paddle"] else []),
            ],
            "rerecognition_candidates": [], "selected_candidate": None,
        }
        label, label_status = _verified_region_label(primary.get("bbox_source"), region_labels or [])
        region["ground_truth_status"] = label_status if region_labels else "unlabeled"
        if not raw_list or not clean_list or not exact_inputs or split or len(exact_inputs) != 1:
            region["status"] = "skipped_crop_contract"
            region["skip_reason"] = "Thiếu crop recognizer duy nhất hoặc production đã tách thành sub-crop."
            report["regions"].append(region)
            continue

        variants, transform_times = _build_crop_variants_timed(raw_list[0], clean_list[0], resize_scale)
        region_dir = output_dir / "regions" / f"crop_{crop_id:04d}"
        _write_image(region_dir / "production_input.png", exact_inputs[0])
        _write_image(region_dir / "raw_crop.png", raw_list[0])
        for variant in RERUN_VARIANTS:
            if len(region["rerecognition_candidates"]) >= MAX_RERUNS_PER_REGION:
                break
            if calls_used >= max_extra_calls:
                stop_reason = "extra_call_budget_exhausted"
                break
            if time.perf_counter() >= deadline:
                stop_reason = "extra_time_budget_exhausted"
                break
            crop_variant = variants[variant]
            relative_image = f"regions/crop_{crop_id:04d}/{variant}.png"
            _write_image(output_dir / relative_image, crop_variant)
            call_started = time.perf_counter()
            text, confidence = _predict(model, crop_variant)
            call_seconds = time.perf_counter() - call_started
            calls_used += 1
            from app.core.recognition_diagnostics import analyze_crop_quality, find_out_of_vocab_chars
            candidate = {
                "recognizer": getattr(engine, "vietocr_model_name", "vietocr:singleton"),
                "text": text, "confidence": confidence,
                "confidence_comparable_across_models": False,
                "provenance": "experimental_vietocr_rerecognition",
                "source_crop_id": crop_id, "source_bbox": primary.get("bbox_source"),
                "source_polygon": primary.get("polygon_source"),
                "variant": variant, "image": relative_image,
                "transform_seconds": transform_times[variant],
                "inference_seconds": call_seconds,
                "crop_quality": vars(analyze_crop_quality(crop_variant)),
            }
            candidate["out_of_vocabulary_characters"] = find_out_of_vocab_chars(text)
            if label is not None:
                from app.core.recognition_diagnostics import compute_cer, compute_wer
                candidate["ground_truth_metrics"] = {
                    "cer": compute_cer(label["text"], text), "wer": compute_wer(label["text"], text)
                }
            region["rerecognition_candidates"].append(candidate)
        region["status"] = "completed" if region["rerecognition_candidates"] else "budget_exhausted"
        report["regions"].append(region)
        if stop_reason:
            break

    report["additional_vietocr_calls"] = calls_used
    report["additional_paddleocr_calls"] = 0
    report["additional_elapsed_seconds"] = time.perf_counter() - extra_started
    report["stop_reason"] = stop_reason
    report["summary"] = {
        "regions_captured": len(observer.crops), "regions_flagged": len(assessed),
        "regions_selected_within_limit": len(selected),
        "regions_completed": sum(row["status"] == "completed" for row in report["regions"]),
        "regions_not_flagged_unchanged": len(observer.crops) - len(assessed),
        "candidate_texts_differ_from_primary": sum(
            candidate["text"] != region["production_candidates"][0]["text"]
            for region in report["regions"] for candidate in region["rerecognition_candidates"]
        ),
        "labeled_regions": sum(row.get("ground_truth_status") == "matched" for row in report["regions"]),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", nargs="+", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, help="Thư mục ngoài repository; mặc định TEMP.")
    parser.add_argument("--max-regions", type=int, default=12)
    parser.add_argument("--max-extra-calls", type=int, default=36)
    parser.add_argument("--max-extra-seconds", type=float, default=30.0)
    parser.add_argument("--resize-scale", type=float, default=1.5)
    parser.add_argument("--region-ground-truth", type=Path)
    args = parser.parse_args()
    if args.max_regions < 0 or args.max_extra_calls < 0 or args.max_extra_seconds < 0:
        parser.error("Các giới hạn phải không âm")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = (args.output_dir or Path(tempfile.gettempdir()) / f"tdp_selective_ocr_{stamp}").expanduser().resolve()
    try:
        root.relative_to(ROOT.resolve())
    except ValueError:
        pass
    else:
        parser.error("--output-dir phải nằm ngoài repository")
    root.mkdir(parents=True, exist_ok=True)
    labels_by_image: dict[str, list[dict[str, Any]]] = {}
    if args.region_ground_truth:
        payload = json.loads(args.region_ground_truth.expanduser().read_text(encoding="utf-8"))
        rows = payload.get("regions") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            parser.error("Tệp nhãn vùng phải chứa danh sách regions")
        for index, row in enumerate(rows):
            required = ("image", "bbox_source", "text", "verified", "reviewed_by", "reviewed_at", "source_note")
            if (not isinstance(row, dict) or any(key not in row for key in required)
                    or row.get("verified") is not True or not isinstance(row.get("text"), str)
                    or any(not str(row.get(key, "")).strip() for key in ("reviewed_by", "reviewed_at", "source_note"))
                    or not isinstance(row.get("bbox_source"), list) or len(row["bbox_source"]) != 4):
                parser.error(f"Nhãn vùng {index} không đầy đủ hoặc chưa xác minh")
            labels_by_image.setdefault(str(Path(row["image"]).expanduser().resolve()), []).append(row)

    from app.core.ocr_engine import OCREngine
    engine = OCREngine(lang="vi")
    runs = []
    for image_path in args.images:
        image_path = image_path.expanduser().resolve()
        if not image_path.is_file():
            runs.append({"image": str(image_path), "status": "failed", "error": "image_not_found"})
            continue
        destination = root / _safe_name(image_path.stem)
        try:
            report = run_selective(engine, image_path, destination, max_regions=args.max_regions,
                                   max_extra_calls=args.max_extra_calls,
                                   max_extra_seconds=args.max_extra_seconds,
                                   resize_scale=args.resize_scale,
                                   region_labels=labels_by_image.get(str(image_path), []))
            (destination / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            runs.append({"source_file": image_path.name, "status": report["status"],
                         "report": f"{destination.name}/report.json",
                         "additional_vietocr_calls": report.get("additional_vietocr_calls", 0)})
            print(f"{image_path.name}: {report.get('summary', {}).get('regions_flagged', 0)} vùng nghi vấn, "
                  f"{report.get('additional_vietocr_calls', 0)} lượt VietOCR bổ sung")
        except Exception as exc:
            runs.append({"source_file": image_path.name, "status": "failed",
                         "error": type(exc).__name__, "message": str(exc)})
            print(f"LỖI {image_path.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
    (root / "index.json").write_text(json.dumps({"schema_version": 1, "created_utc": stamp,
        "production_mutated": False, "automatic_selection": False, "runs": runs},
        ensure_ascii=False, indent=2), encoding="utf-8")
    archive = root.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(root.parent))
    print(f"Báo cáo ZIP: {archive}")
    return 0 if all(row["status"] == "success" for row in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
