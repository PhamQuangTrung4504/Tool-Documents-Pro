#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tạo manifest ground truth thủ công, chạy OCR và so sánh các phiên bản."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _as_path(value: str | Path) -> Path:
    """Chuẩn hóa đầu vào argparse/API thành Path ngay tại biên xử lý."""
    return Path(value).expanduser()


def _outside_repository(path: str | Path, label: str) -> Path:
    resolved = _as_path(path).resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError:
        return resolved
    raise ValueError(f"{label} phải nằm ngoài repository: {resolved}")


def _json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Không thể tuần tự hóa kiểu dữ liệu: {type(value).__name__}")


def _default_output_dir() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA")
    base = Path(local_app_data) if local_app_data else Path(tempfile.gettempdir())
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return base / "Tool-Documents-Pro" / "benchmarks" / stamp


def _load_manifest(path: Path) -> list[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    documents = payload.get("documents") if isinstance(payload, dict) else None
    if not isinstance(documents, list) or not documents:
        raise ValueError("Manifest phải có danh sách documents không rỗng")
    from app.core.ocr_benchmark import validate_manifest

    errors = []
    ids = set()
    for index, document in enumerate(documents):
        errors.extend(f"documents[{index}]: {message}" for message in validate_manifest(document))
        sample_id = document.get("id") if isinstance(document, dict) else None
        if sample_id in ids:
            errors.append(f"Mã tài liệu bị lặp: {sample_id}")
        ids.add(sample_id)
        image_path = Path(str(document.get("image", "")))
        if not image_path.is_absolute():
            image_path = path.parent / image_path
        if not image_path.is_file():
            errors.append(f"Không tìm thấy ảnh của documents[{index}]: {image_path}")
        document["_resolved_image"] = image_path.resolve()
    if errors:
        raise ValueError("\n".join(errors))
    return documents


def _init_manifest(args: argparse.Namespace) -> int:
    manifest_path = _outside_repository(args.output_manifest, "Manifest")
    image_paths = []
    for item in args.images:
        path = _as_path(item).resolve()
        if path.is_dir():
            image_paths.extend(sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES))
        elif path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            image_paths.append(path)
        else:
            raise ValueError(f"Không phải ảnh tài liệu hợp lệ: {path}")
    if not image_paths:
        raise ValueError("Không tìm thấy ảnh trong các đường dẫn đã cho")
    unique = list(dict.fromkeys(image_paths))
    _json_write(manifest_path, {
        "schema_version": 1,
        "notes": "Chỉ điền ground_truth.text sau khi người dùng đối chiếu ảnh gốc. Không dùng output OCR làm ground truth.",
        "documents": [
            {
                "id": f"doc_{index:04d}",
                "image": str(image),
                "error_groups": [],
                "ground_truth": {
                    "text": None,
                    "verified": False,
                    "reviewed_by": "",
                    "reviewed_at": "",
                    "source_note": "",
                    "table": None,
                    "formatting_annotation": None,
                },
            }
            for index, image in enumerate(unique, 1)
        ],
    })
    print(f"Đã tạo manifest cần kiểm tra thủ công: {manifest_path}")
    print(f"Số ảnh: {len(unique)}. Chưa có mẫu nào được tính là có ground truth.")
    return 0


def _run_benchmark(args: argparse.Namespace) -> int:
    from app.core.ocr_benchmark import (
        aggregate_records,
        ground_truth_digest,
        sha256_file,
        text_metrics,
    )

    manifest_path = _as_path(args.manifest).resolve()
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.version) or args.version in {".", ".."}:
        raise ValueError("Nhãn phiên bản chỉ được gồm chữ, số, dấu chấm, gạch dưới và gạch nối")
    output_dir = _outside_repository(args.output_dir or _default_output_dir(), "Thư mục kết quả")
    documents = _load_manifest(manifest_path)

    import cv2
    from app.core.ocr_engine import OCREngine

    engine = OCREngine(lang=args.language)
    records = []
    for document in documents:
        image_path: Path = document["_resolved_image"]
        result = engine.recognize(
            image_path,
            preprocess=not args.no_preprocess,
            denoise=not args.no_denoise,
            include_diagnostics=args.include_diagnostics,
        )
        gt = document["ground_truth"]
        record: Dict[str, Any] = {
            "id": document["id"],
            "image": str(image_path),
            "image_sha256": sha256_file(image_path),
            "ground_truth_sha256": ground_truth_digest(gt),
            "error_groups": document.get("error_groups", []),
            "ground_truth_text": gt["text"],
            "prediction_text": result.full_text,
            "raw_prediction_text": result.raw_full_text,
            "ocr_seconds": result.elapsed_seconds,
            **text_metrics(gt["text"], result.full_text),
            "box_count": len(result.boxes),
            "table_count": len(result.tables),
            "table_metrics": None,
            "formatting_evaluation": {
                "status": "not_scored",
                "annotation_available": gt.get("formatting_annotation") is not None,
                "note": "Định dạng Word không được gộp vào CER/WER; cần nhãn và phép đánh giá riêng.",
            },
        }
        table_gt = gt.get("table")
        if table_gt is not None:
            from app.core.table_evaluator import TableGroundTruth, evaluate_table_sample

            source_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            table_truth = TableGroundTruth(
                sample_id=document["id"],
                category=table_gt.get("category", "user_annotated"),
                expected_grid=table_gt["expected_grid"],
                expected_non_table_lines=table_gt.get("expected_non_table_lines", []),
                metadata=table_gt.get("metadata", {}),
                expected_cell_spans=[tuple(span) for span in table_gt.get("expected_cell_spans", [])],
                expected_diagonal_cells=[tuple(cell) for cell in table_gt.get("expected_diagonal_cells", [])],
            )
            table_trace = evaluate_table_sample(
                sample_id=document["id"], input_boxes=result.boxes,
                ground_truth=table_truth, image=source_image,
            )
            record["table_metrics"] = table_trace.metrics
            record["table_layout_seconds"] = table_trace.elapsed_ms / 1000.0
            record["table_diagnostics"] = [vars(item) for item in table_trace.diagnostics]
        if args.include_diagnostics:
            record["ocr_diagnostics"] = result.diagnostics
            record["box_evidence"] = [
                {"index": index, "text": box.text, "raw_text": box.raw_text, "confidence": box.confidence,
                 "bbox": box.bbox, "metadata": box.metadata}
                for index, box in enumerate(result.boxes)
            ]
        records.append(record)
        print(f"Đã đo {document['id']}: CER={record['cer']:.4f}, WER={record['wer']:.4f}, OCR={record['ocr_seconds']:.2f}s")

    run = {
        "schema_version": 1,
        "version": args.version,
        "revision": args.revision,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path),
        "language": args.language,
        "preprocess": not args.no_preprocess,
        "denoise": not args.no_denoise,
        "diagnostics_included": args.include_diagnostics,
        "documents": records,
        "summary": aggregate_records(records),
    }
    output_path = output_dir / f"run_{args.version}.json"
    _json_write(output_path, run)
    print(f"Kết quả benchmark (có thể chứa văn bản tài liệu): {output_path}")
    return 0


def _compare(args: argparse.Namespace) -> int:
    from app.core.ocr_benchmark import compare_runs

    base_path = _as_path(args.baseline).resolve()
    candidate_path = _as_path(args.candidate).resolve()
    output_path = _outside_repository(args.output, "Báo cáo so sánh") if args.output else _default_output_dir() / "comparison.json"
    report = compare_runs(json.loads(base_path.read_text(encoding="utf-8")),
                          json.loads(candidate_path.read_text(encoding="utf-8")))
    _json_write(output_path, report)
    print(f"Tài liệu CER tăng: {len(report['regressed_documents'])}/{report['document_count']}")
    print(f"Báo cáo so sánh: {output_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Tạo manifest trống để người dùng nhập ground truth đã kiểm tra")
    init.add_argument("--images", nargs="+", type=Path, required=True)
    init.add_argument("--output-manifest", type=Path, required=True)
    init.set_defaults(handler=_init_manifest)

    run = commands.add_parser("run", help="Chạy OCR và đo trên manifest đã xác minh")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--version", required=True, help="Nhãn baseline hoặc thử nghiệm")
    run.add_argument("--revision", default="", help="Mã commit hoặc mô tả phiên bản tùy chọn")
    run.add_argument("--output-dir", type=Path)
    run.add_argument("--language", default="vi")
    run.add_argument("--no-preprocess", action="store_true")
    run.add_argument("--no-denoise", action="store_true")
    run.add_argument("--include-diagnostics", action="store_true", help="Lưu diagnostics và OCRBox; có thể chứa dữ liệu nhạy cảm")
    run.set_defaults(handler=_run_benchmark)

    compare = commands.add_parser("compare", help="So sánh hai lượt trên cùng ảnh và ground truth")
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--output", type=Path)
    compare.set_defaults(handler=_compare)
    args = parser.parse_args()
    try:
        return args.handler(args)
    except Exception as exc:
        print(f"LỖI: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
