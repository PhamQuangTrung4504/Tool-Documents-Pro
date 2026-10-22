"""Các phép tính và kiểm tra dữ liệu cho benchmark OCR có ground truth thủ công."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from app.core.recognition_diagnostics import compute_cer, compute_wer


def _is_symbol_or_digit(char: str) -> bool:
    category = unicodedata.category(char)
    return char.isdigit() or category.startswith(("P", "S"))


def compute_symbol_accuracy(reference: str, hypothesis: str) -> Optional[float]:
    """Tỷ lệ ký tự số/ký hiệu ground truth được khớp trong căn chỉnh Levenshtein.

    Trả về None nếu ground truth không có chữ số hoặc ký hiệu. Ký tự mất được tính
    là sai; ký tự chèn không làm thay đổi mẫu số.
    """
    ref = unicodedata.normalize("NFC", reference)
    hyp = unicodedata.normalize("NFC", hypothesis)
    symbol_positions = [i for i, char in enumerate(ref) if _is_symbol_or_digit(char)]
    if not symbol_positions:
        return None

    rows, cols = len(ref) + 1, len(hyp) + 1
    distance = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        distance[i][0] = i
    for j in range(cols):
        distance[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            distance[i][j] = min(
                distance[i - 1][j] + 1,
                distance[i][j - 1] + 1,
                distance[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]),
            )

    matched_symbol_count = 0
    i, j = len(ref), len(hyp)
    while i or j:
        if i and j and distance[i][j] == distance[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            if i - 1 in symbol_positions and ref[i - 1] == hyp[j - 1]:
                matched_symbol_count += 1
            i -= 1
            j -= 1
        elif i and distance[i][j] == distance[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    return matched_symbol_count / len(symbol_positions)


def validate_manifest(document: Dict[str, Any]) -> List[str]:
    """Kiểm tra một mục manifest; không suy diễn hay tạo ground truth."""
    errors: List[str] = []
    for field in ("id", "image", "ground_truth"):
        if field not in document:
            errors.append(f"Thiếu trường bắt buộc: {field}")
    gt = document.get("ground_truth")
    if not isinstance(gt, dict):
        errors.append("ground_truth phải là đối tượng đã được người dùng điền")
        return errors
    if not isinstance(gt.get("text"), str):
        errors.append("ground_truth.text phải là chuỗi do người dùng kiểm tra")
    if gt.get("verified") is not True:
        errors.append("ground_truth.verified phải là true sau khi kiểm tra thủ công")
    for field in ("reviewed_by", "reviewed_at", "source_note"):
        if not isinstance(gt.get(field), str) or not gt[field].strip():
            errors.append(f"ground_truth.{field} phải ghi nhận nguồn kiểm chứng")
    if not isinstance(document.get("error_groups", []), list):
        errors.append("error_groups phải là danh sách")
    table = gt.get("table")
    if table is not None:
        if not isinstance(table, dict) or not isinstance(table.get("expected_grid"), list):
            errors.append("ground_truth.table.expected_grid phải là ma trận")
    return errors


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate_records(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    items = list(records)
    if not items:
        return {"document_count": 0, "groups": {}}
    group_items: Dict[str, List[Dict[str, Any]]] = {"__all__": items}
    for item in items:
        for group in item.get("error_groups", []):
            group_items.setdefault(str(group), []).append(item)
    groups: Dict[str, Any] = {}
    for group, subset in group_items.items():
        groups[group] = {
            "document_count": len(subset),
            "macro_cer": sum(row["cer"] for row in subset) / len(subset),
            "macro_wer": sum(row["wer"] for row in subset) / len(subset),
            "macro_symbol_accuracy": _mean_known(row.get("symbol_accuracy") for row in subset),
            "mean_ocr_seconds": sum(row["ocr_seconds"] for row in subset) / len(subset),
            "table_metrics": _mean_table_metrics(subset),
        }
    return {"document_count": len(items), "groups": groups}


def _mean_known(values: Iterable[Optional[float]]) -> Optional[float]:
    known = [value for value in values if value is not None]
    return sum(known) / len(known) if known else None


def _mean_table_metrics(items: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    keys = sorted({key for item in items for key in (item.get("table_metrics") or {})})
    return {
        key: sum(item["table_metrics"][key] for item in items if item.get("table_metrics") and key in item["table_metrics"])
        / sum(1 for item in items if item.get("table_metrics") and key in item["table_metrics"])
        for key in keys
    }


def compare_runs(baseline: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    """So sánh hai lượt trên cùng ảnh/ground truth và nêu rõ từng hồi quy."""
    base = {row["id"]: row for row in baseline["documents"]}
    trial = {row["id"]: row for row in candidate["documents"]}
    if base.keys() != trial.keys():
        raise ValueError("Hai lượt phải có cùng tập mã tài liệu")
    comparisons = []
    for sample_id in sorted(base):
        old, new = base[sample_id], trial[sample_id]
        if old["image_sha256"] != new["image_sha256"] or old["ground_truth_sha256"] != new["ground_truth_sha256"]:
            raise ValueError(f"Ảnh hoặc ground truth không đồng nhất: {sample_id}")
        row = {
            "id": sample_id,
            "cer_delta_baseline_minus_candidate": old["cer"] - new["cer"],
            "wer_delta_baseline_minus_candidate": old["wer"] - new["wer"],
            "symbol_accuracy_delta_candidate_minus_baseline": _subtract_optional(
                new.get("symbol_accuracy"), old.get("symbol_accuracy")
            ),
            "ocr_seconds_delta_candidate_minus_baseline": new["ocr_seconds"] - old["ocr_seconds"],
            "baseline": _metric_slice(old),
            "candidate": _metric_slice(new),
            "regressed": new["cer"] > old["cer"],
        }
        comparisons.append(row)
    return {
        "baseline_version": baseline["version"],
        "candidate_version": candidate["version"],
        "document_count": len(comparisons),
        "regressed_documents": [row["id"] for row in comparisons if row["regressed"]],
        "documents": comparisons,
        "baseline_summary": aggregate_records(base.values()),
        "candidate_summary": aggregate_records(trial.values()),
    }


def _subtract_optional(left: Optional[float], right: Optional[float]) -> Optional[float]:
    return left - right if left is not None and right is not None else None


def _metric_slice(item: Dict[str, Any]) -> Dict[str, Any]:
    return {key: item.get(key) for key in ("cer", "wer", "symbol_accuracy", "ocr_seconds", "table_metrics")}


def text_metrics(reference: str, hypothesis: str) -> Dict[str, Optional[float]]:
    return {
        "cer": compute_cer(reference, hypothesis),
        "wer": compute_wer(reference, hypothesis),
        "symbol_accuracy": compute_symbol_accuracy(reference, hypothesis),
    }


def ground_truth_digest(ground_truth: Dict[str, Any]) -> str:
    encoded = json.dumps(ground_truth, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
