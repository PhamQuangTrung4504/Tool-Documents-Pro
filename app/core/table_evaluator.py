#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
app.core.table_evaluator
========================
Module đánh giá định lượng, truy vết lỗi (Traceability) và chẩn đoán cấu trúc
bảng biểu (Table Structure Recognition Evaluation).

Cung cấp:
1. Data structures:
   - TableGroundTruth: Chứa nhãn chuẩn hàng/cột và các dòng ngoài bảng.
   - CellErrorDiagnostic: Ghi nhận chi tiết từng lỗi ô kèm nhãn chẩn đoán nguyên nhân.
   - TableEvalTrace: Hồ sơ truy vết toàn diện cho 1 lần chạy benchmark kèm export JSON.
2. Metrics:
   - table_detected_exact (0.0 hoặc 1.0)
   - row_count_accuracy (0.0 - 1.0)
   - col_count_accuracy (0.0 - 1.0)
   - cell_exact_match_rate (0.0 - 1.0)
   - cell_content_f1 (0.0 - 1.0)
3. Chẩn đoán tự động phân loại lỗi (Failure Mode Diagnostics):
   - COLUMN_SHIFT: Dữ liệu bị nhảy lệch sang cột trái/phải.
   - MULTI_LINE_SPLIT: Dòng bị tách thành nhiều dòng con.
   - HEADER_DROP: Dòng tiêu đề bị nuốt chữ hoặc sót cột.
   - SPAN_COLLAPSE: Ô gộp làm sập lưới cột.
   - MISSING_CELL: Ô có dữ liệu bị bỏ trống.
   - NOISE_INTRUSION: Ký tự rác (|, [, ]) lọt vào ô.
   - TEXT_MISMATCH: Lỗi sai khác ký tự.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from app.core.layout_analyzer import reconstruct_document_layout
from app.core.models import OCRBox, OCRTable

__all__ = [
    "TableGroundTruth",
    "CellErrorDiagnostic",
    "TableEvalTrace",
    "evaluate_table_sample",
    "generate_benchmark_summary",
]


@dataclass
class TableGroundTruth:
    """Nhãn chuẩn (Ground Truth) cho một mẫu bảng biểu tài liệu."""

    sample_id: str
    category: str  # "bordered", "borderless", "merged_cells", "multi_column_dense", "noisy_scan"
    expected_grid: List[List[str]] = field(default_factory=list)
    expected_non_table_lines: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    expected_cell_spans: List[Tuple[int, int, int, int]] = field(default_factory=list)
    expected_diagonal_cells: List[Tuple[int, int, str]] = field(default_factory=list)


@dataclass
class CellErrorDiagnostic:
    """Thông tin chẩn đoán lỗi tại một ô bảng (Cell-level Diagnostic)."""

    row_idx: int
    col_idx: int
    expected_text: str
    actual_text: str
    failure_mode: str  # "COLUMN_SHIFT", "MULTI_LINE_SPLIT", "HEADER_DROP", "SPAN_COLLAPSE", "MISSING_CELL", "NOISE_INTRUSION", "TEXT_MISMATCH"


@dataclass
class TableEvalTrace:
    """Hồ sơ truy vết toàn diện (Traceability) cho một lần chạy đánh giá bảng."""

    sample_id: str
    category: str
    input_boxes: List[OCRBox]
    predicted_tables: List[OCRTable]
    predicted_lines: List[List[OCRBox]]
    ground_truth: TableGroundTruth
    metrics: Dict[str, float]
    diagnostics: List[CellErrorDiagnostic]
    elapsed_ms: float

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi hồ sơ truy vết thành dict có thể tuần tự hóa JSON."""
        return {
            "sample_id": self.sample_id,
            "category": self.category,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "metrics": {k: round(v, 4) for k, v in self.metrics.items()},
            "diagnostics": [asdict(d) for d in self.diagnostics],
            "ground_truth": {
                "sample_id": self.ground_truth.sample_id,
                "category": self.ground_truth.category,
                "expected_grid": self.ground_truth.expected_grid,
                "expected_non_table_lines": self.ground_truth.expected_non_table_lines,
                "metadata": self.ground_truth.metadata,
                "expected_cell_spans": [list(span) for span in self.ground_truth.expected_cell_spans],
                "expected_diagonal_cells": [list(cell) for cell in self.ground_truth.expected_diagonal_cells],
            },
            "predicted_tables": [
                {
                    "num_rows": tbl.num_rows,
                    "num_cols": tbl.num_cols,
                    "bbox": tbl.bbox,
                    "grid": tbl.grid,
                    "column_boundaries": tbl.column_boundaries,
                    "row_boundaries": tbl.row_boundaries,
                    "cells": [
                        {
                            "row_start": cell.row_start,
                            "col_start": cell.col_start,
                            "row_end": cell.row_end,
                            "col_end": cell.col_end,
                            "bbox": cell.bbox,
                            "diagonal_border": cell.diagonal_border,
                            "border_edges": cell.border_edges,
                        }
                        for cell in tbl.cells
                    ],
                }
                for tbl in self.predicted_tables
            ],
            "predicted_lines_count": len(self.predicted_lines),
            "input_boxes_count": len(self.input_boxes),
        }

    def save_json(self, path: Union[str, Path]) -> None:
        """Lưu toàn bộ hồ sơ truy vết ra file JSON."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


def _tokenize(text: str) -> List[str]:
    """Tách từ/token đơn giản chuẩn hóa khoảng trắng và chữ thường."""
    return [tok for tok in re.split(r"\s+", text.strip().lower()) if tok]


def _set_f1(expected: Set[Tuple[Any, ...]], actual: Set[Tuple[Any, ...]]) -> float:
    if not expected and not actual:
        return 1.0
    if not expected or not actual:
        return 0.0
    true_positive = len(expected & actual)
    precision = true_positive / len(actual)
    recall = true_positive / len(expected)
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def _diagnose_cell_errors(
    expected_grid: List[List[str]],
    actual_grid: List[List[str]],
) -> List[CellErrorDiagnostic]:
    """
    Phân tích và tự động gắn nhãn chẩn đoán nguyên nhân lỗi cấu trúc ô:
    - COLUMN_SHIFT: Dữ liệu ô bị lệch sang cột bên cạnh.
    - MULTI_LINE_SPLIT: Hàng bị tách thành nhiều dòng liên tiếp.
    - HEADER_DROP: Dòng tiêu đề bị nuốt từ / thiếu thông tin.
    - SPAN_COLLAPSE: Lệch số cột do ô gộp span.
    - MISSING_CELL: Ô có dữ liệu bị rỗng / thiếu.
    - NOISE_INTRUSION: Ký tự rác lọt vào ô.
    - TEXT_MISMATCH: Sai khác nội dung thông thường.
    """
    diagnostics: List[CellErrorDiagnostic] = []
    gt_rows = len(expected_grid)
    gt_cols = len(expected_grid[0]) if gt_rows > 0 else 0
    act_rows = len(actual_grid)
    act_cols = len(actual_grid[0]) if act_rows > 0 else 0

    noise_tokens = {"|", "[", "]", "\\", "/", ";", "~", "1", "o"}

    for r in range(gt_rows):
        gt_row = expected_grid[r]
        for c in range(gt_cols):
            exp_text = gt_row[c].strip() if c < len(gt_row) else ""
            act_text = actual_grid[r][c].strip() if (r < act_rows and c < len(actual_grid[r])) else ""

            if exp_text == act_text:
                continue

            exp_norm = exp_text.lower()
            act_norm = act_text.lower()

            # Phân loại failure_mode theo thứ tự ưu tiên:
            failure_mode = "TEXT_MISMATCH"

            # 1. NOISE_INTRUSION: Ô xuất hiện ký tự rác hoặc viền nhiễu
            if act_text in noise_tokens or (len(act_text) <= 2 and act_text in {"[", "]", "|", "\\"}):
                failure_mode = "NOISE_INTRUSION"

            # 2. COLUMN_SHIFT: Nội dung ô kỳ vọng xuất hiện ở cột lân cận c-1 hoặc c+1
            elif r < act_rows and exp_norm and len(exp_norm) >= 2 and (
                (c > 0 and c - 1 < len(actual_grid[r]) and exp_norm in actual_grid[r][c - 1].lower()) or
                (c + 1 < len(actual_grid[r]) and exp_norm in actual_grid[r][c + 1].lower())
            ):
                failure_mode = "COLUMN_SHIFT"

            # 3. HEADER_DROP: Dòng tiêu đề (r == 0) bị thiếu từ hoặc rơi rụng
            elif r == 0 and exp_text:
                if not act_text:
                    failure_mode = "HEADER_DROP"
                elif any(word in exp_norm for word in exp_norm.split()) and not any(w in act_norm for w in exp_norm.split()):
                    failure_mode = "HEADER_DROP"

            # 4. MULTI_LINE_SPLIT: Dòng kỳ vọng bị chia sang hàng tiếp theo r+1
            elif exp_text and r + 1 < act_rows and any(
                (len(cell.strip()) >= 3 and (cell.strip().lower() in exp_norm or exp_norm in cell.strip().lower()))
                for cell in actual_grid[r + 1]
            ):
                failure_mode = "MULTI_LINE_SPLIT"

            # 5. SPAN_COLLAPSE: Bảng có ô gộp làm lệch số cột
            elif gt_cols != act_cols:
                failure_mode = "SPAN_COLLAPSE"

            # 6. MISSING_CELL: Ô kỳ vọng có chữ nhưng thực tế rỗng hoặc vượt ngoài bounds
            elif exp_text and not act_text:
                failure_mode = "MISSING_CELL"

            diagnostics.append(
                CellErrorDiagnostic(
                    row_idx=r,
                    col_idx=c,
                    expected_text=exp_text,
                    actual_text=act_text,
                    failure_mode=failure_mode,
                )
            )

    return diagnostics


def evaluate_table_sample(
    sample_id: str,
    input_boxes: List[OCRBox],
    ground_truth: TableGroundTruth,
    layout_fn: Optional[Callable[[List[OCRBox]], Any]] = None,
    image: Optional[Any] = None,
) -> TableEvalTrace:
    """
    Đánh giá định lượng kết quả nhận diện bố cục bảng trên một mẫu benchmark.
    Kiến trúc Pluggable: cho phép truyền bất kỳ layout_fn nào (mặc định: reconstruct_document_layout).
    """
    if layout_fn is None:
        layout_fn = reconstruct_document_layout

    t0 = time.perf_counter()
    if image is not None:
        try:
            raw_res = layout_fn(input_boxes, image=image)
        except TypeError:
            try:
                raw_res = layout_fn(input_boxes, image)
            except TypeError:
                raw_res = layout_fn(input_boxes)
    else:
        raw_res = layout_fn(input_boxes)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Giải nén kết quả (hỗ trợ cả LayoutResult tuple lẫn dict/object)
    if isinstance(raw_res, (tuple, list)) and len(raw_res) >= 2:
        pred_lines: List[List[OCRBox]] = raw_res[0]
        pred_tables: List[OCRTable] = raw_res[1]
    elif hasattr(raw_res, "tables") and hasattr(raw_res, "lines"):
        pred_lines = getattr(raw_res, "lines")
        pred_tables = getattr(raw_res, "tables")
    else:
        pred_lines = []
        pred_tables = []

    gt_grid = ground_truth.expected_grid
    has_gt_table = len(gt_grid) > 0
    gt_rows = len(gt_grid)
    gt_cols = len(gt_grid[0]) if gt_rows > 0 else 0

    has_pred_table = len(pred_tables) > 0
    first_pred_table = pred_tables[0] if has_pred_table else None
    pred_grid = first_pred_table.grid if first_pred_table else []
    pred_rows = len(pred_grid)
    pred_cols = len(pred_grid[0]) if pred_rows > 0 else 0

    # 1. table_detected_exact
    if has_gt_table:
        table_detected_exact = 1.0 if has_pred_table else 0.0
    else:
        table_detected_exact = 1.0 if not has_pred_table else 0.0

    # 2. row_count_accuracy & col_count_accuracy
    if has_gt_table:
        row_count_acc = max(0.0, 1.0 - abs(pred_rows - gt_rows) / max(1, gt_rows))
        col_count_acc = max(0.0, 1.0 - abs(pred_cols - gt_cols) / max(1, gt_cols))
    else:
        row_count_acc = 1.0 if pred_rows == 0 else 0.0
        col_count_acc = 1.0 if pred_cols == 0 else 0.0

    # 3. cell_exact_match_rate & cell_content_f1
    if not has_gt_table:
        cell_exact_match_rate = 1.0 if not has_pred_table else 0.0
        cell_content_f1 = 1.0 if not has_pred_table else 0.0
    else:
        exact_matches = 0
        total_gt_cells = max(1, gt_rows * gt_cols)

        total_tp_tokens = 0
        total_pred_tokens = 0
        total_gt_tokens = 0

        for r in range(gt_rows):
            gt_row = gt_grid[r]
            for c in range(gt_cols):
                exp_text = gt_row[c].strip() if c < len(gt_row) else ""
                act_text = pred_grid[r][c].strip() if (r < pred_rows and c < len(pred_grid[r])) else ""

                if exp_text == act_text:
                    exact_matches += 1

                exp_tokens = _tokenize(exp_text)
                act_tokens = _tokenize(act_text)

                total_gt_tokens += len(exp_tokens)
                total_pred_tokens += len(act_tokens)

                # Đếm số token khớp vị trí ô
                gt_tok_counts: Dict[str, int] = {}
                for t in exp_tokens:
                    gt_tok_counts[t] = gt_tok_counts.get(t, 0) + 1
                for t in act_tokens:
                    if gt_tok_counts.get(t, 0) > 0:
                        total_tp_tokens += 1
                        gt_tok_counts[t] -= 1

        cell_exact_match_rate = exact_matches / total_gt_cells

        if total_pred_tokens + total_gt_tokens == 0:
            cell_content_f1 = 1.0
        else:
            prec = total_tp_tokens / max(1, total_pred_tokens)
            rec = total_tp_tokens / max(1, total_gt_tokens)
            cell_content_f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    # 4. Chẩn đoán chi tiết lỗi
    diagnostics = _diagnose_cell_errors(gt_grid, pred_grid)

    input_ids = {id(box) for box in input_boxes}
    box_cell_owners: Dict[int, Set[Tuple[int, int]]] = {}
    assigned_box_ids: Set[int] = set()
    for table_index, table in enumerate(pred_tables):
        if table.cells:
            for cell_index, cell in enumerate(table.cells):
                for box in cell.boxes:
                    box_id = id(box)
                    if box_id in input_ids:
                        assigned_box_ids.add(box_id)
                        box_cell_owners.setdefault(box_id, set()).add((table_index, cell_index))
        else:
            for row_index, row in enumerate(table.row_boxes):
                for box in row:
                    box_id = id(box)
                    if box_id in input_ids:
                        assigned_box_ids.add(box_id)
                        box_cell_owners.setdefault(box_id, set()).add((table_index, row_index))
    for line in pred_lines:
        assigned_box_ids.update(id(box) for box in line if id(box) in input_ids)
    assignment_rate = len(assigned_box_ids) / max(1, len(input_ids))
    duplicate_cell_box_count = sum(max(0, len(owners) - 1) for owners in box_cell_owners.values())

    metrics = {
        "table_detected_exact": table_detected_exact,
        "row_count_accuracy": row_count_acc,
        "col_count_accuracy": col_count_acc,
        "cell_exact_match_rate": cell_exact_match_rate,
        "cell_content_f1": cell_content_f1,
        "ocr_box_assignment_rate": assignment_rate,
        "duplicate_cell_box_assignment_count": float(duplicate_cell_box_count),
    }

    if ground_truth.expected_cell_spans:
        expected_spans = set(ground_truth.expected_cell_spans)
        predicted_spans = {
            (cell.row_start, cell.col_start, cell.row_end, cell.col_end)
            for table in pred_tables for cell in table.cells if cell.is_merged
        }
        metrics["cell_span_f1"] = _set_f1(expected_spans, predicted_spans)
    if ground_truth.expected_diagonal_cells:
        expected_diagonals = set(ground_truth.expected_diagonal_cells)
        predicted_diagonals = {
            (cell.row_start, cell.col_start, cell.diagonal_border)
            for table in pred_tables for cell in table.cells if cell.diagonal_border
        }
        metrics["diagonal_cell_f1"] = _set_f1(expected_diagonals, predicted_diagonals)

    return TableEvalTrace(
        sample_id=sample_id,
        category=ground_truth.category,
        input_boxes=input_boxes,
        predicted_tables=pred_tables,
        predicted_lines=pred_lines,
        ground_truth=ground_truth,
        metrics=metrics,
        diagnostics=diagnostics,
        elapsed_ms=elapsed_ms,
    )


def generate_benchmark_summary(traces: List[TableEvalTrace]) -> Dict[str, Any]:
    """
    Tổng hợp báo cáo số liệu trung bình và tần suất các chế độ lỗi trên toàn tập benchmark.
    """
    if not traces:
        return {"num_samples": 0}

    metric_sums: Dict[str, float] = {}
    metric_counts: Dict[str, int] = {}
    failure_mode_counts: Dict[str, int] = {}

    for tr in traces:
        for k, v in tr.metrics.items():
            metric_sums[k] = metric_sums.get(k, 0.0) + v
            metric_counts[k] = metric_counts.get(k, 0) + 1
        for diag in tr.diagnostics:
            failure_mode_counts[diag.failure_mode] = failure_mode_counts.get(diag.failure_mode, 0) + 1

    n = len(traces)
    avg_metrics = {k: round(v / max(1, metric_counts[k]), 4) for k, v in metric_sums.items()}

    return {
        "num_samples": n,
        "average_metrics": avg_metrics,
        "failure_mode_frequencies": failure_mode_counts,
        "sample_summaries": [
            {
                "sample_id": tr.sample_id,
                "category": tr.category,
                "metrics": {k: round(v, 4) for k, v in tr.metrics.items()},
                "error_count": len(tr.diagnostics),
                "failure_modes": list(set(d.failure_mode for d in tr.diagnostics)),
                "elapsed_ms": round(tr.elapsed_ms, 2),
            }
            for tr in traces
        ],
    }
