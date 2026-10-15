# -*- coding: utf-8 -*-
"""
tests.test_table_evaluation_harness
===================================
Bộ kiểm thử hệ thống đánh giá định lượng, lưu vết lỗi (Traceability)
và chạy Benchmark 5 dạng bảng tiêu chuẩn (Phase C2):
1. sample_01_bordered_admin
2. sample_02_borderless_financial
3. sample_03_merged_cells_header
4. sample_04_dense_7cols
5. sample_05_noisy_broken_grid

Xuất baseline kết quả ra logs/c2_table_baseline_trace.json để làm đối chứng cho Phase C3.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pytest
from app.core.layout_analyzer import reconstruct_document_layout
from app.core.models import OCRBox
from app.core.table_evaluator import (
    CellErrorDiagnostic,
    TableEvalTrace,
    TableGroundTruth,
    _diagnose_cell_errors,
    evaluate_table_sample,
    generate_benchmark_summary,
)
from tests.benchmarks.table_benchmark_samples import (
    build_sample_01_bordered_admin,
    build_sample_02_borderless_financial,
    build_sample_03_merged_cells_header,
    build_sample_04_dense_7cols,
    build_sample_05_noisy_broken_grid,
    get_all_benchmark_samples,
)


def test_01_traceability_data_structures_and_json_export(tmp_path: Path):
    """Kiểm tra cấu trúc dữ liệu lưu vết và khả năng xuất/nạp JSON chuẩn xác."""
    gt = TableGroundTruth(
        sample_id="test_sample",
        category="bordered",
        expected_grid=[["A", "B"], ["1", "2"]],
        expected_non_table_lines=["TIÊU ĐỀ"],
        metadata={"note": "test metadata"},
    )
    assert gt.sample_id == "test_sample"
    assert gt.category == "bordered"
    assert len(gt.expected_grid) == 2

    diag = CellErrorDiagnostic(
        row_idx=0,
        col_idx=1,
        expected_text="B",
        actual_text="C",
        failure_mode="TEXT_MISMATCH",
    )
    assert diag.failure_mode == "TEXT_MISMATCH"

    trace = TableEvalTrace(
        sample_id="test_sample",
        category="bordered",
        input_boxes=[
            OCRBox("A", 0.95, [[0, 0], [10, 0], [10, 10], [0, 10]]),
            OCRBox("B", 0.95, [[20, 0], [30, 0], [30, 10], [20, 10]]),
        ],
        predicted_tables=[],
        predicted_lines=[],
        ground_truth=gt,
        metrics={"table_detected_exact": 0.0, "cell_exact_match_rate": 0.0},
        diagnostics=[diag],
        elapsed_ms=12.5,
    )

    t_dict = trace.to_dict()
    assert t_dict["sample_id"] == "test_sample"
    assert t_dict["metrics"]["table_detected_exact"] == 0.0
    assert len(t_dict["diagnostics"]) == 1
    assert t_dict["diagnostics"][0]["failure_mode"] == "TEXT_MISMATCH"

    out_file = tmp_path / "test_trace.json"
    trace.save_json(out_file)
    assert out_file.exists()

    with open(out_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["sample_id"] == "test_sample"
    assert loaded["diagnostics"][0]["expected_text"] == "B"


def test_02_diagnose_cell_errors_failure_modes():
    """Kiểm tra độ nhạy và tính chính xác của các nhãn chẩn đoán lỗi ô (Failure Modes)."""
    # 1. HEADER_DROP
    exp_h = [["STT", "Tiêu chí đánh giá", "Điểm tối đa"]]
    act_h = [["STT", "", "Điểm tối đa"]]
    diags_h = _diagnose_cell_errors(exp_h, act_h)
    assert any(d.failure_mode == "HEADER_DROP" for d in diags_h)

    # 2. COLUMN_SHIFT
    exp_shift = [
        ["Header 1", "Header 2", "Header 3"],
        ["Val 1", "Val 2", "Val 3"],
    ]
    act_shift = [
        ["Header 1", "Header 2", "Header 3"],
        ["Val 1", "", "Val 2"],  # Val 2 nhảy từ col 1 sang col 2
    ]
    diags_shift = _diagnose_cell_errors(exp_shift, act_shift)
    shift_modes = [d.failure_mode for d in diags_shift]
    assert "COLUMN_SHIFT" in shift_modes or "MISSING_CELL" in shift_modes

    # 3. NOISE_INTRUSION
    exp_noise = [["Tên", "Tuổi"], ["An", "25"]]
    act_noise = [["Tên", "Tuổi"], ["|", "25"]]
    diags_noise = _diagnose_cell_errors(exp_noise, act_noise)
    assert any(d.failure_mode == "NOISE_INTRUSION" for d in diags_noise)

    # 4. MULTI_LINE_SPLIT
    exp_split = [
        ["Hạng mục", "Ghi chú"],
        ["Đào tạo nhân sự và chuyển giao công nghệ", "Hoàn thành"],
    ]
    act_split = [
        ["Hạng mục", "Ghi chú"],
        ["Đào tạo nhân sự và", "Hoàn thành"],
        ["chuyển giao công nghệ", ""],
    ]
    diags_split = _diagnose_cell_errors(exp_split, act_split)
    assert any(d.failure_mode == "MULTI_LINE_SPLIT" for d in diags_split)


def test_03_benchmark_sample_01_bordered_admin():
    """Kiểm tra đánh giá mẫu 01: Bảng hành chính có đầy đủ đường kẻ khung."""
    img, boxes, gt = build_sample_01_bordered_admin()
    trace = evaluate_table_sample(
        sample_id=gt.sample_id,
        input_boxes=boxes,
        ground_truth=gt,
        layout_fn=reconstruct_document_layout,
        image=img,
    )

    assert trace.sample_id == "sample_01_bordered_admin"
    assert trace.category == "bordered"
    assert 0.0 <= trace.metrics["table_detected_exact"] <= 1.0
    assert 0.0 <= trace.metrics["row_count_accuracy"] <= 1.0
    assert 0.0 <= trace.metrics["col_count_accuracy"] <= 1.0
    assert 0.0 <= trace.metrics["cell_exact_match_rate"] <= 1.0
    assert 0.0 <= trace.metrics["cell_content_f1"] <= 1.0
    assert trace.elapsed_ms > 0.0


def test_04_benchmark_sample_02_borderless_financial():
    """Kiểm tra đánh giá mẫu 02: Bảng không đường kẻ (Hóa đơn / Báo giá)."""
    img, boxes, gt = build_sample_02_borderless_financial()
    trace = evaluate_table_sample(
        sample_id=gt.sample_id,
        input_boxes=boxes,
        ground_truth=gt,
        layout_fn=reconstruct_document_layout,
        image=img,
    )

    assert trace.sample_id == "sample_02_borderless_financial"
    assert trace.category == "borderless"
    assert "cell_content_f1" in trace.metrics


def test_05_benchmark_sample_03_merged_cells_header():
    """Kiểm tra đánh giá mẫu 03: Bảng có tiêu đề 2 tầng / ô gộp."""
    img, boxes, gt = build_sample_03_merged_cells_header()
    trace = evaluate_table_sample(
        sample_id=gt.sample_id,
        input_boxes=boxes,
        ground_truth=gt,
        layout_fn=reconstruct_document_layout,
        image=img,
    )

    assert trace.sample_id == "sample_03_merged_cells_header"
    assert trace.category == "merged_cells"
    assert "cell_exact_match_rate" in trace.metrics


def test_06_benchmark_sample_04_dense_7cols():
    """Kiểm tra đánh giá mẫu 04: Bảng 7 cột số liệu dày đặc."""
    img, boxes, gt = build_sample_04_dense_7cols()
    trace = evaluate_table_sample(
        sample_id=gt.sample_id,
        input_boxes=boxes,
        ground_truth=gt,
        layout_fn=reconstruct_document_layout,
        image=img,
    )

    assert trace.sample_id == "sample_04_dense_7cols"
    assert trace.category == "multi_column_dense"
    assert "col_count_accuracy" in trace.metrics


def test_07_benchmark_sample_05_noisy_broken_grid():
    """Kiểm tra đánh giá mẫu 05: Bảng đứt nét, lệch Y ngẫu nhiên và có nhiễu viền."""
    img, boxes, gt = build_sample_05_noisy_broken_grid()
    trace = evaluate_table_sample(
        sample_id=gt.sample_id,
        input_boxes=boxes,
        ground_truth=gt,
        layout_fn=reconstruct_document_layout,
        image=img,
    )

    assert trace.sample_id == "sample_05_noisy_broken_grid"
    assert trace.category == "noisy_scan"
    assert "row_count_accuracy" in trace.metrics


def test_08_full_benchmark_suite_baseline_run_and_trace_export():
    """
    Chạy toàn bộ 5 mẫu chuẩn benchmark, xuất hồ sơ đối chứng Baseline vào
    logs/c2_table_baseline_trace.json và in bảng thống kê chi tiết.
    """
    samples = get_all_benchmark_samples()
    assert len(samples) == 5, f"Kỳ vọng 5 mẫu benchmark, thực tế có {len(samples)}"

    traces: List[TableEvalTrace] = []
    for s_id, (img, boxes, gt) in samples.items():
        tr = evaluate_table_sample(
            sample_id=s_id,
            input_boxes=boxes,
            ground_truth=gt,
            layout_fn=reconstruct_document_layout,
            image=img,
        )
        traces.append(tr)

    assert len(traces) == 5
    summary = generate_benchmark_summary(traces)

    assert summary["num_samples"] == 5
    assert "average_metrics" in summary
    assert "table_detected_exact" in summary["average_metrics"]
    assert "cell_content_f1" in summary["average_metrics"]

    # Lưu kết quả baseline ra logs/c2_table_baseline_trace.json
    logs_dir = Path(ROOT_DIR) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    baseline_trace_path = logs_dir / "c2_table_baseline_trace.json"

    export_payload = {
        "version": "Phase C2 Baseline (Current Layout Engine)",
        "summary": summary,
        "traces": [t.to_dict() for t in traces],
    }

    with open(baseline_trace_path, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=2, ensure_ascii=False)

    assert baseline_trace_path.exists()
    assert baseline_trace_path.stat().st_size > 500

    # In báo cáo bảng trực quan cho kiểm thử viên
    print("\n" + "=" * 95)
    print("PHASE C2 — TABLE & LAYOUT EVALUATION HARNESS: 5-ARCHETYPE BASELINE BENCHMARK")
    print("=" * 95)
    print(f"{'Sample ID':<32} | {'Category':<18} | {'Detect':<7} | {'RowAcc':<7} | {'ColAcc':<7} | {'Exact':<7} | {'F1':<7} | {'Time(ms)'}")
    print("-" * 95)
    for s in summary["sample_summaries"]:
        m = s["metrics"]
        print(f"{s['sample_id']:<32} | {s['category']:<18} | {m['table_detected_exact']:<7.2f} | {m['row_count_accuracy']:<7.2f} | {m['col_count_accuracy']:<7.2f} | {m['cell_exact_match_rate']:<7.2f} | {m['cell_content_f1']:<7.2f} | {s['elapsed_ms']}ms")
    print("-" * 95)
    avg = summary["average_metrics"]
    print(f"{'AVERAGE BASELINE':<32} | {'ALL 5 SAMPLES':<18} | {avg['table_detected_exact']:<7.2f} | {avg['row_count_accuracy']:<7.2f} | {avg['col_count_accuracy']:<7.2f} | {avg['cell_exact_match_rate']:<7.2f} | {avg['cell_content_f1']:<7.2f} |")
    print("=" * 95)
    print(f"Recorded Failure Modes: {summary['failure_mode_frequencies']}")
    print(f"Trace saved to: {baseline_trace_path}")
    print("=" * 95 + "\n")
