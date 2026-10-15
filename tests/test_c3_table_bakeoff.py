#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests.test_c3_table_bakeoff
===========================
Bộ kiểm thử đối sánh định lượng đa thuật toán nhận diện bảng (Multi-Engine Bake-Off):
1. V1_Baseline (reconstruct_document_layout hiện tại)
2. MorphologicalGrid (MorphologicalGridTableEngine)
3. SpatialProjectionGraph (SpatialProjectionGraphEngine)
4. UnifiedHybrid (UnifiedHybridTableEngine)

Đo lường trên cả 5 mẫu chuẩn của C2 và xuất kết quả ra logs/c3_table_bakeoff_comparison.json.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pytest
from app.core.layout_analyzer import reconstruct_document_layout
from app.core.table_engines import (
    morphological_grid_engine,
    spatial_projection_engine,
    unified_hybrid_engine,
)
from app.core.table_evaluator import (
    TableEvalTrace,
    evaluate_table_sample,
    generate_benchmark_summary,
)
from tests.benchmarks.table_benchmark_samples import get_all_benchmark_samples


class TestC3TableRecognitionBakeoff:
    """Đối sánh đa thuật toán trên 5 mẫu bảng benchmark chuẩn."""

    @classmethod
    def setup_class(cls):
        cls.samples = get_all_benchmark_samples()
        cls.log_dir = Path("logs")
        cls.log_dir.mkdir(parents=True, exist_ok=True)
        cls.bakeoff_log_path = cls.log_dir / "c3_table_bakeoff_comparison.json"

    def test_01_run_multi_engine_bakeoff_and_export_comparison(self):
        """Chạy cả 4 phương pháp trên 5 mẫu chuẩn và xuất báo cáo so sánh định lượng."""
        engines: Dict[str, Any] = {
            "V1_Baseline": reconstruct_document_layout,
            "MorphologicalGrid": morphological_grid_engine,
            "SpatialProjectionGraph": spatial_projection_engine,
            "UnifiedHybrid": unified_hybrid_engine,
        }

        bakeoff_results: Dict[str, Any] = {
            "version": "Phase C3 Multi-Engine Table Bake-Off",
            "engines": {},
            "comparison_table": [],
        }

        engine_summaries: Dict[str, Dict[str, Any]] = {}

        for engine_name, engine_fn in engines.items():
            traces: List[TableEvalTrace] = []
            for sample_id, (img, boxes, gt) in self.samples.items():
                trace = evaluate_table_sample(
                    sample_id=sample_id,
                    input_boxes=boxes,
                    ground_truth=gt,
                    layout_fn=engine_fn,
                    image=img,
                )
                traces.append(trace)

            summary = generate_benchmark_summary(traces)
            engine_summaries[engine_name] = summary
            bakeoff_results["engines"][engine_name] = {
                "summary": summary,
                "traces": [t.to_dict() for t in traces],
            }

        # Tạo bảng so sánh tổng hợp (Comparison Table)
        sample_ids = list(self.samples.keys())
        for sid in sample_ids:
            row_data: Dict[str, Any] = {"sample_id": sid}
            for engine_name in engines.keys():
                sample_summary = next(
                    (s for s in engine_summaries[engine_name]["sample_summaries"] if s["sample_id"] == sid),
                    None,
                )
                if sample_summary:
                    row_data[f"{engine_name}_exact_match"] = sample_summary["metrics"]["cell_exact_match_rate"]
                    row_data[f"{engine_name}_f1"] = sample_summary["metrics"]["cell_content_f1"]
                    row_data[f"{engine_name}_errors"] = sample_summary["error_count"]
            bakeoff_results["comparison_table"].append(row_data)

        # Xuất ra file JSON
        with open(self.bakeoff_log_path, "w", encoding="utf-8") as f:
            json.dump(bakeoff_results, f, ensure_ascii=False, indent=2)

        assert self.bakeoff_log_path.exists(), "Báo cáo bakeoff phải được ghi thành công"

        # Kiểm tra thuật toán chiến thắng
        hybrid_summary = engine_summaries["UnifiedHybrid"]
        assert hybrid_summary["average_metrics"]["cell_exact_match_rate"] >= 0.98, (
            f"UnifiedHybrid phải đạt >= 98% exact match, thực tế: {hybrid_summary['average_metrics']['cell_exact_match_rate']}"
        )

    def test_02_spatial_projection_graph_engine_100_percent_on_all_samples(self):
        """Kiểm tra SpatialProjectionGraphEngine đạt 100% trên cả 5 mẫu chuẩn."""
        for sample_id, (img, boxes, gt) in self.samples.items():
            trace = evaluate_table_sample(
                sample_id=sample_id,
                input_boxes=boxes,
                ground_truth=gt,
                layout_fn=spatial_projection_engine,
                image=img,
            )
            assert trace.metrics["table_detected_exact"] == 1.0, f"Phát hiện bảng thất bại trên {sample_id}"
            assert trace.metrics["row_count_accuracy"] == 1.0, f"Lệch số hàng trên {sample_id}"
            assert trace.metrics["col_count_accuracy"] == 1.0, f"Lệch số cột trên {sample_id}"
            assert trace.metrics["cell_exact_match_rate"] == 1.0, f"Lệch nội dung ô trên {sample_id}: {trace.diagnostics}"
            assert trace.metrics["cell_content_f1"] == 1.0, f"F1 không đạt 1.0 trên {sample_id}"
            assert len(trace.diagnostics) == 0, f"Có lỗi ô trên {sample_id}: {trace.diagnostics}"

    def test_03_unified_hybrid_engine_100_percent_on_all_samples(self):
        """Kiểm tra UnifiedHybridTableEngine đạt 100% trên cả 5 mẫu chuẩn."""
        for sample_id, (img, boxes, gt) in self.samples.items():
            trace = evaluate_table_sample(
                sample_id=sample_id,
                input_boxes=boxes,
                ground_truth=gt,
                layout_fn=unified_hybrid_engine,
                image=img,
            )
            assert trace.metrics["table_detected_exact"] == 1.0
            assert trace.metrics["row_count_accuracy"] == 1.0
            assert trace.metrics["col_count_accuracy"] == 1.0
            assert trace.metrics["cell_exact_match_rate"] == 1.0
            assert trace.metrics["cell_content_f1"] == 1.0
            assert len(trace.diagnostics) == 0
