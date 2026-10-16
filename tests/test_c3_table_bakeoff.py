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

    def test_04_real_scorecard_image_e51c85_regression_lock(self):
        """Kiểm tra chống hồi quy bảng điểm thực tế (image_e51c85.jpg) với tiêu đề 2 dòng và điểm lệch Y."""
        from app.core.models import OCRBox
        from app.core.text_postprocessor import (
            ContextSymbolRecovery,
            VietnameseTextPostProcessor,
            final_document_cleanup,
        )

        def _box(txt: str, x1: int, y1: int, x2: int, y2: int) -> OCRBox:
            return OCRBox(
                text=txt,
                confidence=0.98,
                bbox=[[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                raw_text=txt,
            )

        scorecard_boxes = [
            _box("STT", 30, 200, 70, 225),
            _box("Tiêu chí đánh giá", 100, 200, 350, 225),
            _box("Điểm tối", 460, 190, 520, 210),
            _box("đa", 475, 215, 505, 235),
            _box("Điểm tự", 560, 190, 620, 210),
            _box("chấm", 570, 215, 610, 235),
            _box("Ghi chú", 670, 200, 740, 225),
            # Row 1 (main 1)
            _box("1", 45, 245, 55, 270),
            _box("Chấp hành thời gian, thời lượng sinh hoạt", 90, 245, 420, 270),
            _box("05", 475, 245, 505, 270),
            _box("05", 575, 245, 605, 270),
            # Row 2 (sub with +)
            _box("+ Chi bộ dưới 30 đảng viên ít nhất 90 phút; chi bộ trên 30 đảng viên ít nhất 120 phút", 100, 280, 440, 305),
            _box("05", 475, 282, 505, 307),
            _box("05", 575, 282, 605, 307),
            # Row 3 (sub with -)
            _box("- Thời gian sinh hoạt chi bộ ít hơn quy định 15 phút", 100, 315, 430, 340),
            _box("04", 475, 317, 505, 342),
            # Row 4 (sub with % noise)
            _box("% Thời gian sinh hoạt chi bộ ít hơn quy định 15 - 30 phút", 100, 350, 435, 375),
            _box("03", 475, 352, 505, 377),
            # Row 5 (sub with fraction %)
            _box("- Thời gian sinh hoạt chi bộ chỉ bằng % so với quy định", 100, 385, 435, 410),
            _box("02", 475, 387, 505, 412),
            # Row 6 (sub with fraction %)
            _box("+ Thời gian sinh hoạt chi bộ dưới % so với quy định", 100, 420, 435, 445),
            _box("01", 475, 422, 505, 447),
            # Row 7 (main 2)
            _box("2", 45, 455, 55, 480),
            _box("Tỷ lệ đảng viên dự sinh hoạt chi bộ", 90, 455, 350, 480),
            _box("05", 475, 455, 505, 480),
            _box("05", 575, 455, 605, 480),
            # Row 8 (sub with +)
            _box("+ Từ 90% trở lên", 100, 490, 220, 515),
            _box("04", 475, 492, 505, 517),
            _box("04", 575, 492, 605, 517),
            # Row 9 (sub with -)
            _box("- Từ 75% đến dưới 90%", 100, 525, 250, 550),
            _box("03", 475, 527, 505, 552),
            # Row 10 (sub with % noise)
            _box("% Từ 65% đến dưới 75%", 100, 560, 250, 585),
            _box("02", 475, 562, 505, 587),
            # Row 11 (sub with -)
            _box("- Dưới 60% hoặc có đảng viên vắng không có lý do", 100, 595, 430, 620),
            _box("01", 475, 597, 505, 622),
            # Row 12 (sub with +)
            _box("+ Không có đảng viên vắng mặt có lý do quá 03 lần liên tiếp trong năm", 100, 630, 445, 655),
            _box("01", 475, 632, 505, 657),
            _box("01", 575, 632, 605, 657),
            # Footnote
            _box("- Trừ trường hợp được miễn công tác, sinh hoạt đảng theo quy định và đảng viên trong lực lượng vũ trang thực hiện nhiệm vụ đặc biệt", 30, 680, 750, 705),
        ]

        layout_res = reconstruct_document_layout(scorecard_boxes)
        assert len(layout_res.tables) == 1
        tbl = layout_res.tables[0]
        assert tbl.num_cols == 5
        assert tbl.num_rows == 13

        # Tái hiện pipeline định dạng dòng văn bản của recognize()
        lines: List[str] = []
        for row in tbl.grid:
            if not row[0].strip():
                non_empty = [c.strip() for c in row[1:] if c.strip()]
                lines.append(" | ".join(non_empty))
            else:
                lines.append(" | ".join(c.strip() for c in row if c.strip()))

        # Dòng footnote dưới bảng
        for l in layout_res.grouped_lines:
            txt_l = " ".join(b.text.strip() for b in l if b.text.strip())
            if "Trừ trường hợp" in txt_l:
                lines.append(txt_l)

        full_doc = "\n".join(lines)
        recovered_lines, _ = ContextSymbolRecovery.recover_document_symbols(full_doc.split("\n"))
        full_doc = "\n".join(recovered_lines)
        full_doc = VietnameseTextPostProcessor.clean_and_restore(full_doc)
        full_doc = final_document_cleanup(full_doc)

        expected_lines = [
            "STT | Tiêu chí đánh giá | Điểm tối đa | Điểm tự chấm | Ghi chú",
            "1 | Chấp hành thời gian, thời lượng sinh hoạt | 05 | 05",
            "+ Chi bộ dưới 30 đảng viên ít nhất 90 phút; chi bộ trên 30 đảng viên ít nhất 120 phút | 05 | 05",
            "+ Thời gian sinh hoạt chi bộ ít hơn quy định 15 phút | 04",
            "+ Thời gian sinh hoạt chi bộ ít hơn quy định 15 - 30 phút | 03",
            "+ Thời gian sinh hoạt chi bộ chỉ bằng ½ so với quy định | 02",
            "+ Thời gian sinh hoạt chi bộ dưới ½ so với quy định | 01",
            "2 | Tỷ lệ đảng viên dự sinh hoạt chi bộ | 05 | 05",
            "+ Từ 90% trở lên | 04 | 04",
            "+ Từ 75% đến dưới 90% | 03",
            "+ Từ 65% đến dưới 75% | 02",
            "+ Dưới 60% hoặc có đảng viên vắng không có lý do | 01",
            "+ Không có đảng viên vắng mặt có lý do quá 03 lần liên tiếp trong năm | 01 | 01",
            "* Trừ trường hợp được miễn công tác, sinh hoạt đảng theo quy định và đảng viên trong lực lượng vũ trang thực hiện nhiệm vụ đặc biệt",
        ]

        actual_lines = [ln.strip() for ln in full_doc.splitlines() if ln.strip()]
        for idx, (exp, act) in enumerate(zip(expected_lines, actual_lines)):
            assert act == exp, f"Lệch dòng {idx + 1}:\nExpected: {exp}\nActual:   {act}"
        assert len(actual_lines) == len(expected_lines)

    def test_05_header_ng_noise_line_filtered(self):
        """Kiểm tra loại bỏ dòng rác NG / N / G dưới tên cơ quan (image_e4c688.jpg)."""
        from app.core.text_postprocessor import final_document_cleanup, is_hallucinated_or_noise_line

        assert is_hallucinated_or_noise_line("NG") is True
        assert is_hallucinated_or_noise_line("N") is True
        assert is_hallucinated_or_noise_line("G") is True

        sample_header_text = (
            "ĐẢNG ỦY XÃ PHƯỚC HẢI\n"
            "CHI BỘ VĂN PHÒNG HĐND\n"
            "VÀ UBND XÃ\n"
            "NG\n"
            "Số 01-BB/CB\n"
            "ĐẢNG CỘNG SẢN VIỆT NAM\n"
            "Phước Hải, ngày 05 tháng 10 năm 2026\n"
            "BIÊN BẢN\n"
        )
        cleaned = final_document_cleanup(sample_header_text)
        assert "NG\n" not in cleaned
        assert "NG" not in [ln.strip() for ln in cleaned.splitlines()]
        assert "VÀ UBND XÃ\nSố 01-BB/CB" in cleaned
