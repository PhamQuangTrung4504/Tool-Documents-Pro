#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests.test_modular_imports
==========================
Kiểm thử tính tương thích ngược và tính toàn vẹn của kiến trúc module hóa Phase C1:
- Import trực tiếp từ từng module đơn nhiệm:
  + app.core.models
  + app.core.vision_preprocess
  + app.core.text_postprocessor
  + app.core.layout_analyzer
  + app.core.docx_exporter
- Import qua facade tương thích ngược app.core.ocr_engine
- Xác nhận 100% object reference đồng nhất (imported_from_module is imported_from_facade).
"""

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pytest


def test_models_modular_imports():
    """Kiểm tra các cấu trúc dữ liệu và Enum từ models.py."""
    import app.core.models as mod
    import app.core.ocr_engine as facade

    symbols = [
        "OCRBox",
        "OCRTable",
        "ParallelHeader",
        "HeaderDetectionResult",
        "LayoutResult",
        "OCRResult",
        "SymbolDecision",
    ]
    for sym in symbols:
        assert hasattr(mod, sym), f"models.py thiếu {sym}"
        assert hasattr(facade, sym), f"ocr_engine facade thiếu {sym}"
        assert getattr(mod, sym) is getattr(facade, sym), f"{sym} không đồng nhất giữa models và ocr_engine"


def test_vision_preprocess_modular_imports():
    """Kiểm tra các hàm xử lý thị giác từ vision_preprocess.py."""
    import app.core.vision_preprocess as mod
    import app.core.ocr_engine as facade

    symbols = [
        "get_rotate_crop_image",
        "clean_crop_borders",
        "pad_small_crop_for_vietocr",
        "is_decorative_pattern_crop",
        "split_dotted_line_crop",
        "estimate_and_correct_skew",
        "filter_border_noise_boxes",
        "assess_image_quality",
    ]
    for sym in symbols:
        assert hasattr(mod, sym), f"vision_preprocess.py thiếu {sym}"
        assert hasattr(facade, sym), f"ocr_engine facade thiếu {sym}"
        assert getattr(mod, sym) is getattr(facade, sym), f"{sym} không đồng nhất giữa vision_preprocess và ocr_engine"


def test_text_postprocessor_modular_imports():
    """Kiểm tra các hàm hậu xử lý văn bản từ text_postprocessor.py."""
    import app.core.text_postprocessor as mod
    import app.core.ocr_engine as facade

    symbols = [
        "is_decorative_or_hallucinated",
        "is_hallucinated_or_noise_line",
        "recover_emails_and_urls",
        "format_progress_table_row",
        "ContextSymbolRecovery",
        "preserve_casing",
        "clean_unmatched_closing_brackets",
        "clean_orphaned_brackets",
        "final_document_cleanup",
        "VietnameseTextPostProcessor",
        "NOISE_SINGLE_CHARS",
        "RE_DECORATIVE_LINE_CHARS",
        "RE_REPEATED_WORD",
        "ROMAN_HEADING_KEYWORDS",
    ]
    for sym in symbols:
        assert hasattr(mod, sym), f"text_postprocessor.py thiếu {sym}"
        assert hasattr(facade, sym), f"ocr_engine facade thiếu {sym}"
        assert getattr(mod, sym) is getattr(facade, sym), f"{sym} không đồng nhất giữa text_postprocessor và ocr_engine"


def test_layout_analyzer_modular_imports():
    """Kiểm tra các hàm phân tích bố cục và tái dựng bảng biểu từ layout_analyzer.py."""
    import app.core.layout_analyzer as mod
    import app.core.ocr_engine as facade

    symbols = [
        "_box_info_list",
        "_group_single_column_lines",
        "_group_boxes_sliding_window",
        "join_ocr_line_boxes",
        "detect_parallel_administrative_header",
        "detect_and_reconstruct_table",
        "reconstruct_document_layout",
        "group_boxes_into_lines",
        "sort_boxes_reading_order",
        "format_progress_table_row",
    ]
    for sym in symbols:
        assert hasattr(mod, sym), f"layout_analyzer.py thiếu {sym}"
        assert hasattr(facade, sym), f"ocr_engine facade thiếu {sym}"
        assert getattr(mod, sym) is getattr(facade, sym), f"{sym} không đồng nhất giữa layout_analyzer và ocr_engine"


def test_docx_exporter_modular_imports():
    """Kiểm tra hàm xuất tài liệu DOCX từ docx_exporter.py."""
    import app.core.docx_exporter as mod
    import app.core.ocr_engine as facade

    assert hasattr(mod, "export_to_docx"), "docx_exporter.py thiếu export_to_docx"
    assert hasattr(facade, "export_to_docx"), "ocr_engine facade thiếu export_to_docx"
    assert mod.export_to_docx is facade.export_to_docx, "export_to_docx không đồng nhất"


def test_ocr_engine_core_symbols():
    """Kiểm tra các symbol cốt lõi và class OCREngine trong ocr_engine.py."""
    import app.core.ocr_engine as facade

    symbols = [
        "LANGUAGE_MAP",
        "DEFAULT_LANG",
        "apply_paddlex_patches",
        "get_display_languages",
        "get_lang_code",
        "get_global_detector",
        "get_global_vietocr",
        "OCREngine",
    ]
    for sym in symbols:
        assert hasattr(facade, sym), f"ocr_engine facade thiếu {sym}"
