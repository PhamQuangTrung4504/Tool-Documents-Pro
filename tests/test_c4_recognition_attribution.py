#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests/test_c4_recognition_attribution.py
========================================
Bộ kiểm thử toàn diện cho Giai đoạn C4 (Phase C4):
1. Bảng từ vựng chuẩn của VietOCR (VIETOCR_STANDARD_VOCAB & find_out_of_vocab_chars).
2. Bộ phân tích chất lượng lát cắt (CropDiagnosticMetrics & analyze_crop_quality).
3. Đệm biên thích ứng bảo toàn ký tự và dấu thanh (adaptive_safe_crop_padding).
4. Tính chỉ số CER (Character Error Rate) và WER (Word Error Rate).
5. Bộ quy kết nguyên nhân lỗi (RecognitionErrorAttributor) trên 4 nhóm lỗi cốt lõi:
   - OUT_OF_VOCAB
   - CROP_OR_BORDER_ARTIFACTS
   - ASPECT_RATIO_ATTENTION_DRIFT
   - INTRINSIC_TONE_CHAR_CONFUSION
6. Xuất báo cáo định lượng và ma trận quyết định kỹ thuật ra logs/c4_recognition_attribution_report.json.
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

# Ensure project root is in path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.recognition_diagnostics import (
    CropDiagnosticMetrics,
    ErrorRootCause,
    RecognitionErrorAttributor,
    SampleAttributionResult,
    VIETOCR_STANDARD_VOCAB,
    VIETOCR_STANDARD_VOCAB_SET,
    analyze_crop_quality,
    compute_cer,
    compute_wer,
    find_out_of_vocab_chars,
)
from app.core.vision_preprocess import adaptive_safe_crop_padding


class TestVietOCRStandardVocab(unittest.TestCase):
    """Kiểm tra bảng từ vựng chuẩn và phát hiện ký tự ngoài từ vựng (OOV)."""

    def test_vocab_constants(self) -> None:
        self.assertEqual(len(VIETOCR_STANDARD_VOCAB), 229)
        self.assertEqual(len(VIETOCR_STANDARD_VOCAB_SET), 229)

        # Ký tự tiếng Việt có dấu chuẩn phải nằm trong vocab
        sample_chars = ["à", "Á", "ả", "Ã", "ạ", "ă", "ằ", "ắ", "ẳ", "ẵ", "ặ", "đ", "Đ", "ơ", "ớ", "ư", "ứ", "ỹ"]
        for ch in sample_chars:
            self.assertIn(ch, VIETOCR_STANDARD_VOCAB_SET)

    def test_find_out_of_vocab_chars_standard_text(self) -> None:
        # Văn bản tiếng Việt thông thường không có ký tự OOV
        text = "Cộng hòa Xã hội Chủ nghĩa Việt Nam - Độc lập Tự do Hạnh phúc 2026!"
        oov = find_out_of_vocab_chars(text)
        self.assertEqual(oov, [])

    def test_find_out_of_vocab_chars_symbols(self) -> None:
        # Các phân số, số mũ và ký tự kỹ thuật đặc thù
        text = "Tỷ lệ ½ so với quy định, diện tích 150m², thể tích 50m³, nhiệt độ 25°C ± 2°C, tiêu chí ≥ 80% • điểm."
        oov = find_out_of_vocab_chars(text)
        self.assertIn("½", oov)
        self.assertIn("²", oov)
        self.assertIn("³", oov)
        self.assertIn("°", oov)
        self.assertIn("±", oov)
        self.assertIn("≥", oov)
        self.assertIn("•", oov)
        # Ký tự '%' và '8' nằm trong vocab nên không có trong oov
        self.assertNotIn("%", oov)
        self.assertNotIn("8", oov)


class TestCropQualityDiagnostics(unittest.TestCase):
    """Kiểm tra bộ phân tích chất lượng lát cắt ảnh."""

    def test_normal_crop_diagnostics(self) -> None:
        # Ảnh trắng kích thước 200x40 (AR = 5.0)
        crop = np.full((40, 200, 3), 255, dtype=np.uint8)
        # Vẽ một số chữ ở giữa (cách xa biên)
        cv2.putText(crop, "VietOCR", (40, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        metrics = analyze_crop_quality(crop)
        self.assertEqual(metrics.height, 40)
        self.assertEqual(metrics.width, 200)
        self.assertAlmostEqual(metrics.aspect_ratio, 5.0, places=1)
        self.assertFalse(metrics.is_high_aspect_ratio)
        self.assertFalse(metrics.has_vertical_border_intrusion)
        self.assertFalse(metrics.has_right_char_clipping)
        self.assertFalse(metrics.has_top_diacritic_clipping)
        self.assertEqual(metrics.flags, [])

    def test_high_aspect_ratio_detection(self) -> None:
        # Ảnh dài hẹp 500x20 (AR = 25.0 > 22.0)
        crop = np.full((20, 500, 3), 255, dtype=np.uint8)
        metrics = analyze_crop_quality(crop)
        self.assertTrue(metrics.is_high_aspect_ratio)
        self.assertIn("HIGH_ASPECT_RATIO", metrics.flags)

    def test_vertical_border_intrusion_detection(self) -> None:
        # Ảnh 40x200 có đường kẻ dọc đen sát mép trái (x=0, chạy suốt 35px >= 75% của 40)
        crop = np.full((40, 200, 3), 255, dtype=np.uint8)
        crop[2:38, 0] = [0, 0, 0]  # Chiều cao 36px >= 0.75 * 40 (30px)

        metrics = analyze_crop_quality(crop)
        self.assertTrue(metrics.has_vertical_border_intrusion)
        self.assertIn("VERTICAL_BORDER_INTRUSION", metrics.flags)

    def test_right_char_clipping_detection(self) -> None:
        # Ảnh 40x200 có nét chữ bị cắt chạm mép phải (nét cao 10px, 10% < ink < 70%)
        crop = np.full((40, 200, 3), 255, dtype=np.uint8)
        crop[15:25, 198:200] = [0, 0, 0]  # 10px trên 40px chiều cao = 25% ink

        metrics = analyze_crop_quality(crop)
        self.assertTrue(metrics.has_right_char_clipping)
        self.assertIn("RIGHT_CHAR_CLIPPING", metrics.flags)

    def test_top_diacritic_clipping_detection(self) -> None:
        # Ảnh 40x200 có dấu thanh chạm mép trên (y=0,1)
        crop = np.full((40, 200, 3), 255, dtype=np.uint8)
        crop[0:2, 80:100] = [0, 0, 0]

        metrics = analyze_crop_quality(crop)
        self.assertTrue(metrics.has_top_diacritic_clipping)
        self.assertIn("TOP_DIACRITIC_CLIPPING", metrics.flags)


class TestAdaptiveSafeCropPadding(unittest.TestCase):
    """Kiểm tra tầng đệm thích ứng bảo toàn nét chữ và dấu thanh."""

    def test_no_padding_needed_on_clean_crop(self) -> None:
        crop = np.full((40, 200, 3), 255, dtype=np.uint8)
        cv2.putText(crop, "Text", (40, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        padded = adaptive_safe_crop_padding(crop, safe_margin=12)
        self.assertEqual(padded.shape, crop.shape)

    def test_safe_padding_added_on_right_clipping(self) -> None:
        crop = np.full((40, 200, 3), 255, dtype=np.uint8)
        crop[15:25, 198:200] = [0, 0, 0]  # Chạm mép phải

        padded = adaptive_safe_crop_padding(crop, safe_margin=12)
        # Chiều rộng phải tăng đúng 12px
        self.assertEqual(padded.shape[0], 40)
        self.assertEqual(padded.shape[1], 212)

    def test_safe_padding_added_on_top_clipping(self) -> None:
        crop = np.full((40, 200, 3), 255, dtype=np.uint8)
        crop[0:2, 80:100] = [0, 0, 0]  # Dấu thanh chạm mép trên

        padded = adaptive_safe_crop_padding(crop, safe_margin=12)
        # Chiều cao phải tăng đúng 12px
        self.assertEqual(padded.shape[0], 52)
        self.assertEqual(padded.shape[1], 200)


class TestCERAndWER(unittest.TestCase):
    """Kiểm tra độ chính xác của hàm tính khoảng cách CER và WER."""

    def test_exact_match(self) -> None:
        self.assertEqual(compute_cer("chi bộ", "chi bộ"), 0.0)
        self.assertEqual(compute_wer("sinh hoạt chi bộ", "sinh hoạt chi bộ"), 0.0)

    def test_cer_substitution(self) -> None:
        # Thay 1 ký tự: 'chi bố' so với 'chi bộ' (độ dài 6 ký tự)
        cer = compute_cer("chi bộ", "chi bố")
        self.assertAlmostEqual(cer, 1.0 / 6.0, places=3)

    def test_wer_substitution(self) -> None:
        # Thay 1 từ: 'sinh hoạt chi bố' so với 'sinh hoạt chi bộ' (4 từ)
        wer = compute_wer("sinh hoạt chi bộ", "sinh hoạt chi bố")
        self.assertAlmostEqual(wer, 1.0 / 4.0, places=3)


class TestRecognitionErrorAttributor(unittest.TestCase):
    """Kiểm tra bộ quy kết nguyên nhân lỗi trên 4 nhóm cốt lõi và xuất báo cáo C4."""

    def setUp(self) -> None:
        self.attributor = RecognitionErrorAttributor()

    def test_group1_out_of_vocab_attribution(self) -> None:
        # Mẫu 1: Ký hiệu phân số ½ bị OCR nhận thành %
        res1 = self.attributor.attribute_sample(
            sample_id="oov_01_fraction_half",
            ground_truth="Thời gian sinh hoạt chi bộ chỉ bằng ½ so với quy định",
            raw_pred="Thời gian sinh hoạt chi bộ chỉ bằng % so với quy định",
            post_pred="Thời gian sinh hoạt chi bộ chỉ bằng ½ so với quy định",
        )
        self.assertEqual(res1.root_cause, ErrorRootCause.OUT_OF_VOCAB)
        self.assertTrue(res1.is_resolved_stage2)

        # Mẫu 2: Số mũ diện tích m² bị nhận thành m2
        res2 = self.attributor.attribute_sample(
            sample_id="oov_02_superscript_m2",
            ground_truth="Diện tích chuyển mục đích sang đất ở: 170m²",
            raw_pred="Diện tích chuyển mục đích sang đất ở: 170m2",
            post_pred="Diện tích chuyển mục đích sang đất ở: 170m²",
        )
        self.assertEqual(res2.root_cause, ErrorRootCause.OUT_OF_VOCAB)
        self.assertTrue(res2.is_resolved_stage2)

        # Mẫu 3: Nhiệt độ °C bị nhận thành oC
        res3 = self.attributor.attribute_sample(
            sample_id="oov_03_degree_celsius",
            ground_truth="Nhiệt độ bảo quản tài liệu: 25°C",
            raw_pred="Nhiệt độ bảo quản tài liệu: 25 oC",
            post_pred="Nhiệt độ bảo quản tài liệu: 25°C",
        )
        self.assertEqual(res3.root_cause, ErrorRootCause.OUT_OF_VOCAB)
        self.assertTrue(res3.is_resolved_stage2)

    def test_group2_crop_border_artifacts_attribution(self) -> None:
        # Mẫu 4: Cắt hụt mép phải làm mất chữ số cuối 'năm 202' -> 'năm 2026'
        crop_clipped = np.full((36, 180, 3), 255, dtype=np.uint8)
        crop_clipped[10:25, 178:180] = [0, 0, 0]  # Chạm mép phải

        res4 = self.attributor.attribute_sample(
            sample_id="crop_01_right_clipped_year",
            ground_truth="Kế hoạch công tác năm 2026",
            raw_pred="Kế hoạch công tác năm 202",
            crop_bgr=crop_clipped,
            repaired_crop_pred="Kế hoạch công tác năm 2026",
            post_pred="Kế hoạch công tác năm 2026",
        )
        self.assertEqual(res4.root_cause, ErrorRootCause.CROP_OR_BORDER_ARTIFACTS)
        self.assertTrue(res4.is_resolved_stage1)

        # Mẫu 5: Dính đường kẻ viền bảng dọc tạo ký tự '[' hoặc '|'
        crop_border = np.full((40, 220, 3), 255, dtype=np.uint8)
        crop_border[:, 0] = [0, 0, 0]  # Đường kẻ bảng dọc bên trái

        res5 = self.attributor.attribute_sample(
            sample_id="crop_02_vertical_border_line",
            ground_truth="I. Thành phần tham dự",
            raw_pred="[ I. Thành phần tham dự |",
            crop_bgr=crop_border,
            repaired_crop_pred="I. Thành phần tham dự",
            post_pred="I. Thành phần tham dự",
        )
        self.assertEqual(res5.root_cause, ErrorRootCause.CROP_OR_BORDER_ARTIFACTS)
        self.assertTrue(res5.is_resolved_stage1)

        # Mẫu 6: Cắt hụt dấu thanh trên chữ hoa 'ỦY BAN'
        crop_top = np.full((36, 150, 3), 255, dtype=np.uint8)
        crop_top[0:2, 20:35] = [0, 0, 0]

        res6 = self.attributor.attribute_sample(
            sample_id="crop_03_top_diacritic_clipped",
            ground_truth="ỦY BAN NHÂN DÂN",
            raw_pred="UY BAN NHÂN DÂN",
            crop_bgr=crop_top,
            repaired_crop_pred="ỦY BAN NHÂN DÂN",
            post_pred="ỦY BAN NHÂN DÂN",
        )
        self.assertEqual(res6.root_cause, ErrorRootCause.CROP_OR_BORDER_ARTIFACTS)
        self.assertTrue(res6.is_resolved_stage1)

        # Mẫu 7: Cắt cụt chữ cuối câu 'không c' -> 'không có.'
        crop_clipped7 = np.full((36, 180, 3), 255, dtype=np.uint8)
        crop_clipped7[10:25, 178:180] = [0, 0, 0]

        res7 = self.attributor.attribute_sample(
            sample_id="crop_04_clipped_sentence_end",
            ground_truth="Giới thiệu đại biểu: không có.",
            raw_pred="Giới thiệu đại biểu: không c",
            crop_bgr=crop_clipped7,
            repaired_crop_pred="Giới thiệu đại biểu: không có.",
            post_pred="Giới thiệu đại biểu: không có.",
        )
        self.assertEqual(res7.root_cause, ErrorRootCause.CROP_OR_BORDER_ARTIFACTS)
        self.assertTrue(res7.is_resolved_stage1)

    def test_group3_aspect_ratio_attention_drift_attribution(self) -> None:
        # Mẫu 8: Dòng chấm lửng quá dài làm trôi attention của Transformer
        crop_long = np.full((18, 520, 3), 255, dtype=np.uint8)  # AR = 28.8 > 22.0

        res8 = self.attributor.attribute_sample(
            sample_id="ar_01_dotted_line_drift",
            ground_truth="Họ và tên: .............................................................",
            raw_pred="Họ và tên: ............ Họ và tên: ......",
            crop_bgr=crop_long,
            repaired_crop_pred="Họ và tên: .............................................................",
            post_pred="Họ và tên: .............................................................",
        )
        self.assertEqual(res8.root_cause, ErrorRootCause.ASPECT_RATIO_ATTENTION_DRIFT)

    def test_group4_intrinsic_tone_char_confusion_attribution(self) -> None:
        # Mẫu 9: Nhầm dấu thanh 'bộ' -> 'bố' trên ảnh chất lượng bình thường
        crop_normal = np.full((36, 200, 3), 255, dtype=np.uint8)

        res9 = self.attributor.attribute_sample(
            sample_id="intrinsic_01_tone_bo",
            ground_truth="Đồng chí Bí thư chi bộ chủ trì",
            raw_pred="Đồng chí Bí thư chi bố chủ trì",
            crop_bgr=crop_normal,
            repaired_crop_pred="Đồng chí Bí thư chi bố chủ trì",
            post_pred="Đồng chí Bí thư chi bố chủ trì",
        )
        self.assertEqual(res9.root_cause, ErrorRootCause.INTRINSIC_TONE_CHAR_CONFUSION)
        self.assertTrue(res9.remaining_error)

        # Mẫu 10: Nhầm dấu hỏi/nặng 'dự' -> 'dư'
        res10 = self.attributor.attribute_sample(
            sample_id="intrinsic_02_tone_du",
            ground_truth="Tỷ lệ đảng viên dự sinh hoạt chi bộ",
            raw_pred="Tỷ lệ đảng viên dư sinh hoạt chi bộ",
            crop_bgr=crop_normal,
            repaired_crop_pred="Tỷ lệ đảng viên dư sinh hoạt chi bộ",
            post_pred="Tỷ lệ đảng viên dư sinh hoạt chi bộ",
        )
        self.assertEqual(res10.root_cause, ErrorRootCause.INTRINSIC_TONE_CHAR_CONFUSION)
        self.assertTrue(res10.remaining_error)

        # Mẫu 11: Nhầm ký tự 'Thiên Ưng' -> 'Thiên Ung'
        res11 = self.attributor.attribute_sample(
            sample_id="intrinsic_03_char_ung",
            ground_truth="Công ty TNHH Thiên Ưng",
            raw_pred="Công ty TNHH Thiên Ung",
            crop_bgr=crop_normal,
            repaired_crop_pred="Công ty TNHH Thiên Ung",
            post_pred="Công ty TNHH Thiên Ung",
        )
        self.assertEqual(res11.root_cause, ErrorRootCause.INTRINSIC_TONE_CHAR_CONFUSION)
        self.assertTrue(res11.remaining_error)

        # Mẫu 12: Nhầm ký tự số '05' -> '0s' (được post-processor Stage 2 phục hồi)
        res12 = self.attributor.attribute_sample(
            sample_id="intrinsic_04_char_0s",
            ground_truth="Điểm tối đa: 05",
            raw_pred="Điểm tối đa: 0s",
            crop_bgr=crop_normal,
            repaired_crop_pred="Điểm tối đa: 0s",
            post_pred="Điểm tối đa: 05",
        )
        # Vì đây là nhầm glyph trong vocab và không có OOV
        self.assertEqual(res12.root_cause, ErrorRootCause.INTRINSIC_TONE_CHAR_CONFUSION)
        self.assertTrue(res12.is_resolved_stage2)

        # Mẫu 13: Nhầm dấu 'họp' -> 'hơp'
        res13 = self.attributor.attribute_sample(
            sample_id="intrinsic_05_tone_hop",
            ground_truth="Nội dung cuộc họp chi bộ",
            raw_pred="Nội dung cuộc hơp chi bộ",
            crop_bgr=crop_normal,
            repaired_crop_pred="Nội dung cuộc hơp chi bộ",
            post_pred="Nội dung cuộc họp chi bộ",
        )
        self.assertEqual(res13.root_cause, ErrorRootCause.INTRINSIC_TONE_CHAR_CONFUSION)
        self.assertTrue(res13.is_resolved_stage2)

    def test_full_attribution_report_export(self) -> None:
        """Chạy tổng hợp 14 mẫu kiểm thử đại diện và xuất báo cáo ra file JSON."""
        crop_clipped = np.full((36, 180, 3), 255, dtype=np.uint8)
        crop_clipped[10:25, 178:180] = [0, 0, 0]

        crop_border = np.full((40, 220, 3), 255, dtype=np.uint8)
        crop_border[:, 0] = [0, 0, 0]

        crop_top = np.full((36, 150, 3), 255, dtype=np.uint8)
        crop_top[0:2, 20:35] = [0, 0, 0]

        crop_long = np.full((18, 520, 3), 255, dtype=np.uint8)
        crop_normal = np.full((36, 200, 3), 255, dtype=np.uint8)

        samples_data = [
            # Group 1: OUT_OF_VOCAB
            ("oov_01", "Thời gian sinh hoạt chi bộ chỉ bằng ½ so với quy định", "Thời gian sinh hoạt chi bộ chỉ bằng % so với quy định", None, "Thời gian sinh hoạt chi bộ chỉ bằng % so với quy định", "Thời gian sinh hoạt chi bộ chỉ bằng ½ so với quy định"),
            ("oov_02", "Diện tích chuyển mục đích sang đất ở: 170m²", "Diện tích chuyển mục đích sang đất ở: 170m2", None, "Diện tích chuyển mục đích sang đất ở: 170m2", "Diện tích chuyển mục đích sang đất ở: 170m²"),
            ("oov_03", "Nhiệt độ bảo quản tài liệu: 25°C", "Nhiệt độ bảo quản tài liệu: 25 oC", None, "Nhiệt độ bảo quản tài liệu: 25 oC", "Nhiệt độ bảo quản tài liệu: 25°C"),

            # Group 2: CROP_OR_BORDER_ARTIFACTS
            ("crop_01", "Kế hoạch công tác năm 2026", "Kế hoạch công tác năm 202", crop_clipped, "Kế hoạch công tác năm 2026", "Kế hoạch công tác năm 2026"),
            ("crop_02", "I. Thành phần tham dự", "[ I. Thành phần tham dự |", crop_border, "I. Thành phần tham dự", "I. Thành phần tham dự"),
            ("crop_03", "ỦY BAN NHÂN DÂN", "UY BAN NHÂN DÂN", crop_top, "ỦY BAN NHÂN DÂN", "ỦY BAN NHÂN DÂN"),
            ("crop_04", "Giới thiệu đại biểu: không có.", "Giới thiệu đại biểu: không c", crop_clipped, "Giới thiệu đại biểu: không có.", "Giới thiệu đại biểu: không có."),

            # Group 3: ASPECT_RATIO_ATTENTION_DRIFT
            ("ar_01", "Họ và tên: .............................................................", "Họ và tên: ............ Họ và tên: ......", crop_long, "Họ và tên: .............................................................", "Họ và tên: ............................................................."),

            # Group 4: INTRINSIC_TONE_CHAR_CONFUSION
            ("intrinsic_01", "Đồng chí Bí thư chi bộ chủ trì", "Đồng chí Bí thư chi bố chủ trì", crop_normal, "Đồng chí Bí thư chi bố chủ trì", "Đồng chí Bí thư chi bố chủ trì"),
            ("intrinsic_02", "Tỷ lệ đảng viên dự sinh hoạt chi bộ", "Tỷ lệ đảng viên dư sinh hoạt chi bộ", crop_normal, "Tỷ lệ đảng viên dư sinh hoạt chi bộ", "Tỷ lệ đảng viên dư sinh hoạt chi bộ"),
            ("intrinsic_03", "Công ty TNHH Thiên Ưng", "Công ty TNHH Thiên Ung", crop_normal, "Công ty TNHH Thiên Ung", "Công ty TNHH Thiên Ung"),
            ("intrinsic_04", "Điểm tối đa: 05", "Điểm tối đa: 0s", crop_normal, "Điểm tối đa: 0s", "Điểm tối đa: 05"),
            ("intrinsic_05", "Nội dung cuộc họp chi bộ", "Nội dung cuộc hơp chi bộ", crop_normal, "Nội dung cuộc hơp chi bộ", "Nội dung cuộc họp chi bộ"),
            ("correct_01", "Cộng hòa xã hội chủ nghĩa Việt Nam", "Cộng hòa xã hội chủ nghĩa Việt Nam", crop_normal, "Cộng hòa xã hội chủ nghĩa Việt Nam", "Cộng hòa xã hội chủ nghĩa Việt Nam"),
        ]

        evaluated_results: List[SampleAttributionResult] = []
        for s_id, gt, raw, crop, s1, s2 in samples_data:
            res = self.attributor.attribute_sample(
                sample_id=s_id,
                ground_truth=gt,
                raw_pred=raw,
                crop_bgr=crop,
                repaired_crop_pred=s1,
                post_pred=s2,
            )
            evaluated_results.append(res)

        report = self.attributor.generate_attribution_report(evaluated_results)

        # Ghi báo cáo ra logs/c4_recognition_attribution_report.json
        logs_dir = ROOT_DIR / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        report_file = logs_dir / "c4_recognition_attribution_report.json"

        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        self.assertTrue(report_file.exists())
        self.assertEqual(report["summary"]["total_samples"], 14)
        self.assertEqual(report["summary"]["total_errors"], 13)

        # Kiểm tra tỷ lệ lỗi CER giảm dần qua 3 giai đoạn
        s0_cer = report["summary"]["average_stage0_cer"]
        s1_cer = report["summary"]["average_stage1_cer"]
        s2_cer = report["summary"]["average_stage2_cer"]
        self.assertGreater(s0_cer, s1_cer)
        self.assertGreater(s1_cer, s2_cer)

        # Quyết định kỹ thuật
        dec = report["decision_matrix"]
        self.assertEqual(dec["decision"], "PROCEED_TO_DIRECTION_B_PROBABILISTIC_LANGUAGE_MODEL")
        self.assertFalse(dec["fine_tune_vietocr_required"])


if __name__ == "__main__":
    unittest.main()
