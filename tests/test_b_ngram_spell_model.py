#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests/test_b_ngram_spell_model.py
=================================
Bộ kiểm thử toàn diện cho Giai đoạn B (Phase B):
1. Ma trận khoảng cách thị giác OCR (VisualOCRConfusionModel & visual_ocr_distance):
   - Khoảng cách giữa các từ cùng gốc khác dấu thanh (bộ vs bố, dự vs dư) < 0.20.
   - Khoảng cách giữa các từ nhầm dấu phụ nguyên âm hoặc thanh (họp vs hơp, đơn vs đon, ưng vs ung) <= 0.32.
   - Khoảng cách giữa 2 từ khác hẳn nhau (bộ vs nhà) >= 1.0.
2. Kiểm tra phát hiện lỗi ngữ cảnh đa lĩnh vực (Multi-Domain Contextual Detection):
   - Hành chính: 'sinh hoạt chi bố' -> 'bộ', 'đảng viên dư sinh hoạt' -> 'dự'.
   - Thương mại / Kế toán: 'xuất hóa đon giá trị' -> 'đơn', 'tổng công thanh toán' -> 'cộng'.
   - Giáo dục / Khoa học: 'kết qua xét nghiệm' -> 'quả', 'làm bài tạp về nhà' -> 'tập'.
   - Kỹ thuật / Sản xuất: 'sản phâm công nghệ' -> 'phẩm'.
3. Bảo toàn Zero False Positives trên văn bản chuẩn đa lĩnh vực (Biên bản, Hóa đơn, Hợp đồng, CNTT).
4. Kiểm tra cấu trúc chỉ mục N-gram siêu tốc O(1) và logic phân rã âm tiết tiếng Việt.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from typing import List

# Ensure project root is in path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.ngram_language_model import (
    ContextualCandidateScore,
    VietnameseNgramLanguageModel,
    VisualOCRConfusionModel,
    get_default_ngram_model,
    visual_ocr_distance,
)
from app.core.spell_corrector import (
    SpellError,
    VietnameseSpellChecker,
    check_text_spelling,
    get_default_spell_checker,
)


class TestVisualOCRConfusionModel(unittest.TestCase):
    """Kiểm tra ma trận khoảng cách nhầm lẫn thị giác OCR."""

    def test_syllable_decomposition(self) -> None:
        """Kiểm tra phân tách âm tiết: onset, base_vowels, diacritics, tone, coda."""
        o, b, d, t, cd = VisualOCRConfusionModel.decompose_syllable("bộ")
        self.assertEqual(o, "b")
        self.assertEqual(b, "o")
        self.assertEqual(d, ("circumflex",))
        self.assertEqual(t, 5)  # Nặng
        self.assertEqual(cd, "")

        o, b, d, t, cd = VisualOCRConfusionModel.decompose_syllable("họp")
        self.assertEqual(o, "h")
        self.assertEqual(b, "o")
        self.assertEqual(d, ("none",))
        self.assertEqual(t, 5)
        self.assertEqual(cd, "p")

        o, b, d, t, cd = VisualOCRConfusionModel.decompose_syllable("nghiệp")
        self.assertEqual(o, "ngh")
        self.assertEqual(b, "ie")
        self.assertEqual(t, 5)
        self.assertEqual(cd, "p")

    def test_tone_confusion_distance_under_0_20(self) -> None:
        """Các từ chỉ khác dấu thanh trên cùng nguyên âm phải có distance < 0.20."""
        # bộ vs bố (nặng vs sắc): cost = 0.15
        dist_bo = visual_ocr_distance("bộ", "bố")
        self.assertLess(dist_bo, 0.20)
        self.assertAlmostEqual(dist_bo, 0.15, places=2)

        # dự vs dư (nặng vs ngang): cost = 0.15
        dist_du = visual_ocr_distance("dự", "dư")
        self.assertLess(dist_du, 0.20)
        self.assertAlmostEqual(dist_du, 0.15, places=2)

        # chỉ vs chi (hỏi vs ngang): cost = 0.15
        dist_chi = visual_ocr_distance("chỉ", "chi")
        self.assertLess(dist_chi, 0.20)
        self.assertAlmostEqual(dist_chi, 0.15, places=2)

    def test_diacritic_and_tone_confusion_distance_under_0_32(self) -> None:
        """Các từ khác dấu phụ nguyên âm hoặc cả dấu phụ + thanh phải có distance <= 0.32."""
        # họp vs hơp (khác dấu phụ ơ/o và thanh nặng/ngang): cost = 0.30
        dist_hop = visual_ocr_distance("họp", "hơp")
        self.assertLessEqual(dist_hop, 0.32)
        self.assertAlmostEqual(dist_hop, 0.30, places=2)

        # đơn vs đon (khác dấu phụ ơ/o): cost = 0.22
        dist_don = visual_ocr_distance("đơn", "đon")
        self.assertLessEqual(dist_don, 0.32)
        self.assertAlmostEqual(dist_don, 0.22, places=2)

        # ưng vs ung (khác dấu phụ ư/u): cost = 0.22
        dist_ung = visual_ocr_distance("ưng", "ung")
        self.assertLessEqual(dist_ung, 0.32)
        self.assertAlmostEqual(dist_ung, 0.22, places=2)

        # bộ vs bồ (khác thanh trên cùng ô): cost = 0.15 <= 0.32
        dist_bobo = visual_ocr_distance("bộ", "bồ")
        self.assertLessEqual(dist_bobo, 0.32)

    def test_visual_letter_confusion_distance(self) -> None:
        """Các cặp chữ cái dễ nhầm lẫn trong OCR (a/d, n/m, o/a, c/e)."""
        dist_ad = visual_ocr_distance("can", "con")
        self.assertLessEqual(dist_ad, 0.36)

    def test_unrelated_words_distance_over_1_0(self) -> None:
        """Hai từ hoàn toàn khác nhau phải có khoảng cách >= 1.0."""
        dist_diff = visual_ocr_distance("bộ", "nhà")
        self.assertGreaterEqual(dist_diff, 1.0)

        dist_diff2 = visual_ocr_distance("họp", "sách")
        self.assertGreaterEqual(dist_diff2, 1.0)


class TestVietnameseNgramLanguageModel(unittest.TestCase):
    """Kiểm tra mô hình ngôn ngữ xác suất N-gram đa lĩnh vực."""

    def setUp(self) -> None:
        self.lm = get_default_ngram_model()

    def test_is_valid_collocation_golden_rule(self) -> None:
        """Kiểm tra quy tắc vàng Zero False Positives trên các cụm từ hợp lệ."""
        # Các cụm từ ghép chuẩn phải trả về True ngay lập tức
        self.assertTrue(self.lm.is_valid_collocation("cuộc", "họp", ""))
        self.assertTrue(self.lm.is_valid_collocation("sinh", "hoạt", "chi"))
        self.assertTrue(self.lm.is_valid_collocation("chi", "bộ", ""))
        self.assertTrue(self.lm.is_valid_collocation("đầy", "đủ", ""))
        self.assertTrue(self.lm.is_valid_collocation("kế", "toán", ""))
        self.assertTrue(self.lm.is_valid_collocation("hóa", "đơn", ""))
        self.assertTrue(self.lm.is_valid_collocation("tổng", "cộng", ""))
        self.assertTrue(self.lm.is_valid_collocation("pháp", "luật", ""))
        self.assertTrue(self.lm.is_valid_collocation("thời", "lượng", ""))
        self.assertTrue(self.lm.is_valid_collocation("điều", "khoản", ""))
        self.assertTrue(self.lm.is_valid_collocation("trao", "đổi", ""))

        # Các cụm từ sai chính tả phải trả về False
        self.assertFalse(self.lm.is_valid_collocation("chi", "bố", ""))
        self.assertFalse(self.lm.is_valid_collocation("hóa", "đon", "giá"))
        self.assertFalse(self.lm.is_valid_collocation("viên", "dư", "sinh"))
        self.assertFalse(self.lm.is_valid_collocation("bài", "tạp", "về"))
        self.assertFalse(self.lm.is_valid_collocation("sản", "phâm", "công"))

    def test_evaluate_token_in_context_scoring(self) -> None:
        """Kiểm tra hàm chấm điểm Noisy Channel trả về ứng viên tối ưu."""
        # 1. 'chi bố' -> gợi ý 'bộ'
        eval_bo = self.lm.evaluate_token_in_context("", "chi", "bố", "", "")
        self.assertIsNotNone(eval_bo)
        assert eval_bo is not None
        self.assertEqual(eval_bo.suggested_token, "bộ")
        self.assertLess(eval_bo.visual_distance, 0.20)

        # 2. 'hóa đon giá trị' -> gợi ý 'đơn'
        eval_don = self.lm.evaluate_token_in_context("", "hóa", "đon", "giá", "trị")
        self.assertIsNotNone(eval_don)
        assert eval_don is not None
        self.assertEqual(eval_don.suggested_token, "đơn")
        self.assertLessEqual(eval_don.visual_distance, 0.25)

        # 3. 'tổng công thanh toán' -> gợi ý 'cộng'
        eval_cong = self.lm.evaluate_token_in_context("", "tổng", "công", "thanh", "toán")
        self.assertIsNotNone(eval_cong)
        assert eval_cong is not None
        self.assertEqual(eval_cong.suggested_token, "cộng")
        self.assertLessEqual(eval_cong.visual_distance, 0.20)

        # 4. 'kết qua xét nghiệm' -> gợi ý 'quả'
        eval_qua = self.lm.evaluate_token_in_context("", "kết", "qua", "xét", "nghiệm")
        self.assertIsNotNone(eval_qua)
        assert eval_qua is not None
        self.assertEqual(eval_qua.suggested_token, "quả")


class TestMultiDomainSpellCorrection(unittest.TestCase):
    """Kiểm tra chức năng kiểm tra chính tả đa lĩnh vực qua VietnameseSpellChecker."""

    def setUp(self) -> None:
        self.checker = get_default_spell_checker()

    # ------------------------------------------------------------------
    # 1. Nhóm lỗi thực tế đa lĩnh vực
    # ------------------------------------------------------------------

    def test_domain_administrative_context(self) -> None:
        """Lĩnh vực Hành chính & Đảng: 'sinh hoạt chi bố' và 'đảng viên dư sinh hoạt'."""
        # 1. 'chi bố' -> 'bộ'
        text1 = "Đồng chí Bí thư chủ trì buổi sinh hoạt chi bố định kỳ."
        errs1 = check_text_spelling(text1)
        tokens1 = [e.token for e in errs1]
        self.assertIn("bố", tokens1, "'bố' trong 'chi bố' phải bị đánh dấu")
        err1 = next(e for e in errs1 if e.token == "bố")
        self.assertEqual(err1.suggestions[0], "bộ", "Gợi ý số 1 phải là 'bộ'")

        # 2. 'dư sinh hoạt' -> 'dự'
        text2 = "Tỷ lệ đảng viên dư sinh hoạt chi bộ đạt 100%."
        errs2 = check_text_spelling(text2)
        tokens2 = [e.token for e in errs2]
        self.assertIn("dư", tokens2, "'dư' trong 'dư sinh hoạt' phải bị đánh dấu")
        err2 = next(e for e in errs2 if e.token == "dư")
        self.assertEqual(err2.suggestions[0], "dự", "Gợi ý số 1 phải là 'dự'")

    def test_domain_commercial_and_accounting(self) -> None:
        """Lĩnh vực Thương mại & Kế toán: 'hóa đon' và 'tổng công thanh toán'."""
        # 1. 'hóa đon' -> 'đơn'
        text1 = "Công ty xuất hóa đon giá trị gia tăng cho khách hàng."
        errs1 = check_text_spelling(text1)
        tokens1 = [e.token for e in errs1]
        self.assertIn("đon", tokens1, "'đon' trong 'hóa đon' phải bị đánh dấu")
        err1 = next(e for e in errs1 if e.token == "đon")
        self.assertEqual(err1.suggestions[0], "đơn", "Gợi ý số 1 phải là 'đơn'")

        # 2. 'tổng công thanh toán' -> 'cộng'
        text2 = "Bảng kê tổng công thanh toán tiền hàng tháng 10."
        errs2 = check_text_spelling(text2)
        tokens2 = [e.token for e in errs2]
        self.assertIn("công", tokens2, "'công' trong 'tổng công thanh toán' phải bị đánh dấu")
        err2 = next(e for e in errs2 if e.token == "công")
        self.assertEqual(err2.suggestions[0], "cộng", "Gợi ý số 1 phải là 'cộng'")

    def test_domain_education_and_science(self) -> None:
        """Lĩnh vực Giáo dục & Khoa học: 'kết qua' và 'bài tạp'."""
        # 1. 'kết qua' -> 'quả'
        text1 = "Bác sĩ trả kết qua xét nghiệm máu cho bệnh nhân."
        errs1 = check_text_spelling(text1)
        tokens1 = [e.token for e in errs1]
        self.assertIn("qua", tokens1, "'qua' trong 'kết qua' phải bị đánh dấu")
        err1 = next(e for e in errs1 if e.token == "qua")
        self.assertEqual(err1.suggestions[0], "quả", "Gợi ý số 1 phải là 'quả'")

        # 2. 'bài tạp' -> 'tập'
        text2 = "Học sinh hoàn thành làm bài tạp về nhà đầy đủ."
        errs2 = check_text_spelling(text2)
        tokens2 = [e.token for e in errs2]
        self.assertIn("tạp", tokens2, "'tạp' trong 'bài tạp' phải bị đánh dấu")
        err2 = next(e for e in errs2 if e.token == "tạp")
        self.assertEqual(err2.suggestions[0], "tập", "Gợi ý số 1 phải là 'tập'")

    def test_domain_technical_and_production(self) -> None:
        """Lĩnh vực Kỹ thuật & Sản xuất: 'sản phâm công nghệ'."""
        text = "Doanh nghiệp ra mắt sản phâm công nghệ thông tin mới."
        errs = check_text_spelling(text)
        tokens = [e.token for e in errs]
        self.assertIn("phâm", tokens, "'phâm' trong 'sản phâm' phải bị đánh dấu")
        err = next(e for e in errs if e.token == "phâm")
        self.assertEqual(err.suggestions[0], "phẩm", "Gợi ý số 1 phải là 'phẩm'")

    # ------------------------------------------------------------------
    # 2. Zero False Positives trên văn bản chuẩn đa lĩnh vực
    # ------------------------------------------------------------------

    def test_zero_false_positives_standard_texts(self) -> None:
        """Văn bản chuẩn thuộc mọi lĩnh vực phải trả về đúng 0 lỗi."""
        standard_documents = [
            # Hành chính
            "Biên bản cuộc họp chi bộ đầy đủ các thành phần tham dự đúng thời lượng quy định.",
            # Tài chính - Kế toán
            "Hóa đơn thanh toán tổng cộng giá trị các mặt hàng bằng tiền mặt và chuyển khoản.",
            # Pháp luật - Hợp đồng
            "Quy định pháp luật và các điều khoản trong hợp đồng lao động đã được thảo luận.",
            # Giáo dục - Y tế
            "Kết quả học tập và làm bài tập về nhà của sinh viên được đánh giá rất tốt.",
            # CNTT - Công nghệ
            "Sản phẩm công nghệ thông tin phục vụ công tác quản lý và vận hành hệ thống phần mềm.",
        ]

        for doc in standard_documents:
            errs = check_text_spelling(doc)
            self.assertEqual(
                len(errs),
                0,
                f"Văn bản chuẩn đa lĩnh vực bị bôi đỏ nhầm: '{doc}' -> Lỗi: {errs}",
            )


if __name__ == "__main__":
    unittest.main()
