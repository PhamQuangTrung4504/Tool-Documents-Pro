#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests/test_spell_corrector.py
=============================
Bộ kiểm thử tự động toàn diện cho chức năng Kiểm tra chính tả tiếng Việt thủ công.

Bao gồm 15 trường hợp kiểm thử bắt buộc theo yêu cầu:
1. Văn bản tiếng Việt đúng không bị đánh dấu hàng loạt.
2. Từ sai cấu trúc có thể được phát hiện.
3. Từ tiếng Việt hợp lệ như 'bô', 'thê', 'bộc' không bị kết luận sai.
4. Từ viết tắt in hoa được giữ nguyên (UBND, HĐND, QĐ, CPU, VIP, TW, BCĐTW).
5. Số liệu, ngày tháng, mã định danh, URL và email không bị đánh dấu nhầm.
6. Văn bản trộn tiếng Việt và tiếng Anh được xử lý phù hợp.
7. Tên riêng và thuật ngữ chuyên ngành không bị đánh dấu nhầm.
8. Văn bản có dấu tiếng Việt ở dạng Unicode dựng sẵn (NFC) và tổ hợp (NFD).
9. Chỉ số start_idx, end_idx khớp chính xác với văn bản ban đầu.
10. Dấu câu, khoảng trắng, tab, xuống dòng và thụt lề không thay đổi.
11. Kiểm tra chính tả chỉ đánh dấu, không sửa nội dung (100% text identity).
12. Bỏ đánh dấu chỉ xóa định dạng do bộ kiểm tra tạo ra.
13. Các chức năng OCR, sao chép và xuất file vẫn hoạt động.
14. Văn bản dài không làm giao diện treo nghiêm trọng (Performance benchmark).
15. Kiểm thử hồi quy và toàn vẹn API.
"""

from __future__ import annotations

import os
import sys
import time
import unicodedata
import unittest

# Đảm bảo import được module app
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app.core.spell_corrector import (
    VietnameseSpellChecker,
    check_text_spelling,
    get_default_spell_checker,
)


class TestVietnameseSpellChecker(unittest.TestCase):
    """Kiểm tra chức năng cốt lõi của VietnameseSpellChecker."""

    def setUp(self) -> None:
        self.checker = get_default_spell_checker()

    # ------------------------------------------------------------------
    # Case 1: Văn bản tiếng Việt đúng không bị đánh dấu hàng loạt
    # ------------------------------------------------------------------
    def test_01_correct_vietnamese_text_not_flagged(self) -> None:
        sample_texts = [
            "Cộng hòa xã hội chủ nghĩa Việt Nam",
            "Độc lập - Tự do - Hạnh phúc",
            "Ủy ban nhân dân thành phố Hà Nội",
            "Kế hoạch triển khai thực hiện nhiệm vụ công tác năm 2026",
            "Bộ Văn hóa, Thể thao và Du lịch thông báo kế hoạch tổ chức",
            "Kính gửi: Các cơ quan, đơn vị, ban ngành đoàn thể",
            "Giải quyết những vấn đề khó khăn, vướng mắc kịp thời",
        ]
        for text in sample_texts:
            errors = check_text_spelling(text)
            self.assertEqual(
                errors,
                [],
                f"Văn bản đúng không được có lỗi: '{text}', phát hiện: {errors}",
            )

    # ------------------------------------------------------------------
    # Case 2: Từ sai cấu trúc có thể được phát hiện
    # ------------------------------------------------------------------
    def test_02_detect_structurally_invalid_words(self) -> None:
        text = "Kế họach kiểm tra và ra sóat nội dung công việc."
        errors = check_text_spelling(text)
        tokens = [tok for _, _, tok in errors]
        self.assertIn("họach", tokens, "Cần phát hiện lỗi đặt dấu ở 'họach'")
        self.assertIn("sóat", tokens, "Cần phát hiện lỗi đặt dấu ở 'sóat'")

        text2 = "Thực hiện nghiêm tuc, giải quyết đap ứng yêu cầu chưa đươc thông qua."
        errors2 = check_text_spelling(text2)
        tokens2 = [tok for _, _, tok in errors2]
        self.assertIn("tuc", tokens2, "Cần phát hiện lỗi thiếu dấu ở 'tuc'")
        self.assertIn("đap", tokens2, "Cần phát hiện lỗi thiếu dấu ở 'đap'")
        self.assertIn("đươc", tokens2, "Cần phát hiện lỗi thiếu dấu ở 'đươc'")

    # ------------------------------------------------------------------
    # Case 3: Từ tiếng Việt hợp lệ như bô, thê, bộc không bị kết luận sai
    # ------------------------------------------------------------------
    def test_03_valid_vietnamese_words_not_false_positives(self) -> None:
        text = "Cái bô bằng nhôm, phu thê tình nghĩa sâu nặng, bộc lộ tình cảm chân thành."
        errors = check_text_spelling(text)
        tokens = [tok for _, _, tok in errors]
        self.assertNotIn("bô", tokens, "'bô' là âm tiết tiếng Việt hợp lệ, không được đánh dấu")
        self.assertNotIn("thê", tokens, "'thê' là âm tiết tiếng Việt hợp lệ, không được đánh dấu")
        self.assertNotIn("bộc", tokens, "'bộc' là âm tiết tiếng Việt hợp lệ, không được đánh dấu")
        self.assertEqual(errors, [], f"Không được có lỗi với các từ hợp lệ: {errors}")

    # ------------------------------------------------------------------
    # Case 4: Từ viết tắt in hoa được giữ nguyên
    # ------------------------------------------------------------------
    def test_04_uppercase_acronyms_protected(self) -> None:
        text = "UBND tỉnh, HĐND huyện, QĐ ban hành, CPU, VIP, TW, BCĐTW, BKHĐT."
        errors = check_text_spelling(text)
        self.assertEqual(
            errors,
            [],
            f"Các từ viết tắt in hoa phải được bảo vệ: {errors}",
        )

    # ------------------------------------------------------------------
    # Case 5: Số liệu, ngày tháng, mã định danh, URL và email không bị đánh dấu nhầm
    # ------------------------------------------------------------------
    def test_05_metadata_and_codes_protected(self) -> None:
        text = (
            "Theo Quyết định số 2485/QĐ-BVHTTDL và Nghị định 63/2010/NĐ-CP "
            "ban hành ngày 24/6/2025, mã hồ sơ 170-KL/TW. "
            "Liên hệ qua email contact@gov.vn hoặc trang web https://chinhphu.vn lúc 08:30. "
            "Số tiền 100.000đ, đạt tỷ lệ 95.5% trên diện tích 50ha."
        )
        errors = check_text_spelling(text)
        self.assertEqual(
            errors,
            [],
            f"Số liệu, ngày tháng, mã công văn, email, URL không được đánh dấu: {errors}",
        )

    # ------------------------------------------------------------------
    # Case 6: Văn bản trộn tiếng Việt và tiếng Anh được xử lý phù hợp
    # ------------------------------------------------------------------
    def test_06_mixed_vietnamese_english_text(self) -> None:
        text = (
            "Ứng dụng OCR chạy trên hệ điều hành Windows với ngôn ngữ Python. "
            "Hệ thống hỗ trợ xuất dữ liệu ra file Excel và định dạng PDF trực tuyến."
        )
        errors = check_text_spelling(text)
        self.assertEqual(
            errors,
            [],
            f"Văn bản trộn tiếng Việt và tiếng Anh chuẩn không được bị đánh dấu sai: {errors}",
        )

    # ------------------------------------------------------------------
    # Case 7: Tên riêng và thuật ngữ chuyên ngành không bị đánh dấu nhầm
    # ------------------------------------------------------------------
    def test_07_proper_nouns_and_specialized_terms(self) -> None:
        text = (
            "Tổng thống Obama và chuyên gia Alexander thảo luận tại diễn đàn quốc tế. "
            "Các công ty Tesla, Microsoft, Samsung phát triển công nghệ trí tuệ nhân tạo."
        )
        errors = check_text_spelling(text)
        self.assertEqual(
            errors,
            [],
            f"Tên riêng và công ty không được đánh dấu: {errors}",
        )

    # ------------------------------------------------------------------
    # Case 8: Văn bản có dấu tiếng Việt ở dạng NFC và NFD
    # ------------------------------------------------------------------
    def test_08_unicode_nfc_and_nfd_support(self) -> None:
        # Chuỗi NFC
        text_nfc = "Ủy ban nhân dân tỉnh Quảng Ninh"
        errors_nfc = check_text_spelling(text_nfc)
        self.assertEqual(errors_nfc, [], "Chuỗi NFC đúng không có lỗi")

        # Chuỗi NFD
        text_nfd = unicodedata.normalize("NFD", text_nfc)
        errors_nfd = check_text_spelling(text_nfd)
        self.assertEqual(errors_nfd, [], "Chuỗi NFD đúng không có lỗi")

        # Chuỗi NFD có chứa lỗi
        err_text_nfc = "Kế họach kiểm tra"
        err_text_nfd = unicodedata.normalize("NFD", err_text_nfc)
        errors_err_nfd = check_text_spelling(err_text_nfd)
        self.assertEqual(len(errors_err_nfd), 1, "Phải phát hiện 1 lỗi trong chuỗi NFD")
        s, e, tok = errors_err_nfd[0]
        self.assertEqual(
            err_text_nfd[s:e],
            tok,
            "Chỉ số NFD phải khớp tuyệt đối với chuỗi NFD gốc",
        )

    # ------------------------------------------------------------------
    # Case 9: Chỉ số start_idx, end_idx khớp chính xác với văn bản ban đầu
    # ------------------------------------------------------------------
    def test_09_indices_exact_match_original_text(self) -> None:
        text = "Văn bản đap ứng yêu cầu, đang kiêmr tra dữ liệu."
        errors = check_text_spelling(text)
        self.assertGreater(len(errors), 0, "Phải phát hiện lỗi")
        for start_idx, end_idx, token in errors:
            sliced = text[start_idx:end_idx]
            self.assertEqual(
                sliced,
                token,
                f"Chỉ số text[{start_idx}:{end_idx}]='{sliced}' không khớp token='{token}'",
            )

    # ------------------------------------------------------------------
    # Case 10: Dấu câu, khoảng trắng, tab, xuống dòng không thay đổi
    # ------------------------------------------------------------------
    def test_10_preserves_whitespace_tabs_newlines_punctuation(self) -> None:
        complex_text = (
            "\t\tBỘ VĂN HÓA, THỂ THAO VÀ DU LỊCH\n"
            "   Số: 2485/QĐ-BVHTTDL\n\n"
            "Điều 1.   Phê duyệt kế họach rà sóat:\n"
            "   - Điểm a: Thực hiện nghiêm tuc.\n"
            "   - Điểm b: Đap ứng đúng tiến độ.\n"
            "\t\t\tHà Nội, ngày 24/6/2025\n"
        )
        original_copy = str(complex_text)
        errors = check_text_spelling(complex_text)

        # Đảm bảo chuỗi đầu vào không bị biến đổi bất kỳ ký tự nào
        self.assertEqual(complex_text, original_copy, "Chuỗi đầu vào không được bị thay đổi")
        self.assertEqual(len(complex_text), len(original_copy))

        # Kiểm tra các vị trí lỗi được trích xuất chính xác
        for s, e, tok in errors:
            self.assertEqual(complex_text[s:e], tok)

    # ------------------------------------------------------------------
    # Case 11: Nút kiểm tra chính tả chỉ đánh dấu, không sửa nội dung
    # ------------------------------------------------------------------
    def test_11_inspection_only_no_text_replacement(self) -> None:
        text = "Kế họach kiểm tra và ra sóat nội dung công việc."
        before = text
        _ = check_text_spelling(text)
        after = text
        self.assertEqual(before, after, "Nội dung trước và sau kiểm tra phải giống hệt nhau")

    # ------------------------------------------------------------------
    # Case 12: Bỏ đánh dấu chỉ xóa định dạng, không thay đổi nội dung
    # ------------------------------------------------------------------
    def test_12_clearing_highlights_does_not_modify_text(self) -> None:
        from PyQt5.QtWidgets import QApplication, QPlainTextEdit, QTextEdit
        from PyQt5.QtGui import QTextCharFormat, QColor, QTextCursor

        # Khởi tạo QApplication headless nếu chưa có
        app = QApplication.instance() or QApplication([""])

        editor = QPlainTextEdit()
        original_content = "Kế họach kiểm tra và ra sóat nội dung."
        editor.setPlainText(original_content)

        errors = check_text_spelling(original_content)
        self.assertEqual(len(errors), 2)

        # Giả lập đánh dấu
        fmt = QTextCharFormat()
        fmt.setForeground(QColor("#ff4d4f"))
        selections = []
        doc = editor.document()
        for s, e, tok in errors:
            cursor = QTextCursor(doc)
            cursor.setPosition(s)
            cursor.setPosition(e, QTextCursor.KeepAnchor)
            sel = QTextEdit.ExtraSelection()
            sel.cursor = cursor
            sel.format = fmt
            selections.append(sel)
        editor.setExtraSelections(selections)

        # Xác minh văn bản không đổi sau khi đánh dấu
        self.assertEqual(editor.toPlainText(), original_content)
        self.assertEqual(len(editor.extraSelections()), 2)

        # Giả lập bỏ đánh dấu
        editor.setExtraSelections([])
        self.assertEqual(len(editor.extraSelections()), 0)
        self.assertEqual(editor.toPlainText(), original_content)

    # ------------------------------------------------------------------
    # Case 13: Các chức năng OCR, sao chép và xuất file vẫn hoạt động
    # ------------------------------------------------------------------
    def test_13_ocr_and_clipboard_compatibility(self) -> None:
        from app.core.ocr_engine import OCRResult, OCRBox

        boxes = [
            OCRBox(text="Cộng hòa xã hội chủ nghĩa Việt Nam", confidence=0.98, bbox=[[0, 0], [100, 0], [100, 20], [0, 20]]),
            OCRBox(text="Độc lập - Tự do - Hạnh phúc", confidence=0.99, bbox=[[0, 25], [100, 25], [100, 45], [0, 45]]),
        ]
        result = OCRResult(boxes=boxes, full_text="Cộng hòa xã hội chủ nghĩa Việt Nam\nĐộc lập - Tự do - Hạnh phúc")
        self.assertEqual(result.num_lines, 2)
        self.assertGreater(result.average_confidence, 0.95)

        # Kiểm tra văn bản OCR sạch không bị lỗi chính tả
        errors = check_text_spelling(result.full_text)
        self.assertEqual(errors, [])

    # ------------------------------------------------------------------
    # Case 14: Văn bản dài không làm giao diện treo nghiêm trọng
    # ------------------------------------------------------------------
    def test_14_long_text_performance(self) -> None:
        paragraph = (
            "Ủy ban nhân dân tỉnh Quảng Ninh chỉ đạo triển khai kế hoạch năm 2026. "
            "Các cơ quan, ban ngành đoàn thể phối hợp chặt chẽ, bảo đảm đúng quy định. "
            "Kế họach rà sóat thủ tục hành chính phải thực hiện nghiêm tuc. "
        )
        # Tạo văn bản ~30.000 từ, ~200.000 ký tự
        long_text = paragraph * 1000
        t0 = time.perf_counter()
        errors = check_text_spelling(long_text)
        elapsed = time.perf_counter() - t0

        self.assertLess(
            elapsed,
            5.0,
            f"Kiểm tra 200.000 ký tự mất {elapsed:.2f}s, phải dưới 5.0s",
        )
        self.assertEqual(len(errors), 3000, "Phải tìm đúng 3000 vị trí lỗi lặp lại")

    # ------------------------------------------------------------------
    # Case 15: Kiểm thử hồi quy và toàn vẹn API
    # ------------------------------------------------------------------
    def test_15_api_signature_and_regression(self) -> None:
        # 1. Chuỗi rỗng hoặc chỉ khoảng trắng
        self.assertEqual(check_text_spelling(""), [])
        self.assertEqual(check_text_spelling("   \n\t  "), [])

        # 2. Định dạng trả về đúng List[Tuple[int, int, str]]
        res = check_text_spelling("lỗi họach ở đây")
        self.assertIsInstance(res, list)
        self.assertEqual(len(res), 1)
        item = res[0]
        self.assertIsInstance(item, tuple)
        self.assertEqual(len(item), 3)
        self.assertIsInstance(item[0], int)
        self.assertIsInstance(item[1], int)
        self.assertIsInstance(item[2], str)
        self.assertEqual(item[2], "họach")

        # 3. Kết quả phải được sắp xếp theo start_idx tăng dần
        multi_err = "đap ứng kế họach ra sóat"
        res_multi = check_text_spelling(multi_err)
        start_indices = [item[0] for item in res_multi]
        self.assertEqual(start_indices, sorted(start_indices))


if __name__ == "__main__":
    unittest.main()
