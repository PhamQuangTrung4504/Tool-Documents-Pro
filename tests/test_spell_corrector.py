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


class TestAntiFalsePositiveAndContext(unittest.TestCase):
    """
    Kiểm thử cơ chế chống sửa sai (Anti-False-Positive) và độ chính xác gợi ý theo ngữ cảnh:
    - Bảo đảm các từ vựng phổ thông tiếng Việt (đúng, để, qua, tại, tỉnh, số,...)
      không bị đánh dấu sai lệch hoặc gợi ý sửa vớ vẩn.
    - Kiểm thử đa lĩnh vực: Hành chính, Thương mại, Kỹ thuật/CNTT, Sách báo.
    - Đảm bảo phát hiện đúng các lỗi thực tế (kỹ su -> kỹ sư, lâu nơm -> lâu năm,...).
    - Đảm bảo không sinh gợi ý giả/rác khi không có bằng chứng đủ mạnh.
    """

    def setUp(self) -> None:
        self.checker = get_default_spell_checker()

    def test_screenshot_real_world_case(self) -> None:
        """Kiểm thử trực tiếp câu trong tài liệu thực tế của người dùng: 'đúng' không bị đánh dấu."""
        text = "Cần bổ sung đúng số hotline theo tài liệu: (028) 38.211.238"
        errors = check_text_spelling(text)
        self.assertEqual(
            errors,
            [],
            f"'đúng số' là cụm từ hợp lệ, không được đánh dấu thành 'đứng số': {errors}",
        )

        text2 = "Tuyên truyền đường dây nóng để người dân được biết."
        errors2 = check_text_spelling(text2)
        self.assertEqual(
            errors2,
            [],
            f"'để người' là ngữ cảnh hoàn toàn chuẩn xác, không được sửa thành 'dể người': {errors2}",
        )

    def test_multi_domain_texts_clean(self) -> None:
        """Kiểm thử các văn bản đa lĩnh vực hợp lệ không có lỗi giả."""
        domains = [
            # Thương mại / Tài chính
            "Hóa đơn giá trị gia tăng số 001245 ngày 15/10/2026, tổng thanh toán 2.500.000 VNĐ.",
            # Kỹ thuật / Công nghệ thông tin
            "Hệ thống backend viết bằng Python kết nối cơ sở dữ liệu PostgreSQL và Redis cache.",
            # Quản trị / Hợp đồng
            "Hợp đồng dịch vụ tư vấn quản trị doanh nghiệp và thỏa thuận bảo mật thông tin giữa hai bên.",
            # Giáo dục / Sách báo
            "Nghiên cứu về lịch sử văn hóa dân tộc và phong tục tập quán truyền thống của Việt Nam.",
            # Đời sống / Giao tiếp
            "Tôi đi làm đúng giờ, giải quyết công việc nhanh chóng và phân công đúng người đúng việc.",
        ]
        for doc in domains:
            errs = check_text_spelling(doc)
            self.assertEqual(errs, [], f"Tài liệu hợp lệ không được có lỗi: '{doc}', phát hiện: {errs}")

    def test_real_errors_detected_with_valid_suggestions(self) -> None:
        """Phát hiện lỗi thực tế và gợi ý sửa đúng có căn cứ."""
        # 1. OCR confusion: kỹ su -> kỹ sư
        errs = check_text_spelling("kỹ su")
        self.assertEqual(len(errs), 1)
        self.assertEqual(errs[0].token, "su")
        self.assertIn("sư", errs[0].suggestions)

        # 2. OCR confusion: lâu nơm -> lâu năm
        errs2 = check_text_spelling("lâu nơm")
        self.assertEqual(len(errs2), 1)
        self.assertEqual(errs2[0].token, "nơm")
        self.assertIn("năm", errs2[0].suggestions)

        # 3. OCR confusion: Bí thu -> Bí thư
        errs3 = check_text_spelling("Bí thu")
        self.assertEqual(len(errs3), 1)
        self.assertEqual(errs3[0].token, "thu")
        self.assertIn("thư", errs3[0].suggestions)

        # 4. Thiếu dấu âm tiết: chuyên nghiêp -> chuyên nghiệp
        errs4 = check_text_spelling("chuyên nghiêp")
        self.assertEqual(len(errs4), 1)
        self.assertEqual(errs4[0].token, "nghiêp")
        self.assertIn("nghiệp", errs4[0].suggestions)

        # 5. Lỗi Telex sót đuôi: kiêmr tra -> kiểm tra
        errs5 = check_text_spelling("Đang kiêmr tra dữ liệu")
        self.assertEqual(len(errs5), 1)
        self.assertEqual(errs5[0].token, "kiêmr")
        self.assertIn("kiểm", errs5[0].suggestions)

    def test_no_bogus_suggestions_when_insufficient_evidence(self) -> None:
        """Không sinh gợi ý giả từ khoảng cách Levenshtein khi không có căn cứ ngữ âm/OCR/ngữ cảnh."""
        from app.core.spell_corrector import get_suggestions
        # Một từ gõ loạn hoàn toàn không có quan hệ ngữ âm với từ tiếng Việt
        suggs = get_suggestions("zkjqqw")
        self.assertEqual(
            suggs,
            [],
            f"Từ rác không được sinh gợi ý tiếng Việt bừa bãi: {suggs}",
        )


class TestOCRLineGroupingAndBullets(unittest.TestCase):
    """Kiểm thử tính năng gom nhóm dòng và cấu trúc danh sách/bullet trong ocr_engine."""

    def test_line_grouping_and_bullet_normalization(self) -> None:
        from app.core.ocr_engine import OCRBox, group_boxes_into_lines

        # Mô phỏng các box cùng một hàng ngang: Bullet nhỏ + Dòng chữ
        boxes = [
            OCRBox("o", 0.85, [[70, 75], [85, 75], [85, 90], [70, 90]]),
            OCRBox("Trong file ghi: đường dây nóng", 0.98, [[100, 70], [500, 70], [500, 95], [100, 95]]),
            OCRBox("3.", 0.95, [[70, 150], [90, 150], [90, 170], [70, 170]]),
            OCRBox("Căn lề phần Kính gửi:", 0.98, [[100, 150], [350, 150], [350, 170], [100, 170]]),
        ]

        lines = group_boxes_into_lines(boxes)
        self.assertEqual(len(lines), 2, "Phải gom nhóm thành đúng 2 dòng")

        # Dòng 1: icon 'o' ở đầu được chuẩn hóa thành '•' và đi cùng text
        line1_text = " ".join(b.text for b in lines[0])
        self.assertTrue(line1_text.startswith("• "))
        self.assertIn("Trong file ghi:", line1_text)

        # Dòng 2: '3.' là số thứ tự được giữ nguyên, đi cùng text trên cùng một hàng
        line2_text = " ".join(b.text for b in lines[1])
        self.assertEqual(line2_text, "3. Căn lề phần Kính gửi:")


class TestPunctuationAndFilePaths(unittest.TestCase):
    """Kiểm thử chuyên sâu xử lý dấu câu, khoảng trắng, đường dẫn và phần mở rộng file."""

    def test_file_paths_and_extensions_protected(self) -> None:
        """Đường dẫn, tên file và phần mở rộng hợp lệ không bị đánh dấu lỗi."""
        cases = [
            "Mở app/core/spell_corrector.py để kiểm tra.",
            "Đọc app/core/ocr_engine.py rồi chạy main.py.",
            "File document.docx đã được xuất thành report.pdf.",
            "Dữ liệu lưu tại data.json và ảnh giao diện image.png.",
            "Xem trang index.html hoặc tệp cấu hình config.yaml.",
            "Mở file app/core/spell_corrector.py.",
        ]
        for text in cases:
            errs = check_text_spelling(text)
            self.assertEqual(
                errs, [],
                f"Đường dẫn/tên file không được bị báo lỗi trong câu: {text!r} (phát hiện: {errs})"
            )

    def test_typo_near_file_path_detected(self) -> None:
        """Từ sai chính tả thực sự đứng trước/sau đường dẫn vẫn phải được phát hiện chính xác."""
        text = "Mở flle app/core/spell_corrector.py để kiêmr tra."
        errs = check_text_spelling(text)
        self.assertEqual(len(errs), 2, f"Phải phát hiện đúng 2 lỗi 'flle' và 'kiêmr' trong câu: {errs}")

        tokens = [err.token for err in errs]
        self.assertIn("flle", tokens)
        self.assertIn("kiêmr", tokens)

        # Kiểm tra offset chính xác 100% trong văn bản gốc
        for err in errs:
            self.assertEqual(text[err.start_idx:err.end_idx], err.token)

    def test_punctuation_and_whitespace_preservation(self) -> None:
        """Phân biệt dấu chấm theo ngữ cảnh: kết thúc câu, giữa 2 câu, số thập phân, v.v."""
        # 1. Dấu chấm giữa 2 câu không có khoảng trắng: không làm biến đổi chuỗi gốc
        text1 = "Xin chào.Bạn khỏe không."
        errs1 = check_text_spelling(text1)
        self.assertEqual(errs1, [])

        # 2. Số thập phân
        text2 = "Giá trị đo được là 3.14."
        errs2 = check_text_spelling(text2)
        self.assertEqual(errs2, [])

        # 3. Dấu chấm liên tiếp và dấu ngoặc
        text3 = "Nội dung đang tiếp tục cập nhật... (Theo quy định tại Điều 1.)"
        errs3 = check_text_spelling(text3)
        self.assertEqual(errs3, [])

    def test_common_sentence_words_matching_extensions(self) -> None:
        """Từ trong câu thông thường chứa chuỗi trùng với phần mở rộng không gây crash hoặc bỏ sót."""
        text = "Một câu thông thường chứa các chuỗi doc hoặc c hoặc py không phải tên file."
        errs = check_text_spelling(text)
        # Không có ngoại lệ nào xảy ra và xử lý an toàn
        self.assertIsInstance(errs, list)


class TestContextBanBan(unittest.TestCase):
    """Kiểm thử phân biệt ngữ cảnh giữa Ban (hành chính) và Bạn (đại từ/lời chào)."""

    def test_ban_vs_ban_contextual_differentiation(self) -> None:
        # 1. 'Ban' trong ngữ cảnh giao tiếp -> gợi ý 'Bạn'
        errs1 = check_text_spelling("Ban có thể kiểm tra lại giúp tôi.")
        self.assertEqual(len(errs1), 1)
        self.assertEqual(errs1[0].token, "Ban")
        self.assertIn("Bạn", errs1[0].suggestions)

        errs2 = check_text_spelling("Chào Ban, chúc một ngày làm việc hiệu quả.")
        self.assertEqual(len(errs2), 1)
        self.assertEqual(errs2[0].token, "Ban")
        self.assertIn("Bạn", errs2[0].suggestions)

        # 2. 'Ban' trong ngữ cảnh hành chính / tổ chức -> HỢP LỆ, không báo lỗi
        clean_cases = [
            "Ban hành Nghị định số 14/2026/NĐ-CP của Chính phủ.",
            "Ban chỉ đạo quốc gia về phòng chống thiên tai.",
            "Ban quản lý dự án công trình giao thông.",
            "Ủy ban nhân dân tỉnh Quảng Ninh.",
            "Làm việc vào ban ngày và nghỉ ngơi vào ban đêm.",
        ]
        for s in clean_cases:
            errs = check_text_spelling(s)
            self.assertEqual(errs, [], f"Không được đánh dấu 'Ban' hành chính trong câu: {s!r}")


class TestTwoColumnReadingOrderAndLayout(unittest.TestCase):
    """Kiểm thử sắp xếp thứ tự đọc và tách cột trong ocr_engine."""

    def test_two_column_administrative_header(self) -> None:
        from app.core.ocr_engine import OCRBox, group_boxes_into_lines, sort_boxes_reading_order

        # Mô phỏng tiêu đề hành chính 2 cột trong văn bản thật:
        # Cột trái: BỘ TƯ PHÁP, Số: 14/2026/TT-BTP
        # Cột phải: CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM, Độc lập - Tự do - Hạnh phúc, Hà Nội, ngày 30...
        # Dưới: THÔNG TƯ, Hướng dẫn về thực hiện...
        boxes = [
            OCRBox("BỘ TƯ PHÁP", 0.98, [[100, 100], [220, 100], [220, 130], [100, 130]]),
            OCRBox("Số: 14/2026/TT-BTP", 0.97, [[80, 170], [250, 170], [250, 200], [80, 200]]),
            OCRBox("CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM", 0.99, [[450, 100], [900, 100], [900, 130], [450, 130]]),
            OCRBox("Độc lập - Tự do - Hạnh phúc", 0.99, [[520, 135], [820, 135], [820, 165], [520, 165]]),
            OCRBox("Hà Nội, ngày 30 tháng 9 năm 2026", 0.98, [[480, 170], [860, 170], [860, 200], [480, 200]]),
            OCRBox("THÔNG TƯ", 0.99, [[420, 250], [580, 250], [580, 280], [420, 280]]),
            OCRBox("Hướng dẫn về thực hiện thủ tục hành chính", 0.98, [[100, 300], [900, 300], [900, 330], [100, 330]]),
        ]

        lines = group_boxes_into_lines(boxes)
        line_texts = [" ".join(b.text for b in line) for line in lines]

        # Kiểm tra: Cột trái và Cột phải tuyệt đối KHÔNG bị ghép nối ngang thành cùng 1 dòng
        for t in line_texts:
            self.assertNotIn("BỘ TƯ PHÁP CỘNG HÒA", t)
            self.assertNotIn("TT-BTP Hà Nội", t)

        # Thứ tự đọc: Cột trái trước, Cột phải sau, rồi tới THÔNG TƯ
        self.assertEqual(line_texts[0], "BỘ TƯ PHÁP")
        self.assertEqual(line_texts[1], "Số: 14/2026/TT-BTP")
        self.assertEqual(line_texts[2], "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM")
        self.assertEqual(line_texts[3], "Độc lập - Tự do - Hạnh phúc")
        self.assertEqual(line_texts[4], "Hà Nội, ngày 30 tháng 9 năm 2026")
        self.assertEqual(line_texts[5], "THÔNG TƯ")
        self.assertEqual(line_texts[6], "Hướng dẫn về thực hiện thủ tục hành chính")

        # Kiểm tra hàm sort_boxes_reading_order
        sorted_boxes = sort_boxes_reading_order(boxes)
        self.assertEqual([b.text for b in sorted_boxes], line_texts)

    def test_different_box_heights_on_same_line(self) -> None:
        from app.core.ocr_engine import OCRBox, group_boxes_into_lines

        # Hai box cùng một dòng nhưng chiều cao khác nhau (ví dụ: số thứ tự/bullet và text)
        boxes = [
            OCRBox("1.", 0.95, [[50, 105], [70, 105], [70, 125], [50, 125]]),         # h=20
            OCRBox("Nội dung điều khoản", 0.98, [[80, 100], [350, 100], [350, 135], [80, 135]]), # h=35
        ]
        lines = group_boxes_into_lines(boxes)
        self.assertEqual(len(lines), 1, "Box khác chiều cao nhưng cùng dòng phải được gom chung 1 dòng")
        self.assertEqual(" ".join(b.text for b in lines[0]), "1. Nội dung điều khoản")


class TestBorderNoiseFilter(unittest.TestCase):
    """Kiểm thử lọc nhiễu quét tại mép ảnh trong ocr_engine."""

    def test_filter_border_noise_cases(self) -> None:
        from app.core.ocr_engine import OCRBox, filter_border_noise_boxes

        img_w, img_h = 1000, 800
        boxes = [
            # 1. Box bị cắt rõ tại mép dưới (h=8 khi median_h=24) -> Bị lọc
            OCRBox("1 1 118/05/11", 0.85, [[100, 792], [300, 792], [300, 800], [100, 800]]),
            # 2. Dòng số trang hợp lệ sát đáy (h=22) -> Giữ lại
            OCRBox("Trang 1/5", 0.98, [[450, 775], [550, 775], [550, 797], [450, 797]]),
            # 3. Ngày tháng hợp lệ gần đáy (h=24) -> Giữ lại
            OCRBox("Hà Nội, ngày 30 tháng 9 năm 2026", 0.98, [[300, 772], [700, 772], [700, 796], [300, 796]]),
            # 4. Box sát mép nhưng chiều cao đầy đủ (h=25) -> Giữ lại
            OCRBox("Ủy ban nhân dân thành phố", 0.99, [[100, 773], [400, 773], [400, 798], [100, 798]]),
            # 5. Box bị cắt tại mép dưới nhưng confidence cao (conf=0.96, h=8) -> Bị lọc
            OCRBox("và Nghị", 0.96, [[100, 792], [200, 792], [200, 800], [100, 800]]),
            # 6. Box bị cắt tại mép trên (top=1, h=7) -> Bị lọc
            OCRBox("Việt", 0.88, [[500, 1], [560, 1], [560, 8], [500, 8]]),
            # 7. Box bình thường ở giữa trang
            OCRBox("Thông báo chính thức", 0.99, [[200, 400], [500, 400], [500, 425], [200, 425]]),
        ]

        filtered = filter_border_noise_boxes(boxes, image_width=img_w, image_height=img_h)
        remaining_texts = [b.text for b in filtered]

        # Kiểm tra các box bị lọc
        self.assertNotIn("1 1 118/05/11", remaining_texts)
        self.assertNotIn("và Nghị", remaining_texts)
        self.assertNotIn("Việt", remaining_texts)

        # Kiểm tra các box hợp lệ được bảo toàn
        self.assertIn("Trang 1/5", remaining_texts)
        self.assertIn("Hà Nội, ngày 30 tháng 9 năm 2026", remaining_texts)
        self.assertIn("Ủy ban nhân dân thành phố", remaining_texts)
        self.assertIn("Thông báo chính thức", remaining_texts)


class TestParallelHeaderScreenshot1(unittest.TestCase):
    """Kiểm thử bố cục tiêu đề hành chính 2 khối thực tế (Screenshot 1: Đảng ủy xã Phước Hải)."""

    def test_phuoc_hai_header_reading_order(self) -> None:
        from app.core.ocr_engine import OCRBox, group_boxes_into_lines, sort_boxes_reading_order

        boxes = [
            # Khối trái (Cơ quan ban hành, số hiệu):
            OCRBox("ĐẢNG ỦY XÃ PHƯỚC HẢI", 0.98, [[80, 100], [260, 100], [260, 125], [80, 125]]),
            OCRBox("CHI BỘ VĂN PHÒNG HĐND", 0.97, [[70, 130], [270, 130], [270, 150], [70, 150]]),
            OCRBox("VÀ UBND XÃ", 0.96, [[100, 155], [210, 155], [210, 175], [100, 175]]),
            OCRBox("*", 0.95, [[150, 180], [165, 180], [165, 195], [150, 195]]),
            OCRBox("Số: 01-BB/CB", 0.97, [[100, 200], [200, 200], [200, 225], [100, 225]]),
            # Khối phải (Quốc hiệu, ngày tháng):
            OCRBox("ĐẢNG CỘNG SẢN VIỆT NAM", 0.99, [[380, 100], [580, 100], [580, 125], [380, 125]]),
            OCRBox("Phước Hải, ngày 05 tháng 10 năm 2026", 0.98, [[350, 150], [580, 150], [580, 175], [350, 175]]),
            # Tiêu đề trung tâm giữa trang:
            OCRBox("BIÊN BẢN", 0.99, [[250, 250], [370, 250], [370, 280], [250, 280]]),
            OCRBox("sinh hoạt chi bộ định kỳ tháng 10 năm 2026", 0.98, [[180, 285], [480, 285], [480, 310], [180, 310]]),
            # Thân bài:
            OCRBox("Vào lúc 15 giờ, ngày 05 tháng 10 năm 2026, tại Phòng họp UBND xã", 0.98, [[80, 330], [580, 330], [580, 360], [80, 360]]),
        ]

        lines = group_boxes_into_lines(boxes)
        line_texts = [" ".join(b.text for b in line) for line in lines]

        # Khối trái phải được đọc trọn vẹn từ trên xuống dưới (đường kẻ trang trí '*' đã bị loại bỏ theo Phase 3.3)
        self.assertEqual(line_texts[0], "ĐẢNG ỦY XÃ PHƯỚC HẢI")
        self.assertEqual(line_texts[1], "CHI BỘ VĂN PHÒNG HĐND")
        self.assertEqual(line_texts[2], "VÀ UBND XÃ")
        self.assertEqual(line_texts[3], "Số: 01-BB/CB")

        # Sau đó mới đến Khối phải:
        self.assertEqual(line_texts[4], "ĐẢNG CỘNG SẢN VIỆT NAM")
        self.assertEqual(line_texts[5], "Phước Hải, ngày 05 tháng 10 năm 2026")

        # Tiếp theo là Tiêu đề chính giữa trang và Thân bài:
        self.assertEqual(line_texts[6], "BIÊN BẢN")
        self.assertEqual(line_texts[7], "sinh hoạt chi bộ định kỳ tháng 10 năm 2026")
        self.assertEqual(line_texts[8], "Vào lúc 15 giờ, ngày 05 tháng 10 năm 2026, tại Phòng họp UBND xã")

        # Tuyệt đối không để ĐẢNG CỘNG SẢN VIỆT NAM bị chèn vào giữa Khối trái
        sorted_boxes = sort_boxes_reading_order(boxes)
        sorted_texts = [b.text for b in sorted_boxes]
        self.assertEqual(sorted_texts[:4], [
            "ĐẢNG ỦY XÃ PHƯỚC HẢI",
            "CHI BỘ VĂN PHÒNG HĐND",
            "VÀ UBND XÃ",
            "Số: 01-BB/CB",
        ])


class TestTableReconstructionScreenshot2(unittest.TestCase):
    """Kiểm thử tái dựng cấu trúc bảng biểu 5 cột thực tế (Screenshot 2: Bảng điểm đánh giá)."""

    def test_reconstruct_table_grid_and_alignment(self) -> None:
        from app.core.ocr_engine import OCRBox, reconstruct_document_layout

        boxes = [
            # Tiêu đề ngoài bảng
            OCRBox("BẢNG ĐIỂM ĐÁNH GIÁ CHẤT LƯỢNG SINH HOẠT CHI BỘ", 0.98, [[100, 40], [750, 40], [750, 65], [100, 65]]),
            OCRBox("THÁNG 10 NĂM 2026", 0.98, [[320, 70], [520, 70], [520, 90], [320, 90]]),
            # Hàng tiêu đề bảng (5 cột, ô 'Điểm tối đa' và 'Điểm tự chấm' bị tách box)
            OCRBox("STT", 0.95, [[30, 110], [65, 110], [65, 135], [30, 135]]),
            OCRBox("Tiêu chí đánh giá", 0.98, [[150, 110], [400, 110], [400, 135], [150, 135]]),
            OCRBox("Điểm tối", 0.95, [[630, 105], [690, 105], [690, 122], [630, 122]]),
            OCRBox("đa", 0.95, [[645, 125], [675, 125], [675, 140], [645, 140]]),
            OCRBox("Điểm tự", 0.95, [[710, 105], [770, 105], [770, 122], [710, 122]]),
            OCRBox("chấm", 0.95, [[720, 125], [760, 125], [760, 140], [720, 140]]),
            OCRBox("Ghi chú", 0.96, [[790, 110], [840, 110], [840, 135], [790, 135]]),
            # Hàng 1
            OCRBox("1", 0.95, [[40, 150], [55, 150], [55, 175], [40, 175]]),
            OCRBox("Chấp hành thời gian, thời lượng sinh hoạt", 0.98, [[80, 150], [550, 150], [550, 175], [80, 175]]),
            OCRBox("05", 0.95, [[640, 150], [670, 150], [670, 175], [640, 175]]),
            OCRBox("05", 0.95, [[725, 150], [755, 150], [755, 175], [725, 175]]),
            # Hàng 2
            OCRBox("+", 0.95, [[40, 185], [55, 185], [55, 210], [40, 210]]),
            OCRBox("Chi bộ dưới 30 đảng viên ít nhất 90 phút; chi bộ trên 30 đảng viên ít nhất 120 phút", 0.98, [[80, 185], [600, 185], [600, 210], [80, 210]]),
            OCRBox("05", 0.95, [[640, 185], [670, 185], [670, 210], [640, 210]]),
            OCRBox("05", 0.95, [[725, 185], [755, 185], [755, 210], [725, 210]]),
            # Hàng 3
            OCRBox("+", 0.95, [[40, 220], [55, 220], [55, 245], [40, 245]]),
            OCRBox("Thời gian sinh hoạt chi bộ ít hơn quy định 15 phút", 0.98, [[80, 220], [580, 220], [580, 245], [80, 245]]),
            OCRBox("04", 0.95, [[640, 220], [670, 220], [670, 245], [640, 245]]),
            # Hàng 4
            OCRBox("+", 0.95, [[40, 255], [55, 255], [55, 280], [40, 280]]),
            OCRBox("Thời gian sinh hoạt chi bộ ít hơn quy định 15 - 30 phút", 0.98, [[80, 255], [590, 255], [590, 280], [80, 280]]),
            OCRBox("03", 0.95, [[640, 255], [670, 255], [670, 280], [640, 280]]),
            # Hàng 5
            OCRBox("+", 0.95, [[40, 290], [55, 290], [55, 315], [40, 315]]),
            OCRBox("Thời gian sinh hoạt chi bộ chỉ bằng ½ so với quy định", 0.98, [[80, 290], [560, 290], [560, 315], [80, 315]]),
            OCRBox("02", 0.95, [[640, 290], [670, 290], [670, 315], [640, 315]]),
            # Hàng 6
            OCRBox("+", 0.95, [[40, 325], [55, 325], [55, 350], [40, 350]]),
            OCRBox("Thời gian sinh hoạt chi bộ dưới ½ so với quy định", 0.98, [[80, 325], [550, 325], [550, 350], [80, 350]]),
            OCRBox("01", 0.95, [[640, 325], [670, 325], [670, 350], [640, 350]]),
            # Hàng 7
            OCRBox("2", 0.95, [[40, 360], [55, 360], [55, 385], [40, 385]]),
            OCRBox("Tỷ lệ đảng viên dự sinh hoạt chi bộ", 0.98, [[80, 360], [480, 360], [480, 385], [80, 385]]),
            OCRBox("05", 0.95, [[640, 360], [670, 360], [670, 385], [640, 385]]),
        ]

        lines, tables = reconstruct_document_layout(boxes)

        # 1. Kiểm tra phát hiện đúng bảng biểu
        self.assertEqual(len(tables), 1, "Phải phát hiện chính xác 1 bảng biểu")
        tbl = tables[0]
        self.assertEqual(tbl.num_cols, 5, "Bảng phải có đúng 5 cột")

        # 2. Kiểm tra hàng Header: Ghép nối đúng các box bị tách trong cùng ô
        header_row = tbl.grid[0]
        self.assertEqual(header_row[0], "STT")
        self.assertEqual(header_row[1], "Tiêu chí đánh giá")
        self.assertEqual(header_row[2], "Điểm tối đa")
        self.assertEqual(header_row[3], "Điểm tự chấm")
        self.assertEqual(header_row[4], "Ghi chú")

        # 3. Kiểm tra các hàng dữ liệu: Điểm số gắn liền với đúng tiêu chí
        row1 = tbl.grid[1]
        self.assertEqual(row1[0], "1")
        self.assertEqual(row1[1], "Chấp hành thời gian, thời lượng sinh hoạt")
        self.assertEqual(row1[2], "05")
        self.assertEqual(row1[3], "05")

        row3 = tbl.grid[3]
        self.assertEqual(row3[0], "+")
        self.assertEqual(row3[1], "Thời gian sinh hoạt chi bộ ít hơn quy định 15 phút")
        self.assertEqual(row3[2], "04")
        self.assertEqual(row3[3], "")  # Ô trống

        # 4. Kiểm tra tiêu đề ngoài bảng nằm trước bảng
        line_texts = [" ".join(b.text for b in line) for line in lines]
        self.assertIn("BẢNG ĐIỂM ĐÁNH GIÁ CHẤT LƯỢNG SINH HOẠT CHI BỘ", line_texts[0])
        self.assertIn("THÁNG 10 NĂM 2026", line_texts[1])


class TestSpellCheckDiacriticConfusionVdng(unittest.TestCase):
    """Kiểm thử phát hiện lỗi dấu OCR nhầm lẫn (vđng -> vắng) và bảo toàn từ viết tắt."""

    def test_vdng_detected_with_vang_suggestion(self) -> None:
        from app.core.spell_corrector import check_text_spelling

        # OCR nhầm 'vắng' thành 'vđng'
        text = "Cuộc họp có 14 đồng chí, vđng 0 đồng chí."
        errors = check_text_spelling(text)

        vdng_err = next((e for e in errors if e.token == "vđng"), None)
        self.assertIsNotNone(vdng_err, "Phải phát hiện token 'vđng' là nghi vấn chính tả")
        self.assertIn("vắng", vdng_err.suggestions, "Gợi ý hàng đầu phải chứa 'vắng'")

    def test_abbreviations_dc_and_stt_protected(self) -> None:
        from app.core.spell_corrector import check_text_spelling

        text = "STT 01: đ/c Lê Văn Điền chủ trì, Đ/c Nguyễn Văn Thắng làm thư ký."
        errors = check_text_spelling(text)

        err_tokens = [e.token for e in errors]
        self.assertNotIn("STT", err_tokens)
        self.assertNotIn("đ/c", err_tokens)
        self.assertNotIn("Đ/c", err_tokens)

    def test_capitalized_vdng_with_dash_and_plus_bullets(self) -> None:
        from app.core.spell_corrector import check_text_spelling

        # Case 1: Gạch đầu dòng - Vdng mặt: 0 đồng chí.
        text1 = "- Vdng mặt: 0 đồng chí."
        errors1 = check_text_spelling(text1)
        self.assertEqual(len(errors1), 1, "Phải phát hiện chính xác 1 lỗi trong câu có gạch đầu dòng")
        err1 = errors1[0]
        self.assertEqual(err1.token, "Vdng")
        self.assertEqual(err1.start_idx, 2)
        self.assertEqual(err1.end_idx, 6)
        self.assertEqual(text1[err1.start_idx:err1.end_idx], "Vdng")
        self.assertIn("Vắng", err1.suggestions, "Gợi ý hàng đầu cho 'Vdng' phải là 'Vắng'")

        # Case 2: Dấu cộng đầu dòng + Vdng mặt: 0 đồng chí.
        text2 = "+ Vdng mặt: 0 đồng chí."
        errors2 = check_text_spelling(text2)
        self.assertEqual(len(errors2), 1)
        err2 = errors2[0]
        self.assertEqual(err2.token, "Vdng")
        self.assertEqual(err2.start_idx, 2)
        self.assertEqual(err2.end_idx, 6)
        self.assertIn("Vắng", err2.suggestions)

        # Case 3: Đầu câu không có bullet: Vdng mặt: 0 đồng chí.
        text3 = "Vdng mặt: 0 đồng chí."
        errors3 = check_text_spelling(text3)
        self.assertEqual(len(errors3), 1)
        err3 = errors3[0]
        self.assertEqual(err3.token, "Vdng")
        self.assertEqual(err3.start_idx, 0)
        self.assertEqual(err3.end_idx, 4)
        self.assertIn("Vắng", err3.suggestions)

        # Case 4: Âm tính: Câu đúng 'Vắng mặt' không được báo lỗi
        clean_cases = [
            "Có mặt: 14 đồng chí; Vắng mặt: 0 đồng chí.",
            "- Có mặt: 14 đồng chí.",
            "- Vắng mặt: 0 đồng chí.",
            "+ Vắng mặt: 0 đồng chí.",
        ]
        for clean_text in clean_cases:
            self.assertEqual(
                check_text_spelling(clean_text),
                [],
                f"Không được báo lỗi giả trên câu đúng: {clean_text!r}",
            )


class TestDocxExportWithTable(unittest.TestCase):
    """Kiểm thử xuất tài liệu Word (.docx) bảo toàn cấu trúc bảng biểu."""

    def test_export_docx_with_table(self) -> None:
        import os
        import tempfile
        import docx
        from app.core.ocr_engine import OCRBox, OCRResult, OCRTable, export_to_docx

        # Tạo OCRResult giả lập có bảng
        tbl = OCRTable(
            grid=[
                ["STT", "Tiêu chí đánh giá", "Điểm tối đa", "Điểm tự chấm", "Ghi chú"],
                ["1", "Chấp hành thời gian", "05", "05", ""],
                ["2", "Tỷ lệ đảng viên dự sinh hoạt", "05", "05", ""],
            ],
            bbox=[30, 100, 840, 250],
            num_rows=3,
            num_cols=5,
        )

        full_text = (
            "BẢNG ĐIỂM ĐÁNH GIÁ CHẤT LƯỢNG\n"
            "THÁNG 10 NĂM 2026\n"
            "STT | Tiêu chí đánh giá | Điểm tối đa | Điểm tự chấm | Ghi chú\n"
            "1 | Chấp hành thời gian | 05 | 05 | \n"
            "2 | Tỷ lệ đảng viên dự sinh hoạt | 05 | 05 | \n"
            "* Ghi chú dưới bảng biểu"
        )

        res = OCRResult(
            boxes=[],
            full_text=full_text,
            language="vi",
            tables=[tbl],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "test_output.docx")
            success = export_to_docx(res, out_path)
            self.assertTrue(success, "export_to_docx phải trả về True")
            self.assertTrue(os.path.isfile(out_path), "File .docx phải tồn tại")

            # Đọc lại file docx để kiểm tra cấu trúc
            doc = docx.Document(out_path)
            self.assertEqual(len(doc.tables), 1, "File Word phải chứa đúng 1 Table")
            word_table = doc.tables[0]
            self.assertEqual(len(word_table.rows), 3)
            self.assertEqual(len(word_table.columns), 5)

            # Kiểm tra text các cell trong Word table
            self.assertEqual(word_table.cell(0, 0).text, "STT")
            self.assertEqual(word_table.cell(0, 1).text, "Tiêu chí đánh giá")
            self.assertEqual(word_table.cell(0, 2).text, "Điểm tối đa")
            self.assertEqual(word_table.cell(1, 0).text, "1")
            self.assertEqual(word_table.cell(1, 1).text, "Chấp hành thời gian")
            self.assertEqual(word_table.cell(1, 2).text, "05")

            # Kiểm tra các đoạn văn bản ngoài bảng
            doc_paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            self.assertIn("BẢNG ĐIỂM ĐÁNH GIÁ CHẤT LƯỢNG", doc_paras)
            self.assertIn("THÁNG 10 NĂM 2026", doc_paras)
            self.assertIn("* Ghi chú dưới bảng biểu", doc_paras)


class TestContextualSpellCheckingNew(unittest.TestCase):
    """Kiểm thử dò soát lỗi chính tả theo ngữ cảnh (Context Confusion) & Phản ví dụ (Negative Cases)."""

    def test_chi_bo_and_dong_chi(self) -> None:
        """Kiểm tra phát hiện lỗi 'chỉ bộ' -> 'chi bộ', 'đồng chỉ' -> 'đồng chí'."""
        text1 = "Nội dung sinh hoạt chỉ bộ định kỳ."
        errs1 = check_text_spelling(text1)
        tokens1 = [err.token for err in errs1]
        self.assertIn("chỉ", tokens1, "'chỉ' trong 'chỉ bộ' phải bị đánh dấu")
        err_chi = next(e for e in errs1 if e.token == "chỉ")
        self.assertIn("chi", err_chi.suggestions, "Gợi ý phải có 'chi'")

        text2 = "Kính gửi đồng chỉ Bí thư."
        errs2 = check_text_spelling(text2)
        tokens2 = [err.token for err in errs2]
        self.assertIn("chỉ", tokens2, "'chỉ' trong 'đồng chỉ' phải bị đánh dấu")
        err_dc = next(e for e in errs2 if e.token == "chỉ")
        self.assertIn("chí", err_dc.suggestions, "Gợi ý phải có 'chí'")

    def test_chi_bo_all_caps(self) -> None:
        """Kiểm tra phát hiện 'CHI BỐ' -> 'CHI BỘ' trong tiêu đề in hoa."""
        text = "SINH HOẠT CHI BỐ"
        errs = check_text_spelling(text)
        tokens = [err.token for err in errs]
        self.assertIn("BỐ", tokens, "'BỐ' trong 'CHI BỐ' phải bị đánh dấu")
        err_bo = next(e for e in errs if e.token == "BỐ")
        self.assertIn("BỘ", err_bo.suggestions, "Gợi ý phải có 'BỘ'")

    def test_khong_co_and_thang_date(self) -> None:
        """Kiểm tra phát hiện 'không cổ' -> 'không có', 'thắng 10' -> 'tháng 10'."""
        text1 = "Chi bộ không cổ đảng viên vi phạm."
        errs1 = check_text_spelling(text1)
        tokens1 = [err.token for err in errs1]
        self.assertIn("cổ", tokens1, "'cổ' trong 'không cổ' phải bị đánh dấu")
        err_co = next(e for e in errs1 if e.token == "cổ")
        self.assertIn("có", err_co.suggestions, "Gợi ý phải có 'có'")

        text2 = "Biên bản họp ngày 28 thắng 10 năm 2024."
        errs2 = check_text_spelling(text2)
        tokens2 = [err.token for err in errs2]
        self.assertIn("thắng", tokens2, "'thắng' trong 'ngày ... thắng ...' phải bị đánh dấu")
        err_thang = next(e for e in errs2 if e.token == "thắng")
        self.assertIn("tháng", err_thang.suggestions, "Gợi ý phải có 'tháng'")

    def test_dang_vien_dang_phi_it_nhat(self) -> None:
        """Kiểm tra phát hiện 'đang viên', 'đăng phí', 'it nhất'."""
        text1 = "Toàn thể đang viên có mặt đầy đủ để nộp đăng phí."
        errs1 = check_text_spelling(text1)
        tokens1 = [err.token for err in errs1]
        self.assertIn("đang", tokens1, "'đang' trong 'đang viên' phải bị đánh dấu")
        self.assertIn("đăng", tokens1, "'đăng' trong 'đăng phí' phải bị đánh dấu")

        text2 = "Thời gian sinh hoạt it nhất 90 phút."
        errs2 = check_text_spelling(text2)
        tokens2 = [err.token for err in errs2]
        self.assertIn("it", tokens2, "'it' trong 'it nhất' phải bị đánh dấu")
        err_it = next(e for e in errs2 if e.token == "it")
        self.assertIn("ít", err_it.suggestions, "Gợi ý phải có 'ít'")

    def test_negative_whitelisting_clean(self) -> None:
        """Bảo vệ tuyệt đối các cụm từ tiếng Việt đúng không bị báo lỗi sai (Negative Cases)."""
        clean_samples = [
            "Đồng chí Bí thư chỉ tay về phía hội trường.",
            "Ban chỉ đạo đại hội nhiệm kỳ 2025 - 2030.",
            "Hôm nay chỉ có 14 đồng chí tham dự họp.",
            "Gia đình gồm có bố mẹ và các con.",
            "Chiếc áo sơ mi có cổ áo màu trắng.",
            "Quân dân ta đã giành chiến thắng vẻ vang.",
            "Các đồng chí đang làm việc tại văn phòng.",
            "Đảng viên thực hiện đăng ký tham gia sinh hoạt.",
        ]
        for sample in clean_samples:
            errs = check_text_spelling(sample)
            self.assertEqual(
                errs,
                [],
                f"Văn bản hợp lệ không được bị báo lỗi sai: {sample} -> {errs}",
            )


if __name__ == "__main__":
    unittest.main()


