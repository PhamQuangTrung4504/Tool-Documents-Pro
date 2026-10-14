#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests/test_ui_spell_check.py
============================
Kiểm thử tích hợp UI cho MainWindow:
- Khởi tạo các nút và Action Kiểm tra chính tả & Bỏ đánh dấu.
- Cơ chế đánh dấu ExtraSelections (màu đỏ, gạch chân đỏ, nền đỏ nhạt).
- Đảm bảo 100% nội dung văn bản không bị sửa đổi trước và sau khi kiểm tra.
- Xóa đánh dấu khi bấm nút Bỏ đánh dấu hoặc khi người dùng chỉnh sửa văn bản.
- Thanh trạng thái hiển thị chính xác thông báo.
"""

from __future__ import annotations

import os
import sys
import unittest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QTextCharFormat
from app.ui.main_window import MainWindow


class TestMainWindowSpellCheckUI(unittest.TestCase):
    """Kiểm thử tích hợp giao diện PyQt5 của MainWindow."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([""])

    def setUp(self) -> None:
        self.window = MainWindow()

    def tearDown(self) -> None:
        self.window.close()

    def test_ui_components_exist(self) -> None:
        """Kiểm tra sự tồn tại của các thành phần giao diện mới."""
        self.assertTrue(hasattr(self.window, "act_spell_check"))
        self.assertTrue(hasattr(self.window, "act_clear_spell"))
        self.assertTrue(hasattr(self.window, "btn_spell_check"))
        self.assertTrue(hasattr(self.window, "btn_clear_spell"))

        self.assertIn("chính tả", self.window.act_spell_check.text().lower())
        self.assertIn("đánh dấu", self.window.act_clear_spell.text().lower())
        self.assertIn("chính tả", self.window.btn_spell_check.text().lower())
        self.assertIn("đánh dấu", self.window.btn_clear_spell.text().lower())

    def test_empty_text_spell_check(self) -> None:
        """Khi văn bản trống, thông báo phù hợp và không phát sinh lỗi."""
        self.window.text_result.setPlainText("")
        self.window._on_check_spelling()
        self.assertEqual(self.window.status_label.text(), "Không đủ dữ liệu để kiểm tra.")
        self.assertEqual(len(self.window.text_result.extraSelections()), 0)

    def test_spell_check_and_highlighting_workflow(self) -> None:
        """Quy trình kiểm tra chính tả: đánh dấu đỏ và bảo toàn 100% nội dung văn bản."""
        sample_text = (
            "Cộng hòa xã hội chủ nghĩa Việt Nam\n"
            "Độc lập - Tự do - Hạnh phúc\n\n"
            "Kế họach kiểm tra và ra sóat thủ tục hành chính nghiêm tuc.\n"
        )
        self.window.text_result.setPlainText(sample_text)

        # Chạy kiểm tra chính tả trực tiếp qua slot và đợi worker
        self.window._on_check_spelling()
        if self.window._spell_worker:
            self.window._spell_worker.wait(3000)
            QApplication.processEvents()

        # Xác minh số lượng vị trí nghi vấn được phát hiện (họach, sóat, tuc -> 3 lỗi)
        selections = self.window.text_result.extraSelections()
        self.assertEqual(len(selections), 3, f"Phải có 3 vị trí được đánh dấu, nhận được: {len(selections)}")
        self.assertIn("3", self.window.status_label.text())

        # Xác minh định dạng đánh dấu: Màu đỏ và WaveUnderline
        first_sel = selections[0]
        fmt: QTextCharFormat = first_sel.format
        self.assertEqual(fmt.foreground().color().name(), "#ff4d4f")
        self.assertEqual(fmt.underlineStyle(), QTextCharFormat.WaveUnderline)
        self.assertEqual(fmt.underlineColor().name(), "#ff4d4f")

        # NGUYÊN TẮC QUAN TRỌNG NHẤT: Nội dung văn bản không bị thay đổi dù chỉ 1 ký tự
        current_text = self.window.text_result.toPlainText()
        self.assertEqual(
            current_text,
            sample_text,
            "Nội dung văn bản phải giống nhau tuyệt đối trước và sau khi đánh dấu",
        )

        # Thao tác bỏ đánh dấu
        self.window._on_clear_spelling_highlights()
        self.assertEqual(len(self.window.text_result.extraSelections()), 0)
        self.assertEqual(self.window.status_label.text(), "Đã xoá các đánh dấu chính tả.")
        self.assertEqual(self.window.text_result.toPlainText(), sample_text)

    def test_text_edit_clears_stale_highlights(self) -> None:
        """Khi người dùng chỉnh sửa văn bản, các đánh dấu cũ phải tự động bị xóa."""
        sample_text = "Văn bản đap ứng yêu cầu"
        self.window.text_result.setPlainText(sample_text)

        self.window._on_check_spelling()
        if self.window._spell_worker:
            self.window._spell_worker.wait(3000)
            QApplication.processEvents()

        self.assertEqual(len(self.window.text_result.extraSelections()), 1)

        # Người dùng sửa văn bản
        self.window.text_result.insertPlainText(" bổ sung")
        # extraSelections phải được tự động dọn dẹp
        self.assertEqual(
            len(self.window.text_result.extraSelections()),
            0,
            "Đánh dấu phải bị xóa khi người dùng chỉnh sửa văn bản",
        )

    def test_apply_suggestion_and_undo(self) -> None:
        """Người dùng chọn gợi ý sửa: thay đúng từ, cập nhật highlight và hỗ trợ Undo."""
        sample_text = "Kế họach triển khai công việc"
        self.window.text_result.setPlainText(sample_text)

        self.window._on_check_spelling()
        if self.window._spell_worker:
            self.window._spell_worker.wait(3000)
            QApplication.processEvents()

        self.assertEqual(len(self.window._active_spell_errors), 1)
        err = self.window._active_spell_errors[0]
        self.assertEqual(err.token, "họach")
        self.assertIn("hoạch", err.suggestions)

        # Áp dụng gợi ý sửa
        self.window._apply_spell_suggestion(err, "hoạch")
        self.assertEqual(
            self.window.text_result.toPlainText(),
            "Kế hoạch triển khai công việc",
            "Văn bản phải được thay thế chính xác token được chọn",
        )
        self.assertEqual(len(self.window.text_result.extraSelections()), 0)

        # Kiểm tra tính năng Undo (Ctrl+Z)
        self.window.text_result.undo()
        self.assertEqual(
            self.window.text_result.toPlainText(),
            sample_text,
            "Undo phải khôi phục nguyên trạng văn bản trước khi áp dụng gợi ý",
        )

    def test_ignore_spell_error(self) -> None:
        """Bỏ qua vị trí nghi vấn: không thay đổi văn bản, chỉ xóa đánh dấu của lỗi đó."""
        sample_text = "Kế họach triển khai công việc"
        self.window.text_result.setPlainText(sample_text)

        self.window._on_check_spelling()
        if self.window._spell_worker:
            self.window._spell_worker.wait(3000)
            QApplication.processEvents()

        self.assertEqual(len(self.window._active_spell_errors), 1)
        err = self.window._active_spell_errors[0]

        # Bỏ qua lỗi
        self.window._ignore_spell_error(err)
        self.assertEqual(self.window.text_result.toPlainText(), sample_text)
        self.assertEqual(len(self.window.text_result.extraSelections()), 0)
        self.assertEqual(len(self.window._active_spell_errors), 0)

    def test_vdng_highlight_and_suggest_in_ui(self) -> None:
        """Kiểm thử tích hợp UI: 'Vdng mặt' được bôi đỏ đúng span [2:6] và gợi ý 'Vắng'."""
        sample_text = "- Vdng mặt: 0 đồng chí."
        self.window.text_result.setPlainText(sample_text)

        self.window._on_check_spelling()
        if self.window._spell_worker:
            self.window._spell_worker.wait(3000)
            QApplication.processEvents()

        # 1. Xác minh có 1 vị trí nghi vấn được phát hiện
        selections = self.window.text_result.extraSelections()
        self.assertEqual(len(selections), 1, "Phải có đúng 1 vị trí được đánh dấu trên UI")
        cursor = selections[0].cursor
        self.assertEqual(cursor.selectionStart(), 2)
        self.assertEqual(cursor.selectionEnd(), 6)
        self.assertEqual(cursor.selectedText(), "Vdng")

        # 2. Xác minh lỗi và gợi ý 'Vắng'
        self.assertEqual(len(self.window._active_spell_errors), 1)
        err = self.window._active_spell_errors[0]
        self.assertEqual(err.token, "Vdng")
        self.assertIn("Vắng", err.suggestions)

        # 3. Áp dụng gợi ý 'Vắng'
        self.window._apply_spell_suggestion(err, "Vắng")
        self.assertEqual(
            self.window.text_result.toPlainText(),
            "- Vắng mặt: 0 đồng chí.",
            "Văn bản sau khi người dùng chọn gợi ý phải trở thành '- Vắng mặt: 0 đồng chí.'",
        )

        # 4. Thử Undo để kiểm tra khôi phục
        self.window.text_result.undo()
        self.assertEqual(
            self.window.text_result.toPlainText(),
            sample_text,
            "Undo phải trả về nguyên vẹn văn bản OCR gốc",
        )


if __name__ == "__main__":
    unittest.main()
