from __future__ import annotations

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from app.core.models import OCRResult
from app.ui.main_window import MainWindow


class TestMainWindowSpellCheckUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([""])

    def setUp(self) -> None:
        self.window = MainWindow()

    def tearDown(self) -> None:
        self.window.close()

    def test_full_text_editing_tab_and_widget_are_removed(self) -> None:
        self.assertEqual(self.window.document_tabs.count(), 2)
        self.assertFalse(hasattr(self.window, "text_result"))
        self.assertEqual(self.window.document_tabs.tabText(0), "Xem trước Word")
        self.assertEqual(self.window.document_tabs.tabText(1), "Rà soát OCR")

    def test_empty_document_spell_check_is_ignored(self) -> None:
        self.window._on_check_spelling()
        self.assertEqual(self.window.status_label.text(), "Không đủ dữ liệu để kiểm tra.")

    def test_spell_check_reports_suggestions_without_mutating_document(self) -> None:
        source = "Kế họach triển khai công việc"
        self.window._current_result = OCRResult(full_text=source)
        with patch("app.ui.main_window.QMessageBox.information") as message:
            self.window._on_check_spelling()
            worker = self.window._spell_worker
            self.assertIsNotNone(worker)
            worker.wait(3000)
            self.app.processEvents()

        self.assertEqual(self.window._current_document_text(), source)
        self.assertIn("1 vị trí", self.window.status_label.text())
        message.assert_called_once()
        self.assertIn("họach", message.call_args.args[2])


if __name__ == "__main__":
    unittest.main()
