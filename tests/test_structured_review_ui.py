from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch

import numpy as np
from PyQt5.QtWidgets import QApplication, QMessageBox

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app.core.models import DocumentBlock, OCRBox, OCRResult
from app.core.structured_editing import StructuredDocumentEditor
from app.ui.main_window import MainWindow


class StructuredReviewUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([""])

    def setUp(self) -> None:
        self.window = MainWindow()
        candidate_box = OCRBox(
            "primary", 0.8, [[5, 5], [55, 5], [55, 25], [5, 25]], raw_text="raw primary",
            metadata={"needs_review": True, "candidates": [
                {"recognizer": "vietocr", "text": "primary"},
                {"recognizer": "paddleocr", "text": "alternative",
                 "mapping": {"status": "matched"}},
            ]},
        )
        result = OCRResult(
            boxes=[candidate_box], full_text="primary",
            document_blocks=[DocumentBlock("paragraph", 0, "primary", "raw primary", [0])],
        )
        self.window._current_image = np.full((40, 70, 3), 255, dtype=np.uint8)
        self.window._current_result = result
        self.window._structured_editor = StructuredDocumentEditor(result)
        self.window._refresh_ocr_review_list(selected_box=0)

    def tearDown(self) -> None:
        self.window._structured_saved = True
        self.window.close()

    def test_candidate_requires_confirmation_and_keeps_structured_result(self) -> None:
        with patch.object(self.window, "_create_document_session"):
            self.window.ocr_review_candidates.setCurrentIndex(2)
            self.assertEqual(self.window.ocr_review_edit.text(), "alternative")
            self.assertEqual(self.window._current_result.full_text, "primary")
            self.window._confirm_review_edit()
        self.assertEqual(self.window._current_result.full_text, "alternative")
        self.assertEqual(self.window._current_result.document_blocks[0].text, "alternative")
        self.assertEqual(self.window._current_result.boxes[0].raw_text, "raw primary")
        self.assertIn("✓", self.window.ocr_review_list.item(0).text())

    def test_manual_edit_can_be_undone_and_redone_without_flattening(self) -> None:
        with patch.object(self.window, "_create_document_session"):
            self.window.ocr_review_edit.setText("confirmed manual text")
            self.window._confirm_review_edit()
            self.assertEqual(self.window._current_result.document_blocks[0].text, "confirmed manual text")
            self.window._undo_structured_edit()
            self.assertEqual(self.window._current_result.full_text, "primary")
            self.window._redo_structured_edit()
        self.assertEqual(self.window._current_result.full_text, "confirmed manual text")
        self.assertEqual(len(self.window._current_result.document_blocks), 1)

    def test_unsaved_structured_changes_require_explicit_discard_confirmation(self) -> None:
        self.window._structured_editor.apply_box_correction(0, "unsaved")
        self.window._structured_saved = False
        with patch("app.ui.main_window.QMessageBox.question", return_value=QMessageBox.No) as ask:
            self.assertFalse(self.window._confirm_discard_unsaved_edits())
        ask.assert_called_once()


if __name__ == "__main__":
    unittest.main()
