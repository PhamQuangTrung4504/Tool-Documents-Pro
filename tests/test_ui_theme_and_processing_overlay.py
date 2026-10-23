from __future__ import annotations

import os
import unittest
from threading import Event
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QEventLoop, QTimer
from PyQt5.QtWidgets import QApplication, QToolBar
from pathlib import Path

from app.core.ocr_engine import OCRCancelledError, OCREngine
from app.ui.main_window import MainWindow, OCRWorker, build_stylesheet


class ThemeAndProcessingOverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([""])

    def test_light_and_dark_stylesheets_use_distinct_surface_tokens(self) -> None:
        dark = build_stylesheet("dark")
        light = build_stylesheet("light")
        self.assertIn("#0f0f14", dark)
        self.assertIn("#f7f7f8", light)
        self.assertIn("QFrame#previewStatePanel", dark)
        self.assertIn("QFrame#previewStatePanel", light)
        self.assertIn("QPushButton#primaryButton", dark)

    def test_processing_state_is_inside_word_preview_and_source_remains_visible(self) -> None:
        window = MainWindow()
        try:
            window.show()
            self.app.processEvents()
            window._set_preview_processing_state("processing", "Đang nhận dạng …")
            self.assertTrue(window.processing_overlay.isVisible())
            self.assertIs(window.processing_overlay.parentWidget(), window.word_preview.content_stack)
            self.assertTrue(window.scroll_image.isVisible())
            self.assertIs(window.processing_indicator.parentWidget(), window.processing_overlay)
            self.assertTrue(window.processing_indicator.isVisible())
            self.assertEqual(window.processing_indicator._mode, "indeterminate")
            self.assertFalse(window.processing_retry.isVisible())
            self.assertEqual(window.processing_message.text(), "Đang nhận dạng …")

            window._set_preview_processing_state("failed", "Không thể xử lý.")
            self.assertTrue(window.processing_overlay.isVisible())
            self.assertFalse(window.processing_indicator.isVisible())
            self.assertTrue(window.processing_retry.isVisible())

            window.word_preview._page_count = 1
            window._set_preview_processing_state("completed")
            self.assertFalse(window.processing_overlay.isVisible())
        finally:
            window.close()

    def test_finished_stale_worker_reenables_controls_without_using_its_result(self) -> None:
        window = MainWindow()
        try:
            worker = object()
            window._ocr_worker = worker
            window.btn_ocr.setEnabled(False)
            window.act_ocr.setEnabled(False)
            window._on_ocr_thread_stopped(worker)
            self.assertIsNone(window._ocr_worker)
            self.assertTrue(window.btn_ocr.isEnabled())
            self.assertTrue(window.act_ocr.isEnabled())
        finally:
            window.close()

    def test_cancel_event_stops_before_model_loading(self) -> None:
        engine = OCREngine()
        cancel_event = Event()
        cancel_event.set()
        with patch.object(engine, "_ensure_loaded") as load:
            with self.assertRaises(OCRCancelledError):
                engine.recognize(object(), cancel_event=cancel_event)
        load.assert_not_called()

    def test_cancel_button_requests_cooperative_worker_stop(self) -> None:
        class FakeWorker:
            def __init__(self) -> None:
                self._cancel_event = Event()

            def isRunning(self) -> bool:
                return True

            def request_cancel(self) -> None:
                self._cancel_event.set()

        window = MainWindow()
        try:
            worker = FakeWorker()
            window._ocr_worker = worker
            window._set_preview_processing_state("processing", "OCR", cancellable=True)
            window._request_ocr_cancel()
            self.assertTrue(worker._cancel_event.is_set())
            self.assertFalse(window.processing_cancel.isEnabled())
            self.assertEqual(window.word_preview._preview_state, "cancelling")
            self.assertIn("Đang hủy", window.processing_message.text())
        finally:
            window._ocr_worker = None
            window.close()

    def test_main_shell_has_one_primary_action_row_and_word_tabs(self) -> None:
        window = MainWindow()
        try:
            self.assertEqual(window.document_tabs.count(), 2)
            self.assertEqual(window.document_tabs.widget(0), window.word_preview)
            self.assertEqual(window.document_tabs.tabText(0), "Xem trước Word")
            self.assertEqual(window.document_tabs.tabText(1), "Rà soát OCR")
            self.assertFalse(any("Chỉnh sửa toàn văn" in window.document_tabs.tabText(i)
                                 for i in range(window.document_tabs.count())))
            self.assertEqual(window.btn_ocr.objectName(), "primaryButton")
            self.assertEqual(window.btn_export_docx.objectName(), "secondaryButton")
            self.assertTrue(window.utility_menu.actions())
            self.assertFalse(hasattr(window, "image_preview_stack"))
            self.assertIsInstance(window.btn_ocr.parentWidget(), QToolBar)
            window.document_tabs.setCurrentIndex(1)
            self.assertEqual(window.document_tabs.currentWidget(), window.document_tabs.widget(1))
            window.document_tabs.setCurrentIndex(0)
            window.show()
            for width, height in ((1366, 768), (1600, 900), (1920, 1080)):
                window.resize(width, height)
                self.app.processEvents()
                self.assertGreaterEqual(window.workspace_splitter.widget(0).width(), 340)
                self.assertGreaterEqual(window.workspace_splitter.widget(1).width(), 500)
        finally:
            window.close()

    def test_circular_progress_is_monotonic_and_caps_before_completion(self) -> None:
        window = MainWindow()
        try:
            ring = window.word_preview.circular_progress
            ring.set_indeterminate()
            self.assertEqual(ring._mode, "indeterminate")
            ring.set_work_progress(68, 100)
            ring.set_work_progress(34, 100)
            self.assertEqual(ring._value, 68)
            ring.set_work_progress(100, 100)
            self.assertEqual(ring._value, 100)
            ring.set_work_progress(5, 0)
            self.assertEqual(ring._mode, "indeterminate")
            ring.show_success()
            self.assertEqual(ring._value, 100)
            self.assertEqual(ring._mode, "complete")
        finally:
            window.close()

    def test_recognition_progress_is_labeled_as_stage_and_uses_real_counts(self) -> None:
        window = MainWindow()
        try:
            preview = window.word_preview
            preview.set_processing_state("processing", "Đang phát hiện vùng chữ")
            self.assertEqual(preview.circular_progress._mode, "indeterminate")
            preview.set_recognition_progress(24, 0)
            self.assertEqual(preview.circular_progress._mode, "indeterminate")
            self.assertNotIn("%", preview.state_message.text())
            preview.set_recognition_progress(24, 60)
            self.assertEqual(preview.circular_progress._value, 40)
            self.assertIn("Giai đoạn nhận dạng: 24/60 vùng · 40%", preview.state_message.text())
            preview.set_recognition_progress(80, 60)
            self.assertEqual(preview.circular_progress._value, 100)
            self.assertIn("60/60 vùng · 100%", preview.state_message.text())
            preview.set_processing_state("processing", "Đang phân tích bố cục")
            self.assertEqual(preview.circular_progress._mode, "indeterminate")
        finally:
            window.close()

    def test_worker_forwards_numeric_progress_callback_without_model_setup(self) -> None:
        class FakeEngine:
            def _ensure_loaded(self) -> None:
                pass

            def recognize(self, _image, **kwargs):
                callback = kwargs["progress_detail_callback"]
                callback(1, 4)
                callback(3, 4)
                return object()

        worker = OCRWorker(FakeEngine(), object())
        seen = []
        worker.progress_detail.connect(lambda done, total: seen.append((done, total)))
        worker.run()
        self.assertEqual(seen, [(1, 4), (3, 4)])

    def test_new_job_ignores_stale_numeric_progress(self) -> None:
        class FakeWorker:
            def __init__(self):
                self._cancel_event = Event()

            def isRunning(self):
                return True

        window = MainWindow()
        try:
            current = FakeWorker()
            stale = FakeWorker()
            window._ocr_worker = current
            generation = window._generation
            window._set_preview_processing_state("processing", "Đang nhận dạng")
            window._on_ocr_progress_detail(24, 60, generation - 1, current)
            window._on_ocr_progress_detail(24, 60, generation, stale)
            self.assertEqual(window.word_preview.circular_progress._mode, "indeterminate")
            window._on_ocr_progress_detail(24, 60, generation, current)
            self.assertEqual(window.word_preview.circular_progress._value, 40)
        finally:
            window._ocr_worker = None
            window.close()

    def test_preview_success_animation_is_guarded_by_document_generation(self) -> None:
        window = MainWindow()
        try:
            preview = window.word_preview
            preview._pdf_path = Path("preview.pdf")
            preview._page_count = 2
            preview.show_success(2)
            self.assertEqual(preview.circular_progress._mode, "complete")
            self.assertIn("background: transparent", preview.state_panel.styleSheet())
            preview.set_theme("light")
            self.assertIn("color: #287a67", preview.state_message.styleSheet())
            loop = QEventLoop()
            QTimer.singleShot(900, loop.quit)
            loop.exec_()
            self.assertEqual(preview.circular_progress._mode, "success")
            self.assertEqual(preview.state_title.text(), "Đã nhận dạng thành công")
            self.assertIn("background: transparent", preview.state_message.styleSheet())
            self.assertNotIn("background-color", preview.state_panel.styleSheet())
            self.assertIs(preview.content_stack.currentWidget(), preview.scroll)

            preview.show_success(2)
            stale_token = preview._success_generation
            preview.set_notice("Tài liệu mới")
            preview._finish_success(stale_token)
            self.assertIs(preview.content_stack.currentWidget(), preview.state_panel)
            self.assertNotEqual(preview._preview_state, "completed")
        finally:
            window.close()

    def test_failure_and_cancel_states_never_show_success(self) -> None:
        window = MainWindow()
        try:
            preview = window.word_preview
            preview.set_processing_state("failed", "Không dựng được preview.")
            self.assertNotEqual(preview.circular_progress._mode, "success")
            self.assertEqual(preview.state_title.text(), "Không thể hoàn tất")
            preview.set_processing_state("cancelling", "Đang hủy...")
            self.assertEqual(preview.state_title.text(), "Đang hủy nhận dạng")
            preview.set_processing_state("cancelled", "Đã hủy nhận dạng.")
            self.assertEqual(preview.state_title.text(), "Đã hủy nhận dạng")
            self.assertNotEqual(preview._preview_state, "success")
        finally:
            window.close()

    def test_reduce_motion_setting_updates_and_persists_indicator_mode(self) -> None:
        window = MainWindow()
        try:
            window.act_reduce_motion.setChecked(True)
            self.assertTrue(window.word_preview.circular_progress._reduce_motion)
            window.act_reduce_motion.setChecked(False)
            self.assertFalse(window.word_preview.circular_progress._reduce_motion)
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
