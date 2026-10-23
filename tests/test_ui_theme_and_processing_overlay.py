from __future__ import annotations

import os
import time
import unittest
from threading import Event
from unittest.mock import patch

import numpy as np
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QEventLoop, QThread, QTimer, Qt
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QApplication, QToolBar
from pathlib import Path

from app.core.ocr_engine import OCRCancelledError, OCREngine
from app.ui.main_window import MainWindow, OCRWorker, _UiEventLoopLatencyProbe, build_stylesheet


class ThemeAndProcessingOverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([""])

    def test_light_and_dark_stylesheets_use_distinct_surface_tokens(self) -> None:
        dark = build_stylesheet("dark")
        light = build_stylesheet("light")
        self.assertIn("#0f0f14", dark)
        self.assertIn("#f7f7f8", light)
        self.assertNotIn("QFrame#previewStatePanel", dark)
        self.assertNotIn("QFrame#previewStatePanel", light)
        self.assertIn("QLabel#previewStateTitle", dark)
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

    def test_close_during_active_ocr_is_nonblocking_and_requests_cancel(self) -> None:
        class RunningWorker:
            def __init__(self) -> None:
                self._cancel_event = Event()

            def isRunning(self) -> bool:
                return True

            def request_cancel(self) -> None:
                self._cancel_event.set()

        window = MainWindow()
        worker = RunningWorker()
        window._ocr_worker = worker
        try:
            with patch.object(window, "_confirm_discard_unsaved_edits", return_value=True):
                started = time.perf_counter()
                window.close()
                elapsed = time.perf_counter() - started
            self.assertLess(elapsed, 0.1)
            self.assertTrue(worker._cancel_event.is_set())
            self.assertTrue(window._close_poll_timer.isActive())
            self.assertTrue(window._close_approved)
        finally:
            window._close_poll_timer.stop()
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
            self.assertIn("24/60 vùng · 40% giai đoạn nhận dạng", preview.state_message.text())
            preview.set_recognition_progress(30, 60)
            self.assertEqual(preview.circular_progress._value, 50)
            self.assertIn("30/60 vùng · 50% giai đoạn nhận dạng", preview.state_message.text())
            preview.set_recognition_progress(80, 60)
            self.assertEqual(preview.circular_progress._value, 100)
            self.assertIn("60/60 vùng · 100% giai đoạn nhận dạng", preview.state_message.text())
            preview.update_processing_message("Đang phân tích bố cục", reset_progress=True)
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

    def test_worker_progress_callbacks_are_delivered_on_gui_thread(self) -> None:
        class FakeEngine:
            def _ensure_loaded(self) -> None:
                pass

            def recognize(self, _image, **kwargs):
                kwargs["progress_callback"]("Đang phát hiện vùng chữ")
                kwargs["progress_detail_callback"](1, 2)
                return object()

        worker = OCRWorker(FakeEngine(), object())
        delivered_on_gui = []
        loop = QEventLoop()
        worker.progress.connect(
            lambda _message: delivered_on_gui.append(
                QThread.currentThread() == self.app.thread()
            )
        )
        worker.progress_detail.connect(
            lambda _done, _total: delivered_on_gui.append(
                QThread.currentThread() == self.app.thread()
            )
        )
        worker.finished.connect(loop.quit)
        worker.start()
        QTimer.singleShot(2000, loop.quit)
        loop.exec_()
        worker.wait(1000)
        self.assertTrue(delivered_on_gui)
        self.assertTrue(all(delivered_on_gui))

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

    def test_pulse_animation_changes_color_while_measured_progress_is_unchanged(self) -> None:
        window = MainWindow()
        try:
            window.show()
            preview = window.word_preview
            preview.set_processing_state("processing", "Đang nhận dạng văn bản")
            preview.set_recognition_progress(19, 46)
            ring = preview.circular_progress
            self.assertEqual(ring._value, 41)
            self.assertEqual(ring._pulse.state(), ring._pulse.Running)
            start_phase = ring._pulse_phase
            start_color = ring._pulse_color().rgba()
            start_hue = ring._pulse_color().hsvHue()
            loop = QEventLoop()
            QTimer.singleShot(180, loop.quit)
            loop.exec_()
            self.assertNotEqual(ring._pulse_phase, start_phase)
            self.assertNotEqual(ring._pulse_color().rgba(), start_color)
            self.assertEqual(ring._pulse_color().hsvHue(), start_hue)
            self.assertEqual(ring._value, 41)
        finally:
            window.close()

    def test_status_surface_is_transparent_and_not_text_selectable(self) -> None:
        window = MainWindow()
        try:
            panel = window.word_preview.state_panel
            self.assertFalse(panel.autoFillBackground())
            self.assertEqual(panel.frameShape(), panel.NoFrame)
            self.assertIn("background: transparent", panel.styleSheet())
            self.assertEqual(window.word_preview.state_title.textInteractionFlags(), Qt.NoTextInteraction)
            self.assertEqual(window.word_preview.state_message.textInteractionFlags(), Qt.NoTextInteraction)
        finally:
            window.close()

    def test_preview_success_animation_is_guarded_by_document_generation(self) -> None:
        window = MainWindow()
        try:
            preview = window.word_preview
            preview.set_perf_job_generation(19)
            perf_events = []
            preview.perf_stage_event.connect(lambda *event: perf_events.append(event))
            preview._pdf_path = Path("preview.pdf")
            preview._generation = 7
            image = QImage(12, 16, QImage.Format_RGB32)
            image.fill(Qt.white)
            preview._pages_ready(7, [(12.0, 16.0)], [(0, image)], 0)
            self.assertEqual(perf_events[-1][:3], (19, "preview_render", "end"))
            self.assertGreater(perf_events[-1][3], 2**31)
            self.assertEqual(preview._preview_state, "success_animating")
            self.assertIs(preview.content_stack.currentWidget(), preview.state_panel)
            self.assertTrue(preview.scroll.isHidden())
            self.assertEqual(preview.circular_progress._mode, "complete")
            self.assertIn("background: transparent", preview.state_panel.styleSheet())
            preview.set_theme("light")
            self.assertIn("color: #287a67", preview.state_message.styleSheet())
            token = preview._success_generation
            preview._success_hold.stop()
            preview._show_success_check(token)
            self.assertEqual(preview.circular_progress._mode, "success")
            self.assertEqual(preview.circular_progress._check_animation.state(), preview.circular_progress._check_animation.Running)
            self.assertTrue(preview.scroll.isHidden())
            self.assertIs(preview.content_stack.currentWidget(), preview.state_panel)
            self.assertEqual(len(preview._render_cache), 1)
            preview._success_finish.stop()
            preview._finish_success(token)
            self.assertEqual(preview._preview_state, "preview_ready")
            self.assertIs(preview.content_stack.currentWidget(), preview.scroll)
            self.assertTrue(preview.state_panel.isHidden())
            self.assertFalse(preview.scroll.isHidden())
            self.assertEqual(preview.circular_progress._check_animation.state(), preview.circular_progress._check_animation.Stopped)
            self.assertEqual(preview.circular_progress._mode, "success")
            self.assertEqual(preview.state_title.text(), "Đã nhận dạng thành công")
            self.assertIn("background: transparent", preview.state_message.styleSheet())
            self.assertNotIn("background-color", preview.state_panel.styleSheet())
            preview._pdf_path = Path("next-preview.pdf")
            preview._page_count = 1
            preview._preview_state = "preparing_preview"
            preview.show_success(1)
            stale_token = preview._success_generation
            preview.set_notice("Tài liệu mới")
            preview._show_success_check(stale_token)
            preview._finish_success(stale_token)
            self.assertIs(preview.content_stack.currentWidget(), preview.state_panel)
            self.assertEqual(preview._preview_state, "idle")
            self.assertFalse(preview.circular_progress.isVisible())
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
            self.assertEqual(preview.circular_progress._spin.state(), preview.circular_progress._spin.Stopped)
            self.assertNotEqual(preview._preview_state, "success")
            preview.set_processing_state("preparing_preview", "Đang chuẩn bị bản xem trước...")
            self.assertEqual(preview.circular_progress._mode, "indeterminate")
            self.assertEqual(preview.state_title.text(), "Đang chuẩn bị bản xem trước...")
            preview.set_processing_state("failed", "Không thể dựng Preview.")
            self.assertEqual(preview.circular_progress._spin.state(), preview.circular_progress._spin.Stopped)
        finally:
            window.close()

    def test_reduce_motion_setting_updates_and_persists_indicator_mode(self) -> None:
        window = MainWindow()
        try:
            window.show()
            preview = window.word_preview
            preview.set_processing_state("processing", "Đang nhận dạng")
            preview.set_recognition_progress(19, 46)
            ring = preview.circular_progress
            self.assertEqual(ring._pulse.state(), ring._pulse.Running)
            window.act_reduce_motion.setChecked(True)
            self.assertTrue(window.word_preview.circular_progress._reduce_motion)
            self.assertEqual(ring._pulse.state(), ring._pulse.Stopped)
            self.assertEqual(ring._value, 41)
            window.act_reduce_motion.setChecked(False)
            self.assertFalse(window.word_preview.circular_progress._reduce_motion)
            self.assertEqual(ring._pulse.state(), ring._pulse.Running)
        finally:
            window.close()

    def test_repeated_processing_state_does_not_restart_ring_mode(self) -> None:
        window = MainWindow()
        try:
            window.show()
            preview = window.word_preview
            ring = preview.circular_progress
            preview.set_processing_state("processing", "Đang nhận dạng")
            preview.set_recognition_progress(19, 46)
            with patch.object(ring, "set_indeterminate", wraps=ring.set_indeterminate) as reset:
                preview.set_processing_state("processing", "Thông báo tiến trình khác")
                self.assertEqual(reset.call_count, 0)
            self.assertEqual(ring._mode, "determinate")
            self.assertEqual(ring._value, 41)
            self.assertEqual(ring._pulse.state(), ring._pulse.Running)
            preview.update_processing_message("Cập nhật nhãn không đổi chế độ", cancellable=True)
            self.assertEqual(ring._mode, "determinate")
            self.assertEqual(ring._value, 41)
        finally:
            window.close()

    def test_set_indeterminate_is_idempotent_and_reduce_motion_keeps_progress(self) -> None:
        window = MainWindow()
        try:
            window.show()
            ring = window.word_preview.circular_progress
            window.word_preview.set_processing_state("processing", "Đang tải")
            self.app.processEvents()
            with patch.object(ring, "set_indeterminate", wraps=ring.set_indeterminate) as reset:
                window.word_preview.set_processing_state("processing", "Thông báo lặp lại")
                self.assertEqual(reset.call_count, 0)
            first_time = ring._spin.currentTime()
            with patch.object(ring, "_start_activity_animation", wraps=ring._start_activity_animation) as start:
                ring.set_indeterminate()
                self.assertEqual(start.call_count, 1)
            self.assertEqual(ring._mode, "indeterminate")
            self.assertEqual(ring._spin.state(), ring._spin.Running)
            self.assertGreaterEqual(ring._spin.currentTime(), first_time)
            ring.set_progress_percent(51)
            phase = ring._pulse_phase
            ring.set_progress_percent(51)
            self.assertEqual(ring._value, 51)
            self.assertEqual(ring._pulse.state(), ring._pulse.Running)
            ring.set_reduce_motion(True)
            self.assertEqual(ring._pulse.state(), ring._pulse.Stopped)
            self.assertEqual(ring._value, 51)
            self.assertEqual(ring._pulse_color().name(), ring._accent.name())
            self.assertGreaterEqual(phase, 0.0)
        finally:
            window.close()

    def test_starting_ocr_does_not_reconvert_already_visible_source(self) -> None:
        class FakeEngine:
            def _ensure_loaded(self):
                pass

            def recognize(self, _image, **_kwargs):
                return object()

        window = MainWindow()
        image = np.zeros((16, 24, 3), dtype="uint8")
        try:
            window._current_image = image
            window._show_image(image)
            with patch.object(window, "_confirm_discard_unsaved_edits", return_value=True), \
                    patch.object(window, "_get_engine", return_value=FakeEngine()), \
                    patch.object(window, "_show_image") as show_image, \
                    patch.object(OCRWorker, "start", autospec=True):
                window._on_run_ocr()
                show_image.assert_not_called()
            self.assertIsNotNone(window._current_pixmap)
            self.assertEqual(window.word_preview._preview_state, "processing")
        finally:
            window._ocr_worker = None
            window.close()

    def test_replacing_source_image_with_same_dimensions_refreshes_pixmap(self) -> None:
        window = MainWindow()
        first = np.zeros((20, 30, 3), dtype="uint8")
        second = np.full((20, 30, 3), 255, dtype="uint8")
        try:
            window.show()
            window._show_image(first)
            self.app.processEvents()
            first_key = window._current_pixmap.cacheKey()
            first_rendered_key = window._rendered_source_cache_key
            window._show_image(second)
            self.app.processEvents()
            self.assertNotEqual(window._current_pixmap.cacheKey(), first_key)
            self.assertEqual(window._rendered_source_cache_key, window._current_pixmap.cacheKey())
            self.assertNotEqual(window._rendered_source_cache_key, first_rendered_key)
        finally:
            window.close()

    def test_event_loop_latency_probe_is_opt_in(self) -> None:
        with patch.dict(os.environ, {"TDP_UI_PERF": "0", "TDP_OCR_PROCESS": "0"}):
            window = MainWindow()
            try:
                self.assertIsNone(window._ui_perf_probe)
                self.assertIsNone(window._ocr_process_client)
            finally:
                window.close()

    def test_process_backend_is_explicitly_opt_in(self) -> None:
        with patch.dict(os.environ, {"TDP_UI_PERF": "0", "TDP_OCR_PROCESS": "1"}):
            window = MainWindow()
            try:
                self.assertIsNotNone(window._ocr_process_client)
                self.assertFalse(window._ocr_process_client.is_running)
            finally:
                window.close()

    def test_event_loop_latency_probe_collects_stage_summary_without_stream_logging(self) -> None:
        with patch.dict(os.environ, {"TDP_UI_PERF": "1"}):
            window = MainWindow()
            try:
                probe = window._ui_perf_probe
                self.assertIsNotNone(probe)
                probe.start(17)
                loop = QEventLoop()
                QTimer.singleShot(90, lambda: probe.record_stage_event(
                    17, "detection", "start", time.perf_counter_ns()
                ))
                QTimer.singleShot(190, lambda: probe.record_stage_event(
                    17, "detection", "end", time.perf_counter_ns()
                ))
                QTimer.singleShot(270, loop.quit)
                loop.exec_()
                summary = probe.finish("test_completed")
                self.assertEqual(summary["job_generation"], 17)
                self.assertEqual(summary["reason"], "test_completed")
                self.assertGreaterEqual(summary["overall"]["n"], 3)
                self.assertIn("detection", summary["stages"])
                self.assertIn("p95_ms", summary["stages"]["detection"])
                self.assertIn("over_500ms", summary["overall"])
                self.assertEqual(summary["worker_stage_durations"][1]["stage"], "detection")
                self.assertIn("worker_signal_delivery", summary)
                self.assertEqual(summary["stage_transitions"][0]["stage"], "startup")
                self.assertFalse(probe._timer.isActive())
            finally:
                window.close()

    def test_recognition_progress_is_not_misclassified_as_detection(self) -> None:
        self.assertEqual(
            MainWindow._classify_ocr_progress_stage("Đã phát hiện 35 vùng; đang nhận dạng …"),
            "recognition",
        )
        self.assertEqual(
            MainWindow._classify_ocr_progress_stage("Đang phát hiện vùng chữ …"),
            "detection",
        )

    def test_probe_attributes_samples_using_worker_event_timestamps(self) -> None:
        window = MainWindow()
        try:
            probe = window._ui_perf_probe or _UiEventLoopLatencyProbe(window)
            probe.start(29)
            probe._timer.stop()
            base = probe._started_ns
            probe._samples = [
                (base + 10_000_000, 1.0),
                (base + 40_000_000, 2.0),
                (base + 120_000_000, 3.0),
            ]
            probe.record_stage_event(29, "recognition", "start", base + 30_000_000)
            probe.record_stage_event(29, "recognition", "end", base + 90_000_000)
            summary = probe.finish("test_completed")
            self.assertEqual(summary["stages"]["startup"]["n"], 1)
            self.assertEqual(summary["stages"]["recognition"]["n"], 1)
            self.assertEqual(summary["stages"]["between_stages"]["n"], 1)
            duration = next(x["duration_ms"] for x in summary["worker_stage_durations"]
                            if x["stage"] == "recognition")
            self.assertEqual(duration, 60.0)
        finally:
            window.close()

    def test_worker_forwards_actual_stage_callbacks_with_job_generation(self) -> None:
        class FakeEngine:
            def _ensure_loaded(self):
                pass

            def recognize(self, _image, **kwargs):
                callback = kwargs["stage_callback"]
                for stage in ("detection", "recognition", "layout"):
                    callback(stage, "start", time.perf_counter_ns())
                    callback(stage, "end", time.perf_counter_ns())
                return object()

        worker = OCRWorker(FakeEngine(), np.zeros((2, 2, 3), dtype="uint8"), generation=31)
        events = []
        worker.stage_event.connect(lambda *event: events.append(event))
        loop = QEventLoop()
        worker.result_ready.connect(lambda _result: loop.quit())
        timeout = QTimer()
        timeout.setSingleShot(True)
        timeout.timeout.connect(loop.quit)
        timeout.start(2000)
        worker.start()
        loop.exec_()
        timeout.stop()
        if worker.isRunning():
            worker.wait(1000)
        self.assertEqual([(event[1], event[2]) for event in events], [
            ("model_loading", "start"), ("model_loading", "end"),
            ("detection", "start"), ("detection", "end"),
            ("recognition", "start"), ("recognition", "end"),
            ("layout", "start"), ("layout", "end"),
        ])
        self.assertTrue(all(event[0] == 31 and event[3] > 0 for event in events))

    def test_opencv_thread_experiment_requires_debug_mode_and_explicit_count(self) -> None:
        worker = OCRWorker(object(), np.zeros((1, 1, 3), dtype="uint8"))
        with patch.dict(os.environ, {"TDP_UI_PERF": "1", "TDP_OPENCV_THREADS": "4"}), \
                patch("app.ui.main_window._DEBUG_OPENCV_THREADS", "4"), \
                patch.object(OCRWorker, "_opencv_debug_threads_applied", None), \
                patch("app.ui.main_window.cv2.getNumThreads", return_value=20), \
                patch("app.ui.main_window.cv2.setNumThreads") as set_threads:
            worker._apply_debug_opencv_thread_override()
            set_threads.assert_called_once_with(4)
        with patch.dict(os.environ, {"TDP_UI_PERF": "0", "TDP_OPENCV_THREADS": "2"}), \
                patch("app.ui.main_window._DEBUG_OPENCV_THREADS", "2"), \
                patch("app.ui.main_window.cv2.setNumThreads") as set_threads:
            worker._apply_debug_opencv_thread_override()
            set_threads.assert_not_called()


if __name__ == "__main__":
    unittest.main()
