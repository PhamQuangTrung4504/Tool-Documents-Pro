"""Lazy, embedded multi-page PDF preview of the generated Word document."""
from __future__ import annotations

import importlib
import time
from collections import OrderedDict
from pathlib import Path

from PyQt5.QtCore import QAbstractAnimation, QEasingCurve, Qt, QThread, QTimer, QVariantAnimation, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QStackedWidget, QVBoxLayout, QWidget


class CircularProgressIndicator(QWidget):
    """Vòng tiến độ quay vô định hoặc hiển thị phần trăm đã xác nhận."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedSize(132, 132)
        self._mode = "indeterminate"
        self._value = 0
        self._angle = 0.0
        self._pulse_phase = 0.0
        self._check_progress = 0.0
        self._accent = QColor("#6c5ce7")
        self._success = QColor("#00b894")
        self._track = QColor(128, 128, 140, 55)
        self._reduce_motion = False
        self._spin = QVariantAnimation(self)
        self._spin.setStartValue(0.0)
        self._spin.setEndValue(360.0)
        self._spin.setDuration(1250)
        self._spin.setLoopCount(-1)
        self._spin.valueChanged.connect(self._set_angle)
        self._pulse = QVariantAnimation(self)
        self._pulse.setStartValue(0.0)
        self._pulse.setKeyValueAt(0.5, 1.0)
        self._pulse.setEndValue(0.0)
        self._pulse.setDuration(1600)
        self._pulse.setLoopCount(-1)
        self._pulse.setEasingCurve(QEasingCurve.InOutSine)
        self._pulse.valueChanged.connect(self._set_pulse_phase)
        self._check_animation = QVariantAnimation(self)
        self._check_animation.setStartValue(0.0)
        self._check_animation.setEndValue(1.0)
        self._check_animation.setDuration(420)
        self._check_animation.valueChanged.connect(self._set_check_progress)

    def _set_angle(self, value) -> None:
        self._angle = float(value)
        self.update()

    def _set_check_progress(self, value) -> None:
        self._check_progress = float(value)
        self.update()

    def _set_pulse_phase(self, value) -> None:
        previous_alpha = self._pulse_color().alpha()
        self._pulse_phase = float(value)
        if self._pulse_color().alpha() != previous_alpha:
            self.update()

    def set_theme(self, theme: str) -> None:
        self._accent = QColor("#63c9bd" if theme == "dark" else "#348f86")
        self._success = QColor("#42d3a6" if theme == "dark" else "#008f70")
        self._track = QColor(255, 255, 255, 38) if theme == "dark" else QColor(24, 24, 32, 32)
        self.update()

    def set_reduce_motion(self, enabled: bool) -> None:
        self._reduce_motion = bool(enabled)
        if enabled:
            self._spin.stop()
            self._pulse.stop()
        elif self.isVisible():
            self._start_activity_animation()

    def _start_activity_animation(self) -> None:
        if self._reduce_motion or not self.isVisible():
            return
        if self._mode == "indeterminate":
            if self._spin.state() == QAbstractAnimation.Paused:
                self._spin.resume()
            elif self._spin.state() != QAbstractAnimation.Running:
                self._spin.start()
        if self._mode in {"indeterminate", "determinate"}:
            if self._pulse.state() == QAbstractAnimation.Paused:
                self._pulse.resume()
            elif self._pulse.state() != QAbstractAnimation.Running:
                self._pulse.start()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._start_activity_animation()

    def hideEvent(self, event) -> None:
        for animation in (self._spin, self._pulse):
            if animation.state() == QAbstractAnimation.Running:
                animation.pause()
        super().hideEvent(event)

    def set_indeterminate(self) -> None:
        if self._mode == "indeterminate":
            self._start_activity_animation()
            return
        self._mode = "indeterminate"
        self._value = 0
        self._check_progress = 0.0
        self._pulse.stop()
        self._spin.stop()
        self._start_activity_animation()
        self._check_animation.stop()
        self.update()

    def stop_animation(self) -> None:
        self._spin.stop()
        self._pulse.stop()
        self._check_animation.stop()

    def set_progress_percent(self, value: int) -> None:
        """Nhận phần trăm từ bộ đếm có tổng; giữ đơn điệu trong cùng giai đoạn."""
        mode_changed = self._mode != "determinate"
        self._mode = "determinate"
        self._value = max(self._value, min(100, max(0, int(value))))
        if mode_changed:
            self._spin.stop()
        self._start_activity_animation()
        self.update()

    def set_work_progress(self, completed: int, total: int) -> None:
        """Dùng bộ đếm có tổng đã biết; nếu tổng không hợp lệ thì quay vô định."""
        if total <= 0:
            self.set_indeterminate()
            return
        self.set_progress_percent(int(max(0, completed) * 100 / total))

    def show_success(self) -> None:
        self._mode = "complete"
        self._value = 100
        self._check_progress = 0.0
        self._spin.stop()
        self._pulse.stop()
        self.update()

    def animate_check(self) -> None:
        self._mode = "success"
        self._spin.stop()
        self._pulse.stop()
        self._check_progress = 1.0 if self._reduce_motion else 0.0
        if not self._reduce_motion:
            self._check_animation.start()
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect().adjusted(13, 13, -13, -13)
        pen = QPen(self._track, 8, Qt.SolidLine, Qt.RoundCap)
        painter.setPen(pen)
        painter.drawArc(rect, 0, 360 * 16)

        if self._mode == "indeterminate":
            pen.setColor(self._pulse_color())
            pen.setWidth(8)
            painter.setPen(pen)
            painter.drawArc(rect, int((90 - self._angle) * 16), 98 * 16)
            return

        color = self._success if self._mode == "success" else self._pulse_color()
        text_color = self._success if self._mode == "success" else self._accent
        pen.setColor(color)
        painter.setPen(pen)
        painter.drawArc(rect, 90 * 16, -int(360 * 16 * self._value / 100))
        painter.setPen(color)
        if self._mode == "success":
            first = (self.width() * 0.32, self.height() * 0.52)
            joint = (self.width() * 0.45, self.height() * 0.65)
            last = (self.width() * 0.70, self.height() * 0.39)
            import math
            lengths = (math.dist(first, joint), math.dist(joint, last))
            remaining = sum(lengths) * self._check_progress
            painter.setPen(QPen(color, 7, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            if remaining > 0:
                fraction = min(1.0, remaining / lengths[0])
                painter.drawLine(int(first[0]), int(first[1]),
                                 int(first[0] + (joint[0] - first[0]) * fraction),
                                 int(first[1] + (joint[1] - first[1]) * fraction))
            if remaining > lengths[0]:
                fraction = min(1.0, (remaining - lengths[0]) / lengths[1])
                painter.drawLine(int(joint[0]), int(joint[1]),
                                 int(joint[0] + (last[0] - joint[0]) * fraction),
                                 int(joint[1] + (last[1] - joint[1]) * fraction))
        elif self._mode == "complete":
            painter.setPen(text_color)
            font = painter.font()
            font.setPointSize(20)
            font.setWeight(600)
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignCenter, "100%")
        else:
            painter.setPen(text_color)
            font = painter.font()
            font.setPointSize(20)
            font.setWeight(600)
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignCenter, f"{self._value}%")

    def _pulse_color(self) -> QColor:
        """Biến thiên alpha nhẹ; không đổi sắc độ, văn bản hay độ dài cung."""
        if self._reduce_motion:
            return QColor(self._accent)
        color = QColor(self._accent)
        color.setAlpha(232 + int(23 * self._pulse_phase))
        return color


class _PdfPageWorker(QThread):
    pages_ready = pyqtSignal(int, object, object, object)
    failed = pyqtSignal(int, str)
    stage_event = pyqtSignal(str, str, object)

    def __init__(self, path: Path, center: int, zoom: float, generation: int, parent=None):
        super().__init__(parent)
        self.path = path
        self.center = center
        self.zoom = zoom
        self.generation = generation

    def run(self) -> None:
        self.stage_event.emit("preview_render", "start", time.perf_counter_ns())
        try:
            try:
                pymupdf = importlib.import_module("pymupdf")
            except ImportError:
                self.failed.emit(self.generation, "PyMuPDF is unavailable in the active Python environment.")
                return
            with pymupdf.open(self.path) as document:
                if document.page_count == 0:
                    raise ValueError("The PDF contains no pages.")
                page_sizes = [(float(page.rect.width), float(page.rect.height)) for page in document]
                indexes = range(max(0, self.center - 1), min(document.page_count, self.center + 2))
                images = []
                matrix = pymupdf.Matrix(self.zoom * 1.35, self.zoom * 1.35)
                for index in indexes:
                    pix = document.load_page(index).get_pixmap(matrix=matrix, alpha=False)
                    image = QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format_RGB888).copy()
                    images.append((index, image))
            self.pages_ready.emit(self.generation, page_sizes, images, self.center)
        except Exception as exc:
            # Exception details from PDF libraries may include a filename; keep the
            # visible/log-safe message to the error class and a generic description.
            self.failed.emit(self.generation, f"PyMuPDF could not open or rasterize the preview PDF ({type(exc).__name__}).")


class WordPreviewWidget(QWidget):
    preview_ready = pyqtSignal(int)
    preview_failed = pyqtSignal(str)
    perf_stage_event = pyqtSignal(int, str, str, object)
    cancel_requested = pyqtSignal()
    retry_requested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._pdf_path: Path | None = None
        self._zoom = 1.0
        self._page_count = 0
        self._page_sizes: list[tuple[float, float]] = []
        self._page_labels: list[QLabel] = []
        self._render_cache: OrderedDict[int, QPixmap] = OrderedDict()
        self._render_worker: _PdfPageWorker | None = None
        self._generation = 0
        self._initial_preview_emitted = False
        self._success_generation = 0
        self._requested_center = 0
        self._theme = "dark"
        self._preview_state = "idle"
        self._perf_job_generation = 0

        outer = QVBoxLayout(self)
        controls = QHBoxLayout()
        self.page_label = QLabel("Trang — / —")
        controls.addWidget(self.page_label)
        controls.addStretch(1)
        self.zoom_out = QPushButton("−")
        self.zoom_in = QPushButton("+")
        self.fit_width = QPushButton("Vừa chiều rộng")
        self.zoom_out.setEnabled(False)
        self.zoom_in.setEnabled(False)
        self.fit_width.setEnabled(False)
        controls.addWidget(self.zoom_out)
        controls.addWidget(self.zoom_in)
        controls.addWidget(self.fit_width)
        outer.addLayout(controls)

        self.content_stack = QStackedWidget()
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setObjectName("wordPageCanvas")
        self.pages = QWidget()
        self.page_layout = QVBoxLayout(self.pages)
        self.page_layout.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        self.scroll.setWidget(self.pages)
        self.content_stack.addWidget(self.scroll)

        self.state_panel = QFrame()
        self.state_panel.setObjectName("previewStatePanel")
        self.state_panel.setFrameShape(QFrame.NoFrame)
        self.state_panel.setAutoFillBackground(False)
        self.state_panel.setStyleSheet(
            "QFrame#previewStatePanel { background: transparent; border: none; }"
        )
        state_layout = QVBoxLayout(self.state_panel)
        state_layout.setContentsMargins(36, 28, 36, 28)
        state_layout.addStretch(1)
        self.circular_progress = CircularProgressIndicator()
        state_layout.addWidget(self.circular_progress, alignment=Qt.AlignHCenter)
        self.state_title = QLabel("Bản xem trước Word")
        self.state_title.setObjectName("previewStateTitle")
        self.state_title.setAlignment(Qt.AlignCenter)
        self.state_title.setTextInteractionFlags(Qt.NoTextInteraction)
        self.state_title.setAutoFillBackground(False)
        state_layout.addWidget(self.state_title)
        self.state_message = QLabel("Chạy OCR để tạo tài liệu Word có thể chỉnh sửa.")
        self.state_message.setObjectName("previewStateMessage")
        self.state_message.setAlignment(Qt.AlignCenter)
        self.state_message.setTextInteractionFlags(Qt.NoTextInteraction)
        self.state_message.setAutoFillBackground(False)
        self.state_message.setWordWrap(True)
        self.state_message.setMaximumWidth(520)
        state_layout.addWidget(self.state_message, alignment=Qt.AlignHCenter)
        state_actions = QHBoxLayout()
        state_actions.addStretch(1)
        self.state_cancel = QPushButton("Hủy tác vụ")
        self.state_cancel.setObjectName("secondaryButton")
        self.state_cancel.clicked.connect(self.cancel_requested.emit)
        state_actions.addWidget(self.state_cancel)
        self.state_retry = QPushButton("Thử lại")
        self.state_retry.setObjectName("secondaryButton")
        self.state_retry.clicked.connect(self.retry_requested.emit)
        state_actions.addWidget(self.state_retry)
        state_actions.addStretch(1)
        state_layout.addLayout(state_actions)
        state_layout.addStretch(1)
        self.content_stack.addWidget(self.state_panel)
        outer.addWidget(self.content_stack, 1)
        self.notice = QLabel("Chưa có tài liệu. Chạy OCR để tạo bản xem trước Word.")
        self.notice.setAlignment(Qt.AlignCenter)
        outer.addWidget(self.notice)

        self.zoom_in.clicked.connect(lambda: self.set_zoom(self._zoom * 1.15))
        self.zoom_out.clicked.connect(lambda: self.set_zoom(self._zoom / 1.15))
        self.fit_width.clicked.connect(self.fit_to_width)
        self.scroll.verticalScrollBar().valueChanged.connect(self._update_page_indicator)
        self.set_notice("Chưa có tài liệu. Chạy OCR để tạo bản xem trước Word.")
        self.set_processing_state("idle", "Chạy OCR để tạo tài liệu Word có thể chỉnh sửa.")

    def set_theme(self, theme: str) -> None:
        """Đổi nền vùng xem quanh trang; ảnh trang vẫn giữ nguyên màu gốc."""
        self._theme = theme if theme in {"light", "dark"} else "dark"
        self.circular_progress.set_theme(self._theme)
        if hasattr(self, "_preview_state"):
            self._set_success_appearance(self._preview_state == "success_animating")
        background = "#e9eaec" if self._theme == "light" else "#292933"
        border = "#c8cbd0" if self._theme == "light" else "#555"
        self.scroll.setStyleSheet(f"QScrollArea#wordPageCanvas {{ background: {background}; border: 0; }}")
        for label in self._page_labels:
            label.setStyleSheet(f"background: white; border: 1px solid {border};")

    def set_perf_job_generation(self, generation: int) -> None:
        self._perf_job_generation = int(generation)

    def set_notice(self, message: str) -> None:
        self.cancel_success_animation()
        self._generation += 1
        self._pdf_path = None
        self._page_count = 0
        self._page_sizes = []
        self._clear_pages()
        self._page_labels = []
        self._render_cache.clear()
        self.fit_width.setEnabled(False)
        self.notice.setText(message)
        self.notice.hide()
        self.set_processing_state("empty", message)

    def set_pdf(self, path: Path) -> None:
        self.cancel_success_animation()
        self._generation += 1
        self._initial_preview_emitted = False
        self._pdf_path = path
        self._page_count = 0
        self._page_sizes = []
        self._render_cache.clear()
        self._clear_pages()
        self._page_labels = []
        self._requested_center = 0
        self.notice.hide()
        self.set_processing_state("preparing_preview", "Đang chuẩn bị bản xem trước …")
        self._request_pages(0)

    def set_processing_state(self, state: str, message: str = "", cancellable: bool = False) -> None:
        """Hiển thị trạng thái của tài liệu ngay trong khung Word Preview."""
        state = {"empty": "idle", "success": "success_animating", "completed": "preview_ready"}.get(state, state)
        previous_state = self._preview_state
        if state != "success_animating" and previous_state == "success_animating":
            self.cancel_success_animation()
        self._set_success_appearance(False)
        self._preview_state = state
        titles = {
            "idle": "Bản xem trước Word",
            "processing": "Đang xử lý tài liệu",
            "preparing_preview": "Đang chuẩn bị bản xem trước...",
            "cancelling": "Đang hủy nhận dạng",
            "failed": "Không thể hoàn tất",
            "cancelled": "Đã hủy nhận dạng",
            "preview_ready": "Bản xem trước Word",
        }
        self.state_title.setText(titles.get(state, "Bản xem trước Word"))
        self.state_message.setText(message)
        busy = state in {"processing", "preparing_preview", "cancelling"}
        self.circular_progress.setVisible(busy)
        if busy and (state != previous_state or self.circular_progress._mode not in {"indeterminate", "determinate"}):
            self.circular_progress.set_indeterminate()
        elif not busy:
            self.circular_progress.stop_animation()
        self.state_cancel.setVisible(state == "processing" and cancellable)
        self.state_cancel.setEnabled(state == "processing" and cancellable)
        self.state_retry.setVisible(state in {"failed", "cancelled"})
        preview_ready = state == "preview_ready" and self._page_count > 0
        self.zoom_in.setEnabled(preview_ready)
        self.zoom_out.setEnabled(preview_ready)
        self.fit_width.setEnabled(preview_ready)
        self.content_stack.setCurrentWidget(
            self.scroll if preview_ready else self.state_panel
        )
        self.notice.hide()

    def update_processing_message(
        self, message: str, cancellable: bool = False, reset_progress: bool = False,
        title: str | None = None,
    ) -> None:
        """Cập nhật nhãn; chế độ vòng tròn chỉ đổi ở ranh giới giai đoạn."""
        if self._preview_state not in {"processing", "preparing_preview", "cancelling"}:
            return
        if title:
            self.state_title.setText(title)
        self.state_message.setText(message)
        if reset_progress and self.circular_progress._mode != "indeterminate":
            self.circular_progress.set_indeterminate()
        self.state_cancel.setVisible(self._preview_state == "processing" and cancellable)
        self.state_cancel.setEnabled(self._preview_state == "processing" and cancellable)

    def set_recognition_progress(
        self, completed: int, total: int, cancellable: bool = True,
    ) -> None:
        """Hiển thị tỷ lệ có mẫu số thật, ghi rõ đây là giai đoạn nhận dạng."""
        if self._preview_state != "processing" or total <= 0:
            return
        safe_total = max(1, int(total))
        safe_completed = min(safe_total, max(0, int(completed)))
        percent = int(safe_completed * 100 / safe_total)
        self.state_title.setText("Đang nhận dạng văn bản")
        self.state_message.setText(
            f"{safe_completed}/{safe_total} vùng · {percent}% giai đoạn nhận dạng"
        )
        self.circular_progress.setVisible(True)
        self.circular_progress.set_work_progress(safe_completed, safe_total)
        self.state_cancel.setVisible(cancellable)
        self.state_cancel.setEnabled(cancellable)

    def _set_success_appearance(self, active: bool) -> None:
        if active:
            self.state_title.setStyleSheet(
                f"background: transparent; color: {'#f7f7f8' if self._theme == 'dark' else '#202123'}; "
                "font-size: 17px; font-weight: 500;"
            )
            color = "#42d3a6" if self._theme == "dark" else "#287a67"
            self.state_message.setStyleSheet(
                f"background: transparent; color: {color}; font-size: 14px;"
            )
        else:
            self.state_title.setStyleSheet("")
            self.state_message.setStyleSheet("")

    def cancel_success_animation(self) -> None:
        self._success_generation += 1
        if hasattr(self, "_success_hold"):
            self._success_hold.stop()
        if hasattr(self, "_success_finish"):
            self._success_finish.stop()
        if hasattr(self, "circular_progress"):
            self.circular_progress.stop_animation()

    def show_success(self, page_count: int) -> None:
        """Chỉ thành công sau khi trang đầu đã được dựng nhưng vẫn đang ẩn."""
        if self._pdf_path is None or self._page_count <= 0 or page_count <= 0:
            return
        if self._preview_state in {"success_animating", "preview_ready"}:
            return
        self.cancel_success_animation()
        self._set_success_appearance(True)
        token = self._success_generation
        self._preview_state = "success_animating"
        self.state_title.setText("Đang hoàn tất")
        self.state_message.setText(f"Bản xem trước Word đã sẵn sàng · {page_count} trang")
        self.content_stack.setCurrentWidget(self.state_panel)
        self.circular_progress.setVisible(True)
        self.circular_progress.show_success()
        if self.circular_progress._reduce_motion:
            self._show_success_check(token)
            return
        self.state_cancel.hide()
        self.state_retry.hide()
        self._success_hold = QTimer(self)
        self._success_hold.setSingleShot(True)
        self._success_hold.timeout.connect(lambda: self._show_success_check(token))
        self._success_hold.start(160)

    def _show_success_check(self, token: int) -> None:
        if (token != self._success_generation or self._pdf_path is None
                or self._preview_state != "success_animating"):
            return
        self.circular_progress.animate_check()
        self.state_title.setText("Đã nhận dạng thành công")
        self.state_message.setText("Bản xem trước Word đã sẵn sàng.")
        if self.circular_progress._reduce_motion:
            self._finish_success(token)
            return
        self._success_finish = QTimer(self)
        self._success_finish.setSingleShot(True)
        self._success_finish.timeout.connect(lambda: self._finish_success(token))
        self._success_finish.start(420)

    def _finish_success(self, token: int) -> None:
        if (token != self._success_generation or self._pdf_path is None
                or self._preview_state != "success_animating"):
            return
        self._preview_state = "preview_ready"
        self.circular_progress.stop_animation()
        self.circular_progress.setVisible(False)
        self.zoom_in.setEnabled(True)
        self.zoom_out.setEnabled(True)
        self.fit_width.setEnabled(bool(self._page_count))
        self.content_stack.setCurrentWidget(self.scroll)

    def set_zoom(self, zoom: float) -> None:
        self._zoom = max(0.5, min(2.5, zoom))
        if self._pdf_path:
            self._render_cache.clear()
            self._rebuild_page_placeholders()
            self._request_pages(self._requested_center)

    def fit_to_width(self) -> None:
        if not self._pdf_path or not self._page_sizes:
            return
        width = max((size[0] for size in self._page_sizes), default=0)
        if width <= 0:
            return
        self._zoom = max(0.5, min(2.5, (self.scroll.viewport().width() - 48) / (width * 1.35)))
        self._render_cache.clear()
        self._rebuild_page_placeholders()
        self._request_pages(self._requested_center)

    def _clear_pages(self) -> None:
        while self.page_layout.count():
            item = self.page_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

    def _rebuild_page_placeholders(self) -> None:
        self._clear_pages()
        self._page_labels = []
        scale = self._zoom * 1.35
        for width, height in self._page_sizes:
            label = QLabel()
            label.setFixedSize(int(width * scale) + 14, int(height * scale) + 14)
            border = "#c8cbd0" if self._theme == "light" else "#555"
            label.setStyleSheet(f"background: white; border: 1px solid {border};")
            label.setContentsMargins(6, 6, 6, 6)
            self.page_layout.addWidget(label)
            self._page_labels.append(label)

    def _request_pages(self, center: int) -> None:
        if not self._pdf_path:
            return
        self._requested_center = min(max(0, center), max(0, self._page_count - 1))
        if self._render_worker is not None and self._render_worker.isRunning():
            self._generation += 1
            return
        self._generation += 1
        worker = _PdfPageWorker(self._pdf_path, self._requested_center, self._zoom, self._generation, self)
        worker.pages_ready.connect(self._pages_ready)
        worker.failed.connect(self._render_failed)
        perf_generation = self._perf_job_generation
        worker.stage_event.connect(
            lambda stage, transition, timestamp_ns, g=perf_generation:
                self.perf_stage_event.emit(g, stage, transition, timestamp_ns)
        )
        worker.finished.connect(lambda w=worker: self._worker_finished(w))
        self._render_worker = worker
        worker.start()

    def _pages_ready(self, generation: int, page_sizes, images, center: int) -> None:
        if generation != self._generation or self._pdf_path is None:
            return
        if not self._page_sizes:
            self._page_sizes = list(page_sizes)
            self._page_count = len(self._page_sizes)
            self._rebuild_page_placeholders()
            self.notice.hide()
        try:
            for index, image in images:
                if index < len(self._page_labels):
                    pixmap = QPixmap.fromImage(image)
                    self._render_cache[index] = pixmap
                    self._render_cache.move_to_end(index)
                    self._page_labels[index].setPixmap(pixmap)
        except Exception as exc:
            message = f"Qt could not display a rendered PDF page ({type(exc).__name__})."
            if not self._initial_preview_emitted:
                self.perf_stage_event.emit(
                    self._perf_job_generation, "preview_render", "end", time.perf_counter_ns()
                )
            self.set_notice(message)
            self.preview_failed.emit(message)
            return
        while len(self._render_cache) > 5:
            evicted, _ = self._render_cache.popitem(last=False)
            if evicted < len(self._page_labels):
                self._page_labels[evicted].setPixmap(QPixmap())
        current = min(self._page_count, max(1, center + 1))
        self.page_label.setText(f"Trang {current} / {self._page_count}")
        if not self._initial_preview_emitted:
            self._initial_preview_emitted = True
            self.perf_stage_event.emit(
                self._perf_job_generation, "preview_render", "end", time.perf_counter_ns()
            )
            self.preview_ready.emit(self._page_count)

    def _render_failed(self, generation: int, message: str) -> None:
        if generation == self._generation:
            if not self._initial_preview_emitted:
                self.perf_stage_event.emit(
                    self._perf_job_generation, "preview_render", "end", time.perf_counter_ns()
                )
            self.set_notice(message)
            self.preview_failed.emit(message)

    def _worker_finished(self, worker: _PdfPageWorker) -> None:
        if self._render_worker is worker:
            self._render_worker = None
        if self._pdf_path is not None and self._requested_center >= 0:
            expected = range(max(0, self._requested_center - 1), min(self._page_count, self._requested_center + 2))
            if any(index not in self._render_cache for index in expected):
                self._request_pages(self._requested_center)

    def _update_page_indicator(self, _value: int) -> None:
        if not self._pdf_path or not self._page_count:
            return
        position = self.scroll.verticalScrollBar().value()
        maximum = max(1, self.scroll.verticalScrollBar().maximum())
        current = min(self._page_count, max(1, int(position / maximum * self._page_count) + 1))
        self.page_label.setText(f"Trang {current} / {self._page_count}")
        self._request_pages(current - 1)

    def wait_for_idle(self, timeout_ms: int = 1000) -> bool:
        worker = self._render_worker
        if worker is not None and worker.isRunning():
            worker.wait(timeout_ms)
        return worker is None or not worker.isRunning()
