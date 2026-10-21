"""Lazy, embedded multi-page PDF preview of the generated Word document."""
from __future__ import annotations

import importlib
from collections import OrderedDict
from pathlib import Path

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget


class _PdfPageWorker(QThread):
    pages_ready = pyqtSignal(int, object, object, object)
    failed = pyqtSignal(int, str)

    def __init__(self, path: Path, center: int, zoom: float, generation: int, parent=None):
        super().__init__(parent)
        self.path = path
        self.center = center
        self.zoom = zoom
        self.generation = generation

    def run(self) -> None:
        try:
            pymupdf = importlib.import_module("pymupdf")
        except ImportError:
            self.failed.emit(self.generation, "PyMuPDF is unavailable in the active Python environment.")
            return
        try:
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
        self._requested_center = 0

        outer = QVBoxLayout(self)
        controls = QHBoxLayout()
        self.page_label = QLabel("Trang — / —")
        controls.addWidget(self.page_label)
        controls.addStretch(1)
        self.zoom_out = QPushButton("−")
        self.zoom_in = QPushButton("+")
        self.fit_width = QPushButton("Vừa chiều rộng")
        self.fit_width.setEnabled(False)
        controls.addWidget(self.zoom_out)
        controls.addWidget(self.zoom_in)
        controls.addWidget(self.fit_width)
        outer.addLayout(controls)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("QScrollArea { background: #292933; border: 0; }")
        self.pages = QWidget()
        self.page_layout = QVBoxLayout(self.pages)
        self.page_layout.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        self.scroll.setWidget(self.pages)
        outer.addWidget(self.scroll, 1)
        self.notice = QLabel("Chưa có tài liệu. Chạy OCR để tạo bản xem trước Word.")
        self.notice.setAlignment(Qt.AlignCenter)
        outer.addWidget(self.notice)

        self.zoom_in.clicked.connect(lambda: self.set_zoom(self._zoom * 1.15))
        self.zoom_out.clicked.connect(lambda: self.set_zoom(self._zoom / 1.15))
        self.fit_width.clicked.connect(self.fit_to_width)
        self.scroll.verticalScrollBar().valueChanged.connect(self._update_page_indicator)
        self.set_notice("Chưa có tài liệu. Chạy OCR để tạo bản xem trước Word.")

    def set_notice(self, message: str) -> None:
        self._generation += 1
        self._pdf_path = None
        self._page_count = 0
        self._page_sizes = []
        self._clear_pages()
        self._page_labels = []
        self._render_cache.clear()
        self.fit_width.setEnabled(False)
        self.notice.setText(message)
        self.notice.show()
        self.scroll.hide()

    def set_pdf(self, path: Path) -> None:
        self._generation += 1
        self._pdf_path = path
        self._page_count = 0
        self._page_sizes = []
        self._render_cache.clear()
        self._clear_pages()
        self._page_labels = []
        self._requested_center = 0
        self.notice.setText("Đang mở PDF xem trước …")
        self.notice.show()
        self.scroll.hide()
        self._request_pages(0)

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
            label.setStyleSheet("background: white; border: 1px solid #555;")
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
            self.fit_width.setEnabled(True)
            self.notice.hide()
            self.scroll.show()
        try:
            for index, image in images:
                if index < len(self._page_labels):
                    pixmap = QPixmap.fromImage(image)
                    self._render_cache[index] = pixmap
                    self._render_cache.move_to_end(index)
                    self._page_labels[index].setPixmap(pixmap)
        except Exception as exc:
            message = f"Qt could not display a rendered PDF page ({type(exc).__name__})."
            self.set_notice(message)
            self.preview_failed.emit(message)
            return
        while len(self._render_cache) > 5:
            evicted, _ = self._render_cache.popitem(last=False)
            if evicted < len(self._page_labels):
                self._page_labels[evicted].setPixmap(QPixmap())
        current = min(self._page_count, max(1, center + 1))
        self.page_label.setText(f"Trang {current} / {self._page_count}")
        self.preview_ready.emit(self._page_count)

    def _render_failed(self, generation: int, message: str) -> None:
        if generation == self._generation:
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
