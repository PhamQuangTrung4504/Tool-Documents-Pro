#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
app.ui.main_window
===================
Cửa sổ chính của ứng dụng OCR Desktop — PyQt5.

Tính năng:
- Mở ảnh từ file / clipboard / kéo thả
- Hiển thị preview ảnh gốc + ảnh có bounding box
- Chọn ngôn ngữ nhận dạng
- Nút OCR / Copy / Export
- Thanh trạng thái + progress bar
- Phím tắt đầy đủ
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from loguru import logger

from PyQt5.QtCore import (
    QMimeData,
    QSize,
    Qt,
    QThread,
    QTimer,
    pyqtSignal,
    pyqtSlot,
)
from PyQt5.QtGui import (
    QClipboard,
    QColor,
    QDragEnterEvent,
    QDropEvent,
    QFont,
    QIcon,
    QImage,
    QKeySequence,
    QPainter,
    QPalette,
    QPixmap,
)
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QGraphicsDropShadowEffect,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QShortcut,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.core.ocr_engine import (
    LANGUAGE_MAP,
    OCREngine,
    OCRResult,
    get_display_languages,
    get_lang_code,
)

# ---------------------------------------------------------------------------
# Hằng số giao diện
# ---------------------------------------------------------------------------

APP_TITLE = "Tool-Documents-Pro — OCR"
MIN_WIDTH = 1100
MIN_HEIGHT = 720

# Bảng màu Dark theme
COLORS = {
    "bg_primary": "#0f0f14",
    "bg_secondary": "#16161d",
    "bg_card": "#1c1c26",
    "bg_input": "#22222e",
    "border": "#2a2a3a",
    "border_focus": "#6c5ce7",
    "accent": "#6c5ce7",
    "accent_hover": "#7f70f0",
    "accent_pressed": "#5a4bd6",
    "success": "#00b894",
    "warning": "#fdcb6e",
    "danger": "#e17055",
    "text_primary": "#e8e8f0",
    "text_secondary": "#8888a0",
    "text_muted": "#555570",
    "scrollbar_bg": "#1a1a24",
    "scrollbar_handle": "#333348",
}

# ---------------------------------------------------------------------------
# Image file filter
# ---------------------------------------------------------------------------

IMAGE_FILTER = (
    "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.webp);;"
    "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp);;"
    "TIFF (*.tiff *.tif);;WebP (*.webp);;All Files (*)"
)

SUPPORTED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp",
}


# ---------------------------------------------------------------------------
# Worker thread — chạy OCR nền
# ---------------------------------------------------------------------------


class OCRWorker(QThread):
    """Thread chạy OCR để không block giao diện."""

    finished = pyqtSignal(object)  # OCRResult | Exception
    progress = pyqtSignal(str)     # Thông báo tiến trình

    def __init__(
        self,
        engine: OCREngine,
        image: np.ndarray,
        parent: Optional[QThread] = None,
    ) -> None:
        super().__init__(parent)
        self._engine = engine
        self._image = image

    def run(self) -> None:
        try:
            self.progress.emit("Đang tải model OCR …")
            self._engine._ensure_loaded()

            self.progress.emit("Đang nhận dạng …")
            result = self._engine.recognize(
                self._image, preprocess=True, denoise=True,
            )
            self.finished.emit(result)

        except Exception as exc:
            logger.exception("OCR worker error:")
            self.finished.emit(exc)


# ---------------------------------------------------------------------------
# Helper — chuyển đổi QImage ↔ numpy
# ---------------------------------------------------------------------------


def qimage_to_numpy(qimg: QImage) -> np.ndarray:
    """Chuyển QImage → numpy array (BGR)."""
    qimg = qimg.convertToFormat(QImage.Format_RGB32)
    width = qimg.width()
    height = qimg.height()

    ptr = qimg.bits()
    ptr.setsize(height * width * 4)
    arr = np.frombuffer(ptr, dtype=np.uint8).reshape((height, width, 4))

    # BGRA → BGR
    bgr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    return bgr


def numpy_to_qpixmap(image: np.ndarray) -> QPixmap:
    """Chuyển numpy array (BGR) → QPixmap."""
    if len(image.shape) == 2:
        h, w = image.shape
        qimg = QImage(image.data, w, h, w, QImage.Format_Grayscale8)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


# ---------------------------------------------------------------------------
# Stylesheet
# ---------------------------------------------------------------------------


def build_stylesheet() -> str:
    """Tạo stylesheet cho toàn bộ ứng dụng."""
    c = COLORS
    return f"""
    /* ===== GLOBAL ===== */
    QMainWindow, QWidget {{
        background-color: {c['bg_primary']};
        color: {c['text_primary']};
        font-family: 'Segoe UI', 'Arial', sans-serif;
        font-size: 13px;
    }}

    /* ===== TOOLBAR ===== */
    QToolBar {{
        background-color: {c['bg_secondary']};
        border-bottom: 1px solid {c['border']};
        padding: 6px 10px;
        spacing: 8px;
    }}
    QToolBar QToolButton {{
        background-color: transparent;
        border: 1px solid transparent;
        border-radius: 6px;
        padding: 6px 14px;
        color: {c['text_primary']};
        font-weight: 500;
    }}
    QToolBar QToolButton:hover {{
        background-color: {c['bg_card']};
        border-color: {c['border']};
    }}
    QToolBar QToolButton:pressed {{
        background-color: {c['accent_pressed']};
    }}

    /* ===== BUTTONS ===== */
    QPushButton {{
        background-color: {c['bg_card']};
        color: {c['text_primary']};
        border: 1px solid {c['border']};
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 600;
        font-size: 13px;
        min-height: 18px;
    }}
    QPushButton:hover {{
        background-color: {c['bg_input']};
        border-color: {c['accent']};
    }}
    QPushButton:pressed {{
        background-color: {c['accent_pressed']};
    }}
    QPushButton:disabled {{
        color: {c['text_muted']};
        border-color: {c['border']};
    }}

    QPushButton#btnOCR {{
        background: qlineargradient(
            x1:0, y1:0, x2:1, y2:1,
            stop:0 {c['accent']}, stop:1 #a29bfe
        );
        color: #ffffff;
        border: none;
        font-size: 14px;
        padding: 10px 28px;
        border-radius: 10px;
    }}
    QPushButton#btnOCR:hover {{
        background: qlineargradient(
            x1:0, y1:0, x2:1, y2:1,
            stop:0 {c['accent_hover']}, stop:1 #b8b0ff
        );
    }}
    QPushButton#btnOCR:pressed {{
        background-color: {c['accent_pressed']};
    }}
    QPushButton#btnOCR:disabled {{
        background: {c['bg_input']};
        color: {c['text_muted']};
    }}

    QPushButton#btnCopy {{
        background-color: {c['success']};
        color: #ffffff;
        border: none;
    }}
    QPushButton#btnCopy:hover {{
        background-color: #00d9a5;
    }}

    /* ===== COMBO BOX ===== */
    QComboBox {{
        background-color: {c['bg_input']};
        color: {c['text_primary']};
        border: 1px solid {c['border']};
        border-radius: 8px;
        padding: 6px 12px;
        min-width: 170px;
        font-size: 13px;
    }}
    QComboBox:hover {{
        border-color: {c['accent']};
    }}
    QComboBox::drop-down {{
        subcontrol-origin: padding;
        subcontrol-position: top right;
        width: 28px;
        border-left: 1px solid {c['border']};
        border-top-right-radius: 8px;
        border-bottom-right-radius: 8px;
    }}
    QComboBox::down-arrow {{
        image: none;
        border: none;
    }}
    QComboBox QAbstractItemView {{
        background-color: {c['bg_card']};
        color: {c['text_primary']};
        border: 1px solid {c['border']};
        selection-background-color: {c['accent']};
        selection-color: #ffffff;
        outline: none;
    }}

    /* ===== GROUP BOX ===== */
    QGroupBox {{
        background-color: {c['bg_secondary']};
        border: 1px solid {c['border']};
        border-radius: 10px;
        margin-top: 14px;
        padding: 16px 12px 10px 12px;
        font-weight: 600;
        font-size: 13px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 2px 12px;
        color: {c['accent']};
    }}

    /* ===== SCROLL AREA ===== */
    QScrollArea {{
        background-color: {c['bg_card']};
        border: 1px solid {c['border']};
        border-radius: 8px;
    }}
    QScrollBar:vertical {{
        background-color: {c['scrollbar_bg']};
        width: 8px;
        border-radius: 4px;
    }}
    QScrollBar::handle:vertical {{
        background-color: {c['scrollbar_handle']};
        border-radius: 4px;
        min-height: 30px;
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0px;
    }}
    QScrollBar:horizontal {{
        background-color: {c['scrollbar_bg']};
        height: 8px;
        border-radius: 4px;
    }}
    QScrollBar::handle:horizontal {{
        background-color: {c['scrollbar_handle']};
        border-radius: 4px;
        min-width: 30px;
    }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        width: 0px;
    }}

    /* ===== TEXT EDITOR ===== */
    QPlainTextEdit {{
        background-color: {c['bg_input']};
        color: {c['text_primary']};
        border: 1px solid {c['border']};
        border-radius: 8px;
        padding: 10px;
        font-family: 'Consolas', 'Cascadia Code', monospace;
        font-size: 13px;
        selection-background-color: {c['accent']};
        selection-color: #ffffff;
    }}
    QPlainTextEdit:focus {{
        border-color: {c['accent']};
    }}

    /* ===== STATUS BAR ===== */
    QStatusBar {{
        background-color: {c['bg_secondary']};
        color: {c['text_secondary']};
        border-top: 1px solid {c['border']};
        font-size: 12px;
        padding: 2px 8px;
    }}
    QStatusBar QLabel {{
        color: {c['text_secondary']};
        padding: 0 6px;
    }}

    /* ===== PROGRESS BAR ===== */
    QProgressBar {{
        background-color: {c['bg_input']};
        border: 1px solid {c['border']};
        border-radius: 6px;
        text-align: center;
        color: {c['text_primary']};
        font-size: 11px;
        min-height: 14px;
        max-height: 14px;
    }}
    QProgressBar::chunk {{
        background: qlineargradient(
            x1:0, y1:0, x2:1, y2:0,
            stop:0 {c['accent']}, stop:1 #a29bfe
        );
        border-radius: 5px;
    }}

    /* ===== LABEL ===== */
    QLabel#labelTitle {{
        color: {c['text_primary']};
        font-size: 14px;
        font-weight: 700;
    }}
    QLabel#labelHint {{
        color: {c['text_muted']};
        font-size: 12px;
    }}
    QLabel#labelDropZone {{
        color: {c['text_muted']};
        font-size: 15px;
        font-style: italic;
        qproperty-alignment: AlignCenter;
    }}

    /* ===== SPLITTER ===== */
    QSplitter::handle {{
        background-color: {c['border']};
    }}
    QSplitter::handle:horizontal {{
        width: 2px;
    }}

    /* ===== FRAME separator ===== */
    QFrame#separator {{
        background-color: {c['border']};
        max-height: 1px;
    }}
    """


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------


class MainWindow(QMainWindow):
    """Cửa sổ chính của ứng dụng OCR."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        # State
        self._current_image: Optional[np.ndarray] = None  # BGR
        self._current_pixmap: Optional[QPixmap] = None
        self._current_result: Optional[OCRResult] = None
        self._image_path: Optional[str] = None
        self._ocr_worker: Optional[OCRWorker] = None

        # Engine (lazy)
        self._engine: Optional[OCREngine] = None

        # UI setup
        self._init_window()
        self._init_toolbar()
        self._init_central()
        self._init_statusbar()
        self._init_shortcuts()

        # Accept drag & drop
        self.setAcceptDrops(True)

        logger.info("MainWindow initialized.")

    # ==================================================================
    # INIT — Window properties
    # ==================================================================

    def _init_window(self) -> None:
        self.setWindowTitle(APP_TITLE)
        self.setMinimumSize(MIN_WIDTH, MIN_HEIGHT)
        self.resize(1280, 820)
        self.setStyleSheet(build_stylesheet())

        # Đặt icon nếu có
        icon_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "icon.ico"
        )
        if os.path.isfile(icon_path):
            self.setWindowIcon(QIcon(icon_path))

    # ==================================================================
    # INIT — Toolbar
    # ==================================================================

    def _init_toolbar(self) -> None:
        tb = QToolBar("Toolbar")
        tb.setMovable(False)
        tb.setIconSize(QSize(20, 20))
        self.addToolBar(Qt.TopToolBarArea, tb)

        # --- Mở ảnh ---
        self.act_open = QAction("📂 Mở ảnh", self)
        self.act_open.setToolTip("Mở file ảnh (Ctrl+O)")
        self.act_open.setShortcut(QKeySequence("Ctrl+O"))
        self.act_open.triggered.connect(self._on_open_file)
        tb.addAction(self.act_open)

        # --- Dán ảnh ---
        self.act_paste = QAction("📋 Dán ảnh", self)
        self.act_paste.setToolTip("Dán ảnh từ clipboard (Ctrl+V)")
        self.act_paste.setShortcut(QKeySequence("Ctrl+V"))
        self.act_paste.triggered.connect(self._on_paste_image)
        tb.addAction(self.act_paste)

        tb.addSeparator()

        # --- Chọn ngôn ngữ ---
        lang_label = QLabel("  Ngôn ngữ: ")
        lang_label.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-size: 12px;"
        )
        tb.addWidget(lang_label)

        self.combo_lang = QComboBox()
        self.combo_lang.addItems(get_display_languages())
        self.combo_lang.setCurrentIndex(0)  # Tiếng Việt
        self.combo_lang.currentTextChanged.connect(self._on_language_changed)
        tb.addWidget(self.combo_lang)

        tb.addSeparator()

        # --- Nút OCR ---
        self.act_ocr = QAction("🔍 Nhận dạng (Ctrl+R)", self)
        self.act_ocr.setToolTip("Chạy nhận dạng chữ")
        self.act_ocr.triggered.connect(self._on_run_ocr)
        tb.addAction(self.act_ocr)

        # --- Spacer ---
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        spacer.setStyleSheet("background: transparent;")
        tb.addWidget(spacer)

        # --- Xoá ---
        self.act_clear = QAction("🗑️ Xoá", self)
        self.act_clear.setToolTip("Xoá ảnh và kết quả")
        self.act_clear.triggered.connect(self._on_clear)
        tb.addAction(self.act_clear)

    # ==================================================================
    # INIT — Central widget
    # ==================================================================

    def _init_central(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(10, 8, 10, 4)
        main_layout.setSpacing(8)

        # Splitter: bên trái = ảnh, bên phải = kết quả
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(3)
        main_layout.addWidget(splitter, stretch=1)

        # ---------- LEFT: Image Panel ----------
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        # Title
        img_title = QLabel("📷  Ảnh đầu vào")
        img_title.setObjectName("labelTitle")
        left_layout.addWidget(img_title)

        # Scroll area chứa ảnh
        self.scroll_image = QScrollArea()
        self.scroll_image.setWidgetResizable(True)
        self.scroll_image.setAlignment(Qt.AlignCenter)

        self.label_image = QLabel()
        self.label_image.setObjectName("labelDropZone")
        self.label_image.setText(
            "Kéo thả ảnh vào đây\n\n"
            "— hoặc —\n\n"
            "Ctrl+O  để mở file\n"
            "Ctrl+V  để dán từ clipboard"
        )
        self.label_image.setAlignment(Qt.AlignCenter)
        self.label_image.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        self.label_image.setMinimumSize(400, 300)
        self.scroll_image.setWidget(self.label_image)

        left_layout.addWidget(self.scroll_image, stretch=1)
        splitter.addWidget(left_panel)

        # ---------- RIGHT: Result Panel ----------
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        # Title
        result_title = QLabel("📝  Kết quả nhận dạng")
        result_title.setObjectName("labelTitle")
        right_layout.addWidget(result_title)

        # Text editor
        self.text_result = QPlainTextEdit()
        self.text_result.setPlaceholderText(
            "Kết quả OCR sẽ hiển thị tại đây …"
        )
        self.text_result.setReadOnly(False)
        right_layout.addWidget(self.text_result, stretch=1)

        # Info label
        self.label_info = QLabel("")
        self.label_info.setObjectName("labelHint")
        right_layout.addWidget(self.label_info)

        # Separator
        sep = QFrame()
        sep.setObjectName("separator")
        sep.setFrameShape(QFrame.HLine)
        right_layout.addWidget(sep)

        # Button row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        self.btn_ocr = QPushButton("🔍  Nhận dạng OCR")
        self.btn_ocr.setObjectName("btnOCR")
        self.btn_ocr.setFixedHeight(42)
        self.btn_ocr.clicked.connect(self._on_run_ocr)
        btn_row.addWidget(self.btn_ocr)

        self.btn_copy = QPushButton("📋  Sao chép")
        self.btn_copy.setObjectName("btnCopy")
        self.btn_copy.setFixedHeight(42)
        self.btn_copy.clicked.connect(self._on_copy_result)
        btn_row.addWidget(self.btn_copy)

        self.btn_export = QPushButton("💾  Xuất TXT")
        self.btn_export.setFixedHeight(42)
        self.btn_export.clicked.connect(self._on_export_txt)
        btn_row.addWidget(self.btn_export)

        right_layout.addLayout(btn_row)
        splitter.addWidget(right_panel)

        # Tỷ lệ splitter: 55% ảnh — 45% text
        splitter.setSizes([620, 500])

        # ---------- Bottom: Progress bar ----------
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # indeterminate
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(16)
        main_layout.addWidget(self.progress_bar)

    # ==================================================================
    # INIT — Status bar
    # ==================================================================

    def _init_statusbar(self) -> None:
        sb = QStatusBar()
        self.setStatusBar(sb)

        self.status_label = QLabel("Sẵn sàng")
        sb.addWidget(self.status_label, stretch=1)

        self.status_size = QLabel("")
        sb.addPermanentWidget(self.status_size)

        self.status_lang = QLabel(f"🌐 {self.combo_lang.currentText()}")
        sb.addPermanentWidget(self.status_lang)

    # ==================================================================
    # INIT — Keyboard shortcuts
    # ==================================================================

    def _init_shortcuts(self) -> None:
        QShortcut(QKeySequence("Ctrl+R"), self, self._on_run_ocr)
        QShortcut(QKeySequence("Ctrl+S"), self, self._on_export_txt)
        QShortcut(QKeySequence("Ctrl+Shift+C"), self, self._on_copy_result)
        QShortcut(QKeySequence("Ctrl+Q"), self, self.close)

    # ==================================================================
    # HELPERS
    # ==================================================================

    def _get_engine(self) -> OCREngine:
        """Trả về OCREngine hiện tại (tạo lazy nếu chưa có)."""
        lang_code = get_lang_code(self.combo_lang.currentText())
        if self._engine is None or self._engine.lang != lang_code:
            self._engine = OCREngine.get_engine(lang=lang_code)
        return self._engine

    def _set_status(self, msg: str) -> None:
        self.status_label.setText(msg)

    def _show_image(self, image: np.ndarray) -> None:
        """Hiển thị numpy image (BGR) lên label."""
        pix = numpy_to_qpixmap(image)
        self._current_pixmap = pix

        # Scale vừa scroll area
        available = self.scroll_image.size()
        scaled = pix.scaled(
            available, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.label_image.setPixmap(scaled)
        self.label_image.setAlignment(Qt.AlignCenter)

        h, w = image.shape[:2]
        self.status_size.setText(f"📐 {w} × {h} px")

    def _load_image_from_path(self, path: str) -> bool:
        """Đọc ảnh từ đường dẫn, hiển thị, lưu state."""
        path = os.path.normpath(path)
        if not os.path.isfile(path):
            QMessageBox.warning(
                self, "Lỗi", f"Không tìm thấy file:\n{path}"
            )
            return False

        ext = os.path.splitext(path)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            QMessageBox.warning(
                self, "Lỗi",
                f"Định dạng không hỗ trợ: {ext}\n"
                f"Hỗ trợ: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
            )
            return False

        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if image is None:
            QMessageBox.warning(
                self, "Lỗi", f"Không thể đọc ảnh:\n{path}"
            )
            return False

        # BGRA → BGR
        if len(image.shape) == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        self._current_image = image
        self._image_path = path
        self._current_result = None
        self._show_image(image)

        fname = os.path.basename(path)
        self.setWindowTitle(f"{APP_TITLE} — {fname}")
        self._set_status(f"Đã mở: {fname}")
        self.label_info.setText("")
        self.text_result.clear()

        logger.info(f"Image loaded: {path} ({image.shape})")
        return True

    def _load_image_from_clipboard(self) -> bool:
        """Đọc ảnh từ clipboard."""
        clipboard = QApplication.clipboard()
        mimedata = clipboard.mimeData()

        if mimedata.hasImage():
            qimg = clipboard.image()
            if qimg.isNull():
                return False
            image = qimage_to_numpy(qimg)
        elif mimedata.hasUrls():
            urls = mimedata.urls()
            for url in urls:
                if url.isLocalFile():
                    return self._load_image_from_path(url.toLocalFile())
            return False
        else:
            return False

        self._current_image = image
        self._image_path = None
        self._current_result = None
        self._show_image(image)

        self.setWindowTitle(f"{APP_TITLE} — (Clipboard)")
        self._set_status("Đã dán ảnh từ clipboard")
        self.label_info.setText("")
        self.text_result.clear()

        logger.info(f"Image loaded from clipboard ({image.shape})")
        return True

    # ==================================================================
    # DRAG & DROP
    # ==================================================================

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        mimedata = event.mimeData()
        if mimedata.hasUrls() or mimedata.hasImage():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        mimedata = event.mimeData()

        if mimedata.hasUrls():
            for url in mimedata.urls():
                if url.isLocalFile():
                    path = url.toLocalFile()
                    ext = os.path.splitext(path)[1].lower()
                    if ext in SUPPORTED_EXTENSIONS:
                        self._load_image_from_path(path)
                        event.acceptProposedAction()
                        return

        if mimedata.hasImage():
            qimg = QImage(mimedata.imageData())
            if not qimg.isNull():
                image = qimage_to_numpy(qimg)
                self._current_image = image
                self._image_path = None
                self._show_image(image)
                self.setWindowTitle(f"{APP_TITLE} — (Dropped)")
                self._set_status("Đã nhận ảnh kéo thả")
                event.acceptProposedAction()
                return

        event.ignore()

    # ==================================================================
    # SLOTS — User actions
    # ==================================================================

    @pyqtSlot()
    def _on_open_file(self) -> None:
        """Mở hộp thoại chọn file ảnh."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Mở ảnh", "", IMAGE_FILTER,
        )
        if path:
            self._load_image_from_path(path)

    @pyqtSlot()
    def _on_paste_image(self) -> None:
        """Dán ảnh từ clipboard."""
        if not self._load_image_from_clipboard():
            self._set_status("Clipboard không chứa ảnh")
            logger.info("Clipboard has no image data.")

    @pyqtSlot(str)
    def _on_language_changed(self, display_name: str) -> None:
        """Thay đổi ngôn ngữ OCR."""
        self.status_lang.setText(f"🌐 {display_name}")
        lang_code = get_lang_code(display_name)
        logger.info(f"Language changed: {display_name} ({lang_code})")

    @pyqtSlot()
    def _on_run_ocr(self) -> None:
        """Chạy nhận dạng OCR trên ảnh hiện tại."""
        if self._current_image is None:
            QMessageBox.information(
                self, "Thông báo",
                "Chưa có ảnh.\n"
                "Mở ảnh bằng Ctrl+O hoặc dán bằng Ctrl+V.",
            )
            return

        if self._ocr_worker is not None and self._ocr_worker.isRunning():
            self._set_status("OCR đang chạy, vui lòng đợi …")
            return

        # Disable controls
        self.btn_ocr.setEnabled(False)
        self.act_ocr.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.text_result.clear()
        self.label_info.setText("")
        self._set_status("Đang khởi tạo OCR …")

        engine = self._get_engine()

        self._ocr_worker = OCRWorker(engine, self._current_image, self)
        self._ocr_worker.progress.connect(self._on_ocr_progress)
        self._ocr_worker.finished.connect(self._on_ocr_finished)
        self._ocr_worker.start()

    @pyqtSlot(str)
    def _on_ocr_progress(self, msg: str) -> None:
        self._set_status(msg)

    @pyqtSlot(object)
    def _on_ocr_finished(self, result_or_error) -> None:
        """Callback khi OCR hoàn thành."""
        self.btn_ocr.setEnabled(True)
        self.act_ocr.setEnabled(True)
        self.progress_bar.setVisible(False)

        if isinstance(result_or_error, Exception):
            err_msg = str(result_or_error)
            self._set_status(f"Lỗi OCR: {err_msg}")
            QMessageBox.critical(
                self, "Lỗi OCR",
                f"Không thể nhận dạng:\n\n{err_msg}",
            )
            logger.error(f"OCR failed: {err_msg}")
            return

        result: OCRResult = result_or_error
        self._current_result = result

        # Hiển thị kết quả text
        self.text_result.setPlainText(result.full_text)

        # Vẽ bounding box lên ảnh
        if self._current_image is not None and result.boxes:
            annotated = OCREngine.draw_boxes(
                self._current_image, result,
                color=(108, 92, 231),  # accent color
                thickness=2,
                show_text=False,
            )
            self._show_image(annotated)

        # Cập nhật info
        avg_conf = result.average_confidence
        conf_color = COLORS["success"] if avg_conf > 0.8 else COLORS["warning"]
        info_text = (
            f"✅ {result.num_lines} dòng  |  "
            f"Độ tin cậy: <span style='color:{conf_color};'>"
            f"{avg_conf:.1%}</span>  |  "
            f"⏱ {result.elapsed_seconds:.2f}s"
        )
        self.label_info.setText(info_text)

        self._set_status(
            f"Hoàn thành: {result.num_lines} dòng, "
            f"{result.elapsed_seconds:.2f}s"
        )

        logger.info(
            f"OCR result displayed: {result.num_lines} lines, "
            f"avg_conf={avg_conf:.2f}"
        )

    @pyqtSlot()
    def _on_copy_result(self) -> None:
        """Sao chép kết quả vào clipboard."""
        text = self.text_result.toPlainText().strip()
        if not text:
            self._set_status("Chưa có kết quả để sao chép")
            return

        clipboard = QApplication.clipboard()
        clipboard.setText(text)
        self._set_status("✅ Đã sao chép kết quả vào clipboard")

        # Flash hiệu ứng — đổi text nút tạm thời
        original_text = self.btn_copy.text()
        self.btn_copy.setText("✅  Đã sao chép!")
        QTimer.singleShot(1500, lambda: self.btn_copy.setText(original_text))

    @pyqtSlot()
    def _on_export_txt(self) -> None:
        """Lưu kết quả ra file TXT."""
        text = self.text_result.toPlainText().strip()
        if not text:
            self._set_status("Chưa có kết quả để lưu")
            return

        # Tên file mặc định
        default_name = "ocr_result.txt"
        if self._image_path:
            base = os.path.splitext(os.path.basename(self._image_path))[0]
            default_name = f"{base}_ocr.txt"

        path, _ = QFileDialog.getSaveFileName(
            self, "Lưu kết quả OCR",
            default_name,
            "Text Files (*.txt);;All Files (*)",
        )
        if not path:
            return

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            self._set_status(f"✅ Đã lưu: {os.path.basename(path)}")
            logger.info(f"Result exported to: {path}")
        except OSError as exc:
            QMessageBox.critical(
                self, "Lỗi", f"Không thể lưu file:\n{exc}"
            )

    @pyqtSlot()
    def _on_clear(self) -> None:
        """Xoá ảnh và kết quả."""
        self._current_image = None
        self._current_pixmap = None
        self._current_result = None
        self._image_path = None

        self.label_image.clear()
        self.label_image.setText(
            "Kéo thả ảnh vào đây\n\n"
            "— hoặc —\n\n"
            "Ctrl+O  để mở file\n"
            "Ctrl+V  để dán từ clipboard"
        )
        self.text_result.clear()
        self.label_info.setText("")
        self.status_size.setText("")

        self.setWindowTitle(APP_TITLE)
        self._set_status("Đã xoá. Sẵn sàng.")

    # ==================================================================
    # RESIZE — Cập nhật ảnh khi resize cửa sổ
    # ==================================================================

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._current_pixmap and not self._current_pixmap.isNull():
            available = self.scroll_image.size()
            scaled = self._current_pixmap.scaled(
                available, Qt.KeepAspectRatio, Qt.SmoothTransformation,
            )
            self.label_image.setPixmap(scaled)

    # ==================================================================
    # CLOSE — Cleanup
    # ==================================================================

    def closeEvent(self, event) -> None:
        if self._ocr_worker and self._ocr_worker.isRunning():
            self._ocr_worker.terminate()
            self._ocr_worker.wait(2000)
        OCREngine.clear_cache()
        logger.info("MainWindow closed.")
        event.accept()
