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
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from loguru import logger

from PyQt5.QtCore import (
    QMimeData,
    QPoint,
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
    QTextCharFormat,
    QTextCursor,
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
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QShortcut,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTextEdit,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.core.ocr_engine import (
    LANGUAGE_MAP,
    OCREngine,
    OCRResult,
    export_to_docx,
    format_confidence,
    get_display_languages,
    get_lang_code,
)
from app.core.spell_corrector import SpellError, check_text_spelling
from app.core.document_session import DocumentSession
from app.core.document_renderer import DocumentRenderService
from app.ui.word_preview import WordPreviewWidget

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


class SpellCheckWorker(QThread):
    """Thread chạy kiểm tra chính tả tiếng Việt để không block giao diện."""

    finished = pyqtSignal(list, str)  # (suspicious_list, original_text)

    def __init__(
        self,
        text: str,
        parent: Optional[QThread] = None,
    ) -> None:
        super().__init__(parent)
        self._text = text

    def run(self) -> None:
        try:
            results = check_text_spelling(self._text)
            self.finished.emit(results, self._text)
        except Exception as exc:
            logger.exception("Spell check worker error:")
            self.finished.emit([], self._text)


class DocumentWorker(QThread):
    """Create the session DOCX and render that same file to PDF off the UI thread."""
    docx_ready = pyqtSignal(str, object)
    rendered = pyqtSignal(str, object)

    def __init__(self, result: OCRResult, session: DocumentSession, generation: int, parent=None):
        super().__init__(parent)
        self.result = result
        self.session = session
        self.generation = generation

    def run(self) -> None:
        if not export_to_docx(self.result, str(self.session.docx_path)):
            self.docx_ready.emit(self.session.session_id, RuntimeError("Không thể tạo file Word tạm."))
            return
        if not self.session.docx_path.is_file() or self.session.docx_path.stat().st_size == 0:
            self.docx_ready.emit(self.session.session_id, RuntimeError("File Word tạm không hợp lệ."))
            return
        try:
            with zipfile.ZipFile(self.session.docx_path) as archive:
                if "word/document.xml" not in archive.namelist():
                    raise zipfile.BadZipFile("missing word/document.xml")
                bad_member = archive.testzip()
                if bad_member:
                    raise zipfile.BadZipFile(f"corrupt member: {bad_member}")
        except (OSError, zipfile.BadZipFile) as exc:
            self.docx_ready.emit(self.session.session_id, RuntimeError(f"File Word tạm không hợp lệ: {exc}"))
            return
        self.docx_ready.emit(self.session.session_id, self.session.docx_path)
        try:
            pdf = DocumentRenderService().render_docx(self.session.docx_path, self.session.pdf_path)
            self.rendered.emit(self.session.session_id, pdf)
        except Exception as exc:
            self.rendered.emit(self.session.session_id, exc)


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

    QPushButton#btnSpellCheck {{
        background-color: #d63031;
        color: #ffffff;
        border: none;
    }}
    QPushButton#btnSpellCheck:hover {{
        background-color: #e17055;
    }}

    QPushButton#btnClearSpell {{
        background-color: #2d3436;
        color: #dfe6e9;
        border: 1px solid #4a4a5a;
    }}
    QPushButton#btnClearSpell:hover {{
        background-color: #3b3b4f;
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
        self._spell_worker: Optional[SpellCheckWorker] = None
        self._active_spell_errors: List[SpellError] = []
        self._is_replacing_suggestion: bool = False
        self._generation = 0
        self._document_session: Optional[DocumentSession] = None
        self._document_worker: Optional[DocumentWorker] = None
        self._document_workers: List[DocumentWorker] = []
        self._retired_sessions: Dict[str, DocumentSession] = {}
        self._document_dirty = False

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

        # Thiết lập AppUserModelID trên Windows để icon hiển thị đúng trên Windows Taskbar
        if sys.platform == "win32":
            try:
                import ctypes
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                    "ToolDocumentsPro.OCR.Desktop.v2"
                )
            except Exception:
                pass

        # Đặt icon ứng dụng chính thức
        icon_path = (
            Path(__file__).resolve().parents[2]
            / "assets"
            / "icon"
            / "icon_tool_documents_pro.ico"
        )
        if icon_path.exists():
            app_icon = QIcon(str(icon_path))
            self.setWindowIcon(app_icon)
            qapp = QApplication.instance()
            if qapp is not None:
                qapp.setWindowIcon(app_icon)

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

        # --- Kiểm tra chính tả ---
        self.act_spell_check = QAction("📝 Kiểm tra chính tả", self)
        self.act_spell_check.setToolTip("Kiểm tra chính tả tiếng Việt trong khung văn bản (Ctrl+K)")
        self.act_spell_check.triggered.connect(self._on_check_spelling)
        tb.addAction(self.act_spell_check)

        # --- Bỏ đánh dấu ---
        self.act_clear_spell = QAction("🧹 Bỏ đánh dấu", self)
        self.act_clear_spell.setToolTip("Xóa các đánh dấu chính tả (Ctrl+Shift+K)")
        self.act_clear_spell.triggered.connect(self._on_clear_spelling_highlights)
        tb.addAction(self.act_clear_spell)

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
        result_title = QLabel("📄  Xem trước tài liệu Word")
        result_title.setObjectName("labelTitle")
        right_layout.addWidget(result_title)

        self.document_tabs = QTabWidget()
        self.word_preview = WordPreviewWidget()
        self.document_tabs.addTab(self.word_preview, "Xem trước Word")
        right_layout.addWidget(self.document_tabs, stretch=1)

        # Text editor (kept as an explicit editing/spell-check mode)
        self.text_result = QPlainTextEdit()
        self.text_result.setPlaceholderText(
            "Kết quả OCR sẽ hiển thị tại đây …"
        )
        self.text_result.setReadOnly(False)
        self.text_result.textChanged.connect(self._on_text_changed)
        self.text_result.setContextMenuPolicy(Qt.CustomContextMenu)
        self.text_result.customContextMenuRequested.connect(self._on_text_context_menu)
        self.document_tabs.addTab(self.text_result, "Chỉnh sửa văn bản")

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

        self.btn_spell_check = QPushButton("📝  Kiểm tra chính tả")
        self.btn_spell_check.setObjectName("btnSpellCheck")
        self.btn_spell_check.setFixedHeight(42)
        self.btn_spell_check.setToolTip("Dò soát và đánh dấu nghi vấn chính tả trên Textbox (Ctrl+K)")
        self.btn_spell_check.clicked.connect(self._on_check_spelling)
        btn_row.addWidget(self.btn_spell_check)

        self.btn_clear_spell = QPushButton("🧹  Bỏ đánh dấu")
        self.btn_clear_spell.setObjectName("btnClearSpell")
        self.btn_clear_spell.setFixedHeight(42)
        self.btn_clear_spell.setToolTip("Xóa định dạng đánh dấu chính tả (Ctrl+Shift+K)")
        self.btn_clear_spell.clicked.connect(self._on_clear_spelling_highlights)
        btn_row.addWidget(self.btn_clear_spell)

        self.btn_export = QPushButton("💾  Xuất TXT")
        self.btn_export.setFixedHeight(42)
        self.btn_export.clicked.connect(self._on_export_txt)
        btn_row.addWidget(self.btn_export)

        self.btn_export_docx = QPushButton("📄  Lưu Word")
        self.btn_export_docx.setObjectName("btnExportDocx")
        self.btn_export_docx.setFixedHeight(42)
        self.btn_export_docx.setToolTip("Lưu bản Word đang được xem trước vào vị trí bạn chọn")
        self.btn_export_docx.clicked.connect(self._on_export_docx)
        btn_row.addWidget(self.btn_export_docx)

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
        QShortcut(QKeySequence("Ctrl+K"), self, self._on_check_spelling)
        QShortcut(QKeySequence("Ctrl+Shift+K"), self, self._on_clear_spelling_highlights)
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

        self._invalidate_document()

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

        self._invalidate_document()

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
                self._invalidate_document()
                image = qimage_to_numpy(qimg)
                self._current_image = image
                self._image_path = None
                self._current_result = None
                self._show_image(image)
                self.text_result.clear()
                self.label_info.setText("")
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

        self._invalidate_document()
        self._current_result = None

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
        generation = self._generation
        self._ocr_worker.finished.connect(lambda result, g=generation: self._on_ocr_finished(result, g))
        self._ocr_worker.start()

    @pyqtSlot(str)
    def _on_ocr_progress(self, msg: str) -> None:
        self._set_status(msg)

    @pyqtSlot(object)
    def _on_ocr_finished(self, result_or_error, generation: Optional[int] = None) -> None:
        """Callback khi OCR hoàn thành."""
        if generation is not None and generation != self._generation:
            return
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
        self._document_dirty = False
        self.word_preview.set_notice("Đang tạo tài liệu Word và bản xem trước …")
        self.btn_export_docx.setEnabled(False)
        self._create_document_session(result)

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
        conf_color = COLORS["success"] if avg_conf is not None and avg_conf > 0.8 else COLORS["warning"]
        conf_text = format_confidence(avg_conf, as_percent=True)
        conf_log = format_confidence(avg_conf)
        info_text = (
            f"✅ {result.num_lines} dòng  |  "
            f"Độ tin cậy: <span style='color:{conf_color};'>"
            f"{conf_text}</span>  |  "
            f"⏱ {result.elapsed_seconds:.2f}s"
        )
        self.label_info.setText(info_text)

        self._set_status(
            f"Hoàn thành: {result.num_lines} dòng, "
            f"{result.elapsed_seconds:.2f}s"
        )

        logger.info(
            f"OCR result displayed: {result.num_lines} lines, "
            f"avg_conf={conf_log}"
        )

    def _create_document_session(self, result: OCRResult) -> None:
        """Generate the temporary DOCX first, then render that same file."""
        old_session = self._document_session
        old_worker = self._document_worker
        if old_session is not None:
            if old_worker is not None and old_worker.isRunning():
                self._retired_sessions[old_session.session_id] = old_session
                old_worker.finished.connect(lambda sid=old_session.session_id: self._cleanup_retired_session(sid))
            else:
                old_session.cleanup()
        self._document_session = DocumentSession.create()
        worker = DocumentWorker(result, self._document_session, self._generation, self)
        worker.docx_ready.connect(self._on_temp_docx_ready)
        worker.rendered.connect(self._on_preview_rendered)
        worker.finished.connect(lambda w=worker: self._forget_document_worker(w))
        self._document_worker = worker
        self._document_workers.append(worker)
        self._set_status("Đang tạo tài liệu Word …")
        worker.start()

    @pyqtSlot(str, object)
    def _on_temp_docx_ready(self, session_id: str, result_or_error) -> None:
        session = self._document_session
        if session is None or session.session_id != session_id:
            return
        if isinstance(result_or_error, Exception):
            self.btn_export_docx.setEnabled(False)
            self.word_preview.set_notice(f"Không tạo được DOCX: {result_or_error}")
            self._set_status("OCR hoàn tất nhưng không tạo được file Word.")
            logger.error("Temporary DOCX generation failed ({})", type(result_or_error).__name__)
            return
        self.btn_export_docx.setEnabled(True)
        self._set_status("DOCX sẵn sàng. Đang tạo bản xem trước …")

    @pyqtSlot(str, object)
    def _on_preview_rendered(self, session_id: str, result_or_error) -> None:
        session = self._document_session
        if session is None or session.session_id != session_id:
            return
        if isinstance(result_or_error, Exception):
            self.word_preview.set_notice(str(result_or_error))
            self._set_status("DOCX đã sẵn sàng; không thể tạo bản xem trước.")
            logger.error(
                "Document preview failed at renderer stage ({}, code={}).",
                type(result_or_error).__name__,
                getattr(result_or_error, "diagnostic_code", "not-applicable"),
            )
            return
        self.word_preview.set_pdf(Path(result_or_error))
        self._set_status("Bản xem trước Word đã sẵn sàng.")

    def _invalidate_document(self) -> None:
        """Invalidate pending callbacks and clean only this window's owned session."""
        self._generation += 1
        self.btn_ocr.setEnabled(True)
        self.act_ocr.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.word_preview.set_notice("Chưa có tài liệu. Chạy OCR để tạo bản xem trước Word.")
        self.btn_export_docx.setEnabled(False)
        self._document_dirty = False
        session, self._document_session = self._document_session, None
        worker = self._document_worker
        if session is not None:
            if worker is not None and worker.isRunning():
                self._retired_sessions[session.session_id] = session
                worker.finished.connect(lambda sid=session.session_id: self._cleanup_retired_session(sid))
            elif not session.cleanup():
                logger.warning(f"Temporary document session cleanup deferred: {session.directory}")

    def _cleanup_retired_session(self, session_id: str) -> None:
        session = self._retired_sessions.pop(session_id, None)
        if session is not None and not session.cleanup():
            logger.warning(f"Temporary document session cleanup deferred: {session.directory}")

    def _forget_document_worker(self, worker: DocumentWorker) -> None:
        if worker in self._document_workers:
            self._document_workers.remove(worker)

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
    def _on_export_docx(self) -> None:
        """Save the exact generated DOCX without modifying its session copy."""
        if self._document_dirty:
            self._regenerate_edited_document()
            self._set_status("Đang cập nhật Word từ nội dung đã chỉnh sửa. Bấm Lưu Word khi DOCX sẵn sàng.")
            return
        text = self.text_result.toPlainText().strip()
        if not text or self._document_session is None or not self._document_session.docx_path.is_file():
            self._set_status("Chưa có kết quả để lưu")
            return

        default_name = "ocr_result.docx"
        if self._image_path:
            base = os.path.splitext(os.path.basename(self._image_path))[0]
            default_name = f"{base}_ocr.docx"

        path, _ = QFileDialog.getSaveFileName(
            self, "Lưu kết quả OCR dạng Word",
            default_name,
            "Word Documents (*.docx);;All Files (*)",
        )
        if not path:
            return
        target = Path(path)
        if target.suffix.lower() != ".docx":
            target = target.with_suffix(".docx")
        try:
            target.resolve().relative_to(self._document_session.root if self._document_session else Path(tempfile.gettempdir()) / "Tool-Documents-Pro" / "temp")
            inside_managed_root = True
        except (OSError, RuntimeError, ValueError):
            inside_managed_root = False
        if inside_managed_root:
            QMessageBox.warning(self, "Đường dẫn không hợp lệ", "Không thể lưu bản Word vào thư mục tạm do ứng dụng quản lý.")
            return
        if target.exists():
            answer = QMessageBox.question(self, "Xác nhận ghi đè", f"File đã tồn tại:\n{target}\n\nBạn có muốn ghi đè không?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                return
        temp_path = None
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(prefix=f".{target.stem}-", suffix=".docx", dir=str(target.parent), delete=False) as tmp:
                temp_path = Path(tmp.name)
            shutil.copyfile(self._document_session.docx_path, temp_path)
            os.replace(temp_path, target)
            self._set_status(f"✅ Đã lưu Word: {os.path.basename(path)}")
            QMessageBox.information(self, "Đã lưu Word", f"Đã lưu tài liệu tại:\n{target}")
            logger.info(f"Result exported to DOCX: {target}")
        except OSError as exc:
            if temp_path and temp_path.exists():
                temp_path.unlink()
            QMessageBox.critical(
                self, "Lỗi", f"Không thể lưu file Word:\n{exc}"
            )

    @pyqtSlot()
    def _on_clear(self) -> None:
        """Xoá ảnh và kết quả."""
        self._invalidate_document()
        self._current_image = None
        self._current_pixmap = None
        self._current_result = None
        self._image_path = None
        self._active_spell_errors = []

        self.label_image.clear()
        self.label_image.setText(
            "Kéo thả ảnh vào đây\n\n"
            "— hoặc —\n\n"
            "Ctrl+O  để mở file\n"
            "Ctrl+V  để dán từ clipboard"
        )
        self.text_result.clear()
        self.text_result.setExtraSelections([])
        self.label_info.setText("")
        self.status_size.setText("")

        self.setWindowTitle(APP_TITLE)
        self._set_status("Đã xoá. Sẵn sàng.")

    # ==================================================================
    # SLOTS — Spell Checking & Context Menu
    # ==================================================================

    @pyqtSlot()
    def _on_check_spelling(self) -> None:
        """
        Dò soát nội dung trong khung Textbox, phát hiện và bôi đỏ/gạch chân đỏ
        các vị trí có dấu hiệu sai chính tả tiếng Việt.
        NGUYÊN TẮC: CHỈ KIỂM TRA VÀ ĐÁNH DẤU, KHÔNG TỰ ĐỘNG SỬA HAY THAY ĐỔI VĂN BẢN.
        """
        text = self.text_result.toPlainText()
        if not text or not text.strip():
            self._set_status("Không đủ dữ liệu để kiểm tra.")
            return

        self.document_tabs.setCurrentWidget(self.text_result)

        if self._spell_worker is not None and self._spell_worker.isRunning():
            self._set_status("Đang kiểm tra chính tả, vui lòng đợi …")
            return

        self._set_status("Đang dò soát chính tả tiếng Việt theo ngữ cảnh …")
        self._spell_worker = SpellCheckWorker(text, self)
        self._spell_worker.finished.connect(self._on_spell_check_finished)
        self._spell_worker.start()

    @pyqtSlot(list, str)
    def _on_spell_check_finished(self, results: list, checked_text: str) -> None:
        """Callback khi kiểm tra chính tả hoàn thành."""
        current_text = self.text_result.toPlainText()
        if current_text != checked_text:
            logger.debug("Text was modified during spell check; discarding stale highlights.")
            return

        self._active_spell_errors = list(results)

        if not results:
            self.text_result.setExtraSelections([])
            self._set_status("Không phát hiện lỗi rõ ràng theo các quy tắc hiện có.")
            logger.info("Spell check finished: 0 suspicious tokens found.")
            return

        self._update_spell_highlights()
        count = len(results)
        self._set_status(
            f"Phát hiện {count} vị trí nghi vấn. Nhấp chuột phải vào từ được đánh dấu để xem gợi ý sửa."
        )
        logger.info(f"Spell check finished: {count} suspicious tokens highlighted.")

    def _update_spell_highlights(self) -> None:
        """Vẽ lại định dạng bôi đỏ cho các vị trí nghi vấn còn lại."""
        if not self._active_spell_errors:
            self.text_result.setExtraSelections([])
            return

        fmt = QTextCharFormat()
        fmt.setForeground(QColor("#ff4d4f"))
        fmt.setFontUnderline(True)
        fmt.setUnderlineColor(QColor("#ff4d4f"))
        fmt.setUnderlineStyle(QTextCharFormat.WaveUnderline)
        fmt.setBackground(QColor(255, 77, 79, 40))

        doc = self.text_result.document()
        doc_len = doc.characterCount()
        selections = []
        for err in self._active_spell_errors:
            if err.end_idx <= doc_len:
                cursor = QTextCursor(doc)
                cursor.setPosition(err.start_idx)
                cursor.setPosition(err.end_idx, QTextCursor.KeepAnchor)
                sel = QTextEdit.ExtraSelection()
                sel.cursor = cursor
                sel.format = fmt
                selections.append(sel)

        self.text_result.setExtraSelections(selections)

    @pyqtSlot(QPoint)
    def _on_text_context_menu(self, pos: QPoint) -> None:
        """
        Hiển thị menu chuột phải tương tác:
        Nếu nhấp chuột vào token nghi vấn đã đánh dấu, hiển thị các lựa chọn gợi ý sửa
        ở đầu menu (ví dụ: '✨ Đổi thành: kỹ sư', '🚫 Bỏ qua lỗi này').
        """
        cursor = self.text_result.cursorForPosition(pos)
        click_pos = cursor.position()
        doc_text = self.text_result.toPlainText()

        # Tìm token nghi vấn tại vị trí con trỏ chuột
        target_error: Optional[SpellError] = None
        for err in self._active_spell_errors:
            if err.start_idx <= click_pos <= err.end_idx:
                if doc_text[err.start_idx:err.end_idx] == err.token:
                    target_error = err
                    break

        menu = QMenu(self)

        if target_error is not None:
            # Tiêu đề thông tin lỗi
            header_act = QAction(f"🔍 Nghi vấn: '{target_error.token}'", menu)
            header_act.setEnabled(False)
            menu.addAction(header_act)

            if target_error.reason:
                reason_act = QAction(f"ℹ️ {target_error.reason}", menu)
                reason_act.setEnabled(False)
                menu.addAction(reason_act)

            menu.addSeparator()

            # Danh sách các gợi ý sửa
            if target_error.suggestions:
                for sugg in target_error.suggestions:
                    act = QAction(f"✨ Đổi thành: {sugg}", menu)
                    act.triggered.connect(
                        lambda checked, err=target_error, s=sugg: self._apply_spell_suggestion(err, s)
                    )
                    menu.addAction(act)
            else:
                no_sugg = QAction("Không có gợi ý chắc chắn", menu)
                no_sugg.setEnabled(False)
                menu.addAction(no_sugg)

            menu.addSeparator()

            # Tùy chọn bỏ qua lỗi này
            act_ignore = QAction("🚫 Bỏ qua lỗi này", menu)
            act_ignore.triggered.connect(
                lambda checked, err=target_error: self._ignore_spell_error(err)
            )
            menu.addAction(act_ignore)

            menu.addSeparator()

        # Menu chỉnh sửa văn bản tiêu chuẩn (Undo, Redo, Cut, Copy, Paste, Select All)
        std_menu = self.text_result.createStandardContextMenu()
        menu.addActions(std_menu.actions())

        menu.exec_(self.text_result.mapToGlobal(pos))

    def _apply_spell_suggestion(self, err: SpellError, suggestion: str) -> None:
        """Thay thế token nghi vấn bằng gợi ý sửa do người dùng chủ động chọn."""
        doc_text = self.text_result.toPlainText()
        if doc_text[err.start_idx:err.end_idx] != err.token:
            self._set_status("Văn bản đã thay đổi so với vị trí đánh dấu — vui lòng kiểm tra lại.")
            self._on_clear_spelling_highlights()
            return

        # Bảo toàn kiểu chữ (casing) của token gốc
        replacement = suggestion
        if err.token.isupper() and len(err.token) > 1:
            replacement = suggestion.upper()
        elif err.token and err.token[0].isupper() and not suggestion[0].isupper():
            replacement = suggestion[0].upper() + suggestion[1:]

        # Thay thế bằng QTextCursor với EditBlock để cho phép Undo (Ctrl+Z)
        self._is_replacing_suggestion = True
        try:
            cursor = self.text_result.textCursor()
            cursor.setPosition(err.start_idx)
            cursor.setPosition(err.end_idx, QTextCursor.KeepAnchor)
            cursor.beginEditBlock()
            cursor.insertText(replacement)
            cursor.endEditBlock()
        finally:
            self._is_replacing_suggestion = False

        # Cập nhật vị trí các lỗi còn lại sau khi độ dài thay đổi
        delta = len(replacement) - len(err.token)
        remaining: List[SpellError] = []
        for other in self._active_spell_errors:
            if other is err or (other.start_idx == err.start_idx and other.end_idx == err.end_idx):
                continue
            if other.start_idx > err.start_idx:
                other.start_idx += delta
                other.end_idx += delta
            remaining.append(other)

        self._active_spell_errors = remaining
        self._update_spell_highlights()

        remaining_count = len(self._active_spell_errors)
        if remaining_count > 0:
            self._set_status(f"Đã đổi '{err.token}' → '{replacement}'. Còn {remaining_count} vị trí nghi vấn.")
        else:
            self._set_status(f"Đã đổi '{err.token}' → '{replacement}'. Đã xử lý hết nghi vấn.")
        logger.info(f"Spell suggestion applied: '{err.token}' -> '{replacement}'")

    def _ignore_spell_error(self, err: SpellError) -> None:
        """Bỏ qua vị trí nghi vấn hiện tại."""
        self._active_spell_errors = [
            e for e in self._active_spell_errors
            if not (e.start_idx == err.start_idx and e.end_idx == err.end_idx)
        ]
        self._update_spell_highlights()
        remaining_count = len(self._active_spell_errors)
        self._set_status(f"Đã bỏ qua '{err.token}'. Còn {remaining_count} vị trí nghi vấn.")
        logger.info(f"Spell error ignored: '{err.token}'")

    @pyqtSlot()
    def _on_clear_spelling_highlights(self) -> None:
        """Xóa toàn bộ đánh dấu chính tả trên Textbox mà không thay đổi nội dung."""
        self._active_spell_errors = []
        if self.text_result.extraSelections():
            self.text_result.setExtraSelections([])
        self._set_status("Đã xoá các đánh dấu chính tả.")
        logger.info("Spell check highlights cleared.")

    @pyqtSlot()
    def _on_text_changed(self) -> None:
        """Xóa hoặc cập nhật các đánh dấu cũ khi người dùng chỉnh sửa văn bản trực tiếp."""
        if getattr(self, "_is_replacing_suggestion", False):
            self._mark_document_dirty_if_changed()
            return
        if self._active_spell_errors:
            self._active_spell_errors = []
            self.text_result.setExtraSelections([])
            self._set_status("Văn bản đã chỉnh sửa — đã xoá các đánh dấu cũ.")
        self._mark_document_dirty_if_changed()

    def _mark_document_dirty_if_changed(self) -> None:
        if self._current_result is not None and self._current_result.full_text != self.text_result.toPlainText():
            self._document_dirty = True
            self.word_preview.set_notice("Văn bản đã chỉnh sửa. Khi lưu Word, cấu trúc bảng OCR gốc sẽ không được áp dụng để tránh lệch dữ liệu.")
            self.btn_export_docx.setEnabled(True)

    def _regenerate_edited_document(self) -> None:
        if not self._document_dirty:
            return
        text = self.text_result.toPlainText()
        self._document_dirty = False
        self._create_document_session(OCRResult(full_text=text))

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
        if not self.word_preview.wait_for_idle(1000):
            self._set_status("Đang hoàn tất hiển thị trang PDF. Đóng cửa sổ lần nữa sau khi kết thúc.")
            event.ignore()
            return
        running_workers = [worker for worker in (self._ocr_worker, self._spell_worker, *self._document_workers) if worker is not None and worker.isRunning()]
        for worker in running_workers:
            worker.wait(1000)
        if any(worker.isRunning() for worker in running_workers):
            self._set_status("Đang hoàn tất tác vụ nền. Đóng cửa sổ lần nữa sau khi kết thúc.")
            event.ignore()
            return
        if self._document_session is not None:
            self._document_session.cleanup()
        for session in self._retired_sessions.values():
            session.cleanup()
        OCREngine.clear_cache()
        logger.info("MainWindow closed.")
        event.accept()
