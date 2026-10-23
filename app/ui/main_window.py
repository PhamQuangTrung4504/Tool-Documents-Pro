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
- Thanh trạng thái và tiến trình ngay trong Word Preview
- Phím tắt đầy đủ
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
import zipfile
from threading import Event
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from loguru import logger

from PyQt5.QtCore import (
    QEvent,
    QSize,
    Qt,
    QThread,
    pyqtSignal,
    pyqtSlot,
    QSettings,
)
from PyQt5.QtGui import (
    QClipboard,
    QDragEnterEvent,
    QDropEvent,
    QFont,
    QIcon,
    QImage,
    QKeySequence,
    QPixmap,
)
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QShortcut,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.core.ocr_engine import (
    LANGUAGE_MAP,
    OCREngine,
    OCRCancelledError,
    OCRResult,
    export_to_docx,
    format_confidence,
    get_display_languages,
    get_lang_code,
)
from app.core.spell_corrector import check_text_spelling
from app.core.document_session import DocumentSession
from app.core.structured_editing import EditOwnershipError, StructuredDocumentEditor
from app.core.document_renderer import DocumentRenderService
from app.ui.word_preview import WordPreviewWidget

# ---------------------------------------------------------------------------
# Hằng số giao diện
# ---------------------------------------------------------------------------

APP_TITLE = "Tool-Documents-Pro — OCR"
MIN_WIDTH = 1040
MIN_HEIGHT = 720

# Mã màu giao diện được tập trung để Light/Dark dùng chung thành phần.
THEME_COLORS = {
  "dark": {
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
  },
  "light": {
    "bg_primary": "#f7f7f8", "bg_secondary": "#ffffff", "bg_card": "#ffffff",
    "bg_input": "#f1f2f4", "border": "#e1e3e6", "border_focus": "#6c5ce7",
    "accent": "#6c5ce7", "accent_hover": "#5b4bd6", "accent_pressed": "#4e40c4",
    "success": "#168a65", "warning": "#a66a00", "danger": "#bd3c38",
    "text_primary": "#202123", "text_secondary": "#62656a", "text_muted": "#858990",
    "scrollbar_bg": "#f1f2f4", "scrollbar_handle": "#c5c8ce",
  },
}
DESIGN_TOKENS = {
    "font_family": "'Segoe UI', 'Arial', sans-serif",
    "font_size": 13,
    "space_small": 6,
    "space_medium": 10,
    "radius_small": 6,
    "radius_medium": 8,
    "radius_large": 12,
}
COLORS = THEME_COLORS["dark"]

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

    result_ready = pyqtSignal(object)  # OCRResult | Exception
    progress = pyqtSignal(str)     # Thông báo tiến trình
    progress_detail = pyqtSignal(int, int)  # Hoàn tất/tổng số vùng của giai đoạn nhận dạng

    def __init__(
        self,
        engine: OCREngine,
        image: np.ndarray,
        parent: Optional[QThread] = None,
    ) -> None:
        super().__init__(parent)
        self._engine = engine
        self._image = image
        self._cancel_event = Event()

    def request_cancel(self) -> None:
        self._cancel_event.set()

    def run(self) -> None:
        try:
            if self._cancel_event.is_set():
                raise OCRCancelledError("OCR đã được hủy trước khi bắt đầu.")
            self.progress.emit("Đang tải model OCR …")
            self._engine._ensure_loaded()
            if self._cancel_event.is_set():
                raise OCRCancelledError("OCR đã được hủy sau khi tải model.")

            self.progress.emit("Đang nhận dạng …")
            result = self._engine.recognize(
                self._image, preprocess=True, denoise=True,
                progress_callback=self.progress.emit,
                progress_detail_callback=self.progress_detail.emit,
                cancel_event=self._cancel_event,
            )
            self.result_ready.emit(result)
        except OCRCancelledError as exc:
            self.result_ready.emit(exc)
        except Exception as exc:
            logger.exception("OCR worker error:")
            self.result_ready.emit(exc)


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
    progress = pyqtSignal(str, str)

    def __init__(self, result: OCRResult, session: DocumentSession, generation: int, parent=None):
        super().__init__(parent)
        self.result = result
        self.session = session
        self.generation = generation

    def run(self) -> None:
        self.progress.emit(self.session.session_id, "Đang xuất DOCX …")
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
            self.progress.emit(self.session.session_id, "Đang chuyển Word thành PDF xem trước …")
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


def build_stylesheet(theme: str = "dark") -> str:
    """Tạo stylesheet cho toàn bộ ứng dụng."""
    c = THEME_COLORS.get(theme, THEME_COLORS["dark"])
    return f"""
    /* ===== GLOBAL ===== */
    QMainWindow, QWidget {{
        background-color: {c['bg_primary']};
        color: {c['text_primary']};
        font-family: {DESIGN_TOKENS['font_family']};
        font-size: {DESIGN_TOKENS['font_size']}px;
    }}

    /* ===== TOOLBAR ===== */
    QToolBar {{
        background-color: {c['bg_secondary']};
        border-bottom: 1px solid {c['border']};
        padding: {DESIGN_TOKENS['space_small']}px {DESIGN_TOKENS['space_medium']}px;
        spacing: {DESIGN_TOKENS['space_medium']}px;
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
    QLabel#appBrand {{ font-size: 15px; font-weight: 700; color: {c['text_primary']}; padding: 0 8px; }}
    QLabel#toolbarLabel {{ color: {c['text_secondary']}; padding-left: 4px; }}

    /* ===== BUTTONS ===== */
    QPushButton {{
        background-color: {c['bg_card']};
        color: {c['text_primary']};
        border: 1px solid {c['border']};
        border-radius: {DESIGN_TOKENS['radius_medium']}px;
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

    QTabWidget::pane {{ border: 1px solid {c['border']}; border-radius: 10px; top: -1px; background: {c['bg_secondary']}; }}
    QTabBar::tab {{ background: {c['bg_primary']}; color: {c['text_secondary']}; padding: 9px 14px; margin-right: 4px; border: 1px solid transparent; border-radius: 8px; }}
    QTabBar::tab:selected {{ color: {c['text_primary']}; background: {c['bg_card']}; border-color: {c['border']}; }}
    QTabBar::tab:hover:!selected {{ color: {c['text_primary']}; background: {c['bg_input']}; }}

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

    QFrame#previewStatePanel {{
        background-color: {c['bg_card']};
        border: 1px solid {c['border']};
        border-radius: 12px;
    }}
    QLabel#previewStateTitle {{ color: {c['text_primary']}; font-size: 20px; font-weight: 700; }}
    QLabel#previewStateMessage {{ color: {c['text_secondary']}; font-size: 14px; }}
    QPushButton#primaryButton {{ background-color: {c['accent']}; color: #fff; border: 0; border-radius: {DESIGN_TOKENS['radius_medium']}px; padding: 7px 16px; font-weight: 700; }}
    QPushButton#primaryButton:hover {{ background-color: {c['accent_hover']}; }}
    QPushButton#primaryButton:pressed {{ background-color: {c['accent_pressed']}; }}
    QPushButton#primaryButton:disabled {{ background-color: {c['bg_input']}; color: {c['text_muted']}; }}
    QPushButton#secondaryButton, QPushButton#iconButton {{ background-color: {c['bg_card']}; color: {c['text_primary']}; border: 1px solid {c['border']}; border-radius: {DESIGN_TOKENS['radius_medium']}px; padding: 6px 12px; }}
    QPushButton#secondaryButton:hover, QPushButton#iconButton:hover {{ border-color: {c['accent']}; }}
    QPushButton#iconButton {{ min-width: 32px; max-width: 36px; padding: 4px; }}
    QPushButton:focus, QToolButton:focus, QComboBox:focus {{ border: 1px solid {c['border_focus']}; }}
    QMenu {{ background: {c['bg_card']}; color: {c['text_primary']}; border: 1px solid {c['border']}; padding: 4px; }}
    QMenu::item {{ padding: 7px 22px 7px 10px; border-radius: {DESIGN_TOKENS['radius_small']}px; }}
    QMenu::item:selected {{ background: {c['bg_input']}; color: {c['text_primary']}; }}
    QToolTip {{ background: {c['bg_card']}; color: {c['text_primary']}; border: 1px solid {c['border']}; padding: 5px 7px; }}

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
        self._settings = QSettings("Tool-Documents-Pro", "Tool-Documents-Pro")
        self._theme = str(self._settings.value("theme", "dark"))
        if self._theme not in THEME_COLORS:
            self._theme = "dark"
        global COLORS
        COLORS = THEME_COLORS[self._theme]

        # State
        self._current_image: Optional[np.ndarray] = None  # BGR
        self._current_pixmap: Optional[QPixmap] = None
        self._current_result: Optional[OCRResult] = None
        self._image_path: Optional[str] = None
        self._ocr_worker: Optional[OCRWorker] = None
        self._spell_worker: Optional[SpellCheckWorker] = None
        self._generation = 0
        self._document_session: Optional[DocumentSession] = None
        self._document_worker: Optional[DocumentWorker] = None
        self._document_workers: List[DocumentWorker] = []
        self._retired_sessions: Dict[str, DocumentSession] = {}
        self._structured_editor: Optional[StructuredDocumentEditor] = None
        self._review_resolved_boxes: Set[int] = set()
        self._structured_saved = True

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
        self.setStyleSheet(build_stylesheet(self._theme))
        app = QApplication.instance()
        if app is not None:
            app.setFont(QFont("Segoe UI", 10))

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
        tb = QToolBar("Tác vụ tài liệu")
        tb.setMovable(False)
        tb.setIconSize(QSize(20, 20))
        self.addToolBar(Qt.TopToolBarArea, tb)

        app_name = QLabel("Tool-Documents-Pro")
        app_name.setObjectName("appBrand")
        tb.addWidget(app_name)
        tb.addSeparator()

        tb.setObjectName("mainToolbar")
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
        lang_label = QLabel("Ngôn ngữ")
        lang_label.setObjectName("toolbarLabel")
        tb.addWidget(lang_label)

        self.combo_lang = QComboBox()
        self.combo_lang.addItems(get_display_languages())
        self.combo_lang.setCurrentIndex(0)  # Tiếng Việt
        self.combo_lang.currentTextChanged.connect(self._on_language_changed)
        tb.addWidget(self.combo_lang)

        tb.addSeparator()

        # --- Nút OCR ---
        self.act_ocr = QAction("Nhận dạng OCR (Ctrl+R)", self)
        self.act_ocr.setToolTip("Chạy nhận dạng chữ")
        self.act_ocr.triggered.connect(self._on_run_ocr)

        # --- Spacer ---
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        spacer.setStyleSheet("background: transparent;")
        tb.addWidget(spacer)

        # --- Hành động chính ---
        self.btn_ocr = QPushButton("Nhận dạng OCR")
        self.btn_ocr.setObjectName("primaryButton")
        self.btn_ocr.setMinimumHeight(36)
        self.btn_ocr.setToolTip("Chạy nhận dạng chữ (Ctrl+R)")
        self.btn_ocr.clicked.connect(self._on_run_ocr)
        tb.addWidget(self.btn_ocr)
        self.act_ocr.triggered.connect(self._on_run_ocr)

        self.btn_export_docx = QPushButton("Lưu Word")
        self.btn_export_docx.setObjectName("secondaryButton")
        self.btn_export_docx.setMinimumHeight(36)
        self.btn_export_docx.setToolTip("Lưu bản Word đang được xem trước")
        self.btn_export_docx.clicked.connect(self._on_export_docx)
        tb.addWidget(self.btn_export_docx)

        self.utility_menu = QMenu("Tiện ích", self)
        self.btn_copy = QAction("Sao chép văn bản", self)
        self.btn_copy.triggered.connect(self._on_copy_result)
        self.act_spell_check = QAction("Kiểm tra chính tả", self)
        self.act_spell_check.setToolTip("Dò soát nội dung OCR và xem các vị trí cần kiểm tra")
        self.act_spell_check.triggered.connect(self._on_check_spelling)
        self.btn_export = QAction("Xuất văn bản TXT…", self)
        self.btn_export.triggered.connect(self._on_export_txt)
        self.act_clear = QAction("Xóa tài liệu hiện tại…", self)
        self.act_clear.triggered.connect(self._on_clear)
        for action in (self.btn_copy, self.act_spell_check, self.btn_export):
            self.utility_menu.addAction(action)
        self.utility_menu.addSeparator()
        self.utility_menu.addAction(self.act_clear)
        utility_button = QToolButton()
        utility_button.setText("Tiện ích")
        utility_button.setPopupMode(QToolButton.InstantPopup)
        utility_button.setMenu(self.utility_menu)
        tb.addWidget(utility_button)

        theme_menu = QMenu("Giao diện", self)
        self.act_theme_light = QAction("Sáng", self)
        self.act_theme_light.setCheckable(True)
        self.act_theme_dark = QAction("Tối", self)
        self.act_theme_dark.setCheckable(True)
        self.act_theme_light.triggered.connect(lambda: self._apply_theme("light"))
        self.act_theme_dark.triggered.connect(lambda: self._apply_theme("dark"))
        theme_menu.addAction(self.act_theme_light)
        theme_menu.addAction(self.act_theme_dark)
        theme_menu.addSeparator()
        self.act_reduce_motion = QAction("Giảm chuyển động", self)
        self.act_reduce_motion.setCheckable(True)
        self.act_reduce_motion.setChecked(
            self._settings.value("reduce_motion", False, type=bool)
        )
        self.act_reduce_motion.toggled.connect(self._set_reduce_motion)
        theme_menu.addAction(self.act_reduce_motion)
        self.act_theme_menu = QAction("Giao diện", self)
        self.act_theme_menu.setMenu(theme_menu)
        tb.addAction(self.act_theme_menu)
        self._sync_theme_actions()

    # ==================================================================
    # INIT — Central widget
    # ==================================================================

    def _init_central(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(10, 8, 10, 4)
        main_layout.setSpacing(8)

        # Workspace chia đôi, có giới hạn tối thiểu để Preview luôn dùng được.
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(3)
        splitter.setChildrenCollapsible(False)
        self.workspace_splitter = splitter
        main_layout.addWidget(splitter, stretch=1)

        # ---------- LEFT: Image Panel ----------
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        # Title
        img_title = QLabel("Ảnh nguồn")
        img_title.setObjectName("labelTitle")
        left_layout.addWidget(img_title)

        source_tools = QHBoxLayout()
        self.source_zoom_out = QPushButton("−")
        self.source_zoom_out.setObjectName("iconButton")
        self.source_zoom_out.setToolTip("Thu nhỏ ảnh")
        self.source_zoom_in = QPushButton("+")
        self.source_zoom_in.setObjectName("iconButton")
        self.source_zoom_in.setToolTip("Phóng to ảnh")
        self.source_fit = QPushButton("Vừa khung")
        self.source_fit.setObjectName("secondaryButton")
        self.source_fit.setToolTip("Thu phóng ảnh vừa khung xem")
        self.source_zoom = 1.0
        self.source_fit_mode = True
        self.source_zoom_out.clicked.connect(lambda: self._set_source_zoom(self.source_zoom / 1.15))
        self.source_zoom_in.clicked.connect(lambda: self._set_source_zoom(self.source_zoom * 1.15))
        self.source_fit.clicked.connect(self._fit_source_image)
        source_tools.addStretch(1)
        source_tools.addWidget(self.source_zoom_out)
        source_tools.addWidget(self.source_zoom_in)
        source_tools.addWidget(self.source_fit)
        left_layout.addLayout(source_tools)

        # Scroll area chứa ảnh
        self.scroll_image = QScrollArea()
        self.scroll_image.setWidgetResizable(False)
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
        self.label_image.installEventFilter(self)
        self.label_image.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        self.label_image.setMinimumSize(1, 1)
        self.scroll_image.setWidget(self.label_image)

        left_layout.addWidget(self.scroll_image, stretch=1)
        self.label_source_name = QLabel("Chưa mở ảnh")
        self.label_source_name.setObjectName("labelHint")
        left_layout.addWidget(self.label_source_name)
        left_panel.setMinimumWidth(340)
        splitter.addWidget(left_panel)

        # ---------- RIGHT: Result Panel ----------
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        # Title
        result_title = QLabel("Không gian Word")
        result_title.setObjectName("labelTitle")
        right_layout.addWidget(result_title)

        self.document_tabs = QTabWidget()
        self.word_preview = WordPreviewWidget()
        self.word_preview.set_theme(self._theme)
        self.word_preview.circular_progress.set_reduce_motion(
            self._settings.value("reduce_motion", False, type=bool)
        )
        self.word_preview.preview_ready.connect(self._on_word_preview_ready)
        self.word_preview.preview_failed.connect(self._on_word_preview_failed)
        self.word_preview.cancel_requested.connect(self._request_ocr_cancel)
        self.word_preview.retry_requested.connect(self._retry_processing)
        self.processing_overlay = self.word_preview.state_panel
        self.processing_title = self.word_preview.state_title
        self.processing_message = self.word_preview.state_message
        self.processing_indicator = self.word_preview.circular_progress
        self.processing_cancel = self.word_preview.state_cancel
        self.processing_retry = self.word_preview.state_retry
        self.document_tabs.addTab(self.word_preview, "Xem trước Word")
        right_layout.addWidget(self.document_tabs, stretch=1)

        review_tab = QWidget()
        review_layout = QVBoxLayout(review_tab)
        review_layout.setContentsMargins(10, 10, 10, 10)
        review_layout.addWidget(QLabel("Chọn một vùng OCR để đối chiếu với ảnh nguồn."))
        self.ocr_review_list = QListWidget()
        self.ocr_review_list.currentRowChanged.connect(self._on_review_box_selected)
        review_layout.addWidget(self.ocr_review_list, stretch=2)
        self.ocr_review_crop = QLabel("Ảnh crop sẽ hiển thị tại đây")
        self.ocr_review_crop.setAlignment(Qt.AlignCenter)
        self.ocr_review_crop.setMinimumHeight(100)
        self.ocr_review_crop.setMaximumHeight(150)
        self.ocr_review_crop.setStyleSheet(f"background: {COLORS['bg_input']}; border: 1px solid {COLORS['border']};")
        review_layout.addWidget(self.ocr_review_crop)
        self.ocr_review_evidence = QLabel("Chưa chọn vùng")
        self.ocr_review_evidence.setWordWrap(True)
        review_layout.addWidget(self.ocr_review_evidence)
        self.ocr_review_candidates = QComboBox()
        self.ocr_review_candidates.currentIndexChanged.connect(self._on_review_candidate_selected)
        review_layout.addWidget(self.ocr_review_candidates)
        self.ocr_review_edit = QLineEdit()
        self.ocr_review_edit.setPlaceholderText("Nội dung đã đối chiếu hoặc nhập bản sửa")
        review_layout.addWidget(self.ocr_review_edit)
        review_buttons = QHBoxLayout()
        self.btn_review_confirm = QPushButton("Xác nhận sửa")
        self.btn_review_confirm.clicked.connect(self._confirm_review_edit)
        self.btn_review_cancel = QPushButton("Hủy sửa")
        self.btn_review_cancel.clicked.connect(self._cancel_review_edit)
        review_buttons.addWidget(self.btn_review_confirm)
        review_buttons.addWidget(self.btn_review_cancel)
        review_layout.addLayout(review_buttons)
        history_buttons = QHBoxLayout()
        self.btn_review_next = QPushButton("Vùng cần xem tiếp")
        self.btn_review_next.clicked.connect(self._select_next_review_box)
        self.btn_review_undo = QPushButton("Hoàn tác")
        self.btn_review_undo.clicked.connect(self._undo_structured_edit)
        self.btn_review_redo = QPushButton("Làm lại")
        self.btn_review_redo.clicked.connect(self._redo_structured_edit)
        history_buttons.addWidget(self.btn_review_next)
        history_buttons.addWidget(self.btn_review_undo)
        history_buttons.addWidget(self.btn_review_redo)
        review_layout.addLayout(history_buttons)
        self.document_tabs.addTab(review_tab, "Rà soát OCR")
        self._set_review_controls_enabled(False)

        right_panel.setMinimumWidth(500)
        # Info label
        self.label_info = QLabel("")
        self.label_info.setObjectName("labelHint")
        right_layout.addWidget(self.label_info)

        splitter.addWidget(right_panel)

        # Tỷ lệ splitter: 55% ảnh — 45% text
        splitter.setSizes([560, 700])

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

    def _apply_theme(self, theme: str) -> None:
        if theme not in THEME_COLORS:
            return
        global COLORS
        self._theme = theme
        COLORS = THEME_COLORS[theme]
        self._settings.setValue("theme", theme)
        self.setStyleSheet(build_stylesheet(theme))
        self.ocr_review_crop.setStyleSheet(
            f"background: {COLORS['bg_input']}; border: 1px solid {COLORS['border']};"
        )
        self.word_preview.set_theme(theme)
        self._sync_theme_actions()

    def _sync_theme_actions(self) -> None:
        if hasattr(self, "act_theme_light"):
            self.act_theme_light.setChecked(self._theme == "light")
            self.act_theme_dark.setChecked(self._theme == "dark")

    def _set_reduce_motion(self, enabled: bool) -> None:
        self._settings.setValue("reduce_motion", bool(enabled))
        self._settings.sync()
        if hasattr(self, "word_preview"):
            self.word_preview.circular_progress.set_reduce_motion(enabled)

    def _set_preview_processing_state(
        self, state: str, message: str = "", cancellable: bool = False,
    ) -> None:
        """Hiển thị toàn bộ trạng thái xử lý trong Word Preview."""
        if state == "file_ready":
            self.word_preview.set_processing_state("empty", "Ảnh đã sẵn sàng. Chạy OCR để tạo tài liệu Word.")
            return
        if state == "completed":
            self.word_preview.set_processing_state("completed")
            return
        if state == "processing":
            worker = self._ocr_worker
            if worker is not None and worker.isRunning() and worker._cancel_event.is_set():
                state = "cancelling"
        self.word_preview.set_processing_state(state, message, cancellable=cancellable)

    def _request_ocr_cancel(self) -> None:
        worker = self._ocr_worker
        if worker is None or not worker.isRunning():
            return
        worker.request_cancel()
        self._set_preview_processing_state(
            "cancelling", "Đang hủy... Chờ bước nhận dạng hiện tại dừng an toàn."
        )

    def _retry_processing(self) -> None:
        if self._current_result is not None:
            self._set_preview_processing_state("processing", "Đang tạo lại DOCX và bản xem trước …")
            self.btn_export_docx.setEnabled(False)
            self._create_document_session(self._current_result)
        else:
            self._on_run_ocr()

    def _show_image(self, image: np.ndarray) -> None:
        """Hiển thị numpy image (BGR) lên label."""
        pix = numpy_to_qpixmap(image)
        self._current_pixmap = pix

        self._render_source_image()
        self.label_image.setAlignment(Qt.AlignCenter)

        h, w = image.shape[:2]
        self.status_size.setText(f"📐 {w} × {h} px")

    def _render_source_image(self) -> None:
        if self._current_pixmap is None or self._current_pixmap.isNull():
            return
        available = self.scroll_image.viewport().size()
        size = available if self.source_fit_mode else self._current_pixmap.size() * self.source_zoom
        scaled = self._current_pixmap.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.label_image.setPixmap(scaled)
        self.label_image.setFixedSize(scaled.size())

    def _set_source_zoom(self, zoom: float) -> None:
        self.source_fit_mode = False
        self.source_zoom = max(0.25, min(4.0, zoom))
        self._render_source_image()

    def _fit_source_image(self) -> None:
        self.source_fit_mode = True
        self._render_source_image()

    def _reset_source_view(self) -> None:
        self.source_zoom = 1.0
        self.source_fit_mode = True

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

        if not self._confirm_discard_unsaved_edits():
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
        self._reset_source_view()
        self._set_preview_processing_state("file_ready")
        self._show_image(image)

        fname = os.path.basename(path)
        self.setWindowTitle(f"{APP_TITLE} — {fname}")
        self._set_status(f"Đã mở: {fname}")
        self.label_info.setText("")

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

        if not self._confirm_discard_unsaved_edits():
            return False

        self._invalidate_document()

        self._current_image = image
        self._image_path = None
        self._current_result = None
        self._reset_source_view()
        self._set_preview_processing_state("file_ready")
        self._show_image(image)

        self.setWindowTitle(f"{APP_TITLE} — (Clipboard)")
        self._set_status("Đã dán ảnh từ clipboard")
        self.label_info.setText("")

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
                if not self._confirm_discard_unsaved_edits():
                    event.ignore()
                    return
                self._invalidate_document()
                image = qimage_to_numpy(qimg)
                self._current_image = image
                self._image_path = None
                self._current_result = None
                self._reset_source_view()
                self._set_preview_processing_state("file_ready")
                self._show_image(image)
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

        if not self._confirm_discard_unsaved_edits():
            return

        self._invalidate_document()
        self._current_result = None
        self._set_preview_processing_state("empty")
        self.document_tabs.setCurrentWidget(self.word_preview)
        if self._current_image is not None:
            self._show_image(self._current_image)

        # Disable controls
        self.btn_ocr.setEnabled(False)
        self.act_ocr.setEnabled(False)
        self.label_info.setText("")
        self._set_status("Đang khởi tạo OCR …")

        engine = self._get_engine()

        self._ocr_worker = OCRWorker(engine, self._current_image, self)
        generation = self._generation
        worker = self._ocr_worker
        worker.progress.connect(
            lambda msg, g=generation, w=worker: self._on_ocr_progress(msg, g, w)
        )
        worker.progress_detail.connect(
            lambda done, total, g=generation, w=worker:
                self._on_ocr_progress_detail(done, total, g, w)
        )
        worker.result_ready.connect(lambda result, g=generation: self._on_ocr_finished(result, g))
        worker.finished.connect(lambda w=worker: self._on_ocr_thread_stopped(w))
        self._set_preview_processing_state(
            "processing", "Đang khởi tạo OCR …", cancellable=True
        )
        self._ocr_worker.start()

    def _on_ocr_thread_stopped(self, worker: OCRWorker) -> None:
        """Dọn trạng thái nút khi luồng kết thúc, kể cả kết quả job đã cũ."""
        if worker is self._ocr_worker:
            self._ocr_worker = None
        self.btn_ocr.setEnabled(True)
        self.act_ocr.setEnabled(True)

    def _on_ocr_progress(
        self, msg: str, generation: Optional[int] = None, worker=None,
    ) -> None:
        if generation is not None and generation != self._generation:
            return
        worker = worker or self._ocr_worker
        if worker is not self._ocr_worker:
            return
        cancel_requested = worker is not None and worker._cancel_event.is_set()
        if cancel_requested:
            msg = "Đang hủy... Chờ bước nhận dạng hiện tại dừng an toàn."
        self._set_status(msg)
        cancellable = (
            worker is not None and worker.isRunning()
            and not cancel_requested
        )
        self._set_preview_processing_state("processing", msg, cancellable=cancellable)

    def _on_ocr_progress_detail(
        self, completed: int, total: int, generation: int, worker: OCRWorker,
    ) -> None:
        """Tiến độ đếm vùng thuộc riêng giai đoạn nhận dạng OCR."""
        if generation != self._generation or worker is not self._ocr_worker:
            return
        if worker._cancel_event.is_set() or total <= 0:
            return
        self.word_preview.set_recognition_progress(completed, total)
        self._set_status(
            f"Đang nhận dạng văn bản: {min(max(0, completed), total)}/{total} vùng"
        )

    @pyqtSlot(object)
    def _on_ocr_finished(self, result_or_error, generation: Optional[int] = None) -> None:
        """Callback khi OCR hoàn thành."""
        if generation is not None and generation != self._generation:
            return
        self.btn_ocr.setEnabled(True)
        self.act_ocr.setEnabled(True)

        if isinstance(result_or_error, Exception):
            err_msg = str(result_or_error)
            if isinstance(result_or_error, OCRCancelledError):
                self._set_status("Đã hủy tác vụ OCR.")
                self._set_preview_processing_state("cancelled", "Đã hủy nhận dạng. Có thể chạy lại khi cần.")
                return
            self._set_status(f"Lỗi OCR: {err_msg}")
            self._set_preview_processing_state("failed", f"Không thể nhận dạng. {err_msg}")
            logger.error(f"OCR failed: {err_msg}")
            return

        result: OCRResult = result_or_error
        self._current_result = result
        self._structured_editor = StructuredDocumentEditor(result)
        self._review_resolved_boxes.clear()
        self._structured_saved = True
        self._refresh_ocr_review_list()
        self._set_preview_processing_state("processing", "Đang dựng DOCX và bản xem trước Word …")
        self.btn_export_docx.setEnabled(False)
        self._create_document_session(result)

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
        self.document_tabs.setCurrentWidget(self.word_preview)
        self._set_preview_processing_state("processing", "Đang dựng DOCX và bản xem trước Word …")
        self.btn_export_docx.setEnabled(False)
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
        worker.progress.connect(self._on_document_progress)
        worker.finished.connect(lambda w=worker: self._forget_document_worker(w))
        self._document_worker = worker
        self._document_workers.append(worker)
        self._set_status("Đang tạo tài liệu Word …")
        worker.start()

    @pyqtSlot(str, str)
    def _on_document_progress(self, session_id: str, message: str) -> None:
        session = self._document_session
        if session is None or session.session_id != session_id:
            return
        self._set_preview_processing_state("processing", message)
        self._set_status(message)

    @pyqtSlot(str, object)
    def _on_temp_docx_ready(self, session_id: str, result_or_error) -> None:
        session = self._document_session
        if session is None or session.session_id != session_id:
            return
        if isinstance(result_or_error, Exception):
            self.btn_export_docx.setEnabled(False)
            self.word_preview.set_notice(f"Không tạo được DOCX: {result_or_error}")
            self._set_preview_processing_state("failed", "Không tạo được DOCX. Bạn có thể thử lại.")
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
            self._set_preview_processing_state("failed", "Không tạo được bản xem trước Word. Bạn có thể thử lại.")
            self._set_status("DOCX đã sẵn sàng; không thể tạo bản xem trước.")
            logger.error(
                "Document preview failed at renderer stage ({}, code={}).",
                type(result_or_error).__name__,
                getattr(result_or_error, "diagnostic_code", "not-applicable"),
            )
            return
        self.word_preview.set_pdf(Path(result_or_error))
        self._set_preview_processing_state("processing", "Đang hiển thị trang Word đầu tiên …")
        self._set_status("PDF đã tạo. Đang hiển thị trang xem trước …")

    @pyqtSlot(int)
    def _on_word_preview_ready(self, page_count: int) -> None:
        self.word_preview.show_success(page_count)
        self._set_status(f"Bản xem trước Word đã sẵn sàng ({page_count} trang).")

    @pyqtSlot(str)
    def _on_word_preview_failed(self, message: str) -> None:
        self._set_preview_processing_state("failed", message)
        self._set_status("Không thể hiển thị bản xem trước Word.")

    def _invalidate_document(self) -> None:
        """Invalidate pending callbacks and clean only this window's owned session."""
        self._generation += 1
        self.btn_ocr.setEnabled(True)
        self.act_ocr.setEnabled(True)
        self.word_preview.set_notice("Chưa có tài liệu. Chạy OCR để tạo bản xem trước Word.")
        self.btn_export_docx.setEnabled(False)
        self._structured_editor = None
        self._review_resolved_boxes.clear()
        self._structured_saved = True
        self.ocr_review_list.clear()
        self.ocr_review_crop.setText("Ảnh crop sẽ hiển thị tại đây")
        self.ocr_review_evidence.setText("Chưa chọn vùng")
        self._set_review_controls_enabled(False)
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
        text = self._current_document_text().strip()
        if not text:
            self._set_status("Chưa có kết quả để sao chép")
            return

        clipboard = QApplication.clipboard()
        clipboard.setText(text)
        self._set_status("✅ Đã sao chép kết quả vào clipboard")

        self.statusBar().showMessage("Đã sao chép kết quả vào bộ nhớ tạm", 1800)

    @pyqtSlot()
    def _on_export_txt(self) -> None:
        """Lưu kết quả ra file TXT."""
        text = self._current_document_text().strip()
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
        text = self._current_document_text().strip()
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
            if self._structured_editor is not None and self._structured_editor.dirty:
                self._structured_saved = True
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
        if self._current_image is not None or self._current_result is not None:
            answer = QMessageBox.question(
                self, "Xóa tài liệu hiện tại",
                "Xóa ảnh nguồn và toàn bộ kết quả đang mở?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
        if not self._confirm_discard_unsaved_edits():
            return
        self._invalidate_document()
        self._current_image = None
        self._current_pixmap = None
        self._current_result = None
        self._structured_editor = None
        self._review_resolved_boxes.clear()
        self.ocr_review_list.clear()
        self.ocr_review_crop.setText("Ảnh crop sẽ hiển thị tại đây")
        self.ocr_review_evidence.setText("Chưa chọn vùng")
        self._set_review_controls_enabled(False)
        self._image_path = None

        self.label_image.clear()
        self.label_source_name.setText("Chưa mở ảnh")
        self.label_image.setText(
            "Kéo thả ảnh vào đây\n\n"
            "— hoặc —\n\n"
            "Ctrl+O  để mở file\n"
            "Ctrl+V  để dán từ clipboard"
        )
        self.label_info.setText("")
        self.status_size.setText("")

        self.setWindowTitle(APP_TITLE)
        self._set_status("Đã xoá. Sẵn sàng.")

    # ==================================================================
    # SLOTS — Spell Checking
    # ==================================================================

    def _current_document_text(self) -> str:
        if self._structured_editor is not None:
            return self._structured_editor.result.full_text
        return self._current_result.full_text if self._current_result is not None else ""

    @pyqtSlot()
    def _on_check_spelling(self) -> None:
        """Dò soát nội dung hiện tại mà không sửa văn bản OCR tự động."""
        text = self._current_document_text()
        if not text or not text.strip():
            self._set_status("Không đủ dữ liệu để kiểm tra.")
            return
        if self._spell_worker is not None and self._spell_worker.isRunning():
            self._set_status("Đang kiểm tra chính tả, vui lòng đợi …")
            return
        self._set_status("Đang dò soát chính tả tiếng Việt theo ngữ cảnh …")
        self._spell_worker = SpellCheckWorker(text, self)
        self._spell_worker.finished.connect(self._on_spell_check_finished)
        self._spell_worker.start()

    @pyqtSlot(list, str)
    def _on_spell_check_finished(self, results: list, checked_text: str) -> None:
        """Báo cáo vị trí nghi vấn; việc sửa vẫn diễn ra qua Rà soát OCR."""
        current_text = self._current_document_text()
        if current_text != checked_text:
            logger.debug("Document changed during spell check; discarding stale result.")
            return
        if not results:
            self._set_status("Không phát hiện lỗi rõ ràng theo các quy tắc hiện có.")
            return
        count = len(results)
        details = []
        for error in results[:30]:
            suggestions = ", ".join(error.suggestions[:3]) or "không có gợi ý chắc chắn"
            details.append(f"• {error.token} — gợi ý: {suggestions}")
        if count > len(details):
            details.append(f"… và {count - len(details)} vị trí khác.")
        self._set_status(f"Phát hiện {count} vị trí cần đối chiếu; không tự sửa nội dung.")
        QMessageBox.information(
            self, "Kết quả kiểm tra chính tả",
            f"Phát hiện {count} vị trí cần đối chiếu. Hãy chọn vùng tương ứng trong Rà soát OCR để sửa có cấu trúc.\n\n"
            + "\n".join(details),
        )

    def _set_review_controls_enabled(self, enabled: bool) -> None:
        for widget in (self.ocr_review_list, self.ocr_review_candidates,
                       self.ocr_review_edit, self.btn_review_confirm,
                       self.btn_review_cancel, self.btn_review_next):
            widget.setEnabled(enabled)
        editor = self._structured_editor
        self.btn_review_undo.setEnabled(enabled and editor is not None and editor.can_undo)
        self.btn_review_redo.setEnabled(enabled and editor is not None and editor.can_redo)

    def _refresh_ocr_review_list(self, selected_box: Optional[int] = None) -> None:
        self.ocr_review_list.blockSignals(True)
        self.ocr_review_list.clear()
        result = self._current_result
        if result is not None:
            for index, box in enumerate(result.boxes):
                needs_review = bool(box.metadata.get("needs_review")) and index not in self._review_resolved_boxes
                marker = "⚠ " if needs_review else "✓ "
                text = " ".join(box.text.split())
                item = QListWidgetItem(f"{marker}{index + 1}: {text[:72]}")
                item.setData(Qt.UserRole, index)
                self.ocr_review_list.addItem(item)
                if selected_box == index:
                    self.ocr_review_list.setCurrentItem(item)
        self.ocr_review_list.blockSignals(False)
        self._set_review_controls_enabled(result is not None and bool(result.boxes))
        if selected_box is not None and result is not None:
            self._on_review_box_selected(self.ocr_review_list.currentRow())

    def _selected_review_box_index(self) -> Optional[int]:
        item = self.ocr_review_list.currentItem()
        value = item.data(Qt.UserRole) if item is not None else None
        return value if isinstance(value, int) else None

    def _on_review_box_selected(self, _row: int) -> None:
        index = self._selected_review_box_index()
        result = self._current_result
        image = self._current_image
        if index is None or result is None or image is None or index >= len(result.boxes):
            self.ocr_review_crop.setText("Chưa chọn vùng")
            self.ocr_review_evidence.setText("Chưa chọn vùng")
            return
        box = result.boxes[index]
        height, width = image.shape[:2]
        points = np.asarray(box.bbox, dtype=np.int32)
        x1 = max(0, int(points[:, 0].min()))
        y1 = max(0, int(points[:, 1].min()))
        x2 = min(width, int(points[:, 0].max()) + 1)
        y2 = min(height, int(points[:, 1].max()) + 1)
        if x2 > x1 and y2 > y1:
            crop = image[y1:y2, x1:x2]
            pixmap = numpy_to_qpixmap(crop).scaled(
                self.ocr_review_crop.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.ocr_review_crop.setPixmap(pixmap)
        else:
            self.ocr_review_crop.setText("Không có crop hợp lệ cho vùng này")

        candidates = box.metadata.get("candidates", [])
        evidence_lines = [f"Văn bản chính: {box.text}"]
        evidence_lines.append("Trạng thái: cần xem lại" if box.metadata.get("needs_review") else "Trạng thái: không gắn cờ xem lại")
        for candidate in candidates:
            if isinstance(candidate, dict):
                evidence_lines.append(f"{candidate.get('recognizer', 'OCR')}: {candidate.get('text', '')}")
        mapping = box.metadata.get("alternate_mapping", {})
        if isinstance(mapping, dict):
            evidence_lines.append(f"Ánh xạ ứng viên phụ: {mapping.get('status', 'không rõ')}")
        self.ocr_review_evidence.setText("\n".join(evidence_lines))

        self.ocr_review_candidates.blockSignals(True)
        self.ocr_review_candidates.clear()
        self.ocr_review_candidates.addItem("Chọn ứng viên đã đối chiếu…", None)
        for candidate_index, candidate in enumerate(candidates):
            if isinstance(candidate, dict) and isinstance(candidate.get("text"), str):
                name = candidate.get("recognizer", "OCR")
                self.ocr_review_candidates.addItem(f"{name}: {candidate['text']}", candidate_index)
        self.ocr_review_candidates.setCurrentIndex(0)
        self.ocr_review_candidates.blockSignals(False)
        self.ocr_review_edit.setText(box.text)

        annotated = OCREngine.draw_boxes(image, result, color=(108, 92, 231), thickness=2, show_text=False)
        cv2.polylines(annotated, [points.reshape((-1, 1, 2))], True, (0, 0, 255), 4)
        self._show_image(annotated)

    def _on_review_candidate_selected(self, combo_index: int) -> None:
        if combo_index <= 0:
            return
        candidate_index = self.ocr_review_candidates.itemData(combo_index)
        box_index = self._selected_review_box_index()
        if box_index is None or self._current_result is None or not isinstance(candidate_index, int):
            return
        candidates = self._current_result.boxes[box_index].metadata.get("candidates", [])
        if 0 <= candidate_index < len(candidates):
            self.ocr_review_edit.setText(candidates[candidate_index].get("text", ""))

    def _cancel_review_edit(self) -> None:
        box_index = self._selected_review_box_index()
        if box_index is not None and self._current_result is not None:
            self.ocr_review_edit.setText(self._current_result.boxes[box_index].text)
        self.ocr_review_candidates.setCurrentIndex(0)

    def _confirm_review_edit(self) -> None:
        box_index = self._selected_review_box_index()
        if box_index is None or self._structured_editor is None or self._current_result is None:
            return
        new_text = self.ocr_review_edit.text()
        if new_text == self._current_result.boxes[box_index].text:
            self._review_resolved_boxes.add(box_index)
            self._refresh_ocr_review_list(selected_box=box_index)
            self._set_status("Đã xác nhận giữ nguyên nội dung vùng OCR.")
            return
        combo_index = self.ocr_review_candidates.currentIndex()
        candidate_index = self.ocr_review_candidates.itemData(combo_index) if combo_index > 0 else None
        try:
            if isinstance(candidate_index, int):
                candidates = self._current_result.boxes[box_index].metadata.get("candidates", [])
                if candidates[candidate_index].get("text") == new_text:
                    self._structured_editor.apply_candidate(box_index, candidate_index)
                else:
                    self._structured_editor.apply_box_correction(box_index, new_text)
            else:
                self._structured_editor.apply_box_correction(box_index, new_text)
        except (EditOwnershipError, ValueError, IndexError) as exc:
            self._set_status(f"Không áp dụng được sửa đổi: {exc}")
            return
        self._review_resolved_boxes.add(box_index)
        self._structured_saved = False
        self._publish_structured_edit(box_index)

    def _publish_structured_edit(self, selected_box: int) -> None:
        if self._structured_editor is None:
            return
        result = self._structured_editor.result
        self._current_result = result
        self.btn_review_undo.setEnabled(self._structured_editor.can_undo)
        self.btn_review_redo.setEnabled(self._structured_editor.can_redo)
        self._refresh_ocr_review_list(selected_box=selected_box)
        self._create_document_session(result)
        self._set_status("Đã áp dụng sửa đổi có cấu trúc. Lưu Word để ghi ra tệp.")

    def _undo_structured_edit(self) -> None:
        if self._structured_editor is None:
            return
        record = self._structured_editor.undo()
        if record is None:
            return
        self._review_resolved_boxes.discard(record.target.box_index)
        self._structured_saved = False
        self._publish_structured_edit(record.target.box_index)

    def _redo_structured_edit(self) -> None:
        if self._structured_editor is None:
            return
        record = self._structured_editor.redo()
        if record is None:
            return
        self._review_resolved_boxes.add(record.target.box_index)
        self._structured_saved = False
        self._publish_structured_edit(record.target.box_index)

    def _select_next_review_box(self) -> None:
        result = self._current_result
        if result is None:
            return
        start = self._selected_review_box_index()
        order = list(range((start + 1) if start is not None else 0, len(result.boxes)))
        order.extend(range(0, (start + 1) if start is not None else 0))
        target = next((index for index in order
                       if result.boxes[index].metadata.get("needs_review")
                       and index not in self._review_resolved_boxes), None)
        if target is None:
            self._set_status("Không còn vùng nào được gắn cờ cần xem lại.")
            return
        for row in range(self.ocr_review_list.count()):
            if self.ocr_review_list.item(row).data(Qt.UserRole) == target:
                self.ocr_review_list.setCurrentRow(row)
                self.document_tabs.setCurrentWidget(self.ocr_review_list.parentWidget())
                return

    def eventFilter(self, watched, event) -> bool:
        if watched is self.label_image and event.type() == QEvent.MouseButtonRelease:
            pixmap = self.label_image.pixmap()
            image = self._current_image
            result = self._current_result
            if pixmap is not None and image is not None and result is not None and not pixmap.isNull():
                offset_x = (self.label_image.width() - pixmap.width()) // 2
                offset_y = (self.label_image.height() - pixmap.height()) // 2
                local = event.pos()
                px, py = local.x() - offset_x, local.y() - offset_y
                if 0 <= px < pixmap.width() and 0 <= py < pixmap.height():
                    source_x = px * image.shape[1] / pixmap.width()
                    source_y = py * image.shape[0] / pixmap.height()
                    matches = []
                    for index, box in enumerate(result.boxes):
                        polygon = np.asarray(box.bbox, dtype=np.float32)
                        if polygon.shape == (4, 2) and cv2.pointPolygonTest(
                            polygon, (float(source_x), float(source_y)), False
                        ) >= 0:
                            area = abs(cv2.contourArea(polygon))
                            matches.append((area, index))
                    if matches:
                        selected = min(matches)[1]
                        for row in range(self.ocr_review_list.count()):
                            if self.ocr_review_list.item(row).data(Qt.UserRole) == selected:
                                self.ocr_review_list.setCurrentRow(row)
                                self.document_tabs.setCurrentWidget(self.ocr_review_list.parentWidget())
                                return True
        return super().eventFilter(watched, event)

    def _confirm_discard_unsaved_edits(self) -> bool:
        structured_unsaved = bool(
            self._structured_editor is not None
            and self._structured_editor.dirty
            and not self._structured_saved
        )
        if not structured_unsaved:
            return True
        answer = QMessageBox.question(
            self, "Thay đổi chưa lưu",
            "Có sửa đổi chưa lưu ra tệp Word. Bạn có muốn bỏ các sửa đổi này không?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        return answer == QMessageBox.Yes

    # ==================================================================
    # RESIZE — Cập nhật ảnh khi resize cửa sổ
    # ==================================================================

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._current_pixmap and not self._current_pixmap.isNull():
            if self.source_fit_mode:
                self._render_source_image()

    # ==================================================================
    # CLOSE — Cleanup
    # ==================================================================

    def closeEvent(self, event) -> None:
        if not self._confirm_discard_unsaved_edits():
            event.ignore()
            return
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
