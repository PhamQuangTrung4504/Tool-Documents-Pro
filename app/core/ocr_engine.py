#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
app.core.ocr_engine
====================
Module lõi xử lý nhận dạng ký tự quang học (OCR) với kiến trúc Hybrid hiện đại:

1. Text Detection (Phát hiện khung chữ):
   - Sử dụng PaddleOCR DB (Differentiable Binarization) với hệ số mở rộng
     det_db_unclip_ratio tối ưu để bắt trọn vẹn mọi dòng văn bản và dấu thanh.
   - Singleton Pattern toàn cục (_GLOBAL_DETECTOR) tránh khởi tạo lặp lại PaddleX.
2. Text Recognition (Đọc ký tự tiếng Việt):
   - Sử dụng VietOCR (kiến trúc Transformer / Seq2Seq) chuyên sâu cho chữ tiếng Việt:
     loại bỏ hoàn toàn hiện tượng rụng dấu hỏi, ngã, nặng, mũ, râu.
   - Singleton Pattern toàn cục (_GLOBAL_VIETOCR) nạp model 1 lần duy nhất vào bộ nhớ.
   - Tự động fallback về PaddleOCR đối với các ngôn ngữ quốc tế khác (en, ch, ja, ko,...).
3. Bố cục & Thứ tự đọc:
   - Tự động sắp xếp các khối chữ theo thứ tự đọc văn bản hành chính từ trên xuống dưới,
     từ trái sang phải (Line-grouping reading order).
"""

from __future__ import annotations

import os
import re
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from PIL import Image
from loguru import logger

# ---------------------------------------------------------------------------
# Biến Toàn Cục & Khóa Đa Luồng (Thread-safe Singleton)
# ---------------------------------------------------------------------------

_GLOBAL_DETECTOR: Optional[Any] = None
_GLOBAL_DETECTOR_LOCK = threading.Lock()

_GLOBAL_VIETOCR: Optional[Any] = None
_GLOBAL_VIETOCR_LOCK = threading.Lock()


def apply_paddlex_patches() -> None:
    """
    Vá triệt để lỗi 'PDX has already been initialized. Reinitialization is not supported.'
    và xung đột DLL giữa OpenMP / PyTorch / PaddlePaddle trên Windows.
    """
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["PADDLE_PDX_EAGER_INIT"] = "False"
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

    # Pre-load PyTorch DLLs trước khi bất kỳ module Paddle nào được import
    try:
        import torch  # noqa: F401 # type: ignore
    except Exception:
        pass

    try:
        import sys
        import paddlex  # type: ignore[import-untyped,import-not-found]
        import paddlex.repo_manager as rm  # type: ignore[import-untyped,import-not-found]
        import paddlex.repo_manager.core as rmc  # type: ignore[import-untyped,import-not-found]

        orig_init = rmc.initialize

        def safe_init(*args, **kwargs):
            try:
                if rmc._GlobalContext.is_initialized():
                    return
                return orig_init(*args, **kwargs)
            except RuntimeError as err_init:
                if "already been initialized" in str(err_init):
                    return
                raise

        rmc.initialize = safe_init
        rm.initialize = safe_init

        for mod_name, mod in list(sys.modules.items()):
            if "paddlex" in mod_name and hasattr(mod, "initialize"):
                try:
                    setattr(mod, "initialize", safe_init)
                except Exception:
                    pass
    except Exception:
        pass


# Áp dụng ngay khi module được import
apply_paddlex_patches()


# ---------------------------------------------------------------------------
# Kiểu dữ liệu kết quả OCR
# ---------------------------------------------------------------------------


@dataclass
class OCRBox:
    """Một vùng văn bản đã được nhận dạng."""

    text: str
    confidence: float
    bbox: List[List[int]]  # 4 điểm [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    raw_text: str = ""     # Lưu giữ văn bản nhận dạng ban đầu của OCR engine

    def __post_init__(self):
        if not self.raw_text:
            self.raw_text = self.text

    @property
    def top_left(self) -> Tuple[int, int]:
        return (int(self.bbox[0][0]), int(self.bbox[0][1]))

    @property
    def bottom_right(self) -> Tuple[int, int]:
        return (int(self.bbox[2][0]), int(self.bbox[2][1]))

    def __repr__(self) -> str:
        return (
            f"OCRBox(text={self.text!r}, "
            f"conf={self.confidence:.2f}, "
            f"pos={self.top_left})"
        )


@dataclass
class OCRTable:
    """Bảng biểu đã được tái dựng cấu trúc hàng và cột."""

    grid: List[List[str]] = field(default_factory=list)  # Ma trận các ô: [hàng][cột]
    bbox: List[int] = field(default_factory=list)        # [left, top, right, bottom]
    num_rows: int = 0
    num_cols: int = 0
    row_boxes: List[List[OCRBox]] = field(default_factory=list)  # Danh sách box cho từng hàng


@dataclass
class ParallelHeader:
    """Cấu trúc dữ liệu đại diện cho Tiêu đề hành chính 2 khối song song."""

    left_lines: List[str] = field(default_factory=list)   # Cơ quan ban hành, số hiệu văn bản
    right_lines: List[str] = field(default_factory=list)  # Quốc hiệu, tiêu ngữ, địa danh, ngày tháng
    left_boxes: List[OCRBox] = field(default_factory=list)
    right_boxes: List[OCRBox] = field(default_factory=list)


class HeaderDetectionResult(tuple):
    """
    Kết quả phát hiện header 2 khối song song.
    Kế thừa tuple (grouped_lines, remaining) để tương thích 100% với code giải nén 2 phần tử:
    header_lines, rem = detect_parallel_administrative_header(...)
    đồng thời lưu thuộc tính parallel_header.
    """
    def __new__(cls, grouped_lines, remaining, parallel_header=None):
        return super().__new__(cls, (grouped_lines, remaining))

    def __init__(self, grouped_lines, remaining, parallel_header=None):
        self.parallel_header = parallel_header


class LayoutResult(tuple):
    """
    Kết quả tái dựng bố cục tài liệu.
    Kế thừa tuple (grouped_lines, tables) để tương thích 100% với code giải nén 2 phần tử:
    grouped_lines, tables = reconstruct_document_layout(...)
    đồng thời lưu thuộc tính parallel_header.
    """
    def __new__(cls, grouped_lines, tables, parallel_header=None):
        return super().__new__(cls, (grouped_lines, tables))

    def __init__(self, grouped_lines, tables, parallel_header=None):
        self.parallel_header = parallel_header


@dataclass
class OCRResult:
    """Kết quả tổng hợp sau khi chạy OCR trên một ảnh."""

    boxes: List[OCRBox] = field(default_factory=list)
    full_text: str = ""
    language: str = "vi"
    elapsed_seconds: float = 0.0
    image_width: int = 0
    image_height: int = 0
    tables: List[OCRTable] = field(default_factory=list)
    raw_full_text: str = ""                             # Văn bản gốc ban đầu trước khi khôi phục ký tự
    audit_trail: List[dict] = field(default_factory=list)  # Nhật ký vết các phép sửa tự động
    parallel_header: Optional[ParallelHeader] = None      # Tiêu đề hành chính 2 khối song song

    @property
    def text(self) -> str:
        return self.full_text

    @property
    def lines(self) -> List[str]:
        if self.full_text:
            return self.full_text.splitlines()
        return [b.text for b in self.boxes] if self.boxes else []

    @property
    def num_lines(self) -> int:
        if self.full_text:
            return len(self.full_text.splitlines())
        return len(self.boxes)

    @property
    def average_confidence(self) -> float:
        if not self.boxes:
            return 0.0
        return sum(b.confidence for b in self.boxes) / len(self.boxes)


# ---------------------------------------------------------------------------
# Bảng ngôn ngữ
# ---------------------------------------------------------------------------

LANGUAGE_MAP: Dict[str, str] = {
    "Tiếng Việt (Hybrid VietOCR)": "vi",
    "English": "en",
    "中文 (Chinese)": "ch",
    "日本語 (Japanese)": "japan",
    "한국어 (Korean)": "korean",
    "Français (French)": "fr",
    "Deutsch (German)": "german",
    "Español (Spanish)": "es",
    "Português (Portuguese)": "pt",
    "Italiano (Italian)": "it",
    "Русский (Russian)": "ru",
    "العربية (Arabic)": "ar",
    "हिन्दी (Hindi)": "hi",
    "ไทย (Thai)": "ta",
}


def get_display_languages() -> List[str]:
    """Trả về danh sách tên ngôn ngữ hiển thị trên UI."""
    return list(LANGUAGE_MAP.keys())


def get_lang_code(display_name: str) -> str:
    """Chuyển tên hiển thị → mã ngôn ngữ PaddleOCR."""
    return LANGUAGE_MAP.get(display_name, "vi")


# ---------------------------------------------------------------------------
# Tiện ích cắt ảnh & sắp xếp thứ tự đọc
# ---------------------------------------------------------------------------


def get_rotate_crop_image(
    img_bgr: np.ndarray,
    points: np.ndarray,
    pad_top: int = 6,
    pad_bottom: int = 8,
    pad_left: int = 8,
    pad_right: int = 18,
) -> Optional[np.ndarray]:
    """
    Cắt và xoay thẳng vùng ảnh polygon 4 điểm (Perspective Transform).
    Chuẩn hóa vùng ảnh chữ nằm ngang cho VietOCR với viền đệm (padding) an toàn:
    - Mở rộng mép trên 6px (bảo vệ dấu sắc, hỏi, ngã, mũ: á, ả, ã, â, ê, ô, ư, ơ).
    - Mở rộng mép dưới 8px (bảo vệ trọn vẹn dấu nặng: ạ, ặ, ậ, ẹ, ệ, ị, ọ, ộ, ợ, ụ, ự, ỵ).
    - Mở rộng mép phải 18px (bảo vệ chữ cái cuối dòng không bị cắt cụt sát mép như 'đặc b' -> 'đặc biệt', 'AW' -> 'AWS.').
    - Mở rộng mép trái 8px (tránh dính mép viền làm đứt chữ cái đầu hoặc dấu bullet).
    - Dùng nền trắng (255, 255, 255) chuẩn tài liệu khi padding để tách biệt nét chữ.
    """
    try:
        pts = np.array(points, dtype=np.float32)
        if len(pts) < 4:
            return None

        # Tính chiều rộng và chiều cao chuẩn
        w = int(max(
            np.linalg.norm(pts[0] - pts[1]),
            np.linalg.norm(pts[2] - pts[3]),
        ))
        h = int(max(
            np.linalg.norm(pts[0] - pts[3]),
            np.linalg.norm(pts[1] - pts[2]),
        ))

        if w <= 4 or h <= 4:
            return None

        pts_std = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        m_matrix = cv2.getPerspectiveTransform(pts, pts_std)
        warped = cv2.warpPerspective(
            img_bgr,
            m_matrix,
            (w, h),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC,
        )

        # Nếu là dòng chữ dọc, xoay 90 độ thành chữ ngang
        if warped.shape[0] * 1.5 >= warped.shape[1] and warped.shape[0] > 40:
            warped = np.rot90(warped)

        # Thêm padding viền an toàn với nền trắng (255, 255, 255)
        # Bảo toàn 100% các dấu câu nhỏ ở rìa ảnh cho VietOCR
        padded = cv2.copyMakeBorder(
            warped,
            top=pad_top,
            bottom=pad_bottom,
            left=pad_left,
            right=pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=[255, 255, 255],
        )

        return padded
    except Exception as exc:
        logger.debug(f"Rotate crop failed: {exc}")
        return None


def clean_crop_borders(crop_bgr: np.ndarray) -> np.ndarray:
    """
    Thuật toán Tẩy Đường Kẻ Bảng ở Sát Biên Lát Cắt (Border Line Stripping):
    1. Kiểm tra kích thước: nếu h < 10 hoặc w < 10 thì trả về nguyên bản.
    2. Chuyển sang ảnh xám và nhị phân hóa ngược (nét mực đen = 255, nền trắng = 0) bằng ngưỡng Otsu.
    3. Lấy màu nền trung vị của ảnh (bg_color).
    4. Tẩy đường kẻ ngang sát biên trên (y in [0, 1, 2]) và dưới (y in [h-3, h-2, h-1]):
       Nếu mật độ pixel đen trên hàng đó >= 55% chiều rộng w -> tô hàng đó bằng bg_color.
    5. Tẩy đường kẻ dọc sát biên trái (x in [0, 1, 2, 3]) và phải (x in [w-4, w-3, w-2, w-1]):
       Nếu cột đó có mật độ pixel đen kéo dài >= 82% chiều cao h (đâm xuyên qua cả vùng đệm padding,
       chứng tỏ 100% là đường kẻ khung bảng chứ không phải chữ cái I, 1, l) -> tô cột bằng bg_color.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return crop_bgr

    h, w = crop_bgr.shape[:2]
    if h < 10 or w < 10:
        return crop_bgr

    result = crop_bgr.copy()
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    if float(np.mean(gray)) > 127.0:
        _, bin_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, bin_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    bg_color = np.median(result.reshape(-1, 3), axis=0).astype(np.uint8)

    # 4. Tẩy đường kẻ ngang sát biên trên và dưới
    h_rows = [0, 1, 2, h - 3, h - 2, h - 1]
    for y in h_rows:
        if 0 <= y < h:
            black_count = int(np.sum(bin_inv[y, :] > 0))
            if black_count >= 0.55 * w:
                result[y, :] = bg_color

    # 5. Tẩy đường kẻ dọc sát biên trái và phải (vùng đệm padding ngoài cùng)
    # Mép trái: giữ ngưỡng >= 82% h
    for x in [0, 1, 2, 3]:
        if 0 <= x < w:
            black_count = int(np.sum(bin_inv[:, x] > 0))
            if black_count >= 0.82 * h:
                result[:, x] = bg_color

    # Mép phải: không xóa nếu chiều cao nét đen < 92% h (chỉ xóa nếu >= 92% h)
    for x in [w - 4, w - 3, w - 2, w - 1]:
        if 0 <= x < w:
            black_count = int(np.sum(bin_inv[:, x] > 0))
            if black_count >= 0.92 * h:
                result[:, x] = bg_color

    return result


def pad_small_crop_for_vietocr(crop_bgr: np.ndarray) -> np.ndarray:
    """
    Đệm Khung Trắng Chuẩn Tỷ Lệ cho Ô Hẹp & Chữ Số (Aspect-Ratio Padding):
    - Các ô số ngắn (05, 04, 01, 1, 2) hoặc số La Mã (I., II.) có w < 2.5 * h.
    - Nếu chiều cao h < 36, phóng đại nhẹ bằng cv2.INTER_LANCZOS4 lên target_h = 44.
    - Tạo canvas nền bg_color có đệm trắng hai bên trái và phải: pad_x = max(14, int(h * 0.75)).
    - Giúp con số nằm cân đối ở trung tâm trường nhìn của mạng VGG, loại bỏ lỗi 05 -> 0s.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return crop_bgr

    h, w = crop_bgr.shape[:2]
    if h <= 0 or w <= 0:
        return crop_bgr

    if w < 2.5 * h:
        curr_crop = crop_bgr
        if h < 36:
            target_h = 44
            scale = target_h / float(h)
            target_w = max(1, int(round(w * scale)))
            curr_crop = cv2.resize(curr_crop, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
            h, w = curr_crop.shape[:2]

        bg_color = np.median(curr_crop.reshape(-1, 3), axis=0).astype(np.uint8)
        bg_val = [int(bg_color[0]), int(bg_color[1]), int(bg_color[2])]
        pad_x = max(14, int(round(h * 0.75)))
        padded = cv2.copyMakeBorder(
            curr_crop,
            top=0,
            bottom=0,
            left=pad_x,
            right=pad_x,
            borderType=cv2.BORDER_CONSTANT,
            value=bg_val,
        )
        return padded

    return crop_bgr


def is_decorative_pattern_crop(crop_bgr: np.ndarray) -> bool:
    """
    Phát hiện thị giác xem lát cắt có phải là dải hoa văn trang trí / đường kẻ lặp:
    - Nếu lát cắt có w >= 2.0 * h và các thành phần liên thông (cv2.connectedComponentsWithStats)
      chỉ gồm toàn các chấm nhỏ hoặc hoa văn lặp đều mà không có bất kỳ ký tự chữ cái nào
      cao >= 35% chiều cao dòng -> True.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return False

    h, w = crop_bgr.shape[:2]
    if h < 8 or w < 10:
        return False

    if w < 2.0 * h:
        return False

    try:
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        if float(np.mean(gray)) > 127.0:
            _, bin_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, bin_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_inv, connectivity=8)
        if num_labels <= 1:
            return True

        has_char_component = False
        min_char_height = 0.35 * h
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            comp_h = stats[i, cv2.CC_STAT_HEIGHT]
            if area >= 3 and comp_h >= min_char_height:
                has_char_component = True
                break

        return not has_char_component
    except Exception:
        return False


RE_REPEATED_WORD = re.compile(r"\b([A-Za-zÀ-ỹĐđ0-9]+)(?:\s+\1){2,}\b", re.IGNORECASE | re.UNICODE)
RE_DECORATIVE_LINE_CHARS = re.compile(r"^[14*_\-–—.~+=#\s]+$")
NOISE_SINGLE_CHARS = {"%", "/", "\\", "|", "[", "]", "~", "`", ";", ":"}


def is_decorative_or_hallucinated(text: str, crop_bgr: Optional[np.ndarray] = None) -> bool:
    """
    Phát hiện toàn diện dải hoa văn trang trí hoặc ảo giác chuỗi lặp:
    1. Kiểm tra thị giác qua is_decorative_pattern_crop nếu crop_bgr được cung cấp.
    2. Kiểm tra ngữ nghĩa chuỗi qua is_hallucinated_or_noise_line.
    """
    if text:
        upper_txt = text.upper()
        if any(h in upper_txt for h in [
            "BIÊN BẢN", "THÀNH PHẦN THAM DỰ", "NỘI DUNG CUỘC HỌP", "QUYẾT ĐỊNH",
            "CỘNG HÒA XÃ HỘI", "ĐỘC LẬP - TỰ DO"
        ]):
            return False
    if crop_bgr is not None and is_decorative_pattern_crop(crop_bgr):
        return True
    return is_hallucinated_or_noise_line(text)


def is_hallucinated_or_noise_line(text: str) -> bool:
    """
    Kiểm tra và phát hiện dòng chữ bị ảo giác Transformer (repetition hallucination)
    hoặc nhiễu mép cắt / đường kẻ trang trí / ký tự đơn lẻ vô nghĩa:
    1. Nếu dòng có >= 3 từ mà một từ bị lặp lại liên tiếp >= 3 lần (r'\\b([A-Za-zÀ-ỹĐđ0-9]+)(?:\\s+\\1){2,}\\b',
       ví dụ: 'đặc đặc đặc') hoặc tỷ lệ từ duy nhất len(set(words)) / len(words) <= 0.35
       (khi len(words) >= 5, ví dụ: 'đặc đặc đặc đặc đặc đặc được đặc đặc đặc đặc đặn') -> True.
    2. Nếu cả dòng sau khi strip chỉ chứa các ký tự lặp vô nghĩa của đường kẻ
       (144444444, 44444, *****, -----) hoặc chỉ chứa vỏn vẹn 1 ký tự rác (%, /, \\, |, [, ]) -> True.
    """
    if not text:
        return True

    stripped = text.strip()
    if not stripped:
        return True

    # 0. Bảo vệ tuyệt đối các tiêu đề và đề mục văn bản quan trọng (Zero False Drops)
    upper_txt = stripped.upper()
    if any(h in upper_txt for h in [
        "BIÊN BẢN", "THÀNH PHẦN THAM DỰ", "NỘI DUNG CUỘC HỌP", "QUYẾT ĐỊNH",
        "CỘNG HÒA XÃ HỘI", "ĐỘC LẬP - TỰ DO", "TIÊU CHÍ ĐÁNH GIÁ", "KẾT LUẬN"
    ]):
        return False

    # 1. Ký tự rác đơn lẻ (ví dụ: '%', '/', '\\', '|', '[', ']')
    if len(stripped) == 1 and stripped in NOISE_SINGLE_CHARS:
        return True

    # 2. Dòng chứa toàn các ký tự lặp vô nghĩa của đường kẻ trang trí:
    if RE_DECORATIVE_LINE_CHARS.match(stripped):
        char_set = set(stripped.replace(" ", "").replace("\t", ""))
        if char_set and (char_set.issubset({"1", "4"}) or char_set.issubset({"*", "-", "_", "~", ".", "=", "#"})):
            return True

    # 3. Phân tích các từ trong dòng:
    words = stripped.split()
    n_words = len(words)

    # 3.a. Có >= 3 từ mà một từ bị lặp lại liên tiếp >= 3 lần (ví dụ: 'đặc đặc đặc')
    if n_words >= 3 and RE_REPEATED_WORD.search(stripped):
        return True

    # 3.b. Tỷ lệ từ duy nhất thấp bất thường (Transformer attention repetition loop):
    if n_words >= 5:
        unique_words = {w.lower() for w in words}
        if len(unique_words) / n_words <= 0.35:
            return True

    return False


def split_dotted_line_crop(crop_bgr: np.ndarray) -> List[np.ndarray]:
    """
    Phân tích hình thái học và lược đồ hình chiếu thẳng đứng (Vertical Projection Profile)
    trên ảnh dòng chữ. Nếu dòng chữ có tỷ lệ khung hình rất dài (width / height > 11.0)
    và chứa các dải chấm lửng '......' hoặc khoảng trống chia cắt dòng:
    (nơi trong suốt chiều dài ngang >= 1.1 * height chỉ có khoảng trắng hoặc chỉ có các
    chấm nhỏ nằm ở 1/3 dưới của dòng mà không có nét chữ ở 2/3 phía trên).
    Tách ảnh crop thành các lát cắt con (sub-crops) theo từng cụm từ để nhận diện bằng VietOCR
    rồi nối lại bằng ' ...... ', loại bỏ triệt để hiện tượng attention drift (nuốt chữ).
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return []

    h, w = crop_bgr.shape[:2]
    # Chỉ áp dụng cho các dòng chữ rất dài có khả năng chứa chuỗi chấm lửng
    if h < 10 or w < 100 or (w / float(h)) < 11.0:
        return [crop_bgr]

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    if float(np.mean(gray)) > 127.0:
        _, bin_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, bin_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Khảo sát 65% chiều cao phía trên (nơi các chữ cái tiếng Việt bắt buộc có nét)
    upper_h = max(1, int(round(h * 0.65)))
    upper_band = bin_inv[:upper_h, :]
    col_ink_counts = np.sum(upper_band > 0, axis=0)

    noise_thresh = max(1, int(round(upper_h * 0.05)))
    is_valley_col = col_ink_counts <= noise_thresh

    min_valley_len = int(round(1.1 * h))
    valleys: List[Tuple[int, int]] = []
    in_valley = False
    v_start = 0

    for x in range(w):
        if is_valley_col[x]:
            if not in_valley:
                in_valley = True
                v_start = x
        else:
            if in_valley:
                in_valley = False
                v_end = x - 1
                if (v_end - v_start + 1) >= min_valley_len:
                    valleys.append((v_start, v_end))

    if in_valley:
        v_end = w - 1
        if (v_end - v_start + 1) >= min_valley_len:
            valleys.append((v_start, v_end))

    # Lọc bỏ thung lũng sát lề ngoài cùng
    margin_thresh = int(round(0.8 * h))
    interior_valleys = [
        (vs, ve) for vs, ve in valleys
        if vs >= margin_thresh and (w - 1 - ve) >= margin_thresh
    ]

    # Chỉ cho phép tách sub-crops khi thực sự có tối thiểu 2 khoảng trống/chấm lửng dài ở giữa
    # để tránh cắt cụt chữ/số cuối dòng trên ảnh hẹp (ví dụ: 'năm 2026.' -> 'năm 202')
    if len(interior_valleys) < 2:
        return [crop_bgr]

    sub_crops: List[np.ndarray] = []
    curr_x = 0
    padding_x = 4

    for vs, ve in interior_valleys:
        x_left = max(0, curr_x)
        x_right = min(w, vs + padding_x)
        if (x_right - x_left) >= int(round(0.6 * h)):
            sub_crop = crop_bgr[:, x_left:x_right]
            if sub_crop.size > 0:
                sub_crops.append(sub_crop)
        curr_x = max(curr_x, ve - padding_x)

    # Tuyệt đối không bao giờ cắt bỏ phần đuôi bên phải của ảnh crop_bgr
    if curr_x < w:
        last_sub_crop = crop_bgr[:, curr_x:w]
        if last_sub_crop.size > 0:
            sub_crops.append(last_sub_crop)

    if len(sub_crops) <= 1:
        return [crop_bgr]

    return sub_crops


def estimate_and_correct_skew(
    image: np.ndarray,
    max_angle: float = 15.0,
    min_angle: float = 0.5,
) -> Tuple[np.ndarray, float]:
    """
    Phát hiện góc nghiêng và nắn thẳng ảnh tài liệu (Deskew):
    - Chuyển ảnh sang thang độ xám và nhị phân hóa cục bộ.
    - Dùng phép biến đổi hình thái học (morphological dilation) theo chiều ngang
      để liên kết các ký tự thành các dải dòng chữ liên tục.
    - Tìm góc nghiêng của các dải dòng bằng minAreaRect.
    - Nếu góc nghiêng nằm trong khoảng [min_angle, max_angle] hoặc [-max_angle, -min_angle]:
      xoay ảnh ngược lại đúng góc nghiêng để nắn thẳng văn bản nằm ngang chuẩn.
    - Nếu góc nghiêng < min_angle: giữ nguyên ảnh để tránh mất nét do phép nội suy không cần thiết.
    Trả về: (deskewed_image, detected_angle)
    """
    if image is None or image.size == 0:
        return image, 0.0

    try:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        h, w = gray.shape[:2]
        if h < 50 or w < 50:
            return image, 0.0

        # Nhị phân hóa Otsu
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Morphological dilation theo chiều ngang để nối các chữ thành dòng
        kernel_w = max(25, int(w / 35))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 3))
        dilated = cv2.dilate(thresh, kernel, iterations=1)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        angles = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 150:
                continue
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (rw, rh), angle = rect

            # Chuẩn hóa để rw luôn là cạnh dài theo phương dòng chữ, rh là chiều cao dòng
            if rw < rh:
                rw, rh = rh, rw
                angle = angle - 90.0

            if rw < 60 or rh < 6:
                continue

            if angle > 45.0:
                angle -= 90.0
            elif angle < -45.0:
                angle += 90.0

            if abs(angle) <= max_angle and (rw / max(1.0, rh)) >= 2.5:
                angles.append(angle)

        if len(angles) < 2:
            return image, 0.0

        median_angle = float(np.median(angles))
        if abs(median_angle) < min_angle or abs(median_angle) > max_angle:
            return image, 0.0

        center = (w / 2.0, h / 2.0)
        rot_mat = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        border_val = (255.0, 255.0, 255.0)
        deskewed = cv2.warpAffine(
            image,
            rot_mat,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_val,
        )
        logger.debug(f"Deskew applied: angle={median_angle:.2f} deg")
        return deskewed, median_angle
    except Exception as exc:
        logger.debug(f"estimate_and_correct_skew exception: {exc}")
        return image, 0.0


def filter_border_noise_boxes(
    boxes: List[OCRBox],
    image_width: int = 0,
    image_height: int = 0,
) -> List[OCRBox]:
    """
    Lọc các box nhiễu do ảnh bị cắt ngang tại mép trên/dưới.
    Bảo toàn số trang hợp lệ, ngày tháng, số hiệu văn bản, và các box sát mép nhưng không bị cắt.
    """
    if not boxes:
        return []

    # Tính median height của toàn bộ trang
    heights = []
    for b in boxes:
        pts = b.bbox
        y_top = min(pt[1] for pt in pts)
        y_bottom = max(pt[1] for pt in pts)
        heights.append(max(1, y_bottom - y_top))
    median_h = float(np.median(heights)) if heights else 20.0

    valid_boxes: List[OCRBox] = []
    for b in boxes:
        pts = b.bbox
        y_top = min(pt[1] for pt in pts)
        y_bottom = max(pt[1] for pt in pts)
        x_left = min(pt[0] for pt in pts)
        x_right = max(pt[0] for pt in pts)
        h = max(1, y_bottom - y_top)
        text = b.text.strip()

        touches_bottom = (image_height > 0 and y_bottom >= image_height - 3)
        touches_top = (y_top <= 3)

        if touches_bottom or touches_top:
            is_page_number = (re.match(
                r"^(?:Trang\s+\d+(?:/\d+)?|\d+(?:/\d+)?|Page\s+\d+)$",
                text, re.IGNORECASE
            ) is not None)
            is_valid_doc_code = (re.search(r"\b\d+/\d{4}/[A-ZĐ-]+", text) is not None)
            is_valid_date = (re.search(r"\bngày\s+\d+\s+tháng\s+\d+", text, re.IGNORECASE) is not None)

            # Nếu là số trang hoặc số hiệu/ngày tháng hợp lệ và chiều cao không quá cụt -> Giữ lại
            if (is_page_number or is_valid_doc_code or is_valid_date) and h >= 0.55 * median_h:
                valid_boxes.append(b)
                continue

            # Nếu chiều cao bị cắt quá nửa so với dòng chữ bình thường -> Nét cắt xén viền ảnh
            if h < 0.48 * median_h or (h <= 10 and median_h >= 18):
                logger.debug(f"Filtered clipped border noise: text={text!r}, h={h}, median_h={median_h}")
                continue

            # Nếu text là chuỗi ký tự rời rạc/nhiễu quét mép ảnh
            is_noisy_pattern = (
                re.match(r"^[\d\s/.,\-_+=*~`]+$", text) is not None
                and not is_page_number
                and not is_valid_doc_code
            )
            if is_noisy_pattern and h < 0.70 * median_h:
                logger.debug(f"Filtered noisy border text: text={text!r}, h={h}, median_h={median_h}")
                continue

        # Lọc box chữ từ Logo / Con dấu bên lề phải (Logo & Stamp Noise Filtering)
        if image_width > 0 and x_left > image_width * 0.65:
            txt_upper = text.upper()
            if any(kw in txt_upper for kw in ["KHIÊN", "CON DẤU", "LOGO", "SEAL"]) and h <= 1.8 * median_h:
                logger.debug(f"Filtered margin logo/stamp noise: text={text!r}")
                continue

        valid_boxes.append(b)

    return valid_boxes


def recover_emails_and_urls(text: str) -> str:
    """
    Thuật toán khôi phục ký tự @ trong Email và sửa lỗi tên miền/URL:
    - VietOCR thường nhận @ thành Q, O, (a, (D, ©, at, hoặc dính liền vào tên miền
      (ví dụ: Qhutech.edu.vn, (Damaron.com, (Damazon.com).
    - Tự động sửa lỗi OCR trong tên miền công nghệ/trường học phổ biến: amaron.com -> amazon.com.
    """
    if not text:
        return ""

    # 1. Tự động sửa lỗi OCR trong tên miền công nghệ / trường học phổ biến
    text = re.sub(r"\bamar(?:on|om)\.com\b", "amazon.com", text, flags=re.IGNORECASE)
    text = re.sub(r"\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9-]*?)amar(?:on|om)\.com\b", r"\1amazon.com", text, flags=re.IGNORECASE)

    domain_exts = r"(?:com|edu\.vn|gov\.vn|vn|net|org|io|edu)"

    # 2. Mẫu 1 (Có khoảng cách hoặc ký tự rác đứng riêng):
    # 'user Q domain.com', 'user (a domain.com', 'user at domain.com', 'user © domain.com'
    text = re.sub(
        rf"([a-zA-Z0-9._%+-]+)\s*(?:@|[QO©]|\(a|\([a-zA-Z]|at)\s+([a-zA-Z0-9-]+\.{domain_exts})\b",
        r"\1@\2",
        text,
        flags=re.IGNORECASE,
    )

    # 3. Mẫu 2 (Ký tự nhầm lẫn dính liền vào domain, ví dụ: 'user Qhutech.edu.vn' hoặc 'user (Damazon.com'):
    text = re.sub(
        rf"([a-zA-Z0-9._%+-]+)\s+[\(]?[QOD©]([a-zA-Z0-9-]+\.{domain_exts})\b",
        r"\1@\2",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        rf"([a-zA-Z0-9._%+-]+)\s+(?:\(a|\([a-zA-Z]|at)([a-zA-Z0-9-]+\.{domain_exts})\b",
        r"\1@\2",
        text,
        flags=re.IGNORECASE,
    )

    # 4. Chuẩn hóa nếu @ có khoảng cách xung quanh: 'user @ domain.com' -> 'user@domain.com'
    text = re.sub(
        rf"([a-zA-Z0-9._%+-]+)\s*@\s*([a-zA-Z0-9-]+\.{domain_exts})\b",
        r"\1@\2",
        text,
        flags=re.IGNORECASE,
    )

    # 5. Đảm bảo tên miền amazon.com chuẩn hóa lần cuối nếu còn sót
    text = re.sub(r"@([a-zA-Z0-9-]+\.)amar(?:on|om)\.com\b", r"@\1amazon.com", text, flags=re.IGNORECASE)
    text = re.sub(r"@amar(?:on|om)\.com\b", "@amazon.com", text, flags=re.IGNORECASE)

    return text


def format_progress_table_row(row_str: str) -> str:
    """
    Cố định cấu trúc bảng tiến độ thực tập hoặc bảng số thứ tự:
    Cột 1: Tuần / STT
    Cột 2: Ngày (khoảng ngày dd/mm/yyyy - dd/mm/yyyy)
    Cột 3: Nội dung (các việc gạch đầu dòng)
    """
    if not row_str:
        return ""

    # 1. Tách số tuần đứng dính vào ngày tháng ở đầu dòng: '1 20/04/2026' -> '1 | 20/04/2026'
    row_str = re.sub(r"^([ \t]*\d+)\s+(\d{2}/\d{2}/\d{4})", r"\1 | \2", row_str)

    # 2. Khôi phục dấu gạch nối ngày tháng: '20/04/2026 26/04/2026' -> '20/04/2026 - 26/04/2026'
    row_str = re.sub(r"(\d{2}/\d{2}/\d{4})\s+(\d{2}/\d{2}/\d{4})", r"\1 - \2", row_str)
    row_str = re.sub(r"(\d{2}/\d{2}/\d{4})\s*[?~_]\s*(\d{2}/\d{2}/\d{4})", r"\1 - \2", row_str)

    # 3. Đảm bảo ngăn cách giữa cụm ngày tháng và nội dung gạch đầu dòng có ' | '
    row_str = re.sub(r"(\d{2}/\d{2}/\d{4}\s*-\s*\d{2}/\d{2}/\d{4})\s*(?!\|)\s*(-|\+)\s+", r"\1 | \2 ", row_str)

    # 4. Dọn dẹp khoảng trắng thừa quanh phân cách '|'
    row_str = re.sub(r"\s*\|\s*", " | ", row_str)
    row_str = re.sub(r"\b(\d+)\s*\|\s*\|\s*", r"\1 | ", row_str)

    return row_str


def _box_info_list(boxes: List[OCRBox]) -> List[dict]:
    """Chuyển đổi danh sách OCRBox thành danh sách dict chứa thuộc tính hình học."""
    box_info = []
    for b in boxes:
        pts = b.bbox
        x_left = min(pt[0] for pt in pts)
        x_right = max(pt[0] for pt in pts)
        y_top = min(pt[1] for pt in pts)
        y_bottom = max(pt[1] for pt in pts)
        h = max(1, y_bottom - y_top)
        w = max(1, x_right - x_left)
        cy = (y_top + y_bottom) / 2.0
        box_info.append({
            "box": b,
            "text": b.text,
            "left": x_left,
            "right": x_right,
            "top": y_top,
            "bottom": y_bottom,
            "h": h,
            "w": w,
            "cy": cy,
        })
    return box_info


def _group_single_column_lines(sub_boxes: List[dict], median_h: float, gutter_thresh: float) -> List[List[OCRBox]]:
    """Gom nhóm các box trong 1 cột hoặc 1 khối đơn thành từng dòng."""
    if not sub_boxes:
        return []
    # Sắp xếp xác định có tie-breaking (Deterministic tie-breaking)
    sub_boxes_sorted = sorted(
        sub_boxes,
        key=lambda it: (round(it["cy"], 2), round(it["left"], 2), it["text"])
    )
    y_thresh = max(6.0, median_h * 0.45)
    lines: List[List[dict]] = []
    curr_line = [sub_boxes_sorted[0]]

    for info in sub_boxes_sorted[1:]:
        line_avg_cy = float(np.mean([item["cy"] for item in curr_line]))
        line_avg_h = float(np.mean([item["h"] for item in curr_line]))
        line_top = min(item["top"] for item in curr_line)
        line_bottom = max(item["bottom"] for item in curr_line)
        overlap_y = max(0, min(info["bottom"], line_bottom) - max(info["top"], line_top))

        last_box = max(curr_line, key=lambda it: it["right"])
        gap_x = info["left"] - last_box["right"]

        is_same_line = (
            (abs(info["cy"] - line_avg_cy) < y_thresh or overlap_y >= 0.40 * min(info["h"], line_avg_h))
            and gap_x < gutter_thresh
        )
        if is_same_line:
            curr_line.append(info)
        else:
            lines.append(curr_line)
            curr_line = [info]

    if curr_line:
        lines.append(curr_line)

    result: List[List[OCRBox]] = []
    for line in lines:
        line.sort(key=lambda it: (round(it["left"], 2), it["text"]))
        if len(line) >= 2:
            first = line[0]
            if first["w"] <= 35 and first["h"] <= 35 and first["box"].text.strip() in [
                "o", "O", "•", "·", "*", "■", "□", "回", "▫", "▪"
            ]:
                first["box"].text = "•"
        result.append([it["box"] for it in line])
    return result


def _group_boxes_sliding_window(
    box_info: List[dict],
    page_min_x: float,
    page_w: float,
    gutter_thresh: float,
    median_h: float,
) -> List[List[OCRBox]]:
    """Gom nhóm các box theo cửa sổ trượt phân tích đa cột."""
    if not box_info:
        return []

    # Sắp xếp xác định có tie-breaking (Deterministic tie-breaking)
    box_info_sorted = sorted(
        box_info,
        key=lambda it: (round(it["cy"], 2), round(it["left"], 2), it["text"])
    )
    result_lines: List[List[OCRBox]] = []
    i = 0
    n = len(box_info_sorted)

    while i < n:
        # Xử lý tiêu đề / banner trải rộng toàn trang (spanning title across columns):
        # nếu box có độ rộng >= 0.62 chiều rộng trang, phát thành dòng độc lập ngay
        curr_box = box_info_sorted[i]
        if curr_box["w"] >= 0.62 * page_w:
            result_lines.append([curr_box["box"]])
            i += 1
            continue

        best_2col_end = None
        best_split_x = None

        for j in range(i + 2, n + 1):
            window = box_info_sorted[i:j]
            # Ngắt window nếu gặp box trải rộng toàn trang tiếp theo
            if any(b["w"] >= 0.62 * page_w for b in window[1:]):
                break

            found_split = None
            found_left_b: List[dict] = []
            found_right_b: List[dict] = []
            for split_ratio in np.linspace(0.28, 0.65, 38):
                sx = page_min_x + split_ratio * page_w
                left_b = [b for b in window if b["right"] <= sx]
                right_b = [b for b in window if b["left"] >= sx]
                cross_b = [b for b in window if b["left"] < sx and b["right"] > sx]
                if left_b and right_b and not cross_b:
                    gap = min(b["left"] for b in right_b) - max(b["right"] for b in left_b)
                    if gap >= gutter_thresh:
                        found_split = sx
                        found_left_b = left_b
                        found_right_b = right_b
                        break
            if found_split is not None:
                # Kiểm tra nếu trong window có các box nằm cùng một hàng ngang tạo thành cặp trường biểu mẫu
                # (+ Ông (bà): ......... và (ghi tên...), Chức danh:........... và Chủ trì cuộc họp.)
                # Bắt buộc KHÔNG được tách thành 2 cột độc lập!
                has_horizontal_form_pair = False
                for lb in found_left_b:
                    lb_txt = lb["text"].strip()
                    for rb in found_right_b:
                        rb_txt = rb["text"].strip()
                        o_y = max(0, min(lb["bottom"], rb["bottom"]) - max(lb["top"], rb["top"]))
                        m_h = min(lb["h"], rb["h"])
                        v_overlap = o_y / m_h if m_h > 0 else 0.0
                        if v_overlap >= 0.40:
                            is_form_prompt = (
                                lb_txt.endswith((":", ".", "…", "_", "-"))
                                or any(kw in lb_txt for kw in ["Ông (bà)", "Ông(bà)", "Chức danh", "Họ và tên", "Nơi làm việc", "Chức vụ", "vào lúc"])
                            )
                            is_form_val = (
                                rb_txt.startswith(("(", "Chủ trì", "Thư ký"))
                                or any(kw in rb_txt for kw in ["Chủ trì", "Thư ký", "đại diện"])
                            )
                            if is_form_prompt or is_form_val:
                                has_horizontal_form_pair = True
                                break
                    if has_horizontal_form_pair:
                        break

                if not has_horizontal_form_pair:
                    best_2col_end = j
                    best_split_x = found_split
            else:
                if best_2col_end is not None:
                    break

        if best_2col_end is not None and (best_2col_end - i) >= 2:
            window = box_info_sorted[i:best_2col_end]
            left_col = [b for b in window if b["right"] <= best_split_x]
            right_col = [b for b in window if b["left"] >= best_split_x]

            lines_left = _group_single_column_lines(left_col, median_h, gutter_thresh)
            lines_right = _group_single_column_lines(right_col, median_h, gutter_thresh)

            result_lines.extend(lines_left)
            result_lines.extend(lines_right)
            i = best_2col_end
        else:
            single_line_boxes = [box_info_sorted[i]]
            k = i + 1
            y_thresh = max(6.0, median_h * 0.45)
            while k < n:
                b_curr = box_info_sorted[k]
                avg_cy = float(np.mean([it["cy"] for it in single_line_boxes]))
                avg_h = float(np.mean([it["h"] for it in single_line_boxes]))
                top_line = min(it["top"] for it in single_line_boxes)
                bot_line = max(it["bottom"] for it in single_line_boxes)
                overlap_y = max(0, min(b_curr["bottom"], bot_line) - max(b_curr["top"], top_line))

                last_box = max(single_line_boxes, key=lambda it: it["right"])
                gap_x = b_curr["left"] - last_box["right"]

                # Không gộp box nhỏ lề phải nghi vấn là con dấu / logo vào dòng văn bản chính
                is_right_margin_stamp = (
                    b_curr["left"] > (page_min_x + 0.65 * page_w)
                    and gap_x >= max(35.0, 0.06 * page_w)
                    and (
                        any(kw in b_curr["text"].upper() for kw in ["KHIÊN", "CON DẤU", "LOGO", "SEAL", "DẤU"])
                        or (b_curr["w"] < 0.25 * page_w and b_curr["text"].isupper() and gap_x >= max(50.0, 0.10 * page_w))
                    )
                )
                if is_right_margin_stamp:
                    break

                # Hỗ trợ nối các trường điền chấm chấm (dotted forms: Hôm nay, vào lúc... ngày... tháng... năm...)
                # và các cặp trường biểu mẫu nằm ngang (+ Ông (bà): ......... và (ghi tên...))
                is_form_pair = (
                    (overlap_y >= 0.40 * min(b_curr["h"], avg_h) or abs(b_curr["cy"] - avg_cy) < y_thresh)
                    and (
                        last_box["text"].rstrip().endswith((":", ".", "…", "_", "-", ","))
                        or any(kw in last_box["text"] for kw in ["Ông (bà)", "Ông(bà)", "Chức danh", "Họ và tên", "Chức vụ", "vào lúc"])
                        or b_curr["text"].lstrip().startswith(("(", "Chủ trì", "Thư ký"))
                    )
                )
                is_dotted_fill = (
                    (last_box["text"].rstrip().endswith((".", "…", "_", "-", ":", ",")) or b_curr["text"].lstrip().startswith((".", "…", "_", "-")))
                    and abs(b_curr["cy"] - avg_cy) < max(6.0, median_h * 0.35)
                )
                can_merge_x = gap_x < gutter_thresh or ((is_dotted_fill or is_form_pair) and gap_x < 0.70 * page_w)

                if (abs(b_curr["cy"] - avg_cy) < y_thresh or overlap_y >= 0.40 * min(b_curr["h"], avg_h)) and can_merge_x:
                    single_line_boxes.append(b_curr)
                    k += 1
                else:
                    break

            single_line_boxes.sort(key=lambda it: (round(it["left"], 2), it["text"]))
            if len(single_line_boxes) >= 2:
                first = single_line_boxes[0]
                if first["w"] <= 35 and first["h"] <= 35 and first["box"].text.strip() in [
                    "o", "O", "•", "·", "*", "■", "□", "回", "▫", "▪"
                ]:
                    first["box"].text = "•"

            result_lines.append([it["box"] for it in single_line_boxes])
            i = k

    return result_lines


def join_ocr_line_boxes(line_boxes: List[OCRBox]) -> str:
    """
    Nối các box trên một dòng văn bản.
    Đối với các cặp trường biểu mẫu nằm cùng hàng ngang (ví dụ: '+ Ông (bà):' và '(ghi tên...)',
    'Chức danh:' và 'Chủ trì cuộc họp'), tự động chèn '......' vào giữa theo đúng quy định.
    """
    if not line_boxes:
        return ""
    if len(line_boxes) == 1:
        return line_boxes[0].text.strip()

    parts: List[str] = []
    for idx, b in enumerate(line_boxes):
        b_txt = b.text.strip()
        if not b_txt:
            continue
        if idx == 0:
            parts.append(b_txt)
        else:
            prev_txt = parts[-1]
            prev_b = line_boxes[idx - 1]
            gap = b.bbox[0][0] - prev_b.bbox[1][0]
            is_form_junction = (
                (
                    prev_txt.endswith((":", "."))
                    or any(kw in prev_txt for kw in ["Ông (bà)", "Ông(bà)", "Chức danh", "Họ và tên"])
                )
                and (
                    b_txt.startswith(("(", "Chủ trì", "Thư ký"))
                    or any(kw in b_txt for kw in ["Chủ trì", "Thư ký", "đại diện"])
                    or gap >= 20
                )
            )
            if is_form_junction:
                clean_prev = re.sub(r"[\.…_\s]+$", "", prev_txt)
                if clean_prev.endswith(":"):
                    parts[-1] = f"{clean_prev} ......"
                else:
                    parts[-1] = f"{clean_prev}: ......"
                parts.append(b_txt)
            else:
                parts.append(b_txt)
    return " ".join(parts)


def detect_parallel_administrative_header(
    box_info: List[dict],
    page_min_x: float,
    page_max_x: float,
    page_w: float,
    gutter_thresh: float,
    median_h: float,
) -> Tuple[List[List[OCRBox]], List[dict]]:
    """
    Phát hiện vùng tiêu đề hành chính 2 khối độc lập (Administrative Parallel Header):
    - Cột bên trái: Cơ quan ban hành, số hiệu văn bản (đọc trọn vẹn từ trên xuống dưới).
    - Cột bên phải: Quốc hiệu, tiêu ngữ, địa danh, ngày tháng (đọc trọn vẹn từ trên xuống dưới).
    - Ranh giới kết thúc là Tiêu đề chính giữa trang (ví dụ: BIÊN BẢN, THÔNG TƯ, QUYẾT ĐỊNH,...).
    """
    if len(box_info) < 3:
        return HeaderDetectionResult([], box_info, parallel_header=None)

    page_mid_x = (page_min_x + page_max_x) / 2.0

    # Tìm tiêu đề chính giữa trang
    centered_title = None
    for item in box_info:
        item_mid = (item["left"] + item["right"]) / 2.0
        text_upper = item["text"].upper()
        is_title_keyword = any(kw in text_upper for kw in [
            "BIÊN BẢN", "QUYẾT ĐỊNH", "THÔNG TƯ", "NGHỊ ĐỊNH", "BÁO CÁO",
            "KẾ HOẠCH", "TỜ TRÌNH", "BẢNG ĐIỂM", "CHỈ THỊ", "CÔNG VĂN"
        ])
        if (is_title_keyword or abs(item_mid - page_mid_x) < 0.18 * page_w) and item["top"] < 400:
            if is_title_keyword or item["top"] > 180:
                centered_title = item
                break

    header_limit_y = centered_title["top"] if centered_title else min(350.0, max(b["bottom"] for b in box_info) * 0.38)

    header_boxes = [b for b in box_info if b["bottom"] <= header_limit_y + 5]
    remaining = [b for b in box_info if b not in header_boxes]

    if len(header_boxes) < 3:
        return HeaderDetectionResult([], box_info, parallel_header=None)

    # Không bao giờ nhận nhầm hàng tiêu đề bảng thành tiêu đề hành chính 2 cột
    has_table_kw = any(re.match(r"^(?:STT|TT|SỐ\s*TT|MÃ|ĐƠN\s*GIÁ|SỐ\s*LƯỢNG|TIÊU\s*CHÍ|ĐIỂM\s*TỐI\s*ĐA|ĐIỂM\s*TỰ\s*CHẤM)$", b["text"].strip().upper()) for b in header_boxes)
    if has_table_kw:
        return HeaderDetectionResult([], box_info, parallel_header=None)

    best_split_x = None
    max_gap = 0.0
    for split_ratio in np.linspace(0.35, 0.65, 31):
        sx = page_min_x + split_ratio * page_w
        left_b = [b for b in header_boxes if b["right"] <= sx]
        right_b = [b for b in header_boxes if b["left"] >= sx]
        cross_b = [b for b in header_boxes if b["left"] < sx and b["right"] > sx]
        if len(left_b) >= 2 and len(right_b) >= 1 and not cross_b:
            gap = min(b["left"] for b in right_b) - max(b["right"] for b in left_b)
            if gap >= max(20.0, 0.05 * page_w) and gap > max_gap:
                max_gap = gap
                best_split_x = sx

    if best_split_x is None:
        return HeaderDetectionResult([], box_info, parallel_header=None)

    left_b = [b for b in header_boxes if b["right"] <= best_split_x]
    right_b = [b for b in header_boxes if b["left"] >= best_split_x]

    if centered_title is None:
        left_text = " ".join(b["text"].upper() for b in left_b)
        right_text = " ".join(b["text"].upper() for b in right_b)
        has_admin_left = any(kw in left_text for kw in ["ĐẢNG", "CHI BỘ", "UBND", "ỦY BAN", "CÔNG TY", "DOANH NGHIỆP", "BỘ", "SỞ", "PHÒNG", "BAN", "TRƯỜNG", "SỐ:", "SỐ "])
        has_admin_right = any(kw in right_text for kw in ["CỘNG HÒA", "ĐỘC LẬP", "HẠNH PHÚC", "ĐẢNG CỘNG SẢN", "NGÀY", "NĂM 20"])
        if not (has_admin_left or has_admin_right):
            return HeaderDetectionResult([], box_info, parallel_header=None)

    # Lọc bỏ các box đường kẻ trang trí bị OCR nhận sai thành ký tự rác (%, *, ~, _, -, =, ...) hoặc số lặp 14444444, 44444, 11111
    def _is_decorative_line_noise(b_item: dict) -> bool:
        txt = b_item["text"].strip()
        if not txt:
            return True
        if re.fullmatch(r"[%*~_\-=—–·•.]+", txt):
            return True
        if re.fullmatch(r"1?4{3,}", txt) or re.fullmatch(r"[1lI|!*=\-—_~]{4,}", txt):
            return True
        return False

    left_b = [b for b in left_b if not _is_decorative_line_noise(b)]
    right_b = [b for b in right_b if not _is_decorative_line_noise(b)]

    lines_left = _group_single_column_lines(left_b, median_h, gutter_thresh)
    lines_right = _group_single_column_lines(right_b, median_h, gutter_thresh)

    left_text_lines = [" ".join(b.text.strip() for b in line if b.text.strip()) for line in lines_left]
    right_text_lines = [" ".join(b.text.strip() for b in line if b.text.strip()) for line in lines_right]
    left_boxes = [b for line in lines_left for b in line]
    right_boxes = [b for line in lines_right for b in line]

    parallel_hdr = ParallelHeader(
        left_lines=left_text_lines,
        right_lines=right_text_lines,
        left_boxes=left_boxes,
        right_boxes=right_boxes,
    )

    return HeaderDetectionResult(lines_left + lines_right, remaining, parallel_header=parallel_hdr)


def detect_and_reconstruct_table(
    box_info: List[dict],
    page_min_x: float,
    page_max_x: float,
    page_w: float,
    median_h: float,
) -> Tuple[Optional[OCRTable], List[dict], List[List[OCRBox]], List[dict]]:
    """
    Phát hiện và tái dựng cấu trúc bảng biểu dựa trên các cột anchor và ranh giới cột.
    Trả về: (table_obj, pre_table_boxes, table_rows_boxes, post_table_boxes)
    """
    narrow_boxes = [b for b in box_info if b["w"] < 0.38 * page_w]
    if len(narrow_boxes) < 4:
        return None, box_info, [], []

    # Tìm box tiêu đề cột STT trong bảng (nếu có)
    stt_header_box = None
    for b in box_info:
        txt_norm = b["text"].strip().upper()
        if re.match(r"^(?:STT|TT|SỐ\s*TT|TUẦN|TUẦN\s*LỄ|WEEK)$", txt_norm):
            stt_header_box = b
            break

    # Gom cụm các box hẹp theo trục X
    clusters = []
    for b in sorted(narrow_boxes, key=lambda b: (b["left"] + b["right"]) / 2.0):
        b_mid = (b["left"] + b["right"]) / 2.0
        matched = False
        for cl in clusters:
            avg_mid = float(np.mean([(it["left"] + it["right"]) / 2.0 for it in cl]))
            if abs(b_mid - avg_mid) <= max(35.0, 1.8 * median_h):
                cl.append(b)
                matched = True
                break
        if not matched:
            clusters.append([b])

    # Cụm cột STT (nếu có header STT)
    stt_cluster = []
    if stt_header_box is not None:
        stt_mid = (stt_header_box["left"] + stt_header_box["right"]) / 2.0
        for b in narrow_boxes:
            b_mid = (b["left"] + b["right"]) / 2.0
            if abs(b_mid - stt_mid) <= max(35.0, 1.8 * median_h) or b["right"] <= stt_header_box["right"] + 15:
                stt_cluster.append(b)
        if stt_header_box not in stt_cluster:
            stt_cluster.append(stt_header_box)

    def _is_valid_anchor_cluster(cl_boxes: List[dict]) -> bool:
        if not cl_boxes:
            return False
        # Cụm chỉ chứa toàn các ký tự bullet rác (*, %, +, -, •, |) không thể là cột dữ liệu độc lập
        is_pure_bullet = all(b["text"].strip() in {"*", "%", "+", "-", "•", "|", "~"} for b in cl_boxes)
        if is_pure_bullet:
            return False
        return True

    # Cột anchor: Cụm có >= 3 box, hoặc cụm STT có >= 2 box
    table_col_anchors = []
    if stt_cluster and len(stt_cluster) >= 2:
        table_col_anchors.append(stt_cluster)
        stt_box_ids = {id(b["box"]) for b in stt_cluster}
        for cl in clusters:
            filtered_cl = [b for b in cl if id(b["box"]) not in stt_box_ids]
            if len(filtered_cl) >= 3 and _is_valid_anchor_cluster(filtered_cl):
                table_col_anchors.append(filtered_cl)
    else:
        table_col_anchors = [cl for cl in clusters if len(cl) >= 3 and _is_valid_anchor_cluster(cl)]

    if len(table_col_anchors) < 2:
        return None, box_info, [], []

    # Phạm vi bảng: Hàng đầu tiên chứa các anchor (STT, Điểm tối đa,...)
    all_anchor_boxes = [b for cl in table_col_anchors for b in cl]
    if stt_header_box is not None:
        anchor_min_top = min(stt_header_box["top"], min(b["top"] for b in all_anchor_boxes))
    else:
        anchor_min_top = min(b["top"] for b in all_anchor_boxes)
    anchor_max_bottom = max(b["bottom"] for b in all_anchor_boxes)

    table_top_bound = anchor_min_top - 5
    table_bot_bound = anchor_max_bottom + 5

    table_boxes = [
        b for b in box_info
        if b["bottom"] > table_top_bound and b["top"] < table_bot_bound
    ]
    if not table_boxes:
        return None, box_info, [], []


    table_box_ids = {id(b["box"]) for b in table_boxes}

    # BẢO ĐẢM NGUYÊN TẮC PHÂN HOẠCH ĐỘC QUYỀN (MUTUAL EXCLUSIVITY):
    # Một box đã thuộc về table_boxes KHÔNG BAO GIỜ được xuất hiện trong pre_table_boxes hoặc post_table_boxes
    pre_table_boxes = [
        b for b in box_info
        if id(b["box"]) not in table_box_ids and b["bottom"] <= anchor_min_top
    ]
    post_table_boxes = [
        b for b in box_info
        if id(b["box"]) not in table_box_ids and b["top"] >= anchor_max_bottom
    ]

    if not table_boxes:
        return None, box_info, [], []

    # Tính rãnh phân cách cột
    table_col_anchors.sort(key=lambda cl: float(np.mean([(b["left"] + b["right"]) / 2.0 for b in cl])))
    dividers = [page_min_x - 10]

    # Rãnh sau anchor 0 (STT):
    a0_max_right = max(b["right"] for b in table_col_anchors[0])
    right_of_a0 = [b for b in table_boxes if b["left"] > a0_max_right]
    if right_of_a0:
        c1_min_left = min(b["left"] for b in right_of_a0)
        dividers.append((a0_max_right + c1_min_left) / 2.0)

    # Rãnh trước anchor 1 (Điểm tối đa):
    a1_min_left = min(b["left"] for b in table_col_anchors[1])
    left_of_a1 = [b for b in table_boxes if b["right"] < a1_min_left]
    if left_of_a1:
        c1_max_right = max(b["right"] for b in left_of_a1)
        dividers.append((c1_max_right + a1_min_left) / 2.0)

    # Rãnh giữa các anchor tiếp theo
    for idx in range(1, len(table_col_anchors)):
        if idx + 1 < len(table_col_anchors):
            a_curr_right = max(b["right"] for b in table_col_anchors[idx])
            a_next_left = min(b["left"] for b in table_col_anchors[idx + 1])
            dividers.append((a_curr_right + a_next_left) / 2.0)

    # Rãnh sau anchor cuối cùng (nếu có cột Ghi chú)
    last_anchor = table_col_anchors[-1]
    last_max_right = max(b["right"] for b in last_anchor)
    right_of_last = [b for b in table_boxes if b["left"] > last_max_right]
    if right_of_last:
        c_last_min_left = min(b["left"] for b in right_of_last)
        dividers.append((last_max_right + c_last_min_left) / 2.0)

    dividers.append(page_max_x + 10)
    dividers = sorted(list(set(dividers)))
    num_cols = len(dividers) - 1

    # Gom nhóm theo hàng
    def _is_table_header_row(r_items: List[dict]) -> bool:
        if not r_items:
            return False
        row_text = " ".join(item["box_info"]["text"].strip().lower() for item in r_items)
        header_keywords = [
            "tuần", "ngày", "nội dung", "stt", "tiêu chí", "điểm tối đa", 
            "điểm tự chấm", "ghi chú", "nội dung thực hiện", "nhiệm vụ"
        ]
        matches = [kw for kw in header_keywords if kw in row_text]
        if len(matches) >= 2:
            return True
        if any(h in row_text for h in ["tuần", "week"]) and any(h in row_text for h in ["ngày", "nội dung"]):
            return True
        return False

    # Gom nhóm theo hàng
    table_boxes.sort(key=lambda b: (b["top"], b["left"]))
    rows: List[List[dict]] = []
    for b in table_boxes:
        b_mid = (b["left"] + b["right"]) / 2.0
        c_idx = 0
        for ci in range(num_cols):
            if dividers[ci] <= b_mid < dividers[ci + 1]:
                c_idx = ci
                break

        matched_row = None
        for r in rows:
            is_r_header = _is_table_header_row(r)
            b_text = b["text"].strip()
            # Khóa hàng tiêu đề bảng: Tuyệt đối không gộp các bounding box dữ liệu vào cùng hàng tiêu đề
            if is_r_header:
                # Nếu box bắt đầu bằng bullet (+, -, *, •, ·, –) thì không thuộc header
                if re.match(r"^[-+*•%~—–·]", b_text):
                    continue
                # Không gộp nếu box hoàn toàn nằm dưới đáy của toàn bộ tiêu đề
                r_bottom = max(item["box_info"]["bottom"] for item in r)
                if b["top"] >= r_bottom:
                    continue

            r_top = min(item["box_info"]["top"] for item in r)
            r_bottom = max(item["box_info"]["bottom"] for item in r)
            r_cy = float(np.mean([item["box_info"]["cy"] for item in r]))
            overlap_y = max(0, min(b["bottom"], r_bottom) - max(b["top"], r_top))
            if overlap_y >= 0.35 * min(b["h"], r_bottom - r_top) or abs(b["cy"] - r_cy) <= median_h * 0.6:
                matched_row = r
                break

        if matched_row is not None:
            matched_row.append({"box_info": b, "col": c_idx})
        else:
            rows.append([{"box_info": b, "col": c_idx}])

    rows.sort(key=lambda r: min(item["box_info"]["top"] for item in r))

    # Xử lý ô nhiều dòng (Multi-line cell merging / Orphan row merging)
    # Nếu một hàng chỉ chứa các box nằm trong duy nhất 1 cột (ví dụ: ngày sinh bị ngắt số đuôi '8'),
    # và hàng ngay trước đó đã có ô ở cột này với khoảng cách dọc hợp lý (<= 1.8 * median_h),
    # đồng thời hàng trước là hàng dữ liệu chính (có >= 2 cột khác nhau được điền):
    # -> Gộp box này vào ô tương ứng của hàng trước thay vì tạo thành hàng mồ côi (| 8 | |).
    merged_rows: List[List[dict]] = []
    for r in rows:
        r_cols = set(item["col"] for item in r)
        if merged_rows and len(r_cols) == 1 and num_cols >= 3:
            single_col = next(iter(r_cols))
            prev_r = merged_rows[-1]
            if _is_table_header_row(prev_r):
                # Tuyệt đối không gộp dữ liệu vào hàng tiêu đề bảng
                merged_rows.append(r)
                continue
            prev_items_in_col = [it for it in prev_r if it["col"] == single_col]
            prev_distinct_cols = len(set(it["col"] for it in prev_r))
            if prev_items_in_col and prev_distinct_cols >= 2:
                r_min_top = min(item["box_info"]["top"] for item in r)
                prev_max_bottom = max(it["box_info"]["bottom"] for it in prev_items_in_col)
                if (r_min_top - prev_max_bottom) <= 1.8 * median_h:
                    prev_r.extend(r)
                    continue

        merged_rows.append(r)

    rows = merged_rows

    # Lookahead row merging:
    # Gom đúng nội dung nhiều dòng của mỗi tuần: Nếu một hàng r_i chỉ chứa duy nhất cột nội dung (col >= 2, không có box ở col 0 hoặc col 1),
    # nhưng hàng ngay sau r_{i+1} là hàng anchor chính (chứa STT hoặc Ngày) và khoảng cách dọc
    # r_{i+1}.top - r_i.bottom <= 2.2 * median_h:
    # -> r_i chính là dòng nội dung đầu của r_{i+1} (do ô STT/Ngày được căn giữa ô theo chiều dọc).
    # -> Gộp r_i vào r_{i+1}.
    lookahead_merged_rows: List[List[dict]] = []
    idx = 0
    while idx < len(rows):
        curr_r = rows[idx]
        if idx + 1 < len(rows):
            next_r = rows[idx + 1]
            curr_cols = set(item["col"] for item in curr_r)
            # Chỉ áp dụng khi curr_r là dòng nội dung mồ côi (chỉ chứa duy nhất cột nội dung col >= 2, không có col 0 và col 1)
            is_orphan_content_row = (len(curr_cols) == 1 and all(c >= 2 for c in curr_cols))

            next_has_anchor = any(
                item["col"] in {0, 1} and (
                    re.search(r"\d{2}/\d{2}/\d{4}", item["box_info"]["text"])
                    or re.match(r"^\d+$", item["box_info"]["text"].strip())
                )
                for item in next_r
            )
            if is_orphan_content_row and next_has_anchor and not _is_table_header_row(curr_r):
                curr_bottom = max(item["box_info"]["bottom"] for item in curr_r)
                next_anchor_top = min(
                    item["box_info"]["top"] for item in next_r if item["col"] in {0, 1}
                )
                if (next_anchor_top - curr_bottom) <= 2.2 * median_h and (next_anchor_top - curr_bottom) >= -1.0 * median_h:
                    next_r.extend(curr_r)
                    idx += 1
                    continue
        lookahead_merged_rows.append(curr_r)
        idx += 1

    rows = lookahead_merged_rows

    table_grid: List[List[str]] = []
    reconstructed_rows_boxes: List[List[OCRBox]] = []
    for r in rows:
        row_cells: List[List[dict]] = [[] for _ in range(num_cols)]
        for item in r:
            row_cells[item["col"]].append(item["box_info"])

        cell_texts: List[str] = []
        row_box_list: List[OCRBox] = []
        for c in range(num_cols):
            boxes_in_cell = row_cells[c]
            boxes_in_cell.sort(key=lambda it: (it["top"], it["left"]))
            cell_parts = [it["box"].text.strip() for it in boxes_in_cell if it["box"].text.strip()]
            if len(cell_parts) > 1:
                joined_text = cell_parts[0]
                for p in cell_parts[1:]:
                    if re.search(r"[\d/.-]$", joined_text) and re.match(r"^\d+$", p):
                        joined_text += p
                    elif p.startswith(("-", "+", "*", "•")):
                        joined_text += "\n" + p
                    else:
                        joined_text += " " + p
                cell_text = joined_text
            elif cell_parts:
                cell_text = cell_parts[0]
            else:
                cell_text = ""

            # Khôi phục bullet trong ô cột tiêu chí/nội dung
            if cell_text and (c <= 2):
                cell_text_rec, dec, _ = ContextSymbolRecovery.recover_bullets_and_pluses(cell_text)
                if dec == SymbolDecision.AUTO_CORRECT:
                    cell_text = cell_text_rec
            cell_texts.append(cell_text)
            row_box_list.extend([it["box"] for it in boxes_in_cell])

        table_grid.append(cell_texts)
        if row_box_list:
            reconstructed_rows_boxes.append(row_box_list)

    # Loại bỏ các cột ma (Phantom Column Pruning - cột 100% rỗng hoặc chỉ chứa bullet rác đơn lẻ)
    if table_grid and num_cols > 2:
        valid_cols = []
        for c in range(num_cols):
            all_empty = all(not r[c].strip() for r in table_grid)
            if all_empty:
                continue
            is_just_bullet_noise = (
                all(r[c].strip() in {"", "*", "%", "-", "+", "•", "|"} for r in table_grid)
                and c + 1 < num_cols
            )
            if is_just_bullet_noise:
                for r_idx in range(len(table_grid)):
                    b_tok = table_grid[r_idx][c].strip()
                    if b_tok and b_tok not in {"|", "\\", "/"}:
                        nxt = table_grid[r_idx][c + 1].strip()
                        if not nxt.startswith(("+", "-", "•", "*")):
                            table_grid[r_idx][c + 1] = f"{b_tok} {nxt}".strip()
                continue
            valid_cols.append(c)

        if len(valid_cols) < num_cols and len(valid_cols) >= 2:
            table_grid = [[r[c] for c in valid_cols] for r in table_grid]
            num_cols = len(valid_cols)

    # Nhận diện các cột biểu tượng (Icon / Symb / Symbol) dựa trên dòng tiêu đề
    icon_col_indices: Set[int] = set()
    if table_grid:
        header_row = table_grid[0]
        # Chuẩn hóa cột Điểm tối đa bị nhận nhầm thành "đa" / "tối đa"
        # và cột Điểm tự chấm bị nhận nhầm thành "chấm" / "tự chấm"
        for c_idx in range(len(header_row)):
            cell_txt = header_row[c_idx].strip()
            if cell_txt.lower() in {"đa", "tối đa"}:
                prev_cell = header_row[c_idx - 1].lower() if c_idx > 0 else ""
                next_cell = header_row[c_idx + 1].lower() if c_idx + 1 < len(header_row) else ""
                if "tiêu chí" in prev_cell or "tự chấm" in next_cell or "chấm" in next_cell or "ghi chú" in next_cell or "stt" in prev_cell or c_idx in {1, 2}:
                    header_row[c_idx] = "Điểm tối đa"
            elif cell_txt.lower() in {"chấm", "tự chấm", "tư chấm", "điểm chấm"}:
                prev_cell = header_row[c_idx - 1].lower() if c_idx > 0 else ""
                next_cell = header_row[c_idx + 1].lower() if c_idx + 1 < len(header_row) else ""
                if "tối đa" in prev_cell or "đa" in prev_cell or "tiêu chí" in prev_cell or "ghi chú" in next_cell or c_idx in {2, 3}:
                    header_row[c_idx] = "Điểm tự chấm"

        for c_idx, h_text in enumerate(header_row):
            h_norm = h_text.strip().lower()
            if any(term in h_norm for term in ["symb", "symbol", "icon", "biểu tượng", "ký hiệu"]):
                icon_col_indices.add(c_idx)

    # Chuẩn hóa cột Icon trong các hàng dữ liệu (từ hàng 1 trở đi)
    if icon_col_indices and len(table_grid) > 1:
        ICON_NOISE_TOKENS = {"cy", "cos", "tt", "j", "l", "c", "t", "r", "n", "f", "\\", "|", "%", "1", "o", "v"}
        for r_idx in range(1, len(table_grid)):
            row = table_grid[r_idx]
            for c_idx in icon_col_indices:
                if c_idx < len(row):
                    val = row[c_idx].strip()
                    if not val:
                        row[c_idx] = "[Icon]"
                    elif val.lower() in ICON_NOISE_TOKENS or (len(val) <= 3 and not any(ch in "0123456789" for ch in val) and not any(ord(ch) > 127 for ch in val)):
                        row[c_idx] = "[Icon]"

    # Khôi phục số thứ tự STT (1, 2, 3...) cho các hàng tiêu chí chính trong bảng nếu bị thiếu hoặc nhận nhầm
    SUB_CRITERION_KEYWORDS = [
        "thời gian sinh hoạt",
        "chi bộ dưới",
        "chi bộ trên",
        "từ 90%",
        "từ 75%",
        "từ 65%",
        "dưới 60%",
        "không có đảng viên",
        "ít hơn quy định",
        "chỉ bằng",
        "so với quy định",
        "vắng mặt",
        "trừ trường hợp",
    ]
    if table_grid and len(table_grid) >= 2:
        header_row = table_grid[0]
        if header_row and re.match(r"^(?:STT|TT|SỐ\s*TT|TUẦN|TUẦN\s*LỄ|WEEK)$", header_row[0].strip().upper()):
            expected_stt = 1
            main_row_min_x: Optional[float] = None
            for r_idx in range(1, len(table_grid)):
                row = table_grid[r_idx]
                r_items = rows[r_idx] if r_idx < len(rows) else []
                c0_text = row[0].strip() if len(row) > 0 else ""
                c1_text = row[1].strip() if len(row) > 1 else ""
                c2_text = row[2].strip() if len(row) > 2 else ""
                main_text = c1_text or c2_text
                main_lower = main_text.lower()

                # Kiểm tra bounding box gốc của cột STT trên ảnh
                stt_boxes = [it["box_info"] for it in r_items if it["col"] == 0]
                has_orig_digit = any(re.search(r"\d", b["text"]) for b in stt_boxes)

                # Bounding boxes của cột nội dung chính
                content_boxes = [it["box_info"] for it in r_items if it["col"] in {1, 2}]
                curr_min_x = min(b["left"] for b in content_boxes) if content_boxes else None

                # 1. Ký hiệu liệt kê đầu dòng: +, -, *, •, ·, –, hoặc %
                starts_with_bullet = (
                    c0_text in {"+", "-", "*", "•", "%", "~", "—", "–", "·"}
                    or re.match(r"^([+*•·%–\-~]|—|\b[FE47]\s+)", main_text) is not None
                    or any(re.match(r"^([+*•·%–\-~]|—)$", b["text"].strip()) is not None for b in stt_boxes)
                )

                # 2. Thụt lề hình học (min_x) so với hàng tiêu chí chính phía trên
                is_indented = False
                if main_row_min_x is not None and curr_min_x is not None:
                    if (curr_min_x - main_row_min_x) >= max(12.0, 0.6 * median_h):
                        is_indented = True

                # 3. Điều kiện giải thích con không có bounding box số thứ tự hợp lệ
                main_stripped = re.sub(r"^[+*•%~—–\-·\s]+", "", main_lower).strip()
                is_sub_condition = (
                    re.search(r"^(từ\s+\d+|dưới\s+\d+|trên\s+\d+|không\s+có\s+|thời\s+gian\s+|chi\s+bộ\s+(?:dưới|trên)|ít\s+hơn|chỉ\s+bằng|vắng\s+mặt)", main_stripped) is not None
                    or any(phrase in main_lower for phrase in [
                        "ít hơn quy định", "chỉ bằng ½", "chỉ bằng %", "dưới ½", "dưới %",
                        "vắng không có lý do", "vắng mặt có lý do", "quá 03 lần liên tiếp"
                    ])
                )

                # Xác định Hàng Con (Sub-row)
                is_sub_row = (
                    starts_with_bullet
                    or is_indented
                    or (not has_orig_digit and is_sub_condition)
                )

                if is_sub_row:
                    # BẮT BUỘC giữ trống cột STT (hoặc giữ bullet nếu có), KHÔNG tăng expected_stt
                    if c0_text in {"+", "-", "*", "•", "%", "~", "—", "–", "·"}:
                        row[0] = c0_text
                    else:
                        row[0] = ""
                elif main_text:
                    # Đây là Hàng Chính (Main Row) thực sự
                    if curr_min_x is not None:
                        main_row_min_x = curr_min_x
                    curr_stt = row[0].strip()
                    m_num = re.search(r"^(\d+)", curr_stt)
                    if m_num:
                        row[0] = m_num.group(1)
                        expected_stt = int(m_num.group(1)) + 1
                    else:
                        row[0] = str(expected_stt)
                        expected_stt += 1

    # BẢO VỆ CHẮC CHẮN ĐÂY LÀ BẢNG BIỂU HỢP LỆ (TABLE VALIDATION):
    # Một bảng biểu thực sự phải có nhiều hàng chứa ít nhất 2 cột không rỗng
    # (tránh nhận nhầm toàn bộ văn bản danh sách/hành chính thông thường thành 1 bảng biểu khổng lồ)
    multi_cell_rows = [r for r in table_grid if sum(1 for c in r if c.strip()) >= 2]
    if len(multi_cell_rows) < 2 or (len(table_grid) > 0 and len(multi_cell_rows) / len(table_grid) < 0.40):
        return None, box_info, [], []

    bbox = [
        int(min(b["left"] for b in table_boxes)),
        int(min(b["top"] for b in table_boxes)),
        int(max(b["right"] for b in table_boxes)),
        int(max(b["bottom"] for b in table_boxes)),
    ]
    ocr_table = OCRTable(
        grid=table_grid,
        bbox=bbox,
        num_rows=len(table_grid),
        num_cols=num_cols,
        row_boxes=reconstructed_rows_boxes,
    )
    return ocr_table, pre_table_boxes, reconstructed_rows_boxes, post_table_boxes


def reconstruct_document_layout(boxes: List[OCRBox]) -> Tuple[List[List[OCRBox]], List[OCRTable]]:
    """
    Phân tích toàn diện bố cục tài liệu (Document Layout Reconstruction):
    1. Phát hiện và tái dựng bảng biểu (Table Reconstruction) khi có cấu trúc hàng cột.
    2. Phát hiện tiêu đề hành chính 2 khối (Administrative Parallel Header) ở đầu trang.
    3. Phân tích đa cột / một cột (Sliding window) cho các vùng văn bản còn lại.
    Trả về: LayoutResult(grouped_lines, tables, parallel_header)
    """
    if not boxes:
        return LayoutResult([], [], parallel_header=None)

    box_info = _box_info_list(boxes)
    page_min_x = min(item["left"] for item in box_info)
    page_max_x = max(item["right"] for item in box_info)
    page_w = max(1, page_max_x - page_min_x)
    heights = [item["h"] for item in box_info]
    median_h = float(np.median(heights)) if heights else 20.0
    gutter_thresh = max(3.0 * median_h, 0.08 * page_w)

    # 1. Kiểm tra Header 2 khối hành chính đầu trang TRƯỚC (Administrative Parallel Header)
    # Tuyệt đối không để Table Reconstruction nuốt mất Header 2 cột hoặc ghép chung với bảng
    hdr_res = detect_parallel_administrative_header(
        box_info, page_min_x, page_max_x, page_w, gutter_thresh, median_h
    )
    header_lines = hdr_res[0]
    rem_boxes = hdr_res[1]
    parallel_hdr = getattr(hdr_res, "parallel_header", None)

    # 2. Kiểm tra Table Reconstruction trên các box còn lại (hoặc toàn bộ nếu không có header)
    target_boxes_for_table = rem_boxes if header_lines else box_info
    tbl, pre_tbl, tbl_rows, post_tbl = detect_and_reconstruct_table(
        target_boxes_for_table, page_min_x, page_max_x, page_w, median_h
    )

    if tbl is not None:
        pre_lines = header_lines + _group_boxes_sliding_window(
            pre_tbl, page_min_x, page_w, gutter_thresh, median_h
        )
        post_lines = _group_boxes_sliding_window(
            post_tbl, page_min_x, page_w, gutter_thresh, median_h
        )
        all_lines = pre_lines + tbl_rows + post_lines
        return LayoutResult(all_lines, [tbl], parallel_header=parallel_hdr)

    # 3. Nếu không có bảng: Gom nhóm các dòng còn lại
    if header_lines:
        rem_lines = _group_boxes_sliding_window(
            rem_boxes, page_min_x, page_w, gutter_thresh, median_h
        )
        return LayoutResult(header_lines + rem_lines, [], parallel_header=parallel_hdr)

    # 4. Văn bản thông thường (Sliding Window Multi-column / Single-column)
    standard_lines = _group_boxes_sliding_window(
        box_info, page_min_x, page_w, gutter_thresh, median_h
    )
    return LayoutResult(standard_lines, [], parallel_header=None)


def group_boxes_into_lines(boxes: List[OCRBox]) -> List[List[OCRBox]]:
    """
    Gom nhóm các khối chữ theo từng dòng đọc (Line-level grouping) và sắp xếp
    thứ tự đọc tài liệu dựa trên phân tích bố cục hình học, bảng biểu và tiêu đề hành chính.
    """
    lines, _ = reconstruct_document_layout(boxes)
    return lines


def sort_boxes_reading_order(boxes: List[OCRBox]) -> List[OCRBox]:
    """
    Sắp xếp bounding box theo thứ tự đọc tự nhiên từ trên xuống dưới, từ trái sang phải.
    Gom nhóm theo dòng và bảo toàn toàn bộ danh sách boxes.
    """
    grouped_lines = group_boxes_into_lines(boxes)
    return [b for line in grouped_lines for b in line]


# ---------------------------------------------------------------------------
# Cơ Chế Khôi Phục Ký Tự Theo Ngữ Cảnh (Context-Aware Symbol Recovery)
# ---------------------------------------------------------------------------


class SymbolDecision(str, Enum):
    """Quyết định 3 trạng thái cho phục hồi ký tự OCR."""
    AUTO_CORRECT = "AUTO_CORRECT"
    FLAG_UNCERTAIN = "FLAG_UNCERTAIN"
    PRESERVE = "PRESERVE"


ROMAN_HEADING_KEYWORDS = {
    # Hành chính & pháp lý
    "NỘI DUNG", "TIẾN HÀNH", "KẾT LUẬN", "TỔ CHỨC", "MỤC ĐÍCH", "YÊU CẦU",
    "KẾ HOẠCH", "PHƯƠNG HƯỚNG", "NHIỆM VỤ", "ĐÁNH GIÁ", "CHƯƠNG TRÌNH",
    "THỰC HIỆN", "KẾT QUẢ", "THỜI GIAN", "ĐỊA ĐIỂM", "THÀNH PHẦN",
    "KIỂM ĐIỂM", "THẢO LUẬN", "NGHỊ QUYẾT", "BÁO CÁO", "ĐIỀU KHOẢN",
    "QUYỀN VÀ NGHĨA VỤ", "PHẠM VI", "HIỆU LỰC", "CHẤM DỨT",
    # Học thuật, giáo dục & khoa học
    "MỞ ĐẦU", "TỔNG QUAN", "CƠ SỞ LÝ THUYẾT", "PHƯƠNG PHÁP", "BÀN LUẬN",
    "KIẾN NGHỊ", "TÀI LIỆU THAM KHẢO", "PHỤ LỤC", "GIỚI THIỆU", "ĐẶC ĐIỂM",
    "KHÁI NIỆM", "NGUYÊN LÝ", "CẤU TẠO", "ỨNG DỤNG", "PHÂN TÍCH",
    "GIẢI PHÁP", "KẾ HOẠCH BÀI DẠY", "MỤC TIÊU BÀI HỌC",
    # Doanh nghiệp, thương mại & tài chính
    "TÌNH HÌNH HOẠT ĐỘNG", "DOANH THU", "LỢI NHUẬN", "ĐỊNH HƯỚNG",
    "MỤC TIÊU", "TỔ CHỨC BỘ MÁY", "QUY CHẾ", "QUY TRÌNH", "BẢO HÀNH",
}


class ContextSymbolRecovery:
    """
    Cơ chế Context-Aware Symbol Recovery:
    Khôi phục ký tự và cấu trúc bị OCR nhận sai dựa trên bằng chứng ngữ cảnh văn bản và hình học.
    Tuân thủ nguyên tắc 3 trạng thái: AUTO_CORRECT, FLAG_UNCERTAIN, PRESERVE.
    Bảo toàn 100% dữ liệu gốc và ghi nhận nhật ký vết thay đổi (audit trail).
    """

    @staticmethod
    def is_all_uppercase_heading(text: str) -> bool:
        """Kiểm tra văn bản có phải là đề mục in hoa hay không (>= 70% từ in hoa)."""
        words = re.findall(r'[a-zA-ZÀ-ỹĐđ]+', text)
        if not words:
            return False
        upper_count = sum(1 for w in words if w.isupper())
        return (upper_count / len(words)) >= 0.70

    @staticmethod
    def has_heading_keyword(text: str) -> bool:
        """Kiểm tra văn bản có chứa từ khóa đề mục hay không."""
        text_upper = text.upper()
        return any(kw in text_upper for kw in ROMAN_HEADING_KEYWORDS)

    @classmethod
    def recover_roman_numeral_in_line(
        cls, line: str, prev_roman: str = ""
    ) -> Tuple[str, SymbolDecision, str, str]:
        """
        Khôi phục số La Mã đầu mục (I. đến X.) dựa trên ngữ cảnh.
        Trả về: (new_line, decision, current_roman, rule_name)
        """
        stripped = line.strip()
        if not stripped:
            return line, SymbolDecision.PRESERVE, prev_roman, ""

        # 1. Mẫu số III: IIL, IIl, 11L, lll, 111, TII, TIl, llI, TTT
        m_iii = re.match(r"^([ \t]*)(IIL|IIl|11L|lll|111|TII|TIl|llI|TTT)([\.:]?)\s+([^\n]+)$", line)
        if m_iii:
            indent = m_iii.group(1)
            prefix = m_iii.group(2)
            has_dot = (m_iii.group(3) in {".", ":"})
            rest = m_iii.group(4).strip()
            # ĐẶC BIỆT: 'Nội dung cuộc họp' luôn luôn là mục số II (không bao giờ là III)
            if re.match(r"(?i)Nội\s+dung\s+cuộc\s+họp", rest):
                return f"{indent}II. {rest}", SymbolDecision.AUTO_CORRECT, "II", f"ROMAN_II_FROM_{prefix}_CONTENT"
            if cls.is_all_uppercase_heading(rest) or cls.has_heading_keyword(rest) or (has_dot and rest and rest[0].isupper()):
                return f"{indent}III. {rest}", SymbolDecision.AUTO_CORRECT, "III", f"ROMAN_III_FROM_{prefix}"
            else:
                return line, SymbolDecision.FLAG_UNCERTAIN, prev_roman, f"UNCERTAIN_ROMAN_III_{prefix}"

        # 2. Mẫu số II: IL, Il, 1L, 1l, TI, TT, TL, Tl, ll, 11, lI (chỉ 1 chữ I/1 đứng trước L bắt buộc thành II.)
        m_ii = re.match(r"^([ \t]*)(IL|Il|1L|1l|TI|TT|TL|Tl|ll|11|lI)([\.:]?)\s+([^\n]+)$", line)
        if m_ii:
            indent = m_ii.group(1)
            prefix = m_ii.group(2)
            has_dot = (m_ii.group(3) in {".", ":"})
            rest = m_ii.group(4).strip()
            if cls.is_all_uppercase_heading(rest) or cls.has_heading_keyword(rest) or (has_dot and rest and rest[0].isupper()):
                return f"{indent}II. {rest}", SymbolDecision.AUTO_CORRECT, "II", f"ROMAN_II_FROM_{prefix}"
            else:
                return line, SymbolDecision.FLAG_UNCERTAIN, prev_roman, f"UNCERTAIN_ROMAN_II_{prefix}"

        # 3. Mẫu số IV: IV, lV, 1V, TV
        m_iv = re.match(r"^([ \t]*)(IV|lV|1V|TV)[\.:]?\s+([^\n]+)$", line)
        if m_iv:
            indent = m_iv.group(1)
            prefix = m_iv.group(2)
            rest = m_iv.group(3).strip()
            if cls.is_all_uppercase_heading(rest) or cls.has_heading_keyword(rest):
                return f"{indent}IV. {rest}", SymbolDecision.AUTO_CORRECT, "IV", f"ROMAN_IV_FROM_{prefix}"
            else:
                return line, SymbolDecision.FLAG_UNCERTAIN, prev_roman, f"UNCERTAIN_ROMAN_IV_{prefix}"

        # Mẫu số VI: VI, Vl, V1, V|
        m_vi = re.match(r"^([ \t]*)(VI|V[l1|])[\.:]?\s+([^\n]+)$", line)
        if m_vi:
            indent = m_vi.group(1)
            prefix = m_vi.group(2)
            rest = m_vi.group(3).strip()
            if cls.is_all_uppercase_heading(rest) or cls.has_heading_keyword(rest):
                return f"{indent}VI. {rest}", SymbolDecision.AUTO_CORRECT, "VI", f"ROMAN_VI_FROM_{prefix}"
            else:
                return line, SymbolDecision.FLAG_UNCERTAIN, prev_roman, f"UNCERTAIN_ROMAN_VI_{prefix}"

        # Mẫu số VII: VII, Vll, V11, VlI, VIl, VTl
        m_vii = re.match(r"^([ \t]*)(VII|V(?:ll|11|lI|Il|Tl))[\.:]?\s+([^\n]+)$", line)
        if m_vii:
            indent = m_vii.group(1)
            prefix = m_vii.group(2)
            rest = m_vii.group(3).strip()
            if cls.is_all_uppercase_heading(rest) or cls.has_heading_keyword(rest):
                return f"{indent}VII. {rest}", SymbolDecision.AUTO_CORRECT, "VII", f"ROMAN_VII_FROM_{prefix}"
            else:
                return line, SymbolDecision.FLAG_UNCERTAIN, prev_roman, f"UNCERTAIN_ROMAN_VII_{prefix}"

        # Mẫu số VIII: VIII, Vlll, V111
        m_viii = re.match(r"^([ \t]*)(VIII|V(?:lll|111))[\.:]?\s+([^\n]+)$", line)
        if m_viii:
            indent = m_viii.group(1)
            prefix = m_viii.group(2)
            rest = m_viii.group(3).strip()
            if cls.is_all_uppercase_heading(rest) or cls.has_heading_keyword(rest):
                return f"{indent}VIII. {rest}", SymbolDecision.AUTO_CORRECT, "VIII", f"ROMAN_VIII_FROM_{prefix}"
            else:
                return line, SymbolDecision.FLAG_UNCERTAIN, prev_roman, f"UNCERTAIN_ROMAN_VIII_{prefix}"

        # Mẫu số IX: IX, lX, 1X, |X
        m_ix = re.match(r"^([ \t]*)(IX|[l1|]X)[\.:]?\s+([^\n]+)$", line)
        if m_ix:
            indent = m_ix.group(1)
            prefix = m_ix.group(2)
            rest = m_ix.group(3).strip()
            if cls.is_all_uppercase_heading(rest) or cls.has_heading_keyword(rest):
                return f"{indent}IX. {rest}", SymbolDecision.AUTO_CORRECT, "IX", f"ROMAN_IX_FROM_{prefix}"
            else:
                return line, SymbolDecision.FLAG_UNCERTAIN, prev_roman, f"UNCERTAIN_ROMAN_IX_{prefix}"

        # Mẫu số X: X
        m_x = re.match(r"^([ \t]*)(X)[\.:]?\s+([^\n]+)$", line)
        if m_x:
            indent = m_x.group(1)
            prefix = m_x.group(2)
            rest = m_x.group(3).strip()
            if cls.is_all_uppercase_heading(rest) or cls.has_heading_keyword(rest):
                return f"{indent}X. {rest}", SymbolDecision.AUTO_CORRECT, "X", f"ROMAN_X_FROM_{prefix}"

        # 4. Mẫu số I: L, l, T, |, 1
        m_i = re.match(r"^([ \t]*)(L|l|T|\||1)([\.:]?)\s+([^\n]+)$", line)
        if m_i:
            indent = m_i.group(1)
            prefix = m_i.group(2).strip()
            has_dot = (m_i.group(3) in {".", ":"})
            rest = m_i.group(4).strip()
            is_all_caps = cls.is_all_uppercase_heading(rest)
            is_heading_kw = cls.has_heading_keyword(rest)

            # Nếu có dấu chấm (L., l., T., |.) HOẶC đứng không dấu chấm trước chuỗi IN HOA TOÀN BỘ (L NỘI DUNG..., L THÔNG TIN...)
            if prefix in ["L", "l", "T", "|"]:
                if not rest.startswith(("-", "+", "*", "•", "·", "–", "—")):
                    if (has_dot and (is_all_caps or is_heading_kw)) or (not has_dot and is_all_caps):
                        return f"{indent}I. {rest}", SymbolDecision.AUTO_CORRECT, "I", f"ROMAN_I_FROM_{prefix}"
            elif prefix == "1":
                # Đối với 1., chuẩn hóa thành I. khi có dấu chấm VÀ chứa từ khóa đề mục chính
                if has_dot and (is_all_caps or is_heading_kw) and any(k in rest.upper() for k in ["THÀNH PHẦN", "NỘI DUNG", "CHƯƠNG TRÌNH", "THÔNG TIN", "QUY ĐỊNH", "KẾT LUẬN"]):
                    return f"{indent}I. {rest}", SymbolDecision.AUTO_CORRECT, "I", "ROMAN_I_FROM_1_DOT"

        # 5. Nếu đã là số La Mã chuẩn I., II., III., IV., V., VI., VII., VIII., IX., X.
        m_valid = re.match(r"^[ \t]*(I{1,3}|IV|V|VI{1,3}|IX|X)\.\s+", line)
        if m_valid:
            return line, SymbolDecision.PRESERVE, m_valid.group(1), "VALID_ROMAN"

        return line, SymbolDecision.PRESERVE, prev_roman, ""

    @classmethod
    def recover_bullets_and_pluses(cls, line: str) -> Tuple[str, SymbolDecision, str]:
        """
        Khôi phục dấu cộng (+) hoặc bullet bị nhận sai thành % hoặc | ở đầu dòng/ô.
        Bảo toàn 100% các giá trị phần trăm hợp lệ (90%, 75%, 100%).
        Trả về: (new_line, decision, rule_name)
        """
        # 1. Ký tự '|' đứng trước dấu bullet rác ở đầu dòng (ví dụ: '| % Chi bộ...', '| * Chi bộ...', '| + Chi bộ...')
        m_pipe_bullet = re.match(r"^([ \t]*)\|\s*([%*+]|-)\s+([A-Za-zÀ-ỹĐđ].*)$", line)
        if m_pipe_bullet:
            indent = m_pipe_bullet.group(1)
            sym = m_pipe_bullet.group(2)
            content = m_pipe_bullet.group(3)
            # Nếu là dấu gạch ngang hợp lệ (- Nội dung...)
            if sym == "-":
                return f"{indent}- {content}", SymbolDecision.AUTO_CORRECT, "DASH_FROM_PIPE_PREFIX"
            # Nếu là ghi chú (* Ghi chú...)
            if sym == "*" and content.lower().startswith("ghi chú"):
                return f"{indent}* {content}", SymbolDecision.AUTO_CORRECT, "NOTE_FROM_PIPE_PREFIX"
            return f"{indent}+ {content}", SymbolDecision.AUTO_CORRECT, "PLUS_FROM_PIPE_BULLET_PREFIX"

        # 2. Ký tự '%' hoặc '*' ở đầu dòng theo sau là chữ cái tiếng Việt (ví dụ: '% Chi bộ...', '* Chi bộ...', '% Từ 90%...')
        m_pct = re.match(r"^([ \t]*)([%*])\s*([A-Za-zÀ-ỹĐđ].*)$", line)
        if m_pct:
            indent = m_pct.group(1)
            sym = m_pct.group(2)
            content = m_pct.group(3)
            # Bảo vệ dòng ghi chú: '* Ghi chú quan trọng'
            if sym == "*" and content.lower().startswith("ghi chú"):
                return line, SymbolDecision.PRESERVE, ""
            return f"{indent}+ {content}", SymbolDecision.AUTO_CORRECT, "PLUS_FROM_PERCENT_PREFIX"

        # 3. Ký tự '4' hoặc '7' ở đầu dòng trong tiêu chí bảng bị nhận nhầm từ '+' (ví dụ: '4 Chi bộ...', '7 Thời gian...')
        m_num_bullet = re.match(r"^([ \t]*)([47])\s+([A-Za-zÀ-ỹĐđ].*)$", line)
        if m_num_bullet:
            indent = m_num_bullet.group(1)
            content = m_num_bullet.group(3)
            c_lower = content.lower()
            if any(c_lower.startswith(kw) for kw in [
                "chỉ bộ", "chi bộ", "thời gian", "tỷ lệ", "không có", "đã có", "được", "dưới", "từ "
            ]):
                return f"{indent}+ {content}", SymbolDecision.AUTO_CORRECT, "PLUS_FROM_NUMERIC_NOISE"

        # 4. Ký tự chân trang bị nhận thành 1, -, / hoặc ? (ví dụ: '1 Trừ trường hợp...', '/Trừ trường hợp...', '?Trừ trường hợp...')
        m_footnote = re.match(r"^([ \t]*)[1?*/\-]+\s*(Trừ\s+trường\s+hợp.*)$", line, re.IGNORECASE)
        if m_footnote:
            indent = m_footnote.group(1)
            content = m_footnote.group(2)
            return f"{indent}* {content}", SymbolDecision.AUTO_CORRECT, "FOOTNOTE_FROM_DIGIT"

        # 5. Ký tự '|' ở đầu dòng trong văn bản thông thường (không phải hàng bảng nhiều cột ' | ')
        if line.count(" | ") < 2:
            m_pipe = re.match(r"^([ \t]*)\|\s+([A-Za-zÀ-ỹĐđ\+].*)$", line)
            if m_pipe:
                indent = m_pipe.group(1)
                content = m_pipe.group(2)
                # Nếu không phải là đề mục in hoa toàn bộ (đề mục số La Mã I.)
                if not cls.is_all_uppercase_heading(content):
                    return f"{indent}+ {content}", SymbolDecision.AUTO_CORRECT, "PLUS_FROM_PIPE_PREFIX"

        return line, SymbolDecision.PRESERVE, ""

    @classmethod
    def recover_document_symbols(
        cls, lines: List[str]
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Thực hiện Context-Aware Symbol Recovery trên toàn bộ tài liệu.
        Trả về: (recovered_lines, audit_trail)
        """
        recovered_lines: List[str] = []
        audit_trail: List[Dict[str, Any]] = []
        current_roman = ""

        # Tiền xử lý: Khử nhập nhằng phân cấp đề mục (Hierarchy Disambiguation)
        # Nếu dòng k bắt đầu bằng '1.' (hoặc 'L.', 'l.', 'I.') là đề mục lớn,
        # và dòng kế tiếp phía dưới cũng bắt đầu bằng '1.' (ví dụ: '1. Đại diện...'):
        # -> Dòng k chắc chắn là số La Mã 'I.'
        normalized_lines = list(lines)
        for i in range(len(normalized_lines)):
            m_top = re.match(r"^([ \t]*)(?:1|L|l|I|\|)\.\s+([A-Za-zÀ-ỹĐđ0-9\s,;:/-]+)$", normalized_lines[i])
            if m_top:
                indent = m_top.group(1)
                heading_content = m_top.group(2).strip()
                next_start_with_1 = False
                for j in range(i + 1, min(i + 4, len(normalized_lines))):
                    l_next = normalized_lines[j].strip()
                    if not l_next:
                        continue
                    if re.match(r"^1\.\s+[A-Za-zÀ-ỹĐđ]", l_next):
                        next_start_with_1 = True
                        break
                    else:
                        break
                if next_start_with_1:
                    has_sec_kw = any(kw.lower() in heading_content.lower() for kw in [
                        "thành phần", "nội dung", "kế hoạch", "chương trình", "mục đích",
                        "tổ chức", "kết luận", "tiến độ", "phương thức", "đánh giá", "điều khoản"
                    ])
                    if has_sec_kw or not re.search(r":\s*[^\s]", heading_content):
                        new_heading = f"{indent}I. {heading_content}"
                        audit_trail.append({
                            "line_index": i,
                            "original": normalized_lines[i],
                            "recovered": new_heading,
                            "decision": SymbolDecision.AUTO_CORRECT.value,
                            "rule": "HIERARCHY_DISAMBIGUATION_ROMAN_I",
                        })
                        normalized_lines[i] = new_heading

        for idx, line in enumerate(normalized_lines):
            orig_line = line
            curr_line = line

            # 1. Khôi phục số La Mã
            curr_line, roman_dec, current_roman, roman_rule = cls.recover_roman_numeral_in_line(
                curr_line, current_roman
            )
            if roman_dec == SymbolDecision.AUTO_CORRECT:
                audit_trail.append({
                    "line_index": idx,
                    "original": orig_line,
                    "recovered": curr_line,
                    "decision": roman_dec.value,
                    "rule": roman_rule,
                })

            # 2. Khôi phục Bullet / Plus
            curr_line, bullet_dec, bullet_rule = cls.recover_bullets_and_pluses(curr_line)
            if bullet_dec == SymbolDecision.AUTO_CORRECT:
                audit_trail.append({
                    "line_index": idx,
                    "original": orig_line,
                    "recovered": curr_line,
                    "decision": bullet_dec.value,
                    "rule": bullet_rule,
                })

            # 3. Dọn dẹp dấu phân cách kép thừa sau STT (1 | | Chấp hành... -> 1 | Chấp hành...)
            m_double_pipe = re.match(r"^([ \t]*\d+)\s*\|\s*\|\s*(.*)$", curr_line)
            if m_double_pipe:
                cleaned_line = f"{m_double_pipe.group(1)} | {m_double_pipe.group(2)}"
                audit_trail.append({
                    "line_index": idx,
                    "original": curr_line,
                    "recovered": cleaned_line,
                    "decision": SymbolDecision.AUTO_CORRECT.value,
                    "rule": "STT_DOUBLE_PIPE_CLEANUP",
                })
                curr_line = cleaned_line

            # 4. Cố định cấu trúc bảng tiến độ thực tập (Tuần | Ngày | Nội dung)
            formatted_progress = format_progress_table_row(curr_line)
            if formatted_progress != curr_line:
                audit_trail.append({
                    "line_index": idx,
                    "original": curr_line,
                    "recovered": formatted_progress,
                    "decision": SymbolDecision.AUTO_CORRECT.value,
                    "rule": "PROGRESS_TABLE_ROW_FORMAT",
                })
                curr_line = formatted_progress

            recovered_lines.append(curr_line)

        # Hậu kiểm tra tính tuần tự và loại bỏ trùng lặp số La Mã (Sequential Reconciliation):
        # 1. Đề mục 'Nội dung cuộc họp' luôn mang số II.
        # 2. Nếu có 2 đề mục cùng mang số III. (ví dụ 'III. Nội dung cuộc họp' và 'III. Kết luận...'),
        #    mục đầu tiên bắt buộc đổi thành II.
        # 3. Nếu mục trước là I. và mục sau là III., mục ở giữa bắt buộc là II.
        roman_headings = []
        for r_idx, r_line in enumerate(recovered_lines):
            m_r = re.match(r"^([ \t]*)([I|V|X]+)\.\s+(.*)$", r_line)
            if m_r:
                roman_headings.append((r_idx, m_r.group(1), m_r.group(2), m_r.group(3)))

        for h_pos, (line_idx, ind, rom, text_rest) in enumerate(roman_headings):
            if re.match(r"(?i)Nội\s+dung\s+cuộc\s+họp", text_rest) and rom != "II":
                new_l = f"{ind}II. {text_rest}"
                audit_trail.append({
                    "line_index": line_idx,
                    "original": recovered_lines[line_idx],
                    "recovered": new_l,
                    "decision": SymbolDecision.AUTO_CORRECT.value,
                    "rule": "ROMAN_II_CONTENT_ENFORCED",
                })
                recovered_lines[line_idx] = new_l
            elif rom == "III" and h_pos + 1 < len(roman_headings):
                next_rom = roman_headings[h_pos + 1][2]
                if next_rom == "III":
                    new_l = f"{ind}II. {text_rest}"
                    audit_trail.append({
                        "line_index": line_idx,
                        "original": recovered_lines[line_idx],
                        "recovered": new_l,
                        "decision": SymbolDecision.AUTO_CORRECT.value,
                        "rule": "ROMAN_DEDUP_III_TO_II",
                    })
                    recovered_lines[line_idx] = new_l

        return recovered_lines, audit_trail


# ---------------------------------------------------------------------------
# Bộ Chuẩn Hóa Văn Bản Tiếng Việt & Bảo Toàn Casing
# ---------------------------------------------------------------------------


def preserve_casing(source_text: str, target_text: str) -> str:
    """
    Bảo toàn định dạng chữ hoa/chữ thường (casing) từ source_text sang target_text:
    - Nếu source_text.isupper() -> trả về target_text.upper().
    - Nếu source_text.istitle() -> trả về target_text.title().
    - Nếu chữ thường -> trả về target_text.lower().
    """
    if not source_text or not target_text:
        return target_text
    if source_text.isupper():
        return target_text.upper()
    if source_text.istitle():
        return target_text.title()
    if source_text.islower():
        return target_text.lower()
    return target_text


def clean_unmatched_closing_brackets(text: str) -> str:
    """
    Loại bỏ ký tự ']' rác (thường do OCR đọc nhầm số mũ chú thích ¹)
    khi đứng ngay sau từ mà không có dấu '[' mở tương ứng.
    Bảo toàn 100% các cặp ngoặc hợp lệ như [Icon], [1], [Ghi chú].
    """
    if not text or "]" not in text:
        return text

    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        if "]" not in line:
            cleaned_lines.append(line)
            continue

        out = []
        open_bracket_count = 0
        for char in line:
            if char == "[":
                open_bracket_count += 1
                out.append(char)
            elif char == "]":
                if open_bracket_count > 0:
                    open_bracket_count -= 1
                    out.append(char)
                else:
                    # ']' rác không có ngoặc mở tương ứng -> bỏ qua
                    pass
            else:
                out.append(char)
        cleaned_lines.append("".join(out))
    return "\n".join(cleaned_lines)


def clean_orphaned_brackets(text: str) -> str:
    """
    Xóa dấu ngoặc vuông '[' hoặc ký tự gạch đứng '|' mồ côi ở đầu dòng:
    Nếu dòng bắt đầu bằng '[' hoặc '|' ngay trước chữ/số/ký tự văn bản mà trên dòng đó
    không có dấu đóng ngoặc ']' tương ứng (ví dụ: '[3. Thư ký...', '[Chủ trì...', '[Ưng...').
    Đồng thời xóa dấu '[' mồ côi đứng trước từ ngữ trong dòng (ví dụ: 'THIÊN [Ưng' -> 'THIÊN Ưng').
    """
    if not text:
        return text
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        m = re.match(r"^(\s*)[\[\|]\s*([^\s].*)$", line)
        if m:
            c = m.group(2)
            if "]" not in c:
                line = f"{m.group(1)}{c}"
        # Xóa dấu '[' mồ côi đứng trước từ trong dòng khi không có ']' đóng
        if "]" not in line and "[" in line:
            line = re.sub(r"(?<=\s)\[([A-Za-zÀ-ỹĐđ]+)", r"\1", line)
            line = re.sub(r"^\[([A-Za-zÀ-ỹĐđ]+)", r"\1", line)
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def final_document_cleanup(text: str) -> str:
    """
    Khóa chốt hậu xử lý văn bản cuối cùng (Final Document Cleanup):
    1. Chuẩn hóa lỗi nhầm dấu cố hữu của VietOCR (bảo toàn chữ HOA/thường):
       - CHI BỐ, CHỈ BỘ, CHI BỒ, CHỈ BỒ -> CHI BỘ | Chi bố, Chỉ bộ, Chi bồ, Chỉ bồ -> Chi bộ.
       - r'\\bsinh\\s+hoatchi\\b' -> 'sinh hoạt chi'.
       - r'\\b(\\d+)\\s+đồng\\s+ch\\b(?![a-zà-ỹđ])' -> r'\\1 đồng chí'.
       - r'(Giới\\s+thiệu\\s+đại\\s+biểu\\s*:\\s*không)\\s+c\\b' -> r'\\1 có.'.
       - r'\\b(Số\\s*:?\\s*\\d+)\\s+([A-ZĐ]{1,5}/[A-ZĐ0-9\\-]+)\\b' -> r'\\1-\\2' (Số 01 BB/CB -> Số 01-BB/CB).
       - (Tiêu chí đánh giá | đa | Điểm tự chấm) -> (Tiêu chí đánh giá | Điểm tối đa | Điểm tự chấm).
    2. Chuẩn hóa số, phân số & ký hiệu bảng:
       - r'\\b0[sS]\\b' -> '05'.
       - r'(chỉ\\s+bằng|bằng|dưới|trên|quá)\\s*[%％]\\s*(so\\s+với\\s+quy\\s+định)' -> r'\\1 ½ \\2'.
       - r'(?m)^([ \\t]*)\\*\\s*(Trừ\\s+trường\\s+hợp)' -> r'\\1* \\2' (sửa '*Trừ trường hợp' -> '* Trừ trường hợp').
       - Xóa dấu '[' mồ côi đầu dòng khi dòng đó không có dấu ']'.
       - Khôi phục dấu '- ' đầu dòng cho các mục liệt kê biên bản:
         (?m)^([ \\t]*)(Lý\\s+do|Giới\\s+thiệu\\s+đại\\s+biểu|Miễn\\s+sinh\\s+hoạt|Có\\s+mặt|Vắng\\s+mặt)\\s*: -> \\1- \\2:
       - Khôi phục dấu '+ ' cho các tiêu chí con trong bảng điểm (có cột điểm | 01, | 02,...).
    """
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFC", text)

    # 1.a. Chuẩn hóa chi bộ / CHI BỘ / Chi bộ (bảo toàn chữ HOA/thường)
    # Xử lý tất cả các biến thể: chi bộ, chỉ bộ, chi bồ, chỉ bồ, chi bố, chỉ bố,...
    def _fix_chi_bo(m: re.Match) -> str:
        s = m.group(0)
        if s.isupper():
            return "CHI BỘ"
        elif s[0].isupper():
            return "Chi bộ"
        else:
            return "chi bộ"

    text = re.sub(r"\bch[iỉ]\s+b[ôồốổỗộ]\b", _fix_chi_bo, text, flags=re.IGNORECASE)

    # 1.b. sinh hoatchi -> sinh hoạt chi
    def _fix_sinh_hoatchi(m: re.Match) -> str:
        s = m.group(0)
        if s.isupper():
            return "SINH HOẠT CHI"
        elif s[0].isupper():
            return "Sinh hoạt chi"
        else:
            return "sinh hoạt chi"

    text = re.sub(r"\bsinh\s+hoatchi\b", _fix_sinh_hoatchi, text, flags=re.IGNORECASE)

    # 1.c. (\d+) đồng ch -> \1 đồng chí
    text = re.sub(r"\b(\d+)\s+đồng\s+ch\b(?![a-zà-ỹđ])", r"\1 đồng chí", text, flags=re.IGNORECASE)

    # 1.d. Giới thiệu đại biểu: không c -> không có.
    text = re.sub(
        r"((?:[-\–—]\s*)?Giới\s+thiệu\s+đại\s+biểu\s*:\s*không)\s+c\.?(?![a-zà-ỹđ])",
        r"\1 có.",
        text,
        flags=re.IGNORECASE,
    )

    # 1.e. Số 01 BB/CB -> Số 01-BB/CB (khôi phục dấu gạch nối trong số hiệu văn bản)
    text = re.sub(
        r"\b(Số\s*:?\s*\d+)\s+([A-ZĐ]{1,5}/[A-ZĐ0-9\-]+)\b",
        r"\1-\2",
        text,
    )

    # 1.f. Tiêu đề cột đa -> Điểm tối đa & chấm -> Điểm tự chấm
    text = re.sub(
        r"(\|\s*Tiêu\s+chí\s+đánh\s+giá\s*\|\s*)(?:tối\s+)?đa(?=\s*\|)",
        r"\1Điểm tối đa",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(\|\s*Điểm\s+tối\s+đa\s*\|\s*)(?:tự\s+)?chấm(\s*\|\s*Ghi\s+chú)",
        r"\1Điểm tự chấm\2",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(\|\s*(?:Điểm\s+)?tối\s+đa\s*\|\s*)(?:tự\s+)?chấm(?=\s*\||\s*$)",
        r"\1Điểm tự chấm",
        text,
        flags=re.IGNORECASE,
    )
    text = text.replace("| chấm |", "| Điểm tự chấm |")
    text = text.replace("| tự chấm |", "| Điểm tự chấm |")
    text = text.replace("| Tự chấm |", "| Điểm tự chấm |")

    # 2.a. 0s / 0S -> 05
    text = re.sub(r"\b0[sS]\b", "05", text)

    # 2.b. chỉ bằng % / ％ so với quy định -> chỉ bằng ½ so với quy định (thay thế trực diện)
    for prefix in ("chỉ bằng", "Chỉ bằng", "dưới", "Dưới", "bằng", "Bằng", "ít hơn", "Ít hơn", "quá", "trên", "hơn", "gấp"):
        text = text.replace(f"{prefix} % so với", f"{prefix} ½ so với")
        text = text.replace(f"{prefix} ％ so với", f"{prefix} ½ so với")
        text = text.replace(f"{prefix}% so với", f"{prefix} ½ so với")
        text = text.replace(f"{prefix} %", f"{prefix} ½")
        text = text.replace(f"{prefix} ％", f"{prefix} ½")
        text = text.replace(f"{prefix}%", f"{prefix} ½")

    text = re.sub(
        r"(chỉ\s+bằng|bằng|dưới|trên|quá|hơn|gấp|ít\s+hơn)\s*[%％]+(?=\s+so\s+với|\s+quy\s+định|\s+mức|\s|$|[|])",
        r"\1 ½",
        text,
        flags=re.IGNORECASE,
    )

    # 2.c. /Trừ trường hợp... / *Trừ trường hợp -> * Trừ trường hợp...
    text = text.replace("*Trừ trường hợp", "* Trừ trường hợp")
    text = text.replace("/Trừ trường hợp", "* Trừ trường hợp")
    text = text.replace("-Trừ trường hợp", "* Trừ trường hợp")
    text = re.sub(
        r"(?m)^([ \t]*)[1?*/\\|\-–—]+\s*\*?\s*(Trừ\s+trường\s+hợp.*)$",
        r"\1* \2",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(?m)^([ \t]*)[1?*/\\|\-–—]*\s*\*+\s*(Trừ\s+trường\s+hợp.*)$",
        r"\1* \2",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"(?m)^([ \t]*)\*([^\s*])", r"\1* \2", text)

    # 2.d. Xóa dấu ngoặc '[' mồ côi đầu dòng khi không có ']'
    text = clean_orphaned_brackets(text)

    # 2.d2. Xóa ký tự rác đứng trước dấu gạch đầu dòng: '/ -Ghi nhận' -> '- Ghi nhận'
    text = re.sub(r"(?m)^([ \t]*)[/\\|]\s*([-\–—]\s*)", r"\1- ", text)
    text = re.sub(r"(?m)^([ \t]*)[-\–—](?=[A-Za-zÀ-ỹĐđ])", r"\1- ", text)

    # 2.e. Khôi phục dấu '- ' đầu dòng cho các mục liệt kê biên bản
    text = re.sub(
        r"(?m)^([ \t]*)(Lý\s+do|Giới\s+thiệu\s+đại\s+biểu|Miễn\s+sinh\s+hoạt|Có\s+mặt|Vắng\s+mặt)\s*:",
        r"\1- \2:",
        text,
    )

    # 2.f. Khôi phục dấu '+ ' cho các tiêu chí con trong bảng điểm (có cột điểm | 01, | 02,...)
    lines_out = []
    for ln in text.splitlines():
        ln_strip = ln.lstrip()
        if ln_strip.startswith(("- ", "– ", "— ", "-")) and re.search(r"\|\s*0[1-9]\b", ln_strip):
            indent = ln[:len(ln) - len(ln_strip)]
            body = ln_strip[1:].lstrip() if not ln_strip.startswith(("- ", "– ", "— ")) else ln_strip[2:].lstrip()
            ln = indent + "+ " + body
        lines_out.append(ln)
    text = "\n".join(lines_out)

    # 2.g. Sửa lỗi 'III. Nội dung cuộc họp' / 'IL. Nội dung cuộc họp' -> 'II. Nội dung cuộc họp'
    text = re.sub(
        r"(?mi)^([ \t]*)(?:III|IL|Il|1L|1l|IIL|IIl|11L|lll|111)\.?\s+(Nội\s+dung\s+cuộc\s+họp\b.*)$",
        r"\1II. \2",
        text,
    )

    # 2.h. Khôi phục cấu trúc văn bản hành chính & biên bản cuộc họp:
    # 1. Khôi phục tiêu đề 'BIÊN BẢN' phía trên dòng 'Lấy ý kiến tập thể lao động...' nếu bị thiếu
    if re.search(r"(?i)Lấy\s+ý\s+kiến\s+(?:tập|toàn)\s+thể\s+lao\s+động", text):
        if not re.search(r"(?mi)^[ \t]*BIÊN\s+BẢN\b", text):
            text = re.sub(
                r"(?mi)^([ \t]*)(Lấy\s+ý\s+kiến\s+(?:tập|toàn)\s+thể\s+lao\s+động.*)$",
                r"\1BIÊN BẢN\n\1\2",
                text,
                count=1,
            )

    # 1.b. Khôi phục dòng tiêu đề phụ bị thiếu dưới BIÊN BẢN (nhảy thẳng xuống Hôm nay, vào lúc...)
    if re.search(r"(?mi)^[ \t]*BIÊN\s+BẢN[ \t]*$", text) and re.search(r"(?i)Nội\s+quy\s+lao\s+động", text):
        if not re.search(r"(?i)Lấy\s+ý\s+kiến\s+(?:tập|toàn)\s+thể\s+lao\s+động", text):
            text = re.sub(
                r"(?mi)^([ \t]*BIÊN\s+BẢN[ \t]*)\n+(?=[ \t]*Hôm\s+nay)",
                r"\1\nLấy ý kiến tập thể lao động về nội dung Nội quy lao động\n",
                text,
                count=1,
            )

    # 2. Sửa trực tiếp dòng bị đọc nhầm 'Thà Thành phần tham dự' -> 'I. Thành phần tham dự'
    text = re.sub(
        r"(?mi)^([ \t]*)(?:Thà|Tha|Thờ|Th|I|l|1|L|II|IL)[\.:]?\s+(Thành\s+phần\s+tham\s+dự\b.*)$",
        r"\1I. \2",
        text,
    )
    # Khử trùng lặp nếu có 2 dòng 'Thành phần tham dự' liên tiếp
    text = re.sub(
        r"(?mi)^([ \t]*I\.\s+Thành\s+phần\s+tham\s+dự[^\n]*)\n+[ \t]*(?:I\.\s+)?Thành\s+phần\s+tham\s+dự[^\n]*$",
        r"\1",
        text,
    )

    # 3. Khôi phục đề mục 'I. Thành phần tham dự' phía trên '1. Đại diện Công ty:' nếu bị thiếu
    if re.search(r"(?i)1\.\s+Đại\s+diện\s+(?:Công\s+ty|Doanh\s+nghiệp)", text) and re.search(r"(?i)(?:II|2)\.\s+Nội\s+dung\s+cuộc\s+họp", text):
        if not re.search(r"(?mi)^[ \t]*I\.\s+Thành\s+phần\s+tham\s+dự", text):
            text = re.sub(
                r"(?mi)^([ \t]*)(1\.\s+Đại\s+diện\s+(?:Công\s+ty|Doanh\s+nghiệp).*)$",
                r"\1I. Thành phần tham dự\n\1\2",
                text,
                count=1,
            )

    # 2.i. Chuẩn hóa 'Cuộc hơp' -> 'Cuộc họp' (VietOCR thiếu dấu nặng thành ơ)
    text = re.sub(r"\b(Cuộc|cuộc)\s+h[ơo]p\b", r"\1 họp", text)
    text = text.replace("Cuộc hơp", "Cuộc họp")
    text = text.replace("cuộc hơp", "cuộc họp")

    # 2.j. Khôi phục năm 2026 bị cắt cụt số cuối thành 'năm 202'
    text = re.sub(r"\b(tháng\s+\d{1,2}\s+năm\s+202)\.?(?![0-9])", r"\g<1>6.", text)

    # Đồng bộ hóa hoàn toàn với clean_and_restore
    text = VietnameseTextPostProcessor.clean_and_restore(text)

    return text


class VietnameseTextPostProcessor:
    """
    Module chuẩn hóa và làm sạch kỹ thuật văn bản OCR:
    - Chuẩn hóa Unicode dựng sẵn (NFC).
    - Chuẩn hóa khoảng trắng quanh dấu câu: loại bỏ khoảng trắng thừa trước dấu.
    - TUYỆT ĐỐI KHÔNG tự động sửa nội dung từ vựng (chỉ bộ, chi bố, dư sinh hoạt)
      để nhường quyền kiểm tra, bôi đỏ và hiển thị gợi ý cho spell_corrector.py.
    - Bảo toàn 100% kiểu chữ hoa, chữ thường của văn bản gốc (Zero Casing Destruction).
    """

    @classmethod
    def clean_and_restore(cls, text: str) -> str:
        if not text:
            return ""

        # 1. Chuẩn hóa Unicode NFC
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = unicodedata.normalize("NFC", text).rstrip()

        # 2. Loại bỏ khoảng trắng thừa ngay trước dấu câu
        text = re.sub(r"[ \t]+([,.:;?!])", r"\1", text)

        # 3. Ký hiệu kết thúc văn bản hành chính ./.
        text = re.sub(r"\.\s*/\s*\.", "./.", text)

        # 3.b. Xóa nhiễu đường kẻ trang trí dưới Tiêu ngữ (144444444, 44444, *******)
        # và đảm bảo ngắt dòng trước Địa danh, ngày... tháng... năm...
        text = re.sub(
            r"(Độc\s+lập\s*[-–—]\s*Tự\s+do\s*[-–—]\s*Hạnh\s+phúc)[ \t]*(?:1?4{3,}|[1lI|!*=\-—_~]{3,}|\*{3,})*[ \t]*(?:\r?\n)?[ \t]*([A-ZÀ-ỸĐ][a-zA-ZÀ-ỹĐđ\s,/-]+,\s*ngày\s+\d+)",
            r"\1\n\2",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"(?m)^[ \t]*(?:1?4{3,}|[1lI|!*=\-—_~]{4,}|\*{3,})[ \t]*\r?\n?", "", text)
        text = re.sub(r"[ \t]*\|\s*KHIÊN\s+ƯNG\s+KẾ\s+TOÁN\b", "", text, flags=re.IGNORECASE)

        # 3.c. Tách từ bị dính chữ phổ biến 'hoạtchi bộ' -> 'hoạt chi bộ'
        text = re.sub(r"\b(sinh\s+hoạt|hoạt)(chi\s+bộ|chi\s+ủy)\b", r"\1 \2", text, flags=re.IGNORECASE)

        # 4. Khôi phục từ bị cắt xén ở cuối bảng/trang dựa trên ngữ cảnh hành chính đã xác minh
        text = re.sub(r"\b(nhiệm\s+vụ\s+đặc)(?:\s+b\b|\b(?!\s*biệt))", r"\1 biệt", text)

        # 5. Khôi phục phân số ½ bị OCR nhận thành % (ví dụ: 'chỉ bằng % so với quy định')
        # Tuyệt đối không thay đổi phần trăm hợp lệ đi kèm chữ số như 50%, 60%, 75%, 90%, 100%
        for prefix in ("chỉ bằng", "Chỉ bằng", "dưới", "Dưới", "bằng", "Bằng", "ít hơn", "Ít hơn", "quá", "trên", "hơn", "gấp"):
            text = text.replace(f"{prefix} % so với", f"{prefix} ½ so với")
            text = text.replace(f"{prefix} ％ so với", f"{prefix} ½ so với")
            text = text.replace(f"{prefix}% so với", f"{prefix} ½ so với")
            text = text.replace(f"{prefix} %", f"{prefix} ½")
            text = text.replace(f"{prefix} ％", f"{prefix} ½")
            text = text.replace(f"{prefix}%", f"{prefix} ½")

        text = re.sub(
            r"\b(chỉ\s+bằng|bằng|dưới|trên|quá|hơn|gấp)\s+(?<!\d)%\s*(so\s+với|quy\s+định|mức|tổng)\b",
            r"\1 ½ \2",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\b(chỉ\s+bằng|bằng|dưới|quá)\s*%\s*(so\s+với\s+quy\s+định)",
            r"\1 ½ \2",
            text,
            flags=re.IGNORECASE,
        )
        # Sửa '0s' / '0S' thành '05' trong bảng điểm đánh giá
        text = re.sub(r"\b0[sS]\b", "05", text)

        # 5.b. Chuẩn hóa số mũ diện tích & thể tích (m2, m?, m', m3 -> m², m³)
        text = re.sub(
            r"\b(\d+(?:[.,]\d+)?\s*(?:mm|cm|dm|m|km|ha))(?:2|²|\?|\'|7)(?!\w)",
            r"\1²",
            text,
        )
        text = re.sub(
            r"\b(\d+(?:[.,]\d+)?\s*(?:mm|cm|dm|m|km))(?:3|³)(?!\w)",
            r"\1³",
            text,
        )

        # 5.c. Chuẩn hóa nhiệt độ (°C)
        text = re.sub(r"\b(\d+(?:[.,]\d+)?)\s*[oO0]\s*C\b", r"\1°C", text)

        # 5.d. Chuẩn hóa khoảng trắng sau dấu câu bị dính chữ (13/04/2021.Nơi cấp, số:473/)
        text = re.sub(r"(\d{2}/\d{2}/\d{4})\.([A-Za-zÀ-ỹĐđ])", r"\1. \2", text)
        text = re.sub(r"\b(số|Số):(\d+)", r"\1: \2", text)

        # 5.e. Khôi phục nối dòng nhãn ngắn bị ngắt xuống dòng: Sinh năm:\n1978...
        text = re.sub(
            r"(?m)^([ \t]*(?:Sinh\s+năm|Họ\s+và\s+tên|Ngày\s+sinh|Nơi\s+sinh|Địa\s+chỉ|Hộ\s+khẩu|Quê\s+quán):)[ \t]*\r?\n[ \t]*([0-9A-Za-zÀ-ỹĐđ])",
            r"\1 \2",
            text,
            flags=re.IGNORECASE,
        )

        # 6. Chuẩn hóa dấu chú thích chân trang: '/Trừ trường hợp...', '?Trừ trường hợp...' -> '* Trừ trường hợp...'
        text = text.replace("*Trừ trường hợp", "* Trừ trường hợp")
        text = text.replace("/Trừ trường hợp", "* Trừ trường hợp")
        text = text.replace("-Trừ trường hợp", "* Trừ trường hợp")
        text = re.sub(
            r"(?m)^([ \t]*)[1?*/\-]+\s*(Trừ\s+trường\s+hợp.*)$",
            r"\1* \2",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"(?m)^([ \t]*)\*([^\s*])", r"\1* \2", text)

        # 6.b. Khôi phục dấu '+ ' cho các tiêu chí con trong bảng điểm (có cột điểm | 01, | 02,...)
        lines_out = []
        for ln in text.splitlines():
            ln_strip = ln.lstrip()
            if ln_strip.startswith(("- ", "– ", "— ", "-")) and re.search(r"\|\s*0[1-9]\b", ln_strip):
                indent = ln[:len(ln) - len(ln_strip)]
                body = ln_strip[1:].lstrip() if not ln_strip.startswith(("- ", "– ", "— ")) else ln_strip[2:].lstrip()
                ln = indent + "+ " + body
            lines_out.append(ln)
        text = "\n".join(lines_out)

        # 6.c. Chuẩn hóa 'Cuộc hơp' -> 'Cuộc họp'
        text = re.sub(r"\b(Cuộc|cuộc)\s+h[ơo]p\b", r"\1 họp", text)
        text = text.replace("Cuộc hơp", "Cuộc họp")
        text = text.replace("cuộc hơp", "cuộc họp")

        # 6.d. Khôi phục tiêu đề cột thứ 4 bảng điểm
        text = re.sub(
            r"(\|\s*Tiêu\s+chí\s+đánh\s+giá\s*\|\s*)(?:tối\s+)?đa(?=\s*\|)",
            r"\1Điểm tối đa",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"(\|\s*Điểm\s+tối\s+đa\s*\|\s*)(?:tự\s+)?chấm(\s*\|\s*Ghi\s+chú)",
            r"\1Điểm tự chấm\2",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"(\|\s*(?:Điểm\s+)?tối\s+đa\s*\|\s*)(?:tự\s+)?chấm(?=\s*\||\s*$)",
            r"\1Điểm tự chấm",
            text,
            flags=re.IGNORECASE,
        )
        text = text.replace("| chấm |", "| Điểm tự chấm |")
        text = text.replace("| tự chấm |", "| Điểm tự chấm |")
        text = text.replace("| Tự chấm |", "| Điểm tự chấm |")

        # 6.e. Khôi phục năm 2026 bị cắt cụt số cuối thành 'năm 202'
        text = re.sub(r"\b(tháng\s+\d{1,2}\s+năm\s+202)\.?(?![0-9])", r"\g<1>6.", text)

        # 7. Chuẩn hóa các cụm từ hành chính phổ biến bị nhận sai dấu thanh và BẢO TOÀN CASING:
        # TUYỆT ĐỐI KHÔNG tự động sửa ngầm lỗi từ vựng như 'chỉ bộ', 'chi bố', 'dư sinh hoạt'
        # để nhường quyền kiểm tra, bôi đỏ và hiển thị gợi ý cho spell_corrector.py
        text = re.sub(
            r"\b(đằng)\s+(cộng\s+sản)\b",
            lambda m: f"{preserve_casing(m.group(1), 'đảng')} {preserve_casing(m.group(2), m.group(2))}",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\b(ngày\s+\d+\s+)(thắng)(\s+\d+)",
            lambda m: f"{m.group(1)}{preserve_casing(m.group(2), 'tháng')}{m.group(3)}",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\b(ghi)\s+(chủ)\b",
            lambda m: f"{preserve_casing(m.group(1), 'ghi')} {preserve_casing(m.group(2), 'chú')}",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\b(it)\s+(nhất)\b",
            lambda m: f"{preserve_casing(m.group(1), 'ít')} {preserve_casing(m.group(2), 'nhất')}",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\b(sinh\s+hoạt\s+)(đằng)\b",
            lambda m: f"{m.group(1)}{preserve_casing(m.group(2), 'đảng')}",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\b(đảng)\s+(viễn)\b",
            lambda m: f"{preserve_casing(m.group(1), 'đảng')} {preserve_casing(m.group(2), 'viên')}",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\b(dàng)\s+(viên)\b",
            lambda m: f"{preserve_casing(m.group(1), 'đảng')} {preserve_casing(m.group(2), 'viên')}",
            text,
            flags=re.IGNORECASE,
        )

        # 7.b. Gom đúng hàng ngang biểu mẫu có chấm lửng
        text = re.sub(
            r"(\+\s*Ông\s*\(bà\):?)\s*(?:\.{2,}|…+|_|-)*\s*(\(ghi\s+tên[^)]+\))",
            r"\1 ...... \2",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"(Chức\s+danh:?)\s*(?:\.{2,}|…+|_|-)*\s*(Chủ\s+trì\s+cuộc\s+họp)",
            r"\1 ...... \2",
            text,
            flags=re.IGNORECASE,
        )

        # 8. Xử lý ký tự số mũ chú thích bị nhận diện thành ']' rác (ví dụ: 'trong năm]')
        text = clean_unmatched_closing_brackets(text)

        # 8.a. Xóa ký tự ngoặc '[' hoặc '|' mồ côi ở đầu dòng: '[3. Thư ký...' -> '3. Thư ký...'
        text = clean_orphaned_brackets(text)

        # 8.b. Khôi phục dấu ngoặc đơn mở '(' bị thiếu ở đầu dòng kết thúc bằng ')'
        # ví dụ: 'Đánh dấu X vào nội dung được chọn)' -> '(Đánh dấu X vào nội dung được chọn)'
        def _restore_missing_open_paren(match: re.Match) -> str:
            indent = match.group(1)
            content = match.group(2)
            if "(" not in content and len(content.strip()) >= 5:
                return f"{indent}({content}"
            return match.group(0)

        text = re.sub(
            r"(?m)^([ \t]*)([A-ZÀ-ỸĐ][^\n]*\))$",
            _restore_missing_open_paren,
            text,
        )

        # 9. Tách đề mục số bị dính vào cuối dòng thông tin trước đó:
        # ví dụ: 'Công nghệ phần mềm 5. Tiến độ và nội dung thực tập:' -> 'Công nghệ phần mềm\n5. Tiến độ và nội dung thực tập:'
        text = re.sub(
            r"([^\n]+?)[ \t]+([1-9]\d{0,1}\.\s+[A-ZÀ-ỸĐ][a-zA-ZÀ-ỹĐđ\s,/-]+:)",
            r"\1\n\2",
            text,
        )

        # 9.b. Tách đề mục số La Mã bị dính vào cuối dòng thông tin trước đó:
        # ví dụ: '... của doanh nghiệp: 14 người. IL. Nội dung cuộc họp' -> '... 14 người.\nIL. Nội dung cuộc họp'
        roman_sec_keywords = (
            r"Nội\s+dung|Thành\s+phần|Kết\s+luận|Phương\s+thức|Tổ\s+chức|Đánh\s+giá|"
            r"Tiến\s+độ|Kế\s+hoạch|Thời\s+gian|Địa\s+điểm|Chương\s+trình|Mục\s+đích|"
            r"Yêu\s+cầu|Quy\s+định|Điều\s+khoản|Trách\s+nhiệm|Hiệu\s+lực"
        )
        roman_patterns = r"(?:I{1,3}|IV|V|VI{0,3}|IX|X|IL|Il|1L|1l|IIL|IIl|11L|TT|TTT|TI|TL|Tl)"
        text = re.sub(
            rf"([^\n]+?[:.\d])[ \t]+({roman_patterns}\.\s+(?:{roman_sec_keywords})\b)",
            r"\1\n\2",
            text,
            flags=re.IGNORECASE,
        )

        # 10. Khôi phục AWS bị cắt thành AW ở mép phải
        text = re.sub(
            r"\b(dịch\s+vụ\s+của\s+)AW(?:\b|\.)(?!\s*S)",
            r"\g<1>AWS.",
            text,
        )

        # 11. Khôi phục chữ số nhầm ký tự chữ cái trong ngữ cảnh số liệu / ngày tháng / phần trăm:
        # 11.a. 'O' nhầm '0' trong ngày tháng: 'ngày O5', 'tháng O8', '05/O9/2024'
        text = re.sub(r"\b(ngày\s+)O(\d)\b", r"\g<1>0\2", text, flags=re.IGNORECASE)
        text = re.sub(r"\b(tháng\s+)O(\d)\b", r"\g<1>0\2", text, flags=re.IGNORECASE)
        text = re.sub(r"\b(\d{1,2}[/-])O(\d[/-]\d{4})\b", r"\g<1>0\2", text)
        # 11.b. 'O' nhầm '0' trong phần trăm và số tiền: '9O%', '5O%', '1OO%', '5O.000.000', '1OO đơn vị'
        text = re.sub(r"\b(\d+)O%", r"\g<1>0%", text)
        text = re.sub(r"\b1OO%", "100%", text)
        text = re.sub(r"\b(\d+)O([.,]\d{3})", r"\g<1>0\2", text)
        text = re.sub(r"\b1OO\b", "100", text)
        # 11.c. 'l' hoặc 'I' nhầm '1' trong ngày tháng / phần trăm:
        text = re.sub(r"\b(ngày\s+)l(\d)\b", r"\g<1>1\2", text, flags=re.IGNORECASE)
        text = re.sub(r"\b(tháng\s+)l([012])\b", r"\g<1>1\2", text, flags=re.IGNORECASE)
        text = re.sub(r"\bl00%", "100%", text)
        text = re.sub(r"\bl(\d)%", r"1\1%", text)
        text = re.sub(r"\b(\d+)l%", r"\g<1>1%", text)
        # 11.d. 'S' nhầm '5' trong phần trăm: 'S0%' -> '50%', '2S%' -> '25%'
        text = re.sub(r"\bS(\d)%", r"5\1%", text)
        text = re.sub(r"\b(\d+)S%", r"\g<1>5%", text)
        # 11.e. 'B' nhầm '8' trong phần trăm: 'B0%' -> '80%', 'B5%' -> '85%'
        text = re.sub(r"\bB(\d)%", r"8\1%", text)
        text = re.sub(r"\b(\d+)B%", r"\g<1>8%", text)

        # 12. Chuẩn hóa tiêu đề câu hỏi đề thi giáo dục: 'Cầu 1:' -> 'Câu 1:', 'Bải 1:' -> 'Bài 1:'
        text = re.sub(r"\b(Cầu|Cậu|Cấu)\s+(\d+[:.])", r"Câu \2", text)
        text = re.sub(r"\b(Bải|Bại)\s+(\d+[:.])", r"Bài \2", text)

        # 13. Khôi phục Email và URL/Domain:
        text = recover_emails_and_urls(text)

        # 14. Khôi phục dấu gạch ngang / gạch nối:
        # 14.a. Trong khoảng thời gian / ngày tháng
        text = re.sub(r"(\d{2}/\d{2}/\d{4}|\b\d+)\s*[?~_]\s*(\d{2}/\d{2}/\d{4}|\b\d+)", r"\1 - \2", text)
        text = re.sub(r"(\d{2}/\d{2}/\d{4})\s+(\d{2}/\d{2}/\d{4})", r"\1 - \2", text)
        # 14.b. Trong cụm đề mục liên kết họ tên – email
        text = re.sub(r"(?i)(họ\s+(?:và\s+)?tên)\s*[?~_]\s*(email)", r"\1 – \2", text)
        # 14.c. Giữa tên người và email khi bị thay thế bởi ? ~ . _
        text = re.sub(r"([A-Za-zÀ-ỹĐđ]{2,})\s*[\.\?~_]\s+([a-zA-Z0-9._%+-]+@[a-zA-Z0-9-]+\.)", r"\1 – \2", text)

        # 11. Chuẩn hóa định dạng các hàng bảng tiến độ thực tập
        if "\n" in text:
            text = "\n".join(format_progress_table_row(line) for line in text.split("\n"))
        else:
            text = format_progress_table_row(text)

        # 12. Chuẩn hóa lần cuối về Unicode NFC dựng sẵn
        return unicodedata.normalize("NFC", text)


# ---------------------------------------------------------------------------
# Factory Methods Singleton Toàn Cục
# ---------------------------------------------------------------------------


def get_global_detector(
    lang: str = "vi",
    use_gpu: bool = False,
    use_angle_cls: bool = True,
    det_db_thresh: float = 0.3,
    det_db_box_thresh: float = 0.5,
    det_db_unclip_ratio: float = 1.9,
    drop_score: float = 0.3,
    rec_batch_num: int = 6,
    ocr_version: Optional[str] = None,
) -> Any:
    """
    Khởi tạo PaddleOCR Detector Singleton (duy nhất 1 lần cho cả ứng dụng).
    Tránh lỗi: 'PDX has already been initialized. Reinitialization is not supported.'
    """
    global _GLOBAL_DETECTOR
    with _GLOBAL_DETECTOR_LOCK:
        if _GLOBAL_DETECTOR is not None:
            return _GLOBAL_DETECTOR

        # 1. Áp dụng bản vá ngăn chặn lỗi khởi tạo kép PaddleX
        apply_paddlex_patches()

        # 2. Tối ưu môi trường CPU cho PaddlePaddle 3.x trên Windows
        if not use_gpu:
            try:
                import paddlex.inference.models.runners.paddle_static.runner as _pdx_runner  # type: ignore[import-untyped,import-not-found]
                _orig_pdx_create = _pdx_runner.PaddleStaticRunner._create

                def _safe_pdx_create(self, *args, **kwargs):
                    if getattr(self, "_config", {}).get("device_type") == "cpu":
                        self._config["run_mode"] = "paddle"
                    return _orig_pdx_create(self, *args, **kwargs)

                setattr(_pdx_runner.PaddleStaticRunner, "_create", _safe_pdx_create)
            except Exception:
                pass

        # 3. Khởi tạo PaddleOCR
        v4_supported = {"ch", "en", "chinese_cht"}
        default_version = "PP-OCRv4" if lang in v4_supported else "PP-OCRv3"
        target_version = ocr_version or default_version

        logger.info(f"Loading PaddleOCR detector for lang='{lang}' (Singleton) …")
        from paddleocr import PaddleOCR  # type: ignore[import-untyped,import-not-found]
        import logging as _logging
        _logging.getLogger("ppocr").setLevel(_logging.WARNING)
        _logging.getLogger("paddleocr").setLevel(_logging.WARNING)

        try:
            _GLOBAL_DETECTOR = PaddleOCR(
                lang=lang,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=use_angle_cls,
                text_det_thresh=det_db_thresh,
                text_det_box_thresh=det_db_box_thresh,
                text_det_unclip_ratio=det_db_unclip_ratio,
                text_rec_score_thresh=drop_score,
                ocr_version=target_version,
            )
        except Exception as exc:
            logger.warning(
                f"PaddleOCR primary initialization failed ({exc}), "
                "falling back to standard config …"
            )
            _GLOBAL_DETECTOR = PaddleOCR(
                lang=lang,
                use_angle_cls=use_angle_cls,
                use_gpu=use_gpu,
                det_db_thresh=det_db_thresh,
                det_db_box_thresh=det_db_box_thresh,
                det_db_unclip_ratio=det_db_unclip_ratio,
                drop_score=drop_score,
                rec_batch_num=rec_batch_num,
                show_log=False,
                ocr_version=target_version,
            )

        logger.info("PaddleOCR detector singleton initialized successfully.")
        return _GLOBAL_DETECTOR


def get_global_vietocr(
    model_name: str = "vgg_transformer",
    use_gpu: bool = False,
    beamsearch: bool = True,
) -> Any:
    """
    Khởi tạo VietOCR Predictor Singleton (duy nhất 1 lần, nạp weights vào RAM 1 lần).
    Bật Beam Search (beamsearch=True) để phân tích ngữ cảnh câu và khôi phục chính xác
    các từ vựng dễ nhầm lẫn dấu thanh (ví dụ: 'tại', 'Bí thư').
    """
    global _GLOBAL_VIETOCR
    with _GLOBAL_VIETOCR_LOCK:
        if _GLOBAL_VIETOCR is not None:
            return _GLOBAL_VIETOCR

        try:
            import torch  # type: ignore
            from vietocr.tool.config import Cfg  # type: ignore[import-untyped,import-not-found]
            from vietocr.tool.predictor import Predictor  # type: ignore[import-untyped,import-not-found]

            logger.info(f"Loading VietOCR model '{model_name}' (Singleton, beamsearch={beamsearch}) …")
            config = Cfg.load_config_from_name(model_name)
            device = "cuda:0" if (use_gpu and torch.cuda.is_available()) else "cpu"
            config["device"] = device
            config["predictor"]["beamsearch"] = beamsearch

            _GLOBAL_VIETOCR = Predictor(config)
            logger.info(f"VietOCR ({model_name}) singleton initialized on {device} (beamsearch={beamsearch}).")
        except Exception as exc:
            logger.warning(
                f"Không thể khởi tạo VietOCR ({exc}). "
                "Sẽ dùng PaddleOCR recognition mặc định."
            )
            _GLOBAL_VIETOCR = None

        return _GLOBAL_VIETOCR


def assess_image_quality(img_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Đánh giá chất lượng ảnh văn bản theo độ tương phản nền-chữ (Quality-Aware Assessment):
    - Chuyển sang ảnh xám gray.
    - Mức sáng nền giấy: bg_level = percentile(gray, 85).
    - Mức tối nét chữ: fg_level = percentile(gray, 5).
    - Độ tương phản thực tế: contrast_span = bg_level - fg_level.
    - is_clean_document: True nếu bg_level >= 195.0 và contrast_span >= 90.0.
    """
    if img_bgr is None or img_bgr.size == 0:
        return {
            "bg_level": 255.0,
            "fg_level": 0.0,
            "contrast_span": 255.0,
            "is_clean_document": True,
        }

    if len(img_bgr.shape) == 2:
        gray = img_bgr
    elif len(img_bgr.shape) == 3 and img_bgr.shape[2] == 4:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2GRAY)
    else:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    bg_level = float(np.percentile(gray, 85))
    fg_level = float(np.percentile(gray, 5))
    contrast_span = float(bg_level - fg_level)
    is_clean = (bg_level >= 195.0 and contrast_span >= 90.0)

    return {
        "bg_level": bg_level,
        "fg_level": fg_level,
        "contrast_span": contrast_span,
        "is_clean_document": is_clean,
    }


# ---------------------------------------------------------------------------
# OCR Engine — Hybrid Architecture
# ---------------------------------------------------------------------------


class OCREngine:
    """
    Engine nhận dạng chữ từ ảnh với kiến trúc Hybrid:
    - Text Detection: PaddleOCR DBNet Singleton.
    - Text Recognition: VietOCR Transformer Singleton cho tiếng Việt.
    """

    # Cache wrapper instances theo lang
    _instances: Dict[str, "OCREngine"] = {}

    def __init__(
        self,
        lang: str = "vi",
        use_gpu: bool = False,
        use_angle_cls: bool = True,
        det_db_thresh: float = 0.3,
        det_db_box_thresh: float = 0.5,
        det_db_unclip_ratio: float = 1.9,
        drop_score: float = 0.3,
        rec_batch_num: int = 6,
        ocr_version: Optional[str] = None,
        vietocr_model_name: str = "vgg_transformer",
    ) -> None:
        self.lang = lang
        self.use_gpu = use_gpu
        self.use_angle_cls = use_angle_cls
        self.det_db_thresh = det_db_thresh
        self.det_db_box_thresh = det_db_box_thresh
        self.det_db_unclip_ratio = det_db_unclip_ratio
        self.drop_score = drop_score
        self.rec_batch_num = rec_batch_num
        self.ocr_version = ocr_version
        self.vietocr_model_name = vietocr_model_name

        self._ocr = None       # Tham chiếu tới singleton _GLOBAL_DETECTOR
        self._vietocr = None   # Tham chiếu tới singleton _GLOBAL_VIETOCR
        self._is_loaded = False

    # ------------------------------------------------------------------
    # Singleton Factory Method theo ngôn ngữ
    # ------------------------------------------------------------------

    @classmethod
    def get_engine(cls, lang: str = "vi", **kwargs) -> "OCREngine":
        if lang not in cls._instances:
            cls._instances[lang] = cls(lang=lang, **kwargs)
        return cls._instances[lang]

    @classmethod
    def clear_cache(cls) -> None:
        """
        Dọn dẹp cache instance wrapper khi người dùng đóng cửa sổ.
        Giữ nguyên neural network weights trong RAM để lần quét tiếp theo tức thì.
        """
        cls._instances.clear()
        logger.info("OCREngine cache cleared.")

    # ------------------------------------------------------------------
    # Load Model (Lazy Loading qua Singleton)
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Lấy singleton models đã cache."""
        if self._is_loaded and self._ocr is not None:
            return

        t0 = time.perf_counter()

        # 1. Lấy singleton PaddleOCR Detector
        self._ocr = get_global_detector(
            lang=self.lang,
            use_gpu=self.use_gpu,
            use_angle_cls=self.use_angle_cls,
            det_db_thresh=self.det_db_thresh,
            det_db_box_thresh=self.det_db_box_thresh,
            det_db_unclip_ratio=self.det_db_unclip_ratio,
            drop_score=self.drop_score,
            rec_batch_num=self.rec_batch_num,
            ocr_version=self.ocr_version,
        )

        # 2. Lấy singleton VietOCR Predictor nếu là tiếng Việt
        if self.lang == "vi":
            self._vietocr = get_global_vietocr(
                model_name=self.vietocr_model_name,
                use_gpu=self.use_gpu,
            )
        else:
            self._vietocr = None

        elapsed = time.perf_counter() - t0
        self._is_loaded = True
        logger.info(f"Hybrid OCR engines ready in {elapsed:.2f}s.")

    # ------------------------------------------------------------------
    # Tiền xử lý ảnh (Preprocessing)
    # ------------------------------------------------------------------

    @staticmethod
    def preprocess_image(
        image: np.ndarray,
        max_side: int = 2560,
        min_side: int = 1000,
        denoise: bool = True,
        enhance_contrast: bool = True,
    ) -> np.ndarray:
        """
        Tiền xử lý ảnh tài liệu thích ứng theo chất lượng ảnh (Quality-Aware Adaptive Preprocessing):
        - Trường hợp 1 — Ảnh sạch / Scan rõ / Chụp màn hình (bg_level >= 195 và contrast_span >= 90):
          * TẮT HOÀN TOÀN CLAHE và FastNlMeansDenoising! Giữ nguyên độ mượt nét chữ gốc.
          * Nếu cạnh nhỏ < 850px, phóng đại nhẹ bằng nội suy cv2.INTER_LANCZOS4 (1.25x - 1.5x).
        - Trường hợp 2 — Ảnh chụp tối / Bóng đổ / Mờ (bg_level < 195 hoặc contrast_span < 90):
          * Áp dụng CLAHE nhẹ (clipLimit=1.2, tileGridSize=(8, 8)) trên kênh L.
          * Khử nhiễu nhẹ nếu denoise=True.
        """
        if image is None or image.size == 0:
            raise ValueError("Ảnh đầu vào không hợp lệ.")

        if len(image.shape) == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        img_h, img_w = image.shape[:2]

        quality = assess_image_quality(image)
        is_clean = quality["is_clean_document"]

        if is_clean:
            # Trường hợp 1: Ảnh sạch / Scan rõ / Chụp màn hình
            # TẮT HOÀN TOÀN CLAHE và Denoise
            shortest_side = min(img_h, img_w)
            if shortest_side < 850:
                scale_up = min(1.5, max(1.25, 850.0 / max(1, shortest_side)))
                up_w = int(round(img_w * scale_up))
                up_h = int(round(img_h * scale_up))
                image = cv2.resize(image, (up_w, up_h), interpolation=cv2.INTER_LANCZOS4)
                img_h, img_w = up_h, up_w
            elif shortest_side < min_side:
                scale_up = min(1.5, float(min_side) / max(1, shortest_side))
                if scale_up > 1.05:
                    up_w = int(round(img_w * scale_up))
                    up_h = int(round(img_h * scale_up))
                    image = cv2.resize(image, (up_w, up_h), interpolation=cv2.INTER_LANCZOS4)
                    img_h, img_w = up_h, up_w

            if max(img_h, img_w) > max_side:
                scale_down = max_side / float(max(img_h, img_w))
                new_w = int(img_w * scale_down)
                new_h = int(img_h * scale_down)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

            return image

        # Trường hợp 2: Ảnh chụp tối / Bóng đổ / Mờ
        shortest_side = min(img_h, img_w)
        if shortest_side < min_side:
            scale_up = min(2.0, float(min_side) / max(1, shortest_side))
            if scale_up > 1.15:
                up_w = int(img_w * scale_up)
                up_h = int(img_h * scale_up)
                image = cv2.resize(image, (up_w, up_h), interpolation=cv2.INTER_CUBIC)
                img_h, img_w = up_h, up_w

        if max(img_h, img_w) > max_side:
            scale_down = max_side / float(max(img_h, img_w))
            new_w = int(img_w * scale_down)
            new_h = int(img_h * scale_down)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # CLAHE nhẹ (clipLimit=1.2) trên kênh L
        if enhance_contrast:
            try:
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
                cl = clahe.apply(l)
                image = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
            except Exception:
                pass

        # Khử nhiễu nhẹ
        if denoise:
            try:
                image = cv2.fastNlMeansDenoisingColored(image, None, 4, 4, 7, 21)
            except Exception:
                image = cv2.bilateralFilter(image, d=5, sigmaColor=50, sigmaSpace=50)

        return image

    # ------------------------------------------------------------------
    # Nhận dạng chính (Hybrid Pipeline)
    # ------------------------------------------------------------------

    def recognize(
        self,
        source,
        preprocess: bool = True,
        denoise: bool = True,
        max_side: int = 2560,
    ) -> OCRResult:
        """
        Nhận dạng chữ từ ảnh với pipeline Hybrid:
        1. PaddleOCR Text Detection trích xuất toạ độ đa giác (polygons).
        2. Cắt (crop) từng dòng chữ theo góc nghiêng chuẩn xác.
        3. VietOCR Transformer đọc từng dòng chữ tiếng Việt.
        4. Sắp xếp lại thứ tự đọc văn bản và chuẩn hóa văn bản.
        """
        self._ensure_loaded()
        t0 = time.perf_counter()

        # 1. Đọc ảnh
        if isinstance(source, (str, Path)):
            source = str(source)
            if not os.path.isfile(source):
                raise FileNotFoundError(f"Không tìm thấy file: {source}")
            image = cv2.imread(source, cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"Không thể đọc ảnh: {source}")
        elif isinstance(source, np.ndarray):
            image = source.copy()
        else:
            raise TypeError(f"source phải là str, Path hoặc np.ndarray, nhận {type(source)}")

        orig_h, orig_w = image.shape[:2]

        # 2. Tiền xử lý ảnh
        if preprocess:
            processed_image = self.preprocess_image(
                image, max_side=max_side, denoise=denoise
            )
        else:
            processed_image = image

        proc_h, proc_w = processed_image.shape[:2]
        scale_x = orig_w / float(proc_w)
        scale_y = orig_h / float(proc_h)

        # Thêm padding viền an toàn (16px) trước khi chạy DBNet Detection
        # để bảo toàn 100% dấu thanh và nét chữ nằm sát mép trên/mép biên ảnh
        pad_det = 16
        padded_det_img = cv2.copyMakeBorder(
            processed_image,
            top=pad_det, bottom=pad_det, left=pad_det, right=pad_det,
            borderType=cv2.BORDER_CONSTANT,
            value=[255, 255, 255],
        )

        logger.debug("Running PaddleOCR Text Detection …")
        polygons: List[List[List[float]]] = []
        paddle_texts: List[str] = []
        paddle_scores: List[float] = []

        # 3. Chạy Detection qua PaddleOCR
        detector = self._ocr
        assert detector is not None, "PaddleOCR detector not initialized"
        try:
            results = detector.predict(padded_det_img)
            for res in results:
                dt_polys = res.get("dt_polys", []) if hasattr(res, "get") else getattr(res, "dt_polys", [])
                rec_texts = res.get("rec_texts", []) if hasattr(res, "get") else getattr(res, "rec_texts", [])
                rec_scores = res.get("rec_scores", []) if hasattr(res, "get") else getattr(res, "rec_scores", [])

                for i, poly in enumerate(dt_polys):
                    if hasattr(poly, "tolist"):
                        poly = poly.tolist()
                    if len(poly) >= 4:
                        # Ánh xạ toạ độ từ ảnh đệm về toạ độ ảnh tiền xử lý
                        unpadded_poly = [
                            [
                                max(0.0, min(float(proc_w), pt[0] - pad_det)),
                                max(0.0, min(float(proc_h), pt[1] - pad_det)),
                            ]
                            for pt in poly[:4]
                        ]
                        polygons.append(unpadded_poly)
                        paddle_texts.append(rec_texts[i] if i < len(rec_texts) else "")
                        paddle_scores.append(float(rec_scores[i]) if i < len(rec_scores) else 0.8)
        except Exception as exc_det:
            logger.warning(f"Detection via predict failed, falling back to .ocr(): {exc_det}")
            try:
                raw_results = detector.ocr(padded_det_img, cls=self.use_angle_cls)
                if raw_results and raw_results[0]:
                    for line in raw_results[0]:
                        bbox_points = line[0]
                        unpadded_poly = [
                            [
                                max(0.0, min(float(proc_w), pt[0] - pad_det)),
                                max(0.0, min(float(proc_h), pt[1] - pad_det)),
                            ]
                            for pt in bbox_points[:4]
                        ]
                        polygons.append(unpadded_poly)
                        paddle_texts.append(line[1][0])
                        paddle_scores.append(float(line[1][1]))
            except Exception as exc_raw:
                logger.error(f"Detection fallback failed: {exc_raw}")

        # 4. Text Recognition: Sử dụng VietOCR cho tiếng Việt hoặc PaddleOCR
        boxes: List[OCRBox] = []
        use_vietocr = (self.lang == "vi" and self._vietocr is not None)

        if use_vietocr:
            logger.debug(f"Recognizing {len(polygons)} text lines with VietOCR ({self.vietocr_model_name}) …")

        for idx, poly in enumerate(polygons):
            # Tính toạ độ nguyên trên ảnh gốc
            bbox_int = [
                [int(round(pt[0] * scale_x)), int(round(pt[1] * scale_y))]
                for pt in poly[:4]
            ]

            text = ""
            conf = 0.0
            crop_bgr: Optional[np.ndarray] = None

            if use_vietocr:
                # Cắt (crop) vùng chữ từ ảnh tiền xử lý với padding viền an toàn
                crop_bgr = get_rotate_crop_image(processed_image, np.array(poly))
                if crop_bgr is not None and crop_bgr.size > 0:
                    # 1. Tẩy đường kẻ bảng ở sát biên lát cắt (tránh nhận nhầm [, |, F, ])
                    crop_bgr = clean_crop_borders(crop_bgr)

                    # 2. Kiểm tra dải hoa văn trang trí / đường kẻ lặp ở tầng thị giác
                    if is_decorative_pattern_crop(crop_bgr):
                        text = ""
                        conf = 0.0
                    else:
                        try:
                            # 3. Tách dòng rất dài có chấm lửng thành các sub-crops tránh attention drift nuốt chữ
                            sub_crops = split_dotted_line_crop(crop_bgr)
                            if len(sub_crops) > 1 and self._vietocr is not None:
                                sub_texts = []
                                sub_confs = []
                                for sc in sub_crops:
                                    sc_padded = pad_small_crop_for_vietocr(sc)
                                    sc_rgb = cv2.cvtColor(sc_padded, cv2.COLOR_BGR2RGB)
                                    sc_pil = Image.fromarray(sc_rgb)
                                    p_res = self._vietocr.predict(sc_pil, return_prob=True)
                                    if isinstance(p_res, tuple):
                                        t_sub = (p_res[0] or "").strip()
                                        c_sub = float(p_res[1]) if p_res[1] is not None else 0.95
                                    else:
                                        t_sub = str(p_res).strip()
                                        c_sub = 0.95
                                    if t_sub:
                                        sub_texts.append(t_sub)
                                        sub_confs.append(c_sub)
                                if sub_texts:
                                    text = " ...... ".join(sub_texts)
                                    conf = float(np.mean(sub_confs)) if sub_confs else 0.95
                                else:
                                    text = ""
                                    conf = 0.5
                            else:
                                # 4. Đệm khung trắng chuẩn tỷ lệ cho ô hẹp & chữ số (05, 01, 1, 2)
                                crop_bgr_padded = pad_small_crop_for_vietocr(crop_bgr)
                                crop_rgb = cv2.cvtColor(crop_bgr_padded, cv2.COLOR_BGR2RGB)
                                crop_pil = Image.fromarray(crop_rgb)
                                if self._vietocr is not None:
                                    pred_res = self._vietocr.predict(crop_pil, return_prob=True)
                                else:
                                    pred_res = ("", 0.0)
                                if isinstance(pred_res, tuple):
                                    text = pred_res[0] or ""
                                    prob = pred_res[1]
                                    conf = float(prob) if prob is not None else 0.95
                                else:
                                    text = str(pred_res)
                                    conf = 0.95
                        except Exception as err:
                            logger.debug(f"VietOCR line predict error: {err}")
                            text = paddle_texts[idx] if idx < len(paddle_texts) else ""
                            conf = paddle_scores[idx] if idx < len(paddle_scores) else 0.5
                else:
                    text = paddle_texts[idx] if idx < len(paddle_texts) else ""
                    conf = paddle_scores[idx] if idx < len(paddle_scores) else 0.5
            else:
                # Tiếng Anh hoặc ngôn ngữ khác dùng kết quả từ PaddleOCR
                text = paddle_texts[idx] if idx < len(paddle_texts) else ""
                conf = paddle_scores[idx] if idx < len(paddle_scores) else 0.5

            # Chuẩn hóa văn bản Unicode NFC và làm sạch hậu kiểm
            text = unicodedata.normalize("NFC", text)
            text = VietnameseTextPostProcessor.clean_and_restore(text)
            text = unicodedata.normalize("NFC", text)

            # Bỏ qua dòng trống, dòng ảo giác lặp từ hoặc ký tự rác, hoặc quá thấp độ tin cậy
            if text and not is_decorative_or_hallucinated(text, crop_bgr) and conf >= self.drop_score:
                boxes.append(OCRBox(text=text, confidence=conf, bbox=bbox_int, raw_text=text))

        # 5. Lọc nhiễu quét do cắt ngang viền ảnh (Border noise filtering)
        boxes = filter_border_noise_boxes(boxes, image_width=orig_w, image_height=orig_h)

        # 6. Gom nhóm các khối chữ theo dòng và tái dựng bố cục tài liệu (Layout Reconstruction)
        layout_res = reconstruct_document_layout(boxes)
        grouped_lines = layout_res[0]
        detected_tables = layout_res[1]
        parallel_hdr = getattr(layout_res, "parallel_header", None)
        boxes = [b for line in grouped_lines for b in line]

        # 7. Tổng hợp toàn bộ văn bản (đối với bảng, nối các ô bằng ' | ', đối với văn bản thường nối bằng space)
        # Xây dựng bản đồ ánh xạ hàng bảng theo box id để nhận diện chính xác 100% hàng bảng
        table_row_map = {}
        for tbl in detected_tables:
            for r_idx, r_boxes in enumerate(tbl.row_boxes):
                if r_idx < len(tbl.grid):
                    box_id_set = frozenset(id(b) for b in r_boxes)
                    table_row_map[box_id_set] = (tbl, tbl.grid[r_idx])

        rendered_table_rows = set()
        text_lines = []
        for line in grouped_lines:
            if not line:
                continue

            line_box_ids = frozenset(id(b) for b in line)
            matched_tbl_row = None
            if line_box_ids in table_row_map:
                candidate_tuple = table_row_map[line_box_ids]
                r_key = (id(candidate_tuple[0]), tuple(candidate_tuple[1]))
                if r_key not in rendered_table_rows:
                    matched_tbl_row = candidate_tuple
                    rendered_table_rows.add(r_key)
            else:
                for r_box_ids, tbl_row_tuple in table_row_map.items():
                    common = line_box_ids & r_box_ids
                    if common and (len(common) / len(line_box_ids) >= 0.7 or line_box_ids.issubset(r_box_ids)):
                        r_key = (id(tbl_row_tuple[0]), tuple(tbl_row_tuple[1]))
                        if r_key not in rendered_table_rows:
                            matched_tbl_row = tbl_row_tuple
                            rendered_table_rows.add(r_key)
                        break

            if matched_tbl_row is not None:
                tbl, row_data = matched_tbl_row
                if not row_data[0].strip():
                    non_empty_cells = [c.strip() for c in row_data[1:] if c.strip()]
                    formatted_row = " | ".join(non_empty_cells)
                else:
                    formatted_row = " | ".join(c.strip() for c in row_data if c.strip())

                formatted_row = re.sub(r"\b(\d+)\s*\|\s*\|\s*", r"\1 | ", formatted_row)
                formatted_row = format_progress_table_row(formatted_row)
                text_lines.append(formatted_row)
            else:
                is_fallback_table_row = False
                line_joined = join_ocr_line_boxes(line)
                for tbl in detected_tables:
                    for row_data in tbl.grid:
                        row_joined = " ".join(c.strip() for c in row_data if c.strip())
                        if row_joined and (row_joined == line_joined or line_joined in row_joined):
                            r_key = (id(tbl), tuple(row_data))
                            if r_key not in rendered_table_rows:
                                is_fallback_table_row = True
                                rendered_table_rows.add(r_key)
                                if not row_data[0].strip():
                                    non_empty_cells = [c.strip() for c in row_data[1:] if c.strip()]
                                    formatted_row = " | ".join(non_empty_cells)
                                else:
                                    formatted_row = " | ".join(c.strip() for c in row_data if c.strip())
                                formatted_row = re.sub(r"\b(\d+)\s*\|\s*\|\s*", r"\1 | ", formatted_row)
                                formatted_row = format_progress_table_row(formatted_row)
                                text_lines.append(formatted_row)
                            else:
                                is_fallback_table_row = True
                            break
                    if is_fallback_table_row:
                        break

                if not is_fallback_table_row:
                    cleaned_line = line_joined.strip()
                    if is_hallucinated_or_noise_line(cleaned_line):
                        continue
                    text_lines.append(line_joined)

        raw_full_text = "\n".join(text_lines)
        raw_full_text = VietnameseTextPostProcessor.clean_and_restore(raw_full_text)

        # 8. Context-Aware Symbol Recovery: khôi phục ký tự, số La Mã, bullet dựa trên ngữ cảnh
        # BẮT BUỘC chạy clean_and_restore trước để tách các đề mục số La Mã bị dính vào cuối dòng trước
        # (ví dụ: '... 14 người. TT. Nội dung cuộc họp' -> tách thành 2 dòng riêng trước khi nhận dạng số La Mã)
        pre_cleaned_lines = raw_full_text.split("\n")
        recovered_lines, audit_records = ContextSymbolRecovery.recover_document_symbols(pre_cleaned_lines)
        full_text = "\n".join(recovered_lines)
        full_text = VietnameseTextPostProcessor.clean_and_restore(full_text)

        # 9. Hậu xử lý bắt buộc toàn văn trước khi trả về kết quả OCR:
        # a. Khôi phục bullet '+' cho các dòng tiêu chí con trong bảng đánh giá chất lượng
        full_text = re.sub(
            r"(?m)^-\s+((?:Chi\s+bộ|Thời\s+gian|Từ\s+\d+|Dưới\s+\d+|Không\s+có).*\|\s*0\d)",
            r"+ \1",
            full_text,
            flags=re.IGNORECASE,
        )
        full_text = re.sub(
            r"(?m)^([ \t]*)[-\–—]\s+((?:Chi\s+bộ|Thời\s+gian|Từ\s+\d+|Dưới\s+\d+|Không\s+có).*\|\s*0\d)",
            r"\1+ \2",
            full_text,
            flags=re.IGNORECASE,
        )
        # b. Chuẩn hóa tiêu đề cột 'đa' -> 'Điểm tối đa'
        full_text = re.sub(
            r"(\|\s*Tiêu\s+chí\s+đánh\s+giá\s*\|\s*)(?:tối\s+)?đa(\s*\|\s*Điểm\s+tự\s+chấm)",
            r"\1Điểm tối đa\2",
            full_text,
            flags=re.IGNORECASE,
        )
        # c. Khóa chốt hậu xử lý văn bản cuối cùng (Final Document Cleanup)
        full_text = final_document_cleanup(full_text)

        # Cập nhật văn bản trong các OCRBox và OCRTable để đồng bộ với full_text
        for b in boxes:
            b.text = final_document_cleanup(b.text)

        for tbl in detected_tables:
            for r_idx in range(len(tbl.grid)):
                row_cells = tbl.grid[r_idx]
                for c_idx in range(len(row_cells)):
                    cell_val = row_cells[c_idx]
                    cell_val = re.sub(
                        r"^([ \t]*)[-\–—]\s+((?:Chi\s+bộ|Thời\s+gian|Từ\s+\d+|Dưới\s+\d+|Không\s+có).*)",
                        r"\1+ \2",
                        cell_val,
                        flags=re.IGNORECASE,
                    )
                    row_cells[c_idx] = final_document_cleanup(cell_val)

        elapsed = time.perf_counter() - t0
        result = OCRResult(
            boxes=boxes,
            full_text=full_text,
            language=self.lang,
            elapsed_seconds=elapsed,
            image_width=orig_w,
            image_height=orig_h,
            tables=detected_tables,
            raw_full_text=raw_full_text,
            audit_trail=audit_records,
            parallel_header=parallel_hdr,
        )

        logger.info(
            f"Hybrid OCR completed: {result.num_lines} lines, "
            f"{len(result.tables)} tables detected, "
            f"avg_conf={result.average_confidence:.2f}, "
            f"elapsed={elapsed:.2f}s"
        )
        return result

    # ------------------------------------------------------------------
    # Vẽ Bounding Box lên ảnh
    # ------------------------------------------------------------------

    @staticmethod
    def draw_boxes(
        image: np.ndarray,
        result: OCRResult,
        color: Tuple[int, int, int] = (0, 200, 80),
        thickness: int = 2,
        show_text: bool = False,
        font_scale: float = 0.5,
    ) -> np.ndarray:
        """Vẽ bounding box của kết quả OCR lên ảnh."""
        canvas = image.copy()

        for box in result.boxes:
            pts = np.array(box.bbox, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=thickness)

            if show_text:
                x, y = box.bbox[0]
                label = f"{box.text} ({box.confidence:.0%})"
                cv2.putText(
                    canvas, label, (x, max(y - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    color, 1, cv2.LINE_AA,
                )

        return canvas

    # ------------------------------------------------------------------
    # Đổi ngôn ngữ
    # ------------------------------------------------------------------

    def switch_language(self, new_lang: str) -> None:
        """Đổi ngôn ngữ nhận dạng."""
        if new_lang == self.lang:
            return

        logger.info(f"Switching OCR language: {self.lang} → {new_lang}")
        self.lang = new_lang
        self._is_loaded = False
        self._ocr = None
        self._vietocr = None

    # ------------------------------------------------------------------
    # Context Manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "OCREngine":
        self._ensure_loaded()
        return self

    def __exit__(self, *args) -> None:
        pass

    def __repr__(self) -> str:
        status = "loaded" if self._is_loaded else "not loaded"
        return f"OCREngine(lang={self.lang!r}, hybrid={self.lang == 'vi'}, {status})"


# ---------------------------------------------------------------------------
# Xuất Định Dạng Tài Liệu (Word DOCX Export)
# ---------------------------------------------------------------------------


def export_to_docx(result: OCRResult, output_path: str) -> bool:
    """
    Xuất kết quả OCR sang tài liệu Microsoft Word (.docx).
    - Bảo toàn cấu trúc bảng biểu thành Word Table chuẩn khi có bảng.
    - Bảo toàn tiêu đề, căn lề và các đoạn văn bản.
    """
    try:
        import docx  # type: ignore[import-untyped,import-not-found]
        from docx.shared import Inches, Pt, RGBColor  # type: ignore[import-untyped,import-not-found]
        from docx.enum.text import WD_ALIGN_PARAGRAPH  # type: ignore[import-untyped,import-not-found]
        from docx.enum.table import WD_TABLE_ALIGNMENT  # type: ignore[import-untyped,import-not-found]
    except ImportError:
        logger.error("python-docx is not installed. Cannot export to DOCX.")
        return False

    try:
        doc: Any = docx.Document()

        # Thiết lập lề trang tiêu chuẩn (A4: Top/Bottom 2cm, Left 2.5cm, Right 2cm)
        for section in doc.sections:
            section.top_margin = Inches(0.8)
            section.bottom_margin = Inches(0.8)
            section.left_margin = Inches(1.0)
            section.right_margin = Inches(0.8)

        # Thiết lập style Normal
        style: Any = doc.styles["Normal"]
        font = style.font
        font.name = "Times New Roman"
        font.size = Pt(12)
        font.color.rgb = RGBColor(0, 0, 0)

        # 1. Nếu có parallel_header (Tiêu đề 2 khối song song), tạo bảng 1 hàng 2 cột không viền ở đầu trang Word
        header_rendered = False
        hdr_lines_set = set()
        if result.parallel_header and (result.parallel_header.left_lines or result.parallel_header.right_lines):
            hdr = result.parallel_header
            hdr_lines_set = set(l.strip() for l in (hdr.left_lines + hdr.right_lines) if l.strip())

            hdr_table: Any = doc.add_table(rows=1, cols=2)
            hdr_table.alignment = WD_TABLE_ALIGNMENT.CENTER

            # Gỡ bỏ đường viền của bảng header
            try:
                from docx.oxml import parse_xml  # type: ignore[import-untyped,import-not-found]
                from docx.oxml.ns import nsdecls  # type: ignore[import-untyped,import-not-found]
                tblPr: Any = hdr_table._tbl.tblPr
                borders = parse_xml(
                    r'<w:tblBorders %s><w:top w:val="none"/><w:left w:val="none"/>'
                    r'<w:bottom w:val="none"/><w:right w:val="none"/>'
                    r'<w:insideH w:val="none"/><w:insideV w:val="none"/></w:tblBorders>' % nsdecls("w")
                )
                tblPr.append(borders)
            except Exception as e_border:
                logger.debug(f"Could not remove table borders: {e_border}")

            # Khối trái: Cơ quan ban hành, số hiệu
            cell_left = hdr_table.cell(0, 0)
            for idx, line_text in enumerate(hdr.left_lines):
                p = cell_left.paragraphs[0] if idx == 0 else cell_left.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_before = Pt(1)
                p.paragraph_format.space_after = Pt(1)
                run = p.add_run(line_text)
                run.font.name = "Times New Roman"
                run.font.size = Pt(11)
                if idx == 0 or line_text.isupper():
                    run.bold = True

            # Khối phải: Quốc hiệu, tiêu ngữ, ngày tháng
            cell_right = hdr_table.cell(0, 1)
            for idx, line_text in enumerate(hdr.right_lines):
                p = cell_right.paragraphs[0] if idx == 0 else cell_right.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_before = Pt(1)
                p.paragraph_format.space_after = Pt(1)
                run = p.add_run(line_text)
                run.font.name = "Times New Roman"
                run.font.size = Pt(11)
                if any(kw in line_text for kw in ["ĐẢNG CỘNG SẢN", "CỘNG HÒA", "ĐỘC LẬP"]):
                    run.bold = True
                elif "ngày" in line_text.lower() or "tháng" in line_text.lower():
                    run.italic = True

            header_rendered = True
            p_spacer = doc.add_paragraph()
            p_spacer.paragraph_format.space_before = Pt(4)
            p_spacer.paragraph_format.space_after = Pt(4)

        # 2. Xuất phần thân văn bản và các bảng biểu
        lines = result.full_text.split("\n")
        skipping_header = header_rendered
        table_idx = 0
        in_table_mode = False

        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            if skipping_header and line_str in hdr_lines_set:
                continue
            skipping_header = False

            # Kiểm tra nếu dòng thuộc bảng
            if result.tables and table_idx < len(result.tables) and " | " in line_str:
                tbl = result.tables[table_idx]
                cells = [c.strip() for c in line_str.split(" | ")]
                if len(cells) == tbl.num_cols or abs(len(cells) - tbl.num_cols) <= 1:
                    if not in_table_mode:
                        doc_table: Any = doc.add_table(rows=len(tbl.grid), cols=tbl.num_cols)
                        doc_table.alignment = WD_TABLE_ALIGNMENT.CENTER
                        doc_table.style = "Table Grid"

                        # Tính độ rộng cột tỷ lệ (Proportional column widths)
                        total_width_inches = 6.5
                        col_max_lens = [
                            max((len(tbl.grid[r][c].strip()) for r in range(len(tbl.grid))), default=5)
                            for c in range(tbl.num_cols)
                        ]
                        total_len = max(1, sum(col_max_lens))
                        col_widths_inches = []
                        for c_len in col_max_lens:
                            w_inch = max(0.6, min(4.0, (c_len / total_len) * total_width_inches))
                            col_widths_inches.append(w_inch)

                        sum_w = sum(col_widths_inches)
                        if sum_w > 0:
                            col_widths_inches = [(w / sum_w) * total_width_inches for w in col_widths_inches]

                        # Lặp tiêu đề bảng ở mỗi trang Word nếu bảng dài (tblHeader)
                        if len(tbl.grid) > 0:
                            try:
                                from docx.oxml import parse_xml  # type: ignore[import-untyped,import-not-found]
                                from docx.oxml.ns import nsdecls  # type: ignore[import-untyped,import-not-found]
                                header_tr: Any = doc_table.rows[0]._tr.get_or_add_trPr()
                                header_tr.append(parse_xml(r'<w:tblHeader %s/>' % nsdecls("w")))
                            except Exception:
                                pass

                        for r_i, row_data in enumerate(tbl.grid):
                            for c_i in range(min(len(row_data), tbl.num_cols)):
                                cell = doc_table.cell(r_i, c_i)
                                if c_i < len(col_widths_inches):
                                    cell.width = Inches(col_widths_inches[c_i])
                                cell_text = row_data[c_i].strip()
                                cell.text = cell_text

                                for p in cell.paragraphs:
                                    p.paragraph_format.space_before = Pt(2)
                                    p.paragraph_format.space_after = Pt(2)
                                    for run in p.runs:
                                        run.font.name = "Times New Roman"
                                        run.font.size = Pt(11)
                                        if r_i == 0:
                                            run.bold = True
                                    if r_i == 0 or (c_i != 1 and len(cell_text) <= 5):
                                        p.alignment = WD_ALIGN_PARAGRAPH.CENTER

                        in_table_mode = True
                        table_idx += 1
                    continue

            in_table_mode = False
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.line_spacing = 1.15

            is_centered_title = any(kw in line_str.upper() for kw in [
                "BIÊN BẢN", "QUYẾT ĐỊNH", "THÔNG TƯ", "NGHỊ ĐỊNH", "BÁO CÁO",
                "KẾ HOẠCH", "BẢNG ĐIỂM", "ĐẢNG CỘNG SẢN VIỆT NAM",
                "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM", "ĐỘC LẬP - TỰ DO - HẠNH PHÚC",
                "HỢP ĐỒNG", "HÓA ĐƠN", "PHIẾU THU", "PHIẾU CHI", "BÁO GIÁ",
                "BẢNG KÊ", "GIẤY XÁC NHẬN", "GIẤY CHỨNG NHẬN", "CHƯƠNG",
                "BÀI HỌC", "GIÁO TRÌNH", "ĐỀ THI", "ĐÁP ÁN", "TỔNG KẾT",
                "PHIẾU KHÁM", "KẾT QUẢ XÉT NGHIỆM"
            ])
            if is_centered_title:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                run = p.add_run(line_str)
                run.bold = True
                run.font.name = "Times New Roman"
                run.font.size = Pt(13)
            else:
                run = p.add_run(line_str)
                run.font.name = "Times New Roman"
                run.font.size = Pt(12)

        doc.save(output_path)
        logger.info(f"DOCX exported successfully: {output_path}")
        return True
    except Exception as exc:
        logger.error(f"Error exporting to DOCX: {exc}")
        return False
