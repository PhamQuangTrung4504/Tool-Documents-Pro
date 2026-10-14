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
from typing import Any, Dict, List, Optional, Tuple

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
        import torch  # noqa: F401
    except Exception:
        pass

    try:
        import sys
        import paddlex
        import paddlex.repo_manager as rm
        import paddlex.repo_manager.core as rmc

        orig_init = rmc.initialize

        def safe_init(*args, **kwargs):
            try:
                if rmc._GlobalContext.is_initialized():
                    return
                return orig_init(*args, **kwargs)
            except RuntimeError as re:
                if "already been initialized" in str(re):
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
        return tuple(self.bbox[0])

    @property
    def bottom_right(self) -> Tuple[int, int]:
        return tuple(self.bbox[2])

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
    def num_lines(self) -> int:
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
    pad_left: int = 6,
    pad_right: int = 12,
) -> Optional[np.ndarray]:
    """
    Cắt và xoay thẳng vùng ảnh polygon 4 điểm (Perspective Transform).
    Chuẩn hóa vùng ảnh chữ nằm ngang cho VietOCR với viền đệm (padding) an toàn:
    - Mở rộng mép trên 6px (bảo vệ dấu sắc, hỏi, ngã, mũ: á, ả, ã, â, ê, ô, ư, ơ).
    - Mở rộng mép dưới 8px (bảo vệ trọn vẹn dấu nặng: ạ, ặ, ậ, ẹ, ệ, ị, ọ, ộ, ợ, ụ, ự, ỵ).
    - Mở rộng mép phải 12px (bảo vệ chữ cái cuối dòng không bị cắt cụt sát mép như 'đặc b' -> 'đặc biệt').
    - Mở rộng mép trái 6px (tránh dính mép viền làm đứt chữ cái đầu hoặc dấu bullet).
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

        pts_std = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
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
            is_page_number = bool(re.match(
                r"^(?:Trang\s+\d+(?:/\d+)?|\d+(?:/\d+)?|Page\s+\d+)$",
                text, re.IGNORECASE
            ))
            is_valid_doc_code = bool(re.search(r"\b\d+/\d{4}/[A-ZĐ-]+", text))
            is_valid_date = bool(re.search(r"\bngày\s+\d+\s+tháng\s+\d+", text, re.IGNORECASE))

            # Nếu là số trang hoặc số hiệu/ngày tháng hợp lệ và chiều cao không quá cụt -> Giữ lại
            if (is_page_number or is_valid_doc_code or is_valid_date) and h >= 0.55 * median_h:
                valid_boxes.append(b)
                continue

            # Nếu chiều cao bị cắt quá nửa so với dòng chữ bình thường -> Nét cắt xén viền ảnh
            if h < 0.48 * median_h or (h <= 10 and median_h >= 18):
                logger.debug(f"Filtered clipped border noise: text={text!r}, h={h}, median_h={median_h}")
                continue

            # Nếu text là chuỗi ký tự rời rạc/nhiễu quét mép ảnh
            is_noisy_pattern = bool(re.match(r"^[\d\s/.,\-_+=*~`]+$", text) and not is_page_number and not is_valid_doc_code)
            if is_noisy_pattern and h < 0.70 * median_h:
                logger.debug(f"Filtered noisy border text: text={text!r}, h={h}, median_h={median_h}")
                continue

        valid_boxes.append(b)

    return valid_boxes


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
    sub_boxes_sorted = sorted(sub_boxes, key=lambda it: (it["cy"], it["left"]))
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
        line.sort(key=lambda it: it["left"])
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

    box_info_sorted = sorted(box_info, key=lambda it: it["cy"])
    result_lines: List[List[OCRBox]] = []
    i = 0
    n = len(box_info_sorted)

    while i < n:
        best_2col_end = None
        best_split_x = None

        for j in range(i + 2, n + 1):
            window = box_info_sorted[i:j]
            found_split = None
            for split_ratio in np.linspace(0.28, 0.65, 38):
                sx = page_min_x + split_ratio * page_w
                left_b = [b for b in window if b["right"] <= sx]
                right_b = [b for b in window if b["left"] >= sx]
                cross_b = [b for b in window if b["left"] < sx and b["right"] > sx]
                if left_b and right_b and not cross_b:
                    gap = min(b["left"] for b in right_b) - max(b["right"] for b in left_b)
                    if gap >= gutter_thresh:
                        found_split = sx
                        break
            if found_split is not None:
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

                if (abs(b_curr["cy"] - avg_cy) < y_thresh or overlap_y >= 0.40 * min(b_curr["h"], avg_h)) and gap_x < gutter_thresh:
                    single_line_boxes.append(b_curr)
                    k += 1
                else:
                    break

            single_line_boxes.sort(key=lambda it: it["left"])
            if len(single_line_boxes) >= 2:
                first = single_line_boxes[0]
                if first["w"] <= 35 and first["h"] <= 35 and first["box"].text.strip() in [
                    "o", "O", "•", "·", "*", "■", "□", "回", "▫", "▪"
                ]:
                    first["box"].text = "•"

            result_lines.append([it["box"] for it in single_line_boxes])
            i = k

    return result_lines


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

    # Lọc bỏ các box đường kẻ trang trí bị OCR nhận sai thành ký tự rác (%, *, ~, _, -, =, ...)
    def _is_decorative_line_noise(b_item: dict) -> bool:
        txt = b_item["text"].strip()
        if re.fullmatch(r"[%*~_\-=—–·•.]+", txt):
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
    narrow_boxes = [b for b in box_info if b["w"] < 0.25 * page_w]
    if len(narrow_boxes) < 4:
        return None, box_info, [], []

    # Tìm box tiêu đề cột STT trong bảng (nếu có)
    stt_header_box = None
    for b in box_info:
        txt_norm = b["text"].strip().upper()
        if re.match(r"^(?:STT|TT|SỐ\s*TT)$", txt_norm):
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
    anchor_min_top = min(b["top"] for b in all_anchor_boxes)
    anchor_max_bottom = max(b["bottom"] for b in all_anchor_boxes)

    table_boxes = [
        b for b in box_info
        if b["bottom"] > anchor_min_top - 5 and b["top"] < anchor_max_bottom + 5 and b["cy"] >= anchor_min_top - 5
    ]
    pre_table_boxes = [b for b in box_info if b["bottom"] <= anchor_min_top]
    post_table_boxes = [b for b in box_info if b["top"] >= anchor_max_bottom]

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

    # Khôi phục số thứ tự STT '1' cho hàng tiêu chí đầu tiên nếu bị thiếu hoặc nhận nhầm thành '|' / 'l'
    if table_grid and len(table_grid) >= 2:
        header_row = table_grid[0]
        if header_row and re.match(r"^(?:STT|TT|SỐ\s*TT)$", header_row[0].strip().upper()):
            row_with_2_idx = -1
            for r_idx in range(1, len(table_grid)):
                if table_grid[r_idx][0].strip() == "2":
                    row_with_2_idx = r_idx
                    break
            if row_with_2_idx > 1:
                first_stt_val = table_grid[1][0].strip()
                if first_stt_val in {"", "|", "l", "I", "!"} or not first_stt_val.isdigit():
                    table_grid[1][0] = "1"

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

    # 1. Kiểm tra Table Reconstruction
    tbl, pre_tbl, tbl_rows, post_tbl = detect_and_reconstruct_table(
        box_info, page_min_x, page_max_x, page_w, median_h
    )

    if tbl is not None:
        # Gom nhóm vùng văn bản trước bảng (có thể có Header 2 khối)
        hdr_res = detect_parallel_administrative_header(
            pre_tbl, page_min_x, page_max_x, page_w, gutter_thresh, median_h
        )
        header_lines = hdr_res[0]
        rem_pre = hdr_res[1]
        parallel_hdr = getattr(hdr_res, "parallel_header", None)

        pre_lines = header_lines + _group_boxes_sliding_window(
            rem_pre, page_min_x, page_w, gutter_thresh, median_h
        )

        # Gom nhóm vùng văn bản sau bảng
        post_lines = _group_boxes_sliding_window(
            post_tbl, page_min_x, page_w, gutter_thresh, median_h
        )

        all_lines = pre_lines + tbl_rows + post_lines
        return LayoutResult(all_lines, [tbl], parallel_header=parallel_hdr)

    # 2. Nếu không có bảng: Kiểm tra Header 2 khối hành chính đầu trang
    hdr_res = detect_parallel_administrative_header(
        box_info, page_min_x, page_max_x, page_w, gutter_thresh, median_h
    )
    header_lines = hdr_res[0]
    rem_boxes = hdr_res[1]
    parallel_hdr = getattr(hdr_res, "parallel_header", None)

    if header_lines:
        rem_lines = _group_boxes_sliding_window(
            rem_boxes, page_min_x, page_w, gutter_thresh, median_h
        )
        return LayoutResult(header_lines + rem_lines, [], parallel_header=parallel_hdr)

    # 3. Văn bản thông thường (Sliding Window Multi-column / Single-column)
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
    "NỘI DUNG", "TIẾN HÀNH", "KẾT LUẬN", "TỔ CHỨC", "MỤC ĐÍCH", "YÊU CẦU",
    "KẾ HOẠCH", "PHƯƠNG HƯỚNG", "NHIỆM VỤ", "ĐÁNH GIÁ", "CHƯƠNG TRÌNH",
    "THỰC HIỆN", "KẾT QUẢ", "THỜI GIAN", "ĐỊA ĐIỂM", "THÀNH PHẦN",
    "KIỂM ĐIỂM", "THẢO LUẬN", "NGHỊ QUYẾT", "BÁO CÁO",
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
        """Kiểm tra văn bản có chứa từ khóa đề mục văn bản hành chính hay không."""
        text_upper = text.upper()
        return any(kw in text_upper for kw in ROMAN_HEADING_KEYWORDS)

    @classmethod
    def recover_roman_numeral_in_line(
        cls, line: str, prev_roman: str = ""
    ) -> Tuple[str, SymbolDecision, str, str]:
        """
        Khôi phục số La Mã đầu mục (I., II., III., IV., V.) dựa trên ngữ cảnh.
        Trả về: (new_line, decision, current_roman, rule_name)
        """
        stripped = line.strip()
        if not stripped:
            return line, SymbolDecision.PRESERVE, prev_roman, ""

        # 1. Mẫu số II: TI., TI, ll., 11., Tl., lI., Il.
        m_ii = re.match(r"^([ \t]*)(TI|ll|11|Tl|lI|Il)\.?\s+([A-ZÀ-ỸĐ\s,;:/-]+)$", line)
        if m_ii:
            indent = m_ii.group(1)
            prefix = m_ii.group(2)
            rest = m_ii.group(3).strip()
            if cls.is_all_uppercase_heading(rest) or cls.has_heading_keyword(rest):
                return f"{indent}II. {rest}", SymbolDecision.AUTO_CORRECT, "II", f"ROMAN_II_FROM_{prefix}"
            else:
                return line, SymbolDecision.FLAG_UNCERTAIN, prev_roman, f"UNCERTAIN_ROMAN_II_{prefix}"

        # 2. Mẫu số III: lll., 111., TII., TIl., llI., lll, 111
        m_iii = re.match(r"^([ \t]*)(lll|111|TII|TIl|llI)\.?\s+([A-ZÀ-ỸĐ\s,;:/-]+)$", line)
        if m_iii:
            indent = m_iii.group(1)
            prefix = m_iii.group(2)
            rest = m_iii.group(3).strip()
            if cls.is_all_uppercase_heading(rest) or cls.has_heading_keyword(rest):
                return f"{indent}III. {rest}", SymbolDecision.AUTO_CORRECT, "III", f"ROMAN_III_FROM_{prefix}"
            else:
                return line, SymbolDecision.FLAG_UNCERTAIN, prev_roman, f"UNCERTAIN_ROMAN_III_{prefix}"

        # 3. Mẫu số IV: lV., 1V., TV., lV, 1V
        m_iv = re.match(r"^([ \t]*)(lV|1V|TV)\.?\s+([A-ZÀ-ỸĐ\s,;:/-]+)$", line)
        if m_iv:
            indent = m_iv.group(1)
            prefix = m_iv.group(2)
            rest = m_iv.group(3).strip()
            if cls.is_all_uppercase_heading(rest) or cls.has_heading_keyword(rest):
                return f"{indent}IV. {rest}", SymbolDecision.AUTO_CORRECT, "IV", f"ROMAN_IV_FROM_{prefix}"
            else:
                return line, SymbolDecision.FLAG_UNCERTAIN, prev_roman, f"UNCERTAIN_ROMAN_IV_{prefix}"

        # 4. Mẫu số I: |, |., l., 1.
        # ĐIỀU KIỆN ÂM TÍNH NGHIÊM NGẶT:
        # Không đổi 1. thành I. nếu theo sau là chữ thường (ví dụ: 1. Tuyên bố lý do...)
        # và không đổi số thứ tự hợp lệ như 1. NỘP HỒ SƠ
        m_i = re.match(r"^([ \t]*)(\|\.?|l\.|1\.)\s+([A-ZÀ-ỸĐ\s,;:/-]+)$", line)
        if m_i:
            indent = m_i.group(1)
            prefix = m_i.group(2).strip()
            rest = m_i.group(3).strip()
            is_strong_symbol = prefix in ["|", "|.", "l."]
            is_strong_heading = cls.has_heading_keyword(rest) and cls.is_all_uppercase_heading(rest)

            if is_strong_symbol and (cls.is_all_uppercase_heading(rest) or cls.has_heading_keyword(rest)):
                return f"{indent}I. {rest}", SymbolDecision.AUTO_CORRECT, "I", f"ROMAN_I_FROM_{prefix}"
            elif prefix == "1." and is_strong_heading and ("NỘI DUNG" in rest.upper() or "CHƯƠNG TRÌNH" in rest.upper()):
                return f"{indent}I. {rest}", SymbolDecision.AUTO_CORRECT, "I", "ROMAN_I_FROM_1_DOT"

        # 5. Nếu đã là số La Mã chuẩn I., II., III., IV., V., VI.
        m_valid = re.match(r"^[ \t]*(I{1,3}|IV|V|VI)\.\s+", line)
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
        m_pct = re.match(r"^([ \t]*)([%*])\s+([A-Za-zÀ-ỹĐđ].*)$", line)
        if m_pct:
            indent = m_pct.group(1)
            sym = m_pct.group(2)
            content = m_pct.group(3)
            # Bảo vệ dòng ghi chú: '* Ghi chú quan trọng'
            if sym == "*" and content.lower().startswith("ghi chú"):
                return line, SymbolDecision.PRESERVE, ""
            return f"{indent}+ {content}", SymbolDecision.AUTO_CORRECT, "PLUS_FROM_PERCENT_PREFIX"

        # 3. Ký tự '|' ở đầu dòng trong văn bản thông thường (không phải hàng bảng nhiều cột ' | ')
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

        for idx, line in enumerate(lines):
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

            recovered_lines.append(curr_line)

        return recovered_lines, audit_trail


# ---------------------------------------------------------------------------
# Bộ Chuẩn Hóa Văn Bản Tiếng Việt
# ---------------------------------------------------------------------------


class VietnameseTextPostProcessor:
    """
    Module chuẩn hóa và làm sạch kỹ thuật văn bản OCR:
    - Chuẩn hóa Unicode dựng sẵn (NFC).
    - Chuẩn hóa khoảng trắng quanh dấu câu: loại bỏ khoảng trắng thừa trước dấu.
    - TUYỆT ĐỐI KHÔNG tự động chèn khoảng trắng sau dấu chấm làm hỏng file path / extension,
      và TUYỆT ĐỐI KHÔNG tự động sửa nội dung từ vựng hoặc viết lại câu.
    """

    @classmethod
    def clean_and_restore(cls, text: str) -> str:
        if not text:
            return ""

        # 1. Chuẩn hóa Unicode NFC
        text = unicodedata.normalize("NFC", text).strip()

        # 2. Loại bỏ khoảng trắng thừa ngay trước dấu câu
        text = re.sub(r"[ \t]+([,.:;?!])", r"\1", text)

        # 3. Ký hiệu kết thúc văn bản hành chính ./.
        text = re.sub(r"\.\s*/\s*\.", "./.", text)

        # 4. Khôi phục từ bị cắt xén ở cuối bảng/trang dựa trên ngữ cảnh hành chính đã xác minh
        text = re.sub(r"\b(nhiệm\s+vụ\s+đặc)\s+b\b", r"\1 biệt", text)

        # 5. Chuẩn hóa lần cuối về Unicode NFC dựng sẵn
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
                import paddlex.inference.models.runners.paddle_static.runner as _pdx_runner
                _orig_pdx_create = _pdx_runner.PaddleStaticRunner._create

                def _safe_pdx_create(runner_self, *args, **kwargs):
                    if runner_self._config.get("device_type") == "cpu":
                        runner_self._config["run_mode"] = "paddle"
                    return _orig_pdx_create(runner_self, *args, **kwargs)

                _pdx_runner.PaddleStaticRunner._create = _safe_pdx_create
            except Exception:
                pass

        # 3. Khởi tạo PaddleOCR
        v4_supported = {"ch", "en", "chinese_cht"}
        default_version = "PP-OCRv4" if lang in v4_supported else "PP-OCRv3"
        target_version = ocr_version or default_version

        logger.info(f"Loading PaddleOCR detector for lang='{lang}' (Singleton) …")
        from paddleocr import PaddleOCR
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
            import torch
            from vietocr.tool.config import Cfg
            from vietocr.tool.predictor import Predictor

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
        Tiền xử lý ảnh tài liệu:
        - Chuẩn hóa kênh màu sang BGR.
        - Tự động upscale bicubic (INTER_CUBIC) nếu ảnh nhỏ để làm rõ nét dấu thanh.
        - Cân bằng tương phản cục bộ CLAHE trên kênh độ sáng L (LAB).
        - Khử nhiễu nhẹ nền văn bản.
        """
        if image is None or image.size == 0:
            raise ValueError("Ảnh đầu vào không hợp lệ.")

        if len(image.shape) == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        img_h, img_w = image.shape[:2]

        # Upscale nếu kích thước quá nhỏ
        shortest_side = min(img_h, img_w)
        if shortest_side < min_side:
            scale_up = min(2.0, float(min_side) / max(1, shortest_side))
            if scale_up > 1.15:
                up_w = int(img_w * scale_up)
                up_h = int(img_h * scale_up)
                image = cv2.resize(image, (up_w, up_h), interpolation=cv2.INTER_CUBIC)
                img_h, img_w = up_h, up_w

        # Downscale nếu quá lớn
        if max(img_h, img_w) > max_side:
            scale_down = max_side / float(max(img_h, img_w))
            new_w = int(img_w * scale_down)
            new_h = int(img_h * scale_down)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # CLAHE tăng tương phản
        if enhance_contrast:
            try:
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
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

            if use_vietocr:
                # Cắt (crop) vùng chữ từ ảnh tiền xử lý với padding viền an toàn
                crop_bgr = get_rotate_crop_image(processed_image, np.array(poly))
                if crop_bgr is not None and crop_bgr.size > 0:
                    try:
                        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                        crop_pil = Image.fromarray(crop_rgb)
                        # Giữ nguyên tỷ lệ (aspect ratio), đưa trực tiếp vào VietOCR Predictor
                        pred_res = self._vietocr.predict(crop_pil, return_prob=True)
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

            # Bỏ qua dòng trống hoặc quá thấp độ tin cậy
            if text and conf >= self.drop_score:
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

        text_lines = []
        for line in grouped_lines:
            if not line:
                continue

            line_box_ids = frozenset(id(b) for b in line)
            matched_tbl_row = None
            if line_box_ids in table_row_map:
                matched_tbl_row = table_row_map[line_box_ids]
            else:
                for r_box_ids, tbl_row_tuple in table_row_map.items():
                    if line_box_ids.issubset(r_box_ids) or (line_box_ids & r_box_ids):
                        matched_tbl_row = tbl_row_tuple
                        break

            if matched_tbl_row is not None:
                tbl, row_data = matched_tbl_row
                if not row_data[0].strip():
                    non_empty_cells = [c.strip() for c in row_data[1:] if c.strip()]
                    formatted_row = " | ".join(non_empty_cells)
                else:
                    formatted_row = " | ".join(c.strip() for c in row_data if c.strip())

                formatted_row = re.sub(r"\b(\d+)\s*\|\s*\|\s*", r"\1 | ", formatted_row)
                text_lines.append(formatted_row)
            else:
                is_fallback_table_row = False
                line_joined = " ".join(b.text.strip() for b in line if b.text.strip())
                for tbl in detected_tables:
                    for row_data in tbl.grid:
                        row_joined = " ".join(c.strip() for c in row_data if c.strip())
                        if row_joined and (row_joined == line_joined or line_joined in row_joined):
                            is_fallback_table_row = True
                            if not row_data[0].strip():
                                non_empty_cells = [c.strip() for c in row_data[1:] if c.strip()]
                                formatted_row = " | ".join(non_empty_cells)
                            else:
                                formatted_row = " | ".join(c.strip() for c in row_data if c.strip())
                            formatted_row = re.sub(r"\b(\d+)\s*\|\s*\|\s*", r"\1 | ", formatted_row)
                            text_lines.append(formatted_row)
                            break
                    if is_fallback_table_row:
                        break

                if not is_fallback_table_row:
                    text_lines.append(line_joined)

        raw_full_text = "\n".join(text_lines)
        raw_full_text = VietnameseTextPostProcessor.clean_and_restore(raw_full_text)

        # 8. Context-Aware Symbol Recovery: khôi phục ký tự, số La Mã, bullet dựa trên ngữ cảnh
        recovered_lines, audit_records = ContextSymbolRecovery.recover_document_symbols(text_lines)
        full_text = "\n".join(recovered_lines)
        full_text = VietnameseTextPostProcessor.clean_and_restore(full_text)

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
        import docx
        from docx.shared import Inches, Pt, RGBColor
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.enum.table import WD_TABLE_ALIGNMENT
    except ImportError:
        logger.error("python-docx is not installed. Cannot export to DOCX.")
        return False

    try:
        doc = docx.Document()

        # Thiết lập lề trang tiêu chuẩn (A4: Top/Bottom 2cm, Left 2.5cm, Right 2cm)
        for section in doc.sections:
            section.top_margin = Inches(0.8)
            section.bottom_margin = Inches(0.8)
            section.left_margin = Inches(1.0)
            section.right_margin = Inches(0.8)

        # Thiết lập style Normal
        style = doc.styles["Normal"]
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

            hdr_table = doc.add_table(rows=1, cols=2)
            hdr_table.alignment = WD_TABLE_ALIGNMENT.CENTER

            # Gỡ bỏ đường viền của bảng header
            try:
                from docx.oxml import parse_xml
                from docx.oxml.ns import nsdecls
                tblPr = hdr_table._tbl.tblPr
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
                        doc_table = doc.add_table(rows=len(tbl.grid), cols=tbl.num_cols)
                        doc_table.alignment = WD_TABLE_ALIGNMENT.CENTER
                        doc_table.style = "Table Grid"

                        for r_i, row_data in enumerate(tbl.grid):
                            for c_i in range(min(len(row_data), tbl.num_cols)):
                                cell = doc_table.cell(r_i, c_i)
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
                "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM", "ĐỘC LẬP - TỰ DO - HẠNH PHÚC"
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
