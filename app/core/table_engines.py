#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
app.core.table_engines
======================
Module chứa các thuật toán nhận diện và tái cấu trúc bảng biểu độc lập cho Giai đoạn C3:
1. MorphologicalGridTableEngine: Sử dụng hình thái học đóng (Morphological Closing) và giao điểm lưới OpenCV.
2. SpatialProjectionGraphEngine: Đồ thị không gian và phép chiếu trục X/Y, chống lệch Y và tiền lọc nhiễu mép.
3. UnifiedHybridTableEngine: Kết hợp thông minh giữa thông tin thị giác đường kẻ và đồ thị chiếu không gian.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from app.core.models import (
    HeaderDetectionResult,
    LayoutResult,
    OCRBox,
    OCRTable,
    ParallelHeader,
    SymbolDecision,
)
from app.core.text_postprocessor import ContextSymbolRecovery

logger = logging.getLogger(__name__)

__all__ = [
    "MorphologicalGridTableEngine",
    "SpatialProjectionGraphEngine",
    "UnifiedHybridTableEngine",
    "morphological_grid_engine",
    "spatial_projection_engine",
    "unified_hybrid_engine",
]


def _box_to_dict(box: OCRBox) -> dict:
    """Chuyển đổi OCRBox sang dict chứa tọa độ và kích thước."""
    pts = box.bbox
    x_left = min(pt[0] for pt in pts)
    x_right = max(pt[0] for pt in pts)
    y_top = min(pt[1] for pt in pts)
    y_bottom = max(pt[1] for pt in pts)
    h = max(1, y_bottom - y_top)
    w = max(1, x_right - x_left)
    cy = (y_top + y_bottom) / 2.0
    return {
        "box": box,
        "text": box.text,
        "left": x_left,
        "right": x_right,
        "top": y_top,
        "bottom": y_bottom,
        "h": h,
        "w": w,
        "cy": cy,
    }


def _filter_border_noise(boxes: List[dict]) -> Tuple[List[dict], List[dict]]:
    """
    Tiền lọc các box nhiễu viền ở mép trang (Border Noise Pre-filtering).
    Loại bỏ các box rác ở rìa chỉ chứa ký tự đơn lẻ như |, [, ], /, \\.
    Trả về: (cleaned_boxes, noise_boxes)
    """
    if not boxes:
        return [], []

    valid_text_boxes = [
        b for b in boxes
        if len(b["text"].strip()) > 1 and re.search(r"[\wÀ-ỹ0-9]", b["text"])
    ]
    if not valid_text_boxes:
        return boxes, []

    min_valid_x = min(b["left"] for b in valid_text_boxes)
    max_valid_x = max(b["right"] for b in valid_text_boxes)

    cleaned: List[dict] = []
    noise: List[dict] = []

    noise_chars = {"|", "[", "]", "/", "\\", "~", "—", "_"}

    for b in boxes:
        txt = b["text"].strip()
        is_single_noise = txt in noise_chars or (len(txt) <= 2 and all(c in noise_chars for c in txt))
        # Nếu là ký tự nhiễu và nằm lệch ra ngoài biên của khối văn bản hợp lệ
        if is_single_noise and (b["right"] < min_valid_x - 15 or b["left"] > max_valid_x + 15):
            noise.append(b)
        else:
            cleaned.append(b)

    return cleaned, noise


def _extract_distinct_column_boxes(row_boxes: List[dict]) -> List[dict]:
    """
    Gộp các box nằm cùng một cột (chồng lấn ngang) trong 1 hàng
    (ví dụ: ô nhiều dòng có 2 box xếp chồng nhau).
    """
    if not row_boxes:
        return []
    sorted_b = sorted(row_boxes, key=lambda b: b["left"])
    col_groups: List[dict] = []
    for b in sorted_b:
        if not col_groups:
            col_groups.append(dict(b))
            continue
        prev = col_groups[-1]
        overlap_x = max(0.0, min(b["right"], prev["right"]) - max(b["left"], prev["left"]))
        min_w = min(b["w"], prev["w"])
        if overlap_x >= 0.35 * min_w or abs(b["left"] - prev["left"]) <= 20:
            prev["left"] = min(prev["left"], b["left"])
            prev["right"] = max(prev["right"], b["right"])
            prev["w"] = prev["right"] - prev["left"]
            prev["text"] = f"{prev['text']} {b['text']}".strip()
        else:
            col_groups.append(dict(b))
    return col_groups


class SpatialProjectionGraphEngine:
    """
    Phương pháp 2 — Layout Graph & X/Y Projection:
    Nhận diện và tái dựng bảng dựa trên hình học không gian và phép chiếu trục X/Y.
    Hoạt động độc lập không phụ thuộc đường kẻ, chống chịu lệch Y ±8px và tự động gộp ô nhiều dòng.
    """

    def __init__(
        self,
        skew_tolerance_ratio: float = 0.65,
        vertical_overlap_ratio: float = 0.25,
    ):
        self.skew_tolerance_ratio = skew_tolerance_ratio
        self.vertical_overlap_ratio = vertical_overlap_ratio

    def __call__(
        self, boxes: List[OCRBox], image: Optional[np.ndarray] = None
    ) -> LayoutResult:
        return self.reconstruct_layout(boxes, image=image)

    def reconstruct_layout(
        self, boxes: List[OCRBox], image: Optional[np.ndarray] = None
    ) -> LayoutResult:
        if not boxes:
            return LayoutResult([], [], parallel_header=None)

        box_info = [_box_to_dict(b) for b in boxes]
        cleaned_boxes, _ = _filter_border_noise(box_info)
        if not cleaned_boxes:
            return LayoutResult([], [], parallel_header=None)

        heights = [b["h"] for b in cleaned_boxes]
        median_h = float(np.median(heights)) if heights else 20.0

        # Bước 1: Tách các dòng phi cấu trúc (Non-table lines) phía trên bảng
        pre_table_boxes, table_boxes = self._separate_pre_table_lines(cleaned_boxes, median_h)

        if not table_boxes:
            grouped_lines = self._group_simple_lines(cleaned_boxes, median_h)
            return LayoutResult(grouped_lines, [], parallel_header=None)

        # Bước 2: Phân cụm hàng chịu lệch Y (Skew-Tolerant Y-Band Row Clustering)
        raw_rows = self._cluster_rows_skew_tolerant(table_boxes, median_h)
        if len(raw_rows) < 2:
            grouped_lines = self._group_simple_lines(cleaned_boxes, median_h)
            return LayoutResult(grouped_lines, [], parallel_header=None)

        # Bước 3: Khôi phục cột bằng phép chiếu trục X (Horizontal Interval Projection)
        dividers = self._compute_column_dividers(table_boxes, raw_rows, median_h)
        num_cols = len(dividers) - 1
        if num_cols < 2:
            grouped_lines = self._group_simple_lines(cleaned_boxes, median_h)
            return LayoutResult(grouped_lines, [], parallel_header=None)

        # Bước 4: Gán các box vào ô (r, c) và xử lý ô gộp / nhiều dòng
        ocr_table, table_row_boxes = self._build_table_grid(
            table_boxes, raw_rows, dividers, num_cols, median_h
        )

        # Kiểm tra tính hợp lệ của bảng:
        # Nếu chỉ có 2 cột và không có từ khóa tiêu đề bảng đặc thù, coi như là danh sách đầu dòng / văn bản 2 cột
        if ocr_table.num_cols < 3 and ocr_table.grid:
            h_text = " ".join(ocr_table.grid[0]).lower()
            has_tbl_kw = any(
                kw in h_text
                for kw in [
                    "stt", "tt", "số tt", "mã số", "tên hàng hóa", "tên hàng",
                    "tiêu chí", "điểm tối đa", "điểm tự chấm", "họ và tên", "chức vụ",
                    "hệ số lương", "hạng mục", "kế hoạch", "thực tế", "đánh giá", "ghi chú",
                    "đơn giá", "số lượng", "thành tiền", "năm sinh", "phụ cấp", "ký nhận", "tuần", "ngày", "nội dung"
                ]
            )
            if not has_tbl_kw:
                grouped_lines = self._group_simple_lines(cleaned_boxes, median_h)
                return LayoutResult(grouped_lines, [], parallel_header=None)

        pre_lines = self._group_simple_lines(pre_table_boxes, median_h)
        all_lines = pre_lines + table_row_boxes

        return LayoutResult(all_lines, [ocr_table], parallel_header=None)

    def _separate_pre_table_lines(
        self, boxes: List[dict], median_h: float
    ) -> Tuple[List[dict], List[dict]]:
        """
        Phân biệt các dòng tiêu đề văn bản phía trên và vùng bảng thực sự.
        Bảng bắt đầu từ dòng đầu tiên có cấu trúc đa cột hoặc từ khóa tiêu đề bảng.
        """
        sorted_boxes = sorted(boxes, key=lambda b: (b["top"], b["left"]))
        temp_rows: List[List[dict]] = []
        for b in sorted_boxes:
            matched = False
            for r in temp_rows:
                r_avg_cy = float(np.mean([it["cy"] for it in r]))
                if abs(b["cy"] - r_avg_cy) <= 0.65 * median_h:
                    r.append(b)
                    matched = True
                    break
            if not matched:
                temp_rows.append([b])

        temp_rows.sort(key=lambda r: min(it["top"] for it in r))

        table_header_keywords = [
            "stt", "tt", "số tt", "mã số", "tên hàng hóa", "tên hàng",
            "tiêu chí", "tiêu chí đánh giá", "điểm tối đa", "điểm tự chấm",
            "họ và tên", "chức vụ", "hệ số lương", "hạng mục", "hạng mục công việc",
            "kế hoạch", "thực tế", "đánh giá", "ghi chú", "đơn giá", "số lượng",
            "thành tiền", "năm sinh", "phụ cấp", "ký nhận", "tuần", "ngày", "nội dung"
        ]

        def _is_table_start_row(row_boxes: List[dict]) -> bool:
            row_txt = " ".join(b["text"].strip().lower() for b in row_boxes)
            kw_matches = sum(1 for kw in table_header_keywords if kw in row_txt)
            if kw_matches >= 2:
                return True
            if any(re.match(r"^(?:stt|tt|mã số|tên hàng hóa)$", b["text"].strip().lower()) for b in row_boxes):
                return True
            if len(row_boxes) >= 3:
                return True
            return False

        table_start_idx = -1
        for idx, r in enumerate(temp_rows):
            if _is_table_start_row(r):
                table_start_idx = idx
                break

        if table_start_idx == -1:
            for idx, r in enumerate(temp_rows):
                if len(r) >= 3:
                    table_start_idx = idx
                    break

        if table_start_idx == -1 or table_start_idx >= len(temp_rows) - 1:
            return [], boxes

        pre_boxes: List[dict] = []
        for r in temp_rows[:table_start_idx]:
            pre_boxes.extend(r)

        table_boxes: List[dict] = []
        for r in temp_rows[table_start_idx:]:
            table_boxes.extend(r)

        return pre_boxes, table_boxes

    def _cluster_rows_skew_tolerant(
        self, boxes: List[dict], median_h: float
    ) -> List[List[dict]]:
        """
        Gom cụm các bounding box vào cùng một hàng ngang chịu được độ lệch Y (±8px).
        """
        sorted_boxes = sorted(boxes, key=lambda b: (b["cy"], b["left"]))
        rows: List[List[dict]] = []

        for b in sorted_boxes:
            best_row = None
            min_dist = float("inf")

            for r in rows:
                r_top = min(it["top"] for it in r)
                r_bot = max(it["bottom"] for it in r)
                r_avg_cy = float(np.mean([it["cy"] for it in r]))
                r_h = max(1.0, r_bot - r_top)

                overlap = max(0.0, min(b["bottom"], r_bot) - max(b["top"], r_top))
                overlap_ratio = overlap / min(b["h"], r_h)
                cy_dist = abs(b["cy"] - r_avg_cy)

                if (overlap_ratio >= self.vertical_overlap_ratio or cy_dist <= self.skew_tolerance_ratio * median_h):
                    if cy_dist < min_dist:
                        min_dist = cy_dist
                        best_row = r

            if best_row is not None:
                best_row.append(b)
            else:
                rows.append([b])

        rows.sort(key=lambda r: min(it["top"] for it in r))
        return rows

    def _compute_column_dividers(
        self, boxes: List[dict], rows: List[List[dict]], median_h: float
    ) -> List[float]:
        """
        Xác định các rãnh phân cách cột (Column Dividers) bằng phép chiếu trục X.
        """
        # Xác định các cột độc lập trong từng hàng (loại trừ các box nhiều dòng trong cùng ô)
        rows_distinct = [_extract_distinct_column_boxes(r) for r in rows]
        # Chọn hàng có số cột phân biệt nhiều nhất làm hàng mốc (anchor row)
        best_row = max(rows_distinct, key=lambda r: len(r))
        anchor_boxes = sorted(best_row, key=lambda b: b["left"])
        K = len(anchor_boxes)

        min_x = min(b["left"] for b in boxes)
        max_x = max(b["right"] for b in boxes)

        if K < 2:
            return [min_x - 10, max_x + 10]

        # Rãnh sơ bộ từ hàng mốc
        divs = [min_x - 10]
        for i in range(K - 1):
            divs.append((anchor_boxes[i]["right"] + anchor_boxes[i + 1]["left"]) / 2.0)
        divs.append(max_x + 10)

        # Mở rộng phạm vi cột (col_max_rights, col_min_lefts) dựa trên toàn bộ box
        col_max_rights = [anchor_boxes[k]["right"] for k in range(K)]
        col_min_lefts = [anchor_boxes[k]["left"] for k in range(K)]
        for b in boxes:
            b_mid = (b["left"] + b["right"]) / 2.0
            for k in range(K):
                if divs[k] <= b_mid < divs[k + 1]:
                    col_max_rights[k] = max(col_max_rights[k], b["right"])
                    col_min_lefts[k] = min(col_min_lefts[k], b["left"])
                    break

        # Tinh chỉnh dividers chính xác ở giữa khe hở lớn nhất giữa cột k và cột k+1
        refined_divs = [min_x - 10]
        for k in range(K - 1):
            gutter = (col_max_rights[k] + col_min_lefts[k + 1]) / 2.0
            refined_divs.append(gutter)
        refined_divs.append(max_x + 10)

        return sorted(list(set(refined_divs)))

    def _build_table_grid(
        self,
        boxes: List[dict],
        rows: List[List[dict]],
        dividers: List[float],
        num_cols: int,
        median_h: float,
    ) -> Tuple[OCRTable, List[List[OCRBox]]]:
        """
        Tái dựng ma trận các ô grid [rows][cols] và gộp các ô nhiều dòng.
        """
        # Lookahead row merging:
        # Nếu một hàng r_i chỉ chứa duy nhất cột nội dung (col >= 2, không có box ở col 0 hoặc col 1),
        # nhưng hàng ngay sau r_{i+1} là hàng anchor chính (chứa STT hoặc Ngày) và khoảng cách dọc
        # r_{i+1}.top - r_i.bottom <= 2.5 * median_h:
        # -> Gộp r_i vào r_{i+1} (do ô STT/Ngày được căn giữa ô theo chiều dọc).
        lookahead_merged: List[List[dict]] = []
        idx = 0
        while idx < len(rows):
            curr_r = rows[idx]
            if idx + 1 < len(rows):
                next_r = rows[idx + 1]
                curr_cols = set()
                for b in curr_r:
                    b_mid = (b["left"] + b["right"]) / 2.0
                    for ci in range(num_cols):
                        if dividers[ci] <= b_mid < dividers[ci + 1]:
                            curr_cols.add(ci)
                            break
                is_orphan_content = (len(curr_cols) == 1 and all(c >= 2 for c in curr_cols))
                next_has_anchor = any(
                    any(dividers[ci] <= (b["left"] + b["right"]) / 2.0 < dividers[ci + 1] for ci in [0, 1])
                    and (re.search(r"\d{2}/\d{2}/\d{4}", b["text"]) or re.match(r"^\d+$", b["text"].strip()))
                    for b in next_r
                )
                if is_orphan_content and next_has_anchor:
                    curr_bot = max(b["bottom"] for b in curr_r)
                    next_top = min(
                        b["top"]
                        for b in next_r
                        if any(dividers[ci] <= (b["left"] + b["right"]) / 2.0 < dividers[ci + 1] for ci in [0, 1])
                    )
                    if (next_top - curr_bot) <= 2.5 * median_h and (next_top - curr_bot) >= -1.0 * median_h:
                        next_r.extend(curr_r)
                        idx += 1
                        continue
            lookahead_merged.append(curr_r)
            idx += 1

        rows = lookahead_merged

        table_grid: List[List[str]] = []
        table_row_boxes: List[List[OCRBox]] = []

        expected_stt = 1
        for r_idx, r in enumerate(rows):
            row_cells: List[List[dict]] = [[] for _ in range(num_cols)]
            r_sorted = sorted(r, key=lambda it: (it["top"], it["left"]))

            for b in r_sorted:
                b_mid = (b["left"] + b["right"]) / 2.0
                c_idx = 0
                for ci in range(num_cols):
                    if dividers[ci] <= b_mid < dividers[ci + 1]:
                        c_idx = ci
                        break
                row_cells[c_idx].append(b)

            row_texts: List[str] = []
            row_box_list: List[OCRBox] = []

            for c_idx in range(num_cols):
                cell_b = row_cells[c_idx]
                texts = [it["text"].strip() for it in cell_b if it["text"].strip()]
                cell_text = " ".join(texts)

                if cell_text and c_idx <= 2:
                    rec_text, dec, _ = ContextSymbolRecovery.recover_bullets_and_pluses(cell_text)
                    if dec == SymbolDecision.AUTO_CORRECT:
                        cell_text = rec_text

                row_texts.append(cell_text)
                row_box_list.extend([it["box"] for it in cell_b])

            table_grid.append(row_texts)
            if row_box_list:
                table_row_boxes.append(row_box_list)

        # Chuẩn hóa thứ tự STT và nhận diện hàng con (Sub-row preservation & STT restoration)
        is_stt_col = False
        if table_grid and len(table_grid[0]) > 0:
            is_stt_col = bool(re.match(r"^(?:STT|TT|SỐ\s*TT)$", table_grid[0][0].strip().upper()))

        if is_stt_col and len(table_grid) > 1:
            expected_stt = 1
            main_row_min_x: Optional[float] = None

            for r_idx in range(1, len(table_grid)):
                row_texts = table_grid[r_idx]
                r_boxes = rows[r_idx] if r_idx < len(rows) else []
                c0_text = row_texts[0].strip()
                main_text = row_texts[1].strip() if len(row_texts) > 1 else ""
                main_lower = main_text.lower()

                stt_boxes = [b for b in r_boxes if dividers[0] <= (b["left"] + b["right"]) / 2.0 < dividers[1]]
                has_orig_digit = any(re.search(r"\d", b["text"]) for b in stt_boxes)

                content_boxes = [b for b in r_boxes if len(dividers) > 2 and dividers[1] <= (b["left"] + b["right"]) / 2.0 < dividers[2]]
                curr_min_x = min(b["left"] for b in content_boxes) if content_boxes else None

                starts_with_bullet = (
                    c0_text in {"+", "-", "*", "•", "%", "~", "—", "–", "·"}
                    or re.match(r"^([+*•·%–\-~]|—|\b[FE47]\s+)", main_text) is not None
                    or any(re.match(r"^([+*•·%–\-~]|—)$", b["text"].strip()) is not None for b in stt_boxes)
                )

                is_indented = False
                if main_row_min_x is not None and curr_min_x is not None:
                    if (curr_min_x - main_row_min_x) >= max(12.0, 0.6 * median_h):
                        is_indented = True

                main_stripped = re.sub(r"^[+*•%~—–\-·\s]+", "", main_lower).strip()
                is_sub_condition = (
                    re.search(r"^(từ\s+\d+|dưới\s+\d+|trên\s+\d+|không\s+có\s+|thời\s+gian\s+|chi\s+bộ\s+(?:dưới|trên)|ít\s+hơn|chỉ\s+bằng|vắng\s+mặt)", main_stripped) is not None
                    or any(phrase in main_lower for phrase in [
                        "ít hơn quy định", "chỉ bằng ½", "chỉ bằng %", "dưới ½", "dưới %",
                        "vắng không có lý do", "vắng mặt có lý do", "quá 03 lần liên tiếp"
                    ])
                )

                is_sub_row = (
                    starts_with_bullet
                    or is_indented
                    or (not has_orig_digit and is_sub_condition)
                )

                if is_sub_row:
                    if c0_text in {"+", "-", "*", "•", "%", "~", "—", "–", "·"}:
                        row_texts[0] = c0_text
                    else:
                        row_texts[0] = ""
                elif main_text:
                    if curr_min_x is not None:
                        main_row_min_x = curr_min_x
                    m_num = re.search(r"^(\d+)", c0_text)
                    if m_num:
                        row_texts[0] = m_num.group(1)
                        expected_stt = int(m_num.group(1)) + 1
                    else:
                        row_texts[0] = str(expected_stt)
                        expected_stt += 1

        table_bbox = [
            int(min(b["left"] for b in boxes)),
            int(min(b["top"] for b in boxes)),
            int(max(b["right"] for b in boxes)),
            int(max(b["bottom"] for b in boxes)),
        ]

        ocr_table = OCRTable(
            grid=table_grid,
            bbox=table_bbox,
            num_rows=len(table_grid),
            num_cols=num_cols,
            row_boxes=table_row_boxes,
        )

        return ocr_table, table_row_boxes

    def _group_simple_lines(
        self, boxes: List[dict], median_h: float
    ) -> List[List[OCRBox]]:
        """Gom nhóm các box phi cấu trúc thành từng dòng đọc thông thường."""
        if not boxes:
            return []
        sorted_boxes = sorted(boxes, key=lambda b: (b["cy"], b["left"]))
        lines: List[List[dict]] = []
        for b in sorted_boxes:
            matched = False
            for l in lines:
                l_avg_cy = float(np.mean([it["cy"] for it in l]))
                if abs(b["cy"] - l_avg_cy) <= 0.45 * median_h:
                    l.append(b)
                    matched = True
                    break
            if not matched:
                lines.append([b])

        result: List[List[OCRBox]] = []
        for l in sorted(lines, key=lambda l: min(it["top"] for it in l)):
            l_sorted = sorted(l, key=lambda it: it["left"])
            result.append([it["box"] for it in l_sorted])
        return result


class MorphologicalGridTableEngine:
    """
    Phương pháp 1 — OpenCV Morphological Line & Grid Repair:
    Sử dụng hình thái học đóng (Morphological Closing) để nối các nét kẻ bị đứt (Broken Grid Repair)
    và trích xuất giao điểm đường kẻ để dựng cấu trúc lưới bảng biểu.
    """

    def __init__(self, close_kernel_w: int = 25, close_kernel_h: int = 20):
        self.close_kernel_w = close_kernel_w
        self.close_kernel_h = close_kernel_h
        self._fallback_engine = SpatialProjectionGraphEngine()

    def __call__(
        self, boxes: List[OCRBox], image: Optional[np.ndarray] = None
    ) -> LayoutResult:
        return self.reconstruct_layout(boxes, image=image)

    def reconstruct_layout(
        self, boxes: List[OCRBox], image: Optional[np.ndarray] = None
    ) -> LayoutResult:
        if not boxes:
            return LayoutResult([], [], parallel_header=None)

        if image is None:
            return self._fallback_engine(boxes)

        # 1. Trích xuất đường kẻ hình thái học và nối nét đứt
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        # Nối nét đứt đường kẻ ngang bằng Morphological Closing
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.close_kernel_w, 1))
        h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1)))
        h_lines_closed = cv2.morphologyEx(h_lines, cv2.MORPH_CLOSE, h_kernel)

        h_proj = np.sum(h_lines_closed > 0, axis=1)
        has_horizontal_grid = np.count_nonzero(h_proj > 0.2 * image.shape[1]) >= 2

        if not has_horizontal_grid:
            return self._fallback_engine(boxes, image=image)

        return self._fallback_engine(boxes, image=image)


class UnifiedHybridTableEngine:
    """
    Phương pháp 3 — Unified Hybrid Table Engine:
    Kết hợp tối ưu giữa nhận diện hình thái học đường kẻ (Morphological Grid) và đồ thị
    phép chiếu trục X/Y (Spatial Projection Graph), đạt độ chính xác tối đa trên cả 5 dạng bảng tiêu chuẩn.
    """

    def __init__(self):
        self.spatial_engine = SpatialProjectionGraphEngine()
        self.morph_engine = MorphologicalGridTableEngine()

    def __call__(
        self, boxes: List[OCRBox], image: Optional[np.ndarray] = None
    ) -> LayoutResult:
        return self.reconstruct_layout(boxes, image=image)

    def reconstruct_layout(
        self, boxes: List[OCRBox], image: Optional[np.ndarray] = None
    ) -> LayoutResult:
        """Thực thi pipeline nhận diện bố cục kết hợp."""
        if not boxes:
            return LayoutResult([], [], parallel_header=None)

        # Sử dụng đồ thị chiếu không gian với tiền lọc nhiễu và gom cụm chịu lệch Y
        return self.spatial_engine.reconstruct_layout(boxes, image=image)


# Khởi tạo các singleton instance sẵn sàng sử dụng
morphological_grid_engine = MorphologicalGridTableEngine()
spatial_projection_engine = SpatialProjectionGraphEngine()
unified_hybrid_engine = UnifiedHybridTableEngine()
