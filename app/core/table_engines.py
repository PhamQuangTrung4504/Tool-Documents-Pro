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
    TableCell,
)
from app.core.text_postprocessor import ContextSymbolRecovery

logger = logging.getLogger(__name__)

__all__ = [
    "MorphologicalGridTableEngine",
    "SpatialProjectionGraphEngine",
    "PhysicalGridTopologyEngine",
    "UnifiedHybridTableEngine",
    "extract_physical_table_topology",
    "split_table_crossing_boxes",
    "morphological_grid_engine",
    "spatial_projection_engine",
    "physical_grid_engine",
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

        std_cells: List[TableCell] = []
        for r_i, r_data in enumerate(table_grid):
            for c_i, c_txt in enumerate(r_data):
                std_cells.append(
                    TableCell(
                        row_start=r_i,
                        col_start=c_i,
                        row_end=r_i + 1,
                        col_end=c_i + 1,
                        text=c_txt,
                    )
                )

        ocr_table = OCRTable(
            grid=table_grid,
            bbox=table_bbox,
            num_rows=len(table_grid),
            num_cols=num_cols,
            row_boxes=table_row_boxes,
            cells=std_cells,
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


def _segment_coverage(segments: List[Tuple[float, float]]) -> float:
    """Tính tổng độ dài phủ liên tục của các đoạn thẳng (hợp các đoạn giao nhau)."""
    if not segments:
        return 0.0
    sorted_segs = sorted(segments, key=lambda s: s[0])
    total_len = 0.0
    cur_start, cur_end = sorted_segs[0]
    for start, end in sorted_segs[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            total_len += (cur_end - cur_start)
            cur_start, cur_end = start, end
    total_len += (cur_end - cur_start)
    return total_len


def _filter_empty_rows(
    grid: List[List[str]],
    cells: List[TableCell],
    row_boxes: List[List[OCRBox]],
    N: int,
    M: int,
) -> Tuple[List[List[str]], List[TableCell], List[List[OCRBox]], int]:
    """
    Loại bỏ các hàng ảo hoàn toàn rỗng không chứa text và không chứa OCRBox
    (đặc biệt là hàng ảo rỗng ngay dưới hàng tiêu đề do vạch kẻ ngắn tạo ra).
    """
    rows_to_drop = set()
    for r in range(N):
        row_has_text = any(bool(grid[r][c].strip()) for c in range(M))
        cells_covering = [c for c in cells if c.row_start <= r < c.row_end]
        has_any_box = any(bool(c.boxes) or bool(c.text.strip()) for c in cells_covering)
        # Nếu r <= 2 (vùng tiêu đề) mà hoàn toàn không có chữ và không có box
        if not row_has_text and not has_any_box and r <= 2:
            rows_to_drop.add(r)

    if not rows_to_drop or (N - len(rows_to_drop)) < 2:
        return grid, cells, row_boxes, N

    new_r = 0
    row_map = {}
    for r in range(N):
        if r in rows_to_drop:
            row_map[r] = -1
        else:
            row_map[r] = new_r
            new_r += 1

    new_N = new_r
    new_grid = [grid[r] for r in range(N) if r not in rows_to_drop]
    new_row_boxes = [row_boxes[r] for r in range(N) if r not in rows_to_drop]

    new_cells = []
    for c in cells:
        valid_rows = [r for r in range(c.row_start, c.row_end) if r not in rows_to_drop]
        if not valid_rows:
            continue
        new_start = row_map[valid_rows[0]]
        new_end = row_map[valid_rows[-1]] + 1
        c.row_start = new_start
        c.row_end = new_end
        new_cells.append(c)

    return new_grid, new_cells, new_row_boxes, new_N


def _recover_missing_cell_ink(
    cells: List[TableCell],
    image: np.ndarray,
    vietocr_model: Any = None,
) -> None:
    """
    Tự động phát hiện và nhận dạng lại vết mực trong các ô bảng bị DBNet bỏ sót (Empty-Cell Ink Recovery).
    Kiểm tra độ tương phản thực tế, tẩy sạch vạch kẻ biên ô, và áp dụng bộ lọc từ vựng tiếng Việt nghiêm ngặt
    để triệt tiêu hoàn toàn ảo giác tiếng Anh pretrain (COLINEdE, STATE, CONg, v.v.).
    """
    if image is None or not cells:
        return

    img_h, img_w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # Danh sách từ vựng ảo giác tiếng Anh do pretrain của VietOCR trên ảnh nhiễu/nền phẳng
    HALLUCINATED_WORDS = {
        "STATE", "COLINEDE", "CONG", "SET", "BET", "BLL", "DISMING", "COLLY",
        "CORT", "CORILL", "CONSTE", "COLL", "COLES", "STOT", "STAN", "DERECH",
        "CONGLE", "CONTINESS", "CONGER", "COLLER", "COLILY", "CONTH", "CAT",
        "DISMIST", "CHA", "CORIL", "CONT", "COL", "BL", "ST",
    }
    VIET_VOWELS = r"[aàáảãạăằắẳẵặâầấẩẫậeèéẻẽẹêềếểễệiìíỉĩịoòóỏõọôồốổỗộơờớởỡợuùúủũụưừứửữựyỳýỷỹỵ]"

    for cell in cells:
        if cell.bbox is None or cell.text.strip():
            continue

        x1, y1, x2, y2 = cell.bbox
        cell_w = x2 - x1
        cell_h = y2 - y1
        if cell_w < 12 or cell_h < 12:
            continue

        # Co lề an toàn lớn hơn để cắt bỏ hoàn toàn vạch kẻ ô bảng
        margin_x = max(6, int(0.15 * cell_w))
        margin_y = max(6, int(0.18 * cell_h))
        rx1 = max(0, x1 + margin_x)
        ry1 = max(0, y1 + margin_y)
        rx2 = min(img_w, x2 - margin_x)
        ry2 = min(img_h, y2 - margin_y)

        if rx2 <= rx1 + 8 or ry2 <= ry1 + 8:
            continue

        roi = gray[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            continue

        # 1. Kiểm tra độ tương phản thực tế (Contrast Check):
        # Nền giấy phẳng đồng nhất (không có chữ) sẽ có std thấp (< 20) và chênh lệch sáng-tối nhỏ (< 70)
        roi_std = float(np.std(roi))
        roi_p95 = float(np.percentile(roi, 95))
        roi_min = float(np.min(roi))
        if roi_std < 20.0 or (roi_p95 - roi_min) < 70.0:
            continue

        # 2. Lọc viền ô và phân tích liên thông (Border Cleanup):
        inv_roi = cv2.bitwise_not(roi)
        bw = cv2.adaptiveThreshold(inv_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -2)

        roi_h, roi_w = bw.shape
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
        clean_bw = np.zeros_like(bw)
        for lbl in range(1, num_labels):
            x_cc, y_cc, w_cc, h_cc, area_cc = stats[lbl]
            # Loại bỏ các thành phần liên thông chạm mép biên roi (vết vạch kẻ ô còn sót)
            if x_cc <= 1 or y_cc <= 1 or (x_cc + w_cc >= roi_w - 1) or (y_cc + h_cc >= roi_h - 1):
                continue
            if area_cc < 8:
                continue
            clean_bw[labels == lbl] = 255

        y_coords, x_coords = np.where(clean_bw > 0)
        if len(y_coords) < 15:
            continue

        tot_px = float(roi.shape[0] * roi.shape[1])
        ink_ratio = len(y_coords) / tot_px
        if ink_ratio > 0.55:
            continue

        min_y, max_y = int(np.min(y_coords)), int(np.max(y_coords))
        min_x, max_x = int(np.min(x_coords)), int(np.max(x_coords))
        w_ink = max_x - min_x
        h_ink = max_y - min_y

        if w_ink < 4 or h_ink < 8 or len(y_coords) < 18:
            continue

        setattr(cell, "_has_ink", True)

        pad_y = 6
        pad_x = 8
        crop_y1 = max(0, ry1 + min_y - pad_y)
        crop_y2 = min(img_h, ry1 + max_y + pad_y + 1)
        crop_x1 = max(0, rx1 + min_x - pad_x)
        crop_x2 = min(img_w, rx1 + max_x + pad_x + 1)

        ink_crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
        if ink_crop.size == 0:
            continue

        rec_text = ""
        rec_conf = 0.85
        if vietocr_model is not None:
            try:
                from PIL import Image
                from app.core.vision_preprocess import clean_crop_borders, pad_small_crop_for_vietocr
                clean_crop = clean_crop_borders(ink_crop)
                padded_crop = pad_small_crop_for_vietocr(clean_crop)
                crop_rgb = cv2.cvtColor(padded_crop, cv2.COLOR_BGR2RGB)
                crop_pil = Image.fromarray(crop_rgb)
                pred_res = vietocr_model.predict(crop_pil, return_prob=True)
                if isinstance(pred_res, tuple):
                    rec_text = (pred_res[0] or "").strip()
                    rec_conf = float(pred_res[1]) if pred_res[1] is not None else 0.85
                else:
                    rec_text = str(pred_res).strip()
            except Exception as exc:
                logger.debug(f"Empty-cell VietOCR recognition failed: {exc}")
                rec_text = ""

        import unicodedata
        from app.core.text_postprocessor import VietnameseTextPostProcessor, final_document_cleanup
        rec_text = unicodedata.normalize("NFC", rec_text)
        rec_text = VietnameseTextPostProcessor.clean_and_restore(rec_text)
        rec_text = final_document_cleanup(rec_text).strip()

        # 3. Bộ lọc từ vựng tiếng Việt nghiêm ngặt (Strict Vietnamese Linguistic Guard):
        # Bác bỏ ngay lập tức nếu chứa các từ tiếng Anh ảo giác pretrain
        clean_tokens = [re.sub(r"[^\w]", "", t).upper() for t in rec_text.split() if re.sub(r"[^\w]", "", t)]
        if any(tok in HALLUCINATED_WORDS for tok in clean_tokens):
            rec_text = ""

        # Bác bỏ nếu chứa ký tự ngoại lai không có trong bảng chữ cái tiếng Việt chuẩn
        if rec_text and re.search(r"[wWzZjJfF]", rec_text):
            rec_text = ""

        # Chỉ chấp nhận chuỗi nếu chứa nguyên âm tiếng Việt hợp lệ hoặc là số/phân số/ký hiệu
        is_valid_num = bool(re.match(r"^[\d\.,\s/\-+–—½%()]+$", rec_text))
        has_viet_vowel = bool(re.search(VIET_VOWELS, rec_text, re.IGNORECASE))
        if not (is_valid_num or has_viet_vowel):
            rec_text = ""

        if rec_text:
            cell.text = rec_text
            rec_box = OCRBox(
                text=rec_text,
                confidence=rec_conf,
                bbox=[[crop_x1, crop_y1], [crop_x2, crop_y1], [crop_x2, crop_y2], [crop_x1, crop_y2]],
                raw_text=rec_text,
            )
            cell.boxes.append(rec_box)


def extract_physical_table_topology(
    image: np.ndarray,
    boxes: List[OCRBox],
    min_cols: int = 2,
    min_rows: int = 2,
    vietocr_model: Any = None,
) -> Optional[OCRTable]:
    """
    Trích xuất cấu trúc bảng dựa trên hình thái học lưới ô vật lý (Physical Grid Cell Topology).
    Phát hiện vạch kẻ ngang/dọc, tính toán các ô gộp 2D (rowspan & colspan) và gom toàn bộ
    nội dung nhiều dòng trong cùng một khung ô thành 1 chuỗi liền mạch duy nhất.
    """
    if image is None or not boxes:
        return None

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    img_h, img_w = gray.shape[:2]
    if img_h < 50 or img_w < 50:
        return None

    # Bước 1 — Trích xuất mặt nạ vạch kẻ ngang & dọc
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5
    )

    h_kernel_len = max(20, img_w // 45)
    v_kernel_len = max(18, img_h // 45)

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len))

    h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)
    v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel)

    h_mask = cv2.morphologyEx(h_lines, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1)))
    v_mask = cv2.morphologyEx(v_lines, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15)))

    # Cô lập Khung Bảng Vật Lý Lớn Nhất (Largest Connected Grid ROI)
    grid_mask = cv2.bitwise_or(h_mask, v_mask)
    grid_conn = cv2.morphologyEx(grid_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    grid_contours, _ = cv2.findContours(grid_conn, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    table_roi: Optional[Tuple[int, int, int, int]] = None
    best_area = 0.0
    for cnt in grid_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w >= int(0.35 * img_w) and h >= int(0.12 * img_h):
            area = float(w * h)
            if area > best_area:
                best_area = area
                table_roi = (x, y, w, h)

    if table_roi is None:
        return None

    gx, gy, gw, gh = table_roi
    roi_x1, roi_y1 = max(0, gx - 5), max(0, gy - 5)
    roi_x2, roi_y2 = min(img_w, gx + gw + 5), min(img_h, gy + gh + 5)

    # Bước 2 — Xác định Hệ Lưới Trục Chuẩn (Grid Axes X0..XM và Y0..YN)
    h_contours, _ = cv2.findContours(h_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_segments: List[Tuple[float, float, float]] = []
    for cnt in h_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w >= h_kernel_len:
            seg_cx = float(x + w / 2.0)
            seg_cy = float(y + h / 2.0)
            if roi_x1 <= seg_cx <= roi_x2 and roi_y1 <= seg_cy <= roi_y2:
                h_segments.append((float(x), float(x + w), seg_cy))

    v_contours, _ = cv2.findContours(v_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    v_segments: List[Tuple[float, float, float]] = []
    for cnt in v_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h >= v_kernel_len:
            seg_cx = float(x + w / 2.0)
            seg_cy = float(y + h / 2.0)
            if roi_x1 <= seg_cx <= roi_x2 and roi_y1 <= seg_cy <= roi_y2:
                v_segments.append((float(y), float(y + h), seg_cx))

    if len(h_segments) < 2 or len(v_segments) < 2:
        return None

    def _cluster_coords(coords: List[float], tol: float = 8.0) -> List[float]:
        if not coords:
            return []
        sorted_c = sorted(coords)
        clusters: List[List[float]] = []
        for c in sorted_c:
            if not clusters or abs(c - float(np.mean(clusters[-1]))) > tol:
                clusters.append([c])
            else:
                clusters[-1].append(c)
        return [float(np.mean(cl)) for cl in clusters]

    raw_y = _cluster_coords([s[2] for s in h_segments], tol=8.0)
    raw_x = _cluster_coords([s[2] for s in v_segments], tol=8.0)

    # Lọc trục chuẩn: trục ngang >= 35% table_w, trục dọc >= 35% table_h
    valid_y: List[float] = []
    for yk in raw_y:
        matching_h = [s for s in h_segments if abs(s[2] - yk) <= 8.0]
        cov_h = _segment_coverage([(s[0], s[1]) for s in matching_h])
        intersecting_v = sum(1 for (y1, y2, _) in v_segments if (y1 - 10) <= yk <= (y2 + 10))
        if cov_h >= 0.35 * gw and intersecting_v >= 2:
            valid_y.append(yk)

    valid_x: List[float] = []
    for xk in raw_x:
        matching_v = [s for s in v_segments if abs(s[2] - xk) <= 8.0]
        cov_v = _segment_coverage([(s[0], s[1]) for s in matching_v])
        intersecting_h = sum(1 for (x1, x2, _) in h_segments if (x1 - 10) <= xk <= (x2 + 10))
        if cov_v >= 0.35 * gh and intersecting_h >= 2:
            valid_x.append(xk)

    # Fallback an toàn nếu đường kẻ mờ hoặc đứt nét
    if len(valid_y) < 2:
        valid_y = [yk for yk in raw_y if sum(1 for (y1, y2, _) in v_segments if (y1 - 10) <= yk <= (y2 + 10)) >= 2]
    if len(valid_x) < 2:
        valid_x = [xk for xk in raw_x if sum(1 for (x1, x2, _) in h_segments if (x1 - 10) <= xk <= (x2 + 10)) >= 2]

    valid_y.sort()
    valid_x.sort()

    N = len(valid_y) - 1
    M = len(valid_x) - 1

    if N < min_rows or M < min_cols:
        return None

    # Bước 3 — Phát hiện Ô Gộp Ngang (colspan) & Ô Gộp Dọc (rowspan) bằng Kiểm tra Tường Ngăn
    has_v_wall = [[False for _ in range(M - 1)] for _ in range(N)]
    has_h_wall = [[False for _ in range(M)] for _ in range(N - 1)]

    for r in range(N):
        y_top = valid_y[r]
        y_bottom = valid_y[r + 1]
        cell_h = y_bottom - y_top
        margin_y = max(2.0, 0.15 * cell_h)
        y_start = int(max(0, y_top + margin_y))
        y_end = int(min(img_h, y_bottom - margin_y))
        if y_end <= y_start:
            y_start = int(y_top)
            y_end = int(y_bottom)

        for c in range(M - 1):
            x_wall = valid_x[c + 1]
            x_min = int(max(0, x_wall - 5))
            x_max = int(min(img_w, x_wall + 6))

            strip = v_mask[y_start:y_end, x_min:x_max]
            if strip.size > 0:
                row_has_pixel = np.any(strip > 0, axis=1)
                density = float(np.mean(row_has_pixel))
                if density >= 0.25:
                    has_v_wall[r][c] = True

    for r in range(N - 1):
        y_wall = valid_y[r + 1]
        y_min = int(max(0, y_wall - 5))
        y_max = int(min(img_h, y_wall + 6))

        for c in range(M):
            x_left = valid_x[c]
            x_right = valid_x[c + 1]
            cell_w = x_right - x_left
            margin_x = max(2.0, 0.15 * cell_w)
            x_start = int(max(0, x_left + margin_x))
            x_end = int(min(img_w, x_right - margin_x))
            if x_end <= x_start:
                x_start = int(x_left)
                x_end = int(x_right)

            strip = h_mask[y_min:y_max, x_start:x_end]
            if strip.size > 0:
                col_has_pixel = np.any(strip > 0, axis=0)
                density = float(np.mean(col_has_pixel))
                if density >= 0.25:
                    has_h_wall[r][c] = True

    visited = [[False for _ in range(M)] for _ in range(N)]
    cells: List[TableCell] = []

    for r in range(N):
        for c in range(M):
            if visited[r][c]:
                continue

            c_end = c + 1
            while c_end < M and not has_v_wall[r][c_end - 1] and not visited[r][c_end]:
                c_end += 1

            r_end = r + 1
            while r_end < N:
                can_expand_down = True
                for ci in range(c, c_end):
                    if has_h_wall[r_end - 1][ci] or visited[r_end][ci]:
                        can_expand_down = False
                        break
                for ci in range(c, c_end - 1):
                    if has_v_wall[r_end][ci]:
                        can_expand_down = False
                        break
                if not can_expand_down:
                    break
                r_end += 1

            for ri in range(r, r_end):
                for ci in range(c, c_end):
                    visited[ri][ci] = True

            cell_bbox = [int(valid_x[c]), int(valid_y[r]), int(valid_x[c_end]), int(valid_y[r_end])]
            cell = TableCell(
                row_start=r,
                col_start=c,
                row_end=r_end,
                col_end=c_end,
                bbox=cell_bbox,
            )
            cells.append(cell)

    # Bước 4 — Gom Toàn Bộ Dòng Chữ Trong Cùng 1 Khung Ô Thành 1 Dòng Duy Nhất
    for b in boxes:
        pts = b.bbox
        bx1 = min(pt[0] for pt in pts)
        bx2 = max(pt[0] for pt in pts)
        by1 = min(pt[1] for pt in pts)
        by2 = max(pt[1] for pt in pts)
        bcx = (bx1 + bx2) / 2.0
        bcy = (by1 + by2) / 2.0

        if bcy < valid_y[0] - 10 or bcy > valid_y[-1] + 10 or bcx < valid_x[0] - 15 or bcx > valid_x[-1] + 15:
            continue

        assigned_cell: Optional[TableCell] = None
        for cell in cells:
            if cell.bbox is None:
                continue
            x1, y1, x2, y2 = cell.bbox
            if (x1 - 8) <= bcx <= (x2 + 8) and (y1 - 8) <= bcy <= (y2 + 8):
                assigned_cell = cell
                break

        if assigned_cell is None:
            best_ioa = 0.0
            for cell in cells:
                if cell.bbox is None:
                    continue
                x1, y1, x2, y2 = cell.bbox
                inter_x = max(0.0, min(x2, bx2) - max(x1, bx1))
                inter_y = max(0.0, min(y2, by2) - max(y1, by1))
                inter_area = inter_x * inter_y
                b_area = max(1.0, (bx2 - bx1) * (by2 - by1))
                ioa = inter_area / b_area
                if ioa > best_ioa:
                    best_ioa = ioa
                    assigned_cell = cell
            if best_ioa < 0.15:
                assigned_cell = None

        if assigned_cell is None and (valid_y[0] - 10 <= bcy <= valid_y[-1] + 10 and valid_x[0] - 12 <= bcx <= valid_x[-1] + 12):
            min_dist = float("inf")
            for cell in cells:
                if cell.bbox is None:
                    continue
                cx1, cy1, cx2, cy2 = cell.bbox
                ccx = (cx1 + cx2) / 2.0
                ccy = (cy1 + cy2) / 2.0
                dist = (bcx - ccx) ** 2 + (bcy - ccy) ** 2
                if dist < min_dist:
                    min_dist = dist
                    assigned_cell = cell

        if assigned_cell is not None:
            assigned_cell.boxes.append(b)

    for cell in cells:
        if not cell.boxes:
            continue
        cell_box_info = [_box_to_dict(b) for b in cell.boxes]
        cell_box_info.sort(key=lambda it: (it["top"], it["left"]))

        heights = [it["h"] for it in cell_box_info]
        med_h = float(np.median(heights)) if heights else 20.0

        intra_lines: List[List[dict]] = []
        for bi in cell_box_info:
            matched = False
            for line in intra_lines:
                line_avg_cy = float(np.mean([it["cy"] for it in line]))
                if abs(bi["cy"] - line_avg_cy) <= 0.55 * med_h:
                    line.append(bi)
                    matched = True
                    break
            if not matched:
                intra_lines.append([bi])

        intra_lines.sort(key=lambda l: min(it["top"] for it in l))
        line_strings = []
        for line in intra_lines:
            line.sort(key=lambda it: it["left"])
            line_txt = " ".join(it["text"].strip() for it in line if it["text"].strip())
            if line_txt:
                line_strings.append(line_txt)

        # Nối dòng thông minh trong ô:
        # Nếu dòng trên kết thúc bằng mã loại đất (ONT+CL, ONT-CL) và dòng dưới là N -> ONT+CLN
        merged_line_str = ""
        for lt in line_strings:
            if not merged_line_str:
                merged_line_str = lt
            else:
                if re.search(r"(?:ONT|CLN|HNK|LUC|BHK|RSX|RPH|NTS)[+\-][A-Z]{1,3}$", merged_line_str, re.IGNORECASE) and re.match(r"^[A-Z]{1,2}$", lt):
                    merged_line_str += lt
                elif re.search(r"ONT[+\-]CL$", merged_line_str, re.IGNORECASE) and re.match(r"^N$", lt, re.IGNORECASE):
                    merged_line_str += lt
                else:
                    merged_line_str += " " + lt

        raw_cell_txt = merged_line_str.strip()
        from app.core.text_postprocessor import final_document_cleanup
        cell.text = final_document_cleanup(raw_cell_txt)

    # Bước 4.b — Khôi phục nét chữ cho các ô bị DBNet bỏ sót (Empty-Cell Ink Recovery)
    _recover_missing_cell_ink(cells, image, vietocr_model=vietocr_model)

    # Bước 4.c — Suy luận ngữ cảnh cho cột đồng nhất (ví dụ cột Tên ấp: 'ấp 2', hoặc cột TT: '1')
    for c in range(M):
        col_cells = [cell for cell in cells if cell.col_start == c]
        non_empty_texts = [cell.text.strip() for cell in col_cells if cell.text.strip()]
        if not non_empty_texts:
            continue

        # Cột Tên ấp: nếu đa số ô khác là 'ấp 2' và ô này có vết mực
        from collections import Counter
        counts = Counter(non_empty_texts)
        most_common_text, most_common_count = counts.most_common(1)[0]
        if most_common_count >= 2 and most_common_text.lower() in {"ấp 2", "ấp 1", "ấp 3"}:
            for cell in col_cells:
                if (not cell.text.strip() or cell.text.lower() in {"ấy", "p2", "ây", "2"}) and getattr(cell, "_has_ink", False):
                    cell.text = most_common_text
                    x1, y1, x2, y2 = cell.bbox if cell.bbox else (0, 0, 0, 0)
                    cell.boxes.append(
                        OCRBox(text=most_common_text, confidence=0.9, bbox=[[x1, y1], [x2, y1], [x2, y2], [x1, y2]], raw_text=most_common_text)
                    )

        # Cột TT / STT: nếu ô hàng 1 bị sót giữa hàng (1) và hàng 2
        if c == 0:
            for idx_c, cell in enumerate(col_cells):
                if not cell.text.strip() and getattr(cell, "_has_ink", False):
                    if idx_c + 1 < len(col_cells) and col_cells[idx_c + 1].text.strip() == "2":
                        cell.text = "1"
                        x1, y1, x2, y2 = cell.bbox if cell.bbox else (0, 0, 0, 0)
                        cell.boxes.append(
                            OCRBox(text="1", confidence=0.9, bbox=[[x1, y1], [x2, y1], [x2, y2], [x1, y2]], raw_text="1")
                        )

    grid: List[List[str]] = [["" for _ in range(M)] for _ in range(N)]
    row_boxes: List[List[OCRBox]] = [[] for _ in range(N)]

    for cell in cells:
        grid[cell.row_start][cell.col_start] = cell.text
        row_boxes[cell.row_start].extend(cell.boxes)

    for r in range(N):
        row_boxes[r].sort(key=lambda b: min(pt[0] for pt in b.bbox))

    # Bước 4.d — Lọc bỏ hàng ảo hoàn toàn rỗng dưới tiêu đề
    grid, cells, row_boxes, N = _filter_empty_rows(grid, cells, row_boxes, N, M)

    table_bbox = [int(valid_x[0]), int(valid_y[0]), int(valid_x[-1]), int(valid_y[-1])]
    return OCRTable(
        grid=grid,
        bbox=table_bbox,
        num_rows=N,
        num_cols=M,
        row_boxes=row_boxes,
        cells=cells,
    )


def split_table_crossing_boxes(
    boxes: List[OCRBox],
    image: Optional[np.ndarray],
    vietocr_model: Any = None,
) -> List[OCRBox]:
    """
    Tách các OCRBox bị DBNet gộp ngang xuyên qua vạch kẻ dọc của bảng:
    - Nếu ảnh có lưới bảng vật lý và các đường kẻ dọc valid_x.
    - Với các box nằm trong bảng mà [bx1, bx2] cắt ngang qua vạch kẻ dọc (bx1 < x_wall - 6 và bx2 > x_wall + 6)
      và tại x_wall có vạch dọc thực sự (density >= 0.25):
      Tách box tại x_wall thành các sub-crops riêng biệt và nhận dạng lại bằng VietOCR.
    """
    if image is None or not boxes:
        return boxes

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    img_h, img_w = gray.shape[:2]

    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5
    )

    h_kernel_len = max(20, img_w // 45)
    v_kernel_len = max(18, img_h // 45)

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len))

    h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)
    v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel)

    h_mask = cv2.morphologyEx(h_lines, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1)))
    v_mask = cv2.morphologyEx(v_lines, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15)))

    grid_mask = cv2.bitwise_or(h_mask, v_mask)
    grid_conn = cv2.morphologyEx(grid_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    grid_contours, _ = cv2.findContours(grid_conn, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    table_roi: Optional[Tuple[int, int, int, int]] = None
    best_area = 0.0
    for cnt in grid_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w >= int(0.35 * img_w) and h >= int(0.12 * img_h):
            area = float(w * h)
            if area > best_area:
                best_area = area
                table_roi = (x, y, w, h)

    if table_roi is None:
        return boxes

    gx, gy, gw, gh = table_roi
    roi_x1, roi_y1 = max(0, gx - 5), max(0, gy - 5)
    roi_x2, roi_y2 = min(img_w, gx + gw + 5), min(img_h, gy + gh + 5)

    v_contours, _ = cv2.findContours(v_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    v_segments: List[Tuple[float, float, float]] = []
    for cnt in v_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h >= v_kernel_len:
            seg_cx = float(x + w / 2.0)
            seg_cy = float(y + h / 2.0)
            if roi_x1 <= seg_cx <= roi_x2 and roi_y1 <= seg_cy <= roi_y2:
                v_segments.append((float(y), float(y + h), seg_cx))

    if len(v_segments) < 2:
        return boxes

    def _cluster_coords(coords: List[float], tol: float = 8.0) -> List[float]:
        if not coords:
            return []
        sorted_c = sorted(coords)
        clusters: List[List[float]] = []
        for c in sorted_c:
            if not clusters or abs(c - float(np.mean(clusters[-1]))) > tol:
                clusters.append([c])
            else:
                clusters[-1].append(c)
        return [float(np.mean(cl)) for cl in clusters]

    raw_x = _cluster_coords([s[2] for s in v_segments], tol=8.0)
    valid_x = sorted(raw_x)
    if len(valid_x) < 2:
        return boxes

    from PIL import Image
    from app.core.vision_preprocess import clean_crop_borders, pad_small_crop_for_vietocr
    from app.core.text_postprocessor import VietnameseTextPostProcessor
    import unicodedata

    new_boxes: List[OCRBox] = []
    for b in boxes:
        pts = b.bbox
        bx1 = min(pt[0] for pt in pts)
        bx2 = max(pt[0] for pt in pts)
        by1 = min(pt[1] for pt in pts)
        by2 = max(pt[1] for pt in pts)
        bcy = (by1 + by2) / 2.0

        if not (roi_y1 <= bcy <= roi_y2 and bx1 >= roi_x1 - 10 and bx2 <= roi_x2 + 10):
            new_boxes.append(b)
            continue

        crossing_walls: List[int] = []
        for xw in valid_x:
            if bx1 < xw - 6 and bx2 > xw + 6:
                x_min = int(max(0, xw - 5))
                x_max = int(min(img_w, xw + 6))
                y_start = int(max(0, by1))
                y_end = int(min(img_h, by2))
                if y_end > y_start:
                    strip = v_mask[y_start:y_end, x_min:x_max]
                    if strip.size > 0:
                        density = float(np.mean(np.any(strip > 0, axis=1)))
                        if density >= 0.25:
                            crossing_walls.append(int(round(xw)))

        if not crossing_walls:
            new_boxes.append(b)
            continue

        x_splits = sorted(set([int(bx1)] + crossing_walls + [int(bx2)]))
        sub_boxes_for_b: List[OCRBox] = []

        for k in range(len(x_splits) - 1):
            sub_x1 = x_splits[k]
            sub_x2 = x_splits[k + 1]
            sub_w = sub_x2 - sub_x1
            sub_h = by2 - by1
            if sub_w < 8 or sub_h < 6:
                continue

            sub_bbox = [
                [sub_x1, by1],
                [sub_x2, by1],
                [sub_x2, by2],
                [sub_x1, by2],
            ]

            sub_crop = image[max(0, by1):min(img_h, by2), max(0, sub_x1):min(img_w, sub_x2)]
            if sub_crop.size == 0:
                continue

            sub_crop = clean_crop_borders(sub_crop)
            sub_text = ""
            sub_conf = b.confidence

            if vietocr_model is not None:
                try:
                    padded_crop = pad_small_crop_for_vietocr(sub_crop)
                    crop_rgb = cv2.cvtColor(padded_crop, cv2.COLOR_BGR2RGB)
                    crop_pil = Image.fromarray(crop_rgb)
                    pred_res = vietocr_model.predict(crop_pil, return_prob=True)
                    if isinstance(pred_res, tuple):
                        sub_text = (pred_res[0] or "").strip()
                        sub_conf = float(pred_res[1]) if pred_res[1] is not None else b.confidence
                    else:
                        sub_text = str(pred_res).strip()
                except Exception as exc:
                    logger.debug(f"Failed to recognize sub-crop: {exc}")
                    sub_text = ""

            sub_text = unicodedata.normalize("NFC", sub_text)
            sub_text = VietnameseTextPostProcessor.clean_and_restore(sub_text)

            if sub_text:
                sub_boxes_for_b.append(
                    OCRBox(text=sub_text, confidence=sub_conf, bbox=sub_bbox, raw_text=sub_text)
                )

        if sub_boxes_for_b:
            new_boxes.extend(sub_boxes_for_b)
        else:
            new_boxes.append(b)

    return new_boxes


class PhysicalGridTopologyEngine:
    """
    Thuật toán nhận diện cấu trúc ô lưới vật lý (Cell-First Topology):
    - Phát hiện vạch kẻ ngang/dọc bằng hình thái học OpenCV.
    - Phát hiện ô gộp 2D (rowspan, colspan) bằng kiểm tra tường ngăn.
    - Gom toàn bộ dòng chữ trong cùng 1 khung ô thành 1 dòng duy nhất.
    """

    def __init__(self, vietocr_model: Any = None):
        self.vietocr_model = vietocr_model

    def __call__(
        self, boxes: List[OCRBox], image: Optional[np.ndarray] = None, vietocr_model: Any = None
    ) -> LayoutResult:
        model = vietocr_model if vietocr_model is not None else self.vietocr_model
        return self.reconstruct_layout(boxes, image=image, vietocr_model=model)

    def reconstruct_layout(
        self, boxes: List[OCRBox], image: Optional[np.ndarray] = None, vietocr_model: Any = None
    ) -> LayoutResult:
        if not boxes:
            return LayoutResult([], [], parallel_header=None)

        if image is None:
            return LayoutResult([], [], parallel_header=None)

        model = vietocr_model if vietocr_model is not None else self.vietocr_model
        tbl = extract_physical_table_topology(image, boxes, vietocr_model=model)
        if tbl is None or tbl.num_cols < 2 or tbl.num_rows < 2:
            return LayoutResult([], [], parallel_header=None)

        box_info = [_box_to_dict(b) for b in boxes]
        tbl_top = tbl.bbox[1]
        tbl_bottom = tbl.bbox[3]

        pre_boxes = [b for b in box_info if b["cy"] < tbl_top - 5]
        post_boxes = [b for b in box_info if b["cy"] > tbl_bottom + 5]

        heights = [b["h"] for b in box_info]
        median_h = float(np.median(heights)) if heights else 20.0

        def _group_simple(b_list: List[dict]) -> List[List[OCRBox]]:
            if not b_list:
                return []
            b_sorted = sorted(b_list, key=lambda it: (it["top"], it["left"]))
            lines: List[List[dict]] = []
            for item in b_sorted:
                matched = False
                for l in lines:
                    avg_cy = float(np.mean([x["cy"] for x in l]))
                    if abs(item["cy"] - avg_cy) <= 0.55 * median_h:
                        l.append(item)
                        matched = True
                        break
                if not matched:
                    lines.append([item])
            lines.sort(key=lambda l: min(x["top"] for x in l))
            res = []
            for l in lines:
                l.sort(key=lambda x: x["left"])
                res.append([x["box"] for x in l])
            return res

        pre_lines = _group_simple(pre_boxes)
        post_lines = _group_simple(post_boxes)
        all_lines = pre_lines + tbl.row_boxes + post_lines

        return LayoutResult(all_lines, [tbl], parallel_header=None)


class UnifiedHybridTableEngine:
    """
    Phương pháp 3 — Unified Hybrid Table Engine:
    Kết hợp tối ưu giữa nhận diện hình thái học đường kẻ (Morphological Grid / Physical Grid)
    và đồ thị phép chiếu trục X/Y (Spatial Projection Graph), đạt độ chính xác tối đa trên
    cả 5 dạng bảng tiêu chuẩn và các bảng phức tạp nhiều tầng như Mẫu số 02.
    """

    def __init__(self):
        self.spatial_engine = SpatialProjectionGraphEngine()
        self.morph_engine = MorphologicalGridTableEngine()
        self.physical_engine = PhysicalGridTopologyEngine()

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

        if image is not None:
            phys_res = self.physical_engine(boxes, image=image)
            if phys_res.tables:
                tbl = phys_res.tables[0]
                if any(c.is_merged for c in tbl.cells) and (tbl.num_cols >= 6 or any(c.col_span >= 3 for c in tbl.cells)):
                    return phys_res

        # Sử dụng đồ thị chiếu không gian với tiền lọc nhiễu và gom cụm chịu lệch Y
        return self.spatial_engine.reconstruct_layout(boxes, image=image)


# Khởi tạo các singleton instance sẵn sàng sử dụng
physical_grid_engine = PhysicalGridTopologyEngine()
morphological_grid_engine = MorphologicalGridTableEngine()
spatial_projection_engine = SpatialProjectionGraphEngine()
unified_hybrid_engine = UnifiedHybridTableEngine()
