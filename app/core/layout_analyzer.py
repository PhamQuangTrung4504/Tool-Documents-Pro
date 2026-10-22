#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
app.core.layout_analyzer
========================
Module phân tích bố cục, gom dòng đọc tự nhiên, phát hiện tiêu đề 2 khối song song
và tái dựng cấu trúc bảng biểu hoàn chỉnh từ danh sách bounding box OCR.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

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
from app.core.text_postprocessor import ContextSymbolRecovery, format_progress_table_row

__all__ = [
    "format_progress_table_row",
    "_box_info_list",
    "_group_single_column_lines",
    "_group_boxes_sliding_window",
    "join_ocr_line_boxes",
    "detect_parallel_administrative_header",
    "detect_and_reconstruct_table",
    "reconstruct_document_layout",
    "group_boxes_into_lines",
    "sort_boxes_reading_order",
]

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


def _table_candidate_quality(table: OCRTable, source_boxes: List[OCRBox]) -> Tuple[float, float, float]:
    """Return bounded (quality, box coverage, rectangularity) for one candidate.

    Invalid dimensions or overlapping/out-of-bounds logical cells are rejected.
    Empty candidates score zero; a valid candidate is never zeroed merely
    because it has no topology errors.
    """
    if (
        table.num_rows <= 0 or table.num_cols <= 0 or not table.grid
        or not source_boxes or table.num_rows < 0 or table.num_cols < 0
    ):
        return 0.0, 0.0, 0.0

    assigned_ids = {id(box) for row in table.row_boxes for box in row}
    assigned_ids.update(id(box) for cell in table.cells for box in cell.boxes)
    source_ids = {id(box) for box in source_boxes}
    coverage = min(1.0, max(0.0, len(assigned_ids & source_ids) / len(source_ids)))
    nonempty_rows = sum(any(str(value).strip() for value in row) for row in table.grid)
    row_coverage = min(1.0, max(0.0, nonempty_rows / table.num_rows))
    nonempty_cols = sum(
        any(col < len(row) and str(row[col]).strip() for row in table.grid)
        for col in range(table.num_cols)
    )
    col_coverage = min(1.0, max(0.0, nonempty_cols / table.num_cols))
    rectangularity = min(
        1.0,
        max(0.0, sum(len(row) == table.num_cols for row in table.grid) / len(table.grid)),
    )
    # `topology_issues()` reports structural invalidity. A clean candidate
    # retains its score; overlap and out-of-bounds spans make it unusable.
    if table.topology_issues():
        return 0.0, coverage, rectangularity
    score = 0.60 * coverage + 0.15 * row_coverage + 0.15 * col_coverage + 0.10 * rectangularity
    return min(1.0, max(0.0, score)), coverage, rectangularity


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
                can_merge_x = gap_x < gutter_thresh

                if (abs(b_curr["cy"] - avg_cy) < y_thresh or overlap_y >= 0.40 * min(b_curr["h"], avg_h)) and can_merge_x:
                    single_line_boxes.append(b_curr)
                    k += 1
                else:
                    break

            single_line_boxes.sort(key=lambda it: (round(it["left"], 2), it["text"]))

            result_lines.append([it["box"] for it in single_line_boxes])
            i = k

    return result_lines


def join_ocr_line_boxes(line_boxes: List[OCRBox]) -> str:
    """Join recognized boxes with whitespace; never invent punctuation/fill text."""
    return " ".join(box.text.strip() for box in line_boxes if box.text.strip())

def detect_parallel_administrative_header(
    box_info: List[dict],
    page_min_x: float,
    page_max_x: float,
    page_w: float,
    gutter_thresh: float,
    median_h: float,
) -> Tuple[List[List[OCRBox]], List[dict]]:
    """Separate a two-block upper page region using geometry only.

    The detector preserves every input box. A centered text band can delimit
    the region; otherwise only a compact upper two-column cluster is eligible.
    Text vocabulary is not used to decide which side or region a box belongs to.
    """
    if len(box_info) < 4:
        return HeaderDetectionResult([], box_info, parallel_header=None)

    page_mid_x = (page_min_x + page_max_x) / 2.0
    text_top = min(item["top"] for item in box_info)
    text_bottom = max(item["bottom"] for item in box_info)
    text_height = max(1.0, text_bottom - text_top)
    centered_candidates = [
        item for item in box_info
        if abs((item["left"] + item["right"]) / 2.0 - page_mid_x) <= 0.18 * page_w
        and 0.20 <= (item["top"] - text_top) / text_height <= 0.90
    ]
    centered_title = min(centered_candidates, key=lambda item: item["top"]) if centered_candidates else None
    header_limit = (
        centered_title["top"]
        if centered_title
        else text_top + max(0.38 * text_height, 1.5 * median_h)
    )
    header_boxes = [
        item for item in box_info
        if item["bottom"] <= header_limit - (0.20 * median_h if centered_title else 0.0)
    ]
    if len(header_boxes) < 4:
        return HeaderDetectionResult([], box_info, parallel_header=None)

    best_split: Optional[float] = None
    best_gap = 0.0
    for ratio in np.linspace(0.30, 0.70, 41):
        split_x = page_min_x + ratio * page_w
        left = [item for item in header_boxes if item["right"] <= split_x]
        right = [item for item in header_boxes if item["left"] >= split_x]
        crosses = [item for item in header_boxes if item["left"] < split_x < item["right"]]
        if len(left) < 2 or len(right) < 2 or crosses:
            continue
        gap = min(item["left"] for item in right) - max(item["right"] for item in left)
        required_gap = max(1.25 * median_h, 0.05 * page_w)
        if gap >= required_gap and gap > best_gap:
            best_split, best_gap = split_x, gap

    if best_split is None:
        return HeaderDetectionResult([], box_info, parallel_header=None)

    left_boxes = [item for item in header_boxes if item["right"] <= best_split]
    right_boxes = [item for item in header_boxes if item["left"] >= best_split]
    lines_left = _group_single_column_lines(left_boxes, median_h, gutter_thresh)
    lines_right = _group_single_column_lines(right_boxes, median_h, gutter_thresh)
    parallel_header = ParallelHeader(
        left_lines=[" ".join(box.text.strip() for box in line if box.text.strip()) for line in lines_left],
        right_lines=[" ".join(box.text.strip() for box in line if box.text.strip()) for line in lines_right],
        left_boxes=[box for line in lines_left for box in line],
        right_boxes=[box for line in lines_right for box in line],
    )
    owned_ids = {id(box) for box in parallel_header.left_boxes + parallel_header.right_boxes}
    remaining = [item for item in box_info if id(item["box"]) not in owned_ids]
    return HeaderDetectionResult(lines_left + lines_right, remaining, parallel_header=parallel_header)

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
        bbox=bbox,
        num_rows=len(table_grid),
        num_cols=num_cols,
        row_boxes=reconstructed_rows_boxes,
        cells=std_cells,
    )
    return ocr_table, pre_table_boxes, reconstructed_rows_boxes, post_table_boxes


def reconstruct_document_layout(
    boxes: List[OCRBox],
    image: Optional[np.ndarray] = None,
    vietocr_model: Any = None,
    diagnostics: Optional[dict] = None,
) -> LayoutResult:
    """
    Phân tích toàn diện bố cục tài liệu (Document Layout Reconstruction):
    1. Phát hiện tiêu đề hành chính 2 khối (Administrative Parallel Header) ở đầu trang.
    2. Tái dựng bảng biểu bằng UnifiedHybridTableEngine (Spatial Projection Graph + Morphological Grid).
    3. Phân tích đa cột / một cột (Sliding window) cho các vùng văn bản còn lại.
    Trả về: LayoutResult(grouped_lines, tables, parallel_header)
    """
    if not boxes:
        if diagnostics is not None:
            diagnostics["selected_engine"] = "none"
            diagnostics["reason"] = "no_ocr_boxes"
        return LayoutResult([], [], parallel_header=None)

    box_info = _box_info_list(boxes)
    page_min_x = min(item["left"] for item in box_info)
    page_max_x = max(item["right"] for item in box_info)
    page_w = max(1, page_max_x - page_min_x)
    heights = [item["h"] for item in box_info]
    median_h = float(np.median(heights)) if heights else 20.0
    gutter_thresh = max(3.0 * median_h, 0.08 * page_w)

    # 1. Nếu có ảnh, ưu tiên kiểm tra Bảng Lưới Vật Lý (Physical Grid Topology) TRƯỚC
    # để tránh detect_parallel_administrative_header nhận nhầm các ô bảng thành Header 2 cột
    if image is not None:
        from app.core.table_engines import extract_physical_table_topology
        target_ocr_boxes = [b["box"] for b in box_info]
        physical_diagnostics = {} if diagnostics is not None else None
        phys_tbl = extract_physical_table_topology(
            image,
            target_ocr_boxes,
            vietocr_model=vietocr_model,
            diagnostics=physical_diagnostics,
        )
        if diagnostics is not None:
            diagnostics["physical_grid_candidate"] = physical_diagnostics
        if phys_tbl is not None:
            # A structurally valid ruled grid is stronger evidence than
            # content-derived table heuristics, even for small/simple tables.
            if not phys_tbl.topology_issues():
                if diagnostics is not None:
                    diagnostics["selected_engine"] = "physical_grid_topology"
                physical_box_ids = {
                    id(box)
                    for cell in phys_tbl.cells
                    for box in cell.boxes
                }
                physical_box_ids.update(
                    id(box)
                    for row in phys_tbl.row_boxes
                    for box in row
                )
                tbl_top = phys_tbl.bbox[1]
                tbl_bottom = phys_tbl.bbox[3]
                pre_tbl_boxes = [
                    b for b in box_info
                    if id(b["box"]) not in physical_box_ids and b["cy"] < tbl_top - 4
                ]
                post_tbl_boxes = [
                    b for b in box_info
                    if id(b["box"]) not in physical_box_ids and b["cy"] > tbl_bottom + 4
                ]
                pre_ids = {id(item["box"]) for item in pre_tbl_boxes}
                post_ids = {id(item["box"]) for item in post_tbl_boxes}
                orphan_table_boxes = [
                    item for item in box_info
                    if id(item["box"]) not in physical_box_ids | pre_ids | post_ids
                ]

                # Chạy detect_parallel_administrative_header CHỈ TRÊN pre_tbl_boxes
                if pre_tbl_boxes:
                    pre_min_x = min(b["left"] for b in pre_tbl_boxes)
                    pre_max_x = max(b["right"] for b in pre_tbl_boxes)
                    pre_w = max(1, pre_max_x - pre_min_x)
                    hdr_res = detect_parallel_administrative_header(
                        pre_tbl_boxes, pre_min_x, pre_max_x, pre_w, gutter_thresh, median_h
                    )
                    header_lines = hdr_res[0]
                    rem_pre_boxes = hdr_res[1]
                    parallel_hdr = getattr(hdr_res, "parallel_header", None)
                    pre_body_lines = _group_boxes_sliding_window(
                        rem_pre_boxes, pre_min_x, pre_w, gutter_thresh, median_h
                    )
                    pre_lines = header_lines + pre_body_lines
                else:
                    pre_lines = []
                    parallel_hdr = None

                if post_tbl_boxes:
                    post_min_x = min(b["left"] for b in post_tbl_boxes)
                    post_max_x = max(b["right"] for b in post_tbl_boxes)
                    post_w = max(1, post_max_x - post_min_x)
                    post_lines = _group_boxes_sliding_window(
                        post_tbl_boxes, post_min_x, post_w, gutter_thresh, median_h
                    )
                else:
                    post_lines = []

                orphan_lines = _group_boxes_sliding_window(
                    orphan_table_boxes,
                    min((item["left"] for item in orphan_table_boxes), default=page_min_x),
                    max(1.0, max((item["right"] for item in orphan_table_boxes), default=page_max_x)
                        - min((item["left"] for item in orphan_table_boxes), default=page_min_x)),
                    gutter_thresh,
                    median_h,
                )
                table_region_lines = list(phys_tbl.row_boxes) + orphan_lines
                table_region_lines.sort(
                    key=lambda line: min(min(point[1] for point in box.bbox) for box in line) if line else float("inf")
                )
                all_lines = pre_lines + table_region_lines + post_lines
                return LayoutResult(all_lines, [phys_tbl], parallel_header=parallel_hdr)

    # 2. Kiểm tra Header 2 khối hành chính đầu trang TRƯỚC (Administrative Parallel Header) cho tài liệu thông thường
    # Tuyệt đối không để Table Reconstruction nuốt mất Header 2 cột hoặc ghép chung với bảng
    hdr_res = detect_parallel_administrative_header(
        box_info, page_min_x, page_max_x, page_w, gutter_thresh, median_h
    )
    header_lines = hdr_res[0]
    rem_boxes = hdr_res[1]
    parallel_hdr = getattr(hdr_res, "parallel_header", None)

    # Table candidate generation sees the complete page first. Header regions
    # are classified only after the spatial candidates exist, so text that
    # happens to be in an upper two-column band cannot strip a table header.
    target_boxes_for_table = box_info

    # 2. Ưu tiên số 1 (Primary Engine): Chạy logic detect_and_reconstruct_table nguyên bản
    # (sử dụng phân tích cụm cột neo STT và header 5 cột Tiêu chí đánh giá / Điểm tối đa / Điểm tự chấm).
    tbl_v1, pre_tbl_v1, tbl_rows_v1, post_tbl_v1 = detect_and_reconstruct_table(
        target_boxes_for_table, page_min_x, page_max_x, page_w, median_h
    )

    # For borderless or partially ruled tables, run spatial projection first.
    # The legacy anchor engine remains a fallback if the general engine finds no table.
    from app.core.table_engines import unified_hybrid_engine

    target_ocr_boxes = [b["box"] for b in target_boxes_for_table]
    hybrid_res = unified_hybrid_engine(target_ocr_boxes, image=image)
    if len(hybrid_res[1]) > 0:
        hybrid_table = hybrid_res[1][0]
        if tbl_v1 is not None:
            legacy_score = _table_candidate_quality(tbl_v1, target_ocr_boxes)
            hybrid_score = _table_candidate_quality(hybrid_table, target_ocr_boxes)
            if legacy_score[0] >= hybrid_score[0]:
                if diagnostics is not None:
                    diagnostics["selected_engine"] = "legacy_anchor_quality"
                    diagnostics["table_candidate_quality"] = {
                        "legacy": {"score": legacy_score[0], "box_coverage": legacy_score[1], "rectangularity": legacy_score[2]},
                        "spatial": {"score": hybrid_score[0], "box_coverage": hybrid_score[1], "rectangularity": hybrid_score[2]},
                    }
                pre_lines = header_lines + _group_boxes_sliding_window(
                    [item for item in pre_tbl_v1 if id(item["box"]) not in {id(box) for line in header_lines for box in line}],
                    page_min_x, page_w, gutter_thresh, median_h
                )
                post_lines = _group_boxes_sliding_window(
                    post_tbl_v1, page_min_x, page_w, gutter_thresh, median_h
                )
                return LayoutResult(pre_lines + tbl_rows_v1 + post_lines, [tbl_v1], parallel_header=parallel_hdr)
        if diagnostics is not None:
            diagnostics["selected_engine"] = "unified_hybrid"
            if tbl_v1 is not None:
                legacy_score = _table_candidate_quality(tbl_v1, target_ocr_boxes)
                hybrid_score = _table_candidate_quality(hybrid_table, target_ocr_boxes)
                diagnostics["table_candidate_quality"] = {
                    "legacy": {"score": legacy_score[0], "box_coverage": legacy_score[1], "rectangularity": legacy_score[2]},
                    "spatial": {"score": hybrid_score[0], "box_coverage": hybrid_score[1], "rectangularity": hybrid_score[2]},
                }
        tbl = hybrid_table
        header_ids = {id(box) for line in header_lines for box in line}
        remaining_hybrid_lines = [
            [box for box in line if id(box) not in header_ids]
            for line in hybrid_res[0]
        ]
        all_lines = header_lines + [line for line in remaining_hybrid_lines if line]
        return LayoutResult(all_lines, [tbl], parallel_header=parallel_hdr)

    # 4. Fallback cuối: Nếu UnifiedHybrid không tìm thấy nhưng detect_and_reconstruct_table có bảng
    if tbl_v1 is not None:
        if diagnostics is not None:
            diagnostics["selected_engine"] = "legacy_anchor_fallback"
        pre_lines = header_lines + _group_boxes_sliding_window(
            pre_tbl_v1, page_min_x, page_w, gutter_thresh, median_h
        )
        post_lines = _group_boxes_sliding_window(
            post_tbl_v1, page_min_x, page_w, gutter_thresh, median_h
        )
        all_lines = pre_lines + tbl_rows_v1 + post_lines
        return LayoutResult(all_lines, [tbl_v1], parallel_header=parallel_hdr)

    # 3. Nếu không có bảng: Gom nhóm các dòng còn lại
    if header_lines:
        if diagnostics is not None:
            diagnostics["selected_engine"] = "line_grouping_with_parallel_header"
        rem_lines = _group_boxes_sliding_window(
            rem_boxes, page_min_x, page_w, gutter_thresh, median_h
        )
        return LayoutResult(header_lines + rem_lines, [], parallel_header=parallel_hdr)

    # 4. Văn bản thông thường (Sliding Window Multi-column / Single-column)
    standard_lines = _group_boxes_sliding_window(
        box_info, page_min_x, page_w, gutter_thresh, median_h
    )
    if diagnostics is not None:
        diagnostics["selected_engine"] = "line_grouping"
    return LayoutResult(standard_lines, [], parallel_header=None)


def group_boxes_into_lines(
    boxes: List[OCRBox],
    image: Optional[np.ndarray] = None,
) -> List[List[OCRBox]]:
    """
    Gom nhóm các khối chữ theo từng dòng đọc (Line-level grouping) và sắp xếp
    thứ tự đọc tài liệu dựa trên phân tích bố cục hình học, bảng biểu và tiêu đề hành chính.
    """
    lines, _ = reconstruct_document_layout(boxes, image=image)
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



