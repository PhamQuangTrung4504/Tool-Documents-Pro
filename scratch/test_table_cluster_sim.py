import sys, os
sys.path.insert(0, os.path.abspath("."))
import numpy as np
from dataclasses import dataclass

@dataclass
class Box:
    text: str
    left: int
    right: int
    top: int
    bottom: int
    confidence: float = 0.95

    @property
    def h(self):
        return self.bottom - self.top

    @property
    def w(self):
        return self.right - self.left

    @property
    def cy(self):
        return (self.top + self.bottom) / 2.0

def run_reconstruct_debug():
    boxes = [
        # Tiêu đề ngoài bảng
        Box("BẢNG ĐIỂM ĐÁNH GIÁ CHẤT LƯỢNG SINH HOẠT CHI BỘ", 100, 750, 40, 65),
        Box("THÁNG 10 NĂM 2026", 320, 520, 70, 90),
        # Table Header
        Box("STT", 30, 65, 110, 135),
        Box("Tiêu chí đánh giá", 150, 400, 110, 135),
        Box("Điểm tối", 630, 690, 105, 122),
        Box("đa", 645, 675, 125, 140),
        Box("Điểm tự", 710, 770, 105, 122),
        Box("chấm", 720, 760, 125, 140),
        Box("Ghi chú", 790, 840, 110, 135),
        # Row 1 (STT: 1, Tiêu chí: Chấp hành..., Điểm: 05, 05)
        Box("1", 40, 55, 150, 175),
        Box("Chấp hành thời gian, thời lượng sinh hoạt", 80, 550, 150, 175),
        Box("05", 640, 670, 150, 175),
        Box("05", 725, 755, 150, 175),
        # Sub-item row 2
        Box("% Chi bộ dưới 30 đảng viên ít nhất 90 phút...", 80, 600, 185, 210),
        Box("05", 640, 670, 185, 210),
        Box("05", 725, 755, 185, 210),
        # Row 7 (STT: 2, Tiêu chí: Tỷ lệ..., Điểm: 05, 05)
        Box("2", 40, 55, 360, 385),
        Box("Tỷ lệ đảng viên dự sinh hoạt chi bộ", 80, 480, 360, 385),
        Box("05", 640, 670, 360, 385),
        Box("05", 725, 755, 360, 385),
    ]

    box_info = []
    for b in boxes:
        box_info.append({
            "box": b,
            "text": b.text,
            "left": b.left,
            "right": b.right,
            "top": b.top,
            "bottom": b.bottom,
            "w": b.w,
            "h": b.h,
            "cy": b.cy,
        })

    page_min_x = min(item["left"] for item in box_info)
    page_max_x = max(item["right"] for item in box_info)
    page_w = max(1, page_max_x - page_min_x)
    heights = [item["h"] for item in box_info]
    median_h = float(np.median(heights))

    narrow_boxes = [b for b in box_info if b["w"] < 0.25 * page_w]
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

    table_col_anchors = [cl for cl in clusters if len(cl) >= 3]
    all_anchor_boxes = [b for cl in table_col_anchors for b in cl]
    anchor_min_top = min(b["top"] for b in all_anchor_boxes)
    anchor_max_bottom = max(b["bottom"] for b in all_anchor_boxes)

    table_boxes = [
        b for b in box_info
        if b["bottom"] > anchor_min_top - 5 and b["top"] < anchor_max_bottom + 5 and b["cy"] >= anchor_min_top - 5
    ]

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
    print("Dividers:", dividers)

    num_cols = len(dividers) - 1
    # Check từng box thuộc col nào
    for b in table_boxes:
        b_mid = (b["left"] + b["right"]) / 2.0
        c_idx = 0
        for ci in range(num_cols):
            if dividers[ci] <= b_mid < dividers[ci + 1]:
                c_idx = ci
                break
        if b["text"] in ["STT", "1", "2", "Tỷ lệ đảng viên dự sinh hoạt chi bộ", "Chấp hành thời gian, thời lượng sinh hoạt"]:
            print(f"Box {b['text']:<45} mid={b_mid:<6.1f} -> col={c_idx}")

    from app.core.ocr_engine import detect_and_reconstruct_table, OCRBox
    for item in box_info:
        b = item["box"]
        item["box"] = OCRBox(text=b.text, confidence=0.95, bbox=[[b.left, b.top],[b.right, b.top],[b.right, b.bottom],[b.left, b.bottom]])

    tbl, _, _, _ = detect_and_reconstruct_table(box_info, page_min_x, page_max_x, page_w, median_h)
    print("\n--- RECONSTRUCTED TABLE GRID ---")
    if tbl:
        for r_i, row in enumerate(tbl.grid):
            print(f"Row {r_i}: {' | '.join(row)}")
    else:
        print("Table is None")

run_reconstruct_debug()

