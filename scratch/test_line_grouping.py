import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class OCRBox:
    text: str
    confidence: float
    bbox: List[List[int]]

def group_boxes_into_lines(boxes: List[OCRBox]) -> List[List[OCRBox]]:
    if not boxes:
        return []

    # Tính toạ độ hình học cho mỗi box
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
            "left": x_left,
            "right": x_right,
            "top": y_top,
            "bottom": y_bottom,
            "h": h,
            "w": w,
            "cy": cy,
        })

    # Ước lượng chiều cao dòng trung bình
    heights = [info["h"] for info in box_info]
    median_h = float(np.median(heights)) if heights else 20.0
    y_thresh = max(6.0, median_h * 0.45)

    # Sắp xếp sơ bộ theo cy tăng dần, rồi left tăng dần
    box_info.sort(key=lambda item: (item["cy"], item["left"]))

    # Gom nhóm theo dòng
    grouped_lines: List[List[dict]] = []
    current_line: List[dict] = [box_info[0]]

    for info in box_info[1:]:
        line_avg_cy = float(np.mean([item["cy"] for item in current_line]))
        line_avg_h = float(np.mean([item["h"] for item in current_line]))
        line_top = min(item["top"] for item in current_line)
        line_bottom = max(item["bottom"] for item in current_line)

        overlap_y = max(0, min(info["bottom"], line_bottom) - max(info["top"], line_top))
        is_same_line = (
            abs(info["cy"] - line_avg_cy) < y_thresh
            or overlap_y >= 0.40 * min(info["h"], line_avg_h)
        )

        if is_same_line:
            current_line.append(info)
        else:
            grouped_lines.append(current_line)
            current_line = [info]

    if current_line:
        grouped_lines.append(current_line)

    # Sắp xếp các dòng từ trên xuống dưới, và trong từng dòng sắp xếp từ trái sang phải
    result_lines: List[List[OCRBox]] = []
    for line in grouped_lines:
        line.sort(key=lambda item: item["left"])

        # Chuẩn hóa biểu tượng bullet có căn cứ (box nhỏ ở đầu dòng)
        if len(line) >= 2:
            first = line[0]
            # Box đầu dòng nhỏ hình vuông/tròn (w<=35, h<=35) và chứa ký tự dạng bullet
            if first["w"] <= 35 and first["h"] <= 35 and first["box"].text.strip() in ["o", "O", "•", "·", "*", "■", "□", "回", "▫", "▪"]:
                first["box"].text = "•"

        result_lines.append([item["box"] for item in line])

    return result_lines

# Test với các box mô phỏng từ ảnh thực tế của người dùng
mock_boxes = [
    OCRBox("o", 0.85, [[70, 75], [85, 75], [85, 90], [70, 90]]),
    OCRBox("Trong file ghi: '... tuyên truyền số điện thoại ...'", 0.98, [[100, 70], [600, 70], [600, 95], [100, 95]]),
    OCRBox("DOCX", 0.92, [[100, 105], [150, 105], [150, 125], [100, 125]]),
    OCRBox("o", 0.85, [[70, 145], [85, 145], [85, 160], [70, 160]]),
    OCRBox("Cần bổ sung đúng số hotline theo tài liệu: (028) 38.211.238", 0.98, [[100, 140], [550, 140], [550, 165], [100, 165]]),
    OCRBox("3.", 0.95, [[70, 190], [90, 190], [90, 210], [70, 210]]),
    OCRBox("Căn lề phần Kính gửi:", 0.98, [[100, 190], [300, 190], [300, 210], [100, 210]]),
    OCRBox("o", 0.85, [[70, 235], [85, 235], [85, 250], [70, 250]]),
    OCRBox("Khối chữ Kính gửi: và các gạch đầu dòng đang bị thụt tab...", 0.97, [[100, 230], [580, 230], [580, 255], [100, 255]]),
    OCRBox("4.", 0.95, [[70, 280], [90, 280], [90, 300], [70, 300]]),
    OCRBox("Ký hiệu người soạn và dòng văn bản gửi kèm:", 0.98, [[100, 280], [450, 280], [450, 300], [100, 300]]),
]

lines = group_boxes_into_lines(mock_boxes)
formatted_text = "\n".join(" ".join(b.text for b in line) for line in lines)
print("=== FORMATTED OCR TEXT ===")
print(formatted_text)
