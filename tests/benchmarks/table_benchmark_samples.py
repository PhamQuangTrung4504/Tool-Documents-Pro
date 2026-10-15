#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests.benchmarks.table_benchmark_samples
========================================
Bộ sinh dữ liệu benchmark 5 dạng bảng thực tế cho Giai đoạn C2:
1. sample_01_bordered_admin: Bảng có đường kẻ đầy đủ + ô trống.
2. sample_02_borderless_financial: Bảng không đường kẻ (Hóa đơn / Báo giá).
3. sample_03_merged_cells_header: Bảng có tiêu đề 2 tầng / Ô gộp ngang & dọc.
4. sample_04_dense_7cols: Bảng nhiều cột dày đặc (7 cột hẹp).
5. sample_05_noisy_broken_grid: Bảng bị nhiễu viền, đứt nét đường kẻ & lệch tọa độ Y.
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import cv2
import numpy as np

from app.core.models import OCRBox
from app.core.table_evaluator import TableGroundTruth

__all__ = [
    "build_sample_01_bordered_admin",
    "build_sample_02_borderless_financial",
    "build_sample_03_merged_cells_header",
    "build_sample_04_dense_7cols",
    "build_sample_05_noisy_broken_grid",
    "get_all_benchmark_samples",
]


def _create_box(text: str, x1: int, y1: int, x2: int, y2: int, conf: float = 0.98) -> OCRBox:
    """Tạo OCRBox với 4 điểm góc [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]."""
    bbox = [
        [int(x1), int(y1)],
        [int(x2), int(y1)],
        [int(x2), int(y2)],
        [int(x1), int(y2)],
    ]
    return OCRBox(text=text, confidence=conf, bbox=bbox, raw_text=text)


def build_sample_01_bordered_admin() -> Tuple[np.ndarray, List[OCRBox], TableGroundTruth]:
    """
    Sample 1: Bảng hành chính có đầy đủ đường kẻ khung & có ô trống (Admin Score Table).
    Cấu trúc: 5 cột (STT, Tiêu chí đánh giá, Điểm tối đa, Điểm tự chấm, Ghi chú).
    """
    img = np.ones((700, 800, 3), dtype=np.uint8) * 255
    boxes: List[OCRBox] = []

    # Non-table lines
    t1 = "BẢNG ĐIỂM ĐÁNH GIÁ CHẤT LƯỢNG SINH HOẠT CHI BỘ"
    boxes.append(_create_box(t1, 120, 30, 680, 55))
    t2 = "THÁNG 10 NĂM 2026"
    boxes.append(_create_box(t2, 320, 65, 480, 85))

    # Grid columns x-ranges:
    # Col 0 (STT): 50 - 110
    # Col 1 (Tiêu chí): 110 - 450
    # Col 2 (Điểm tối đa): 450 - 550
    # Col 3 (Điểm tự chấm): 550 - 660
    # Col 4 (Ghi chú): 660 - 750
    row_y = [
        (110, 140),  # Header: row 0
        (150, 180),  # Row 1
        (190, 220),  # Row 2 (sub-item 1)
        (230, 260),  # Row 3 (sub-item 2)
        (270, 300),  # Row 4
        (310, 340),  # Row 5 (sub-item 3)
    ]

    expected_grid = [
        ["STT", "Tiêu chí đánh giá", "Điểm tối đa", "Điểm tự chấm", "Ghi chú"],
        ["1", "Chấp hành thời gian, thời lượng sinh hoạt", "05", "05", ""],
        ["", "Chi bộ dưới 30 đảng viên ít nhất 90 phút", "05", "", ""],
        ["", "Thời gian sinh hoạt chi bộ chỉ bằng ½ so với quy định", "02", "", ""],
        ["2", "Tỷ lệ đảng viên dự sinh hoạt chi bộ", "05", "05", ""],
        ["", "Từ 90% trở lên", "04", "04", ""],
    ]

    # Vẽ đường kẻ bảng trên ảnh
    cv2.rectangle(img, (50, 100), (750, 350), (0, 0, 0), 2)
    for y1, _ in row_y[1:]:
        cv2.line(img, (50, y1 - 5), (750, y1 - 5), (100, 100, 100), 1)
    for x in [110, 450, 550, 660]:
        cv2.line(img, (x, 100), (x, 350), (100, 100, 100), 1)

    # Thêm boxes tương ứng
    boxes.append(_create_box("STT", 60, 115, 100, 135))
    boxes.append(_create_box("Tiêu chí đánh giá", 130, 115, 380, 135))
    boxes.append(_create_box("Điểm tối đa", 460, 115, 540, 135))
    boxes.append(_create_box("Điểm tự chấm", 560, 115, 650, 135))
    boxes.append(_create_box("Ghi chú", 680, 115, 740, 135))

    # Row 1
    boxes.append(_create_box("1", 75, 155, 85, 175))
    boxes.append(_create_box("Chấp hành thời gian, thời lượng sinh hoạt", 120, 155, 440, 175))
    boxes.append(_create_box("05", 490, 155, 515, 175))
    boxes.append(_create_box("05", 595, 155, 620, 175))

    # Row 2
    boxes.append(_create_box("Chi bộ dưới 30 đảng viên ít nhất 90 phút", 130, 195, 435, 215))
    boxes.append(_create_box("05", 490, 195, 515, 215))

    # Row 3
    boxes.append(_create_box("Thời gian sinh hoạt chi bộ chỉ bằng ½ so với quy định", 130, 235, 445, 255))
    boxes.append(_create_box("02", 490, 235, 515, 255))

    # Row 4
    boxes.append(_create_box("2", 75, 275, 85, 295))
    boxes.append(_create_box("Tỷ lệ đảng viên dự sinh hoạt chi bộ", 120, 275, 390, 295))
    boxes.append(_create_box("05", 490, 275, 515, 295))
    boxes.append(_create_box("05", 595, 275, 620, 295))

    # Row 5
    boxes.append(_create_box("Từ 90% trở lên", 130, 315, 250, 335))
    boxes.append(_create_box("04", 490, 315, 515, 335))
    boxes.append(_create_box("04", 595, 315, 620, 335))

    gt = TableGroundTruth(
        sample_id="sample_01_bordered_admin",
        category="bordered",
        expected_grid=expected_grid,
        expected_non_table_lines=[t1, t2],
        metadata={"total_rows": 6, "total_cols": 5, "description": "Admin score table with empty sub-criteria cells"},
    )

    return img, boxes, gt


def build_sample_02_borderless_financial() -> Tuple[np.ndarray, List[OCRBox], TableGroundTruth]:
    """
    Sample 2: Bảng không đường kẻ (Borderless Financial Table - Báo giá / Hóa đơn).
    Phân tách hoàn toàn bằng khoảng cách ngang (Whitespace Columns).
    """
    img = np.ones((650, 800, 3), dtype=np.uint8) * 255
    boxes: List[OCRBox] = []

    # Non-table lines
    t1 = "HÓA ĐƠN BÁN HÀNG"
    boxes.append(_create_box(t1, 300, 30, 500, 55))
    t2 = "Số: 00124/HĐ-2026"
    boxes.append(_create_box(t2, 330, 65, 470, 85))

    expected_grid = [
        ["Tên hàng hóa", "Số lượng", "Đơn giá", "Thành tiền"],
        ["Giấy in A4 Double A", "10", "85.000", "850.000"],
        ["Bút bi Thiên Long", "50", "5.000", "250.000"],
        ["Bìa còng 7cm", "15", "32.000", "480.000"],
        ["Tổng cộng", "", "", "1.580.000"],
    ]

    # Col 0 (Tên hàng): 60 - 320
    # Col 1 (Số lượng): 350 - 430
    # Col 2 (Đơn giá): 470 - 560
    # Col 3 (Thành tiền): 600 - 720

    # Row 0: Header
    boxes.append(_create_box("Tên hàng hóa", 60, 120, 200, 140))
    boxes.append(_create_box("Số lượng", 360, 120, 420, 140))
    boxes.append(_create_box("Đơn giá", 480, 120, 545, 140))
    boxes.append(_create_box("Thành tiền", 620, 120, 710, 140))

    # Row 1
    boxes.append(_create_box("Giấy in A4 Double A", 60, 160, 235, 180))
    boxes.append(_create_box("10", 380, 160, 400, 180))
    boxes.append(_create_box("85.000", 485, 160, 540, 180))
    boxes.append(_create_box("850.000", 630, 160, 695, 180))

    # Row 2
    boxes.append(_create_box("Bút bi Thiên Long", 60, 200, 210, 220))
    boxes.append(_create_box("50", 380, 200, 400, 220))
    boxes.append(_create_box("5.000", 495, 200, 535, 220))
    boxes.append(_create_box("250.000", 630, 200, 695, 220))

    # Row 3
    boxes.append(_create_box("Bìa còng 7cm", 60, 240, 180, 260))
    boxes.append(_create_box("15", 380, 240, 400, 260))
    boxes.append(_create_box("32.000", 485, 240, 540, 260))
    boxes.append(_create_box("480.000", 630, 240, 695, 260))

    # Row 4
    boxes.append(_create_box("Tổng cộng", 60, 280, 150, 300))
    boxes.append(_create_box("1.580.000", 620, 280, 705, 300))

    gt = TableGroundTruth(
        sample_id="sample_02_borderless_financial",
        category="borderless",
        expected_grid=expected_grid,
        expected_non_table_lines=[t1, t2],
        metadata={"total_rows": 5, "total_cols": 4, "description": "Borderless financial invoice without grid lines"},
    )

    return img, boxes, gt


def build_sample_03_merged_cells_header() -> Tuple[np.ndarray, List[OCRBox], TableGroundTruth]:
    """
    Sample 3: Bảng có tiêu đề 2 tầng / ô gộp ngang và ô nội dung nhiều dòng (Merged Cells).
    Cột 2 gộp trên 2 cột con: Kết quả thực hiện -> Kế hoạch | Thực tế.
    """
    img = np.ones((700, 850, 3), dtype=np.uint8) * 255
    boxes: List[OCRBox] = []

    # Non-table line
    t1 = "BÁO CÁO TIẾN ĐỘ THỰC HIỆN DỰ ÁN"
    boxes.append(_create_box(t1, 230, 30, 620, 55))

    # Grid columns:
    # Col 0 (STT): 50 - 110
    # Col 1 (Hạng mục): 110 - 430
    # Col 2 (Kế hoạch): 430 - 550
    # Col 3 (Thực tế): 550 - 670
    # Col 4 (Đánh giá): 670 - 780
    expected_grid = [
        ["STT", "Hạng mục công việc", "Kế hoạch", "Thực tế", "Đánh giá"],
        ["1", "Triển khai hệ thống OCR", "100%", "95%", "Đạt"],
        ["2", "Đào tạo nhân sự và chuyển giao công nghệ", "50", "48", "Tốt"],
    ]

    # Vẽ đường kẻ
    cv2.rectangle(img, (50, 90), (780, 320), (0, 0, 0), 2)
    cv2.line(img, (50, 140), (780, 140), (0, 0, 0), 1)
    cv2.line(img, (50, 180), (780, 180), (0, 0, 0), 1)
    cv2.line(img, (50, 240), (780, 240), (0, 0, 0), 1)
    for x in [110, 430, 550, 670]:
        cv2.line(img, (x, 90), (x, 320), (0, 0, 0), 1)

    # Boxes:
    boxes.append(_create_box("STT", 65, 125, 95, 145))
    boxes.append(_create_box("Hạng mục công việc", 180, 125, 360, 145))
    boxes.append(_create_box("Kế hoạch", 460, 125, 520, 145))
    boxes.append(_create_box("Thực tế", 585, 125, 640, 145))
    boxes.append(_create_box("Đánh giá", 700, 125, 760, 145))

    # Row 1
    boxes.append(_create_box("1", 75, 195, 85, 215))
    boxes.append(_create_box("Triển khai hệ thống OCR", 125, 195, 340, 215))
    boxes.append(_create_box("100%", 470, 195, 510, 215))
    boxes.append(_create_box("95%", 595, 195, 625, 215))
    boxes.append(_create_box("Đạt", 715, 195, 745, 215))

    # Row 2 (Ô hạng mục tách thành 2 box nhiều dòng)
    boxes.append(_create_box("2", 75, 265, 85, 285))
    boxes.append(_create_box("Đào tạo nhân sự và", 125, 250, 280, 270))
    boxes.append(_create_box("chuyển giao công nghệ", 125, 275, 320, 295))
    boxes.append(_create_box("50", 480, 265, 500, 285))
    boxes.append(_create_box("48", 600, 265, 620, 285))
    boxes.append(_create_box("Tốt", 715, 265, 745, 285))

    gt = TableGroundTruth(
        sample_id="sample_03_merged_cells_header",
        category="merged_cells",
        expected_grid=expected_grid,
        expected_non_table_lines=[t1],
        metadata={"total_rows": 3, "total_cols": 5, "description": "Table with multi-level header and multi-line cell"},
    )

    return img, boxes, gt


def build_sample_04_dense_7cols() -> Tuple[np.ndarray, List[OCRBox], TableGroundTruth]:
    """
    Sample 4: Bảng 7 cột số liệu dày đặc (Dense Multi-column Table).
    Khoảng cách ngang giữa các cột hẹp (10-15px).
    """
    img = np.ones((600, 950, 3), dtype=np.uint8) * 255
    boxes: List[OCRBox] = []

    t1 = "DANH SÁCH CÁN BỘ CÔNG NHÂN VIÊN"
    boxes.append(_create_box(t1, 280, 25, 660, 48))

    expected_grid = [
        ["TT", "Họ và tên", "Năm sinh", "Chức vụ", "Hệ số lương", "Phụ cấp", "Ký nhận"],
        ["1", "Nguyễn Văn An", "1985", "Trưởng phòng", "4.40", "0.5", "An"],
        ["2", "Trần Thị Bình", "1990", "Phó phòng", "3.66", "0.3", "Bình"],
        ["3", "Lê Văn Cường", "1992", "Chuyên viên", "3.00", "0.2", "Cường"],
    ]

    # 7 cols x-offsets:
    # Col 0 (TT): 40 - 80
    # Col 1 (Họ tên): 90 - 270
    # Col 2 (Năm sinh): 280 - 370
    # Col 3 (Chức vụ): 380 - 520
    # Col 4 (Hệ số lương): 530 - 660
    # Col 5 (Phụ cấp): 670 - 770
    # Col 6 (Ký nhận): 780 - 890

    # Vẽ khung
    cv2.rectangle(img, (40, 80), (890, 260), (0, 0, 0), 2)
    for y in [120, 165, 210]:
        cv2.line(img, (40, y), (890, y), (120, 120, 120), 1)
    for x in [80, 270, 370, 520, 660, 770]:
        cv2.line(img, (x, 80), (x, 260), (120, 120, 120), 1)

    # Row 0: Header
    boxes.append(_create_box("TT", 50, 92, 70, 112))
    boxes.append(_create_box("Họ và tên", 140, 92, 220, 112))
    boxes.append(_create_box("Năm sinh", 295, 92, 355, 112))
    boxes.append(_create_box("Chức vụ", 420, 92, 480, 112))
    boxes.append(_create_box("Hệ số lương", 555, 92, 635, 112))
    boxes.append(_create_box("Phụ cấp", 695, 92, 745, 112))
    boxes.append(_create_box("Ký nhận", 805, 92, 865, 112))

    # Row 1
    boxes.append(_create_box("1", 55, 135, 65, 155))
    boxes.append(_create_box("Nguyễn Văn An", 110, 135, 230, 155))
    boxes.append(_create_box("1985", 305, 135, 345, 155))
    boxes.append(_create_box("Trưởng phòng", 400, 135, 500, 155))
    boxes.append(_create_box("4.40", 580, 135, 615, 155))
    boxes.append(_create_box("0.5", 710, 135, 730, 155))
    boxes.append(_create_box("An", 825, 135, 845, 155))

    # Row 2
    boxes.append(_create_box("2", 55, 180, 65, 200))
    boxes.append(_create_box("Trần Thị Bình", 110, 180, 225, 200))
    boxes.append(_create_box("1990", 305, 180, 345, 200))
    boxes.append(_create_box("Phó phòng", 415, 180, 485, 200))
    boxes.append(_create_box("3.66", 580, 180, 615, 200))
    boxes.append(_create_box("0.3", 710, 180, 730, 200))
    boxes.append(_create_box("Bình", 815, 180, 850, 200))

    # Row 3
    boxes.append(_create_box("3", 55, 225, 65, 245))
    boxes.append(_create_box("Lê Văn Cường", 110, 225, 225, 245))
    boxes.append(_create_box("1992", 305, 225, 345, 245))
    boxes.append(_create_box("Chuyên viên", 405, 225, 495, 245))
    boxes.append(_create_box("3.00", 580, 225, 615, 245))
    boxes.append(_create_box("0.2", 710, 225, 730, 245))
    boxes.append(_create_box("Cường", 810, 225, 855, 245))

    gt = TableGroundTruth(
        sample_id="sample_04_dense_7cols",
        category="multi_column_dense",
        expected_grid=expected_grid,
        expected_non_table_lines=[t1],
        metadata={"total_rows": 4, "total_cols": 7, "description": "7-column dense table with narrow gutters"},
    )

    return img, boxes, gt


def build_sample_05_noisy_broken_grid() -> Tuple[np.ndarray, List[OCRBox], TableGroundTruth]:
    """
    Sample 5: Bảng bị đứt nét kẻ, lệch tọa độ Y ngẫu nhiên (±5px) và có box nhiễu viền.
    """
    img = np.ones((500, 700, 3), dtype=np.uint8) * 255
    boxes: List[OCRBox] = []

    # Box nhiễu viền ở sát lề (phải được lọc hoặc không làm hỏng cấu trúc bảng)
    boxes.append(_create_box("|", 10, 120, 18, 145, conf=0.45))
    boxes.append(_create_box("[", 680, 210, 688, 235, conf=0.40))

    expected_grid = [
        ["Mã số", "Họ tên", "Ghi chú"],
        ["NV01", "Phạm Quang Trung", "Chính thức"],
        ["NV02", "Đỗ Minh Tuấn", "Thử việc"],
    ]

    # Vẽ đường kẻ đứt nét
    for x in range(50, 650, 20):
        cv2.line(img, (x, 80), (x + 10, 80), (0, 0, 0), 1)
        cv2.line(img, (x, 140), (x + 10, 140), (0, 0, 0), 1)
        cv2.line(img, (x, 200), (x + 10, 200), (0, 0, 0), 1)
        cv2.line(img, (x, 260), (x + 10, 260), (0, 0, 0), 1)

    # Row 0: Lệch Y nhẹ (93, 88, 95)
    boxes.append(_create_box("Mã số", 70, 93, 130, 115))
    boxes.append(_create_box("Họ tên", 240, 88, 320, 110))
    boxes.append(_create_box("Ghi chú", 480, 95, 550, 117))

    # Row 1: Lệch Y nhẹ (152, 147, 155)
    boxes.append(_create_box("NV01", 75, 152, 125, 172))
    boxes.append(_create_box("Phạm Quang Trung", 200, 147, 360, 169))
    boxes.append(_create_box("Chính thức", 470, 155, 560, 177))

    # Row 2: Lệch Y nhẹ (214, 208, 216)
    boxes.append(_create_box("NV02", 75, 214, 125, 234))
    boxes.append(_create_box("Đỗ Minh Tuấn", 220, 208, 340, 229))
    boxes.append(_create_box("Thử việc", 480, 216, 550, 238))

    gt = TableGroundTruth(
        sample_id="sample_05_noisy_broken_grid",
        category="noisy_scan",
        expected_grid=expected_grid,
        expected_non_table_lines=[],
        metadata={"total_rows": 3, "total_cols": 3, "description": "Noisy broken grid with vertical jitter and border noise"},
    )

    return img, boxes, gt


def get_all_benchmark_samples() -> Dict[str, Tuple[np.ndarray, List[OCRBox], TableGroundTruth]]:
    """Trả về toàn bộ 5 mẫu benchmark chuẩn của Giai đoạn C2."""
    return {
        "sample_01_bordered_admin": build_sample_01_bordered_admin(),
        "sample_02_borderless_financial": build_sample_02_borderless_financial(),
        "sample_03_merged_cells_header": build_sample_03_merged_cells_header(),
        "sample_04_dense_7cols": build_sample_04_dense_7cols(),
        "sample_05_noisy_broken_grid": build_sample_05_noisy_broken_grid(),
    }
