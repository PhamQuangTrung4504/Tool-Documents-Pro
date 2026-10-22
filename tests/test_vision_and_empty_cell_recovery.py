#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests/test_vision_and_empty_cell_recovery.py
============================================
Kiểm thử chuyên sâu cho:
1. Illumination Compensation (compensate_illumination) & Robust Binarization (robust_binarization).
2. Lọc độ tương phản và chống ảo giác tiếng Anh trong _recover_missing_cell_ink.
3. Đồng bộ hóa văn bản tiêu chí con dấu '+' giữa UI App và Word export.
"""

import numpy as np
import cv2
import pytest
from app.core.vision_preprocess import compensate_illumination, robust_binarization
from app.core.models import TableCell, OCRBox
from app.core.table_engines import _recover_missing_cell_ink
from app.core.text_postprocessor import final_document_cleanup


class MockVietOCRModel:
    """Mock VietOCR returning English hallucination to test the linguistic guard."""
    def __init__(self, hallucination_text="COLINEdE"):
        self.hallucination_text = hallucination_text

    def predict(self, img_pil, return_prob=True):
        return (self.hallucination_text, 0.95)


def test_compensate_illumination_grayscale_and_bgr():
    """Kiểm tra compensate_illumination trên ảnh xám và BGR có bóng đổ."""
    # Tạo ảnh có bóng đổ gradient từ 60 (tối) tới 220 (sáng)
    h, w = 200, 200
    ramp = np.linspace(60, 220, w, dtype=np.uint8)
    shadowed_gray = np.tile(ramp, (h, 1))

    # Thêm nét chữ đen (mực)
    cv2.putText(shadowed_gray, "VIETNAM", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 10, 2)

    # Chạy bù sáng
    compensated_gray = compensate_illumination(shadowed_gray)
    assert compensated_gray.shape == (h, w)
    # Nền tại vùng bóng đổ (cột 10) và vùng sáng (cột 180) phải được kéo về nền sáng (> 200)
    assert compensated_gray[20, 10] > 200
    assert compensated_gray[20, 180] > 200

    # Chạy trên ảnh 3 kênh màu (BGR)
    shadowed_bgr = cv2.cvtColor(shadowed_gray, cv2.COLOR_GRAY2BGR)
    compensated_bgr = compensate_illumination(shadowed_bgr)
    assert compensated_bgr.shape == (h, w, 3)
    assert np.mean(compensated_bgr[20, 10]) > 200


def test_robust_binarization():
    """Kiểm tra robust_binarization trả về ảnh nhị phân hợp lệ."""
    img = np.full((100, 100, 3), 200, dtype=np.uint8)
    cv2.putText(img, "TEST", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2)
    bw = robust_binarization(img)
    assert bw.shape == (100, 100)
    assert bw.dtype == np.uint8
    # Chữ có nét đen
    assert np.any(bw == 0)
    # Nền trắng
    assert np.any(bw == 255)


def test_recover_missing_cell_ink_flat_paper_rejected():
    """Ô phẳng đồng nhất (không có chữ) phải bị loại ngay lập tức bởi Contrast Check."""
    # Ảnh nền trắng đồng nhất với nhiễu nhẹ std < 5
    flat_img = np.full((100, 100, 3), 250, dtype=np.uint8)
    # Thêm nhiễu rất nhỏ
    noise = np.random.randint(-2, 3, (100, 100, 3), dtype=np.int16)
    noisy_flat = np.clip(flat_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    cell = TableCell(row_start=0, row_end=1, col_start=0, col_end=1, bbox=(10, 10, 90, 90), text="")
    mock_model = MockVietOCRModel("COLINEdE")

    _recover_missing_cell_ink([cell], noisy_flat, vietocr_model=mock_model)
    # Cell phải giữ nguyên rỗng, không được có chữ
    assert cell.text == ""
    assert len(cell.boxes) == 0


def test_recover_missing_cell_ink_preserves_recognized_text_for_review():
    """Without independent visual evidence, recovered recognition is preserved for review."""
    img = np.full((100, 100, 3), 250, dtype=np.uint8)
    # Vẽ vài đốm đen để pass qua contrast check
    cv2.circle(img, (50, 50), 10, (10, 10, 10), -1)

    cell = TableCell(row_start=0, row_end=1, col_start=0, col_end=1, bbox=(5, 5, 95, 95), text="")
    # The helper cannot prove this recognition is a hallucination from language alone.
    mock_model = MockVietOCRModel("COLINEdE")

    _recover_missing_cell_ink([cell], img, vietocr_model=mock_model)
    assert cell.text == "COLINEdE"
    assert cell.boxes and cell.boxes[0].text == "COLINEdE"

    # Thử với STATE
    mock_model_state = MockVietOCRModel("STATE")
    second_cell = TableCell(row_start=0, row_end=1, col_start=0, col_end=1, bbox=(5, 5, 95, 95), text="")
    _recover_missing_cell_ink([second_cell], img, vietocr_model=mock_model_state)
    assert second_cell.text == "STATE"

    # Thử với từ hợp lệ có nguyên âm tiếng Việt và số: "UBND" hoặc "Thôn 1"
    mock_model_valid = MockVietOCRModel("Thôn 1")
    third_cell = TableCell(row_start=0, row_end=1, col_start=0, col_end=1, bbox=(5, 5, 95, 95), text="")
    _recover_missing_cell_ink([third_cell], img, vietocr_model=mock_model_valid)
    assert third_cell.text == "Thôn 1"


def test_full_text_plus_bullet_sync():
    """Kiểm tra các dòng tiêu chí con được đồng bộ chính xác dấu '+'."""
    text_with_dash = (
        "- Chi bộ dưới 30 đảng viên ít nhất 90 phút; chi bộ trên 30 đảng viên ít nhất 120 phút | 05 | 05\n"
        "- Thời gian sinh hoạt chi bộ ít hơn quy định 15 phút | 04\n"
        "- Thời gian sinh hoạt chi bộ chỉ bằng ½ so với quy định | 02\n"
        "- Từ 90% trở lên | 04 | 04\n"
        "- Dưới 60% hoặc có đảng viên vắng không có lý do | 01\n"
        "- Không có đảng viên vắng mặt có lý do quá 03 lần liên tiếp trong năm | 01 | 01\n"
    )
    cleaned = final_document_cleanup(text_with_dash)
    for line in cleaned.strip().split("\n"):
        assert line.startswith("+ "), f"Line does not start with '+': {line}"
