#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
app.core.vision_preprocess
==========================
Module tiền xử lý thị giác máy tính và hình thái học ảnh tài liệu OpenCV/NumPy:
- Cắt và xoay nắn dòng chữ polygon 4 điểm (Perspective Transform).
- Tẩy xóa đường viền bảng và nhiễu biên lát cắt.
- Đệm padding nền trắng an toàn cho VietOCR.
- Phát hiện dải hoa văn trang trí và đường kẻ lặp ở tầng thị giác.
- Tách dòng có chấm lửng thành các sub-crops.
- Phát hiện góc nghiêng và nắn thẳng ảnh tài liệu (Deskew).
- Lọc nhiễu quét do cắt ngang viền ảnh (Border noise filtering).
- Đánh giá chất lượng ảnh văn bản (Quality-Aware Assessment).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from app.core.models import OCRBox

__all__ = [
    "get_rotate_crop_image",
    "clean_crop_borders",
    "pad_small_crop_for_vietocr",
    "adaptive_safe_crop_padding",
    "is_decorative_pattern_crop",
    "split_dotted_line_crop",
    "estimate_and_correct_skew",
    "filter_border_noise_boxes",
    "assess_image_quality",
]

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


def adaptive_safe_crop_padding(
    crop_bgr: np.ndarray,
    safe_margin: int = 12,
) -> np.ndarray:
    """
    Đệm Biên Thích Ứng Bảo Toàn Ký Tự và Dấu Thanh (Adaptive Safe Crop Padding):
    - Kiểm tra xem các cột pixel ở sát mép phải hoặc mép trên của crop_bgr có còn nét mực
      của ký tự hay không (không phải đường kẻ bảng dọc/ngang xuyên suốt).
    - Tự động đệm thêm khoảng trắng an toàn (safe_margin = 12px) ở mép phải và/hoặc mép trên
      để VietOCR không bao giờ bị cắt cụt ký tự cuối câu (như 'năm 202', 'không c')
      hoặc mất dấu thanh trên chữ hoa ('ỦY BAN', 'ĐỘC LẬP').
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return crop_bgr

    h, w = crop_bgr.shape[:2]
    if h < 4 or w < 4:
        return crop_bgr

    # 1. Nhị phân hóa xác định điểm mực (Foreground)
    if len(crop_bgr.shape) == 2:
        gray = crop_bgr
    else:
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    if float(np.mean(gray)) > 127.0:
        _, bin_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, bin_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    pad_top = 0
    pad_right = 0

    # 2. Kiểm tra mép trên (top 2px)
    # Nếu có nét mực chạm mép trên (> 5% diện tích slice) nhưng không phải đường kẻ bảng ngang (> 55% w)
    top_slice = bin_inv[0:min(2, h), :]
    top_ink_ratio = float(np.count_nonzero(top_slice)) / max(1.0, float(top_slice.size))
    top_is_horizontal_line = any(
        int(np.sum(bin_inv[y, :] > 0)) >= 0.55 * w for y in range(min(2, h))
    )
    if 0.05 < top_ink_ratio < 0.70 and not top_is_horizontal_line:
        pad_top = safe_margin

    # 3. Kiểm tra mép phải (right 2px - 3px)
    # Nếu có nét mực chạm mép phải (> 10% ink) nhưng không phải đường kẻ dọc (dải đen >= 75% h)
    check_w = min(3, w)
    right_slice = bin_inv[:, max(0, w - check_w):w]
    right_ink_ratio = float(np.count_nonzero(right_slice)) / max(1.0, float(right_slice.size))

    # Tính run dọc lớn nhất ở mép phải
    max_right_run = 0
    for c in range(right_slice.shape[1]):
        col = right_slice[:, c]
        run = 0
        for val in col:
            if val > 0:
                run += 1
                if run > max_right_run:
                    max_right_run = run
            else:
                run = 0

    right_is_vertical_line = max_right_run >= 0.75 * h
    if 0.10 < right_ink_ratio < 0.70 and not right_is_vertical_line:
        pad_right = safe_margin

    if pad_top > 0 or pad_right > 0:
        bg_color = np.median(crop_bgr.reshape(-1, 3 if len(crop_bgr.shape) == 3 else 1), axis=0).astype(np.uint8)
        if len(crop_bgr.shape) == 3:
            bg_val = [int(bg_color[0]), int(bg_color[1]), int(bg_color[2])]
        else:
            bg_val = int(bg_color[0])
        padded = cv2.copyMakeBorder(
            crop_bgr,
            top=pad_top,
            bottom=0,
            left=0,
            right=pad_right,
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



