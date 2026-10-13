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
class OCRResult:
    """Kết quả tổng hợp sau khi chạy OCR trên một ảnh."""

    boxes: List[OCRBox] = field(default_factory=list)
    full_text: str = ""
    language: str = "vi"
    elapsed_seconds: float = 0.0
    image_width: int = 0
    image_height: int = 0

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
    pad_bottom: int = 6,
    pad_left: int = 4,
    pad_right: int = 4,
) -> Optional[np.ndarray]:
    """
    Cắt và xoay thẳng vùng ảnh polygon 4 điểm (Perspective Transform).
    Chuẩn hóa vùng ảnh chữ nằm ngang cho VietOCR với viền đệm (padding) an toàn:
    - Mở rộng mép trên 6px (bảo vệ dấu sắc, hỏi, ngã, mũ: á, ả, ã, â, ê, ô, ư, ơ).
    - Mở rộng mép dưới 6px (bảo vệ trọn vẹn dấu nặng: ạ, ặ, ậ, ẹ, ệ, ị, ọ, ộ, ợ, ụ, ự, ỵ).
    - Mở rộng mép trái/phải 4px (tránh dính mép viền làm đứt chữ cái đầu/cuối).
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


def sort_boxes_reading_order(boxes: List[OCRBox]) -> List[OCRBox]:
    """
    Sắp xếp bounding box theo thứ tự đọc tự nhiên của tài liệu hành chính:
    Từ trên xuống dưới, từ trái sang phải.
    Gom nhóm các khối chữ cùng hàng (chênh lệch toạ độ Y nhỏ) để sắp xếp theo X.
    """
    if not boxes:
        return []

    # Ước lượng chiều cao trung bình của một dòng chữ
    heights = [abs(b.bbox[2][1] - b.bbox[0][1]) for b in boxes if len(b.bbox) >= 4]
    avg_h = float(np.median(heights)) if heights else 20.0
    y_threshold = max(8.0, avg_h * 0.55)

    # Sắp xếp sơ bộ theo toạ độ Y đỉnh trái
    sorted_by_y = sorted(boxes, key=lambda b: (b.bbox[0][1], b.bbox[0][0]))

    lines: List[OCRBox] = []
    current_line: List[OCRBox] = [sorted_by_y[0]]

    for box in sorted_by_y[1:]:
        line_avg_y = float(np.mean([b.bbox[0][1] for b in current_line]))
        if abs(box.bbox[0][1] - line_avg_y) < y_threshold:
            current_line.append(box)
        else:
            # Sắp xếp hàng hiện tại theo X từ trái sang phải
            current_line.sort(key=lambda b: b.bbox[0][0])
            lines.extend(current_line)
            current_line = [box]

    if current_line:
        current_line.sort(key=lambda b: b.bbox[0][0])
        lines.extend(current_line)

    return lines


# ---------------------------------------------------------------------------
# Bộ Chuẩn Hóa Văn Bản Tiếng Việt
# ---------------------------------------------------------------------------


class VietnameseTextPostProcessor:
    """
    Module chuẩn hóa và làm sạch văn bản tiếng Việt sau khi nhận dạng.
    - Chuẩn hóa Unicode dựng sẵn (NFC).
    - Chuẩn hóa viết tắt hành chính (UBND, HĐND, PCT,...).
    - Khôi phục dấu các cụm từ hành chính phổ biến nếu phát hiện sai sót.
    """

    @staticmethod
    def apply_case(original: str, replacement: str) -> str:
        """Bảo toàn định dạng viết hoa của từ gốc."""
        if original.isupper():
            return replacement.upper()
        orig_words = original.split()
        rep_words = replacement.split()
        if len(orig_words) == len(rep_words):
            return " ".join(
                r.upper() if o.isupper() else (r.capitalize() if o and o[0].isupper() else r.lower())
                for o, r in zip(orig_words, rep_words)
            )
        if original and original[0].isupper():
            return replacement[0].upper() + replacement[1:]
        return replacement.lower()

    PHRASE_PATTERNS: List[Tuple[str, str]] = [
        (r"\b(uy|ủy)\s+(ban|ban)\s+(nhan|nhân)\s+(dan|dân)\s+(xa|xã)\b", "Ủy ban nhân dân xã"),
        (r"\b(uy|ủy)\s+(ban|ban)\s+(nhan|nhân)\s+(dan|dân)\b", "Ủy ban nhân dân"),
        (r"\b(hoi|hội)\s+(dong|đồng)\s+(nhan|nhân)\s+(dan|dân)\b", "Hội đồng nhân dân"),
        (r"\b(van|văn)\s+(phong|phòng)\s+(hdnd|hđnd)\s+(va|và)\s+(ubnd|ubnd)\b", "Văn phòng HĐND và UBND"),
        (r"\b(phong|phòng)\s+(van|văn)\s+(hoa|hóa)\s*-\s*(xa|xã)\s+(hoi|hội)\b", "Phòng Văn hóa - Xã hội"),
        (r"\b(bo|bộ)\s+(van|văn)\s+(hoa|hóa),\s*(the|thể)\s+(thao|thao)\s+(va|và)\s+(du|du)\s+(lich|lịch)\b", "Bộ Văn hóa, Thể thao và Du lịch"),
        (r"\b(van|văn)\s+(hoa|hóa),\s*(the|thể)\s+(thao|thao)\s+(va|và)\s+(du|du)\s+(lich|lịch)\b", "Văn hóa, Thể thao và Du lịch"),
        (r"\b(trung|trung)\s+(tam|tâm)\s+(phuc|phục)\s+(vu|vụ)\s+(hanh|hành)\s+(chinh|chính)\s+(cong|công)\b", "Trung tâm Phục vụ hành chính công"),
        (r"\b(thanh|thành)\s+(pho|phố)\s+(ho|hồ)\s+(chi|chí)\s+(minh|minh)\b", "Thành phố Hồ Chí Minh"),
        (r"\b(cong|công)\s+(hoa|hòa)\s+(xa|xã)\s+(hoi|hội)\s+(chu|chủ)\s+(nghia|nghĩa)\s+(viet|việt)\s+(nam|nam)\b", "Cộng hòa xã hội chủ nghĩa Việt Nam"),
        (r"\b(doc|độc)\s+(lap|lập)\s*-\s*(tu|tự)\s+(do|do)\s*-\s*(hanh|hạnh)\s+(phuc|phúc)\b", "Độc lập - Tự do - Hạnh phúc"),
        (r"\b(cong|công)\s+(van|văn)\s+(so|số)\b", "Công văn số"),
        (r"\b(quyet|quyết)\s+(dinh|định)\s+(so|số)\b", "Quyết định số"),
        (r"\b(kinh|kính)\s+(gui|gửi)\b", "Kính gửi"),
        (r"\b(noi|nơi)\s+(nhan|nhận)\b", "Nơi nhận"),
        (r"\b(nhu|như)\s+(tren|trên)\b", "Như trên"),
        (r"\b(chu|chủ)\s+(tich|tịch)\b", "Chủ tịch"),
        (r"\b(pho|phó)\s+(chu|chủ)\s+(tich|tịch)\b", "Phó Chủ tịch"),
        (r"\b(pct|pct)\s+(ubnd|ubnd)\s+(xa|xã)\b", "PCT UBND xã"),
        (r"\b(ubnd|ubnd)\s+(xa|xã)\b", "UBND xã"),
        (r"\b(cap|cấp)\s+(xa|xã)\b", "cấp xã"),
        (r"\b(thu|thủ)\s+(tuc|tục)\s+(hanh|hành)\s+(chinh|chính)\b", "thủ tục hành chính"),
        (r"\b(quan|quản)\s+(ly|lý)\s+(nha|nhà)\s+(nuoc|nước)\b", "quản lý nhà nước"),
        (r"\b(ke|kế)\s+(hoach|hoạch)\s+(trien|triển)\s+(khai|khai)\b", "kế hoạch triển khai"),
        (r"\b(trien|triển)\s+(khai|khai)\s+(thuc|thực)\s+(hien|hiện)\b", "triển khai thực hiện"),
        (r"\b(he|hệ)\s+(thong|thống)\s+(thong|thông)\s+(tin|tin)\b", "hệ thống thông tin"),
        (r"\b(giai|giải)\s+(quyet|quyết)\b", "giải quyết"),
        (r"\b(nghiem|nghiêm)\s+(tuc|túc)\b", "nghiêm túc"),
        (r"\b(khan|khẩn)\s+(truong|trương)\b", "khẩn trương"),
        (r"\b(chu|chủ)\s+(tri|trì)\b", "chủ trì"),
        (r"\b(phoi|phối)\s+(hop|hợp)\b", "phối hợp"),
        (r"\b(chu|chủ)\s+(dong|động)\b", "chủ động"),
        (r"\b(ra|rà)\s+(soat|soát)\b", "rà soát"),
        (r"\b(bao|bảo)\s+(dam|đảm)\b", "bảo đảm"),
        (r"\b(dong|đồng)\s+(bo|bộ)\b", "đồng bộ"),
        (r"\b(dung|đúng)\s+(quy|quy)\s+(dinh|định)\b", "đúng quy định"),
        (r"\b(kip|kịp)\s+(thoi|thời)\b", "kịp thời"),
        (r"\b(bao|báo)\s+(cao|cáo)\b", "báo cáo"),
        (r"\b(y|ý)\s+(kien|kiến)\s+(chi|chỉ)\s+(dao|đạo)\b", "ý kiến chỉ đạo"),
        (r"\b(tham|tham)\s+(muu|mưu)\b", "tham mưu"),
        (r"\b(dieu|điều)\s+(kien|kiện)\s+(can|cần)\s+(thiet|thiết)\b", "điều kiện cần thiết"),
        (r"\b(van|vận)\s+(hanh|hành)\b", "vận hành"),
        (r"\b(vuot|vượt)\s+(tham|thẩm)\s+(quyen|quyền)\b", "vượt thẩm quyền"),
        (r"\b(theo|theo)\s+(doi|dõi)\b", "theo dõi"),
        (r"\b(don|đôn)\s+(doc|đốc)\b", "đôn đốc"),
        (r"\b(yeu|yêu)\s+(cau|cầu)\b", "yêu cầu"),
        (r"\b(co|cơ)\s+(quan|quan),\s*(don|đơn)\s+(vi|vị)\b", "cơ quan, đơn vị"),
        (r"\b(nhung|những)\s+(van|vấn)\s+(de|đề)\b", "những vấn đề"),
        (r"\b(ban|ban)\s+(bi|bí)\s+(thu|thư)\b", "Ban Bí thư"),
        (r"\b(bi|bí)\s+(thu|thư)\b", "Bí thư"),
        (r"\b(tai|tại)\s+(dieu|điều)\b", "tại Điều"),
        (r"\b(tai|tại)\s+(khoan|khoản)\b", "tại Khoản"),
        (r"\b(tai|tại)\s+(diem|điểm)\b", "tại Điểm"),
        (r"\b(tai|tại)\s+(quyet|quyết)\s+(dinh|định)\b", "tại Quyết định"),
        (r"\b(tai|tại)\s+(thong|thông)\s+(bao|báo)\b", "tại Thông báo"),
        (r"\b(tai|tại)\s+(van|văn)\s+(ban|bản)\b", "tại Văn bản"),
        (r"\b(tai|tại)\s+(nghi|nghị)\s+(dinh|định)\b", "tại Nghị định"),
        (r"\b(tai|tại)\s+(ke|kế)\s+(hoach|hoạch)\b", "tại Kế hoạch"),
        (r"\b(tai|tại)\s+(bo|bộ)\s+(phan|phận)\b", "tại Bộ phận"),
        (r"\b(bo|bộ)\s+(phan|phận)\s+(mot|một)\s+(cua|cửa)\b", "Bộ phận Một cửa"),
        (r"\b(mot|một)\s+(cua|cửa)\s+(lien|liên)\s+(thong|thông)\b", "Một cửa liên thông"),
        (r"\bngay\s+(\d{1,2})\s+thang\s+(\d{1,2})\s+nam\s+(\d{4})\b", r"ngày \1 tháng \2 năm \3"),
    ]

    @classmethod
    def clean_and_restore(cls, text: str) -> str:
        if not text:
            return ""

        # 1. Chuẩn hóa Unicode NFC
        text = unicodedata.normalize("NFC", text).strip()

        # 2. Chuẩn hóa các từ viết tắt phổ biến trong văn bản hành chính
        abbrev_map = {
            r"\bubnd\b": "UBND",
            r"\bhdnd\b": "HĐND",
            r"\bpct\b": "PCT",
            r"\bhcc\b": "HCC",
            r"\bvhttdl\b": "BVHTTDL",
            r"\bqd\b": "QĐ",
            r"\btphcm\b": "TP.HCM",
        }
        for pat, rep in abbrev_map.items():
            text = re.sub(pat, rep, text, flags=re.IGNORECASE)

        # 3. Khôi phục cụm từ hành chính
        for pat, rep in cls.PHRASE_PATTERNS:
            def _repl(m, r=rep):
                if r.startswith("ngày"):
                    return m.expand(r)
                return cls.apply_case(m.group(0), r)

            text = re.sub(pat, _repl, text, flags=re.IGNORECASE)

        # 4. Chuẩn hóa khoảng trắng quanh dấu câu
        text = re.sub(r"\s+([,.:;?!])", r"\1", text)
        text = re.sub(r"([,.:;?!])(?=[^\s\d/])", r"\1 ", text)

        # 5. Ký hiệu kết thúc văn bản hành chính ./.
        text = re.sub(r"\.\s*/\s*\.", "./.", text)

        # 6. Chuẩn hóa lần cuối về Unicode NFC dựng sẵn
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

        logger.debug("Running PaddleOCR Text Detection …")
        polygons: List[List[List[float]]] = []
        paddle_texts: List[str] = []
        paddle_scores: List[float] = []

        # 3. Chạy Detection qua PaddleOCR
        detector = self._ocr
        try:
            results = detector.predict(processed_image)
            for res in results:
                dt_polys = res.get("dt_polys", []) if hasattr(res, "get") else getattr(res, "dt_polys", [])
                rec_texts = res.get("rec_texts", []) if hasattr(res, "get") else getattr(res, "rec_texts", [])
                rec_scores = res.get("rec_scores", []) if hasattr(res, "get") else getattr(res, "rec_scores", [])

                for i, poly in enumerate(dt_polys):
                    if hasattr(poly, "tolist"):
                        poly = poly.tolist()
                    if len(poly) >= 4:
                        polygons.append(poly[:4])
                        paddle_texts.append(rec_texts[i] if i < len(rec_texts) else "")
                        paddle_scores.append(float(rec_scores[i]) if i < len(rec_scores) else 0.8)
        except Exception as exc_det:
            logger.warning(f"Detection via predict failed, falling back to .ocr(): {exc_det}")
            try:
                raw_results = detector.ocr(processed_image, cls=self.use_angle_cls)
                if raw_results and raw_results[0]:
                    for line in raw_results[0]:
                        bbox_points = line[0]
                        polygons.append(bbox_points)
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
                boxes.append(OCRBox(text=text, confidence=conf, bbox=bbox_int))

        # 5. Sắp xếp thứ tự đọc tự nhiên (Reading Order)
        boxes = sort_boxes_reading_order(boxes)

        # 6. Tổng hợp toàn bộ văn bản
        full_text = "\n".join(b.text for b in boxes)
        full_text = VietnameseTextPostProcessor.clean_and_restore(full_text)

        elapsed = time.perf_counter() - t0
        result = OCRResult(
            boxes=boxes,
            full_text=full_text,
            language=self.lang,
            elapsed_seconds=elapsed,
            image_width=orig_w,
            image_height=orig_h,
        )

        logger.info(
            f"Hybrid OCR completed: {result.num_lines} lines, "
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
