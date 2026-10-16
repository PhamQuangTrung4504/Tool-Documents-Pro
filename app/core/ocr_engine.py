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

Phase C1 Modular Architecture:
- app.core.models: OCRBox, OCRTable, OCRResult, ParallelHeader,...
- app.core.vision_preprocess: Xử lý ảnh OpenCV, nắn nghiêng, cắt polygon,...
- app.core.text_postprocessor: VietnameseTextPostProcessor, final_document_cleanup,...
- app.core.layout_analyzer: Phân tích bố cục, bảng biểu, tiêu đề 2 khối song song,...
- app.core.docx_exporter: Xuất tài liệu Word .docx chuẩn hành chính.
Tất cả các class và hàm công khai đều được re-export tại đây để bảo đảm tính tương thích ngược 100%.
"""

from __future__ import annotations

import os
import re
import threading
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from PIL import Image
from loguru import logger

# Re-exports từ các module con (Phase C1 Decoupling)
from app.core.models import (
    HeaderDetectionResult,
    LayoutResult,
    OCRBox,
    OCRResult,
    OCRTable,
    ParallelHeader,
    SymbolDecision,
)
from app.core.vision_preprocess import (
    adaptive_safe_crop_padding,
    assess_image_quality,
    clean_crop_borders,
    estimate_and_correct_skew,
    filter_border_noise_boxes,
    get_rotate_crop_image,
    is_decorative_pattern_crop,
    pad_small_crop_for_vietocr,
    split_dotted_line_crop,
)
from app.core.text_postprocessor import (
    ContextSymbolRecovery,
    NOISE_SINGLE_CHARS,
    RE_DECORATIVE_LINE_CHARS,
    RE_REPEATED_WORD,
    ROMAN_HEADING_KEYWORDS,
    VietnameseTextPostProcessor,
    clean_orphaned_brackets,
    clean_unmatched_closing_brackets,
    final_document_cleanup,
    format_progress_table_row,
    is_decorative_or_hallucinated,
    is_hallucinated_or_noise_line,
    preserve_casing,
    recover_emails_and_urls,
)
from app.core.layout_analyzer import (
    _box_info_list,
    _group_boxes_sliding_window,
    _group_single_column_lines,
    detect_and_reconstruct_table,
    detect_parallel_administrative_header,
    group_boxes_into_lines,
    join_ocr_line_boxes,
    reconstruct_document_layout,
    sort_boxes_reading_order,
)
from app.core.docx_exporter import export_to_docx
from app.core.table_evaluator import (
    CellErrorDiagnostic,
    TableEvalTrace,
    TableGroundTruth,
    evaluate_table_sample,
    generate_benchmark_summary,
)
from app.core.table_engines import (
    MorphologicalGridTableEngine,
    PhysicalGridTopologyEngine,
    SpatialProjectionGraphEngine,
    UnifiedHybridTableEngine,
    extract_physical_table_topology,
    morphological_grid_engine,
    physical_grid_engine,
    spatial_projection_engine,
    split_table_crossing_boxes,
    unified_hybrid_engine,
)
from app.core.recognition_diagnostics import (
    CropDiagnosticMetrics,
    ErrorRootCause,
    RecognitionErrorAttributor,
    SampleAttributionResult,
    VIETOCR_STANDARD_VOCAB,
    VIETOCR_STANDARD_VOCAB_SET,
    analyze_crop_quality,
    compute_cer,
    compute_wer,
    find_out_of_vocab_chars,
)
from app.core.ngram_language_model import (
    ContextualCandidateScore,
    VietnameseNgramLanguageModel,
    VisualOCRConfusionModel,
    get_default_ngram_model,
    visual_ocr_distance,
)

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
        import torch  # noqa: F401 # type: ignore
    except Exception:
        pass

    try:
        import sys
        import paddlex  # type: ignore[import-untyped,import-not-found]
        import paddlex.repo_manager as rm  # type: ignore[import-untyped,import-not-found]
        import paddlex.repo_manager.core as rmc  # type: ignore[import-untyped,import-not-found]

        orig_init = rmc.initialize

        def safe_init(*args, **kwargs):
            try:
                if rmc._GlobalContext.is_initialized():
                    return
                return orig_init(*args, **kwargs)
            except RuntimeError as err_init:
                if "already been initialized" in str(err_init):
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

DEFAULT_LANG: str = "vi"


def get_display_languages() -> List[str]:
    """Trả về danh sách tên ngôn ngữ hiển thị trên UI."""
    return list(LANGUAGE_MAP.keys())


def get_lang_code(display_name: str) -> str:
    """Chuyển tên hiển thị → mã ngôn ngữ PaddleOCR."""
    return LANGUAGE_MAP.get(display_name, "vi")




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
                import paddlex.inference.models.runners.paddle_static.runner as _pdx_runner  # type: ignore[import-untyped,import-not-found]
                _orig_pdx_create = _pdx_runner.PaddleStaticRunner._create

                def _safe_pdx_create(self, *args, **kwargs):
                    if getattr(self, "_config", {}).get("device_type") == "cpu":
                        self._config["run_mode"] = "paddle"
                    return _orig_pdx_create(self, *args, **kwargs)

                setattr(_pdx_runner.PaddleStaticRunner, "_create", _safe_pdx_create)
            except Exception:
                pass

        # 3. Khởi tạo PaddleOCR
        v4_supported = {"ch", "en", "chinese_cht"}
        default_version = "PP-OCRv4" if lang in v4_supported else "PP-OCRv3"
        target_version = ocr_version or default_version

        logger.info(f"Loading PaddleOCR detector for lang='{lang}' (Singleton) …")
        from paddleocr import PaddleOCR  # type: ignore[import-untyped,import-not-found]
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
            import torch  # type: ignore
            from vietocr.tool.config import Cfg  # type: ignore[import-untyped,import-not-found]
            from vietocr.tool.predictor import Predictor  # type: ignore[import-untyped,import-not-found]

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
        Tiền xử lý ảnh tài liệu thích ứng theo chất lượng ảnh (Quality-Aware Adaptive Preprocessing):
        - Trường hợp 1 — Ảnh sạch / Scan rõ / Chụp màn hình (bg_level >= 195 và contrast_span >= 90):
          * TẮT HOÀN TOÀN CLAHE và FastNlMeansDenoising! Giữ nguyên độ mượt nét chữ gốc.
          * Nếu cạnh nhỏ < 850px, phóng đại nhẹ bằng nội suy cv2.INTER_LANCZOS4 (1.25x - 1.5x).
        - Trường hợp 2 — Ảnh chụp tối / Bóng đổ / Mờ (bg_level < 195 hoặc contrast_span < 90):
          * Áp dụng CLAHE nhẹ (clipLimit=1.2, tileGridSize=(8, 8)) trên kênh L.
          * Khử nhiễu nhẹ nếu denoise=True.
        """
        if image is None or image.size == 0:
            raise ValueError("Ảnh đầu vào không hợp lệ.")

        if len(image.shape) == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        img_h, img_w = image.shape[:2]

        quality = assess_image_quality(image)
        is_clean = quality["is_clean_document"]

        if is_clean:
            # Trường hợp 1: Ảnh sạch / Scan rõ / Chụp màn hình
            # TẮT HOÀN TOÀN CLAHE và Denoise
            shortest_side = min(img_h, img_w)
            if shortest_side < 850:
                scale_up = min(1.5, max(1.25, 850.0 / max(1, shortest_side)))
                up_w = int(round(img_w * scale_up))
                up_h = int(round(img_h * scale_up))
                image = cv2.resize(image, (up_w, up_h), interpolation=cv2.INTER_LANCZOS4)
                img_h, img_w = up_h, up_w
            elif shortest_side < min_side:
                scale_up = min(1.5, float(min_side) / max(1, shortest_side))
                if scale_up > 1.05:
                    up_w = int(round(img_w * scale_up))
                    up_h = int(round(img_h * scale_up))
                    image = cv2.resize(image, (up_w, up_h), interpolation=cv2.INTER_LANCZOS4)
                    img_h, img_w = up_h, up_w

            if max(img_h, img_w) > max_side:
                scale_down = max_side / float(max(img_h, img_w))
                new_w = int(img_w * scale_down)
                new_h = int(img_h * scale_down)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

            return image

        # Trường hợp 2: Ảnh chụp tối / Bóng đổ / Mờ
        shortest_side = min(img_h, img_w)
        if shortest_side < min_side:
            scale_up = min(2.0, float(min_side) / max(1, shortest_side))
            if scale_up > 1.15:
                up_w = int(img_w * scale_up)
                up_h = int(img_h * scale_up)
                image = cv2.resize(image, (up_w, up_h), interpolation=cv2.INTER_CUBIC)
                img_h, img_w = up_h, up_w

        if max(img_h, img_w) > max_side:
            scale_down = max_side / float(max(img_h, img_w))
            new_w = int(img_w * scale_down)
            new_h = int(img_h * scale_down)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # CLAHE nhẹ (clipLimit=1.2) trên kênh L
        if enhance_contrast:
            try:
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
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

        if len(image.shape) == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

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
        assert detector is not None, "PaddleOCR detector not initialized"
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
            crop_bgr: Optional[np.ndarray] = None

            if use_vietocr:
                # Cắt (crop) vùng chữ từ ảnh tiền xử lý với padding viền an toàn
                crop_bgr = get_rotate_crop_image(processed_image, np.array(poly))
                if crop_bgr is not None and crop_bgr.size > 0:
                    # 1. Tẩy đường kẻ bảng ở sát biên lát cắt (tránh nhận nhầm [, |, F, ])
                    crop_bgr = clean_crop_borders(crop_bgr)

                    # 2. Kiểm tra dải hoa văn trang trí / đường kẻ lặp ở tầng thị giác
                    if is_decorative_pattern_crop(crop_bgr):
                        text = ""
                        conf = 0.0
                    else:
                        try:
                            # 3. Tách dòng rất dài có chấm lửng thành các sub-crops tránh attention drift nuốt chữ
                            sub_crops = split_dotted_line_crop(crop_bgr)
                            if len(sub_crops) > 1 and self._vietocr is not None:
                                sub_texts = []
                                sub_confs = []
                                for sc in sub_crops:
                                    sc_padded = pad_small_crop_for_vietocr(sc)
                                    sc_rgb = cv2.cvtColor(sc_padded, cv2.COLOR_BGR2RGB)
                                    sc_pil = Image.fromarray(sc_rgb)
                                    p_res = self._vietocr.predict(sc_pil, return_prob=True)
                                    if isinstance(p_res, tuple):
                                        t_sub = (p_res[0] or "").strip()
                                        c_sub = float(p_res[1]) if p_res[1] is not None else 0.95
                                    else:
                                        t_sub = str(p_res).strip()
                                        c_sub = 0.95
                                    if t_sub:
                                        sub_texts.append(t_sub)
                                        sub_confs.append(c_sub)
                                if sub_texts:
                                    text = " ...... ".join(sub_texts)
                                    conf = float(np.mean(sub_confs)) if sub_confs else 0.95
                                else:
                                    text = ""
                                    conf = 0.5
                            else:
                                # 4. Đệm khung trắng chuẩn tỷ lệ cho ô hẹp & chữ số (05, 01, 1, 2)
                                crop_bgr_padded = pad_small_crop_for_vietocr(crop_bgr)
                                crop_rgb = cv2.cvtColor(crop_bgr_padded, cv2.COLOR_BGR2RGB)
                                crop_pil = Image.fromarray(crop_rgb)
                                if self._vietocr is not None:
                                    pred_res = self._vietocr.predict(crop_pil, return_prob=True)
                                else:
                                    pred_res = ("", 0.0)
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

            # Bỏ qua dòng trống, dòng ảo giác lặp từ hoặc ký tự rác, hoặc quá thấp độ tin cậy
            if text and not is_decorative_or_hallucinated(text, crop_bgr) and conf >= self.drop_score:
                boxes.append(OCRBox(text=text, confidence=conf, bbox=bbox_int, raw_text=text))

        # 5. Lọc nhiễu quét do cắt ngang viền ảnh (Border noise filtering)
        boxes = filter_border_noise_boxes(boxes, image_width=orig_w, image_height=orig_h)

        # 5.b. Tách các OCRBox bị DBNet gộp ngang xuyên qua vạch kẻ dọc của bảng
        boxes = split_table_crossing_boxes(boxes, image=image, vietocr_model=self._vietocr)

        # 6. Gom nhóm các khối chữ theo dòng và tái dựng bố cục tài liệu (Layout Reconstruction)
        layout_res = reconstruct_document_layout(boxes, image=image)
        grouped_lines = layout_res[0]
        detected_tables = layout_res[1]
        parallel_hdr = getattr(layout_res, "parallel_header", None)
        boxes = [b for line in grouped_lines for b in line]

        # 7. Tổng hợp toàn bộ văn bản (đối với bảng, nối các ô bằng ' | ', đối với văn bản thường nối bằng space)
        # Nhận diện các bảng vật lý có ô gộp phức tạp để render nguyên khối theo đúng thứ tự đọc
        phys_merged_tables = {}
        for tbl in detected_tables:
            cells = getattr(tbl, "cells", [])
            is_phys_merged = any(getattr(c, "is_merged", False) for c in cells) and (
                tbl.num_cols >= 6 or any(getattr(c, "col_span", 1) >= 3 for c in cells)
            )
            if is_phys_merged:
                tbl_box_ids = {id(b) for r_b in tbl.row_boxes for b in r_b} | {
                    id(b) for c in cells for b in c.boxes
                }
                phys_merged_tables[id(tbl)] = (tbl, tbl_box_ids)

        rendered_physical_table_ids: Set[int] = set()

        # Xây dựng bản đồ ánh xạ hàng bảng theo box id cho các bảng thông thường (scorecard, simple grid)
        table_row_map = {}
        for tbl in detected_tables:
            if id(tbl) in phys_merged_tables:
                continue
            for r_idx, r_boxes in enumerate(tbl.row_boxes):
                if r_idx < len(tbl.grid):
                    box_id_set = frozenset(id(b) for b in r_boxes)
                    table_row_map[box_id_set] = (tbl, tbl.grid[r_idx])

        rendered_table_rows = set()
        text_lines = []
        for line in grouped_lines:
            if not line:
                continue

            line_box_ids = frozenset(id(b) for b in line)

            # Kiểm tra xem dòng này có chứa box thuộc bảng vật lý ô gộp phức tạp không
            matched_phys_tbl_id = None
            for tbl_id, (tbl, tbl_box_ids) in phys_merged_tables.items():
                if any(id(b) in tbl_box_ids for b in line):
                    matched_phys_tbl_id = tbl_id
                    break

            if matched_phys_tbl_id is not None:
                if matched_phys_tbl_id not in rendered_physical_table_ids:
                    rendered_physical_table_ids.add(matched_phys_tbl_id)
                    tbl, _ = phys_merged_tables[matched_phys_tbl_id]
                    for row_data in tbl.grid:
                        non_empty_cells = [c.strip() for c in row_data if c.strip()]
                        if not non_empty_cells:
                            continue
                        if not row_data[0].strip():
                            formatted_row = " | ".join(non_empty_cells)
                        else:
                            formatted_row = " | ".join(c.strip() for c in row_data if c.strip())
                        formatted_row = re.sub(r"\b(\d+)\s*\|\s*\|\s*", r"\1 | ", formatted_row)
                        formatted_row = format_progress_table_row(formatted_row)
                        text_lines.append(formatted_row)
                continue
            matched_tbl_row = None
            if line_box_ids in table_row_map:
                candidate_tuple = table_row_map[line_box_ids]
                r_key = (id(candidate_tuple[0]), tuple(candidate_tuple[1]))
                if r_key not in rendered_table_rows:
                    matched_tbl_row = candidate_tuple
                    rendered_table_rows.add(r_key)
            else:
                for r_box_ids, tbl_row_tuple in table_row_map.items():
                    common = line_box_ids & r_box_ids
                    if common and (len(common) / len(line_box_ids) >= 0.7 or line_box_ids.issubset(r_box_ids)):
                        r_key = (id(tbl_row_tuple[0]), tuple(tbl_row_tuple[1]))
                        if r_key not in rendered_table_rows:
                            matched_tbl_row = tbl_row_tuple
                            rendered_table_rows.add(r_key)
                        break

            if matched_tbl_row is not None:
                tbl, row_data = matched_tbl_row
                if not row_data[0].strip():
                    non_empty_cells = [c.strip() for c in row_data[1:] if c.strip()]
                    formatted_row = " | ".join(non_empty_cells)
                else:
                    formatted_row = " | ".join(c.strip() for c in row_data if c.strip())

                formatted_row = re.sub(r"\b(\d+)\s*\|\s*\|\s*", r"\1 | ", formatted_row)
                formatted_row = format_progress_table_row(formatted_row)
                text_lines.append(formatted_row)
            else:
                is_fallback_table_row = False
                line_joined = join_ocr_line_boxes(line)
                for tbl in detected_tables:
                    for row_data in tbl.grid:
                        row_joined = " ".join(c.strip() for c in row_data if c.strip())
                        if row_joined and (row_joined == line_joined or line_joined in row_joined):
                            r_key = (id(tbl), tuple(row_data))
                            if r_key not in rendered_table_rows:
                                is_fallback_table_row = True
                                rendered_table_rows.add(r_key)
                                if not row_data[0].strip():
                                    non_empty_cells = [c.strip() for c in row_data[1:] if c.strip()]
                                    formatted_row = " | ".join(non_empty_cells)
                                else:
                                    formatted_row = " | ".join(c.strip() for c in row_data if c.strip())
                                formatted_row = re.sub(r"\b(\d+)\s*\|\s*\|\s*", r"\1 | ", formatted_row)
                                formatted_row = format_progress_table_row(formatted_row)
                                text_lines.append(formatted_row)
                            else:
                                is_fallback_table_row = True
                            break
                    if is_fallback_table_row:
                        break

                if not is_fallback_table_row:
                    cleaned_line = line_joined.strip()
                    if is_hallucinated_or_noise_line(cleaned_line):
                        continue
                    text_lines.append(line_joined)

        raw_full_text = "\n".join(text_lines)
        raw_full_text = VietnameseTextPostProcessor.clean_and_restore(raw_full_text)

        # 8. Context-Aware Symbol Recovery: khôi phục ký tự, số La Mã, bullet dựa trên ngữ cảnh
        # BẮT BUỘC chạy clean_and_restore trước để tách các đề mục số La Mã bị dính vào cuối dòng trước
        # (ví dụ: '... 14 người. TT. Nội dung cuộc họp' -> tách thành 2 dòng riêng trước khi nhận dạng số La Mã)
        pre_cleaned_lines = raw_full_text.split("\n")
        recovered_lines, audit_records = ContextSymbolRecovery.recover_document_symbols(pre_cleaned_lines)
        full_text = "\n".join(recovered_lines)
        full_text = VietnameseTextPostProcessor.clean_and_restore(full_text)

        # 9. Hậu xử lý bắt buộc toàn văn trước khi trả về kết quả OCR:
        # a. Khôi phục bullet '+' cho các dòng tiêu chí con trong bảng đánh giá chất lượng
        full_text = re.sub(
            r"(?m)^([ \t]*)[-–—*•%~]\s+((?:Chi\s+bộ|Thời\s+gian|Từ\s+\d+|Dưới\s+\d+|Không\s+có).*\|\s*0\d)",
            r"\1+ \2",
            full_text,
            flags=re.IGNORECASE,
        )
        full_text = re.sub(
            r"(?m)^([ \t]*)(?:[-–—*•%~]\s+|\+\s*)?((?:Chi\s+bộ\s+dưới|Thời\s+gian\s+sinh\s+hoạt|Từ\s+\d+%|Dưới\s+\d+%|Không\s+có\s+đảng\s+viên).*)",
            r"\1+ \2",
            full_text,
            flags=re.IGNORECASE,
        )
        # a2. Khôi phục ghi chú chân trang '* Trừ trường hợp...'
        full_text = re.sub(
            r"(?m)^([ \t]*)[-*–—/1?]+\s*(Trừ\s+trường\s+hợp)",
            r"\1* \2",
            full_text,
            flags=re.IGNORECASE,
        )
        # b. Chuẩn hóa tiêu đề cột 'đa' -> 'Điểm tối đa'
        full_text = re.sub(
            r"(\|\s*Tiêu\s+chí\s+đánh\s+giá\s*\|\s*)(?:tối\s+)?đa(\s*\|\s*Điểm\s+tự\s+chấm)",
            r"\1Điểm tối đa\2",
            full_text,
            flags=re.IGNORECASE,
        )
        # c. Khóa chốt hậu xử lý văn bản cuối cùng (Final Document Cleanup)
        full_text = final_document_cleanup(full_text)

        # Cập nhật văn bản trong các OCRBox và OCRTable để đồng bộ với full_text
        for b in boxes:
            b.text = final_document_cleanup(b.text)

        for tbl in detected_tables:
            for r_idx in range(len(tbl.grid)):
                row_cells = tbl.grid[r_idx]
                for c_idx in range(len(row_cells)):
                    cell_val = row_cells[c_idx]
                    cell_val = re.sub(
                        r"^([ \t]*)(?:[-–—*•%~]\s+|\+\s*)?((?:Chi\s+bộ\s+dưới|Thời\s+gian\s+sinh\s+hoạt|Từ\s+\d+%|Dưới\s+\d+%|Không\s+có\s+đảng\s+viên).*)",
                        r"\1+ \2",
                        cell_val,
                        flags=re.IGNORECASE,
                    )
                    row_cells[c_idx] = final_document_cleanup(cell_val)

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




__all__ = [
    # Models
    "OCRBox",
    "OCRTable",
    "ParallelHeader",
    "HeaderDetectionResult",
    "LayoutResult",
    "OCRResult",
    "SymbolDecision",
    # Vision Preprocess
    "get_rotate_crop_image",
    "clean_crop_borders",
    "pad_small_crop_for_vietocr",
    "is_decorative_pattern_crop",
    "split_dotted_line_crop",
    "estimate_and_correct_skew",
    "filter_border_noise_boxes",
    "assess_image_quality",
    # Text Postprocessor
    "NOISE_SINGLE_CHARS",
    "RE_DECORATIVE_LINE_CHARS",
    "RE_REPEATED_WORD",
    "ROMAN_HEADING_KEYWORDS",
    "is_decorative_or_hallucinated",
    "is_hallucinated_or_noise_line",
    "recover_emails_and_urls",
    "format_progress_table_row",
    "ContextSymbolRecovery",
    "preserve_casing",
    "clean_unmatched_closing_brackets",
    "clean_orphaned_brackets",
    "final_document_cleanup",
    "VietnameseTextPostProcessor",
    # Layout Analyzer
    "_box_info_list",
    "_group_boxes_sliding_window",
    "_group_single_column_lines",
    "join_ocr_line_boxes",
    "detect_parallel_administrative_header",
    "detect_and_reconstruct_table",
    "reconstruct_document_layout",
    "group_boxes_into_lines",
    "sort_boxes_reading_order",
    # Docx Exporter
    "export_to_docx",
    # Table Evaluator (Phase C2)
    "TableGroundTruth",
    "CellErrorDiagnostic",
    "TableEvalTrace",
    "evaluate_table_sample",
    "generate_benchmark_summary",
    # Table Recognition Engines (Phase C3)
    "MorphologicalGridTableEngine",
    "SpatialProjectionGraphEngine",
    "UnifiedHybridTableEngine",
    "morphological_grid_engine",
    "spatial_projection_engine",
    "unified_hybrid_engine",
    # OCREngine & Helpers
    "LANGUAGE_MAP",
    "DEFAULT_LANG",
    "apply_paddlex_patches",
    "get_display_languages",
    "get_lang_code",
    "get_global_detector",
    "get_global_vietocr",
    "OCREngine",
]
