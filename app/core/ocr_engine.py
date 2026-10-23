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
import inspect
from threading import Event
import unicodedata
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from PIL import Image
from loguru import logger

# Re-exports từ các module con (Phase C1 Decoupling)
from app.core.models import (
    DocumentBlock,
    HeaderDetectionResult,
    LayoutResult,
    OCRBox,
    OCRResult,
    OCRTable,
    ParallelHeader,
    SymbolDecision,
    format_confidence,
)
from app.core.document_diagnostics import build_reconstruction_diagnostic, infer_visual_document_styles
from app.core.vision_preprocess import (
    adaptive_safe_crop_padding,
    assess_image_quality,
    clean_crop_borders,
    compensate_illumination,
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


def _resolve_table_row_identity(
    line_box_ids: frozenset[int],
    table_row_map: List[Tuple[frozenset[int], int, int, List[str]]],
    rendered_rows: Set[Tuple[int, int]],
) -> Optional[Tuple[int, int, List[str]]]:
    """Resolve a line to one table/logical-row pair using in-memory ownership.

    Row content is intentionally absent from identity, so equal-valued rows
    remain distinct. Ambiguous box ownership is left unassigned.
    """
    if not line_box_ids:
        return None
    candidates = []
    for row_box_ids, table_index, row_index, row_data in table_row_map:
        common = line_box_ids & row_box_ids
        if common:
            ownership = len(common) / len(line_box_ids)
            if ownership >= 0.7 or line_box_ids.issubset(row_box_ids):
                candidates.append((ownership, table_index, row_index, row_data))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    if not candidates or (len(candidates) > 1 and candidates[0][0] == candidates[1][0]):
        return None
    _, table_index, row_index, row_data = candidates[0]
    key = (table_index, row_index)
    if key in rendered_rows:
        return None
    rendered_rows.add(key)
    return table_index, row_index, row_data


def _recognition_evidence(
    crop_id: int,
    crop_bbox: List[List[int]],
    primary_name: str,
    primary_text: str,
    primary_confidence: Optional[float],
    alternate_text: str = "",
    alternate_confidence: Optional[float] = None,
    alternate_name: str = "paddleocr",
    alternate_mapping: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Keep same-crop candidates and flag disagreements without score voting.

    Model confidence scales are not calibrated across recognizers. The primary
    prediction remains the displayed text; alternatives are evidence for review.
    """
    primary = unicodedata.normalize("NFC", primary_text or "")
    alternate = unicodedata.normalize("NFC", alternate_text or "")
    candidates = [{
        "recognizer": primary_name, "text": primary_text or "",
        "confidence": primary_confidence, "crop_id": crop_id, "crop_bbox": crop_bbox,
    }]
    if alternate_text:
        candidates.append({
            "recognizer": alternate_name, "text": alternate_text,
            "confidence": alternate_confidence, "crop_id": crop_id, "crop_bbox": crop_bbox,
            "mapping": alternate_mapping or {"status": "unspecified"},
        })
    disagreement = bool(primary and alternate and primary.strip() != alternate.strip())
    # Không có benchmark đủ rộng để phân xử tự động giữa hai mô hình. Vì vậy
    # metadata chỉ đưa ra tập lựa chọn cần người xem; văn bản xuất ra vẫn là
    # candidate chính. Confidence chỉ được giữ để truy vết, không dùng để xếp hạng.
    if disagreement:
        recommendation = {
            "status": "human_review_required",
            "candidate_indices": list(range(len(candidates))),
            "selected_index": None,
            "reason": "verified_candidates_disagree",
        }
    elif not primary and alternate:
        recommendation = {
            "status": "human_review_required",
            "candidate_indices": [1],
            "selected_index": None,
            "reason": "primary_candidate_missing",
        }
    else:
        recommendation = {
            "status": "primary_retained",
            "candidate_indices": [0],
            "selected_index": 0,
            "reason": "single_candidate" if not alternate else "candidate_texts_agree",
        }
    return {
        "candidates": candidates,
        "arbitration": "disagreement_preserved" if disagreement else "agreement" if alternate else "single_candidate",
        "needs_review": disagreement or primary_confidence is None,
        "confidence_source": primary_name if primary_confidence is not None else None,
        "confidence_comparable_across_models": False,
        "alternate_mapping": alternate_mapping or {"status": "unspecified"},
        "selected_candidate": 0,
        "candidate_recommendation": recommendation,
    }


def _prepare_polygon(points: List[List[float]]) -> Optional[Tuple[np.ndarray, float, Tuple[float, float, float, float]]]:
    polygon = np.asarray(points[:4], dtype=np.float32)
    if polygon.shape != (4, 2) or not np.isfinite(polygon).all():
        return None
    area = float(abs(cv2.contourArea(polygon)))
    if area <= 0:
        return None
    bounds = (float(polygon[:, 0].min()), float(polygon[:, 1].min()),
              float(polygon[:, 0].max()), float(polygon[:, 1].max()))
    return polygon, area, bounds


def _paddle_result_value(result: Any, key: str, default: Any = None) -> Any:
    if hasattr(result, "get"):
        return result.get(key, default)
    return getattr(result, key, default)


def _paddle_result_records(result: Any) -> Dict[str, Any]:
    """Extract PaddleX 3.x detections and recognition records conservatively.

    PaddleX 3.7 appends rec_polys, rec_texts and rec_scores together only for
    recognitions passing its score threshold. Therefore dt_polys cannot be
    indexed against rec_texts. rec_polys is the explicit geometry paired with
    each returned recognition text. If that geometry contract is unavailable
    or its cardinality disagrees, recognition candidates are withheld.
    """
    raw_detections = _paddle_result_value(result, "dt_polys", [])
    if raw_detections is None:
        raw_detections = []
    raw_rec_polygons = _paddle_result_value(result, "rec_polys", None)
    raw_texts = _paddle_result_value(result, "rec_texts", None)
    raw_scores = _paddle_result_value(result, "rec_scores", None)
    if raw_texts is None:
        raw_texts = []
    if raw_scores is None:
        raw_scores = []

    def as_polygon(value: Any) -> Optional[List[List[float]]]:
        if hasattr(value, "tolist"):
            value = value.tolist()
        try:
            polygon = [[float(point[0]), float(point[1])] for point in value[:4]]
        except (TypeError, ValueError, IndexError):
            return None
        return polygon if len(polygon) == 4 else None

    detections = []
    for detection_index, raw_polygon in enumerate(raw_detections):
        polygon = as_polygon(raw_polygon)
        if polygon is not None:
            detections.append({"detection_index": detection_index, "polygon": polygon})

    if raw_rec_polygons is None:
        return {"detections": detections, "recognitions": [],
                "recognition_status": "unknown_missing_rec_polys",
                "raw_counts": {"dt_polys": len(raw_detections), "rec_polys": None,
                               "rec_texts": len(raw_texts), "rec_scores": len(raw_scores)}}

    if hasattr(raw_rec_polygons, "tolist"):
        raw_rec_polygons = raw_rec_polygons.tolist()
    if len(raw_rec_polygons) != len(raw_texts):
        return {"detections": detections, "recognitions": [],
                "recognition_status": "unknown_rec_polygon_text_count_mismatch",
                "raw_counts": {"dt_polys": len(raw_detections), "rec_polys": len(raw_rec_polygons),
                               "rec_texts": len(raw_texts), "rec_scores": len(raw_scores)}}

    recognitions = []
    for recognition_index, (raw_polygon, text) in enumerate(zip(raw_rec_polygons, raw_texts)):
        polygon = as_polygon(raw_polygon)
        if polygon is None:
            continue
        score = None
        if recognition_index < len(raw_scores):
            try:
                score = float(raw_scores[recognition_index])
            except (TypeError, ValueError):
                score = None
        recognitions.append({"recognition_index": recognition_index, "polygon": polygon,
                             "text": str(text), "score": score,
                             "score_status": "available" if score is not None else "unknown"})
    return {"detections": detections, "recognitions": recognitions,
            "recognition_status": "verified_rec_polys_text_alignment",
            "raw_counts": {"dt_polys": len(raw_detections), "rec_polys": len(raw_rec_polygons),
                           "rec_texts": len(raw_texts), "rec_scores": len(raw_scores)}}


def _prepared_polygon_overlap(
    first: Optional[Tuple[np.ndarray, float, Tuple[float, float, float, float]]],
    second: Optional[Tuple[np.ndarray, float, Tuple[float, float, float, float]]],
) -> Tuple[float, float]:
    """Return (intersection over smaller polygon, IoU) in one shared pixel space."""
    if first is None or second is None:
        return 0.0, 0.0
    a, area_a, bounds_a = first
    b, area_b, bounds_b = second
    ax1, ay1, ax2, ay2 = bounds_a
    bx1, by1, bx2, by2 = bounds_b
    if ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1:
        return 0.0, 0.0
    try:
        intersection, _ = cv2.intersectConvexConvex(
            a, b
        )
    except cv2.error:
        return 0.0, 0.0
    intersection = max(0.0, float(intersection))
    return intersection / min(area_a, area_b), intersection / max(area_a + area_b - intersection, 1e-9)


def _map_candidate_polygons_to_regions(
    paddle_polygons: List[List[List[float]]],
    target_polygons: List[List[List[float]]],
    min_containment: float = 0.55,
    min_iou: float = 0.25,
) -> List[Dict[str, Any]]:
    """Map original Paddle detections to final OCR regions by geometry only.

    A match is emitted only for a reciprocal, unique one-to-one overlap. Merged
    regions, duplicate detections and other ambiguous relations remain unmapped.
    All polygons must be expressed in the same coordinate space.
    """
    prepared_targets = [_prepare_polygon(polygon) for polygon in target_polygons]
    prepared_paddles = [_prepare_polygon(polygon) for polygon in paddle_polygons]
    overlaps = [[_prepared_polygon_overlap(target, paddle) for paddle in prepared_paddles]
                for target in prepared_targets]
    eligible = [[j for j, (containment, iou) in enumerate(row)
                 if containment >= min_containment and iou >= min_iou]
                for row in overlaps]
    reverse: Dict[int, List[int]] = {}
    for target_index, candidates in enumerate(eligible):
        for candidate_index in candidates:
            reverse.setdefault(candidate_index, []).append(target_index)

    mapped: List[Dict[str, Any]] = []
    for target_index, candidates in enumerate(eligible):
        if len(candidates) != 1:
            mapped.append({"status": "unmapped" if not candidates else "ambiguous",
                           "reason": "no_geometric_overlap" if not candidates else "multiple_paddle_detections",
                           "candidate_indices": candidates})
            continue
        candidate_index = candidates[0]
        if len(reverse.get(candidate_index, [])) != 1:
            mapped.append({"status": "ambiguous", "reason": "paddle_detection_matches_multiple_regions",
                           "candidate_indices": candidates})
            continue
        containment, iou = overlaps[target_index][candidate_index]
        mapped.append({"status": "matched", "reason": "unique_reciprocal_geometric_overlap",
                       "candidate_indices": candidates, "candidate_index": candidate_index,
                       "containment": round(containment, 4), "iou": round(iou, 4),
                       "coordinate_space": "preprocessed_image"})
    return mapped


def _resolve_region_candidate_source(
    mapping: Dict[str, Any],
    source_detection_indices: List[int],
    detection_records: List[Dict[str, Any]],
    recognition_records: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Optional[int]]:
    """Require a verified source detection for a VietOCR region candidate."""
    if len(source_detection_indices) > 1:
        return ({"status": "ambiguous", "reason": "region_merges_multiple_detection_polygons",
                 "candidate_indices": mapping.get("candidate_indices", [])}, None)
    if len(source_detection_indices) != 1 or mapping.get("status") != "matched":
        return mapping, None
    recognition_index = mapping.get("candidate_index")
    if not isinstance(recognition_index, int) or not 0 <= recognition_index < len(recognition_records):
        return ({"status": "unmapped", "reason": "recognition_record_index_invalid",
                 "candidate_indices": mapping.get("candidate_indices", [])}, None)
    detection_index = source_detection_indices[0]
    if not 0 <= detection_index < len(detection_records):
        return ({"status": "unmapped", "reason": "detection_record_index_invalid",
                 "candidate_indices": mapping.get("candidate_indices", [])}, None)
    recognition = recognition_records[recognition_index]
    detection = detection_records[detection_index]
    source_link = recognition.get("source_detection_mapping", {})
    verified = (
        source_link.get("status") == "matched"
        and recognition.get("source_detection_result_index") == detection.get("result_index")
        and recognition.get("source_detection_index") == detection.get("detection_index")
    )
    if not verified:
        return ({"status": "unmapped", "reason": "recognition_detection_link_not_verified",
                 "candidate_indices": mapping.get("candidate_indices", [])}, None)
    return mapping, recognition_index


def _notify_diagnostic_observer(
    observer: Optional[Callable[[Dict[str, Any], Optional[np.ndarray]], None]],
    event: Dict[str, Any],
    image: Optional[np.ndarray] = None,
) -> None:
    """Send opt-in evidence to an observer without changing recognition flow."""
    if observer is None:
        return
    try:
        observer(event, image)
    except Exception as exc:
        # Diagnostics must not affect OCR results; avoid logging document text.
        logger.warning("OCR diagnostic observer failed (%s)", type(exc).__name__)


def _describe_perspective_crop(points: List[List[float]]) -> Dict[str, Any]:
    """Describe the same perspective transform used by get_rotate_crop_image."""
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) < 4:
        return {"valid": False}
    width = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
    height = int(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])))
    if width <= 4 or height <= 4:
        return {"valid": False, "rectified_size_before_padding": [width, height]}
    destination = np.asarray([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(pts[:4], destination)
    rotated = height * 1.5 >= width and height > 40
    return {
        "valid": True,
        "homography_source_to_rectified": matrix.tolist(),
        "rectified_size_before_rotation_and_padding": [width, height],
        "rotate_90_degrees_ccw_after_warp": rotated,
        "padding_px_after_rotation": {"top": 6, "bottom": 8, "left": 8, "right": 18},
        "interpolation": "INTER_CUBIC",
        "border_mode": "BORDER_REPLICATE",
    }
def _table_source_box_indices(table: OCRTable, box_index_by_id: Dict[int, int]) -> List[int]:
    """Return result-local indexes owned by a table, without duplicate row/cell links."""
    owned_ids = {id(box) for row in table.row_boxes for box in row}
    owned_ids.update(id(box) for cell in table.cells for box in cell.boxes)
    return sorted(box_index_by_id[box_id] for box_id in owned_ids if box_id in box_index_by_id)
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


class OCRCancelledError(RuntimeError):
    """Raised when a worker requests cancellation at a safe pipeline boundary."""

_GLOBAL_DETECTOR: Optional[Any] = None
_GLOBAL_DETECTOR_LOCK = threading.Lock()

_GLOBAL_VIETOCR: Optional[Any] = None
_GLOBAL_VIETOCR_LOCK = threading.Lock()


def _create_paddleocr_detector(
    paddleocr_cls: Any,
    *,
    lang: str,
    use_gpu: bool,
    use_angle_cls: bool,
    det_db_thresh: float,
    det_db_box_thresh: float,
    det_db_unclip_ratio: float,
    drop_score: float,
    rec_batch_num: int,
    ocr_version: str,
) -> Any:
    """Tạo detector một lần với tên tham số mà constructor runtime công bố.

    PaddleOCR 3.x đổi tên các tùy chọn từ giao diện 2.x. Chỉ truyền một bộ
    tham số theo chữ ký thực tế; lỗi cache/model/constructor được giữ nguyên,
    không bị nhầm thành lý do để thử một constructor khác.
    """
    parameters = inspect.signature(paddleocr_cls).parameters
    accepts_extra = any(
        item.kind is inspect.Parameter.VAR_KEYWORD for item in parameters.values()
    )
    values = {
        "lang": lang,
        "ocr_version": ocr_version,
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": use_angle_cls,
        "text_det_thresh": det_db_thresh,
        "text_det_box_thresh": det_db_box_thresh,
        "text_det_unclip_ratio": det_db_unclip_ratio,
        "text_rec_score_thresh": drop_score,
        "text_recognition_batch_size": rec_batch_num,
    }
    aliases = {
        "use_textline_orientation": "use_angle_cls",
        "text_det_thresh": "det_db_thresh",
        "text_det_box_thresh": "det_db_box_thresh",
        "text_det_unclip_ratio": "det_db_unclip_ratio",
        "text_rec_score_thresh": "drop_score",
        "text_recognition_batch_size": "rec_batch_num",
    }
    kwargs: Dict[str, Any] = {}
    for name, value in values.items():
        if name in parameters:
            kwargs[name] = value
        elif name in aliases and aliases[name] in parameters:
            kwargs[aliases[name]] = value

    if "use_gpu" in parameters:
        kwargs["use_gpu"] = use_gpu
    elif "device" in parameters or accepts_extra:
        # PaddleOCR 3.x chuyển cấu hình thiết bị sang common runtime arguments.
        kwargs["device"] = "gpu:0" if use_gpu else "cpu"
    if "show_log" in parameters:
        kwargs["show_log"] = False

    return paddleocr_cls(**kwargs)

def apply_paddlex_patches() -> None:
    """
    Vá triệt để lỗi 'PDX has already been initialized. Reinitialization is not supported.'
    và xung đột DLL giữa OpenMP / PyTorch / PaddlePaddle trên Windows.
    """
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["PADDLE_PDX_EAGER_INIT"] = "False"
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

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

        # Đảm bảo thứ tự DLL tại đúng ranh giới nạp runtime native. Import
        # engine/model thuần túy không khởi tạo Torch hoặc Paddle sớm.
        from app.native_runtime import bootstrap_torch_before_paddle
        bootstrap_torch_before_paddle()

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

        _GLOBAL_DETECTOR = _create_paddleocr_detector(
            PaddleOCR,
            lang=lang,
            use_gpu=use_gpu,
            use_angle_cls=use_angle_cls,
            det_db_thresh=det_db_thresh,
            det_db_box_thresh=det_db_box_thresh,
            det_db_unclip_ratio=det_db_unclip_ratio,
            drop_score=drop_score,
            rec_batch_num=rec_batch_num,
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
        diagnostics: Optional[List[dict]] = None,
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
            if diagnostics is not None:
                diagnostics.append({"stage": "channel_normalization", "transformation": "BGRA_to_BGR"})
        elif len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            if diagnostics is not None:
                diagnostics.append({"stage": "channel_normalization", "transformation": "GRAY_to_BGR"})

        img_h, img_w = image.shape[:2]

        quality = assess_image_quality(image)
        is_clean = quality["is_clean_document"]
        if diagnostics is not None:
            diagnostics.append({
                "stage": "quality_assessment",
                "metrics": quality.copy(),
                "branch": "clean_document" if is_clean else "low_quality_document",
            })

        if is_clean:
            # Trường hợp 1: Ảnh sạch / Scan rõ / Chụp màn hình
            # TẮT HOÀN TOÀN CLAHE và Denoise
            shortest_side = min(img_h, img_w)
            if shortest_side < 850:
                scale_up = min(1.5, max(1.25, 850.0 / max(1, shortest_side)))
                up_w = int(round(img_w * scale_up))
                up_h = int(round(img_h * scale_up))
                image = cv2.resize(image, (up_w, up_h), interpolation=cv2.INTER_LANCZOS4)
                if diagnostics is not None:
                    diagnostics.append({"stage": "resize", "from": [img_w, img_h], "to": [up_w, up_h], "interpolation": "LANCZOS4"})
                img_h, img_w = up_h, up_w
            elif shortest_side < min_side:
                scale_up = min(1.5, float(min_side) / max(1, shortest_side))
                if scale_up > 1.05:
                    up_w = int(round(img_w * scale_up))
                    up_h = int(round(img_h * scale_up))
                    image = cv2.resize(image, (up_w, up_h), interpolation=cv2.INTER_LANCZOS4)
                    if diagnostics is not None:
                        diagnostics.append({"stage": "resize", "from": [img_w, img_h], "to": [up_w, up_h], "interpolation": "LANCZOS4"})
                    img_h, img_w = up_h, up_w

            if max(img_h, img_w) > max_side:
                scale_down = max_side / float(max(img_h, img_w))
                new_w = int(img_w * scale_down)
                new_h = int(img_h * scale_down)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
                if diagnostics is not None:
                    diagnostics.append({"stage": "resize", "from": [img_w, img_h], "to": [new_w, new_h], "interpolation": "AREA"})

            if diagnostics is not None:
                diagnostics.append({"stage": "preprocessing_complete", "dimensions": [image.shape[1], image.shape[0]]})
            return image

        # Trường hợp 2: Ảnh chụp tối / Bóng đổ / Mờ / Giấy ố vàng
        # Khử bất đồng nhất ánh sáng & làm phẳng nền giấy
        try:
            image = compensate_illumination(image)
            if diagnostics is not None:
                diagnostics.append({"stage": "illumination_compensation", "status": "applied"})
        except Exception as e_ill:
            logger.debug(f"Illumination compensation failed: {e_ill}")
            if diagnostics is not None:
                diagnostics.append({"stage": "illumination_compensation", "status": "failed", "reason": type(e_ill).__name__})

        shortest_side = min(img_h, img_w)
        if shortest_side < min_side:
            scale_up = min(2.0, float(min_side) / max(1, shortest_side))
            if scale_up > 1.15:
                up_w = int(img_w * scale_up)
                up_h = int(img_h * scale_up)
                image = cv2.resize(image, (up_w, up_h), interpolation=cv2.INTER_CUBIC)
                if diagnostics is not None:
                    diagnostics.append({"stage": "resize", "from": [img_w, img_h], "to": [up_w, up_h], "interpolation": "CUBIC"})
                img_h, img_w = up_h, up_w

        if max(img_h, img_w) > max_side:
            scale_down = max_side / float(max(img_h, img_w))
            new_w = int(img_w * scale_down)
            new_h = int(img_h * scale_down)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            if diagnostics is not None:
                diagnostics.append({"stage": "resize", "from": [img_w, img_h], "to": [new_w, new_h], "interpolation": "AREA"})

        # CLAHE nhẹ (clipLimit=1.2) trên kênh L
        if enhance_contrast:
            try:
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
                cl = clahe.apply(l)
                image = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
                if diagnostics is not None:
                    diagnostics.append({"stage": "contrast_enhancement", "status": "applied", "method": "CLAHE", "clip_limit": 1.2, "tile_grid": [8, 8]})
            except Exception as exc:
                if diagnostics is not None:
                    diagnostics.append({"stage": "contrast_enhancement", "status": "failed", "reason": type(exc).__name__})

        # Khử nhiễu nhẹ
        if denoise:
            try:
                image = cv2.fastNlMeansDenoisingColored(image, None, 4, 4, 7, 21)
                if diagnostics is not None:
                    diagnostics.append({"stage": "denoising", "status": "applied", "method": "fastNlMeansDenoisingColored"})
            except Exception:
                image = cv2.bilateralFilter(image, d=5, sigmaColor=50, sigmaSpace=50)
                if diagnostics is not None:
                    diagnostics.append({"stage": "denoising", "status": "fallback", "method": "bilateralFilter"})

        if diagnostics is not None:
            diagnostics.append({"stage": "preprocessing_complete", "dimensions": [image.shape[1], image.shape[0]]})

        return image

    @staticmethod
    def _merge_horizontally_adjacent_subboxes(
        polygons: List[List[List[float]]],
        paddle_texts: List[str],
        paddle_scores: List[float],
        img_bgr: np.ndarray,
        return_source_indices: bool = False,
    ) -> Any:
        """
        Gộp các sub-box nằm ngang kề sát nhau trên cùng 1 dòng trước khi đưa vào VietOCR
        (Intra-Cell Horizontal Box Merging):
        - Nếu 2 box nằm trên cùng dòng ngang (overlap_y >= 0.55 và abs(cy1 - cy2) <= 0.45 * min_h).
        - Khoảng cách ngang rất nhỏ (-4px <= gap_x <= max(14px, 0.85 * min_h)).
        - Giữa chúng KHÔNG CÓ vạch kẻ dọc phân tách cột trên v_mask.
        -> Gộp 2 box thành 1 box duy nhất để VietOCR đọc nguyên cụm (ví dụ 'ấp' + '2' -> 'ấp 2').
        """
        if len(polygons) <= 1:
            if return_source_indices:
                return polygons, paddle_texts, paddle_scores, [[index] for index in range(len(polygons))]
            return polygons, paddle_texts, paddle_scores

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr
        img_h, img_w = gray.shape[:2]
        inv_gray = cv2.bitwise_not(gray)
        bw = cv2.adaptiveThreshold(inv_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -2)
        v_kernel_len = max(12, int(img_h / 45))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len))
        v_mask = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel)

        boxes_info = []
        for i, poly in enumerate(polygons):
            pts = poly[:4]
            x_min = min(p[0] for p in pts)
            x_max = max(p[0] for p in pts)
            y_min = min(p[1] for p in pts)
            y_max = max(p[1] for p in pts)
            txt = paddle_texts[i] if i < len(paddle_texts) else ""
            sc = paddle_scores[i] if i < len(paddle_scores) else 0.8
            boxes_info.append({
                "poly": poly,
                "source_indices": [i],
                "text": txt,
                "score": sc,
                "x_min": x_min, "x_max": x_max,
                "y_min": y_min, "y_max": y_max,
                "w": x_max - x_min, "h": y_max - y_min,
                "cy": (y_min + y_max) / 2.0,
            })

        boxes_info.sort(key=lambda b: (b["y_min"], b["x_min"]))

        merged_any = True
        while merged_any:
            merged_any = False
            new_boxes = []
            skip_indices = set()

            for i in range(len(boxes_info)):
                if i in skip_indices:
                    continue
                b1 = boxes_info[i]
                merge_target = None

                for j in range(i + 1, len(boxes_info)):
                    if j in skip_indices:
                        continue
                    b2 = boxes_info[j]

                    if b2["y_min"] - b1["y_max"] > max(15.0, b1["h"]):
                        break

                    min_h = min(b1["h"], b2["h"])
                    if min_h <= 2.0:
                        continue
                    y_overlap = max(0.0, min(b1["y_max"], b2["y_max"]) - max(b1["y_min"], b2["y_min"]))
                    vert_overlap_ratio = y_overlap / min_h
                    cy_diff = abs(b1["cy"] - b2["cy"])

                    if vert_overlap_ratio >= 0.55 and cy_diff <= 0.45 * min_h:
                        if b1["x_min"] <= b2["x_min"]:
                            left_b, right_b = b1, b2
                        else:
                            left_b, right_b = b2, b1

                        gap_x = right_b["x_min"] - left_b["x_max"]
                        max_gap = max(14.0, 0.85 * min_h)

                        if -4.0 <= gap_x <= max_gap:
                            chk_x1 = int(max(0, left_b["x_max"] - 2))
                            chk_x2 = int(min(img_w, right_b["x_min"] + 2))
                            chk_y1 = int(max(0, min(left_b["y_min"], right_b["y_min"])))
                            chk_y2 = int(min(img_h, max(left_b["y_max"], right_b["y_max"])))

                            has_vert_wall = False
                            if chk_x2 > chk_x1 and chk_y2 > chk_y1:
                                strip = v_mask[chk_y1:chk_y2, chk_x1:chk_x2]
                                if strip.size > 0:
                                    row_has_v = np.any(strip > 0, axis=1)
                                    if float(np.mean(row_has_v)) >= 0.35:
                                        has_vert_wall = True

                            if not has_vert_wall:
                                merge_target = j
                                break

                if merge_target is not None:
                    b2 = boxes_info[merge_target]
                    skip_indices.add(merge_target)
                    merged_any = True
                    mx_min = min(b1["x_min"], b2["x_min"])
                    mx_max = max(b1["x_max"], b2["x_max"])
                    my_min = min(b1["y_min"], b2["y_min"])
                    my_max = max(b1["y_max"], b2["y_max"])
                    new_poly = [
                        [float(mx_min), float(my_min)],
                        [float(mx_max), float(my_min)],
                        [float(mx_max), float(my_max)],
                        [float(mx_min), float(my_max)],
                    ]
                    comb_text = f"{b1['text']} {b2['text']}".strip()
                    comb_score = min(b1["score"], b2["score"])
                    new_boxes.append({
                        "poly": new_poly,
                        "source_indices": b1["source_indices"] + b2["source_indices"],
                        "text": comb_text,
                        "score": comb_score,
                        "x_min": mx_min, "x_max": mx_max,
                        "y_min": my_min, "y_max": my_max,
                        "w": mx_max - mx_min, "h": my_max - my_min,
                        "cy": (my_min + my_max) / 2.0,
                    })
                else:
                    new_boxes.append(b1)

            boxes_info = new_boxes

        out_polys = [b["poly"] for b in boxes_info]
        out_texts = [b["text"] for b in boxes_info]
        out_scores = [b["score"] for b in boxes_info]
        if return_source_indices:
            return out_polys, out_texts, out_scores, [b["source_indices"] for b in boxes_info]
        return out_polys, out_texts, out_scores

    # ------------------------------------------------------------------
    # Nhận dạng chính (Hybrid Pipeline)
    # ------------------------------------------------------------------

    def recognize(
        self,
        source,
        preprocess: bool = True,
        denoise: bool = True,
        max_side: int = 2560,
        include_diagnostics: bool = False,
        diagnostic_observer: Optional[Callable[[Dict[str, Any], Optional[np.ndarray]], None]] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
        progress_detail_callback: Optional[Callable[[int, int], None]] = None,
        cancel_event: Optional[Event] = None,
        stage_callback: Optional[Callable[[str, str, int], None]] = None,
    ) -> OCRResult:
        """
        Nhận dạng chữ từ ảnh với pipeline Hybrid:
        1. PaddleOCR Text Detection trích xuất toạ độ đa giác (polygons).
        2. Cắt (crop) từng dòng chữ theo góc nghiêng chuẩn xác.
        3. VietOCR Transformer đọc từng dòng chữ tiếng Việt.
        4. Sắp xếp lại thứ tự đọc văn bản và chuẩn hóa văn bản.
        """
        def report_progress(message: str) -> None:
            if progress_callback is not None:
                try:
                    progress_callback(message)
                except Exception as exc:
                    logger.debug("Ignoring OCR progress observer failure ({})", type(exc).__name__)

        def report_progress_detail(completed: int, total: int) -> None:
            if progress_detail_callback is not None:
                try:
                    progress_detail_callback(completed, total)
                except Exception as exc:
                    logger.debug("Ignoring OCR detailed progress observer failure ({})", type(exc).__name__)

        def report_stage(stage: str, transition: str) -> None:
            if stage_callback is not None:
                try:
                    stage_callback(stage, transition, time.perf_counter_ns())
                except Exception as exc:
                    logger.debug("Ignoring OCR stage observer failure ({})", type(exc).__name__)

        def check_cancelled() -> None:
            if cancel_event is not None and cancel_event.is_set():
                raise OCRCancelledError("OCR đã được hủy tại ranh giới an toàn của pipeline.")

        check_cancelled()
        self._ensure_loaded()
        check_cancelled()
        t0 = time.perf_counter()

        report_progress("Đang đọc và tiền xử lý ảnh …")
        report_stage("preprocessing", "start")

        # 1. Đọc ảnh
        source_path = str(source) if isinstance(source, (str, Path)) else None
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
        _notify_diagnostic_observer(diagnostic_observer, {
            "event": "source_image", "coordinate_space": "source_pixels",
            "image_size": [orig_w, orig_h], "source_path": source_path,
        }, image)
        preprocessing_steps: List[dict] = []
        diagnostic_report = None
        if include_diagnostics:
            diagnostic_report = {
                "schema_version": 1,
                "image": {"dimensions": [orig_w, orig_h], "quality": assess_image_quality(image)},
                "preprocessing": preprocessing_steps,
                "detection": {},
                "recognition": [],
                "layout": {},
            }

        # 2. Tiền xử lý ảnh
        if preprocess:
            processed_image = self.preprocess_image(
                image,
                max_side=max_side,
                denoise=denoise,
                diagnostics=preprocessing_steps if include_diagnostics else None,
            )
        else:
            processed_image = image
            if include_diagnostics:
                preprocessing_steps.append({"stage": "preprocessing", "status": "skipped"})

        proc_h, proc_w = processed_image.shape[:2]
        check_cancelled()
        scale_x = orig_w / float(proc_w)
        scale_y = orig_h / float(proc_h)
        _notify_diagnostic_observer(diagnostic_observer, {
            "event": "preprocessed_page", "coordinate_space": "preprocessed_pixels",
            "image_size": [proc_w, proc_h], "scale_to_source": [scale_x, scale_y],
            "preprocessing_enabled": preprocess, "denoise": denoise,
            "max_side": max_side,
        }, processed_image)
        report_stage("preprocessing", "end")
        report_stage("detection", "start")
        report_progress("Đang phát hiện vùng chữ …")

        # Thêm padding viền an toàn (16px) trước khi chạy DBNet Detection
        # để bảo toàn 100% dấu thanh và nét chữ nằm sát mép trên/mép biên ảnh
        pad_det = 16
        padded_det_img = cv2.copyMakeBorder(
            processed_image,
            top=pad_det, bottom=pad_det, left=pad_det, right=pad_det,
            borderType=cv2.BORDER_CONSTANT,
            value=[255, 255, 255],
        )
        _notify_diagnostic_observer(diagnostic_observer, {
            "event": "paddle_detector_input", "coordinate_space": "preprocessed_plus_padding",
            "image_size": [int(padded_det_img.shape[1]), int(padded_det_img.shape[0])],
            "padding_px": {"top": pad_det, "bottom": pad_det, "left": pad_det, "right": pad_det},
        }, padded_det_img)

        logger.debug("Running PaddleOCR Text Detection …")
        detection_records: List[Dict[str, Any]] = []
        recognition_records: List[Dict[str, Any]] = []
        raw_result_diagnostics: List[Dict[str, Any]] = []
        raw_alignment_statuses: List[str] = []
        collect_raw_diagnostics = include_diagnostics or diagnostic_observer is not None

        def unpad_polygon(polygon: List[List[float]]) -> List[List[float]]:
            return [[max(0.0, min(float(proc_w), x - pad_det)),
                     max(0.0, min(float(proc_h), y - pad_det))] for x, y in polygon]

        def append_parsed_result(result_index: int, parsed: Dict[str, Any]) -> None:
            result_detections = []
            for record in parsed["detections"]:
                detection = {
                    **record,
                    "detection_record_index": len(detection_records),
                    "result_index": result_index,
                    "polygon_detector_output": record["polygon"],
                    "polygon_preprocessed": unpad_polygon(record["polygon"]),
                }
                result_detections.append(detection)
                detection_records.append(detection)

            result_recognitions = []
            for record in parsed["recognitions"]:
                recognition = {
                    **record,
                    "recognition_record_index": len(recognition_records),
                    "result_index": result_index,
                    "polygon_detector_output": record["polygon"],
                    "polygon_preprocessed": unpad_polygon(record["polygon"]),
                }
                result_recognitions.append(recognition)
                recognition_records.append(recognition)

            detection_to_recognition = _map_candidate_polygons_to_regions(
                [item["polygon_preprocessed"] for item in result_detections],
                [item["polygon_preprocessed"] for item in result_recognitions],
            )
            for recognition, link in zip(result_recognitions, detection_to_recognition):
                recognition["source_detection_mapping"] = link
                if link.get("status") == "matched":
                    recognition["source_detection_index"] = result_detections[
                        link["candidate_index"]
                    ]["detection_index"]
                    recognition["source_detection_result_index"] = result_index
            raw_alignment_statuses.append(parsed["recognition_status"])
            if collect_raw_diagnostics:
                raw_result_diagnostics.append({
                    "result_index": result_index,
                    "recognition_status": parsed["recognition_status"],
                    "raw_counts": parsed["raw_counts"],
                    "detections": result_detections,
                    "recognitions": result_recognitions,
                })
            if diagnostic_observer is not None:
                _notify_diagnostic_observer(diagnostic_observer, {
                    "event": "paddle_raw_result_alignment",
                    "result_index": result_index,
                    "coordinate_space": "predict_input_pixels_then_unpadded_preprocessed_pixels",
                    "recognition_status": parsed["recognition_status"],
                    "raw_counts": parsed["raw_counts"],
                    "detections": result_detections,
                    "recognitions": result_recognitions,
                    "transforms": {"remove_detector_padding_px": pad_det,
                                   "clamp_to_preprocessed_size": [proc_w, proc_h]},
                })

        # 3. Chạy Detection qua PaddleOCR
        detector = self._ocr
        assert detector is not None, "PaddleOCR detector not initialized"
        detection_started = time.perf_counter()
        try:
            results = detector.predict(padded_det_img)
            check_cancelled()
            for result_index, res in enumerate(results):
                append_parsed_result(result_index, _paddle_result_records(res))
        except OCRCancelledError:
            raise
        except Exception as exc_det:
            logger.warning(f"Detection via predict failed, falling back to .ocr(): {exc_det}")
            detection_records = []
            recognition_records = []
            raw_result_diagnostics = []
            raw_alignment_statuses = []
            try:
                check_cancelled()
                try:
                    raw_results = detector.ocr(padded_det_img, cls=self.use_angle_cls)
                except TypeError as exc_legacy_signature:
                    # PaddleOCR 3.x keeps ocr() as a wrapper around predict(),
                    # whose signature no longer accepts the legacy `cls` kwarg.
                    if "cls" not in str(exc_legacy_signature):
                        raise
                    raw_results = detector.ocr(padded_det_img)
                if raw_results and _paddle_result_value(raw_results[0], "dt_polys") is not None:
                    for result_index, result in enumerate(raw_results):
                        append_parsed_result(result_index, _paddle_result_records(result))
                elif raw_results and raw_results[0]:
                    legacy_lines = raw_results[0]
                    for line_index, line in enumerate(legacy_lines):
                        bbox_points = line[0]
                        raw_polygon = [[float(pt[0]), float(pt[1])] for pt in bbox_points[:4]]
                        if len(raw_polygon) != 4:
                            continue
                        confidence = None
                        try:
                            confidence = float(line[1][1])
                        except (TypeError, ValueError, IndexError):
                            pass
                        paired_polygon = unpad_polygon(raw_polygon)
                        detection_records.append({
                            "detection_index": line_index, "result_index": 0,
                            "polygon_detector_output": raw_polygon,
                            "polygon_preprocessed": paired_polygon,
                        })
                        recognition_records.append({
                            "recognition_index": line_index, "result_index": 0,
                            "polygon_detector_output": raw_polygon,
                            "polygon_preprocessed": paired_polygon,
                            "text": str(line[1][0]), "score": confidence,
                            "score_status": "available" if confidence is not None else "unknown",
                            "source_detection_index": line_index,
                            "source_detection_result_index": 0,
                            "source_detection_mapping": {"status": "matched",
                                "reason": "legacy_row_contains_polygon_text_score_together",
                                "coordinate_space": "preprocessed_image"},
                        })
                    raw_alignment_statuses.append("verified_legacy_row_polygon_text_pairing")
                    legacy_diagnostics = {
                        "result_index": 0,
                        "recognition_status": "verified_legacy_row_polygon_text_pairing",
                        "raw_counts": {"dt_polys": len(detection_records),
                                       "rec_polys": len(recognition_records),
                                       "rec_texts": len(recognition_records),
                                       "rec_scores": sum(item["score"] is not None for item in recognition_records)},
                        "detections": detection_records,
                        "recognitions": recognition_records,
                    }
                    if collect_raw_diagnostics:
                        raw_result_diagnostics.append(legacy_diagnostics)
                    if diagnostic_observer is not None:
                        _notify_diagnostic_observer(diagnostic_observer, {
                            "event": "paddle_raw_result_alignment",
                            "coordinate_space": "predict_input_pixels_then_unpadded_preprocessed_pixels",
                            **legacy_diagnostics,
                            "transforms": {"remove_detector_padding_px": pad_det,
                                           "clamp_to_preprocessed_size": [proc_w, proc_h]},
                        })
            except OCRCancelledError:
                raise
            except Exception as exc_raw:
                logger.error(f"Detection fallback failed: {exc_raw}")

        logger.debug("PaddleOCR detection stage total (worker thread): {:.1f}ms", (time.perf_counter() - detection_started) * 1000)
        report_stage("detection", "end")

        # 3.b. Only combine detections when VietOCR will read the fused image
        # region. PaddleOCR text belongs to each original polygon and must not
        # be concatenated and later presented as a prediction for one polygon.
        use_vietocr = (self.lang == "vi" and self._vietocr is not None)
        polygons = [record["polygon_preprocessed"] for record in detection_records]
        if not use_vietocr:
            polygons = [record["polygon_preprocessed"] for record in recognition_records]
        geometric_merge_applied = False
        region_source_detection_groups: List[List[int]] = []
        paddle_texts = [record["text"] for record in recognition_records]
        paddle_scores = [record["score"] for record in recognition_records]
        if use_vietocr:
            polygons, _, _, region_source_detection_groups = self._merge_horizontally_adjacent_subboxes(
                polygons, [""] * len(polygons), [0.0] * len(polygons), processed_image,
                return_source_indices=True,
            )
            geometric_merge_applied = any(len(group) > 1 for group in region_source_detection_groups)
        raw_region_mappings = _map_candidate_polygons_to_regions(
            [record["polygon_preprocessed"] for record in recognition_records], polygons
        )
        unverified_result_statuses = [
            status for status in raw_alignment_statuses
            if status not in {"verified_rec_polys_text_alignment",
                              "verified_legacy_row_polygon_text_pairing"}
        ]
        region_paddle_texts: List[str] = []
        region_paddle_scores: List[Optional[float]] = []
        region_paddle_records: List[Optional[Dict[str, Any]]] = []
        paddle_candidate_mappings: List[Dict[str, Any]] = []
        for region_index, mapping in enumerate(raw_region_mappings):
            if not mapping.get("candidate_indices") and unverified_result_statuses:
                mapping = {"status": "unknown", "reason": unverified_result_statuses[0],
                           "candidate_indices": []}
            recognition_index = mapping.get("candidate_index") if mapping.get("status") == "matched" else None
            record = recognition_records[recognition_index] if recognition_index is not None else None
            mapping = dict(mapping)
            if use_vietocr:
                mapping, recognition_index = _resolve_region_candidate_source(
                    mapping, region_source_detection_groups[region_index],
                    detection_records, recognition_records,
                )
                record = recognition_records[recognition_index] if recognition_index is not None else None
            if record is not None:
                mapping["recognition_index"] = record["recognition_index"]
                mapping["recognition_record_index"] = record["recognition_record_index"]
                mapping["recognition_result_index"] = record["result_index"]
                mapping["recognition_polygon_preprocessed"] = record["polygon_preprocessed"]
                mapping["source_detection_mapping"] = record.get("source_detection_mapping", {
                    "status": "unknown", "reason": "source_detection_link_unavailable"})
                mapping["source_detection_index"] = record.get("source_detection_index")
            region_paddle_records.append(record)
            paddle_candidate_mappings.append(mapping)
            region_paddle_texts.append(record["text"] if record is not None else "")
            region_paddle_scores.append(record["score"] if record is not None else None)
        if diagnostic_report is not None:
            diagnostic_report["detection"] = {
                "count_after_geometric_merge": len(polygons),
                "coordinate_space": "preprocessed_image",
                "scale_to_original": [scale_x, scale_y],
                "predict_results": raw_result_diagnostics,
                "detections": detection_records,
                "recognitions": recognition_records,
                "polygons": [
                    {
                        "polygon": [[float(x), float(y)] for x, y in polygon[:4]],
                        "paddle_text": region_paddle_texts[index],
                        "confidence": region_paddle_scores[index],
                        "paddle_mapping": paddle_candidate_mappings[index],
                    }
                    for index, polygon in enumerate(polygons)
                ],
            }
        report_stage("recognition", "start")
        report_progress(f"Đã phát hiện {len(polygons)} vùng; đang nhận dạng …")

        # 4. Text Recognition: Sử dụng VietOCR cho tiếng Việt hoặc PaddleOCR
        boxes: List[OCRBox] = []

        if use_vietocr:
            logger.debug(f"Recognizing {len(polygons)} text lines with VietOCR ({self.vietocr_model_name}) …")

        progress_interval = max(1, len(polygons) // 100)
        recognition_started = time.perf_counter()
        vietocr_inference_seconds = 0.0
        for idx, poly in enumerate(polygons):
            check_cancelled()
            # Preserve legacy textual progress for non-UI callers. The desktop
            # UI consumes the numeric callback below to avoid resetting its ring.
            if progress_detail_callback is None and idx % progress_interval == 0:
                report_progress(f"Đang nhận dạng vùng {idx + 1}/{len(polygons)} …")
            paddle_text = region_paddle_texts[idx]
            paddle_conf = region_paddle_scores[idx]
            original_poly = [
                [float(pt[0] * scale_x), float(pt[1] * scale_y)] for pt in poly[:4]
            ]
            # Tính toạ độ nguyên trên ảnh gốc
            bbox_int = [
                [int(round(pt[0] * scale_x)), int(round(pt[1] * scale_y))]
                for pt in poly[:4]
            ]
            if diagnostic_observer is not None:
                source_rectified_crop = get_rotate_crop_image(image, np.array(original_poly))
                _notify_diagnostic_observer(diagnostic_observer, {
                    "event": "source_rectified_crop", "crop_id": idx,
                    "coordinate_space": "source_pixels", "polygon_source": original_poly,
                    "source_bbox_rounded": bbox_int,
                    "crop_transform": "perspective_rectification_then_6_8_8_18_px_white_padding",
                    "perspective_transform_source": _describe_perspective_crop(original_poly),
                    "mapping": {"source_x_per_preprocessed_x": scale_x,
                                "source_y_per_preprocessed_y": scale_y,
                                "detector_padding_removed_px": pad_det},
                }, source_rectified_crop)

            text = ""
            conf: Optional[float] = None
            crop_bgr: Optional[np.ndarray] = None
            recognizer_name = "none"

            if use_vietocr:
                # Cắt (crop) vùng chữ từ ảnh tiền xử lý với padding viền an toàn
                crop_bgr = get_rotate_crop_image(processed_image, np.array(poly))
                if crop_bgr is not None and crop_bgr.size > 0:
                    _notify_diagnostic_observer(diagnostic_observer, {
                        "event": "preprocessed_rectified_crop", "crop_id": idx,
                        "coordinate_space": "preprocessed_pixels",
                        "polygon_preprocessed": [[float(x), float(y)] for x, y in poly[:4]],
                        "polygon_padded_detector": [[float(x + pad_det), float(y + pad_det)] for x, y in poly[:4]],
                        "polygon_source": original_poly,
                        "source_bbox_rounded": bbox_int,
                        "perspective_transform_preprocessed": _describe_perspective_crop(poly),
                        "perspective_transform_source": _describe_perspective_crop(original_poly),
                        "mapping": {"source_x_per_preprocessed_x": scale_x,
                                    "source_y_per_preprocessed_y": scale_y,
                                    "detector_padding_removed_px": pad_det,
                                    "geometric_merge_applied": geometric_merge_applied},
                        "crop_transform": "perspective_rectification_then_6_8_8_18_px_white_padding",
                        "stage": "before_clean_crop_borders",
                    }, crop_bgr)
                    _notify_diagnostic_observer(diagnostic_observer, {
                        "event": "paddle_polygon_crop_preview", "crop_id": idx,
                        "stage": "perspective_crop_of_preprocessed_detector_page",
                        "recognizer_input_exact": False,
                        "reason": "PaddleOCR predict API returns text and polygon but does not expose its internal recognition tensor/crop",
                    }, crop_bgr)
                    # 1. Tẩy đường kẻ bảng ở sát biên lát cắt (tránh nhận nhầm [, |, F, ])
                    crop_bgr = clean_crop_borders(crop_bgr)
                    _notify_diagnostic_observer(diagnostic_observer, {
                        "event": "vietocr_border_cleaned_crop", "crop_id": idx,
                        "stage": "after_clean_crop_borders_before_split_or_model",
                        "polygon_preprocessed": [[float(x), float(y)] for x, y in poly[:4]],
                        "polygon_source": original_poly,
                    }, crop_bgr)

                    # 2. Kiểm tra dải hoa văn trang trí / đường kẻ lặp ở tầng thị giác
                    if is_decorative_pattern_crop(crop_bgr):
                        text = ""
                        conf = 0.0
                    else:
                        recognizer_name = f"vietocr:{self.vietocr_model_name}"
                        try:
                            # 3. Tách dòng rất dài có chấm lửng thành các sub-crops tránh attention drift nuốt chữ
                            sub_crops = split_dotted_line_crop(crop_bgr)
                            if len(sub_crops) > 1 and self._vietocr is not None:
                                sub_texts = []
                                sub_confs = []
                                for subcrop_index, sc in enumerate(sub_crops):
                                    sc_padded = pad_small_crop_for_vietocr(sc)
                                    _notify_diagnostic_observer(diagnostic_observer, {
                                        "event": "vietocr_subcrop_before_padding", "crop_id": idx,
                                        "subcrop_index": subcrop_index,
                                        "split_count": len(sub_crops),
                                        "stage": "split_dotted_line_crop_output",
                                    }, sc)
                                    _notify_diagnostic_observer(diagnostic_observer, {
                                        "event": "vietocr_recognizer_input", "crop_id": idx,
                                        "subcrop_index": subcrop_index,
                                        "split_count": len(sub_crops),
                                        "stage": "after_pad_small_crop_for_vietocr",
                                        "recognizer": f"vietocr:{self.vietocr_model_name}",
                                        "color_conversion": "BGR_to_RGB_for_PIL",
                                    }, sc_padded)
                                    sc_rgb = cv2.cvtColor(sc_padded, cv2.COLOR_BGR2RGB)
                                    sc_pil = Image.fromarray(sc_rgb)
                                    inference_started = time.perf_counter()
                                    p_res = self._vietocr.predict(sc_pil, return_prob=True)
                                    vietocr_inference_seconds += time.perf_counter() - inference_started
                                    if isinstance(p_res, tuple):
                                        t_sub = (p_res[0] or "").strip()
                                        c_sub = float(p_res[1]) if p_res[1] is not None else None
                                    else:
                                        t_sub = str(p_res).strip()
                                        c_sub = None
                                    _notify_diagnostic_observer(diagnostic_observer, {
                                        "event": "recognition_subcandidate",
                                        "crop_id": idx,
                                        "subcrop_index": subcrop_index,
                                        "recognizer": f"vietocr:{self.vietocr_model_name}",
                                        "text": t_sub,
                                        "confidence": c_sub,
                                        "confidence_scale_calibrated_with_other_model": False,
                                        "recognizer_input_exact": True,
                                    })
                                    if t_sub:
                                        sub_texts.append(t_sub)
                                        sub_confs.append(c_sub)
                                if sub_texts:
                                    text = " ...... ".join(sub_texts)
                                    known_sub_confs = [value for value in sub_confs if value is not None]
                                    conf = float(np.mean(known_sub_confs)) if known_sub_confs else None
                                else:
                                    text = ""
                                    conf = None
                            else:
                                # 4. Đệm khung trắng chuẩn tỷ lệ cho ô hẹp & chữ số (05, 01, 1, 2)
                                crop_bgr_padded = pad_small_crop_for_vietocr(crop_bgr)
                                _notify_diagnostic_observer(diagnostic_observer, {
                                    "event": "vietocr_recognizer_input", "crop_id": idx,
                                    "subcrop_index": None,
                                    "stage": "after_pad_small_crop_for_vietocr",
                                    "recognizer": f"vietocr:{self.vietocr_model_name}",
                                    "color_conversion": "BGR_to_RGB_for_PIL",
                                    "polygon_preprocessed": [[float(x), float(y)] for x, y in poly[:4]],
                                    "polygon_source": original_poly,
                                }, crop_bgr_padded)
                                crop_rgb = cv2.cvtColor(crop_bgr_padded, cv2.COLOR_BGR2RGB)
                                crop_pil = Image.fromarray(crop_rgb)
                                if self._vietocr is not None:
                                    inference_started = time.perf_counter()
                                    pred_res = self._vietocr.predict(crop_pil, return_prob=True)
                                    vietocr_inference_seconds += time.perf_counter() - inference_started
                                    recognizer_name = f"vietocr:{self.vietocr_model_name}"
                                else:
                                    pred_res = ("", 0.0)
                                if isinstance(pred_res, tuple):
                                    text = pred_res[0] or ""
                                    prob = pred_res[1]
                                    conf = float(prob) if prob is not None else None
                                else:
                                    text = str(pred_res)
                                    conf = None
                        except Exception as err:
                            logger.debug(f"VietOCR line predict error: {err}")
                            text = paddle_text
                            conf = paddle_conf
                            recognizer_name = "paddleocr" if paddle_text else "none"
                else:
                    text = paddle_text
                    conf = paddle_conf
                    recognizer_name = "paddleocr" if paddle_text else "none"
            else:
                # Tiếng Anh hoặc ngôn ngữ khác dùng kết quả từ PaddleOCR
                if diagnostic_observer is not None:
                    paddle_preview = get_rotate_crop_image(processed_image, np.array(poly))
                    _notify_diagnostic_observer(diagnostic_observer, {
                        "event": "paddle_polygon_crop_preview", "crop_id": idx,
                        "stage": "perspective_crop_of_preprocessed_detector_page",
                        "recognizer_input_exact": False,
                        "reason": "PaddleOCR predict API returns text and polygon but does not expose its internal recognition tensor/crop",
                    }, paddle_preview)
                text = paddle_text
                conf = paddle_conf
                recognizer_name = "paddleocr" if paddle_text else "none"

            raw_recognized_text = text
            # NFC is non-semantic; retain recognizer output separately.
            text = unicodedata.normalize("NFC", text)

            # Bỏ qua dòng trống, dòng ảo giác lặp từ hoặc ký tự rác, hoặc quá thấp độ tin cậy
            # These are two predictions for the same detected polygon. Their
            # confidence scales are not calibrated against each other; retain
            # both and flag disagreement instead of choosing by score.
            accepted = bool(text and (conf is None or conf >= self.drop_score))
            recognition_metadata = _recognition_evidence(
                crop_id=idx, crop_bbox=bbox_int, primary_name=recognizer_name,
                primary_text=raw_recognized_text, primary_confidence=conf,
                alternate_text=paddle_text if use_vietocr and recognizer_name != "paddleocr" else "",
                alternate_confidence=paddle_conf, alternate_name="paddleocr",
                alternate_mapping=paddle_candidate_mappings[idx],
            )
            _notify_diagnostic_observer(diagnostic_observer, {
                "event": "recognition_candidate",
                "crop_id": idx,
                "recognizer": recognizer_name,
                "text": raw_recognized_text,
                "confidence": conf,
                "confidence_scale_calibrated_with_other_model": False,
                "bbox_source": bbox_int,
                "polygon_preprocessed": [[float(x), float(y)] for x, y in poly[:4]],
                "polygon_source": original_poly,
                "paddle_candidate_mapping": paddle_candidate_mappings[idx],
                "paddle_recognition_record": region_paddle_records[idx],
                "crop_mapping": recognition_metadata,
                "recognizer_input_exact": recognizer_name.startswith("vietocr:"),
            })
            if use_vietocr and recognizer_name != "paddleocr":
                _notify_diagnostic_observer(diagnostic_observer, {
                    "event": "recognition_candidate",
                    "crop_id": idx,
                    "recognizer": "paddleocr",
                    "text": paddle_text,
                    "confidence": paddle_conf,
                    "confidence_scale_calibrated_with_other_model": False,
                    "bbox_source": bbox_int,
                    "polygon_preprocessed": [[float(x), float(y)] for x, y in poly[:4]],
                    "polygon_source": original_poly,
                    "paddle_candidate_mapping": paddle_candidate_mappings[idx],
                    "paddle_recognition_record": region_paddle_records[idx],
                    "recognizer_input_exact": False,
                    "input_artifact_event": "paddle_detector_input",
                })
            if accepted:
                boxes.append(OCRBox(text=text, confidence=float(conf) if conf is not None else None, bbox=bbox_int,
                                    raw_text=raw_recognized_text, metadata=recognition_metadata))
            if diagnostic_report is not None:
                diagnostic_report["recognition"].append({
                    "box_index": idx,
                    "bbox_original": [[int(x), int(y)] for x, y in bbox_int],
                    "paddle_text": paddle_text,
                    "paddle_confidence": paddle_conf,
                    "recognizer_raw_text": raw_recognized_text,
                    "normalized_text": text,
                    "confidence": conf,
                    **recognition_metadata,
                    "accepted": accepted,
                })
            # Count completed detector regions, including rejected/empty text.
            # This is recognition-stage progress only, never whole-job progress.
            report_progress_detail(idx + 1, len(polygons))
        logger.debug(
            "OCR recognition stage including crop and bookkeeping ({} regions, worker thread): {:.1f}ms; VietOCR model inference total: {:.1f}ms",
            len(polygons), (time.perf_counter() - recognition_started) * 1000,
            vietocr_inference_seconds * 1000,
        )
        report_stage("recognition", "end")

        report_stage("layout", "start")
        # 5. Lọc nhiễu quét do cắt ngang viền ảnh (Border noise filtering)
        box_count_before_border_filter = len(boxes)
        boxes = filter_border_noise_boxes(boxes, image_width=orig_w, image_height=orig_h)
        box_count_after_border_filter = len(boxes)

        # 5.b. Tách các OCRBox bị DBNet gộp ngang xuyên qua vạch kẻ dọc của bảng
        boxes = split_table_crossing_boxes(boxes, image=image, vietocr_model=self._vietocr)
        if diagnostic_report is not None:
            diagnostic_report["detection"]["recognized_box_counts"] = {
                "before_border_filter": box_count_before_border_filter,
                "after_border_filter": box_count_after_border_filter,
                "after_boundary_split": len(boxes),
            }
        diagnostic_input_boxes = list(boxes) if diagnostic_report is not None else []

        # 6. Gom nhóm các khối chữ theo dòng và tái dựng bố cục tài liệu (Layout Reconstruction)
        check_cancelled()
        report_progress("Đang tái dựng bố cục và bảng …")
        layout_res = reconstruct_document_layout(
            boxes,
            image=image,
            vietocr_model=self._vietocr,
            diagnostics=diagnostic_report["layout"] if diagnostic_report is not None else None,
        )
        grouped_lines = layout_res[0]
        detected_tables = layout_res[1]
        parallel_hdr = getattr(layout_res, "parallel_header", None)
        boxes = [b for line in grouped_lines for b in line]

        # 7. Tổng hợp toàn bộ văn bản (đối với bảng, nối các ô bằng ' | ', đối với văn bản thường nối bằng space)
        # Nhận diện các bảng vật lý có ô gộp phức tạp để render nguyên khối theo đúng thứ tự đọc
        phys_merged_tables: Dict[int, Tuple[OCRTable, Set[int]]] = {}
        phys_merged_table_indexes: Set[int] = set()
        for table_index, tbl in enumerate(detected_tables):
            cells = getattr(tbl, "cells", [])
            is_phys_merged = any(getattr(c, "is_merged", False) for c in cells) and (
                tbl.num_cols >= 6 or any(getattr(c, "col_span", 1) >= 3 for c in cells)
            )
            if is_phys_merged:
                phys_merged_table_indexes.add(table_index)
                tbl_box_ids = {id(b) for r_b in tbl.row_boxes for b in r_b} | {
                    id(b) for c in cells for b in c.boxes
                }
                phys_merged_tables[table_index] = (tbl, tbl_box_ids)

        rendered_physical_table_indexes: Set[int] = set()
        emitted_table_blocks: Set[int] = set()
        document_blocks: List[DocumentBlock] = []
        box_index_by_id = {id(box): index for index, box in enumerate(boxes)}
        if parallel_hdr is not None:
            header_boxes = parallel_hdr.left_boxes + parallel_hdr.right_boxes
            header_ids = {id(box) for box in header_boxes}
            document_blocks.append(DocumentBlock(
                kind="parallel_header", order=0,
                text="\n".join(parallel_hdr.left_lines + parallel_hdr.right_lines),
                raw_text="\n".join(box.raw_text for box in header_boxes if box.raw_text),
                box_indices=[box_index_by_id[id(box)] for box in header_boxes if id(box) in box_index_by_id],
                bbox=[min(point[0] for box in header_boxes for point in box.bbox),
                      min(point[1] for box in header_boxes for point in box.bbox),
                      max(point[0] for box in header_boxes for point in box.bbox),
                      max(point[1] for box in header_boxes for point in box.bbox)] if header_boxes else None,
            ))
        else:
            header_ids = set()

        # Row identity is (table index, logical row index) within this result.
        # OCRBox ids are used only to associate current in-memory geometry with
        # that identity; row text is never an identifier.
        table_row_map = []
        for table_index, tbl in enumerate(detected_tables):
            if table_index in phys_merged_table_indexes:
                continue
            for r_idx, r_boxes in enumerate(tbl.row_boxes):
                if r_idx < len(tbl.grid):
                    box_id_set = frozenset(id(b) for b in r_boxes)
                    if box_id_set:
                        table_row_map.append((box_id_set, table_index, r_idx, tbl.grid[r_idx]))

        rendered_table_rows: Set[Tuple[int, int]] = set()
        text_lines = []
        for line in grouped_lines:
            if not line:
                continue
            if all(id(box) in header_ids for box in line):
                text_lines.append(join_ocr_line_boxes(line))
                continue

            line_box_ids = frozenset(id(b) for b in line)

            # Kiểm tra xem dòng này có chứa box thuộc bảng vật lý ô gộp phức tạp không
            matched_phys_tbl_index = None
            for table_index, (tbl, tbl_box_ids) in phys_merged_tables.items():
                if any(id(b) in tbl_box_ids for b in line):
                    matched_phys_tbl_index = table_index
                    break

            if matched_phys_tbl_index is not None:
                if matched_phys_tbl_index not in emitted_table_blocks:
                    table = detected_tables[matched_phys_tbl_index]
                    document_blocks.append(DocumentBlock(
                        kind="table", order=len(document_blocks), table_index=matched_phys_tbl_index,
                        bbox=list(table.bbox) if table.bbox else None,
                        box_indices=_table_source_box_indices(table, box_index_by_id),
                    ))
                    emitted_table_blocks.add(matched_phys_tbl_index)
                if matched_phys_tbl_index not in rendered_physical_table_indexes:
                    rendered_physical_table_indexes.add(matched_phys_tbl_index)
                    tbl, _ = phys_merged_tables[matched_phys_tbl_index]
                    for row_data in tbl.grid:
                        text_lines.append(" | ".join(c.strip() for c in row_data))
                continue
            matched_row_identity = _resolve_table_row_identity(
                line_box_ids, table_row_map, rendered_table_rows
            )

            if matched_row_identity is not None:
                table_index, _, row_data = matched_row_identity
                tbl = detected_tables[table_index]
                if table_index not in emitted_table_blocks:
                    document_blocks.append(DocumentBlock(
                        kind="table", order=len(document_blocks), table_index=table_index,
                        bbox=list(tbl.bbox) if tbl.bbox else None,
                        box_indices=_table_source_box_indices(tbl, box_index_by_id),
                    ))
                    emitted_table_blocks.add(table_index)
                text_lines.append(" | ".join(c.strip() for c in row_data))
            else:
                line_joined = join_ocr_line_boxes(line)
                text_lines.append(line_joined)
                document_blocks.append(DocumentBlock(
                    kind="paragraph", order=len(document_blocks), text=line_joined,
                    raw_text=" ".join(box.raw_text.strip() for box in line if box.raw_text.strip()),
                    box_indices=[box_index_by_id[id(box)] for box in line if id(box) in box_index_by_id],
                    bbox=[min(point[0] for box in line for point in box.bbox),
                          min(point[1] for box in line for point in box.bbox),
                          max(point[0] for box in line for point in box.bbox),
                          max(point[1] for box in line for point in box.bbox)] if line else None,
                ))

        # Place boxless/empty tables using their geometry among the paragraph
        # events. Their position is explicit metadata, not inferred from text.
        for table_index, table in enumerate(detected_tables):
            if table_index in emitted_table_blocks:
                continue
            table_top = table.bbox[1] if len(table.bbox) >= 4 else float("inf")
            insert_at = next((idx for idx, block in enumerate(document_blocks)
                              if block.kind != "parallel_header" and block.bbox
                              and block.bbox[1] >= table_top), len(document_blocks))
            document_blocks.insert(insert_at, DocumentBlock(
                kind="table", order=insert_at, table_index=table_index,
                bbox=list(table.bbox) if table.bbox else None,
                box_indices=_table_source_box_indices(table, box_index_by_id),
            ))
            emitted_table_blocks.add(table_index)
        for order, block in enumerate(document_blocks):
            block.order = order
        report_progress("Đang phân tích kiểu chữ và tổng hợp kết quả …")
        check_cancelled()
        style_diagnostics: Optional[List[dict]] = [] if diagnostic_observer is not None else None
        infer_visual_document_styles(
            document_blocks, boxes, image, diagnostics=style_diagnostics
        )
        if style_diagnostics is not None:
            for style_record in style_diagnostics:
                _notify_diagnostic_observer(diagnostic_observer, {
                    "event": "style_analysis",
                    "coordinate_space": "source_pixels",
                    **style_record,
                })

        raw_parts: List[str] = []
        for block in document_blocks:
            if block.kind == "parallel_header":
                if block.raw_text:
                    raw_parts.extend(block.raw_text.splitlines())
            elif block.kind == "paragraph":
                raw_parts.append(block.raw_text or block.text)
            elif block.kind == "table" and block.table_index is not None:
                table = detected_tables[block.table_index]
                cells_by_origin = {(cell.row_start, cell.col_start): cell for cell in table.cells}
                for row_index, row_data in enumerate(table.grid):
                    raw_row = []
                    for col_index, value in enumerate(row_data):
                        cell = cells_by_origin.get((row_index, col_index))
                        if cell is not None and cell.boxes:
                            raw_value = " ".join(box.raw_text.strip() for box in cell.boxes if box.raw_text.strip())
                        else:
                            raw_value = value
                        raw_row.append(raw_value)
                    raw_parts.append(" | ".join(raw_row))
        raw_full_text = "\n".join(raw_parts)
        normalized_parts: List[str] = []
        for block in document_blocks:
            if block.kind == "parallel_header":
                normalized_parts.extend(block.text.splitlines())
            elif block.kind == "paragraph":
                normalized_parts.append(block.text)
            elif block.kind == "table" and block.table_index is not None:
                table = detected_tables[block.table_index]
                normalized_parts.extend(" | ".join(value for value in row) for row in table.grid)
        # NFC and whitespace normalization are represented by box/block text;
        # semantic correction remains a separate, user-confirmed operation.
        full_text = "\n".join(normalized_parts)
        audit_records: List[dict] = []
        if diagnostic_report is not None:
            input_indexes = {id(box): index for index, box in enumerate(diagnostic_input_boxes)}
            line_membership: Dict[int, List[int]] = {}
            for line_index, line in enumerate(grouped_lines):
                for box in line:
                    if id(box) in input_indexes:
                        line_membership.setdefault(input_indexes[id(box)], []).append(line_index)
            table_membership: Dict[int, List[Tuple[int, int]]] = {}
            for table_index, table in enumerate(detected_tables):
                for cell in table.cells:
                    for box in cell.boxes:
                        if id(box) in input_indexes:
                            table_membership.setdefault(input_indexes[id(box)], []).append((table_index, cell.row_start, cell.col_start))
            diagnostic_report["layout"]["ownership"] = [
                {
                    "box_index": index,
                    "line_indexes": line_membership.get(index, []),
                    "table_cells": table_membership.get(index, []),
                    "owner": "table_cell" if index in table_membership else "paragraph" if index in line_membership else "unassigned",
                }
                for index in range(len(diagnostic_input_boxes))
            ]
            diagnostic_report["layout"]["ownership_invariants"] = {
                "recognized_box_count": len(diagnostic_input_boxes),
                "unassigned_box_count": sum(
                    not line_membership.get(index) and index not in table_membership
                    for index in range(len(diagnostic_input_boxes))
                ),
                "duplicate_line_assignment_count": sum(
                    max(0, len(line_membership.get(index, [])) - 1)
                    for index in range(len(diagnostic_input_boxes))
                ),
            }
            diagnostic_report["layout"]["final_tables"] = [
                {
                    "bbox": list(table.bbox),
                    "rows": table.num_rows,
                    "columns": table.num_cols,
                    "grid": [list(row) for row in table.grid],
                    "cells": [
                        {
                            "row_start": cell.row_start,
                            "col_start": cell.col_start,
                            "row_end": cell.row_end,
                            "col_end": cell.col_end,
                            "bbox": list(cell.bbox) if cell.bbox is not None else None,
                            "text": cell.text,
                            "diagonal_border": cell.diagonal_border,
                        }
                        for cell in table.cells
                    ],
                }
                for table in detected_tables
            ]

        elapsed = time.perf_counter() - t0
        result = OCRResult(
            boxes=boxes,
            full_text=full_text,
            language=self.lang,
            elapsed_seconds=elapsed,
            image_width=orig_w,
            image_height=orig_h,
            tables=detected_tables,
            document_blocks=document_blocks,
            raw_full_text=raw_full_text,
            audit_trail=audit_records,
            parallel_header=parallel_hdr,
            diagnostics=diagnostic_report,
        )
        if diagnostic_report is not None:
            diagnostic_report["reconstruction_integrity"] = build_reconstruction_diagnostic(
                result,
                source_boxes=diagnostic_input_boxes,
            )

        average_confidence = result.average_confidence
        confidence_summary = format_confidence(average_confidence)
        logger.info(
            f"Hybrid OCR completed: {result.num_lines} lines, "
            f"{len(result.tables)} tables detected, "
            f"avg_conf={confidence_summary}, "
            f"elapsed={elapsed:.2f}s"
        )
        check_cancelled()
        report_progress("Đang hoàn tất kết quả OCR …")
        report_stage("layout", "end")
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
                confidence_label = f"{box.confidence:.0%}" if box.confidence is not None else "?"
                label = f"{box.text} ({confidence_label})"
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
    "format_confidence",
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
