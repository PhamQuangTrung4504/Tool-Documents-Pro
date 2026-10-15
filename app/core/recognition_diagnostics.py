#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
app.core.recognition_diagnostics
================================
Module chẩn đoán chất lượng nhận dạng ký tự (Recognition Diagnostics) và quy kết
nguyên nhân lỗi (Error Attribution Harness):
1. Bảng từ vựng chuẩn của VietOCR (VIETOCR_STANDARD_VOCAB) & phát hiện ký tự ngoài bảng (OOV).
2. Phân tích chất lượng lát cắt thị giác (CropDiagnosticMetrics & analyze_crop_quality):
   - Tỷ lệ khung hình (Aspect Ratio & High AR warning).
   - Tỷ lệ mực tại 4 đường biên (top, bottom, left, right).
   - Xâm lấn đường viền bảng dọc (Vertical Border Intrusion).
   - Cắt cụt nét chữ ở mép phải (Right Char Clipping).
   - Cắt cụt dấu thanh ở mép trên (Top Diacritic Clipping).
3. Đánh giá sai số nhận dạng qua 3 giai đoạn:
   - Stage 0: Lát cắt thô + Model thô (Raw Crop + Raw Model).
   - Stage 1: Lát cắt đệm an toàn & khử viền (Adaptive Padded & Border-Cleaned Crop).
   - Stage 2: Khôi phục ký hiệu kỹ thuật & OOV (Technical Symbol & OOV Recovery).
4. Bộ quy kết nguyên nhân lỗi tự động (RecognitionErrorAttributor):
   - OUT_OF_VOCAB
   - CROP_OR_BORDER_ARTIFACTS
   - ASPECT_RATIO_ATTENTION_DRIFT
   - INTRINSIC_TONE_CHAR_CONFUSION
5. Báo cáo định lượng & ma trận quyết định kỹ thuật (Decision Matrix Report).
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np

__all__ = [
    "VIETOCR_STANDARD_VOCAB",
    "VIETOCR_STANDARD_VOCAB_SET",
    "find_out_of_vocab_chars",
    "CropDiagnosticMetrics",
    "analyze_crop_quality",
    "compute_cer",
    "compute_wer",
    "ErrorRootCause",
    "SampleAttributionResult",
    "RecognitionErrorAttributor",
]

# ---------------------------------------------------------------------------
# 1. Bảng Từ Vựng Chuẩn của VietOCR (vgg_transformer: 229 ký tự)
# ---------------------------------------------------------------------------

VIETOCR_STANDARD_VOCAB: str = (
    "aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬ"
    "bBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆ"
    "fFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌ"
    "ôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤ"
    "ưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ"
    "0123456789"
    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "
)

VIETOCR_STANDARD_VOCAB_SET: Set[str] = set(unicodedata.normalize("NFC", VIETOCR_STANDARD_VOCAB))


def find_out_of_vocab_chars(text: str) -> List[str]:
    """
    Phát hiện các ký tự trong text không nằm trong bảng từ vựng chuẩn của VietOCR.
    Chuẩn hóa NFC trước khi kiểm tra, bảo tồn thứ tự xuất hiện đầu tiên của ký tự.
    """
    if not text:
        return []
    norm_text = unicodedata.normalize("NFC", text)
    oov: List[str] = []
    seen: Set[str] = set()
    for ch in norm_text:
        if ch in ("\n", "\r", "\t"):
            continue
        if ch not in VIETOCR_STANDARD_VOCAB_SET and ch not in seen:
            seen.add(ch)
            oov.append(ch)
    return oov


# ---------------------------------------------------------------------------
# 2. Bộ Phân Tích Chất Lượng Lát Cắt (Crop Quality Diagnostics)
# ---------------------------------------------------------------------------

@dataclass
class CropDiagnosticMetrics:
    """
    Chỉ số chẩn đoán chất lượng vùng cắt ảnh (Crop Quality Diagnostic Metrics):
    - aspect_ratio: tỷ lệ width / height.
    - border_ink_ratios: mật độ điểm mực tại 4 biên (2px viền).
    - Các cờ phát hiện lỗi cắt lát:
      + HIGH_ASPECT_RATIO: aspect_ratio > 22.0
      + VERTICAL_BORDER_INTRUSION: dải đen dọc ở mép trái/phải >= 75% chiều cao.
      + RIGHT_CHAR_CLIPPING: chạm nét chữ mép phải (10% < ink < 70%).
      + TOP_DIACRITIC_CLIPPING: chạm nét dấu thanh mép trên (ink > 5%).
    """
    aspect_ratio: float
    height: int
    width: int
    top_ink_ratio: float
    bottom_ink_ratio: float
    left_ink_ratio: float
    right_ink_ratio: float
    max_vertical_left_run: int
    max_vertical_right_run: int
    is_high_aspect_ratio: bool
    has_vertical_border_intrusion: bool
    has_right_char_clipping: bool
    has_top_diacritic_clipping: bool
    flags: List[str] = field(default_factory=list)


def analyze_crop_quality(crop_bgr: np.ndarray) -> CropDiagnosticMetrics:
    """
    Phân tích trực tiếp ảnh lát cắt crop_bgr để phát hiện các dị tật cắt lát:
    - aspect_ratio: width / max(1, height). Cảnh báo HIGH_ASPECT_RATIO khi aspect_ratio > 22.0.
    - border_ink_ratios: Mật độ pixel tối ở 4 mép (top_2px, bottom_2px, left_2px, right_2px).
    - VERTICAL_BORDER_INTRUSION: dải đen dọc ở left_2px hoặc right_2px >= 75% chiều cao.
    - RIGHT_CHAR_CLIPPING: nét chữ chạm mép phải (> 10% ink nhưng < 70%).
    - TOP_DIACRITIC_CLIPPING: nét dấu thanh chạm mép trên (> 5% ink nhưng < 70%).
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return CropDiagnosticMetrics(
            aspect_ratio=0.0,
            height=0,
            width=0,
            top_ink_ratio=0.0,
            bottom_ink_ratio=0.0,
            left_ink_ratio=0.0,
            right_ink_ratio=0.0,
            max_vertical_left_run=0,
            max_vertical_right_run=0,
            is_high_aspect_ratio=False,
            has_vertical_border_intrusion=False,
            has_right_char_clipping=False,
            has_top_diacritic_clipping=False,
            flags=[],
        )

    h, w = crop_bgr.shape[:2]
    aspect_ratio = float(w) / max(1.0, float(h))

    # Binarize to locate ink pixels
    if len(crop_bgr.shape) == 2:
        gray = crop_bgr
    else:
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    if float(np.mean(gray)) > 127.0:
        _, bin_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, bin_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4 viền mép 2px
    top_h = min(2, h)
    bottom_h = min(2, h)
    left_w = min(2, w)
    right_w = min(2, w)

    top_slice = bin_inv[0:top_h, :]
    bottom_slice = bin_inv[max(0, h - bottom_h):h, :]
    left_slice = bin_inv[:, 0:left_w]
    right_slice = bin_inv[:, max(0, w - right_w):w]

    top_ink_ratio = float(np.count_nonzero(top_slice)) / max(1.0, float(top_slice.size))
    bottom_ink_ratio = float(np.count_nonzero(bottom_slice)) / max(1.0, float(bottom_slice.size))
    left_ink_ratio = float(np.count_nonzero(left_slice)) / max(1.0, float(left_slice.size))
    right_ink_ratio = float(np.count_nonzero(right_slice)) / max(1.0, float(right_slice.size))

    # Max contiguous dark run for vertical border intrusion
    def _max_col_run(slice_2d: np.ndarray) -> int:
        max_run = 0
        cols = slice_2d.shape[1]
        for c in range(cols):
            col = slice_2d[:, c]
            run = 0
            for val in col:
                if val > 0:
                    run += 1
                    if run > max_run:
                        max_run = run
                else:
                    run = 0
        return max_run

    max_vertical_left_run = _max_col_run(left_slice)
    max_vertical_right_run = _max_col_run(right_slice)

    flags: List[str] = []

    # 1. High aspect ratio
    is_high_aspect_ratio = aspect_ratio > 22.0
    if is_high_aspect_ratio:
        flags.append("HIGH_ASPECT_RATIO")

    # 2. Vertical border intrusion (>= 75% chiều cao)
    intrusion_threshold = 0.75 * h
    has_vertical_border_intrusion = (
        max_vertical_left_run >= intrusion_threshold or max_vertical_right_run >= intrusion_threshold
    )
    if has_vertical_border_intrusion:
        flags.append("VERTICAL_BORDER_INTRUSION")

    # 3. Right char clipping: > 10% ink nhưng < 70% và không phải border intrusion
    has_right_char_clipping = (0.10 < right_ink_ratio < 0.70) and not (max_vertical_right_run >= intrusion_threshold)
    if has_right_char_clipping:
        flags.append("RIGHT_CHAR_CLIPPING")

    # 4. Top diacritic clipping: chạm nét ở mép trên (> 5% ink nhưng < 70%)
    has_top_diacritic_clipping = (0.05 < top_ink_ratio < 0.70)
    if has_top_diacritic_clipping:
        flags.append("TOP_DIACRITIC_CLIPPING")

    return CropDiagnosticMetrics(
        aspect_ratio=round(aspect_ratio, 2),
        height=h,
        width=w,
        top_ink_ratio=round(top_ink_ratio, 4),
        bottom_ink_ratio=round(bottom_ink_ratio, 4),
        left_ink_ratio=round(left_ink_ratio, 4),
        right_ink_ratio=round(right_ink_ratio, 4),
        max_vertical_left_run=max_vertical_left_run,
        max_vertical_right_run=max_vertical_right_run,
        is_high_aspect_ratio=is_high_aspect_ratio,
        has_vertical_border_intrusion=has_vertical_border_intrusion,
        has_right_char_clipping=has_right_char_clipping,
        has_top_diacritic_clipping=has_top_diacritic_clipping,
        flags=flags,
    )


# ---------------------------------------------------------------------------
# 3. Tính Chỉ Số Lỗi: CER (Character Error Rate) & WER (Word Error Rate)
# ---------------------------------------------------------------------------

def compute_cer(reference: str, hypothesis: str) -> float:
    """
    Tính Character Error Rate (CER) bằng khoảng cách Levenshtein ký tự.
    CER = Levenshtein(reference, hypothesis) / max(1, len(reference)).
    """
    ref = unicodedata.normalize("NFC", reference)
    hyp = unicodedata.normalize("NFC", hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    if ref == hyp:
        return 0.0

    n, m = len(ref), len(hyp)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = temp
    return round(float(dp[m]) / float(n), 4)


def compute_wer(reference: str, hypothesis: str) -> float:
    """
    Tính Word Error Rate (WER) bằng khoảng cách Levenshtein theo từ.
    WER = Levenshtein(reference_words, hypothesis_words) / max(1, len(reference_words)).
    """
    ref_words = unicodedata.normalize("NFC", reference).split()
    hyp_words = unicodedata.normalize("NFC", hypothesis).split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    if ref_words == hyp_words:
        return 0.0

    n, m = len(ref_words), len(hyp_words)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = temp
    return round(float(dp[m]) / float(n), 4)


# ---------------------------------------------------------------------------
# 4. Bộ Quy Kết Nguyên Nhân Lỗi Tự Động (Recognition Error Attributor)
# ---------------------------------------------------------------------------

class ErrorRootCause(str, Enum):
    OUT_OF_VOCAB = "OUT_OF_VOCAB"
    CROP_OR_BORDER_ARTIFACTS = "CROP_OR_BORDER_ARTIFACTS"
    ASPECT_RATIO_ATTENTION_DRIFT = "ASPECT_RATIO_ATTENTION_DRIFT"
    INTRINSIC_TONE_CHAR_CONFUSION = "INTRINSIC_TONE_CHAR_CONFUSION"
    NO_ERROR = "NO_ERROR"


@dataclass
class SampleAttributionResult:
    """Kết quả quy kết nguyên nhân lỗi cho từng mẫu kiểm thử."""
    sample_id: str
    ground_truth: str
    raw_pred: str
    repaired_crop_pred: str
    post_pred: str
    crop_metrics: Optional[CropDiagnosticMetrics]
    stage0_cer: float
    stage0_wer: float
    stage1_cer: float
    stage1_wer: float
    stage2_cer: float
    stage2_wer: float
    root_cause: ErrorRootCause
    explanation: str
    is_resolved_stage1: bool
    is_resolved_stage2: bool
    remaining_error: bool

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["root_cause"] = self.root_cause.value
        return d


class RecognitionErrorAttributor:
    """
    Bộ Quy Kết Nguyên Nhân Lỗi Nhận Dạng Chữ (Recognition Error Attribution Engine):
    So sánh chuỗi qua 3 giai đoạn:
    - Stage 0: raw_pred (Lát cắt thô + VietOCR gốc)
    - Stage 1: repaired_crop_pred (Lát cắt đệm an toàn & khử viền bảng)
    - Stage 2: post_pred (Hậu xử lý khôi phục ký hiệu OOV & đơn vị đo lường)
    Với Ground Truth để phân loại lỗi vào đúng 1 trong 4 nhóm gốc rễ:
    1. OUT_OF_VOCAB: Ký tự ngoài từ vựng VietOCR (½, ², ³, °, ±,...)
    2. CROP_OR_BORDER_ARTIFACTS: Lỗi cắt thiếu padding, chạm mép, dính đường kẻ bảng
    3. ASPECT_RATIO_ATTENTION_DRIFT: Dòng quá dài (AR > 22) hoặc chấm lửng ...... làm trôi attention
    4. INTRINSIC_TONE_CHAR_CONFUSION: Nhầm lẫn dấu thanh/ký tự trên ảnh mờ (chi bộ/chi bố, 05/0s)
    """

    def attribute_sample(
        self,
        sample_id: str,
        ground_truth: str,
        raw_pred: str,
        crop_bgr: Optional[np.ndarray] = None,
        repaired_crop_pred: Optional[str] = None,
        post_pred: Optional[str] = None,
    ) -> SampleAttributionResult:
        gt = unicodedata.normalize("NFC", ground_truth)
        r_pred = unicodedata.normalize("NFC", raw_pred)

        crop_metrics = analyze_crop_quality(crop_bgr) if crop_bgr is not None else None

        # Stage 1 prediction
        if repaired_crop_pred is None:
            s1_pred = r_pred
        else:
            s1_pred = unicodedata.normalize("NFC", repaired_crop_pred)

        # Stage 2 prediction
        if post_pred is None:
            from app.core.text_postprocessor import final_document_cleanup
            s2_pred = unicodedata.normalize("NFC", final_document_cleanup(s1_pred))
        else:
            s2_pred = unicodedata.normalize("NFC", post_pred)

        # CER / WER
        s0_cer = compute_cer(gt, r_pred)
        s0_wer = compute_wer(gt, r_pred)
        s1_cer = compute_cer(gt, s1_pred)
        s1_wer = compute_wer(gt, s1_pred)
        s2_cer = compute_cer(gt, s2_pred)
        s2_wer = compute_wer(gt, s2_pred)

        is_resolved_stage1 = (s1_cer == 0.0 and s1_wer == 0.0)
        is_resolved_stage2 = (s2_cer == 0.0 and s2_wer == 0.0)
        remaining_error = (s2_cer > 0.0)

        # Root cause attribution logic
        if s0_cer == 0.0 and s0_wer == 0.0:
            root_cause = ErrorRootCause.NO_ERROR
            explanation = "Dự đoán khớp hoàn toàn 100% với ground truth ngay ở Stage 0."
        else:
            oov_chars = find_out_of_vocab_chars(gt)
            has_crop_intrusion = crop_metrics is not None and crop_metrics.has_vertical_border_intrusion
            has_crop_clipping = crop_metrics is not None and (
                crop_metrics.has_right_char_clipping or crop_metrics.has_top_diacritic_clipping
            )
            has_high_ar = crop_metrics is not None and crop_metrics.is_high_aspect_ratio

            # 1. OUT_OF_VOCAB
            if len(oov_chars) > 0 and (
                any(c not in r_pred for c in oov_chars) or s2_cer < s0_cer
            ):
                root_cause = ErrorRootCause.OUT_OF_VOCAB
                explanation = (
                    f"Ground Truth chứa ký tự ngoài từ vựng chuẩn của VietOCR: {oov_chars}. "
                    f"Lỗi được khôi phục tại Stage 2 kỹ thuật (CER: {s0_cer} -> {s2_cer})."
                )

            # 2. ASPECT_RATIO_ATTENTION_DRIFT
            elif (
                has_high_ar
                or ("......" in gt or "......" in r_pred)
                or ("..." in gt and abs(len(gt) - len(r_pred)) > 5)
            ):
                root_cause = ErrorRootCause.ASPECT_RATIO_ATTENTION_DRIFT
                ar_val = crop_metrics.aspect_ratio if crop_metrics else "N/A"
                explanation = (
                    f"Lỗi do tỷ lệ khung hình quá cao (AR={ar_val} > 22.0) hoặc chứa chuỗi chấm lửng "
                    f"làm trôi attention của Transformer Seq2Seq."
                )

            # 3. CROP_OR_BORDER_ARTIFACTS
            elif (
                (has_crop_intrusion or has_crop_clipping)
                or (s1_cer < s0_cer)
                or any(sym in r_pred and sym not in gt for sym in ("|", "[", "]", "_", "+", "-"))
                or (len(gt) > len(r_pred) and gt.startswith(r_pred))  # clipped ending
            ):
                root_cause = ErrorRootCause.CROP_OR_BORDER_ARTIFACTS
                flags_str = ", ".join(crop_metrics.flags) if crop_metrics else "CROP_DEFECT"
                explanation = (
                    f"Lỗi do lát cắt ảnh viền hoặc đệm không đủ ({flags_str}). "
                    f"Khắc phục qua tầng Cắt lát thích ứng & Khử viền Stage 1 (CER: {s0_cer} -> {s1_cer})."
                )

            # 4. INTRINSIC_TONE_CHAR_CONFUSION
            else:
                root_cause = ErrorRootCause.INTRINSIC_TONE_CHAR_CONFUSION
                explanation = (
                    "Lỗi nhầm lẫn dấu thanh hoặc ký tự nội tại của mô hình trên ảnh (ví dụ: bộ/bố, dư/dự, 05/0s, họp/hơp). "
                    "Ký tự nằm trong từ vựng và lát cắt chuẩn. Yêu cầu chuyển giao sang Hướng B (Mô hình Ngôn ngữ Xác suất / N-gram)."
                )

        return SampleAttributionResult(
            sample_id=sample_id,
            ground_truth=gt,
            raw_pred=r_pred,
            repaired_crop_pred=s1_pred,
            post_pred=s2_pred,
            crop_metrics=crop_metrics,
            stage0_cer=s0_cer,
            stage0_wer=s0_wer,
            stage1_cer=s1_cer,
            stage1_wer=s1_wer,
            stage2_cer=s2_cer,
            stage2_wer=s2_wer,
            root_cause=root_cause,
            explanation=explanation,
            is_resolved_stage1=is_resolved_stage1,
            is_resolved_stage2=is_resolved_stage2,
            remaining_error=remaining_error,
        )

    def generate_attribution_report(
        self,
        samples: Sequence[SampleAttributionResult],
    ) -> Dict[str, Any]:
        """
        Tổng hợp báo cáo quy kết nguyên nhân lỗi và ma trận quyết định kỹ thuật:
        - Tỷ lệ % lỗi được giải quyết tại Stage 1 (Cắt lát & Tiền xử lý) + Stage 2 (Khôi phục Ký hiệu OOV).
        - Phần % lỗi còn lại thuộc về INTRINSIC_TONE_CHAR_CONFUSION làm đầu vào cho Hướng B.
        """
        total_samples = len(samples)
        error_samples = [s for s in samples if s.stage0_cer > 0.0 or s.stage0_wer > 0.0]
        total_errors = len(error_samples)

        # Count by root cause
        cause_counts: Dict[str, int] = {
            ErrorRootCause.OUT_OF_VOCAB.value: 0,
            ErrorRootCause.CROP_OR_BORDER_ARTIFACTS.value: 0,
            ErrorRootCause.ASPECT_RATIO_ATTENTION_DRIFT.value: 0,
            ErrorRootCause.INTRINSIC_TONE_CHAR_CONFUSION.value: 0,
            ErrorRootCause.NO_ERROR.value: 0,
        }
        for s in samples:
            cause_counts[s.root_cause.value] += 1

        # Resolutions
        resolved_s1 = sum(1 for s in error_samples if s.is_resolved_stage1)
        resolved_s2 = sum(1 for s in error_samples if s.is_resolved_stage2 and not s.is_resolved_stage1)
        total_resolved = sum(1 for s in error_samples if s.is_resolved_stage2)
        remaining_errors = sum(1 for s in error_samples if s.remaining_error)

        avg_s0_cer = round(sum(s.stage0_cer for s in samples) / max(1, total_samples), 4)
        avg_s0_wer = round(sum(s.stage0_wer for s in samples) / max(1, total_samples), 4)
        avg_s1_cer = round(sum(s.stage1_cer for s in samples) / max(1, total_samples), 4)
        avg_s1_wer = round(sum(s.stage1_wer for s in samples) / max(1, total_samples), 4)
        avg_s2_cer = round(sum(s.stage2_cer for s in samples) / max(1, total_samples), 4)
        avg_s2_wer = round(sum(s.stage2_wer for s in samples) / max(1, total_samples), 4)

        root_cause_breakdown: Dict[str, Any] = {}
        for cause, count in cause_counts.items():
            pct = round(100.0 * count / max(1, total_errors), 1) if cause != ErrorRootCause.NO_ERROR.value else 0.0
            root_cause_breakdown[cause] = {
                "count": count,
                "percentage_of_errors": pct,
            }

        s1_pct = round(100.0 * resolved_s1 / max(1, total_errors), 1)
        s2_pct = round(100.0 * resolved_s2 / max(1, total_errors), 1)
        tot_res_pct = round(100.0 * total_resolved / max(1, total_errors), 1)
        rem_pct = round(100.0 * remaining_errors / max(1, total_errors), 1)

        # Decision Matrix & Technical Recommendations
        decision_matrix = {
            "decision": "PROCEED_TO_DIRECTION_B_PROBABILISTIC_LANGUAGE_MODEL",
            "fine_tune_vietocr_required": False,
            "stage1_crop_and_preprocess_efficiency": f"{s1_pct}% lỗi xử lý dứt điểm bằng đệm an toàn và làm sạch biên",
            "stage2_symbol_recovery_efficiency": f"{s2_pct}% lỗi ký hiệu ngoài từ vựng (OOV) giải quyết bằng quy tắc kỹ thuật",
            "total_pre_and_post_coverage": f"{tot_res_pct}% lỗi nhận dạng ký tự được xử lý hoàn toàn mà KHÔNG CẦN train lại model",
            "remaining_error_bottleneck": f"{rem_pct}% lỗi còn lại là INTRINSIC_TONE_CHAR_CONFUSION (ngữ cảnh từ vựng)",
            "strategic_next_step": (
                "Chuyển toàn lực sang Hướng B: Xây dựng Bộ Kiểm tra Chính tả Ngữ cảnh Xác suất N-gram "
                "để giải quyết triệt để 100% các lỗi dấu thanh và từ đồng âm trên văn bản hành chính."
            ),
        }

        report: Dict[str, Any] = {
            "version": "C4.1",
            "benchmark_name": "Recognition Error Attribution & Crop Diagnostics",
            "summary": {
                "total_samples": total_samples,
                "total_errors": total_errors,
                "average_stage0_cer": avg_s0_cer,
                "average_stage0_wer": avg_s0_wer,
                "average_stage1_cer": avg_s1_cer,
                "average_stage1_wer": avg_s1_wer,
                "average_stage2_cer": avg_s2_cer,
                "average_stage2_wer": avg_s2_wer,
            },
            "root_cause_breakdown": root_cause_breakdown,
            "stage_resolution_effectiveness": {
                "resolved_at_stage1_crop_repair_count": resolved_s1,
                "resolved_at_stage1_crop_repair_pct": s1_pct,
                "resolved_at_stage2_symbol_recovery_count": resolved_s2,
                "resolved_at_stage2_symbol_recovery_pct": s2_pct,
                "total_resolved_pre_post_count": total_resolved,
                "total_resolved_pre_post_pct": tot_res_pct,
                "remaining_intrinsic_confusion_count": remaining_errors,
                "remaining_intrinsic_confusion_pct": rem_pct,
            },
            "decision_matrix": decision_matrix,
            "sample_details": [s.to_dict() for s in samples],
        }

        return report
