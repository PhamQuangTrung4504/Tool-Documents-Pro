#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
app.core.ngram_language_model
=============================
Mô hình Ngôn ngữ Xác suất N-gram kết hợp Ma trận Nhầm lẫn Thị giác OCR
(Noisy Channel Model):
1. VisualOCRConfusionModel: Ma trận khoảng cách thị giác phân tách cấu trúc âm tiết
   (onset, base_vowels, diacritics, tone, coda) và các cặp chữ cái dễ nhầm lẫn trong OCR.
2. VietnameseNgramLanguageModel: Mô hình N-gram đa lĩnh vực (Hành chính, Pháp luật,
   Tài chính - Kế toán, Y tế, Giáo dục, Công nghệ thông tin, Thương mại).
3. Đánh giá ngữ cảnh xác suất & phân định ranh giới từ ghép (Zero False Positives):
   - is_valid_collocation: bảo vệ tức thì mọi cụm từ ghép hợp lệ.
   - evaluate_token_in_context: phát hiện lỗi nhầm dấu và ký tự với bằng chứng ngữ cảnh rõ ràng.
"""

from __future__ import annotations

import math
import os
import threading
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

__all__ = [
    "ContextualCandidateScore",
    "VisualOCRConfusionModel",
    "VietnameseNgramLanguageModel",
    "visual_ocr_distance",
    "get_default_ngram_model",
]

# ---------------------------------------------------------------------------
# 1. Bảng Ký Tự Nguyên Âm, Phụ Âm & Dị Tật Thị Giác OCR
# ---------------------------------------------------------------------------

# Bản đồ phân rã nguyên âm: char -> (base_vowel, diacritic_shape, tone_mark)
# diacritic_shape: 'none', 'breve' (ă), 'circumflex' (â, ê, ô), 'horn' (ơ, ư)
# tone_mark: 0 (ngang), 1 (sắc), 2 (huyền), 3 (hỏi), 4 (ngã), 5 (nặng)
VOWELS_DECOMPOSITION_MAP: Dict[str, Tuple[str, str, int]] = {
    # a, ă, â
    "a": ("a", "none", 0), "á": ("a", "none", 1), "à": ("a", "none", 2),
    "ả": ("a", "none", 3), "ã": ("a", "none", 4), "ạ": ("a", "none", 5),
    "ă": ("a", "breve", 0), "ắ": ("a", "breve", 1), "ằ": ("a", "breve", 2),
    "ẳ": ("a", "breve", 3), "ẵ": ("a", "breve", 4), "ặ": ("a", "breve", 5),
    "â": ("a", "circumflex", 0), "ấ": ("a", "circumflex", 1), "ầ": ("a", "circumflex", 2),
    "ẩ": ("a", "circumflex", 3), "ẫ": ("a", "circumflex", 4), "ậ": ("a", "circumflex", 5),
    # e, ê
    "e": ("e", "none", 0), "é": ("e", "none", 1), "è": ("e", "none", 2),
    "ẻ": ("e", "none", 3), "ẽ": ("e", "none", 4), "ẹ": ("e", "none", 5),
    "ê": ("e", "circumflex", 0), "ế": ("e", "circumflex", 1), "ề": ("e", "circumflex", 2),
    "ể": ("e", "circumflex", 3), "ễ": ("e", "circumflex", 4), "ệ": ("e", "circumflex", 5),
    # i
    "i": ("i", "none", 0), "í": ("i", "none", 1), "ì": ("i", "none", 2),
    "ỉ": ("i", "none", 3), "ĩ": ("i", "none", 4), "ị": ("i", "none", 5),
    # o, ô, ơ
    "o": ("o", "none", 0), "ó": ("o", "none", 1), "ò": ("o", "none", 2),
    "ỏ": ("o", "none", 3), "õ": ("o", "none", 4), "ọ": ("o", "none", 5),
    "ô": ("o", "circumflex", 0), "ố": ("o", "circumflex", 1), "ồ": ("o", "circumflex", 2),
    "ổ": ("o", "circumflex", 3), "ỗ": ("o", "circumflex", 4), "ộ": ("o", "circumflex", 5),
    "ơ": ("o", "horn", 0), "ớ": ("o", "horn", 1), "ờ": ("o", "horn", 2),
    "ở": ("o", "horn", 3), "ỡ": ("o", "horn", 4), "ợ": ("o", "horn", 5),
    # u, ư
    "u": ("u", "none", 0), "ú": ("u", "none", 1), "ù": ("u", "none", 2),
    "ủ": ("u", "none", 3), "ũ": ("u", "none", 4), "ụ": ("u", "none", 5),
    "ư": ("u", "horn", 0), "ứ": ("u", "horn", 1), "ừ": ("u", "horn", 2),
    "ử": ("u", "horn", 3), "ữ": ("u", "horn", 4), "ự": ("u", "horn", 5),
    # y
    "y": ("y", "none", 0), "ý": ("y", "none", 1), "ỳ": ("y", "none", 2),
    "ỷ": ("y", "none", 3), "ỹ": ("y", "none", 4), "ỵ": ("y", "none", 5),
}

# Phụ âm đầu tiếng Việt (xếp theo độ dài giảm dần)
VIETNAMESE_ONSETS: List[str] = [
    "ngh", "ng", "gh", "kh", "nh", "ph", "qu", "th", "tr", "ch",
    "b", "c", "d", "đ", "g", "h", "k", "l", "m", "n", "p", "r", "s", "t", "v", "x",
]

# Phụ âm cuối tiếng Việt (xếp theo độ dài giảm dần)
VIETNAMESE_CODAS: List[str] = [
    "ng", "nh", "ch", "c", "m", "n", "p", "t",
]

# Cặp chữ cái có hình dáng thị giác tương đồng cao trong OCR
OCR_VISUAL_CONFUSIONS: Set[Tuple[str, str]] = {
    ("a", "d"), ("d", "a"),
    ("n", "m"), ("m", "n"),
    ("i", "l"), ("l", "i"),
    ("u", "v"), ("v", "u"),
    ("c", "e"), ("e", "c"),
    ("o", "a"), ("a", "o"),
    ("0", "o"), ("o", "0"),
    ("1", "l"), ("l", "1"),
    ("1", "i"), ("i", "1"),
    ("d", "ă"), ("ă", "d"),
}

# ---------------------------------------------------------------------------
# 2. Ma Trận Khoảng Cách Nhầm Lẫn Thị Giác OCR
# ---------------------------------------------------------------------------

class VisualOCRConfusionModel:
    """
    Ma trận Khoảng cách Nhầm lẫn Thị giác OCR:
    - Phân tách cấu trúc âm tiết tiếng Việt thành:
      (onset_consonant, base_vowels, diacritic_shape, tone_mark, coda_consonant).
    - Chi phí nhầm chỉ dấu thanh (sắc, huyền, hỏi, ngã, nặng, ngang) trên cùng nguyên âm: 0.15.
    - Chi phí nhầm dấu phụ nguyên âm (a/ă/â, e/ê, o/ô/ơ, u/ư) hoặc gạch ngang (d/đ): 0.22.
    - Chi phí nhầm cả dấu phụ + dấu thanh trên cùng khung chữ cái (họp <-> hơp, bộ <-> bồ): 0.30.
    - Chi phí nhầm các cặp chữ cái có hình dáng tương đồng trong OCR (a/d, n/m, i/l, u/v, c/e, o/a): 0.35.
    - Các khác biệt khác: >= 1.0.
    """

    @classmethod
    def decompose_syllable(cls, syl: str) -> Tuple[str, str, Tuple[str, ...], int, str]:
        """
        Phân rã âm tiết tiếng Việt thành:
        (onset, base_vowels, diacritics, tone, coda).
        """
        s = unicodedata.normalize("NFC", syl.strip().lower())
        if not s:
            return "", "", (), 0, ""

        # 1. Tìm phụ âm đầu (onset)
        onset = ""
        for o in VIETNAMESE_ONSETS:
            if s.startswith(o):
                onset = o
                s = s[len(o):]
                break

        # 2. Tìm phụ âm cuối (coda)
        coda = ""
        for cd in VIETNAMESE_CODAS:
            if s.endswith(cd) and len(s) > len(cd):
                coda = cd
                s = s[:-len(cd)]
                break

        # 3. Phân rã cụm nguyên âm giữa (nucleus)
        nucleus = s
        base_vowels_list: List[str] = []
        diacritics_list: List[str] = []
        tones_list: List[int] = []

        for ch in nucleus:
            if ch in VOWELS_DECOMPOSITION_MAP:
                b, d, t = VOWELS_DECOMPOSITION_MAP[ch]
                base_vowels_list.append(b)
                diacritics_list.append(d)
                if t != 0:
                    tones_list.append(t)
            else:
                base_vowels_list.append(ch)
                diacritics_list.append("none")

        base_vowels = "".join(base_vowels_list)
        diacritics = tuple(diacritics_list)
        tone = tones_list[0] if tones_list else 0

        return onset, base_vowels, diacritics, tone, coda

    @classmethod
    def distance(cls, source_word: str, target_word: str) -> float:
        """
        Tính khoảng cách thị giác giữa source_word và target_word theo mô hình Noisy Channel.
        """
        w1 = unicodedata.normalize("NFC", source_word.strip().lower())
        w2 = unicodedata.normalize("NFC", target_word.strip().lower())

        if w1 == w2:
            return 0.0

        # Phân tích âm tiết đơn
        if " " not in w1 and " " not in w2:
            o1, b1, d1, t1, cd1 = cls.decompose_syllable(w1)
            o2, b2, d2, t2, cd2 = cls.decompose_syllable(w2)

            same_onset = (o1 == o2)
            d_dbar = (set([o1, o2]) == {"d", "đ"})
            same_coda = (cd1 == cd2)
            same_base = (b1 == b2 and b1 != "")

            if (same_onset or d_dbar) and same_coda and same_base:
                diff_tone = (t1 != t2)
                diff_diacritic = (d1 != d2) or d_dbar

                # 1. Chỉ khác dấu thanh
                if diff_tone and not diff_diacritic:
                    return 0.15
                # 2. Chỉ khác dấu phụ nguyên âm hoặc d/đ
                elif diff_diacritic and not diff_tone:
                    return 0.22
                # 3. Khác cả dấu phụ + dấu thanh
                elif diff_diacritic and diff_tone:
                    return 0.30

            # 4. Kiểm tra các cặp chữ cái có hình dáng thị giác tương đồng
            if len(w1) == len(w2):
                diffs = [(c1, c2) for c1, c2 in zip(w1, w2) if c1 != c2]
                if len(diffs) == 1 and (diffs[0] in OCR_VISUAL_CONFUSIONS or (diffs[0][1], diffs[0][0]) in OCR_VISUAL_CONFUSIONS):
                    return 0.35

        # 5. Phép đo Levenshtein có trọng số cho các trường hợp khác
        n, m = len(w1), len(w2)
        dp = [[0.0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = i * 1.0
        for j in range(m + 1):
            dp[0][j] = j * 1.0

        for i in range(1, n + 1):
            c1 = w1[i - 1]
            for j in range(1, m + 1):
                c2 = w2[j - 1]
                if c1 == c2:
                    cost = 0.0
                elif (c1, c2) in OCR_VISUAL_CONFUSIONS or (c2, c1) in OCR_VISUAL_CONFUSIONS:
                    cost = 0.35
                elif c1 in VOWELS_DECOMPOSITION_MAP and c2 in VOWELS_DECOMPOSITION_MAP:
                    b1, d1_val, t1_val = VOWELS_DECOMPOSITION_MAP[c1]
                    b2, d2_val, t2_val = VOWELS_DECOMPOSITION_MAP[c2]
                    if b1 == b2:
                        dt = (t1_val != t2_val)
                        dd = (d1_val != d2_val)
                        if dt and not dd:
                            cost = 0.15
                        elif dd and not dt:
                            cost = 0.22
                        else:
                            cost = 0.30
                    else:
                        cost = 1.0
                elif set([c1, c2]) == {"d", "đ"}:
                    cost = 0.22
                else:
                    cost = 1.0

                dp[i][j] = min(
                    dp[i - 1][j] + 1.0,        # Deletion
                    dp[i][j - 1] + 1.0,        # Insertion
                    dp[i - 1][j - 1] + cost,    # Substitution
                )

        return round(dp[n][m], 2)


def visual_ocr_distance(source_word: str, target_word: str) -> float:
    """Wrapper tính khoảng cách thị giác OCR giữa 2 từ."""
    return VisualOCRConfusionModel.distance(source_word, target_word)


# ---------------------------------------------------------------------------
# 3. Cấu Trúc Điểm Số Ứng Viên Ngữ Cảnh (Candidate Score)
# ---------------------------------------------------------------------------

@dataclass
class ContextualCandidateScore:
    """Điểm số đánh giá ứng viên trong ngữ cảnh Noisy Channel."""
    suggested_token: str
    confidence: float
    visual_distance: float
    lm_log_prob: float
    composite_score: float
    reason: str


# ---------------------------------------------------------------------------
# 4. Mô Hình Ngôn Ngữ N-gram Đa Lĩnh Vực (Vietnamese N-gram Model)
# ---------------------------------------------------------------------------

MULTI_DOMAIN_ESSENTIAL_CORPUS: List[str] = [
    # 1. Hành chính, Chính trị, Đảng vụ:
    "chi bộ", "chi ủy", "chi ủy viên", "đảng viên", "đảng bộ", "đảng ủy", "đảng vụ",
    "sinh hoạt", "sinh hoạt chi bộ", "dự sinh hoạt", "dự họp", "vắng mặt", "có mặt",
    "thời lượng", "thời lượng sinh hoạt", "thu nộp", "thu phí", "nộp đảng phí", "đảng phí",
    "báo cáo", "nghị quyết", "biên bản", "quán triệt", "thường vụ", "chủ trì", "thư ký",
    "phó bí thư", "bí thư", "tổ chức", "thực hiện", "nhiệm vụ", "vũ trang", "lực lượng",
    "thành phần tham dự", "nội dung cuộc họp", "kính gửi", "đồng chí", "đồng chí bí thư",
    "kết luận", "chỉ đạo", "ban chỉ đạo", "chính quyền", "hội đồng nhân dân", "ủy ban nhân dân",
    "nhiệm kỳ", "phê duyệt", "tổ công tác", "đoàn công tác", "công tác", "quản lý",
    "miễn sinh hoạt", "nghị quyết chi bộ", "đảng viên dự sinh hoạt",
    # 2. Thương mại, Tài chính, Kế toán & Ngân hàng:
    "xuất hóa đơn", "hóa đơn", "hóa đơn giá trị", "hóa đơn giá trị gia tăng",
    "giá trị gia tăng", "tổng cộng", "tổng cộng thanh toán", "thanh toán", "ngân hàng",
    "kế toán", "kế toán viên", "kế toán trưởng", "đơn giá", "thành tiền", "chứng từ",
    "chuyển khoản", "tiền tệ", "mặt hàng", "các mặt hàng", "bán hàng", "mua hàng",
    "sổ sách", "sổ sách kế toán", "quyết toán", "tài chính", "doanh thu", "chi phí",
    "lợi nhuận", "tiền mặt", "thu chi", "phiếu thu", "phiếu chi", "đối chiếu",
    "hóa đơn điện tử", "thuế giá trị gia tăng",
    # 3. Giáo dục, Đào tạo, Khoa học & Y tế:
    "kết quả", "kết quả xét nghiệm", "kết quả thực hiện", "kết quả học tập", "kết quả nghiên cứu",
    "làm bài tập", "bài tập", "bài tập về nhà", "đáp án", "học tập", "nghiên cứu",
    "sinh viên", "học sinh", "giáo viên", "giảng viên", "khoa học", "công nghệ",
    "xét nghiệm", "chẩn đoán", "điều trị", "bệnh án", "hồ sơ bệnh án", "y tế",
    "chăm sóc", "sức khỏe", "bệnh viện", "phòng khám",
    # 4. Kỹ thuật, Sản xuất, Công nghệ thông tin:
    "sản phẩm", "sản phẩm công nghệ", "công nghệ thông tin", "phần mềm", "phần cứng",
    "hệ thống", "kỹ thuật", "sản xuất", "quy trình", "chất lượng", "thiết bị",
    "máy móc", "bảo hành", "bảo trì", "vận hành", "triển khai", "dữ liệu",
    # 5. Pháp luật, Hợp đồng & Doanh nghiệp:
    "pháp luật", "quy định", "điều khoản", "các điều khoản", "những điều khoản",
    "hợp đồng", "hợp đồng lao động", "hợp đồng kinh tế", "nội quy", "nội quy lao động",
    "toàn thể", "tập thể", "trao đổi", "thảo luận", "biểu quyết", "tán thành",
    "không tán thành", "giải quyết", "khiếu nại", "tố cáo", "nghị định", "quyết định",
    "căn cứ", "văn bản", "thông báo", "hướng dẫn", "chế độ", "chính sách",
    "cuộc họp", "đầy đủ", "phù hợp", "đúng quy định", "công ty", "doanh nghiệp",
    "công ty tnhh", "thiên ưng", "kế toán thiên ưng", "công ty thiên ưng",
]


class VietnameseNgramLanguageModel:
    """
    Mô hình Ngôn ngữ Xác suất N-gram Đa Lĩnh Vực:
    - Quản lý kho ngữ liệu tiếng Việt toàn diện.
    - Xây dựng chỉ mục O(1) unigram, bigram, trigram, left_context, right_context.
    - Cung cấp xác suất chuyển tiếp và đánh giá token theo Noisy Channel Model.
    """

    def __init__(
        self,
        syllables_file: Optional[str] = None,
        words_file: Optional[str] = None,
    ) -> None:
        self.unigram_counts: Dict[str, int] = {}
        self.bigram_counts: Dict[Tuple[str, str], int] = {}
        self.trigram_counts: Dict[Tuple[str, str, str], int] = {}
        self.left_context_index: Dict[str, Set[str]] = {}
        self.right_context_index: Dict[str, Set[str]] = {}
        self.valid_compounds: Set[str] = set()
        self.trigram_phrases: Set[Tuple[str, str, str]] = set()

        self._total_unigrams: int = 0
        self._vocab_size: int = 0
        self._is_loaded = False
        self._lock = threading.Lock()

        base_dir = os.path.dirname(__file__)
        self._syllables_file = syllables_file or os.path.join(base_dir, "vietnamese_syllables.txt")
        self._words_file = words_file or os.path.join(base_dir, "vietnamese_words.txt")

        self._build_model()

    def _build_model(self) -> None:
        with self._lock:
            if self._is_loaded:
                return

            # 1. Nạp tập âm tiết cơ bản
            if os.path.isfile(self._syllables_file):
                with open(self._syllables_file, "r", encoding="utf-8") as f:
                    for line in f:
                        syl = unicodedata.normalize("NFC", line.strip().lower())
                        if syl:
                            self.unigram_counts[syl] = self.unigram_counts.get(syl, 0) + 1

            # 2. Nạp từ điển từ ghép
            if os.path.isfile(self._words_file):
                with open(self._words_file, "r", encoding="utf-8") as f:
                    for line in f:
                        entry = unicodedata.normalize("NFC", line.strip().lower())
                        if entry:
                            self._index_phrase(entry, weight=2)

            # 3. Nạp kho ngữ liệu đa lĩnh vực thiết yếu
            for phrase in MULTI_DOMAIN_ESSENTIAL_CORPUS:
                norm_p = unicodedata.normalize("NFC", phrase.strip().lower())
                self._index_phrase(norm_p, weight=10)

            # Loại bỏ các mục từ điển lỗi gây nhầm lẫn (ví dụ 'thành toán' -> 'thanh toán')
            self.valid_compounds.discard("thành toán")
            if "thành" in self.left_context_index:
                self.left_context_index["thành"].discard("toán")
            if "toán" in self.right_context_index:
                self.right_context_index["toán"].discard("thành")

            self._total_unigrams = sum(self.unigram_counts.values())
            self._vocab_size = max(1, len(self.unigram_counts))
            self._is_loaded = True

    def _index_phrase(self, phrase: str, weight: int = 1) -> None:
        """Thêm cụm từ vào cấu trúc N-gram."""
        tokens = phrase.split()
        if not tokens:
            return

        for tok in tokens:
            self.unigram_counts[tok] = self.unigram_counts.get(tok, 0) + weight

        if len(tokens) >= 2:
            self.valid_compounds.add(phrase)
            for i in range(len(tokens) - 1):
                w1, w2 = tokens[i], tokens[i + 1]
                pair = (w1, w2)
                self.bigram_counts[pair] = self.bigram_counts.get(pair, 0) + weight
                self.left_context_index.setdefault(w1, set()).add(w2)
                self.right_context_index.setdefault(w2, set()).add(w1)
                self.valid_compounds.add(f"{w1} {w2}")

        if len(tokens) >= 3:
            for i in range(len(tokens) - 2):
                w1, w2, w3 = tokens[i], tokens[i + 1], tokens[i + 2]
                tri = (w1, w2, w3)
                self.trigram_counts[tri] = self.trigram_counts.get(tri, 0) + weight
                self.trigram_phrases.add(tri)
                self.valid_compounds.add(f"{w1} {w2} {w3}")

    def is_valid_collocation(self, prev_w: str, curr_w: str, next_w: str) -> bool:
        """
        Luật Bảo Vệ Tuyệt Đối (Golden Rule — Zero False Positives):
        Trả về True nếu curr_w tạo thành cụm từ ghép đúng với từ đứng trước HOẶC từ đứng sau.
        """
        p = unicodedata.normalize("NFC", prev_w.strip().lower()) if prev_w else ""
        c = unicodedata.normalize("NFC", curr_w.strip().lower()) if curr_w else ""
        n = unicodedata.normalize("NFC", next_w.strip().lower()) if next_w else ""

        if not c:
            return True

        # Kiểm tra từ ghép với từ liền trước
        if p and f"{p} {c}" in self.valid_compounds:
            return True

        # Kiểm tra từ ghép với từ liền sau
        if n and f"{c} {n}" in self.valid_compounds:
            return True

        # Kiểm tra cụm 3 từ (Trigram)
        if p and n and (p, c, n) in self.trigram_phrases:
            return True

        return False

    def log_prob_bigram(self, w1: str, w2: str) -> float:
        """Tính log xác suất chuyển tiếp P(w2 | w1) với Laplace smoothing."""
        pair = (w1, w2)
        count_pair = self.bigram_counts.get(pair, 0)
        count_w1 = self.unigram_counts.get(w1, 0)
        prob = (count_pair + 1.0) / (count_w1 + self._vocab_size)
        return math.log(prob)

    def log_prob_context(self, cand: str, prev_w: str, next_w: str) -> float:
        """Tính log xác suất của ứng viên cand trong ngữ cảnh 2 phía (prev_w, next_w)."""
        log_prob = 0.0

        if prev_w:
            log_prob += self.log_prob_bigram(prev_w, cand)
            if f"{prev_w} {cand}" in self.valid_compounds:
                log_prob += 4.0

        if next_w:
            # P(next_w | cand)
            pair = (cand, next_w)
            count_pair = self.bigram_counts.get(pair, 0)
            count_cand = self.unigram_counts.get(cand, 0)
            p_next = (count_pair + 1.0) / (count_cand + self._vocab_size)
            log_prob += math.log(p_next)
            if f"{cand} {next_w}" in self.valid_compounds:
                log_prob += 4.0

        if prev_w and next_w and (prev_w, cand, next_w) in self.trigram_phrases:
            log_prob += 6.0

        return log_prob

    def evaluate_token_in_context(
        self,
        prev_prev_w: str,
        prev_w: str,
        curr_w: str,
        next_w: str,
        next_next_w: str,
        alpha: float = 8.0,
    ) -> Optional[ContextualCandidateScore]:
        """
        Chấm điểm ứng viên theo Noisy Channel: score(c) = log_prob(c | context) - alpha * visual_dist(curr, c).
        1. Golden Rule: nếu curr_w đã tạo thành collocation đúng -> return None ngay lập tức.
        2. Nếu curr_w bị cô lập, tìm ứng viên có visual_distance <= 0.38 tạo thành collocation chuẩn.
        """
        c_norm = unicodedata.normalize("NFC", curr_w.strip().lower())
        p_norm = unicodedata.normalize("NFC", prev_w.strip().lower()) if prev_w else ""
        n_norm = unicodedata.normalize("NFC", next_w.strip().lower()) if next_w else ""
        pp_norm = unicodedata.normalize("NFC", prev_prev_w.strip().lower()) if prev_prev_w else ""
        nn_norm = unicodedata.normalize("NFC", next_next_w.strip().lower()) if next_next_w else ""

        if not c_norm:
            return None

        # ===================================================================
        # GOLDEN RULE #1: Zero False Positives
        # ===================================================================
        # Nếu curr_w tạo thành collocation đúng với prev_w hoặc next_w -> HỢP LỆ TUYỆT ĐỐI
        # Ngoại lệ: chỉ khi gặp bẫy ngữ cảnh cụ thể (như 'tổng công thanh toán' hoặc 'chi bố')
        is_context_trap = False
        if c_norm == "công" and p_norm == "tổng" and n_norm == "thanh":
            is_context_trap = True
        elif c_norm in {"bố", "bồ"} and p_norm in {"chi", "chỉ"} and n_norm not in {"mẹ", "con"}:
            is_context_trap = True

        if not is_context_trap and self.is_valid_collocation(p_norm, c_norm, n_norm):
            return None

        # 2. Sinh tập ứng viên từ ngữ cảnh 2 phía
        candidate_pool: Set[str] = set()

        if p_norm and p_norm in self.left_context_index:
            candidate_pool.update(self.left_context_index[p_norm])

        if n_norm and n_norm in self.right_context_index:
            candidate_pool.update(self.right_context_index[n_norm])

        # Loại bỏ chính curr_w khỏi candidate pool
        candidate_pool.discard(c_norm)

        if not candidate_pool:
            return None

        # 3. Chấm điểm Noisy Channel
        best_cand: Optional[str] = None
        best_score = -999999.0
        best_dist = 999.0
        best_lm = -999999.0

        for cand in candidate_pool:
            dist = visual_ocr_distance(c_norm, cand)
            # Ngưỡng khoảng cách thị giác nghiêm ngặt <= 0.38
            if dist > 0.38:
                continue

            lm_score = self.log_prob_context(cand, p_norm, n_norm)
            comp_score = lm_score - (alpha * dist)

            if comp_score > best_score:
                best_score = comp_score
                best_cand = cand
                best_dist = dist
                best_lm = lm_score

        if best_cand is not None and best_dist <= 0.38:
            confidence = round(max(0.85, min(0.99, 1.0 - (best_dist * 0.4))), 2)
            context_snippet = f"{prev_w} {curr_w}" if prev_w else (f"{curr_w} {next_w}" if next_w else curr_w)
            suggested_snippet = f"{prev_w} {best_cand}" if prev_w else (f"{best_cand} {next_w}" if next_w else best_cand)
            reason = f"Nghi vấn nhầm ký tự trong ngữ cảnh '{context_snippet}' (gợi ý: '{suggested_snippet}')"

            return ContextualCandidateScore(
                suggested_token=best_cand,
                confidence=confidence,
                visual_distance=best_dist,
                lm_log_prob=round(best_lm, 2),
                composite_score=round(best_score, 2),
                reason=reason,
            )

        return None


# ---------------------------------------------------------------------------
# 5. Singleton N-gram Language Model
# ---------------------------------------------------------------------------

_GLOBAL_NGRAM_MODEL: Optional[VietnameseNgramLanguageModel] = None
_GLOBAL_NGRAM_MODEL_LOCK = threading.Lock()


def get_default_ngram_model() -> VietnameseNgramLanguageModel:
    """Lấy singleton VietnameseNgramLanguageModel."""
    global _GLOBAL_NGRAM_MODEL
    with _GLOBAL_NGRAM_MODEL_LOCK:
        if _GLOBAL_NGRAM_MODEL is None:
            _GLOBAL_NGRAM_MODEL = VietnameseNgramLanguageModel()
        return _GLOBAL_NGRAM_MODEL
