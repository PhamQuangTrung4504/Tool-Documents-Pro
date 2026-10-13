#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
app.core.spell_corrector
========================
Module kiểm tra chính tả tiếng Việt theo ngữ cảnh và sinh gợi ý sửa tương tác
cho ứng dụng OCR Desktop.

NGUYÊN TẮC CỐT LÕI:
- CHỈ KIỂM TRA VÀ ĐÁNH DẤU, KHÔNG TỰ ĐỘNG SỬA, KHÔNG VIẾT LẠI VÀ KHÔNG THAY THẾ
  BẤT KỲ NỘI DUNG OCR NÀO.
- Bảo toàn tuyệt đối 100% nội dung gốc, thứ tự từ, dấu câu, khoảng trắng,
  xuống dòng và thụt lề khi người dùng chưa chọn gợi ý.
- Chỉ thay thế token khi người dùng chủ động nhấp chuột phải và chấp nhận gợi ý sửa.

Tính năng:
1. Bộ thẩm định âm tiết tiếng Việt (VALID_SYLLABLES):
   - 7.700+ âm tiết chuẩn (bao gồm cả phong cách đặt dấu mới và truyền thống).
   - Bảo vệ toàn bộ các từ tiếng Việt hợp lệ (bô, thê, bộc, tai, bi,...).
2. Từ điển từ vựng đa âm tiết & cụm từ (VIETNAMESE_WORDS):
   - Hơn 72.500 từ đơn, từ ghép, thuật ngữ hành chính & chuyên ngành.
   - Quản lý tiền tố / hậu tố từ ghép phục vụ phân tích ngữ cảnh (Bigram/Trigram).
3. Phân tích ngữ cảnh N-gram phát hiện lỗi nhầm dấu / ký tự OCR:
   - Phát hiện các trường hợp âm tiết riêng lẻ có thể hợp lệ nhưng sai trong ngữ cảnh:
     ví dụ: 'kỹ su' -> gợi ý 'kỹ sư', 'lâu nơm' -> gợi ý 'lâu năm', 'chuyên nghiêp' -> gợi ý 'chuyên nghiệp'.
   - Chỉ đánh dấu khi có bằng chứng rõ ràng (khoảng cách weighted edit distance <= 0.7).
   - Không đánh dấu các cụm từ đúng chỉ vì chưa có trong từ điển.
4. Thuật toán sinh gợi ý sửa đa tiêu chí (get_suggestions):
   - Tối đa 5 gợi ý xếp theo độ ưu tiên: khoảng cách chỉnh sửa có trọng số OCR,
     ngữ cảnh từ ghép, tính hợp lệ âm tiết, tần suất từ vựng.
5. Bảo vệ thông minh (Whitelisting):
   - Số đếm, ngày tháng, thời gian, số liệu, tiền tệ, đơn vị đo.
   - Mã hiệu văn bản hành chính (170-KL/TW, 2485/QĐ-BVHTTDL, 63/2010/NĐ-CP).
   - Từ viết tắt in hoa (UBND, HĐND, QĐ, CPU, VIP, TW, BCĐTW,...).
   - URL, Email, đường dẫn.
   - Từ tiếng Anh chuẩn / thuật ngữ IT (qua từ điển 82.000+ từ).
   - Tên riêng, địa danh, tên tổ chức viết hoa.
   - Ký tự mục lục (a, b, c, x, y,...).
"""

from __future__ import annotations

import functools
import importlib.util
import os
import re
import threading
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from loguru import logger

# ---------------------------------------------------------------------------
# Cấu Trúc Dữ Liệu Kết Quả Kiểm Tra
# ---------------------------------------------------------------------------


class SpellError(tuple):
    """
    Thông tin chi tiết của một vị trí nghi vấn chính tả hoặc lỗi OCR.
    Kế thừa trực tiếp từ tuple (start_idx, end_idx, token) để tương thích 100%
    với mọi kiểm tra isinstance(err, tuple), unpack (s, e, tok), slicing, và các bài kiểm thử cũ,
    đồng thời cung cấp các thuộc tính nâng cao: error_type, confidence, suggestions, reason.
    """

    def __new__(
        cls,
        start_idx: int,
        end_idx: int,
        token: str,
        error_type: str = "invalid_syllable",
        confidence: float = 0.90,
        suggestions: Optional[List[str]] = None,
        reason: str = "",
    ):
        inst = super().__new__(cls, (start_idx, end_idx, token))
        inst.start_idx = start_idx
        inst.end_idx = end_idx
        inst.token = token
        inst.error_type = error_type
        inst.confidence = confidence
        inst.suggestions = list(suggestions) if suggestions else []
        inst.reason = reason
        return inst

    def __repr__(self) -> str:
        sugg_str = ", ".join(self.suggestions[:3])
        return (
            f"SpellError({self.start_idx}:{self.end_idx}, token={self.token!r}, "
            f"type={self.error_type}, suggs=[{sugg_str}])"
        )


# ---------------------------------------------------------------------------
# Hằng Số Ngữ Âm & Ký Tự Tiếng Việt
# ---------------------------------------------------------------------------

VIETNAMESE_DIACRITICS: Set[str] = set(
    "đĐăĂâÂêÊôÔơƠưƯ"
    "áÁàÀảẢãÃạẠ"
    "ắẮằẰẳẲẵẴặẶ"
    "ấẤầẦẩẨẫẪậẬ"
    "éÉèÈẻẺẽẼẹẸ"
    "ếẾềỀểỂễỄệỆ"
    "íÍìÌỉỈĩĨịỊ"
    "óÓòÒỏỎõÕọỌ"
    "ốỐồỒổỔỗỖộỘ"
    "ớỚờỜởỞỡỠợỢ"
    "úÚùÙủỦũŨụỤ"
    "ứỨừỪửỬữỮựỰ"
    "ýÝỳỲỷỶỹỸỵỴ"
)

# Phụ âm đầu tiếng Việt
ONSET_CONSONANTS = [
    "", "b", "c", "ch", "d", "đ", "g", "gh", "h", "k", "kh",
    "l", "m", "n", "ng", "ngh", "nh", "p", "ph", "qu", "r",
    "s", "t", "th", "tr", "v", "x",
]

# Bảng nhầm lẫn ký tự và dấu OCR thường gặp (weighted cost)
OCR_CONFUSIONS: Dict[Tuple[str, str], float] = {
    ("b", "đ"): 0.6, ("đ", "b"): 0.6,
    ("d", "đ"): 0.5, ("đ", "d"): 0.5,
    ("u", "ư"): 0.5, ("ư", "u"): 0.5,
    ("o", "ô"): 0.5, ("ô", "o"): 0.5,
    ("o", "ơ"): 0.5, ("ơ", "o"): 0.5,
    ("ơ", "ă"): 0.6, ("ă", "ơ"): 0.6,
    ("a", "ă"): 0.5, ("ă", "a"): 0.5,
    ("a", "â"): 0.5, ("â", "a"): 0.5,
    ("e", "ê"): 0.5, ("ê", "e"): 0.5,
    ("o", "0"): 0.6, ("0", "o"): 0.6,
    ("l", "1"): 0.6, ("1", "l"): 0.6,
    ("i", "1"): 0.6, ("1", "i"): 0.6,
}

# Bản đồ khử dấu để tính khoảng cách dấu thanh
DIACRITIC_PAIRS = {
    "a": "áàảãạăắằẳẵặâấầẩẫậ",
    "e": "éèẻẽẹêếềểễệ",
    "i": "íìỉĩị",
    "o": "óòỏõọôốồổỗộơớờởỡợ",
    "u": "úùủũụưứừửữự",
    "y": "ýỳỷỹỵ",
    "d": "đ",
}
BASE_MAP: Dict[str, str] = {}
for base_ch, variants in DIACRITIC_PAIRS.items():
    for v in variants:
        BASE_MAP[v] = base_ch

# Thuật ngữ chuyên ngành, công nghệ, game & từ mượn phổ biến
TECH_LOAN_WORDS: Set[str] = {
    "ocr", "app", "pdf", "gpu", "cpu", "ram", "ai", "api", "ui", "ux",
    "windows", "microsoft", "python", "online", "data", "tool", "pro",
    "format", "word", "excel", "scan", "scanner", "doc", "docx", "txt",
    "setup", "config", "user", "admin", "system", "server", "client",
    "internet", "email", "website", "web", "link", "id", "pass", "code",
    "file", "folder", "menu", "button", "dialog", "popup", "font", "auto",
    "software", "hardware", "input", "output", "text", "image", "video",
    "ok", "no", "yes", "vs", "vip", "pc", "laptop", "wifi", "bluetooth",
    "game", "gamer", "master", "backup", "reset", "smart", "digital",
    "cloud", "platform", "media", "audio", "driver", "page", "check",
    "port", "host", "domain", "view", "edit", "copy", "paste", "save",
    "load", "open", "close", "start", "stop", "time", "date", "year",
    "backend", "frontend", "fullstack", "database", "redis", "postgres",
    "postgresql", "sql", "nosql", "cache", "docker", "k8s", "devops",
    "git", "github", "gitlab", "hotline", "blog", "post", "review",
}

# Độ ưu tiên từ vựng phổ biến khi xếp hạng gợi ý
COMMON_WORD_PRIORITY: Dict[str, float] = {
    "được": 5.0, "trong": 4.5, "và": 4.5, "các": 4.0, "có": 4.0, "là": 4.0,
    "của": 4.0, "người": 4.0, "với": 3.8, "cho": 3.8, "năm": 4.0, "này": 3.5,
    "đã": 3.5, "khi": 3.5, "đáp": 3.5, "kiểm": 3.8, "soát": 3.8, "hoạch": 4.0,
    "thực": 4.0, "sư": 4.0, "nghiệp": 4.0, "hành": 3.8, "chính": 3.8,
    "túc": 3.5, "tục": 3.5, "thức": 3.5,
}

# Tập từ chức năng, giới từ, liên từ, loại từ và từ vựng phổ thông cơ bản của tiếng Việt
# (Bảo vệ tuyệt đối để không bị đánh dấu nhầm lẫn ngữ cảnh sai lệch)
VIETNAMESE_STOPWORDS: Set[str] = {
    # Giới từ / quan hệ từ
    "tại", "ở", "qua", "với", "cho", "của", "từ", "đến", "trong", "trên", "dưới",
    "vào", "ra", "bởi", "theo", "về", "cùng", "do", "như", "sang", "lên", "xuống",
    # Liên từ
    "và", "hoặc", "hay", "nhưng", "mà", "thì", "nếu", "vì", "tuy", "dù", "song",
    "nên", "thế", "vậy", "để", "nhằm", "hoặc_là",
    # Phó từ / lượng từ / hư từ
    "đã", "đang", "sẽ", "vừa", "mới", "đều", "cũng", "rất", "quá", "lắm", "nhất",
    "không", "chưa", "chẳng", "được", "bị", "phải", "có", "thể", "chỉ", "luôn",
    "ngay", "liền", "hết", "xong", "rồi", "lại", "hơn", "kém",
    # Đại từ / chỉ từ
    "này", "đó", "kia", "ấy", "nào", "gì", "ai", "đây", "nọ", "đâu", "sao",
    "các", "những", "mỗi", "mọi", "từng", "cả", "tất", "mình", "ta", "nó", "họ",
    # Danh từ chỉ đơn vị / loại từ hành chính thông dụng
    "tỉnh", "thành", "huyện", "thị", "xã", "phường", "thôn", "bản", "ấp", "khu",
}

PROTECTED_COMMON_WORDS: Set[str] = VIETNAMESE_STOPWORDS | {
    # Danh từ đơn vị / danh từ phổ thông cơ bản
    "số", "ngày", "tháng", "năm", "giờ", "phút", "giây", "lần", "câu", "từ",
    "người", "nhà", "nơi", "việc", "cái", "con", "chiếc", "bức", "tấm", "khối",
    # Tính từ cơ bản phổ biến
    "đúng", "sai", "tốt", "xấu", "lớn", "nhỏ", "nhiều", "ít", "cao", "thấp",
    "xa", "gần", "nhanh", "chậm", "mới", "cũ", "sớm", "muộn", "rõ", "sâu", "sát",
    # Động từ cơ bản phổ biến
    "đi", "làm", "ăn", "uống", "nói", "cười", "viết", "đọc", "xem", "nghe", "thấy",
    "biết", "hiểu", "nhớ", "quên", "mua", "bán", "trả", "nhận", "gửi", "lấy", "bỏ",
    "đặt", "mang", "cầm", "giữ", "tìm", "chọn", "dùng", "sống", "chết", "hỏi", "xin",
    "bảo", "gọi", "mời", "chờ", "đợi", "gặp", "kéo", "ghi", "nêu",
    # Số đếm
    "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín", "mười", "trăm", "nghìn", "triệu", "tỷ",
}

# Regex nhận diện các thành phần được bảo vệ (Whitelisting):
RE_PROTECTED = re.compile(
    r"(?i)"
    r"(https?://\S+|www\.\S+|ftp://\S+)"                               # URL
    r"|([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)"             # Email
    r"|(\b\d+([/.-]\d+)+\b)"                                          # Ngày tháng / số phân cách
    r"|(\b\d+([:,]\d+)+\b)"                                          # Giờ / số tiền
    r"|(\b\d+[a-zA-Z%]+\b)"                                          # Số + đơn vị (ví dụ: 2026a, 50kg, 100%)
    r"|(\b[0-9]+[/-][0-9a-zA-ZÀ-ỹ/-]+\b)"                            # Mã công văn có số đầu
    r"|(\b[a-zA-ZÀ-ỹ0-9]+[/-][0-9a-zA-ZÀ-ỹ/-]+\b)"                   # Mã công văn chữ + ký hiệu
)

# Regex tách từ tiếng Việt và Latin (bao gồm mọi dấu thanh dựng sẵn và tổ hợp)
RE_WORD = re.compile(r"[a-zA-Z\u00C0-\u024F\u0300-\u036F\u1EA0-\u1EF9]+")


# ---------------------------------------------------------------------------
# Tiện Ích Khoảng Cách Chỉnh Sửa Trọng Số
# ---------------------------------------------------------------------------


def remove_diacritics(s: str) -> str:
    """Loại bỏ dấu thanh và dấu mũ/râu, đưa về ký tự Latin cơ sở."""
    s_nfc = unicodedata.normalize("NFC", s)
    return "".join(BASE_MAP.get(c, c) for c in s_nfc)


def weighted_edit_distance(s1: str, s2: str) -> float:
    """
    Tính khoảng cách chỉnh sửa có trọng số giữa s1 và s2.
    - Cùng ký tự cơ sở, khác dấu thanh/mũ: cost 0.5.
    - Cặp ký tự OCR dễ nhầm lẫn (u/ư, o/ô, b/đ, ơ/ă): cost 0.5 - 0.6.
    - Ký tự gõ Telex sót ở đuôi (r, s, x, j, f): cost 0.4.
    - Các phép biến đổi thông thường: cost 1.0.
    """
    if s1 == s2:
        return 0.0

    n, m = len(s1), len(s2)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i * 1.0
    for j in range(m + 1):
        dp[0][j] = j * 1.0

    for i in range(1, n + 1):
        c1 = s1[i - 1]
        for j in range(1, m + 1):
            c2 = s2[j - 1]
            if c1 == c2:
                cost = 0.0
            elif (c1, c2) in OCR_CONFUSIONS:
                cost = OCR_CONFUSIONS[(c1, c2)]
            elif BASE_MAP.get(c1, c1) == BASE_MAP.get(c2, c2):
                cost = 0.5
            else:
                cost = 1.0

            # Xóa phím gõ Telex dư ở đuôi (kiêmr -> kiểm)
            del_cost = 1.0
            if i == n and c1 in "rsxfj":
                del_cost = 0.4

            dp[i][j] = min(
                dp[i - 1][j] + del_cost,      # Deletion
                dp[i][j - 1] + 1.0,           # Insertion
                dp[i - 1][j - 1] + cost,      # Substitution
            )

    return dp[n][m]


# ---------------------------------------------------------------------------
# Lớp Kiểm Tra Chính Tả & Sinh Gợi Ý
# ---------------------------------------------------------------------------


class VietnameseSpellChecker:
    """
    Bộ kiểm tra chính tả tiếng Việt kết hợp ngữ cảnh N-gram và sinh gợi ý sửa:
    1. VALID_SYLLABLES: 7.700+ âm tiết tiếng Việt chuẩn.
    2. VIETNAMESE_WORDS: 72.500+ từ vựng & từ ghép tiếng Việt.
    3. COMPOUND_PREFIXES / COMPOUND_SUFFIXES: Ánh xạ kết hợp từ ghép N-gram.
    4. ENGLISH_WORDS: 82.000+ từ tiếng Anh hỗ trợ văn bản hỗn hợp.
    """

    def __init__(
        self,
        syllables_file: Optional[str] = None,
        words_file: Optional[str] = None,
    ) -> None:
        self._valid_syllables: Set[str] = set()
        self._vietnamese_words: Set[str] = set()
        self._compound_prefixes: Dict[str, Set[str]] = {}
        self._compound_suffixes: Dict[str, Set[str]] = {}
        self._english_words: Set[str] = set()

        self._lock = threading.Lock()
        self._is_loaded = False

        base_dir = os.path.dirname(__file__)
        self._syllables_file = syllables_file or os.path.join(base_dir, "vietnamese_syllables.txt")
        self._words_file = words_file or os.path.join(base_dir, "vietnamese_words.txt")

        self._load_datasets()

    def _load_datasets(self) -> None:
        """Tải các tập dữ liệu từ vựng tiếng Việt và tiếng Anh."""
        with self._lock:
            if self._is_loaded:
                return

            # 1. Tải tập âm tiết tiếng Việt
            if os.path.isfile(self._syllables_file):
                try:
                    with open(self._syllables_file, "r", encoding="utf-8") as f:
                        for line in f:
                            s = unicodedata.normalize("NFC", line.strip().lower())
                            if s:
                                self._valid_syllables.add(s)
                    logger.debug(f"Loaded {len(self._valid_syllables)} syllables from {self._syllables_file}")
                except Exception as exc:
                    logger.error(f"Error loading syllables file: {exc}")

            # Mở rộng biến thể đặt dấu thanh truyền thống & mới (hòa/hoà, thủy/thuỷ,...)
            self._expand_tone_variations()

            # 2. Tải từ điển từ ghép tiếng Việt (VIETNAMESE_WORDS)
            if os.path.isfile(self._words_file):
                try:
                    with open(self._words_file, "r", encoding="utf-8") as f:
                        for line in f:
                            w = unicodedata.normalize("NFC", line.strip().lower())
                            if w:
                                self._vietnamese_words.add(w)
                                parts = w.split()
                                if len(parts) == 2:
                                    w1, w2 = parts
                                    self._compound_prefixes.setdefault(w1, set()).add(w2)
                                    self._compound_suffixes.setdefault(w2, set()).add(w1)
                    logger.debug(f"Loaded {len(self._vietnamese_words)} words from {self._words_file}")
                except Exception as exc:
                    logger.error(f"Error loading words file: {exc}")

            # Mở rộng biến thể đặt dấu thanh truyền thống & mới cho từ ghép (văn hoá <-> văn hóa,...)
            self._expand_word_tone_variations()

            # 3. Tải từ điển tiếng Anh từ symspellpy an toàn bằng importlib (không gây cảnh báo linter)
            try:
                spec = importlib.util.find_spec("symspellpy")
                if spec is not None and spec.origin:
                    pkg_dir = os.path.dirname(spec.origin)
                    dict_path = os.path.join(pkg_dir, "frequency_dictionary_en_82_765.txt")
                    if os.path.isfile(dict_path):
                        with open(dict_path, "r", encoding="utf-8") as f:
                            for line in f:
                                parts = line.strip().split()
                                if parts:
                                    self._english_words.add(parts[0].lower())
                        logger.debug(f"Loaded {len(self._english_words)} English words from symspellpy.")
            except Exception as exc:
                logger.debug(f"symspellpy English dictionary not loaded: {exc}")

            # Bổ sung thuật ngữ kỹ thuật, tin học, viết tắt
            self._english_words.update(TECH_LOAN_WORDS)

            self._is_loaded = True

    def _expand_tone_variations(self) -> None:
        """
        Bổ sung các biến thể đặt dấu thanh kiểu truyền thống (hoà, toà, thuỷ, xoè,...)
        song song với kiểu mới (hòa, tòa, thủy, xòe,...) để đảm bảo các văn bản dùng
        kiểu truyền thống không bị đánh dấu sai.
        """
        pairs = [
            ("oà", "òa"), ("oá", "óa"), ("oả", "ỏa"), ("oã", "õa"), ("oạ", "ọa"),
            ("oè", "òe"), ("oé", "óe"), ("oẻ", "ỏe"), ("oẽ", "õe"), ("oẹ", "ọe"),
            ("uỳ", "ùy"), ("uý", "úy"), ("uỷ", "ủy"), ("uỹ", "ũy"), ("uỵ", "ụy"),
        ]
        to_add = set()
        for c in ONSET_CONSONANTS:
            for old_v, new_v in pairs:
                new_form = unicodedata.normalize("NFC", c + new_v)
                old_form = unicodedata.normalize("NFC", c + old_v)
                if new_form in self._valid_syllables:
                    to_add.add(old_form)
                elif old_form in self._valid_syllables:
                    to_add.add(new_form)
        self._valid_syllables.update(to_add)

    def _expand_word_tone_variations(self) -> None:
        """
        Mở rộng các biến thể đặt dấu thanh truyền thống (hoà, toà, thuỷ, xoè,...)
        và hiện đại (hòa, tòa, thủy, xòe,...) cho từ điển từ ghép tiếng Việt.
        """
        pairs = [
            ("oà", "òa"), ("oá", "óa"), ("oả", "ỏa"), ("oã", "õa"), ("oạ", "ọa"),
            ("oè", "òe"), ("oé", "óe"), ("oẻ", "ỏe"), ("oẽ", "õe"), ("oẹ", "ọe"),
            ("uỳ", "ùy"), ("uý", "úy"), ("uỷ", "ủy"), ("uỹ", "ũy"), ("uỵ", "ụy"),
        ]
        to_add = set()
        for w in self._vietnamese_words:
            w_new = w
            for old_v, new_v in pairs:
                if old_v in w_new:
                    w_new = w_new.replace(old_v, new_v)
            if w_new != w:
                to_add.add(w_new)

            w_old = w
            for old_v, new_v in pairs:
                if new_v in w_old:
                    w_old = w_old.replace(new_v, old_v)
            if w_old != w:
                to_add.add(w_old)

        for w in to_add:
            self._vietnamese_words.add(w)
            parts = w.split()
            if len(parts) == 2:
                self._compound_prefixes.setdefault(parts[0], set()).add(parts[1])
                self._compound_suffixes.setdefault(parts[1], set()).add(parts[0])

    # ------------------------------------------------------------------
    # Thuật Toán Sinh Gợi Ý Sửa Đa Tiêu Chí
    # ------------------------------------------------------------------

    @functools.lru_cache(maxsize=4096)
    def get_suggestions(
        self,
        word: str,
        context_prev: str = "",
        context_next: str = "",
    ) -> List[str]:
        """
        Sinh tối đa 5 gợi ý sửa phù hợp nhất cho từ nghi vấn trong ngữ cảnh.

        Args:
            word: Token nghi vấn gốc.
            context_prev: Từ liền trước (nếu có).
            context_next: Từ liền sau (nếu có).

        Returns:
            List[str]: Tối đa 5 gợi ý xếp theo độ tin cậy giảm dần.
        """
        w_norm = unicodedata.normalize("NFC", word.strip().lower())
        if not w_norm:
            return []

        prev_norm = unicodedata.normalize("NFC", context_prev.strip().lower()) if context_prev else ""
        next_norm = unicodedata.normalize("NFC", context_next.strip().lower()) if context_next else ""

        candidates: Set[str] = set()

        # 1. Khai thác ngữ cảnh từ ghép N-gram (Ưu tiên cao nhất)
        if prev_norm and prev_norm in self._compound_prefixes:
            for c in self._compound_prefixes[prev_norm]:
                if weighted_edit_distance(w_norm, c) <= 1.8:
                    candidates.add(c)

        if next_norm and next_norm in self._compound_suffixes:
            for c in self._compound_suffixes[next_norm]:
                if weighted_edit_distance(w_norm, c) <= 1.8:
                    candidates.add(c)

        # 2. Sửa lỗi vị trí dấu nguyên âm đôi (họach -> hoạch, sóat -> soát,...)
        tone_shifts = [
            ("ọach", "oạch"), ("óat", "soát"), ("òat", "oạt"), ("óan", "oán"),
            ("òan", "oàn"), ("óai", "oái"), ("òai", "oài"), ("ỏai", "oải"),
            ("óang", "oáng"), ("òang", "oàng"), ("ọang", "oạng"),
            ("úy", "uý"), ("ùy", "uỳ"), ("ủy", "uỷ"), ("ụy", "uỵ"),
            ("uý", "úy"), ("uỳ", "ùy"), ("uỷ", "ủy"), ("uỵ", "ụy"),
        ]
        for bad_p, good_p in tone_shifts:
            if bad_p in w_norm:
                cand = w_norm.replace(bad_p, good_p)
                if cand in self._valid_syllables:
                    candidates.add(cand)

        # 3. Khôi phục phím gõ Telex dư ở đuôi (kiêmr -> kiểm, kieems -> kiếm,...)
        if w_norm and w_norm[-1] in "rsxfj":
            trimmed = w_norm[:-1]
            telex_map = {
                "r": {"e": "ẻ", "ê": "ể", "a": "ả", "ă": "ẳ", "â": "ẩ", "o": "ỏ", "ô": "ổ", "ơ": "ở", "u": "ủ", "ư": "ử", "i": "ỉ", "y": "ỷ"},
                "s": {"e": "é", "ê": "ế", "a": "á", "ă": "ắ", "â": "ấ", "o": "ó", "ô": "ố", "ơ": "ớ", "u": "ú", "ư": "ứ", "i": "í", "y": "ý"},
                "f": {"e": "è", "ê": "ề", "a": "à", "ă": "ằ", "â": "ầ", "o": "ò", "ô": "ồ", "ơ": "ờ", "u": "ù", "ư": "ừ", "i": "ì", "y": "ỳ"},
                "x": {"e": "ẽ", "ê": "ễ", "a": "ã", "ă": "ẵ", "â": "ẫ", "o": "õ", "ô": "ỗ", "ơ": "ỡ", "u": "ũ", "ư": "ữ", "i": "ĩ", "y": "ỹ"},
                "j": {"e": "ẹ", "ê": "ệ", "a": "ạ", "ă": "ặ", "â": "ậ", "o": "ọ", "ô": "ộ", "ơ": "ợ", "u": "ụ", "ư": "ự", "i": "ị", "y": "ỵ"},
            }
            ch = w_norm[-1]
            if ch in telex_map:
                for v_old, v_new in telex_map[ch].items():
                    if v_old in trimmed:
                        cand = trimmed.replace(v_old, v_new, 1)
                        if cand in self._valid_syllables:
                            candidates.add(cand)

        # 4. Tìm các âm tiết cùng phụ âm và nguyên âm cơ sở khác dấu thanh (đap -> đáp, đạp; đươc -> được)
        w_base = remove_diacritics(w_norm)
        for syl in self._valid_syllables:
            if remove_diacritics(syl) == w_base:
                if weighted_edit_distance(w_norm, syl) <= 1.5:
                    candidates.add(syl)

        # 5. Khôi phục theo cặp nhầm lẫn OCR có cơ sở (ví dụ u/ư, ơ/ă, o/ô, b/đ)
        if len(candidates) < 3:
            for syl in self._valid_syllables:
                if abs(len(syl) - len(w_norm)) <= 1:
                    dist = weighted_edit_distance(w_norm, syl)
                    if dist <= 0.6:  # Chỉ chấp nhận khi có bằng chứng OCR/ngữ âm mạnh
                        candidates.add(syl)

        # Loại bỏ từ trùng với từ gốc
        candidates.discard(w_norm)
        if not candidates:
            return []

        # Xếp hạng ứng viên theo điểm tổng hợp
        scored: List[Tuple[float, str]] = []
        for c in candidates:
            dist = weighted_edit_distance(w_norm, c)
            # Điểm khoảng cách (càng nhỏ điểm càng cao)
            score = -dist * 2.5

            # Điểm khớp ngữ cảnh N-gram
            if prev_norm and f"{prev_norm} {c}" in self._vietnamese_words:
                score += 5.0
            if next_norm and f"{c} {next_norm}" in self._vietnamese_words:
                score += 5.0

            # Điểm tần suất từ vựng
            if c in self._vietnamese_words:
                score += 1.2
            if c in COMMON_WORD_PRIORITY:
                score += COMMON_WORD_PRIORITY[c]

            scored.append((score, c))

        scored.sort(key=lambda item: item[0], reverse=True)
        top_candidates = [c for _, c in scored[:5]]

        # Bảo toàn kiểu chữ (casing) của token gốc
        result: List[str] = []
        for c in top_candidates:
            if word.isupper() and len(word) > 1:
                result.append(c.upper())
            elif word and word[0].isupper():
                result.append(c[0].upper() + c[1:] if len(c) > 1 else c.upper())
            else:
                result.append(c)

        return result

    # ------------------------------------------------------------------
    # Thẩm Định Token Độc Lập
    # ------------------------------------------------------------------

    @functools.lru_cache(maxsize=16384)
    def is_valid_token(self, token_lower: str, is_capitalized: bool, has_vietnamese_char: bool) -> bool:
        """Kiểm tra tính hợp lệ của một token độc lập."""
        if len(token_lower) <= 1:
            return True

        if token_lower in self._valid_syllables:
            return True

        # Chứa ký tự tiếng Việt nhưng không thuộc tập âm tiết hợp lệ
        if has_vietnamese_char:
            return False

        # Token thuần Latin ASCII
        if token_lower in self._english_words:
            return True

        if is_capitalized:
            return True

        # Nhận diện đặc trưng ngữ âm ngoại lai (ký tự f, j, w, z hoặc cụm phụ âm không có trong tiếng Việt)
        if any(c in token_lower for c in "fjwz"):
            return True
        non_vn_clusters = (
            "ck", "nd", "st", "sh", "ct", "pt", "ft", "lt", "rt", "nt", "xt",
            "mp", "rk", "lk", "ld", "ts", "ps", "pr", "pl", "br", "bl", "cr",
            "cl", "dr", "fr", "fl", "gr", "gl", "str", "spr", "scr",
        )
        if any(cluster in token_lower for cluster in non_vn_clusters):
            return True

        return False

    # ------------------------------------------------------------------
    # Phân Tích Ngữ Cảnh & Dò Soát Lỗi
    # ------------------------------------------------------------------

    def check_text_spelling(self, text: str) -> List[SpellError]:
        """
        Dò soát toàn bộ văn bản và trả về danh sách các vị trí nghi vấn chính tả
        kèm loại lỗi, độ tin cậy và gợi ý sửa có căn cứ.
        """
        if not text or not text.strip():
            return []

        # 1. Thu thập các vùng được bảo vệ (URLs, emails, ngày tháng, mã công văn)
        protected_spans: List[Tuple[int, int]] = []
        for m in RE_PROTECTED.finditer(text):
            protected_spans.append((m.start(), m.end()))

        # 2. Thu thập danh sách token cùng vị trí trong văn bản
        tokens_info: List[Tuple[int, int, str]] = []
        for m in RE_WORD.finditer(text):
            start_idx = m.start()
            end_idx = m.end()
            raw_token = m.group()

            # Bỏ qua nếu nằm trong vùng được bảo vệ
            in_protected = False
            for p_start, p_end in protected_spans:
                if not (end_idx <= p_start or start_idx >= p_end):
                    in_protected = True
                    break
            if in_protected:
                continue

            # Bỏ qua từ viết tắt in hoa toàn bộ (UBND, HĐND, QĐ, CPU, VIP, TW, BCĐTW,...)
            if len(raw_token) >= 2 and raw_token.isupper():
                continue

            tokens_info.append((start_idx, end_idx, raw_token))

        suspicious_list: List[SpellError] = []
        num_tokens = len(tokens_info)

        # 3. Dò soát từng token trong ngữ cảnh N-gram
        for idx, (start_idx, end_idx, raw_token) in enumerate(tokens_info):
            norm_token = unicodedata.normalize("NFC", raw_token)
            token_lower = norm_token.lower()
            is_capitalized = raw_token[0].isupper()
            has_vietnamese = any(c in VIETNAMESE_DIACRITICS for c in norm_token)

            # Lấy ngữ cảnh xung quanh
            prev_token = tokens_info[idx - 1][2] if idx > 0 else ""
            prev_prev_token = tokens_info[idx - 2][2] if idx > 1 else ""
            next_token = tokens_info[idx + 1][2] if idx < num_tokens - 1 else ""
            next_next_token = tokens_info[idx + 2][2] if idx < num_tokens - 2 else ""

            # Thẩm định tính hợp lệ của âm tiết độc lập
            is_valid_standalone = self.is_valid_token(token_lower, is_capitalized, has_vietnamese)

            if not is_valid_standalone:
                # --- LỖI CẤU TRÚC ÂM TIẾT RÕ RÀNG ---
                error_type = "invalid_syllable"
                reason = "Âm tiết sai cấu trúc hoặc sai vị trí dấu thanh"
                confidence = 0.95

                if raw_token and raw_token[-1] in "rsxfj" and has_vietnamese:
                    error_type = "telex_error"
                    reason = "Lỗi ký tự gõ Telex còn sót ở cuối từ"
                elif any(pat in token_lower for pat in ["ọach", "sóat", "òat", "óan"]):
                    error_type = "misplaced_tone"
                    reason = "Lỗi vị trí dấu thanh trong nguyên âm đôi"

                suggestions = self.get_suggestions(raw_token, prev_token, next_token)
                suspicious_list.append(
                    SpellError(
                        start_idx=start_idx,
                        end_idx=end_idx,
                        token=raw_token,
                        error_type=error_type,
                        confidence=confidence,
                        suggestions=suggestions,
                        reason=reason,
                    )
                )
            else:
                # --- PHÂN TÍCH NGỮ CẢNH BIGRAM / TRIGRAM ---
                # Token đứng riêng hợp lệ, nhưng xét trong cụm từ có dấu hiệu nhầm dấu hoặc nhầm ký tự OCR:
                # Ví dụ: 'kỹ su' -> 'kỹ sư', 'lâu nơm' -> 'lâu năm'
                if prev_token or next_token:
                    context_err = self._evaluate_context_confusion(
                        prev_token, raw_token, next_token,
                        prev_prev_token=prev_prev_token,
                        next_next_token=next_next_token,
                    )
                    if context_err is not None:
                        best_sugg, reason = context_err
                        suggestions = self.get_suggestions(raw_token, prev_token, next_token)
                        # Đảm bảo best_sugg nằm đầu danh sách
                        if best_sugg in suggestions:
                            suggestions.remove(best_sugg)
                        suggestions.insert(0, best_sugg)
                        suggestions = suggestions[:5]

                        suspicious_list.append(
                            SpellError(
                                start_idx=start_idx,
                                end_idx=end_idx,
                                token=raw_token,
                                error_type="context_confusion",
                                confidence=0.85,
                                suggestions=suggestions,
                                reason=reason,
                            )
                        )

        # 4. Sắp xếp danh sách kết quả theo start_idx tăng dần
        suspicious_list.sort(key=lambda err: err.start_idx)
        return suspicious_list

    def _evaluate_context_confusion(
        self,
        prev_token: str,
        curr_token: str,
        next_token: str,
        prev_prev_token: str = "",
        next_next_token: str = "",
    ) -> Optional[Tuple[str, str]]:
        """
        Đánh giá xem token hiện tại có bị nhầm lẫn trong ngữ cảnh với từ liền trước / sau hay không.
        Chỉ đánh dấu khi có bằng chứng chắc chắn (weighted edit distance <= 0.7 với một từ ghép chuẩn).
        Bảo vệ tuyệt đối các từ chức năng (stopwords), tên riêng, và ranh giới từ ghép đã nhận diện.
        """
        curr_lower = unicodedata.normalize("NFC", curr_token.lower())
        prev_lower = unicodedata.normalize("NFC", prev_token.lower()) if prev_token else ""
        next_lower = unicodedata.normalize("NFC", next_token.lower()) if next_token else ""
        prev_prev_lower = unicodedata.normalize("NFC", prev_prev_token.lower()) if prev_prev_token else ""
        next_next_lower = unicodedata.normalize("NFC", next_next_token.lower()) if next_next_token else ""

        # 1. Bảo vệ từ chức năng, liên từ, giới từ, loại từ và từ vựng phổ thông cơ bản
        if curr_lower in PROTECTED_COMMON_WORDS:
            return None

        # 2. Bảo vệ tên riêng (cả 2 từ liền nhau đều viết hoa)
        if prev_token and curr_token and prev_token[0].isupper() and curr_token[0].isupper():
            return None
        if next_token and curr_token and next_token[0].isupper() and curr_token[0].isupper():
            return None

        # 3. Nếu cụm từ ghép hiện tại đã có trong từ điển thì hoàn toàn đúng ngữ cảnh
        if prev_lower and f"{prev_lower} {curr_lower}" in self._vietnamese_words:
            return None
        if next_lower and f"{curr_lower} {next_lower}" in self._vietnamese_words:
            return None

        # 4. Ranh giới từ ghép: Nếu prev_token đã thuộc về từ ghép trước đó (prev_prev + prev)
        # thì prev_token không ghép tiếp với curr_token
        prev_is_bound = bool(
            prev_prev_lower and f"{prev_prev_lower} {prev_lower}" in self._vietnamese_words
        )
        if prev_is_bound or prev_lower in PROTECTED_COMMON_WORDS or not prev_lower:
            return None

        # 5. So khớp với tiền tố liền trước (chỉ khi prev_token có từ ghép với candidate)
        if prev_lower in self._compound_prefixes:
            best_cand = None
            min_dist = 999.0
            for cand in self._compound_prefixes[prev_lower]:
                dist = weighted_edit_distance(curr_lower, cand)
                if dist <= 0.6 and dist < min_dist:
                    min_dist = dist
                    best_cand = cand
            if best_cand:
                sugg_fmt = best_cand.capitalize() if curr_token and curr_token[0].isupper() else best_cand
                return sugg_fmt, f"Nghi vấn nhầm ký tự trong cụm '{prev_token} {curr_token}' (gợi ý: '{prev_token} {sugg_fmt}')"

        return None


# ---------------------------------------------------------------------------
# Singleton Instance & Module-Level API
# ---------------------------------------------------------------------------

_GLOBAL_SPELL_CHECKER: Optional[VietnameseSpellChecker] = None
_GLOBAL_SPELL_CHECKER_LOCK = threading.Lock()


def get_default_spell_checker() -> VietnameseSpellChecker:
    """Lấy singleton VietnameseSpellChecker."""
    global _GLOBAL_SPELL_CHECKER
    with _GLOBAL_SPELL_CHECKER_LOCK:
        if _GLOBAL_SPELL_CHECKER is None:
            _GLOBAL_SPELL_CHECKER = VietnameseSpellChecker()
        return _GLOBAL_SPELL_CHECKER


def check_text_spelling(text: str) -> List[SpellError]:
    """
    API chính: Dò soát chính tả tiếng Việt trên văn bản nguyên bản.

    Returns:
        List[SpellError]: Danh sách các lỗi nghi vấn.
        Mỗi phần tử có start_idx, end_idx, token, error_type, confidence, suggestions, reason
        và có thể unpack trực tiếp dạng tuple (start_idx, end_idx, token) để tương thích ngược.
    """
    checker = get_default_spell_checker()
    return checker.check_text_spelling(text)


def check_text_spelling_detailed(text: str) -> List[SpellError]:
    """API tường minh trả về danh sách SpellError đầy đủ chi tiết."""
    return check_text_spelling(text)


def get_suggestions(word: str, context_prev: str = "", context_next: str = "") -> List[str]:
    """
    API bắt buộc: Sinh tối đa 5 gợi ý sửa phù hợp nhất cho từ trong ngữ cảnh.

    Args:
        word: Từ cần sinh gợi ý.
        context_prev: Từ liền trước (nếu có).
        context_next: Từ liền sau (nếu có).

    Returns:
        List[str]: Danh sách gợi ý xếp theo độ ưu tiên.
    """
    checker = get_default_spell_checker()
    return checker.get_suggestions(word, context_prev, context_next)
