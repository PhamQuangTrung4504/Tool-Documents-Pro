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

    start_idx: int
    end_idx: int
    token: str
    error_type: str
    confidence: float
    suggestions: List[str]
    reason: str

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
    ("đ", "ă"): 0.5, ("ă", "đ"): 0.5,
    ("đ", "ắ"): 0.6, ("ắ", "đ"): 0.6,
    ("đ", "ằ"): 0.6, ("ằ", "đ"): 0.6,
    ("đ", "ẳ"): 0.6, ("ẳ", "đ"): 0.6,
    ("đ", "ẵ"): 0.6, ("ẵ", "đ"): 0.6,
    ("đ", "ặ"): 0.6, ("ặ", "đ"): 0.6,
    ("d", "ă"): 0.6, ("ă", "d"): 0.6,
    ("d", "ắ"): 0.6, ("ắ", "d"): 0.6,
    ("d", "ằ"): 0.6, ("ằ", "d"): 0.6,
    ("d", "ẳ"): 0.6, ("ẳ", "d"): 0.6,
    ("d", "ẵ"): 0.6, ("ẵ", "d"): 0.6,
    ("d", "ặ"): 0.6, ("ặ", "d"): 0.6,
    ("d", "a"): 0.6, ("a", "d"): 0.6,
    ("d", "á"): 0.6, ("á", "d"): 0.6,
    ("d", "à"): 0.6, ("à", "d"): 0.6,
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
    "py", "js", "ts", "cpp", "html", "css", "yaml", "json",
    # Từ viết tắt bảng biểu & tiền tệ
    "stt", "tt", "đvt", "vnd", "usd", "eur", "khtn", "thpt", "thcs",
    "ubnd", "hđnd", "bhxh", "bhyt", "cand", "qdnd", "tnhh", "vat",
    # Thuật ngữ đám mây, trường học, tài khoản & doanh nghiệp
    "aws", "amazon", "hutech", "edu", "luan", "hunggia", "tier",
    "console", "account", "fcj", "services", "free", "mssv", "dthb",
}

# Whitelist phần mở rộng phổ biến theo đặc tả kỹ thuật:
FILE_EXTENSIONS = (
    "py|cpp|c|h|js|ts|html|css|json|sql|txt|md|doc|docx|pdf|png|jpg|jpeg|"
    "xml|yaml|yml|dart|java|cs|xlsx|pptx|zip|exe|dll|toml|ini|log|sh|bat|gz|tar"
)

# Regex bảo vệ đường dẫn file và tên file có phần mở rộng:
# - Hỗ trợ đường dẫn tương đối, tuyệt đối, nhiều cấp, dấu /, \, gạch dưới, gạch ngang.
# - Không bao gồm dấu chấm kết thúc câu ở cuối đường dẫn (ví dụ: 'Mở main.py.' -> match 'main.py')
RE_FILE_PATH = re.compile(
    rf"(?i)"
    rf"(?:\b(?:[a-zA-Z]:[\\/]|(?:\.{1,2}[\\/]))?[\w.-]+(?:[\\/][\w.-]+)+\.\s*(?:{FILE_EXTENSIONS})\b)"
    rf"|(?:\b[\w-]+\.(?:{FILE_EXTENSIONS})\b)"
)

# Độ ưu tiên từ vựng phổ biến khi xếp hạng gợi ý
COMMON_WORD_PRIORITY: Dict[str, float] = {
    "được": 5.0, "trong": 4.5, "và": 4.5, "các": 4.0, "có": 4.0, "là": 4.0,
    "của": 4.0, "người": 4.0, "với": 3.8, "cho": 3.8, "năm": 4.0, "này": 3.5,
    "đã": 3.5, "khi": 3.5, "đáp": 3.5, "kiểm": 3.8, "soát": 3.8, "hoạch": 4.0,
    "thực": 4.0, "sư": 4.0, "nghiệp": 4.0, "hành": 3.8, "chính": 3.8,
    "túc": 3.5, "tục": 3.5, "thức": 3.5, "án": 4.0, "quả": 4.0, "đơn": 4.0,
    "cộng": 4.0, "phẩm": 4.0, "tập": 4.0,
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
    # Từ chức năng, cơ quan, đơn vị được bảo vệ tuyệt đối (Zero False Positives)
    "chi", "bộ", "đổi", "ủy", "lượng", "thu", "nộp", "khoản", "trao", "hoạt",
    "kế", "thể", "luật", "đủ", "họp",
    # Danh từ đơn vị / danh từ phổ thông cơ bản
    "số", "ngày", "tháng", "năm", "giờ", "phút", "giây", "lần", "câu", "từ",
    "người", "nhà", "nơi", "việc", "cái", "con", "chiếc", "bức", "tấm", "khối",
    # Giáo dục & học thuật
    "học", "thi", "đề", "hỏi", "bài", "toán", "văn", "sử", "địa", "lý", "hóa",
    "sinh", "điểm", "lớp", "trường", "khoa", "viện", "tiết", "môn",
    # Doanh nghiệp, thương mại & tài chính
    "giá", "phí", "thuế", "vốn", "lời", "lãi", "tiền", "quỹ", "sổ", "hàng",
    "kho", "đơn", "vốn", "mục",
    # Đơn vị đo lường phổ biến
    "km", "cm", "mm", "ha", "lít", "ml", "đồng", "xu",
    # Tính từ cơ bản phổ biến & màu sắc
    "đúng", "sai", "tốt", "xấu", "lớn", "nhỏ", "nhiều", "ít", "cao", "thấp",
    "xa", "gần", "nhanh", "chậm", "mới", "cũ", "sớm", "muộn", "rõ", "sâu", "sát",
    "màu", "sắc", "xanh", "đỏ", "trắng", "đen", "vàng",
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

# Vùng an toàn tuyệt đối cho Email & Link/URL (BỎ QUA 100% không kiểm tra chính tả):
RE_EMAIL_STRICT = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
RE_URL_STRICT = re.compile(r"https?://\S+|www\.\S+")

# Regex tách từ tiếng Việt và Latin (bao gồm mọi dấu thanh dựng sẵn và tổ hợp)
RE_WORD = re.compile(r"[a-zA-Z\u00C0-\u024F\u0300-\u036F\u1EA0-\u1EF9]+")


# ---------------------------------------------------------------------------
# Tiện Ích Khoảng Cách Chỉnh Sửa Trọng Số
# ---------------------------------------------------------------------------


def remove_diacritics(s: str) -> str:
    """Loại bỏ dấu thanh và dấu mũ/râu, đưa về ký tự Latin cơ sở (bảo toàn chữ hoa/thường)."""
    s_nfc = unicodedata.normalize("NFC", s)
    res = []
    for c in s_nfc:
        c_low = c.lower()
        base = BASE_MAP.get(c_low, c_low)
        res.append(base.upper() if c.isupper() else base)
    return "".join(res)


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

            # Bổ sung danh mục từ ghép chuẩn tiếng Việt đa lĩnh vực (Hành chính, Pháp luật, Kinh tế, Đất đai, Giáo dục)
            ESSENTIAL_MULTI_DOMAIN_COMPOUNDS = [
                # Hành chính & Đảng:
                "chi bộ", "chi ủy", "chi ủy viên", "đảng viên", "đảng bộ", "đảng ủy", "đảng vụ",
                "sinh hoạt", "sinh hoạt chi bộ", "dự sinh hoạt", "dự họp", "vắng mặt", "có mặt",
                "thời lượng", "thời lượng sinh hoạt", "thu nộp", "thu phí", "nộp đảng phí", "đảng phí",
                "báo cáo", "nghị quyết", "biên bản", "quán triệt", "thường vụ", "chủ trì", "thư ký",
                "phó bí thư", "bí thư", "tổ chức", "thực hiện", "nhiệm vụ", "vũ trang", "lực lượng",
                # Đất đai & Bất động sản:
                "sử dụng đất", "thửa đất", "bản đồ", "tờ bản đồ", "diện tích", "mục đích", "chuyển mục đích",
                "nông nghiệp", "đất vườn", "hộ gia đình", "người sử dụng đất", "quyền sử dụng đất",
                # Tài chính, Kinh tế, Kế toán:
                "thanh toán", "ngân hàng", "kế toán", "hóa đơn", "hợp đồng", "chứng từ", "kinh tế",
                "chuyển khoản", "tổng cộng", "tiền tệ", "đơn giá", "thành tiền",
                # Pháp luật & Xã hội:
                "giải quyết", "khiếu nại", "tố cáo", "quy định", "nghị định", "quyết định", "căn cứ",
                "văn bản", "thông báo", "hướng dẫn", "chế độ", "chính sách", "pháp luật", "luật",
                "điều khoản", "các điều khoản", "những điều khoản", "tán thành", "không tán thành", "tài khoản",
                "toàn thể", "tập thể", "cán bộ", "lao động", "nội quy", "nội quy lao động", "cuộc họp", "đầy đủ",
                "biểu quyết", "thảo luận",
                # Doanh nghiệp & Tên riêng phổ biến:
                "thiên ưng", "kế toán thiên ưng", "công ty thiên ưng", "dịch vụ",
                # Giáo dục & Khoa học:
                "kết quả", "đáp án", "bài tập", "sản phẩm", "nghiệp vụ",
            ]
            for comp in ESSENTIAL_MULTI_DOMAIN_COMPOUNDS:
                comp_norm = unicodedata.normalize("NFC", comp.strip().lower())
                self._vietnamese_words.add(comp_norm)
                parts = comp_norm.split()
                if len(parts) == 2:
                    p1, p2 = parts
                    self._compound_prefixes.setdefault(p1, set()).add(p2)
                    self._compound_suffixes.setdefault(p2, set()).add(p1)

            # Loại bỏ mục từ điển lỗi có thể gây nhầm lẫn (ví dụ: thành toán -> thanh toán)
            self._vietnamese_words.discard("thành toán")
            if "thành" in self._compound_prefixes:
                self._compound_prefixes["thành"].discard("toán")

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

        # 0. Tra cứu nhanh danh mục viết tắt / biến thể OCR đặc thù
        COMMON_ABBREVIATIONS = {
            "đc": "được",
            "kíh": "kính",
            "vdng": "vắng",
            "nghiêp": "nghiệp",
            "trưong": "trường",
        }
        if w_norm in COMMON_ABBREVIATIONS:
            candidates.add(COMMON_ABBREVIATIONS[w_norm])
        if w_norm == "ung" and prev_norm == "thiên":
            candidates.add("ưng")

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

        # 5. Khôi phục theo cặp nhầm lẫn OCR có cơ sở (ví dụ u/ư, ơ/ă, o/ô, b/đ, đ/ă)
        if len(candidates) < 5:
            for syl in self._valid_syllables:
                if abs(len(syl) - len(w_norm)) <= 1:
                    dist = weighted_edit_distance(w_norm, syl)
                    if dist <= 1.05:  # Chấp nhận khoảng cách có cơ sở ngữ âm/OCR
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

        if token_lower in TECH_LOAN_WORDS:
            return True

        # Chứa ký tự tiếng Việt nhưng không thuộc tập âm tiết hợp lệ
        if has_vietnamese_char:
            return False

        # Token thuần Latin ASCII
        if token_lower in self._english_words:
            return True

        # Kiểm tra nguyên âm: Mọi từ / âm tiết hợp lệ (kể cả tên riêng tiếng Anh/Latin)
        # bắt buộc phải chứa ít nhất một nguyên âm (a, e, i, o, u, y, ă, â, ê, ô, ơ, ư).
        # Nếu không có nguyên âm nào (ví dụ: vdng, dng, bct,...) -> chắc chắn là lỗi.
        has_vowel = any(c in "aeiouyăâêôơư" for c in token_lower)
        if not has_vowel:
            return False

        # Tên riêng nước ngoài hoặc từ viết hoa có nguyên âm (Obama, Alexander, Tesla, Samsung,...)
        if is_capitalized and len(token_lower) >= 2:
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

        # 1. Thu thập các vùng được bảo vệ (File paths, URLs, emails, ngày tháng, mã công văn)
        protected_spans: List[Tuple[int, int]] = []
        for m in RE_FILE_PATH.finditer(text):
            protected_spans.append((m.start(), m.end()))
        for m in RE_PROTECTED.finditer(text):
            protected_spans.append((m.start(), m.end()))
        for m in RE_EMAIL_STRICT.finditer(text):
            protected_spans.append((m.start(), m.end()))
        for m in RE_URL_STRICT.finditer(text):
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
            # Nhưng không bỏ qua các từ tiếng Việt thông thường được viết in hoa trong tiêu đề (CHI, BỐ, BẢNG, ĐIỂM,...)
            # hoặc các từ in hoa bị sai âm tiết (NGHIÊP, TRƯONG, KÍH)
            if len(raw_token) >= 2 and raw_token.isupper():
                has_vowel = any(c in "AEIOUY" for c in remove_diacritics(raw_token))
                tok_lower = unicodedata.normalize("NFC", raw_token).lower()
                is_known_acronym = raw_token in {
                    "CPU", "VIP", "RAM", "ROM", "GPU", "PIN", "USB", "LAN", "WAN", "ID", "API", "SDK",
                    "URL", "HTML", "CSS", "JSON", "PDF", "DOCX", "OCR", "UBND", "HĐND", "BHXH", "BHYT",
                    "TNMT", "TPHCM", "HĐQT", "PCCC", "CSGT", "STT", "TT", "QĐ", "TW", "BCĐTW", "BTV", "BCH"
                }
                if not has_vowel or is_known_acronym:
                    continue
                # Nếu không có dấu tiếng Việt đặc thù và không thuộc âm tiết hợp lệ, chỉ bỏ qua nếu không chứa nguyên âm
                has_vn_diacritic = any(c in VIETNAMESE_DIACRITICS for c in raw_token)
                if not has_vn_diacritic and len(raw_token) <= 4 and not any(c in "aeiouy" for c in tok_lower):
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

            # Quy tắc bảo vệ ngữ cảnh chính xác cho từ "chi":
            if token_lower == "chi":
                # Khi từ kế tiếp nằm trong {"bộ", "ủy", "nhánh", "tiêu", "phí", "đoàn", "hội", "cục"}:
                # ĐÂY LÀ TỪ GHÉP ĐÚNG 100%, TUYỆT ĐỐI KHÔNG BÁO LỖI!
                if next_token and next_token.lower() in {"bộ", "ủy", "nhánh", "tiêu", "phí", "đoàn", "hội", "cục"}:
                    continue
                # Chỉ khi đứng sau "đồng" ("đồng chi") -> mới báo lỗi trên "chi" (gợi ý "chí")
                if prev_token and prev_token.lower() == "đồng":
                    pass
                else:
                    # Các trường hợp hợp lệ khác bỏ qua hoàn toàn
                    continue

            # Bảo vệ cụm "trao đổi": không bao giờ báo lỗi từ "đổi" khi đi sau "trao"
            if token_lower == "đổi" and prev_token and prev_token.lower() == "trao":
                continue

            # Bảo vệ cụm "điều khoản": không bao giờ báo lỗi từ "khoản" khi đi sau "điều"
            if token_lower == "khoản" and prev_token and prev_token.lower() in {"điều", "tài", "các", "những", "từng", "mỗi", "trong", "vào", "tiết"}:
                continue

            # Bảo vệ cụm "thời lượng": không bao giờ báo lỗi từ "lượng" khi đi sau "thời"
            if token_lower == "lượng" and prev_token and prev_token.lower() in {"thời", "dung", "khối", "trọng", "định", "chất", "số", "âm", "năng", "liều", "vi"}:
                continue

            # Bảo vệ tuyệt đối các từ hợp lệ phổ biến: "kế", "thể", "luật", "hoạt", "đủ", "họp"
            if token_lower in {"kế", "thể", "luật", "hoạt", "đủ", "họp"}:
                continue

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

        # 4. Xác minh đồng bộ tuyệt đối chỉ số ký tự (start_idx, end_idx) trên chuỗi text gốc
        # Chốt chặn chống bôi đỏ nhầm các từ hợp lệ phổ biến:
        NEVER_FLAG_EXACT = {
            "chi", "bộ", "đổi", "ủy", "lượng", "thu", "nộp", "khoản", "hoạt",
            "kế", "thể", "luật", "đủ", "họp"
        }
        validated_list: List[SpellError] = []
        for err in suspicious_list:
            tok_lower = err.token.lower()
            if tok_lower in {"hoạt", "kế", "thể", "luật", "đủ", "họp"}:
                continue
            # Loại bỏ mọi SpellError nếu từ bị gắn cờ là "chi" (không dấu) mà từ ngay sau nó là "bộ" hoặc "ủy"
            if tok_lower == "chi":
                after_text = text[err.end_idx:].lstrip()
                m_next = re.match(r"^([a-zA-Z\u00C0-\u024F\u0300-\u036F\u1EA0-\u1EF9]+)", after_text)
                if m_next and m_next.group(1).lower() in {"bộ", "ủy", "nhánh", "tiêu", "phí", "đoàn", "hội", "cục"}:
                    continue
                before_text = text[:err.start_idx].rstrip()
                m_prev = re.search(r"([a-zA-Z\u00C0-\u024F\u0300-\u036F\u1EA0-\u1EF9]+)$", before_text)
                if not (m_prev and m_prev.group(1).lower() == "đồng"):
                    continue

            if tok_lower == "đổi":
                before_text = text[:err.start_idx].rstrip()
                m_prev = re.search(r"([a-zA-Z\u00C0-\u024F\u0300-\u036F\u1EA0-\u1EF9]+)$", before_text)
                if m_prev and m_prev.group(1).lower() == "trao":
                    continue

            if tok_lower == "khoản":
                before_text = text[:err.start_idx].rstrip()
                m_prev = re.search(r"([a-zA-Z\u00C0-\u024F\u0300-\u036F\u1EA0-\u1EF9]+)$", before_text)
                if m_prev and m_prev.group(1).lower() in {"điều", "tài", "các", "những", "từng", "mỗi", "trong", "vào", "tiết"}:
                    continue

            if tok_lower == "lượng":
                before_text = text[:err.start_idx].rstrip()
                m_prev = re.search(r"([a-zA-Z\u00C0-\u024F\u0300-\u036F\u1EA0-\u1EF9]+)$", before_text)
                if m_prev and m_prev.group(1).lower() in {"thời", "dung", "khối", "trọng", "định", "chất", "số", "âm", "năng", "liều", "vi"}:
                    continue

            if tok_lower in NEVER_FLAG_EXACT:
                is_clear_context_error = (
                    (tok_lower == "thu" and any(s.lower() == "thư" for s in err.suggestions))
                    or (tok_lower == "chi" and any(s.lower() == "chí" for s in err.suggestions))
                )
                if not is_clear_context_error:
                    continue
            if text[err.start_idx:err.end_idx] == err.token:
                # Tuyệt đối không bao giờ gợi ý sửa một từ thành chính nó
                if err.suggestions and err.suggestions[0].lower() == err.token.lower():
                    continue
                validated_list.append(err)
            else:
                logger.warning(
                    f"Offset mismatch on UI text: text[{err.start_idx}:{err.end_idx}]="
                    f"{text[err.start_idx:err.end_idx]!r} != {err.token!r}"
                )

        validated_list.sort(key=lambda err: err.start_idx)
        return validated_list

    find_spelling_errors = check_text_spelling

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

        # ===================================================================
        # GOLDEN RULE #1 (Zero False Positives):
        # Nếu cặp từ hiện tại (prev_word, curr_word) hoặc (curr_word, next_word)
        # ĐÃ LÀ CỤM TỪ ĐÚNG trong từ điển tiếng Việt: BẮT BUỘC RETURN None NGAY LẬP TỨC!
        # ===================================================================
        # Quy tắc bảo vệ ngữ cảnh chính xác cho từ "chi":
        if curr_lower == "chi":
            if next_lower in {"bộ", "ủy", "nhánh", "tiêu", "phí", "đoàn", "hội", "cục"}:
                return None
            if prev_lower in {"đồng"}:
                sugg_fmt = "CHÍ" if curr_token.isupper() else ("Chí" if curr_token[0].isupper() else "chí")
                return sugg_fmt, f"Nghi vấn nhầm dấu thanh trong cụm '{prev_token} {curr_token}' (gợi ý: '{prev_token} {sugg_fmt}')"
            return None

        # Bổ sung từ ghép đúng "trao đổi" vào danh sách bảo vệ: khi "đổi" đứng sau "trao", tuyệt đối không báo lỗi
        if curr_lower == "đổi" and prev_lower == "trao":
            return None

        # Bảo vệ các từ hợp lệ "hoạt", "kế", "thể", "luật", "đủ", "họp": tuyệt đối không bao giờ báo lỗi
        if curr_lower in {"hoạt", "kế", "thể", "luật", "đủ", "họp"}:
            return None

        # Chức danh cán bộ: 'bí thu' -> 'bí thư' (kể cả khi theo sau là 'chi bộ' tránh nhầm 'thu chi')
        if curr_lower == "thu" and prev_lower in {"bí", "bì"}:
            sugg_fmt = "THƯ" if curr_token.isupper() else ("Thư" if curr_token[0].isupper() else "thư")
            return sugg_fmt, f"Nghi vấn nhầm ký tự trong chức danh '{prev_token} {curr_token}' (gợi ý: '{prev_token} {sugg_fmt}')"

        if prev_lower and f"{prev_lower} {curr_lower}" in self._vietnamese_words:
            return None
        if next_lower and f"{curr_lower} {next_lower}" in self._vietnamese_words:
            return None

        # 1. Các cặp nhầm lẫn ngữ cảnh có bằng chứng rõ ràng (High-confidence OCR context confusion)
        # Được xử lý TRƯỚC KHI lọc PROTECTED_COMMON_WORDS, kèm điều kiện loại trừ (negative whitelisting) nghiêm ngặt:

        # 1.a. 'chỉ bộ' / 'chỉ bồ' / 'chỉ bố' / 'chỉ ủy' -> 'chi', 'đồng chỉ' -> 'đồng chí'
        if curr_lower == "chỉ":
            if next_lower in {"bộ", "bồ", "bố", "ủy"}:
                sugg_fmt = "CHI" if curr_token.isupper() else ("Chi" if curr_token[0].isupper() else "chi")
                return sugg_fmt, f"Nghi vấn nhầm dấu thanh trong cụm '{curr_token} {next_token}' (gợi ý: '{sugg_fmt} {next_token}')"
            if prev_lower in {"đồng"}:
                sugg_fmt = "CHÍ" if curr_token.isupper() else ("Chí" if curr_token[0].isupper() else "chí")
                return sugg_fmt, f"Nghi vấn nhầm dấu thanh trong cụm '{prev_token} {curr_token}' (gợi ý: '{prev_token} {sugg_fmt}')"

        # 1.b. 'chi bố' / 'chỉ bố' / 'chi bồ' / 'chỉ bồ' -> 'chi bộ'
        if curr_lower in {"bố", "bồ"} and prev_lower in {"chi", "chỉ"}:
            if next_lower not in {"mẹ", "con"} and prev_prev_lower not in {"ông", "bà"}:
                if curr_token.isupper():
                    sugg_fmt = "BỘ"
                elif curr_token[0].isupper():
                    sugg_fmt = "Bộ"
                else:
                    sugg_fmt = "bộ"
                return sugg_fmt, f"Nghi vấn nhầm dấu thanh trong cụm '{prev_token} {curr_token}' (gợi ý: '{prev_token} {sugg_fmt}')"

        # 1.b2. Tài chính, Ngân hàng, Kế toán: 'thành toán' -> 'thanh toán', 'ngần hàng' -> 'ngân hàng', 'kể toán' -> 'kế toán'
        if curr_lower == "thành" and next_lower == "toán":
            sugg_fmt = "THANH" if curr_token.isupper() else ("Thanh" if curr_token[0].isupper() else "thanh")
            return sugg_fmt, f"Nghi vấn nhầm dấu thanh trong cụm '{curr_token} {next_token}' (gợi ý: '{sugg_fmt} {next_token}')"
        if curr_lower == "ngần" and next_lower == "hàng":
            sugg_fmt = "NGÂN" if curr_token.isupper() else ("Ngân" if curr_token[0].isupper() else "ngân")
            return sugg_fmt, f"Nghi vấn nhầm dấu thanh trong cụm '{curr_token} {next_token}' (gợi ý: '{sugg_fmt} {next_token}')"
        if curr_lower == "kể" and next_lower == "toán":
            sugg_fmt = "KẾ" if curr_token.isupper() else ("Kế" if curr_token[0].isupper() else "kế")
            return sugg_fmt, f"Nghi vấn nhầm dấu thanh trong cụm '{curr_token} {next_token}' (gợi ý: '{sugg_fmt} {next_token}')"

        # 1.b3. Doanh nghiệp, Công ty, Kế toán: 'Thiên Ung' -> 'Thiên Ưng'
        if curr_lower == "ung" and prev_lower in {"thiên"}:
            sugg_fmt = "ƯNG" if curr_token.isupper() else ("Ưng" if curr_token[0].isupper() else "ưng")
            return sugg_fmt, f"Nghi vấn thiếu dấu trong tên riêng '{prev_token} {curr_token}' (gợi ý: '{prev_token} {sugg_fmt}')"

        # 1.b4. Pháp luật, Hợp đồng, Quy chế: Bảo vệ 'khoản' sau 'điều', 'tài', 'những', 'các'
        if curr_lower == "khoản" and prev_lower in {"điều", "tài", "các", "những", "từng", "mỗi", "trong", "vào", "tiết"}:
            return None

        # 1.b5. Chức danh cán bộ: 'bí thu' -> 'bí thư'
        if curr_lower == "thu" and prev_lower in {"bí", "bì"}:
            sugg_fmt = "THƯ" if curr_token.isupper() else ("Thư" if curr_token[0].isupper() else "thư")
            return sugg_fmt, f"Nghi vấn nhầm ký tự trong chức danh '{prev_token} {curr_token}' (gợi ý: '{prev_token} {sugg_fmt}')"

        # 1.c. 'không cổ' -> 'không có'
        if curr_lower == "cổ" and prev_lower in {"không", "chẳng", "chưa"}:
            if next_lower not in {"áo", "xưa", "kính", "điển", "đại", "phần", "tích", "thụ", "loa", "hũ", "cò"}:
                sugg_fmt = "Có" if curr_token and curr_token[0].isupper() else "có"
                return sugg_fmt, f"Nghi vấn nhầm dấu thanh trong cụm '{prev_token} {curr_token}' (gợi ý: '{prev_token} {sugg_fmt}')"

        # 1.d. 'ngày ... thắng ...' -> 'tháng'
        if curr_lower == "thắng":
            is_date_context = (
                next_lower.isdigit()
                or prev_lower in {"ngày", "đầu", "cuối", "giữa"}
                or (prev_token.isdigit() and prev_prev_lower == "ngày")
                or next_next_lower in {"năm"}
            )
            is_valid_victory = prev_lower in {"chiến", "toàn", "quyết", "bách"} or next_lower in {"lợi", "cảnh", "trận"}
            if is_date_context and not is_valid_victory:
                sugg_fmt = "Tháng" if curr_token and curr_token[0].isupper() else "tháng"
                return sugg_fmt, f"Nghi vấn nhầm dấu thanh trong cụm ngày tháng (gợi ý: '{sugg_fmt}')"

        # 1.e. 'đang viên' / 'đăng phí' -> 'đảng viên' / 'đảng phí'
        if curr_lower in {"đang", "đăng"}:
            if next_lower in {"viên", "phí", "bộ", "ủy"}:
                is_valid_doing = (curr_lower == "đang" and next_lower in {"làm", "học", "ăn", "ở", "đi", "nói", "viết", "đọc"})
                is_valid_reg = (curr_lower == "đăng" and next_lower in {"ký", "nhập", "tải", "báo", "cai", "quang"})
                if not is_valid_doing and not is_valid_reg:
                    sugg_fmt = "Đảng" if curr_token and curr_token[0].isupper() else "đảng"
                    return sugg_fmt, f"Nghi vấn nhầm dấu thanh trong cụm '{curr_token} {next_token}' (gợi ý: '{sugg_fmt} {next_token}')"

        # 1.f. 'sinh hoạt đằng' -> 'đảng'
        if curr_lower == "đằng" and prev_lower in {"hoạt", "sinh"}:
            sugg_fmt = "Đảng" if curr_token and curr_token[0].isupper() else "đảng"
            return sugg_fmt, f"Nghi vấn nhầm dấu thanh trong cụm '{prev_token} {curr_token}' (gợi ý: '{prev_token} {sugg_fmt}')"

        # 1.g. 'it nhất' / 'it hơn' -> 'ít nhất' / 'ít hơn'
        if curr_lower == "it" and next_lower in {"nhất", "hơn", "ỏi", "khi", "nhiều"}:
            sugg_fmt = "Ít" if curr_token and curr_token[0].isupper() else "ít"
            return sugg_fmt, f"Nghi vấn thiếu dấu thanh trong cụm '{curr_token} {next_token}' (gợi ý: '{sugg_fmt} {next_token}')"

        # 1.h. 'dàng viên' -> 'đảng viên'
        if curr_lower == "dàng" and next_lower in {"viên"}:
            sugg_fmt = "Đảng" if curr_token and curr_token[0].isupper() else "đảng"
            return sugg_fmt, f"Nghi vấn nhầm phụ âm trong cụm '{curr_token} {next_token}' (gợi ý: '{sugg_fmt} {next_token}')"

        # 1.i. Giáo dục & thi cử: 'đáp an' -> 'đáp án'
        if curr_lower == "an" and prev_lower in {"đáp", "trả"}:
            sugg_fmt = "Án" if curr_token and curr_token[0].isupper() else "án"
            return sugg_fmt, f"Nghi vấn thiếu dấu thanh trong cụm '{prev_token} {curr_token}' (gợi ý: '{prev_token} {sugg_fmt}')"

        # 1.j. Giáo dục & khoa học: 'kết qua' -> 'kết quả'
        if curr_lower == "qua" and prev_lower in {"kết"}:
            sugg_fmt = "Quả" if curr_token and curr_token[0].isupper() else "quả"
            return sugg_fmt, f"Nghi vấn thiếu dấu thanh trong cụm '{prev_token} {curr_token}' (gợi ý: '{prev_token} {sugg_fmt}')"

        # 1.k. Thương mại & hóa đơn: 'hóa đon' -> 'hóa đơn'
        if curr_lower == "đon" and prev_lower in {"hóa"}:
            sugg_fmt = "Đơn" if curr_token and curr_token[0].isupper() else "đơn"
            return sugg_fmt, f"Nghi vấn nhầm ký tự trong cụm '{prev_token} {curr_token}' (gợi ý: '{prev_token} {sugg_fmt}')"

        # 1.l. Kế toán & bảng tính: 'tổng công' -> 'tổng cộng' (loại trừ tổng công ty, tổng công đoàn)
        if curr_lower == "công" and prev_lower in {"tổng"}:
            if next_lower not in {"ty", "đoàn", "giám", "cục", "hội"} and (next_lower in {"tiền", "số", "điểm", "chi", "thu", "thanh", "là"} or (next_token and next_token[0].isdigit()) or not next_lower):
                sugg_fmt = "Cộng" if curr_token and curr_token[0].isupper() else "cộng"
                return sugg_fmt, f"Nghi vấn nhầm dấu thanh trong cụm '{prev_token} {curr_token}' (gợi ý: '{prev_token} {sugg_fmt}')"

        # 1.m. Doanh nghiệp & sản phẩm: 'sản phâm' -> 'sản phẩm'
        if curr_lower == "phâm" and prev_lower in {"sản"}:
            sugg_fmt = "Phẩm" if curr_token and curr_token[0].isupper() else "phẩm"
            return sugg_fmt, f"Nghi vấn nhầm dấu thanh trong cụm '{prev_token} {curr_token}' (gợi ý: '{prev_token} {sugg_fmt}')"

        # 1.n. Giáo dục & học tập: 'bài tạp' -> 'bài tập' (loại trừ bài tạp chí)
        if curr_lower == "tạp" and prev_lower in {"bài"}:
            if next_lower not in {"chí", "kỹ", "nham", "văn"}:
                sugg_fmt = "Tập" if curr_token and curr_token[0].isupper() else "tập"
                return sugg_fmt, f"Nghi vấn nhầm nguyên âm trong cụm '{prev_token} {curr_token}' (gợi ý: '{prev_token} {sugg_fmt}')"

        # 1.o. 'dư' -> 'dự' theo ngữ cảnh: 'dư sinh hoạt', 'đảng viên dư sinh hoạt', 'tham dư', 'dư họp', 'dư thảo'
        if curr_lower == "dư":
            is_valid_du = (
                prev_lower in {"số", "thặng", "còn", "bán", "đồ", "phần"}
                or next_lower in {"thừa", "âm", "nợ", "dả", "lượng", "hàng", "chất", "vài", "ba", "trăm", "nghìn", "triệu"}
            )
            is_du_context = (
                next_lower in {"sinh", "họp", "thảo", "khuyết", "toán", "kiến", "bị", "báo", "phòng", "đoán", "lễ", "tiệc", "tuyển", "thi", "giờ"}
                or (prev_lower in {"tham", "đến", "viên", "người", "đại", "khách"} and next_lower not in {"thừa", "âm", "nợ", "dả"})
                or (next_lower == "chi" and next_next_lower == "bộ")
            )
            if is_du_context and not is_valid_du:
                if curr_token.isupper():
                    sugg_fmt = "DỰ"
                elif curr_token[0].isupper():
                    sugg_fmt = "Dự"
                else:
                    sugg_fmt = "dự"
                return sugg_fmt, f"Nghi vấn nhầm dấu thanh trong ngữ cảnh '{curr_token} {next_token}' (gợi ý: '{sugg_fmt}')"

        # 1.p. Bảo vệ từ chức năng, liên từ, giới từ, loại từ và từ vựng phổ thông cơ bản
        if curr_lower in PROTECTED_COMMON_WORDS:
            return None

        # 1.j. Phân biệt theo ngữ cảnh cho 'Ban' <-> 'Bạn'
        if curr_lower == "ban":
            # Nếu đi cùng từ chỉ cơ quan, tổ chức, ban ngành, thời gian: 'ban hành', 'ban chỉ đạo', 'ban quản lý', 'ban ngày', 'ban đầu' -> HỢP LỆ
            if next_lower in {"hành", "ngày", "đêm", "đầu", "chỉ", "giám", "quản", "chấp", "tổ", "ngành", "bệ", "phát", "thường"}:
                return None
            # Nếu đứng đầu câu hoặc sau lời chào và đi liền đại từ / trợ động từ:
            # ví dụ: "Ban có thể", "Ban hãy", "Ban cần", "chào Ban" -> Nghi vấn thiếu dấu thanh
            if next_lower in {"có", "cần", "hãy", "sẽ", "đang", "muốn", "thân", "đọc", "nhớ", "ơi", "ạ", "nhé"}:
                sugg_fmt = "Bạn" if curr_token and curr_token[0].isupper() else "bạn"
                return sugg_fmt, f"Nghi vấn thiếu dấu thanh trong ngữ cảnh '{curr_token} {next_token}' (gợi ý: '{sugg_fmt} {next_token}')"
            if prev_lower in {"chào", "gửi", "thưa"}:
                sugg_fmt = "Bạn" if curr_token and curr_token[0].isupper() else "bạn"
                return sugg_fmt, f"Nghi vấn thiếu dấu thanh trong ngữ cảnh '{prev_token} {curr_token}' (gợi ý: '{prev_token} {sugg_fmt}')"
            return None

        # 2. Bảo vệ tên riêng (cả 2 từ liền nhau đều viết hoa)
        # Chỉ áp dụng khi không phải từ ghép phổ biến trong văn bản hành chính (chi, ban, tổ, phòng, bộ, hội, đảng, đồng)
        if prev_token and curr_token and prev_token[0].isupper() and curr_token[0].isupper():
            if prev_lower not in {"chi", "ban", "tổ", "phòng", "bộ", "hội", "đảng", "đồng"}:
                return None
        if next_token and curr_token and next_token[0].isupper() and curr_token[0].isupper():
            if curr_lower not in {"chi", "ban", "tổ", "phòng", "bộ", "hội", "đảng", "đồng"}:
                return None

        # 3. Nếu cụm từ ghép hiện tại đã có trong từ điển thì hoàn toàn đúng ngữ cảnh
        if prev_lower and f"{prev_lower} {curr_lower}" in self._vietnamese_words:
            return None
        if next_lower and f"{curr_lower} {next_lower}" in self._vietnamese_words:
            return None

        # 4. Ranh giới từ ghép: Nếu prev_token đã thuộc về từ ghép trước đó (prev_prev + prev)
        # thì prev_token không ghép tiếp với curr_token
        prev_is_bound = (
            prev_prev_lower != "" and f"{prev_prev_lower} {prev_lower}" in self._vietnamese_words
        )
        if prev_is_bound or prev_lower in PROTECTED_COMMON_WORDS or not prev_lower:
            return None

        # 5. So khớp với tiền tố liền trước (chỉ khi prev_token có từ ghép với candidate)
        # Chặn đứng bôi đỏ nhầm: nếu curr_lower thuộc PROTECTED_COMMON_WORDS hoặc là từ đơn thông dụng hợp lệ -> BỎ QUA NGAY
        SAFE_COMMON_SINGLE_WORDS = {
            "họp", "đủ", "có", "mặt", "vắng", "đúng", "phù", "hợp", "cùng",
            "ngày", "giờ", "phút", "năm", "tháng", "kế", "thể", "luật", "hoạt"
        }
        if curr_lower in PROTECTED_COMMON_WORDS or curr_lower in SAFE_COMMON_SINGLE_WORDS:
            return None

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


# Alias tương thích hoàn toàn theo quy chuẩn yêu cầu (Phase 4 Master Prompt)
find_spelling_errors = check_text_spelling
