#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
app.core.spell_corrector
========================
Module kiểm tra chính tả tiếng Việt thủ công cho ứng dụng OCR.

NGUYÊN TẮC QUAN TRỌNG NHẤT:
- CHỈ KIỂM TRA VÀ ĐÁNH DẤU, KHÔNG TỰ ĐỘNG SỬA, KHÔNG VIẾT LẠI VÀ KHÔNG THAY THẾ
  BẤT KỲ NỘI DUNG OCR NÀO.
- Bảo toàn tuyệt đối 100% nội dung gốc, thứ tự từ, dấu câu, khoảng trắng,
  xuống dòng và thụt lề.

Tính năng:
1. Bộ thẩm định âm tiết tiếng Việt (VALID_SYLLABLES):
   - Hơn 7.700 âm tiết chuẩn tiếng Việt (bao gồm cả phong cách đặt dấu mới
     và truyền thống).
   - Bảo vệ toàn bộ các từ tiếng Việt hợp lệ (ví dụ: 'bô', 'thê', 'bộc',...).
2. Bảo vệ thông minh (Whitelisting):
   - Số đếm, ngày tháng, thời gian, số liệu, tiền tệ, đơn vị đo.
   - Mã hiệu văn bản hành chính (ví dụ: '170-KL/TW', '2485/QĐ-BVHTTDL', '63/2010/NĐ-CP').
   - Từ viết tắt in hoa (UBND, HĐND, QĐ, CPU, VIP, TW, BCĐTW,...).
   - URL, Email, đường dẫn web/file.
   - Từ tiếng Anh / thuật ngữ quốc tế phổ biến trong văn bản hỗn hợp (qua từ điển 82.000+ từ).
   - Tên riêng, địa danh, tên tổ chức viết hoa hợp lệ.
   - Ký tự mục lục / số thứ tự (a, b, c, x, y,...).
3. Phát hiện lỗi chính tả & lỗi OCR:
   - Âm tiết sai cấu trúc theo ngữ âm / chữ viết tiếng Việt.
   - Từ có phụ âm cuối tắc (c, ch, p, t) thiếu dấu thanh hoặc mang thanh không hợp lệ (ví dụ: 'đap', 'tuc').
   - Lỗi vị trí dấu thanh / sai tổ hợp nguyên âm (ví dụ: 'họach', 'sóat').
   - Lỗi rụng dấu thanh / sai dấu do OCR (ví dụ: 'đươc', 'thưc').
   - Lỗi gõ Telex / ký tự bất thường ở đuôi (ví dụ: 'kiêmr').
4. API chuẩn xác:
   - check_text_spelling(text: str) -> List[Tuple[int, int, str]]
   - start_idx, end_idx tương ứng chính xác với chỉ số Unicode của chuỗi Python đầu vào.
"""

from __future__ import annotations

import functools
import os
import re
import threading
import unicodedata
from pathlib import Path
from typing import List, Optional, Set, Tuple

from loguru import logger

# ---------------------------------------------------------------------------
# Hằng số Ngữ âm & Ký tự Tiếng Việt
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
}

# Regex nhận diện các thành phần được bảo vệ (Whitelisting):
# - URLs, emails
# - Ngày tháng, thời gian, số liệu có phân cách
# - Số liệu kèm đơn vị đo lường (%/kg/m/đ/...)
# - Mã hiệu văn bản hành chính (ví dụ: 170-KL/TW, 2485/QĐ-BVHTTDL, 63/2010/NĐ-CP)
# - Từ viết tắt có dấu gạch chéo hoặc dấu chấm (TP.HCM, QĐ-UBND,...)
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
# Lớp Kiểm Tra Chính Tả Tiếng Việt
# ---------------------------------------------------------------------------


class VietnameseSpellChecker:
    """
    Lớp lõi thực hiện kiểm tra chính tả tiếng Việt cho ứng dụng OCR.

    Cơ chế thẩm định:
    1. VALID_SYLLABLES: Tập hợp các âm tiết tiếng Việt hợp lệ được tải một lần
       duy nhất vào bộ nhớ và tái sử dụng.
    2. ENGLISH_WORDS: Tập từ vựng tiếng Anh (82.000+ từ) để nhận biết văn bản hỗn hợp.
    3. Bộ quy tắc ngữ âm tiếng Việt phát hiện âm tiết sai cấu trúc, lỗi thanh điệu
       với phụ âm cuối tắc, lỗi gõ Telex, lỗi dập dấu.
    4. Không tự động sửa từ, chỉ trả về danh sách (start_idx, end_idx, token).
    """

    def __init__(self, syllables_file: Optional[str] = None) -> None:
        self._valid_syllables: Set[str] = set()
        self._english_words: Set[str] = set()
        self._lock = threading.Lock()
        self._is_loaded = False
        self._syllables_file = syllables_file or os.path.join(
            os.path.dirname(__file__), "vietnamese_syllables.txt"
        )
        self._load_datasets()

    def _load_datasets(self) -> None:
        """Tải tập âm tiết tiếng Việt và từ điển tiếng Anh."""
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
                    logger.debug(
                        f"Loaded {len(self._valid_syllables)} syllables from {self._syllables_file}"
                    )
                except Exception as exc:
                    logger.error(f"Error loading syllables file: {exc}")

            # Đảm bảo bổ sung cả 2 phong cách đặt dấu thanh (truyền thống & mới)
            # ví dụ: 'hòa' và 'hoà', 'thủy' và 'thuỷ', 'khỏe' và 'khoẻ'
            self._expand_tone_variations()

            # 2. Tải từ điển tiếng Anh từ symspellpy nếu có
            try:
                import importlib.util

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
                        logger.debug(
                            f"Loaded {len(self._english_words)} English words from symspellpy."
                        )
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

    # ------------------------------------------------------------------
    # Thẩm định Token độc lập (với @lru_cache)
    # ------------------------------------------------------------------

    @functools.lru_cache(maxsize=16384)
    def is_valid_token(self, token_lower: str, is_capitalized: bool, has_vietnamese_char: bool) -> bool:
        """
        Kiểm tra tính hợp lệ của một token độc lập.
        Sử dụng LRU Cache để tối ưu hóa hiệu năng trên văn bản dài.
        """
        # 1. Ký tự đơn (a, b, c, x, y,...) luôn hợp lệ (dùng làm đề mục hoặc biến số)
        if len(token_lower) <= 1:
            return True

        # 2. Thuộc tập âm tiết tiếng Việt hợp lệ?
        if token_lower in self._valid_syllables:
            return True

        # 3. Chứa ký tự / dấu thanh tiếng Việt nhưng KHÔNG thuộc tập âm tiết hợp lệ?
        # -> Chắc chắn là lỗi cấu trúc âm tiết, lỗi dấu, hoặc lỗi OCR
        if has_vietnamese_char:
            return False

        # 4. Token thuần ký tự Latin ASCII (không dấu):
        # - Thuộc từ điển tiếng Anh / từ mượn quốc tế?
        if token_lower in self._english_words:
            return True

        # - Tên riêng / danh từ riêng viết hoa (Obama, Tesla, Tokyo, Covid,...)?
        if is_capitalized:
            return True

        # - Các trường hợp còn lại (như 'tuc', 'gap', 'dduoc', 'neen', 'asdf',...):
        # Không phải từ tiếng Anh, không phải âm tiết tiếng Việt hợp lệ -> Nghi vấn lỗi.
        return False

    # ------------------------------------------------------------------
    # API Kiểm Tra Chính Tả Văn Bản
    # ------------------------------------------------------------------

    def check_text_spelling(self, text: str) -> List[Tuple[int, int, str]]:
        """
        Dò soát toàn bộ văn bản và trả về danh sách các vị trí nghi vấn chính tả.

        Args:
            text: Văn bản đầu vào nguyên bản lấy từ Textbox.

        Returns:
            Danh sách tuple: (start_idx, end_idx, token)
            - start_idx: vị trí bắt đầu trong chuỗi text (0-indexed).
            - end_idx: vị trí kết thúc trong chuỗi text (exclusive).
            - token: chuỗi con nguyên bản text[start_idx:end_idx].
            Được sắp xếp theo start_idx tăng dần, không có trùng lặp.
        """
        if not text or not text.strip():
            return []

        # 1. Thu thập các vùng được bảo vệ (URLs, emails, ngày tháng, mã công văn)
        protected_spans: List[Tuple[int, int]] = []
        for m in RE_PROTECTED.finditer(text):
            protected_spans.append((m.start(), m.end()))

        suspicious_list: List[Tuple[int, int, str]] = []

        # 2. Duyệt qua từng từ trong văn bản
        for m in RE_WORD.finditer(text):
            start_idx = m.start()
            end_idx = m.end()
            raw_token = m.group()

            # Bỏ qua nếu token nằm trong vùng được bảo vệ
            in_protected = False
            for p_start, p_end in protected_spans:
                if not (end_idx <= p_start or start_idx >= p_end):
                    in_protected = True
                    break
            if in_protected:
                continue

            # Bảo vệ từ viết tắt in hoa toàn bộ (UBND, HĐND, QĐ, CPU, VIP, TW, BCĐTW,...)
            if len(raw_token) >= 2 and raw_token.isupper():
                continue

            # Chuẩn hóa tạm thời để so khớp từ điển (không đổi chuỗi gốc text)
            norm_token = unicodedata.normalize("NFC", raw_token)
            token_lower = norm_token.lower()
            is_capitalized = raw_token[0].isupper()
            has_vietnamese = any(c in VIETNAMESE_DIACRITICS for c in norm_token)

            # Thẩm định tính hợp lệ của token
            valid = self.is_valid_token(token_lower, is_capitalized, has_vietnamese)
            if not valid:
                suspicious_list.append((start_idx, end_idx, raw_token))

        # 3. Sắp xếp kết quả theo start_idx tăng dần
        suspicious_list.sort(key=lambda item: item[0])

        return suspicious_list


# ---------------------------------------------------------------------------
# Singleton Instance & Module-Level API Bắt Buộc
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


def check_text_spelling(text: str) -> List[Tuple[int, int, str]]:
    """
    API bắt buộc: Kiểm tra chính tả tiếng Việt trên văn bản nguyên bản.

    Args:
        text: Chuỗi văn bản cần kiểm tra.

    Returns:
        List[Tuple[int, int, str]]: Danh sách (start_idx, end_idx, token)
        tương ứng chính xác với chỉ số Unicode của Python trên chuỗi đầu vào.
    """
    checker = get_default_spell_checker()
    return checker.check_text_spelling(text)
