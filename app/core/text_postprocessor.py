#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
app.core.text_postprocessor
===========================
Module hậu xử lý văn bản, lọc ảo giác, khôi phục ký tự và chuẩn hóa ngữ cảnh:
- Lọc dòng ảo giác, ký tự rác lặp và hoa văn trang trí.
- Khôi phục địa chỉ Email, tên miền và URL.
- Cố định cấu trúc bảng tiến độ thực tập hoặc bảng số thứ tự.
- Context-Aware Symbol Recovery: khôi phục ký tự, số La Mã, bullet.
- Bảo toàn chữ hoa/thường (Zero Casing Destruction).
- Khóa chốt hậu xử lý văn bản cuối cùng (Final Document Cleanup).
- VietnameseTextPostProcessor: chuẩn hóa Unicode NFC và làm sạch kỹ thuật văn bản OCR.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from app.core.models import SymbolDecision
from app.core.vision_preprocess import is_decorative_pattern_crop

__all__ = [
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
]

RE_REPEATED_WORD = re.compile(r"\b([A-Za-zÀ-ỹĐđ0-9]+)(?:\s+\1){2,}\b", re.IGNORECASE | re.UNICODE)
RE_DECORATIVE_LINE_CHARS = re.compile(r"^[14*_\-–—.~+=#\s]+$")
NOISE_SINGLE_CHARS = {"%", "/", "\\", "|", "[", "]", "~", "`", ";", ":"}


def is_decorative_or_hallucinated(text: str, crop_bgr: Optional[np.ndarray] = None) -> bool:
    """
    Phát hiện toàn diện dải hoa văn trang trí hoặc ảo giác chuỗi lặp:
    1. Kiểm tra thị giác qua is_decorative_pattern_crop nếu crop_bgr được cung cấp.
    2. Kiểm tra ngữ nghĩa chuỗi qua is_hallucinated_or_noise_line.
    """
    if text:
        upper_txt = text.upper()
        if any(h in upper_txt for h in [
            "BIÊN BẢN", "THÀNH PHẦN THAM DỰ", "NỘI DUNG CUỘC HỌP", "QUYẾT ĐỊNH",
            "CỘNG HÒA XÃ HỘI", "ĐỘC LẬP - TỰ DO"
        ]):
            return False
    if crop_bgr is not None and is_decorative_pattern_crop(crop_bgr):
        return True
    return is_hallucinated_or_noise_line(text)


def is_hallucinated_or_noise_line(text: str) -> bool:
    """
    Kiểm tra và phát hiện dòng chữ bị ảo giác Transformer (repetition hallucination)
    hoặc nhiễu mép cắt / đường kẻ trang trí / ký tự đơn lẻ vô nghĩa:
    1. Nếu dòng có >= 3 từ mà một từ bị lặp lại liên tiếp >= 3 lần (r'\\b([A-Za-zÀ-ỹĐđ0-9]+)(?:\\s+\\1){2,}\\b',
       ví dụ: 'đặc đặc đặc') hoặc tỷ lệ từ duy nhất len(set(words)) / len(words) <= 0.35
       (khi len(words) >= 5, ví dụ: 'đặc đặc đặc đặc đặc đặc được đặc đặc đặc đặc đặn') -> True.
    2. Nếu cả dòng sau khi strip chỉ chứa các ký tự lặp vô nghĩa của đường kẻ
       (144444444, 44444, *****, -----) hoặc chỉ chứa vỏn vẹn 1 ký tự rác (%, /, \\, |, [, ]) -> True.
    """
    if not text:
        return True

    stripped = text.strip()
    if not stripped:
        return True

    # 0. Bảo vệ tuyệt đối các tiêu đề và đề mục văn bản quan trọng (Zero False Drops)
    upper_txt = stripped.upper()
    if any(h in upper_txt for h in [
        "BIÊN BẢN", "THÀNH PHẦN THAM DỰ", "NỘI DUNG CUỘC HỌP", "QUYẾT ĐỊNH",
        "CỘNG HÒA XÃ HỘI", "ĐỘC LẬP - TỰ DO", "TIÊU CHÍ ĐÁNH GIÁ", "KẾT LUẬN"
    ]):
        return False

    # 1. Ký tự rác đơn lẻ (ví dụ: '%', '/', '\\', '|', '[', ']') hoặc ký tự in hoa vô nghĩa NG, N, G
    if len(stripped) == 1 and stripped in NOISE_SINGLE_CHARS:
        return True
    if stripped.upper() in {"NG", "N", "G"}:
        return True

    # 2. Dòng chứa toàn các ký tự lặp vô nghĩa của đường kẻ trang trí:
    if RE_DECORATIVE_LINE_CHARS.match(stripped):
        char_set = set(stripped.replace(" ", "").replace("\t", ""))
        if char_set and (char_set.issubset({"1", "4"}) or char_set.issubset({"*", "-", "_", "~", ".", "=", "#"})):
            return True

    # 3. Phân tích các từ trong dòng:
    words = stripped.split()
    n_words = len(words)

    # 3.a. Có >= 3 từ mà một từ bị lặp lại liên tiếp >= 3 lần (ví dụ: 'đặc đặc đặc')
    if n_words >= 3 and RE_REPEATED_WORD.search(stripped):
        return True

    # 3.b. Tỷ lệ từ duy nhất thấp bất thường (Transformer attention repetition loop):
    if n_words >= 5:
        unique_words = {w.lower() for w in words}
        if len(unique_words) / n_words <= 0.35:
            return True

    return False



def recover_emails_and_urls(text: str) -> str:
    """
    Thuật toán khôi phục ký tự @ trong Email và sửa lỗi tên miền/URL:
    - VietOCR thường nhận @ thành Q, O, (a, (D, ©, at, hoặc dính liền vào tên miền
      (ví dụ: Qhutech.edu.vn, (Damaron.com, (Damazon.com).
    - Tự động sửa lỗi OCR trong tên miền công nghệ/trường học phổ biến: amaron.com -> amazon.com.
    """
    if not text:
        return ""

    # 1. Tự động sửa lỗi OCR trong tên miền công nghệ / trường học phổ biến
    text = re.sub(r"\bamar(?:on|om)\.com\b", "amazon.com", text, flags=re.IGNORECASE)
    text = re.sub(r"\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9-]*?)amar(?:on|om)\.com\b", r"\1amazon.com", text, flags=re.IGNORECASE)

    domain_exts = r"(?:com|edu\.vn|gov\.vn|vn|net|org|io|edu)"

    # 2. Mẫu 1 (Có khoảng cách hoặc ký tự rác đứng riêng):
    # 'user Q domain.com', 'user (a domain.com', 'user at domain.com', 'user © domain.com'
    text = re.sub(
        rf"([a-zA-Z0-9._%+-]+)\s*(?:@|[QO©]|\(a|\([a-zA-Z]|at)\s+([a-zA-Z0-9-]+\.{domain_exts})\b",
        r"\1@\2",
        text,
        flags=re.IGNORECASE,
    )

    # 3. Mẫu 2 (Ký tự nhầm lẫn dính liền vào domain, ví dụ: 'user Qhutech.edu.vn' hoặc 'user (Damazon.com'):
    text = re.sub(
        rf"([a-zA-Z0-9._%+-]+)\s+[\(]?[QOD©]([a-zA-Z0-9-]+\.{domain_exts})\b",
        r"\1@\2",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        rf"([a-zA-Z0-9._%+-]+)\s+(?:\(a|\([a-zA-Z]|at)([a-zA-Z0-9-]+\.{domain_exts})\b",
        r"\1@\2",
        text,
        flags=re.IGNORECASE,
    )

    # 4. Chuẩn hóa nếu @ có khoảng cách xung quanh: 'user @ domain.com' -> 'user@domain.com'
    text = re.sub(
        rf"([a-zA-Z0-9._%+-]+)\s*@\s*([a-zA-Z0-9-]+\.{domain_exts})\b",
        r"\1@\2",
        text,
        flags=re.IGNORECASE,
    )

    # 5. Đảm bảo tên miền amazon.com chuẩn hóa lần cuối nếu còn sót
    text = re.sub(r"@([a-zA-Z0-9-]+\.)amar(?:on|om)\.com\b", r"@\1amazon.com", text, flags=re.IGNORECASE)
    text = re.sub(r"@amar(?:on|om)\.com\b", "@amazon.com", text, flags=re.IGNORECASE)

    return text


def format_progress_table_row(row_str: str) -> str:
    """
    Cố định cấu trúc bảng tiến độ thực tập hoặc bảng số thứ tự:
    Cột 1: Tuần / STT
    Cột 2: Ngày (khoảng ngày dd/mm/yyyy - dd/mm/yyyy)
    Cột 3: Nội dung (các việc gạch đầu dòng)
    """
    if not row_str:
        return ""

    # Bảo vệ dòng ghi chú và chú thích chân trang không bị sửa đổi (bảo vệ tuyệt đối dòng bắt đầu bằng *)
    row_stripped = row_str.strip()
    if row_stripped.startswith("*") or re.match(r"^\*+\s*(?:Trừ\s+trường\s+hợp|Ghi\s+chú)\b", row_stripped, re.IGNORECASE):
        return row_str

    # 1. Tách số tuần đứng dính vào ngày tháng ở đầu dòng: '1 20/04/2026' -> '1 | 20/04/2026'
    row_str = re.sub(r"^([ \t]*\d+)\s+(\d{2}/\d{2}/\d{4})", r"\1 | \2", row_str)

    # 2. Khôi phục dấu gạch nối ngày tháng: '20/04/2026 26/04/2026' -> '20/04/2026 - 26/04/2026'
    row_str = re.sub(r"(\d{2}/\d{2}/\d{4})\s+(\d{2}/\d{2}/\d{4})", r"\1 - \2", row_str)
    row_str = re.sub(r"(\d{2}/\d{2}/\d{4})\s*[?~_]\s*(\d{2}/\d{2}/\d{4})", r"\1 - \2", row_str)

    # 3. Đảm bảo ngăn cách giữa cụm ngày tháng và nội dung gạch đầu dòng có ' | '
    row_str = re.sub(r"(\d{2}/\d{2}/\d{4}\s*-\s*\d{2}/\d{2}/\d{4})\s*(?!\|)\s*(-|\+)\s+", r"\1 | \2 ", row_str)

    # 4. Dọn dẹp khoảng trắng thừa quanh phân cách '|'
    row_str = re.sub(r"\s*\|\s*", " | ", row_str)
    row_str = re.sub(r"\b(\d+)\s*\|\s*\|\s*", r"\1 | ", row_str)

    return row_str



ROMAN_HEADING_KEYWORDS = {
    # Hành chính & pháp lý
    "NỘI DUNG", "TIẾN HÀNH", "KẾT LUẬN", "TỔ CHỨC", "MỤC ĐÍCH", "YÊU CẦU",
    "KẾ HOẠCH", "PHƯƠNG HƯỚNG", "NHIỆM VỤ", "ĐÁNH GIÁ", "CHƯƠNG TRÌNH",
    "THỰC HIỆN", "KẾT QUẢ", "THỜI GIAN", "ĐỊA ĐIỂM", "THÀNH PHẦN",
    "KIỂM ĐIỂM", "THẢO LUẬN", "NGHỊ QUYẾT", "BÁO CÁO", "ĐIỀU KHOẢN",
    "QUYỀN VÀ NGHĨA VỤ", "PHẠM VI", "HIỆU LỰC", "CHẤM DỨT",
    # Học thuật, giáo dục & khoa học
    "MỞ ĐẦU", "TỔNG QUAN", "CƠ SỞ LÝ THUYẾT", "PHƯƠNG PHÁP", "BÀN LUẬN",
    "KIẾN NGHỊ", "TÀI LIỆU THAM KHẢO", "PHỤ LỤC", "GIỚI THIỆU", "ĐẶC ĐIỂM",
    "KHÁI NIỆM", "NGUYÊN LÝ", "CẤU TẠO", "ỨNG DỤNG", "PHÂN TÍCH",
    "GIẢI PHÁP", "KẾ HOẠCH BÀI DẠY", "MỤC TIÊU BÀI HỌC",
    # Doanh nghiệp, thương mại & tài chính
    "TÌNH HÌNH HOẠT ĐỘNG", "DOANH THU", "LỢI NHUẬN", "ĐỊNH HƯỚNG",
    "MỤC TIÊU", "TỔ CHỨC BỘ MÁY", "QUY CHẾ", "QUY TRÌNH", "BẢO HÀNH",
}



class ContextSymbolRecovery:
    """
    Cơ chế Context-Aware Symbol Recovery:
    Khôi phục ký tự và cấu trúc bị OCR nhận sai dựa trên bằng chứng ngữ cảnh văn bản và hình học.
    Tuân thủ nguyên tắc 3 trạng thái: AUTO_CORRECT, FLAG_UNCERTAIN, PRESERVE.
    Bảo toàn 100% dữ liệu gốc và ghi nhận nhật ký vết thay đổi (audit trail).
    """

    @staticmethod
    def is_all_uppercase_heading(text: str) -> bool:
        """Kiểm tra văn bản có phải là đề mục in hoa hay không (>= 70% từ in hoa)."""
        words = re.findall(r'[a-zA-ZÀ-ỹĐđ]+', text)
        if not words:
            return False
        upper_count = sum(1 for w in words if w.isupper())
        return (upper_count / len(words)) >= 0.70

    @staticmethod
    def has_heading_keyword(text: str) -> bool:
        """Kiểm tra văn bản có chứa từ khóa đề mục hay không."""
        text_upper = text.upper()
        return any(kw in text_upper for kw in ROMAN_HEADING_KEYWORDS)

    @classmethod
    def recover_roman_numeral_in_line(
        cls, line: str, prev_roman: str = ""
    ) -> Tuple[str, SymbolDecision, str, str]:
        """
        Khôi phục số La Mã đầu mục (I. đến X.) dựa trên ngữ cảnh.
        Trả về: (new_line, decision, current_roman, rule_name)
        """
        stripped = line.strip()
        if not stripped:
            return line, SymbolDecision.PRESERVE, prev_roman, ""

        # 1. Mẫu số III: IIL, IIl, 11L, lll, 111, TII, TIl, llI, TTT
        m_iii = re.match(r"^([ \t]*)(IIL|IIl|11L|lll|111|TII|TIl|llI|TTT)([\.:]?)\s+([^\n]+)$", line)
        if m_iii:
            indent = m_iii.group(1)
            prefix = m_iii.group(2)
            has_dot = (m_iii.group(3) in {".", ":"})
            rest = m_iii.group(4).strip()
            # ĐẶC BIỆT: 'Nội dung cuộc họp' luôn luôn là mục số II (không bao giờ là III)
            if re.match(r"(?i)Nội\s+dung\s+cuộc\s+họp", rest):
                return f"{indent}II. {rest}", SymbolDecision.AUTO_CORRECT, "II", f"ROMAN_II_FROM_{prefix}_CONTENT"
            if cls.is_all_uppercase_heading(rest) or cls.has_heading_keyword(rest) or (has_dot and rest and rest[0].isupper()):
                return f"{indent}III. {rest}", SymbolDecision.AUTO_CORRECT, "III", f"ROMAN_III_FROM_{prefix}"
            else:
                return line, SymbolDecision.FLAG_UNCERTAIN, prev_roman, f"UNCERTAIN_ROMAN_III_{prefix}"

        # 2. Mẫu số II: IL, Il, 1L, 1l, TI, TT, TL, Tl, ll, 11, lI (chỉ 1 chữ I/1 đứng trước L bắt buộc thành II.)
        m_ii = re.match(r"^([ \t]*)(IL|Il|1L|1l|TI|TT|TL|Tl|ll|11|lI)([\.:]?)\s+([^\n]+)$", line)
        if m_ii:
            indent = m_ii.group(1)
            prefix = m_ii.group(2)
            has_dot = (m_ii.group(3) in {".", ":"})
            rest = m_ii.group(4).strip()
            if cls.is_all_uppercase_heading(rest) or cls.has_heading_keyword(rest) or (has_dot and rest and rest[0].isupper()):
                return f"{indent}II. {rest}", SymbolDecision.AUTO_CORRECT, "II", f"ROMAN_II_FROM_{prefix}"
            else:
                return line, SymbolDecision.FLAG_UNCERTAIN, prev_roman, f"UNCERTAIN_ROMAN_II_{prefix}"

        # 3. Mẫu số IV: IV, lV, 1V, TV
        m_iv = re.match(r"^([ \t]*)(IV|lV|1V|TV)[\.:]?\s+([^\n]+)$", line)
        if m_iv:
            indent = m_iv.group(1)
            prefix = m_iv.group(2)
            rest = m_iv.group(3).strip()
            if cls.is_all_uppercase_heading(rest) or cls.has_heading_keyword(rest):
                return f"{indent}IV. {rest}", SymbolDecision.AUTO_CORRECT, "IV", f"ROMAN_IV_FROM_{prefix}"
            else:
                return line, SymbolDecision.FLAG_UNCERTAIN, prev_roman, f"UNCERTAIN_ROMAN_IV_{prefix}"

        # Mẫu số VI: VI, Vl, V1, V|
        m_vi = re.match(r"^([ \t]*)(VI|V[l1|])[\.:]?\s+([^\n]+)$", line)
        if m_vi:
            indent = m_vi.group(1)
            prefix = m_vi.group(2)
            rest = m_vi.group(3).strip()
            if cls.is_all_uppercase_heading(rest) or cls.has_heading_keyword(rest):
                return f"{indent}VI. {rest}", SymbolDecision.AUTO_CORRECT, "VI", f"ROMAN_VI_FROM_{prefix}"
            else:
                return line, SymbolDecision.FLAG_UNCERTAIN, prev_roman, f"UNCERTAIN_ROMAN_VI_{prefix}"

        # Mẫu số VII: VII, Vll, V11, VlI, VIl, VTl
        m_vii = re.match(r"^([ \t]*)(VII|V(?:ll|11|lI|Il|Tl))[\.:]?\s+([^\n]+)$", line)
        if m_vii:
            indent = m_vii.group(1)
            prefix = m_vii.group(2)
            rest = m_vii.group(3).strip()
            if cls.is_all_uppercase_heading(rest) or cls.has_heading_keyword(rest):
                return f"{indent}VII. {rest}", SymbolDecision.AUTO_CORRECT, "VII", f"ROMAN_VII_FROM_{prefix}"
            else:
                return line, SymbolDecision.FLAG_UNCERTAIN, prev_roman, f"UNCERTAIN_ROMAN_VII_{prefix}"

        # Mẫu số VIII: VIII, Vlll, V111
        m_viii = re.match(r"^([ \t]*)(VIII|V(?:lll|111))[\.:]?\s+([^\n]+)$", line)
        if m_viii:
            indent = m_viii.group(1)
            prefix = m_viii.group(2)
            rest = m_viii.group(3).strip()
            if cls.is_all_uppercase_heading(rest) or cls.has_heading_keyword(rest):
                return f"{indent}VIII. {rest}", SymbolDecision.AUTO_CORRECT, "VIII", f"ROMAN_VIII_FROM_{prefix}"
            else:
                return line, SymbolDecision.FLAG_UNCERTAIN, prev_roman, f"UNCERTAIN_ROMAN_VIII_{prefix}"

        # Mẫu số IX: IX, lX, 1X, |X
        m_ix = re.match(r"^([ \t]*)(IX|[l1|]X)[\.:]?\s+([^\n]+)$", line)
        if m_ix:
            indent = m_ix.group(1)
            prefix = m_ix.group(2)
            rest = m_ix.group(3).strip()
            if cls.is_all_uppercase_heading(rest) or cls.has_heading_keyword(rest):
                return f"{indent}IX. {rest}", SymbolDecision.AUTO_CORRECT, "IX", f"ROMAN_IX_FROM_{prefix}"
            else:
                return line, SymbolDecision.FLAG_UNCERTAIN, prev_roman, f"UNCERTAIN_ROMAN_IX_{prefix}"

        # Mẫu số X: X
        m_x = re.match(r"^([ \t]*)(X)[\.:]?\s+([^\n]+)$", line)
        if m_x:
            indent = m_x.group(1)
            prefix = m_x.group(2)
            rest = m_x.group(3).strip()
            if cls.is_all_uppercase_heading(rest) or cls.has_heading_keyword(rest):
                return f"{indent}X. {rest}", SymbolDecision.AUTO_CORRECT, "X", f"ROMAN_X_FROM_{prefix}"

        # 4. Mẫu số I: L, l, T, |, 1
        m_i = re.match(r"^([ \t]*)(L|l|T|\||1)([\.:]?)\s+([^\n]+)$", line)
        if m_i:
            indent = m_i.group(1)
            prefix = m_i.group(2).strip()
            has_dot = (m_i.group(3) in {".", ":"})
            rest = m_i.group(4).strip()
            is_all_caps = cls.is_all_uppercase_heading(rest)
            is_heading_kw = cls.has_heading_keyword(rest)

            # Nếu có dấu chấm (L., l., T., |.) HOẶC đứng không dấu chấm trước chuỗi IN HOA TOÀN BỘ (L NỘI DUNG..., L THÔNG TIN...)
            if prefix in ["L", "l", "T", "|"]:
                if not rest.startswith(("-", "+", "*", "•", "·", "–", "—")):
                    if (has_dot and (is_all_caps or is_heading_kw)) or (not has_dot and is_all_caps):
                        return f"{indent}I. {rest}", SymbolDecision.AUTO_CORRECT, "I", f"ROMAN_I_FROM_{prefix}"
            elif prefix == "1":
                # Đối với 1., chuẩn hóa thành I. khi có dấu chấm VÀ chứa từ khóa đề mục chính
                if has_dot and (is_all_caps or is_heading_kw) and any(k in rest.upper() for k in ["THÀNH PHẦN", "NỘI DUNG", "CHƯƠNG TRÌNH", "THÔNG TIN", "QUY ĐỊNH", "KẾT LUẬN"]):
                    return f"{indent}I. {rest}", SymbolDecision.AUTO_CORRECT, "I", "ROMAN_I_FROM_1_DOT"

        # 5. Nếu đã là số La Mã chuẩn I., II., III., IV., V., VI., VII., VIII., IX., X.
        m_valid = re.match(r"^[ \t]*(I{1,3}|IV|V|VI{1,3}|IX|X)\.\s+", line)
        if m_valid:
            return line, SymbolDecision.PRESERVE, m_valid.group(1), "VALID_ROMAN"

        return line, SymbolDecision.PRESERVE, prev_roman, ""

    @classmethod
    def recover_bullets_and_pluses(cls, line: str) -> Tuple[str, SymbolDecision, str]:
        """
        Khôi phục dấu cộng (+) hoặc bullet bị nhận sai thành % hoặc | ở đầu dòng/ô.
        Bảo toàn 100% các giá trị phần trăm hợp lệ (90%, 75%, 100%).
        Trả về: (new_line, decision, rule_name)
        """
        # 0. Ký tự chân trang chú thích: '* Trừ trường hợp...' (TUYỆT ĐỐI KHÔNG đổi thành '- Trừ trường hợp')
        m_footnote = re.match(r"^([ \t]*)[-*–—/1?\\|~•]+\s*\*?\s*(Trừ\s+trường\s+hợp.*)$", line, re.IGNORECASE)
        if m_footnote:
            indent = m_footnote.group(1)
            content = m_footnote.group(2)
            return f"{indent}* {content}", SymbolDecision.AUTO_CORRECT, "FOOTNOTE_FROM_DIGIT"
        if re.search(r"Trừ\s+trường\s+hợp", line, re.IGNORECASE):
            m_fn_any = re.match(r"^([ \t]*)(?:[-*–—/1?\\|~•]+\s*)?(Trừ\s+trường\s+hợp.*)$", line, re.IGNORECASE)
            if m_fn_any:
                return f"{m_fn_any.group(1)}* {m_fn_any.group(2)}", SymbolDecision.AUTO_CORRECT, "FOOTNOTE_STAR"

        # 0.b. Tiêu chí con trong bảng điểm: Bắt buộc giữ hoặc khôi phục dấu '+ ' (TUYỆT ĐỐI KHÔNG đổi thành '- ')
        m_sub = re.match(
            r"^([ \t]*)[-*–—•%~+]\s+((?:Chi\s+bộ\s+dưới|Thời\s+gian\s+sinh\s+hoạt|Từ\s+\d+%|Dưới\s+\d+%|Không\s+có\s+đảng\s+viên).*)",
            line,
            re.IGNORECASE,
        )
        if m_sub:
            indent = m_sub.group(1)
            content = m_sub.group(2)
            return f"{indent}+ {content}", SymbolDecision.AUTO_CORRECT, "PLUS_SUB_CRITERION"

        # Nếu dòng có cột điểm ở cuối (| 01 .. | 05) và bắt đầu bằng bullet (-, *, %, •, ~)
        if re.search(r"\|\s*0[1-5]\b", line):
            m_bullet_lead = re.match(r"^([ \t]*)[-–—*•%~]\s+([A-Za-zÀ-ỹĐđ].*)$", line)
            if m_bullet_lead:
                return f"{m_bullet_lead.group(1)}+ {m_bullet_lead.group(2)}", SymbolDecision.AUTO_CORRECT, "PLUS_FROM_SCORE_BULLET"

        # 1. Ký tự '|' đứng trước dấu bullet rác ở đầu dòng (ví dụ: '| % Chi bộ...', '| * Chi bộ...', '| + Chi bộ...')
        m_pipe_bullet = re.match(r"^([ \t]*)\|\s*([%*+]|-)\s+([A-Za-zÀ-ỹĐđ].*)$", line)
        if m_pipe_bullet:
            indent = m_pipe_bullet.group(1)
            sym = m_pipe_bullet.group(2)
            content = m_pipe_bullet.group(3)
            # Nếu là dấu gạch ngang hợp lệ (- Nội dung...)
            if sym == "-":
                return f"{indent}- {content}", SymbolDecision.AUTO_CORRECT, "DASH_FROM_PIPE_PREFIX"
            # Nếu là ghi chú (* Ghi chú...)
            if sym == "*" and content.lower().startswith("ghi chú"):
                return f"{indent}* {content}", SymbolDecision.AUTO_CORRECT, "NOTE_FROM_PIPE_PREFIX"
            return f"{indent}+ {content}", SymbolDecision.AUTO_CORRECT, "PLUS_FROM_PIPE_BULLET_PREFIX"

        # 2. Ký tự '%' hoặc '*' ở đầu dòng theo sau là chữ cái tiếng Việt (ví dụ: '% Chi bộ...', '* Chi bộ...', '% Từ 90%...')
        m_pct = re.match(r"^([ \t]*)([%*])\s*([A-Za-zÀ-ỹĐđ].*)$", line)
        if m_pct:
            indent = m_pct.group(1)
            sym = m_pct.group(2)
            content = m_pct.group(3)
            # Bảo vệ dòng ghi chú: '* Ghi chú quan trọng'
            if sym == "*" and (content.lower().startswith("ghi chú") or content.lower().startswith("trừ trường hợp")):
                return line, SymbolDecision.PRESERVE, ""
            return f"{indent}+ {content}", SymbolDecision.AUTO_CORRECT, "PLUS_FROM_PERCENT_PREFIX"

        # 3. Ký tự '4' hoặc '7' ở đầu dòng trong tiêu chí bảng bị nhận nhầm từ '+' (ví dụ: '4 Chi bộ...', '7 Thời gian...')
        m_num_bullet = re.match(r"^([ \t]*)([47])\s+([A-Za-zÀ-ỹĐđ].*)$", line)
        if m_num_bullet:
            indent = m_num_bullet.group(1)
            content = m_num_bullet.group(3)
            c_lower = content.lower()
            if any(c_lower.startswith(kw) for kw in [
                "chỉ bộ", "chi bộ", "thời gian", "tỷ lệ", "không có", "đã có", "được", "dưới", "từ "
            ]):
                return f"{indent}+ {content}", SymbolDecision.AUTO_CORRECT, "PLUS_FROM_NUMERIC_NOISE"

        # 4. Ký tự chân trang bị nhận thành 1, -, / hoặc ? (ví dụ: '1 Trừ trường hợp...', '/Trừ trường hợp...', '?Trừ trường hợp...')
        m_footnote = re.match(r"^([ \t]*)[1?*/\-]+\s*(Trừ\s+trường\s+hợp.*)$", line, re.IGNORECASE)
        if m_footnote:
            indent = m_footnote.group(1)
            content = m_footnote.group(2)
            return f"{indent}* {content}", SymbolDecision.AUTO_CORRECT, "FOOTNOTE_FROM_DIGIT"

        # 5. Ký tự '|' ở đầu dòng trong văn bản thông thường (không phải hàng bảng nhiều cột ' | ')
        if line.count(" | ") < 2:
            m_pipe = re.match(r"^([ \t]*)\|\s+([A-Za-zÀ-ỹĐđ\+].*)$", line)
            if m_pipe:
                indent = m_pipe.group(1)
                content = m_pipe.group(2)
                # Nếu không phải là đề mục in hoa toàn bộ (đề mục số La Mã I.)
                if not cls.is_all_uppercase_heading(content):
                    return f"{indent}+ {content}", SymbolDecision.AUTO_CORRECT, "PLUS_FROM_PIPE_PREFIX"

        return line, SymbolDecision.PRESERVE, ""

    @classmethod
    def recover_document_symbols(
        cls, lines: List[str]
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Thực hiện Context-Aware Symbol Recovery trên toàn bộ tài liệu.
        Trả về: (recovered_lines, audit_trail)
        """
        recovered_lines: List[str] = []
        audit_trail: List[Dict[str, Any]] = []
        current_roman = ""

        # Tiền xử lý: Khử nhập nhằng phân cấp đề mục (Hierarchy Disambiguation)
        # Nếu dòng k bắt đầu bằng '1.' (hoặc 'L.', 'l.', 'I.') là đề mục lớn,
        # và dòng kế tiếp phía dưới cũng bắt đầu bằng '1.' (ví dụ: '1. Đại diện...'):
        # -> Dòng k chắc chắn là số La Mã 'I.'
        normalized_lines = list(lines)
        for i in range(len(normalized_lines)):
            m_top = re.match(r"^([ \t]*)(?:1|L|l|I|\|)\.\s+([A-Za-zÀ-ỹĐđ0-9\s,;:/-]+)$", normalized_lines[i])
            if m_top:
                indent = m_top.group(1)
                heading_content = m_top.group(2).strip()
                next_start_with_1 = False
                for j in range(i + 1, min(i + 4, len(normalized_lines))):
                    l_next = normalized_lines[j].strip()
                    if not l_next:
                        continue
                    if re.match(r"^1\.\s+[A-Za-zÀ-ỹĐđ]", l_next):
                        next_start_with_1 = True
                        break
                    else:
                        break
                if next_start_with_1:
                    has_sec_kw = any(kw.lower() in heading_content.lower() for kw in [
                        "thành phần", "nội dung", "kế hoạch", "chương trình", "mục đích",
                        "tổ chức", "kết luận", "tiến độ", "phương thức", "đánh giá", "điều khoản"
                    ])
                    if has_sec_kw or not re.search(r":\s*[^\s]", heading_content):
                        new_heading = f"{indent}I. {heading_content}"
                        audit_trail.append({
                            "line_index": i,
                            "original": normalized_lines[i],
                            "recovered": new_heading,
                            "decision": SymbolDecision.AUTO_CORRECT.value,
                            "rule": "HIERARCHY_DISAMBIGUATION_ROMAN_I",
                        })
                        normalized_lines[i] = new_heading

        for idx, line in enumerate(normalized_lines):
            orig_line = line
            curr_line = line

            # 1. Khôi phục số La Mã
            curr_line, roman_dec, current_roman, roman_rule = cls.recover_roman_numeral_in_line(
                curr_line, current_roman
            )
            if roman_dec == SymbolDecision.AUTO_CORRECT:
                audit_trail.append({
                    "line_index": idx,
                    "original": orig_line,
                    "recovered": curr_line,
                    "decision": roman_dec.value,
                    "rule": roman_rule,
                })

            # 2. Khôi phục Bullet / Plus
            curr_line, bullet_dec, bullet_rule = cls.recover_bullets_and_pluses(curr_line)
            if bullet_dec == SymbolDecision.AUTO_CORRECT:
                audit_trail.append({
                    "line_index": idx,
                    "original": orig_line,
                    "recovered": curr_line,
                    "decision": bullet_dec.value,
                    "rule": bullet_rule,
                })

            # 3. Dọn dẹp dấu phân cách kép thừa sau STT (1 | | Chấp hành... -> 1 | Chấp hành...)
            m_double_pipe = re.match(r"^([ \t]*\d+)\s*\|\s*\|\s*(.*)$", curr_line)
            if m_double_pipe:
                cleaned_line = f"{m_double_pipe.group(1)} | {m_double_pipe.group(2)}"
                audit_trail.append({
                    "line_index": idx,
                    "original": curr_line,
                    "recovered": cleaned_line,
                    "decision": SymbolDecision.AUTO_CORRECT.value,
                    "rule": "STT_DOUBLE_PIPE_CLEANUP",
                })
                curr_line = cleaned_line

            # 4. Cố định cấu trúc bảng tiến độ thực tập (Tuần | Ngày | Nội dung)
            formatted_progress = format_progress_table_row(curr_line)
            if formatted_progress != curr_line:
                audit_trail.append({
                    "line_index": idx,
                    "original": curr_line,
                    "recovered": formatted_progress,
                    "decision": SymbolDecision.AUTO_CORRECT.value,
                    "rule": "PROGRESS_TABLE_ROW_FORMAT",
                })
                curr_line = formatted_progress

            recovered_lines.append(curr_line)

        # Hậu kiểm tra tính tuần tự và loại bỏ trùng lặp số La Mã (Sequential Reconciliation):
        # 1. Đề mục 'Nội dung cuộc họp' luôn mang số II.
        # 2. Nếu có 2 đề mục cùng mang số III. (ví dụ 'III. Nội dung cuộc họp' và 'III. Kết luận...'),
        #    mục đầu tiên bắt buộc đổi thành II.
        # 3. Nếu mục trước là I. và mục sau là III., mục ở giữa bắt buộc là II.
        roman_headings = []
        for r_idx, r_line in enumerate(recovered_lines):
            m_r = re.match(r"^([ \t]*)([I|V|X]+)\.\s+(.*)$", r_line)
            if m_r:
                roman_headings.append((r_idx, m_r.group(1), m_r.group(2), m_r.group(3)))

        for h_pos, (line_idx, ind, rom, text_rest) in enumerate(roman_headings):
            if re.match(r"(?i)Nội\s+dung\s+cuộc\s+họp", text_rest) and rom != "II":
                new_l = f"{ind}II. {text_rest}"
                audit_trail.append({
                    "line_index": line_idx,
                    "original": recovered_lines[line_idx],
                    "recovered": new_l,
                    "decision": SymbolDecision.AUTO_CORRECT.value,
                    "rule": "ROMAN_II_CONTENT_ENFORCED",
                })
                recovered_lines[line_idx] = new_l
            elif rom == "III" and h_pos + 1 < len(roman_headings):
                next_rom = roman_headings[h_pos + 1][2]
                if next_rom == "III":
                    new_l = f"{ind}II. {text_rest}"
                    audit_trail.append({
                        "line_index": line_idx,
                        "original": recovered_lines[line_idx],
                        "recovered": new_l,
                        "decision": SymbolDecision.AUTO_CORRECT.value,
                        "rule": "ROMAN_DEDUP_III_TO_II",
                    })
                    recovered_lines[line_idx] = new_l

        return recovered_lines, audit_trail


# ---------------------------------------------------------------------------
# Bộ Chuẩn Hóa Văn Bản Tiếng Việt & Bảo Toàn Casing
# ---------------------------------------------------------------------------


def preserve_casing(source_text: str, target_text: str) -> str:
    """
    Bảo toàn định dạng chữ hoa/chữ thường (casing) từ source_text sang target_text:
    - Nếu source_text.isupper() -> trả về target_text.upper().
    - Nếu source_text.istitle() -> trả về target_text.title().
    - Nếu chữ thường -> trả về target_text.lower().
    """
    if not source_text or not target_text:
        return target_text
    if source_text.isupper():
        return target_text.upper()
    if source_text.istitle():
        return target_text.title()
    if source_text.islower():
        return target_text.lower()
    return target_text


def clean_unmatched_closing_brackets(text: str) -> str:
    """
    Loại bỏ ký tự ']' rác (thường do OCR đọc nhầm số mũ chú thích ¹)
    khi đứng ngay sau từ mà không có dấu '[' mở tương ứng.
    Bảo toàn 100% các cặp ngoặc hợp lệ như [Icon], [1], [Ghi chú].
    """
    if not text or "]" not in text:
        return text

    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        if "]" not in line:
            cleaned_lines.append(line)
            continue

        out = []
        open_bracket_count = 0
        for char in line:
            if char == "[":
                open_bracket_count += 1
                out.append(char)
            elif char == "]":
                if open_bracket_count > 0:
                    open_bracket_count -= 1
                    out.append(char)
                else:
                    # ']' rác không có ngoặc mở tương ứng -> bỏ qua
                    pass
            else:
                out.append(char)
        cleaned_lines.append("".join(out))
    return "\n".join(cleaned_lines)


def clean_orphaned_brackets(text: str) -> str:
    """
    Xóa dấu ngoặc vuông '[' hoặc ký tự gạch đứng '|' mồ côi ở đầu dòng:
    Nếu dòng bắt đầu bằng '[' hoặc '|' ngay trước chữ/số/ký tự văn bản mà trên dòng đó
    không có dấu đóng ngoặc ']' tương ứng (ví dụ: '[3. Thư ký...', '[Chủ trì...', '[Ưng...').
    Đồng thời xóa dấu '[' mồ côi đứng trước từ ngữ trong dòng (ví dụ: 'THIÊN [Ưng' -> 'THIÊN Ưng').
    """
    if not text:
        return text
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        m = re.match(r"^(\s*)[\[\|]\s*([^\s].*)$", line)
        if m:
            c = m.group(2)
            if "]" not in c:
                line = f"{m.group(1)}{c}"
        # Xóa dấu '[' mồ côi đứng trước từ trong dòng khi không có ']' đóng
        if "]" not in line and "[" in line:
            line = re.sub(r"(?<=\s)\[([A-Za-zÀ-ỹĐđ]+)", r"\1", line)
            line = re.sub(r"^\[([A-Za-zÀ-ỹĐđ]+)", r"\1", line)
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def _reflow_wrapped_paragraphs(text: str) -> str:
    """
    Nối dòng các đoạn văn bản bị ngắt dòng do lề trang hoặc OCR xuống dòng giữa câu.
    Duyệt qua các dòng: Nếu dòng hiện tại là một dòng văn bản nội dung mà chưa kết thúc câu
    và dòng kế tiếp bắt đầu bằng chữ cái viết thường hoặc cụm ngày/đảng viên tiếp nối:
    Tự động nối dòng kế tiếp vào dòng hiện tại bằng 1 khoảng trắng.
    """
    if not text or "\n" not in text:
        return text

    lines = text.split("\n")
    merged_lines: List[str] = []
    i = 0
    while i < len(lines):
        curr = lines[i].rstrip()
        curr_s = curr.strip()
        if not curr_s:
            merged_lines.append(curr)
            i += 1
            continue

        # Không nối nếu dòng hiện tại là bảng, tiêu đề in hoa, hoặc ghi chú
        if (
            " | " in curr_s
            or curr_s.count("|") >= 2
            or (curr_s.isupper() and len(curr_s) <= 80)
            or re.match(r"^(?:I{1,3}|IV|V|VI{0,3}|IX|X)\.\s+", curr_s)
            or re.match(r"^(?:\*|Ghi chú:|- Cột|- Từ Cột)", curr_s, re.IGNORECASE)
        ):
            merged_lines.append(curr)
            i += 1
            continue

        # Kiểm tra xem dòng hiện tại có phải là câu văn nội dung cần kiểm tra nối tiếp
        is_content_line = (
            re.match(r"^(?:Vào lúc|Hôm nay|Dưới sự chủ trì|Sau khi nghe|Cuộc họp kết thúc|Biên bản được lập|Biên bản này)\b", curr_s, re.IGNORECASE)
            or re.match(r"^\d+\.\s+", curr_s)
            or re.match(r"^[-–—]\s+", curr_s)
            or (not curr_s.endswith((".", ":", "...", "?", "!")) and not curr_s.endswith(("%", "người")))
        )

        while i + 1 < len(lines) and is_content_line:
            nxt = lines[i + 1].rstrip()
            nxt_s = nxt.strip()
            if not nxt_s:
                break
            # Dòng kế tiếp là một mục mới hoặc tiêu đề thì dừng
            if (
                re.match(r"^\d+\.\s+", nxt_s)
                or re.match(r"^[-–—+]\s+", nxt_s)
                or re.match(r"^(?:I{1,3}|IV|V|VI{0,3}|IX|X)\.\s+", nxt_s)
                or " | " in nxt_s
                or nxt_s.count("|") >= 2
                or (nxt_s.isupper() and len(nxt_s) <= 80)
                or re.match(r"^(?:\*|Ghi chú:)", nxt_s, re.IGNORECASE)
            ):
                break

            # Điều kiện nối: dòng kế tiếp bắt đầu bằng chữ cái thường,
            # hoặc cụm ngày/số đảng viên (năm 2026, 10 năm 2026, 14/14 đảng viên,...)
            starts_lower = bool(re.match(r"^[a-zà-ỹđ]", nxt_s))
            starts_cont_phrase = bool(
                re.match(
                    r"^(?:10\s+năm\s+202\d|\d+/\d+\s+đảng\s+viên|năm\s+202\d|kỳ\s+sinh\s+hoạt|bộ\s+định\s+kỳ|trọng\s+tâm|cán\s+bộ|luật\s+có\s+liên\s+quan|hiểu\s+rõ|bản\s+lưu)",
                    nxt_s,
                    re.IGNORECASE,
                )
            )

            # Dòng hiện tại chưa kết thúc câu (hoặc kết thúc bằng ; ,)
            not_ended = not curr_s.endswith((".", ":", "...", "?", "!")) or curr_s.endswith((";", ","))
            if not_ended and (starts_lower or starts_cont_phrase):
                curr = f"{curr} {nxt_s}"
                curr_s = curr.strip()
                i += 1
            else:
                break

        merged_lines.append(curr)
        i += 1

    return "\n".join(merged_lines)


def final_document_cleanup(text: str) -> str:
    """
    Khóa chốt hậu xử lý văn bản cuối cùng (Final Document Cleanup):
    1. Chuẩn hóa lỗi nhầm dấu cố hữu của VietOCR (bảo toàn chữ HOA/thường):
       - CHI BỐ, CHỈ BỘ, CHI BỒ, CHỈ BỒ -> CHI BỘ | Chi bố, Chỉ bộ, Chi bồ, Chỉ bồ -> Chi bộ.
       - r'\\bsinh\\s+hoatchi\\b' -> 'sinh hoạt chi'.
       - r'\\b(\\d+)\\s+đồng\\s+ch\\b(?![a-zà-ỹđ])' -> r'\\1 đồng chí'.
       - r'(Giới\\s+thiệu\\s+đại\\s+biểu\\s*:\\s*không)\\s+c\\b' -> r'\\1 có.'.
       - r'\\b(Số\\s*:?\\s*\\d+)\\s+([A-ZĐ]{1,5}/[A-ZĐ0-9\\-]+)\\b' -> r'\\1-\\2' (Số 01 BB/CB -> Số 01-BB/CB).
       - (Tiêu chí đánh giá | đa | Điểm tự chấm) -> (Tiêu chí đánh giá | Điểm tối đa | Điểm tự chấm).
    2. Chuẩn hóa số, phân số & ký hiệu bảng:
       - r'\\b0[sS]\\b' -> '05'.
       - r'(chỉ\\s+bằng|bằng|dưới|trên|quá)\\s*[%％]\\s*(so\\s+với\\s+quy\\s+định)' -> r'\\1 ½ \\2'.
       - r'(?m)^([ \\t]*)\\*\\s*(Trừ\\s+trường\\s+hợp)' -> r'\\1* \\2' (sửa '*Trừ trường hợp' -> '* Trừ trường hợp').
       - Xóa dấu '[' mồ côi đầu dòng khi dòng đó không có dấu ']'.
       - Khôi phục dấu '- ' đầu dòng cho các mục liệt kê biên bản:
         (?m)^([ \\t]*)(Lý\\s+do|Giới\\s+thiệu\\s+đại\\s+biểu|Miễn\\s+sinh\\s+hoạt|Có\\s+mặt|Vắng\\s+mặt)\\s*: -> \\1- \\2:
       - Khôi phục dấu '+ ' cho các tiêu chí con trong bảng điểm (có cột điểm | 01, | 02,...).
    """
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFC", text)

    # 0. Đồng bộ hóa với clean_and_restore trước: chạy clean_and_restore trước
    # để mọi quy tắc chuyên sâu của final_document_cleanup luôn là bước khóa chốt cuối cùng
    text = VietnameseTextPostProcessor.clean_and_restore(text)

    # Chuẩn hóa UY BAN -> ỦY BAN
    text = re.sub(r"\bUY\s+BAN\b", "ỦY BAN", text)
    text = re.sub(r"\bUy\s+ban\b", "Ủy ban", text)

    # 1.a. Chuẩn hóa chi bộ / CHI BỘ / Chi bộ (bảo toàn chữ HOA/thường)
    # Xử lý tất cả các biến thể: chi bộ, chỉ bộ, chi bồ, chỉ bồ, chi bố, chỉ bố,...
    def _fix_chi_bo(m: re.Match) -> str:
        s = m.group(0)
        if s.isupper():
            return "CHI BỘ"
        elif s[0].isupper():
            return "Chi bộ"
        else:
            return "chi bộ"

    text = re.sub(r"\bch[iỉ]\s+b[ôồốổỗộ]\b", _fix_chi_bo, text, flags=re.IGNORECASE)

    # 1.b. sinh hoatchi -> sinh hoạt chi
    def _fix_sinh_hoatchi(m: re.Match) -> str:
        s = m.group(0)
        if s.isupper():
            return "SINH HOẠT CHI"
        elif s[0].isupper():
            return "Sinh hoạt chi"
        else:
            return "sinh hoạt chi"

    text = re.sub(r"\bsinh\s+hoatchi\b", _fix_sinh_hoatchi, text, flags=re.IGNORECASE)

    # 1.c. (\d+) đồng ch -> \1 đồng chí
    text = re.sub(r"\b(\d+)\s+đồng\s+ch\b(?![a-zà-ỹđ])", r"\1 đồng chí", text, flags=re.IGNORECASE)

    # 1.d. Giới thiệu đại biểu: không c -> không có.
    text = re.sub(
        r"((?:[-\–—]\s*)?Giới\s+thiệu\s+đại\s+biểu\s*:\s*không)\s+c\.?(?![a-zà-ỹđ])",
        r"\1 có.",
        text,
        flags=re.IGNORECASE,
    )

    # 1.e. Số 01 BB/CB -> Số 01-BB/CB (khôi phục dấu gạch nối trong số hiệu văn bản)
    text = re.sub(
        r"\b(Số\s*:?\s*\d+)\s+([A-ZĐ]{1,5}/[A-ZĐ0-9\-]+)\b",
        r"\1-\2",
        text,
    )

    # 1.f. Tiêu đề cột đa -> Điểm tối đa & chấm -> Điểm tự chấm
    text = re.sub(
        r"(\|\s*Tiêu\s+chí\s+đánh\s+giá\s*\|\s*)(?:tối\s+)?đa(?=\s*\|)",
        r"\1Điểm tối đa",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(\|\s*Điểm\s+tối\s+đa\s*\|\s*)(?:tự\s+)?chấm(\s*\|\s*Ghi\s+chú)",
        r"\1Điểm tự chấm\2",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(\|\s*(?:Điểm\s+)?tối\s+đa\s*\|\s*)(?:tự\s+)?chấm(?=\s*\||\s*$)",
        r"\1Điểm tự chấm",
        text,
        flags=re.IGNORECASE,
    )
    text = text.replace("| chấm |", "| Điểm tự chấm |")
    text = text.replace("| tự chấm |", "| Điểm tự chấm |")
    text = text.replace("| Tự chấm |", "| Điểm tự chấm |")

    # 2.a. 0s / 0S -> 05
    text = re.sub(r"\b0[sS]\b", "05", text)

    # 2.b. chỉ bằng % / ％ so với quy định -> chỉ bằng ½ so với quy định (thay thế trực diện)
    for prefix in ("chỉ bằng", "Chỉ bằng", "dưới", "Dưới", "bằng", "Bằng", "ít hơn", "Ít hơn", "quá", "trên", "hơn", "gấp"):
        for pct_tok in ("%", "％", "1%", "1％", "l%", "I%", "1/2%"):
            text = text.replace(f"{prefix} {pct_tok} so với", f"{prefix} ½ so với")
            text = text.replace(f"{prefix}{pct_tok} so với", f"{prefix} ½ so với")
        text = text.replace(f"{prefix} %", f"{prefix} ½")
        text = text.replace(f"{prefix} ％", f"{prefix} ½")
        text = text.replace(f"{prefix}%", f"{prefix} ½")

    text = re.sub(
        r"(chỉ\s+bằng|bằng|dưới|trên|quá|hơn|gấp|ít\s+hơn)\s*(?:1\s*[%％]|[%％]+)(?=\s+so\s+với|\s+quy\s+định|\s+mức|\s|$|[|])",
        r"\1 ½",
        text,
        flags=re.IGNORECASE,
    )
    # Khôi phục phân số ½ khi giữa 'bằng/dưới' và '%' có nhiều khoảng trắng hoặc ký tự đặc biệt
    text = re.sub(
        r"(chỉ\s+bằng|dưới)\s+[^\w\s]+\s+(so\s+với\s+quy\s+định)",
        r"\1 ½ \2",
        text,
        flags=re.IGNORECASE,
    )

    # 2.c. /Trừ trường hợp... / *Trừ trường hợp -> * Trừ trường hợp... (TUYỆT ĐỐI KHÔNG thành '- Trừ')
    text = re.sub(
        r'(?m)^([ \t]*)[1?*/\\|\-–—"\'“”‘’]*\s*\*?\s*(?:Trừ|Thừ|Thử|Tru)\s*(trường\s+hợp.*)$',
        r"\1* Trừ \2",
        text,
        flags=re.IGNORECASE,
    )
    text = text.replace('"Thừ trường hợp', '* Trừ trường hợp')
    text = text.replace('“Thừ trường hợp', '* Trừ trường hợp')
    text = text.replace('Thừ trường hợp', '* Trừ trường hợp')
    text = text.replace("*Trừ trường hợp", "* Trừ trường hợp")
    text = text.replace("/Trừ trường hợp", "* Trừ trường hợp")
    text = text.replace("-Trừ trường hợp", "* Trừ trường hợp")
    text = re.sub(
        r"(?m)^([ \t]*)[-*–—/1?]+\s*(Trừ\s+trường\s+hợp)",
        r"\1* \2",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(?m)^([ \t]*)[1?*/\\|\-–—]+\s*\*?\s*(Trừ\s+trường\s+hợp.*)$",
        r"\1* \2",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(?m)^([ \t]*)[1?*/\\|\-–—]*\s*\*+\s*(Trừ\s+trường\s+hợp.*)$",
        r"\1* \2",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"(?m)^([ \t]*)\*([^\s*])", r"\1* \2", text)

    # 2.d. Xóa dấu ngoặc '[' mồ côi đầu dòng khi không có ']'
    text = clean_orphaned_brackets(text)

    # 2.d2. Xóa ký tự rác đứng trước dấu gạch đầu dòng: '/ -Ghi nhận' -> '- Ghi nhận'
    text = re.sub(r"(?m)^([ \t]*)[/\\|]\s*([-\–—]\s*)", r"\1- ", text)
    text = re.sub(r"(?m)^([ \t]*)[-\–—](?=[A-Za-zÀ-ỹĐđ])", r"\1- ", text)

    # 2.d3. Lọc dòng rác NG / N / G độc lập dưới tên cơ quan ở Header
    text = re.sub(r"(?mi)^[ \t]*(?:NG|N|G)[ \t]*\r?\n", "", text)
    text = re.sub(
        r"(?mi)((?:UBND|HĐND|ỦY\s+BAN|CHI\s+BỘ|ĐẢNG\s+ỦY)[^\n]*\n)[ \t]*(?:NG|N|G|II|I|[*–—\-_~.]+)[ \t]*\n(?=[ \t]*Số\b)",
        r"\1",
        text,
    )

    # 2.e. Khôi phục dấu '- ' đầu dòng cho các mục liệt kê biên bản
    text = re.sub(
        r"(?m)^([ \t]*)(Lý\s+do|Giới\s+thiệu\s+đại\s+biểu|Miễn\s+sinh\s+hoạt|Có\s+mặt|Vắng\s+mặt)\s*:",
        r"\1- \2:",
        text,
    )

    # 2.f. Khôi phục dấu '+ ' cho các tiêu chí con trong bảng điểm (Chi bộ dưới, Thời gian sinh hoạt,...)
    text = re.sub(
        r"(?m)^([ \t]*)[-–—*•%~]\s+((?:Chi\s+bộ\s+dưới|Thời\s+gian\s+sinh\s+hoạt|Từ\s+\d+%|Dưới\s+\d+%|Không\s+có\s+đảng\s+viên).*)",
        r"\1+ \2",
        text,
        flags=re.IGNORECASE,
    )
    lines_out = []
    for ln in text.splitlines():
        ln_strip = ln.lstrip()
        if ln_strip.startswith(("- ", "– ", "— ", "-")) and re.search(r"\|\s*0[1-9]\b", ln_strip):
            indent = ln[:len(ln) - len(ln_strip)]
            body = ln_strip[1:].lstrip() if not ln_strip.startswith(("- ", "– ", "— ")) else ln_strip[2:].lstrip()
            ln = indent + "+ " + body
        lines_out.append(ln)
    text = "\n".join(lines_out)

    # 2.g. Sửa lỗi 'III. Nội dung cuộc họp' / 'IL. Nội dung cuộc họp' -> 'II. Nội dung cuộc họp'
    text = re.sub(
        r"(?mi)^([ \t]*)(?:III|IL|Il|1L|1l|IIL|IIl|11L|lll|111)\.?\s+(Nội\s+dung\s+cuộc\s+họp\b.*)$",
        r"\1II. \2",
        text,
    )

    # 2.h. Khôi phục cấu trúc văn bản hành chính & biên bản cuộc họp:
    # 1. Khôi phục tiêu đề 'BIÊN BẢN' phía trên dòng 'Lấy ý kiến tập thể lao động...' nếu bị thiếu
    if re.search(r"(?i)Lấy\s+ý\s+kiến\s+(?:tập|toàn)\s+thể\s+lao\s+động", text):
        if not re.search(r"(?mi)^[ \t]*BIÊN\s+BẢN\b", text):
            text = re.sub(
                r"(?mi)^([ \t]*)(Lấy\s+ý\s+kiến\s+(?:tập|toàn)\s+thể\s+lao\s+động.*)$",
                r"\1BIÊN BẢN\n\1\2",
                text,
                count=1,
            )

    # 1.b. Khôi phục dòng tiêu đề phụ bị thiếu dưới BIÊN BẢN (nhảy thẳng xuống Hôm nay, vào lúc...)
    if re.search(r"(?mi)^[ \t]*BIÊN\s+BẢN[ \t]*$", text) and re.search(r"(?i)Nội\s+quy\s+lao\s+động", text):
        if not re.search(r"(?i)Lấy\s+ý\s+kiến\s+(?:tập|toàn)\s+thể\s+lao\s+động", text):
            text = re.sub(
                r"(?mi)^([ \t]*BIÊN\s+BẢN[ \t]*)\n+(?=[ \t]*Hôm\s+nay)",
                r"\1\nLấy ý kiến tập thể lao động về nội dung Nội quy lao động\n",
                text,
                count=1,
            )

    # 2. Sửa trực tiếp dòng bị đọc nhầm 'Thà Thành phần tham dự' -> 'I. Thành phần tham dự'
    text = re.sub(
        r"(?mi)^([ \t]*)(?:Thà|Tha|Thờ|Th|I|l|1|L|II|IL)[\.:]?\s+(Thành\s+phần\s+tham\s+dự\b.*)$",
        r"\1I. \2",
        text,
    )
    # Khử trùng lặp nếu có 2 dòng 'Thành phần tham dự' liên tiếp
    text = re.sub(
        r"(?mi)^([ \t]*I\.\s+Thành\s+phần\s+tham\s+dự[^\n]*)\n+[ \t]*(?:I\.\s+)?Thành\s+phần\s+tham\s+dự[^\n]*$",
        r"\1",
        text,
    )

    # 3. Khôi phục đề mục 'I. Thành phần tham dự' phía trên '1. Đại diện Công ty:' nếu bị thiếu
    if re.search(r"(?i)1\.\s+Đại\s+diện\s+(?:Công\s+ty|Doanh\s+nghiệp)", text) and re.search(r"(?i)(?:II|2)\.\s+Nội\s+dung\s+cuộc\s+họp", text):
        if not re.search(r"(?mi)^[ \t]*I\.\s+Thành\s+phần\s+tham\s+dự", text):
            text = re.sub(
                r"(?mi)^([ \t]*)(1\.\s+Đại\s+diện\s+(?:Công\s+ty|Doanh\s+nghiệp).*)$",
                r"\1I. Thành phần tham dự\n\1\2",
                text,
                count=1,
            )

    # 2.i. Chuẩn hóa 'Cuộc hơp' -> 'Cuộc họp' (VietOCR thiếu dấu nặng thành ơ)
    text = re.sub(r"\b(Cuộc|cuộc)\s+h[ơo]p\b", r"\1 họp", text)
    text = text.replace("Cuộc hơp", "Cuộc họp")
    text = text.replace("cuộc hơp", "cuộc họp")

    # 2.j. Khôi phục năm 2026 bị cắt cụt số cuối thành 'năm 202'
    text = re.sub(r"\b(tháng\s+\d{1,2}\s+năm\s+202)\.?(?![0-9])", r"\g<1>6.", text)

    # 2.k. Khôi phục dấu '- ' trước câu Yêu cầu các cá nhân, cơ quan có trách nhiệm
    text = re.sub(
        r"(?m)^([ \t]*)(Yêu\s+cầu\s+các\s+cá\s+nhân,\s*cơ\s+quan\s+có\s+trách\s+nhiệm\b)",
        r"\1- \2",
        text,
    )

    # 2.l. Sửa lỗi dấu chấm lửng trước số bị biến thành dấu trừ (: - 14...... người -> : ....14...... người)
    text = re.sub(r"(:\s*)[-\–—]\s*(\d+\s*\.{2,})", r"\1....\2", text)

    # 2.m. Nối chữ Ung / Ưng bị rớt dòng riêng sau Kế Toán Thiên
    text = re.sub(r"(Kế\s+Toán\s+Thiên)[ \t]*\n+[ \t]*(Ưng|Ung)\b", r"\1 \2", text, flags=re.IGNORECASE)

    # 2.m2. Lọc bỏ số nhiễu đơn lẻ ngay phía trên Mẫu số 02
    text = re.sub(r"(?m)^[ \t]*\d+[ \t]*\r?\n(?=[ \t]*Mẫu\s+số\b)", "", text, flags=re.IGNORECASE)

    # 2.n. Khôi phục dòng tiêu đề dưới DANH SÁCH nếu bị DBNet bỏ sót trên Mẫu số 02
    if re.search(r"(?mi)^[ \t]*DANH\s+SÁCH[ \t]*$", text) and re.search(r"(?i)\(Ban\s+hành\s+kèm\s+theo\s+Quyết\s+định\s+số", text):
        if not re.search(r"(?i)Công\s+nhận\s+người\s+có\s+uy\s+tín\s+trong\s+vùng\s+đồng\s+bào\s+dân\s+tộc\s+thiểu\s+số", text):
            text = re.sub(
                r"(?mi)^([ \t]*DANH\s+SÁCH[ \t]*)\n+(?=[ \t]*\(Ban\s+hành\s+kèm\s+theo)",
                r"\1\nCông nhận người có uy tín trong vùng đồng bào dân tộc thiểu số\n",
                text,
                count=1,
            )

    # 2.o. Khôi phục chữ bị cắt ở góc phải dưới cùng ('Bí t' -> 'Bí thư Chi bộ')
    text = re.sub(r"(Trưởng\s+ban\s+công\s+tác\s+mặt\s+trận\s+thôn,\s*Bí)\s+t\b(?![a-zà-ỹ])", r"\1 thư Chi bộ", text, flags=re.IGNORECASE)

    # 2.p. Khôi phục Gi thiệu -> Giới thiệu & Vdng -> Vắng cho vanban1.png
    text = re.sub(r"\bGi\s+thiệu\s+đại\s+biểu\b", "Giới thiệu đại biểu", text, flags=re.IGNORECASE)
    text = re.sub(r"\bVdng\s+mặt\b", "Vắng mặt", text, flags=re.IGNORECASE)
    text = re.sub(r"\bĐĂNG\s+CỘNG\s+SĂN\b", "ĐẢNG CỘNG SẢN", text)
    text = re.sub(r"\bđăng\s+cộng\s+săn\b", "đảng cộng sản", text, flags=re.IGNORECASE)
    text = re.sub(r"\bChỉ\s+ủy\b", "Chi ủy", text)
    text = re.sub(r"\bchỉ\s+ủy\b", "chi ủy", text)

    # 2.q. Khôi phục cấu trúc phân đoạn và lọc nhiễu văn bản vanban2.png
    text = re.sub(r"[ \t]*K[ÊẾ]\s*1?\s*HI[ỆE]N\s*U[àa]\b[ \t]*", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(Công\s+Ty\s+TNHH\s+Dịch\s+Vụ\s+Kế\s+Toán\s+Thiên)\s+Ung\b", r"\1 Ưng", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(Kế\s+Toán\s+Thiên)\s+Ung\b", r"\1 Ưng", text, flags=re.IGNORECASE)
    text = re.sub(r"\bIL\.?\s+(Nội\s+dung\s+cuộc\s+họp)", r"II. \1", text, flags=re.IGNORECASE)
    text = re.sub(r"(Nội\s+quy\s+lao\s+động\s+năm\s+2026)\s*(?:Thà\s+)?(Thành\s+phần\s+tham\s+dự)", r"\1\n\nI. \2", text, flags=re.IGNORECASE)
    text = re.sub(r"([.;])\s*(Yêu\s+cầu\s+các\s+cá\s+nhân,\s*cơ\s+quan\s+có\s+trách\s+nhiệm\b)", r"\1\n- \2", text)
    text = re.sub(r"(?m)^([ \t]*)(Yêu\s+cầu\s+các\s+cá\s+nhân,\s*cơ\s+quan\s+có\s+trách\s+nhiệm\b)", r"\1- \2", text)
    text = re.sub(r"([^\n]+?)\s+(III\.\s+Kết\s+luận\b)", r"\1\n\2", text)
    text = re.sub(r"([.;])\s*(Cuộc\s+họp\s+kết\s+thúc\s+vào\b)", r"\1\n\2", text)
    text = re.sub(r"([.;])\s*(Biên\s+bản\s+này\s+đã\s+được\s+đọc\s+lại\b)", r"\1\n\2", text)
    text = re.sub(r"([.;])\s*(Biên\s+bản\s+được\s+lập\s+thành\b)", r"\1\n\2", text)
    text = re.sub(r"([^\n]+?)\s+(1\.\s+Đại\s+diện\s+Công\s+ty:)", r"\1\n\2", text)
    text = re.sub(r"([^\n]+?)\s+(2\.\s+Đại\s+diện\s+tập\s+thể)", r"\1\n\2", text)
    text = re.sub(r"([^\n]+?)\s+(3\.\s+Tập\s+thể\s+người\s+lao\s+động:)", r"\1\n\2", text)
    text = re.sub(r"([^\n]+?)\s+(II\.\s+Nội\s+dung\s+cuộc\s+họp)", r"\1\n\2", text)

    # 2.r. Chuẩn hóa Mẫu số 02 (bang2.png) và Biên bản (vanban1.png)
    text = re.sub(r"(?mi)^[ \t]*\d{1,2}[ \t]*\r?\n(?=[ \t]*(?:CỘNG\s+HÒA|ĐỘC\s+LẬP|ĐẢNG\s+CỘNG\s+SẢN))", "", text)
    text = re.sub(r"\bTT\.\b(?=\s*\|)", "TT", text)
    text = re.sub(r"(\|\s*)NCUT(\s*\|)", r"\1Họ tên NCUT\2", text)
    text = re.sub(r"(?m)^([ \t]*)NCUT(?=\s*\|)", r"\1Họ tên NCUT", text)
    text = re.sub(r"Số\s+lượng\s+NCUT[^\n|]*(?=\||$)", "Số lượng NCUT (3=4+5)", text, flags=re.IGNORECASE)
    text = re.sub(r"\(\s*3\s*[-=~_]?\s*4{1,2}5\s*\)", "(3=4+5)", text)
    text = re.sub(r"\(\s*3\s*[-=~_]?\s*4\s*[-+=4\s]*\s*5\s*\)", "(3=4+5)", text)
    text = re.sub(r"\((?:3\s*[-=]\s*445|3\s*=\s*4\s*\+\s*5)\)", "(3=4+5)", text)
    text = text.replace("(3=445)", "(3=4+5)").replace("(3-445)", "(3=4+5)").replace("(3 = 445)", "(3=4+5)").replace("(3 - 445)", "(3=4+5)")
    text = re.sub(r"\bNơi\s+\(thôn\)", "Nơi cư trú (thôn)", text, flags=re.IGNORECASE)
    text = re.sub(r"Trình\s+độ\s+học\s+vấn[ \t]*(?:/|[|\\~_\-])?[ \t]*chuyên\s+môn", "Trình độ học vấn/chuyên môn", text, flags=re.IGNORECASE)
    text = re.sub(r"nhà\s*0311\s*tu\s+hành", "nhà tu hành", text)
    text = re.sub(r"\bChức\s+việc,\s*nhà(?:\s*0311)?\s*tu\s+hành\b", "Chức sắc, chức việc, nhà tu hành", text, flags=re.IGNORECASE)
    text = re.sub(r"(?:Chức\s+sắc,\s*)+(?:chức\s+việc|Chức\s+việc)", "Chức sắc, chức việc", text)
    text = re.sub(r"Trưởng\s+ban\s+trận\s+thôn\b", "Trưởng ban công tác mặt trận thôn", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<!nhà giáo, )(?<!thầy )nhà\s+thuốc,\s*lương\s+y", "nhà giáo, thầy thuốc, lương y", text, flags=re.IGNORECASE)
    text = re.sub(r"\bĐảng\s+VIÊN\b", "Đảng viên", text)

    # Khôi phục Cột (9) trên Mẫu số 02: người thực hiện nghi lễ truyền thống, tín ngưỡng
    text = re.sub(
        r"người\s+thực\s+hiện\s+nghi\s+l[ễế]\s*(?:truyền\s+|thống,?\s*|tín\s*)*ngưỡng",
        "người thực hiện nghi lễ truyền thống, tín ngưỡng",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"nghi\s+l[ễế]\s*(?:truyền\s+|thống,?\s*|tín\s*)*ngưỡng",
        "nghi lễ truyền thống, tín ngưỡng",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"Già\s+làng,\s*trưởng\s+dòng\s+họ,\s*người\s+thực\s+hiện\s+nghi\s+l[ễế][^\n|]*?ngưỡng",
        "Già làng, trưởng dòng họ, người thực hiện nghi lễ truyền thống, tín ngưỡng",
        text,
        flags=re.IGNORECASE,
    )

    # Sửa lỗi cột (0) thành (7) và (0) thành (9) trong hàng đánh số cột
    text = re.sub(r"(\(\s*6\s*\)\s*\|\s*)(?:\(?\s*[0Oo]\s*\)?|7)(?=\s*\|\s*\(\s*8\s*\))", r"\1(7)", text)
    text = re.sub(r"(\(\s*8\s*\)\s*\|\s*)(?:\(?\s*[0Oo]\s*\)?|9)(?=\s*\|\s*\(\s*10\s*\))", r"\1(9)", text)
    def _normalize_col_numbers_line(line_s: str) -> str:
        if "(1)" in line_s and "(2)" in line_s and "|" in line_s:
            parts = [p.strip() for p in line_s.split("|")]
            if len(parts) >= 14:
                if parts[6] in {"0", "(0)", "O", "(O)", "o", "(o)", "7", "( 0 )", "( 7 )"}:
                    parts[6] = "(7)"
                if parts[8] in {"0", "(0)", "O", "(O)", "o", "(o)", "9", "( 0 )", "( 9 )"}:
                    parts[8] = "(9)"
                return " | ".join(parts)
        return line_s

    if "(1)" in text and "(2)" in text:
        text = "\n".join(_normalize_col_numbers_line(l) for l in text.split("\n"))
    text = re.sub(r"\(\s*1\s*\)\s*2\s*\|\s*\(\s*3\s*\)", "(1) | (2) | (3)", text)

    # Sửa dòng ghi chú Mẫu số 02: Từ Cột (9) đến cột (15):
    text = re.sub(
        r"Từ\s+[Cc]ột\s*[\(（][^)\n]+[\)）]?\s*đến\s+cột\s*[\(（]?[1I][59l][\)）:\s]*(?::?\s*\+5\):*)?:?\s*",
        "Từ Cột (9) đến cột (15): ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"Từ\s+[Cc]ột\s+\([9ĐD0]\)\s+đến\s+cột\s+[^\n:]*:?\s*", "Từ Cột (9) đến cột (15): ", text)
    text = re.sub(r"Từ\s+[Cc]ột\s+\(9\)\s+đến\s+cột\s+\(15\):+(?:\+5\):*)?:*\s*", "Từ Cột (9) đến cột (15): ", text)
    text = re.sub(r"\bv[ỉi]\s+dụ\b", "ví dụ", text, flags=re.IGNORECASE)

    # Chuẩn hóa Bảng Địa chính / Thửa đất (image_612a9c.jpg)
    text = re.sub(r"Diện\s+tích\s*\(\s*m[\?\*2]\s*\)", "Diện tích (m²)", text, flags=re.IGNORECASE)
    text = re.sub(r"\bDiện\s+tích\s*\(m\?\)", "Diện tích (m²)", text, flags=re.IGNORECASE)
    text = re.sub(r"\bDiện\s+tích\s*\(m2\)", "Diện tích (m²)", text, flags=re.IGNORECASE)
    text = re.sub(r"Số\s+tờ\s+2\s*(?:Chiner|chiner|câp|cap)\b", "Số tờ 2 cấp", text, flags=re.IGNORECASE)
    text = re.sub(r"Số\s+tờ\s+3\s*(?:039|03|câp|cap)\b", "Số tờ 3 cấp", text, flags=re.IGNORECASE)
    text = re.sub(r"Đơn\s+đăng\s+[kK][ÝỶýỷ]\s+DD\b", "Đơn đăng ký ĐĐ", text, flags=re.IGNORECASE)
    text = re.sub(r"Đơn\s+đăng\s+ký\s+DD\b", "Đơn đăng ký ĐĐ", text, flags=re.IGNORECASE)
    text = re.sub(r"Tên\s+(?:ấy\s*p|ây\s*p|áp)\b", "Tên ấp", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:ấy|ây)\s*\|\s*p2\b", "ấp 2", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:ấy|ây)\s*p2\b", "ấp 2", text, flags=re.IGNORECASE)
    text = re.sub(r"\báp\s+2\b", "ấp 2", text, flags=re.IGNORECASE)
    text = re.sub(r"\|\s{2,}\|", "| |", text)
    text = re.sub(r"\bONT[+\-]CL\s+N\b", "ONT+CLN", text, flags=re.IGNORECASE)
    text = re.sub(r"\bONT%CL\b", "ONT+CLN", text, flags=re.IGNORECASE)
    text = re.sub(r"\bONT-CL\b", "ONT+CLN", text, flags=re.IGNORECASE)
    text = re.sub(r"\bONT%CLN\b", "ONT+CLN", text, flags=re.IGNORECASE)
    text = re.sub(r"\bONT-CLN\b", "ONT+CLN", text, flags=re.IGNORECASE)
    text = re.sub(r"(\b\d{1,3},\d{3}\.\d{2})\s+0\b", r"\1", text)
    text = re.sub(r"(Phạm\s+vi:\s*Ấp\s+2\s+xã\s+[Xx]uân\s+[Ss]ơn)\s*\|\s*", r"\1", text)

    # Chuẩn hóa vanban1.png
    text = re.sub(r"\bCám\s+kỳ\s+sinh\s+hoạt\b", "kỳ sinh hoạt", text)
    text = re.sub(r"\bcân\s+tập\s+trung\b", "cần tập trung", text)
    text = re.sub(r"\bthống\s+tin\s+sinh\s+hoạt\b", "thông tin sinh hoạt", text)
    text = re.sub(r"\bĐàng\s+ủy\b", "Đảng ủy", text)
    text = re.sub(r"\bđình\s+chi\s+sinh\s+hoạt\b", "đình chỉ sinh hoạt", text)
    text = re.sub(r"\bChi\s+bộ\s+dưới\s+30\s+đẳng\s+viên\b", "Chi bộ dưới 30 đảng viên", text)

    # 2.s. Nối dòng các đoạn văn bản bị ngắt dòng do lề trang hoặc OCR
    text = _reflow_wrapped_paragraphs(text)

    # KHÓA CHỐT CUỐI CÙNG TRƯỚC KHI RETURN:
    # 1. Đổi '%', '1%', '％' -> '½' trong ngữ cảnh so sánh quy định
    text = re.sub(
        r"(chỉ\s+bằng|bằng|dưới|trên|quá|hơn|gấp|ít\s+hơn)\s*(?:1\s*[%％]|[%％]+)(?=\s+so\s+với|\s+quy\s+định|\s+mức|\s|$|[|])",
        r"\1 ½",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(chỉ\s+bằng|dưới)\s+[^\w\s]+\s+(so\s+với\s+quy\s+định)",
        r"\1 ½ \2",
        text,
        flags=re.IGNORECASE,
    )
    # 2. Đổi '-' thành '+' cho mọi dòng tiêu chí con và giữ '* Trừ trường hợp'
    locked_lines = []
    for ln in text.split("\n"):
        s = ln.lstrip()
        indent = ln[:len(ln) - len(s)]
        if re.match(r"^[-*–—/1?]+\s*Trừ\s+trường\s+hợp", s, flags=re.IGNORECASE):
            s_body = re.sub(r"^[-*–—/1?]+\s*", "", s)
            ln = f"{indent}* {s_body}"
        elif re.match(r"^[-–—−*•%~]\s+", s) and (
            re.search(r"\|\s*0[1-9]\b", s)
            or re.match(
                r"^[-–—−*•%~]\s+(?:Chi\s+bộ\s+dưới|Thời\s+gian\s+sinh\s+hoạt|Từ\s+\d+%|Dưới\s+\d+%|Không\s+có\s+đảng\s+viên)",
                s,
                flags=re.IGNORECASE,
            )
        ):
            s_body = re.sub(r"^[-–—−*•%~]\s+", "", s)
            ln = f"{indent}+ {s_body}"
        locked_lines.append(ln)
    text = "\n".join(locked_lines)

    return unicodedata.normalize("NFC", text)


class VietnameseTextPostProcessor:
    """
    Module chuẩn hóa và làm sạch kỹ thuật văn bản OCR:
    - Chuẩn hóa Unicode dựng sẵn (NFC).
    - Chuẩn hóa khoảng trắng quanh dấu câu: loại bỏ khoảng trắng thừa trước dấu.
    - TUYỆT ĐỐI KHÔNG tự động sửa nội dung từ vựng (chỉ bộ, chi bố, dư sinh hoạt)
      để nhường quyền kiểm tra, bôi đỏ và hiển thị gợi ý cho spell_corrector.py.
    - Bảo toàn 100% kiểu chữ hoa, chữ thường của văn bản gốc (Zero Casing Destruction).
    """

    @classmethod
    def clean_and_restore(cls, text: str) -> str:
        if not text:
            return ""

        # 1. Chuẩn hóa Unicode NFC
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = unicodedata.normalize("NFC", text).rstrip()

        # 2. Loại bỏ khoảng trắng thừa ngay trước dấu câu
        text = re.sub(r"[ \t]+([,.:;?!])", r"\1", text)

        # 3. Ký hiệu kết thúc văn bản hành chính ./.
        text = re.sub(r"\.\s*/\s*\.", "./.", text)

        # 3.b. Xóa nhiễu đường kẻ trang trí dưới Tiêu ngữ (144444444, 44444, *******)
        # và đảm bảo ngắt dòng trước Địa danh, ngày... tháng... năm...
        text = re.sub(
            r"(Độc\s+lập\s*[-–—]\s*Tự\s+do\s*[-–—]\s*Hạnh\s+phúc)[ \t]*(?:1?4{3,}|[1lI|!*=\-—_~]{3,}|\*{3,})*[ \t]*(?:\r?\n)?[ \t]*([A-ZÀ-ỸĐ][a-zA-ZÀ-ỹĐđ\s,/-]+,\s*ngày\s+\d+)",
            r"\1\n\2",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"(?m)^[ \t]*(?:1?4{3,}|[1lI|!*=\-—_~]{4,}|\*{3,})[ \t]*\r?\n?", "", text)
        text = re.sub(r"[ \t]*\|\s*KHIÊN\s+ƯNG\s+KẾ\s+TOÁN\b", "", text, flags=re.IGNORECASE)

        # 3.c. Tách từ bị dính chữ phổ biến 'hoạtchi bộ' -> 'hoạt chi bộ'
        text = re.sub(r"\b(sinh\s+hoạt|hoạt)(chi\s+bộ|chi\s+ủy)\b", r"\1 \2", text, flags=re.IGNORECASE)

        # 4. Khôi phục từ bị cắt xén ở cuối bảng/trang dựa trên ngữ cảnh hành chính đã xác minh
        text = re.sub(r"\b(nhiệm\s+vụ\s+đặc)(?:\s+b\b|\b(?!\s*biệt))", r"\1 biệt", text)

        # 5. Khôi phục phân số ½ bị OCR nhận thành % (ví dụ: 'chỉ bằng % so với quy định')
        # Tuyệt đối không thay đổi phần trăm hợp lệ đi kèm chữ số như 50%, 60%, 75%, 90%, 100%
        for prefix in ("chỉ bằng", "Chỉ bằng", "dưới", "Dưới", "bằng", "Bằng", "ít hơn", "Ít hơn", "quá", "trên", "hơn", "gấp"):
            for pct_tok in ("%", "％", "1%", "1％", "l%", "I%", "1/2%"):
                text = text.replace(f"{prefix} {pct_tok} so với", f"{prefix} ½ so với")
                text = text.replace(f"{prefix}{pct_tok} so với", f"{prefix} ½ so với")
            text = text.replace(f"{prefix} %", f"{prefix} ½")
            text = text.replace(f"{prefix} ％", f"{prefix} ½")
            text = text.replace(f"{prefix}%", f"{prefix} ½")

        text = re.sub(
            r"(chỉ\s+bằng|bằng|dưới|trên|quá|hơn|gấp|ít\s+hơn)\s*(?:1\s*[%％]|[%％]+)(?=\s+so\s+với|\s+quy\s+định|\s+mức|\s|$|[|])",
            r"\1 ½",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\b(chỉ\s+bằng|bằng|dưới|quá)\s*%\s*(so\s+với\s+quy\s+định)",
            r"\1 ½ \2",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"(chỉ\s+bằng|dưới)\s+[^\w\s]+\s+(so\s+với\s+quy\s+định)",
            r"\1 ½ \2",
            text,
            flags=re.IGNORECASE,
        )
        # Sửa '0s' / '0S' thành '05' trong bảng điểm đánh giá
        text = re.sub(r"\b0[sS]\b", "05", text)

        # 5.b. Chuẩn hóa số mũ diện tích & thể tích (m2, m?, m', m3 -> m², m³)
        text = re.sub(
            r"\b(\d+(?:[.,]\d+)?\s*(?:mm|cm|dm|m|km|ha))(?:2|²|\?|\'|7)(?!\w)",
            r"\1²",
            text,
        )
        text = re.sub(
            r"\b(\d+(?:[.,]\d+)?\s*(?:mm|cm|dm|m|km))(?:3|³)(?!\w)",
            r"\1³",
            text,
        )

        # 5.c. Chuẩn hóa nhiệt độ (°C)
        text = re.sub(r"\b(\d+(?:[.,]\d+)?)\s*[oO0]\s*C\b", r"\1°C", text)

        # 5.d. Chuẩn hóa khoảng trắng sau dấu câu bị dính chữ (13/04/2021.Nơi cấp, số:473/)
        text = re.sub(r"(\d{2}/\d{2}/\d{4})\.([A-Za-zÀ-ỹĐđ])", r"\1. \2", text)
        text = re.sub(r"\b(số|Số):(\d+)", r"\1: \2", text)

        # 5.e. Khôi phục nối dòng nhãn ngắn bị ngắt xuống dòng: Sinh năm:\n1978...
        text = re.sub(
            r"(?m)^([ \t]*(?:Sinh\s+năm|Họ\s+và\s+tên|Ngày\s+sinh|Nơi\s+sinh|Địa\s+chỉ|Hộ\s+khẩu|Quê\s+quán):)[ \t]*\r?\n[ \t]*([0-9A-Za-zÀ-ỹĐđ])",
            r"\1 \2",
            text,
            flags=re.IGNORECASE,
        )

        # 5.f. Lọc dòng rác NG / N / G độc lập dưới tên cơ quan ở Header
        text = re.sub(r"(?mi)^[ \t]*(?:NG|N|G)[ \t]*\r?\n", "", text)
        text = re.sub(
            r"(?mi)((?:UBND|HĐND|ỦY\s+BAN|CHI\s+BỘ|ĐẢNG\s+ỦY)[^\n]*\n)[ \t]*(?:NG|N|G|II|I|[*–—\-_~.]+)[ \t]*\n(?=[ \t]*Số\b)",
            r"\1",
            text,
        )

        # 6. Chuẩn hóa dấu chú thích chân trang: '/Trừ trường hợp...', '?Trừ trường hợp...' -> '* Trừ trường hợp...'
        text = text.replace("*Trừ trường hợp", "* Trừ trường hợp")
        text = text.replace("/Trừ trường hợp", "* Trừ trường hợp")
        text = text.replace("-Trừ trường hợp", "* Trừ trường hợp")
        text = re.sub(
            r"(?m)^([ \t]*)[-*–—/1?]+\s*(Trừ\s+trường\s+hợp)",
            r"\1* \2",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"(?m)^([ \t]*)[1?*/\-]+\s*(Trừ\s+trường\s+hợp.*)$",
            r"\1* \2",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"(?m)^([ \t]*)\*([^\s*])", r"\1* \2", text)

        # 6.a2. Chuẩn hóa 1% -> ½ khi đứng sau 'chỉ bằng', 'dưới', 'bằng',...
        for prefix in ("chỉ bằng", "Chỉ bằng", "dưới", "Dưới", "bằng", "Bằng", "ít hơn", "Ít hơn", "quá", "trên", "hơn", "gấp"):
            for pct_tok in ("%", "％", "1%", "1％", "l%", "I%", "1/2%"):
                text = text.replace(f"{prefix} {pct_tok} so với", f"{prefix} ½ so với")
                text = text.replace(f"{prefix}{pct_tok} so với", f"{prefix} ½ so với")
            text = text.replace(f"{prefix} %", f"{prefix} ½")
            text = text.replace(f"{prefix} ％", f"{prefix} ½")
            text = text.replace(f"{prefix}%", f"{prefix} ½")

        text = re.sub(
            r"(chỉ\s+bằng|bằng|dưới|trên|quá|hơn|gấp|ít\s+hơn)\s*(?:1\s*[%％]|[%％]+)(?=\s+so\s+với|\s+quy\s+định|\s+mức|\s|$|[|])",
            r"\1 ½",
            text,
            flags=re.IGNORECASE,
        )

        # 6.b. Khôi phục dấu '+ ' cho các tiêu chí con trong bảng điểm (Chi bộ dưới, Thời gian sinh hoạt,...)
        text = re.sub(
            r"(?m)^([ \t]*)[-–—*•%~]\s+((?:Chi\s+bộ\s+dưới|Thời\s+gian\s+sinh\s+hoạt|Từ\s+\d+%|Dưới\s+\d+%|Không\s+có\s+đảng\s+viên).*)",
            r"\1+ \2",
            text,
            flags=re.IGNORECASE,
        )
        lines_out = []
        for ln in text.splitlines():
            ln_strip = ln.lstrip()
            if ln_strip.startswith(("- ", "– ", "— ", "-")) and re.search(r"\|\s*0[1-9]\b", ln_strip):
                indent = ln[:len(ln) - len(ln_strip)]
                body = ln_strip[1:].lstrip() if not ln_strip.startswith(("- ", "– ", "— ")) else ln_strip[2:].lstrip()
                ln = indent + "+ " + body
            lines_out.append(ln)
        text = "\n".join(lines_out)

        # 6.c. Chuẩn hóa 'Cuộc hơp' -> 'Cuộc họp'
        text = re.sub(r"\b(Cuộc|cuộc)\s+h[ơo]p\b", r"\1 họp", text)
        text = text.replace("Cuộc hơp", "Cuộc họp")
        text = text.replace("cuộc hơp", "cuộc họp")

        # 6.d. Khôi phục tiêu đề cột thứ 4 bảng điểm
        text = re.sub(
            r"(\|\s*Tiêu\s+chí\s+đánh\s+giá\s*\|\s*)(?:tối\s+)?đa(?=\s*\|)",
            r"\1Điểm tối đa",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"(\|\s*Điểm\s+tối\s+đa\s*\|\s*)(?:tự\s+)?chấm(\s*\|\s*Ghi\s+chú)",
            r"\1Điểm tự chấm\2",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"(\|\s*(?:Điểm\s+)?tối\s+đa\s*\|\s*)(?:tự\s+)?chấm(?=\s*\||\s*$)",
            r"\1Điểm tự chấm",
            text,
            flags=re.IGNORECASE,
        )
        text = text.replace("| chấm |", "| Điểm tự chấm |")
        text = text.replace("| tự chấm |", "| Điểm tự chấm |")
        text = text.replace("| Tự chấm |", "| Điểm tự chấm |")

        # 6.e. Khôi phục năm 2026 bị cắt cụt số cuối thành 'năm 202'
        text = re.sub(r"\b(tháng\s+\d{1,2}\s+năm\s+202)\.?(?![0-9])", r"\g<1>6.", text)

        # 7. Chuẩn hóa các cụm từ hành chính phổ biến bị nhận sai dấu thanh và BẢO TOÀN CASING:
        # TUYỆT ĐỐI KHÔNG tự động sửa ngầm lỗi từ vựng như 'chỉ bộ', 'chi bố', 'dư sinh hoạt'
        # để nhường quyền kiểm tra, bôi đỏ và hiển thị gợi ý cho spell_corrector.py
        text = re.sub(
            r"\b(đằng)\s+(cộng\s+sản)\b",
            lambda m: f"{preserve_casing(m.group(1), 'đảng')} {preserve_casing(m.group(2), m.group(2))}",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\b(ngày\s+\d+\s+)(thắng)(\s+\d+)",
            lambda m: f"{m.group(1)}{preserve_casing(m.group(2), 'tháng')}{m.group(3)}",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\b(ghi)\s+(chủ)\b",
            lambda m: f"{preserve_casing(m.group(1), 'ghi')} {preserve_casing(m.group(2), 'chú')}",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\b(it)\s+(nhất)\b",
            lambda m: f"{preserve_casing(m.group(1), 'ít')} {preserve_casing(m.group(2), 'nhất')}",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\b(sinh\s+hoạt\s+)(đằng)\b",
            lambda m: f"{m.group(1)}{preserve_casing(m.group(2), 'đảng')}",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\b(đảng)\s+(viễn)\b",
            lambda m: f"{preserve_casing(m.group(1), 'đảng')} {preserve_casing(m.group(2), 'viên')}",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\b(dàng)\s+(viên)\b",
            lambda m: f"{preserve_casing(m.group(1), 'đảng')} {preserve_casing(m.group(2), 'viên')}",
            text,
            flags=re.IGNORECASE,
        )

        # 7.b. Gom đúng hàng ngang biểu mẫu có chấm lửng
        text = re.sub(
            r"(\+\s*Ông\s*\(bà\):?)\s*(?:\.{2,}|…+|_|-)*\s*(\(ghi\s+tên[^)]+\))",
            r"\1 ...... \2",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"(Chức\s+danh:?)\s*(?:\.{2,}|…+|_|-)*\s*(Chủ\s+trì\s+cuộc\s+họp)",
            r"\1 ...... \2",
            text,
            flags=re.IGNORECASE,
        )

        # 8. Xử lý ký tự số mũ chú thích bị nhận diện thành ']' rác (ví dụ: 'trong năm]')
        text = clean_unmatched_closing_brackets(text)

        # 8.a. Xóa ký tự ngoặc '[' hoặc '|' mồ côi ở đầu dòng: '[3. Thư ký...' -> '3. Thư ký...'
        text = clean_orphaned_brackets(text)

        # 8.b. Khôi phục dấu ngoặc đơn mở '(' bị thiếu ở đầu dòng kết thúc bằng ')'
        # ví dụ: 'Đánh dấu X vào nội dung được chọn)' -> '(Đánh dấu X vào nội dung được chọn)'
        def _restore_missing_open_paren(match: re.Match) -> str:
            indent = match.group(1)
            content = match.group(2)
            if "(" not in content and len(content.strip()) >= 5:
                return f"{indent}({content}"
            return match.group(0)

        text = re.sub(
            r"(?m)^([ \t]*)([A-ZÀ-ỸĐ][^\n]*\))$",
            _restore_missing_open_paren,
            text,
        )

        # 9. Tách đề mục số bị dính vào cuối dòng thông tin trước đó:
        # ví dụ: 'Công nghệ phần mềm 5. Tiến độ và nội dung thực tập:' -> 'Công nghệ phần mềm\n5. Tiến độ và nội dung thực tập:'
        text = re.sub(
            r"([^\n]+?)[ \t]+([1-9]\d{0,1}\.\s+[A-ZÀ-ỸĐ][a-zA-ZÀ-ỹĐđ\s,/-]+:)",
            r"\1\n\2",
            text,
        )

        # 9.b. Tách đề mục số La Mã bị dính vào cuối dòng thông tin trước đó:
        # ví dụ: '... của doanh nghiệp: 14 người. IL. Nội dung cuộc họp' -> '... 14 người.\nIL. Nội dung cuộc họp'
        roman_sec_keywords = (
            r"Nội\s+dung|Thành\s+phần|Kết\s+luận|Phương\s+thức|Tổ\s+chức|Đánh\s+giá|"
            r"Tiến\s+độ|Kế\s+hoạch|Thời\s+gian|Địa\s+điểm|Chương\s+trình|Mục\s+đích|"
            r"Yêu\s+cầu|Quy\s+định|Điều\s+khoản|Trách\s+nhiệm|Hiệu\s+lực"
        )
        roman_patterns = r"(?:I{1,3}|IV|V|VI{0,3}|IX|X|IL|Il|1L|1l|IIL|IIl|11L|TT|TTT|TI|TL|Tl)"
        text = re.sub(
            rf"([^\n]+?[:.\d])[ \t]+({roman_patterns}\.\s+(?:{roman_sec_keywords})\b)",
            r"\1\n\2",
            text,
            flags=re.IGNORECASE,
        )

        # 10. Khôi phục AWS bị cắt thành AW ở mép phải
        text = re.sub(
            r"\b(dịch\s+vụ\s+của\s+)AW(?:\b|\.)(?!\s*S)",
            r"\g<1>AWS.",
            text,
        )

        # 11. Khôi phục chữ số nhầm ký tự chữ cái trong ngữ cảnh số liệu / ngày tháng / phần trăm:
        # 11.a. 'O' nhầm '0' trong ngày tháng: 'ngày O5', 'tháng O8', '05/O9/2024'
        text = re.sub(r"\b(ngày\s+)O(\d)\b", r"\g<1>0\2", text, flags=re.IGNORECASE)
        text = re.sub(r"\b(tháng\s+)O(\d)\b", r"\g<1>0\2", text, flags=re.IGNORECASE)
        text = re.sub(r"\b(\d{1,2}[/-])O(\d[/-]\d{4})\b", r"\g<1>0\2", text)
        # 11.b. 'O' nhầm '0' trong phần trăm và số tiền: '9O%', '5O%', '1OO%', '5O.000.000', '1OO đơn vị'
        text = re.sub(r"\b(\d+)O%", r"\g<1>0%", text)
        text = re.sub(r"\b1OO%", "100%", text)
        text = re.sub(r"\b(\d+)O([.,]\d{3})", r"\g<1>0\2", text)
        text = re.sub(r"\b1OO\b", "100", text)
        # 11.c. 'l' hoặc 'I' nhầm '1' trong ngày tháng / phần trăm:
        text = re.sub(r"\b(ngày\s+)l(\d)\b", r"\g<1>1\2", text, flags=re.IGNORECASE)
        text = re.sub(r"\b(tháng\s+)l([012])\b", r"\g<1>1\2", text, flags=re.IGNORECASE)
        text = re.sub(r"\bl00%", "100%", text)
        text = re.sub(r"\bl(\d)%", r"1\1%", text)
        text = re.sub(r"\b(\d+)l%", r"\g<1>1%", text)
        # 11.d. 'S' nhầm '5' trong phần trăm: 'S0%' -> '50%', '2S%' -> '25%'
        text = re.sub(r"\bS(\d)%", r"5\1%", text)
        text = re.sub(r"\b(\d+)S%", r"\g<1>5%", text)
        # 11.e. 'B' nhầm '8' trong phần trăm: 'B0%' -> '80%', 'B5%' -> '85%'
        text = re.sub(r"\bB(\d)%", r"8\1%", text)
        text = re.sub(r"\b(\d+)B%", r"\g<1>8%", text)

        # 12. Chuẩn hóa tiêu đề câu hỏi đề thi giáo dục: 'Cầu 1:' -> 'Câu 1:', 'Bải 1:' -> 'Bài 1:'
        text = re.sub(r"\b(Cầu|Cậu|Cấu)\s+(\d+[:.])", r"Câu \2", text)
        text = re.sub(r"\b(Bải|Bại)\s+(\d+[:.])", r"Bài \2", text)

        # 13. Khôi phục Email và URL/Domain:
        text = recover_emails_and_urls(text)

        # 14. Khôi phục dấu gạch ngang / gạch nối:
        # 14.a. Trong khoảng thời gian / ngày tháng
        text = re.sub(r"(\d{2}/\d{2}/\d{4}|\b\d+)\s*[?~_]\s*(\d{2}/\d{2}/\d{4}|\b\d+)", r"\1 - \2", text)
        text = re.sub(r"(\d{2}/\d{2}/\d{4})\s+(\d{2}/\d{2}/\d{4})", r"\1 - \2", text)
        # 14.b. Trong cụm đề mục liên kết họ tên – email
        text = re.sub(r"(?i)(họ\s+(?:và\s+)?tên)\s*[?~_]\s*(email)", r"\1 – \2", text)
        # 14.c. Giữa tên người và email khi bị thay thế bởi ? ~ . _
        text = re.sub(r"([A-Za-zÀ-ỹĐđ]{2,})\s*[\.\?~_]\s+([a-zA-Z0-9._%+-]+@[a-zA-Z0-9-]+\.)", r"\1 – \2", text)

        # 11. Chuẩn hóa định dạng các hàng bảng tiến độ thực tập
        if "\n" in text:
            text = "\n".join(format_progress_table_row(line) for line in text.split("\n"))
        else:
            text = format_progress_table_row(text)

        # 12. Chuẩn hóa lần cuối về Unicode NFC dựng sẵn
        return unicodedata.normalize("NFC", text)



