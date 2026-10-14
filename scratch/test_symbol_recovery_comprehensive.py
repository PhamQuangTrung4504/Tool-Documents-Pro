import re
from typing import List, Tuple, Dict, Any
from enum import Enum

class SymbolDecision(str, Enum):
    AUTO_CORRECT = "AUTO_CORRECT"
    FLAG_UNCERTAIN = "FLAG_UNCERTAIN"
    PRESERVE = "PRESERVE"

ROMAN_HEADING_KEYWORDS = {
    "NỘI DUNG", "TIẾN HÀNH", "KẾT LUẬN", "TỔ CHỨC", "MỤC ĐÍCH", "YÊU CẦU",
    "KẾ HOẠCH", "PHƯƠNG HƯỚNG", "NHIỆM VỤ", "ĐÁNH GIÁ", "CHƯƠNG TRÌNH",
    "THỰC HIỆN", "KẾT QUẢ", "THỜI GIAN", "ĐỊA ĐIỂM", "THÀNH PHẦN",
    "KIỂM ĐIỂM", "THẢO LUẬN", "NGHỊ QUYẾT", "BÁO CÁO",
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
        """Kiểm tra văn bản có chứa từ khóa đề mục văn bản hành chính hay không."""
        text_upper = text.upper()
        return any(kw in text_upper for kw in ROMAN_HEADING_KEYWORDS)

    @classmethod
    def recover_roman_numeral_in_line(
        cls, line: str, prev_roman: str = ""
    ) -> Tuple[str, SymbolDecision, str, str]:
        """
        Khôi phục số La Mã đầu mục (I., II., III., IV., V.) dựa trên ngữ cảnh.
        Trả về: (new_line, decision, current_roman, rule_name)
        """
        stripped = line.strip()
        if not stripped:
            return line, SymbolDecision.PRESERVE, prev_roman, ""

        # 1. Mẫu số II: TI., TI, ll., 11., Tl., lI., Il.
        m_ii = re.match(r"^([ \t]*)(TI|ll|11|Tl|lI|Il)\.?\s+([A-ZÀ-ỸĐ\s,;:/-]+)$", line)
        if m_ii:
            indent = m_ii.group(1)
            prefix = m_ii.group(2)
            rest = m_ii.group(3).strip()
            if cls.is_all_uppercase_heading(rest) or cls.has_heading_keyword(rest):
                return f"{indent}II. {rest}", SymbolDecision.AUTO_CORRECT, "II", f"ROMAN_II_FROM_{prefix}"
            else:
                return line, SymbolDecision.FLAG_UNCERTAIN, prev_roman, f"UNCERTAIN_ROMAN_II_{prefix}"

        # 2. Mẫu số III: lll., 111., TII., TIl., llI., lll, 111
        m_iii = re.match(r"^([ \t]*)(lll|111|TII|TIl|llI)\.?\s+([A-ZÀ-ỸĐ\s,;:/-]+)$", line)
        if m_iii:
            indent = m_iii.group(1)
            prefix = m_iii.group(2)
            rest = m_iii.group(3).strip()
            if cls.is_all_uppercase_heading(rest) or cls.has_heading_keyword(rest):
                return f"{indent}III. {rest}", SymbolDecision.AUTO_CORRECT, "III", f"ROMAN_III_FROM_{prefix}"
            else:
                return line, SymbolDecision.FLAG_UNCERTAIN, prev_roman, f"UNCERTAIN_ROMAN_III_{prefix}"

        # 3. Mẫu số IV: lV., 1V., TV., lV, 1V
        m_iv = re.match(r"^([ \t]*)(lV|1V|TV)\.?\s+([A-ZÀ-ỸĐ\s,;:/-]+)$", line)
        if m_iv:
            indent = m_iv.group(1)
            prefix = m_iv.group(2)
            rest = m_iv.group(3).strip()
            if cls.is_all_uppercase_heading(rest) or cls.has_heading_keyword(rest):
                return f"{indent}IV. {rest}", SymbolDecision.AUTO_CORRECT, "IV", f"ROMAN_IV_FROM_{prefix}"
            else:
                return line, SymbolDecision.FLAG_UNCERTAIN, prev_roman, f"UNCERTAIN_ROMAN_IV_{prefix}"

        # 4. Mẫu số I: |, |., l., 1.
        # ĐIỀU KIỆN ÂM TÍNH NGHIÊM NGẶT:
        # Không đổi 1. thành I. nếu theo sau là chữ thường (ví dụ: 1. Tuyên bố lý do...)
        m_i = re.match(r"^([ \t]*)(\|\.?|l\.|1\.)\s+([A-ZÀ-ỸĐ\s,;:/-]+)$", line)
        if m_i:
            indent = m_i.group(1)
            prefix = m_i.group(2).strip()
            rest = m_i.group(3).strip()
            is_strong_symbol = prefix in ["|", "|.", "l."]
            is_strong_heading = cls.has_heading_keyword(rest) and cls.is_all_uppercase_heading(rest)

            if is_strong_symbol and (cls.is_all_uppercase_heading(rest) or cls.has_heading_keyword(rest)):
                return f"{indent}I. {rest}", SymbolDecision.AUTO_CORRECT, "I", f"ROMAN_I_FROM_{prefix}"
            elif prefix == "1." and is_strong_heading and ("NỘI DUNG" in rest.upper() or "CHƯƠNG TRÌNH" in rest.upper()):
                return f"{indent}I. {rest}", SymbolDecision.AUTO_CORRECT, "I", "ROMAN_I_FROM_1_DOT"

        # 5. Nếu đã là số La Mã chuẩn I., II., III., IV., V., VI.
        m_valid = re.match(r"^[ \t]*(I{1,3}|IV|V|VI)\.\s+", line)
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
        # 1. Ký tự '%' ở đầu dòng theo sau là chữ cái tiếng Việt (ví dụ: '% Chi bộ...', '% Từ 90%...')
        # Tiếng Việt không có câu văn/tiêu chí nào hợp lệ bắt đầu bằng '% ' theo sau là chữ cái
        m_pct = re.match(r"^([ \t]*)%\s+([A-Za-zÀ-ỹĐđ].*)$", line)
        if m_pct:
            indent = m_pct.group(1)
            content = m_pct.group(2)
            return f"{indent}+ {content}", SymbolDecision.AUTO_CORRECT, "PLUS_FROM_PERCENT_PREFIX"

        # 2. Ký tự '|' ở đầu dòng trong văn bản thông thường (không phải hàng bảng nhiều cột ' | ')
        if line.count(" | ") < 2:
            # Nếu bắt đầu bằng '| ' theo sau là chữ thường hoặc dấu cộng hoặc ký tự bullet
            m_pipe = re.match(r"^([ \t]*)\|\s+([a-zà-ỹđ\+].*)$", line)
            if m_pipe:
                indent = m_pipe.group(1)
                content = m_pipe.group(2)
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

        for idx, line in enumerate(lines):
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

            recovered_lines.append(curr_line)

        return recovered_lines, audit_trail
