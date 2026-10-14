import re
from typing import List, Tuple

ROMAN_HEADING_KEYWORDS = {
    "NỘI DUNG", "TIẾN HÀNH", "KẾT LUẬN", "TỔ CHỨC", "MỤC ĐÍCH", "YÊU CẦU",
    "KẾ HOẠCH", "PHƯƠNG HƯỚNG", "NHIỆM VỤ", "ĐÁNH GIÁ", "CHƯƠNG TRÌNH",
    "THỰC HIỆN", "KẾT QUẢ", "THỜI GIAN", "ĐỊA ĐIỂM", "THÀNH PHẦN",
}

def is_all_uppercase_heading(text: str) -> bool:
    """Kiểm tra văn bản có phải là đề mục in hoa toàn bộ hay không."""
    words = [w for w in re.findall(r'[a-zA-ZÀ-ỹĐđ]+', text)]
    if not words:
        return False
    # Ít nhất 70% từ in hoa
    upper_count = sum(1 for w in words if w.isupper())
    return (upper_count / len(words)) >= 0.7

def has_heading_keyword(text: str) -> bool:
    """Kiểm tra văn bản có chứa từ khóa đề mục hay không."""
    text_upper = text.upper()
    return any(kw in text_upper for kw in ROMAN_HEADING_KEYWORDS)

def recover_roman_numerals_in_line(line: str, prev_roman: str = "") -> Tuple[str, str]:
    """
    Khôi phục số La Mã đầu mục (I., II., III., IV., V.) dựa trên ngữ cảnh.
    Trả về (new_line, current_roman).
    """
    stripped = line.strip()
    if not stripped:
        return line, prev_roman

    # 1. Kiểm tra mẫu số II: TI., ll., 11., Tl., lI., Il.
    m_ii = re.match(r"^([ \t]*)(?:TI|ll|11|Tl|lI|Il)\.?\s+([A-ZÀ-ỸĐ\s,;:/-]+)$", line)
    if m_ii:
        rest = m_ii.group(2).strip()
        if is_all_uppercase_heading(rest) or has_heading_keyword(rest):
            return f"{m_ii.group(1)}II. {rest}", "II"

    # 2. Kiểm tra mẫu số III: lll., 111., TII., TIl.
    m_iii = re.match(r"^([ \t]*)(?:lll|111|TII|TIl|llI)\.?\s+([A-ZÀ-ỸĐ\s,;:/-]+)$", line)
    if m_iii:
        rest = m_iii.group(2).strip()
        if is_all_uppercase_heading(rest) or has_heading_keyword(rest):
            return f"{m_iii.group(1)}III. {rest}", "III"

    # 3. Kiểm tra mẫu số IV: lV., 1V., TV.
    m_iv = re.match(r"^([ \t]*)(?:lV|1V|TV)\.?\s+([A-ZÀ-ỸĐ\s,;:/-]+)$", line)
    if m_iv:
        rest = m_iv.group(2).strip()
        if is_all_uppercase_heading(rest) or has_heading_keyword(rest):
            return f"{m_iv.group(1)}IV. {rest}", "IV"

    # 4. Kiểm tra mẫu số I: |, |., l., 1.
    # CHÚ Ý ĐIỀU KIỆN ÂM TÍNH: Không đổi 1. thành I. nếu theo sau là chữ thường (ví dụ: 1. Tuyên bố...)
    m_i = re.match(r"^([ \t]*)(?:\|\.?|l\.|1\.)\s+([A-ZÀ-ỸĐ\s,;:/-]+)$", line)
    if m_i:
        rest = m_i.group(2).strip()
        prefix = m_i.group(0).strip().split()[0]
        # Nếu prefix là '|' hoặc '|.' hoặc 'l.': Bằng chứng rất mạnh là số La Mã I.
        # Nếu prefix là '1.': Chỉ đổi thành I. khi có từ khóa đề mục (NỘI DUNG, CHƯƠNG TRÌNH,...)
        is_strong_symbol = prefix in ["|", "|.", "l."]
        is_strong_heading = has_heading_keyword(rest) and is_all_uppercase_heading(rest)
        if is_strong_symbol and (is_all_uppercase_heading(rest) or is_strong_heading):
            return f"{m_i.group(1)}I. {rest}", "I"
        elif prefix == "1." and is_strong_heading and ("NỘI DUNG" in rest or "CHƯƠNG TRÌNH" in rest):
            return f"{m_i.group(1)}I. {rest}", "I"

    # Nếu đã là I., II., III.,... chuẩn
    m_valid = re.match(r"^[ \t]*(I{1,3}|IV|V|VI)\.\s+", line)
    if m_valid:
        return line, m_valid.group(1)

    return line, prev_roman

def recover_bullets_and_pluses_in_line(line: str) -> str:
    """
    Khôi phục dấu cộng (+) hoặc bullet bị nhận sai thành % hoặc | ở đầu dòng/ô.
    Bảo toàn 100% các giá trị phần trăm hợp lệ (90%, 75%, 100%).
    """
    # 1. Ký tự '%' ở đầu dòng theo sau là chữ cái tiếng Việt (ví dụ: '% Chi bộ...', '% Từ 90%...')
    m_pct = re.match(r"^([ \t]*)%\s+([A-Za-zÀ-ỹĐđ].*)$", line)
    if m_pct:
        indent = m_pct.group(1)
        content = m_pct.group(2)
        return f"{indent}+ {content}"

    # 2. Ký tự '|' ở đầu dòng trong văn bản thông thường (không phải bảng có nhiều cột ' | ')
    # Nếu dòng chứa >= 2 dấu ' | ' thì đây là hàng bảng, KHÔNG ĐỔI
    if line.count(" | ") < 2:
        m_pipe = re.match(r"^([ \t]*)\|\s+([a-zà-ỹđ\+].*)$", line)
        if m_pipe:
            indent = m_pipe.group(1)
            content = m_pipe.group(2)
            return f"{indent}+ {content}"

    return line

def recover_symbols_in_document(lines: List[str]) -> List[str]:
    """Phục hồi symbol và số La Mã toàn diện cho danh sách các dòng tài liệu."""
    result = []
    current_roman = ""
    for line in lines:
        line_clean = line
        # 1. Khôi phục số La Mã
        line_clean, current_roman = recover_roman_numerals_in_line(line_clean, current_roman)
        # 2. Khôi phục Bullet / Plus
        line_clean = recover_bullets_and_pluses_in_line(line_clean)
        result.append(line_clean)
    return result

# --- RUN TESTS ---
test_lines = [
    "| NỘI DUNG, CHƯƠNG TRÌNH SINH HOẠT CHI BỘ",
    "1. Tuyên bố lý do, giới thiệu đại biểu:",
    "- Lý do: Tổ chức sinh hoạt chi bộ định kỳ tháng 10 năm 2026.",
    "2. Chủ trì: Đồng chí Lê Văn Điền, Bí thư chi bộ.",
    "TI. TIẾN HÀNH SINH HOẠT",
    "1. Đồng chí Lê Văn Điền, Bí thư Chi bộ quán triệt mục đích...",
    "% Chi bộ dưới 30 đảng viên ít nhất 90 phút; chi bộ trên 30 đảng viên ít nhất 120 phút",
    "% Từ 90% trở lên",
    "% Từ 75% đến dưới 90%",
    "Tỷ lệ đạt 90%.",
    "Phép tính 50% + 50% = 100%",
]

recovered = recover_symbols_in_document(test_lines)
print("\n=== RECOVERED RESULTS ===")
for orig, rec in zip(test_lines, recovered):
    print(f"ORIG: {orig}")
    print(f"REC : {rec}")
    print("-" * 50)
