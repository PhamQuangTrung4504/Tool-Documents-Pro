#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
app.core.docx_exporter
======================
Module xuất kết quả nhận dạng OCR ra định dạng Microsoft Word (.docx):
- Bảo toàn cấu trúc bảng biểu thành Word Table chuẩn.
- Bảo toàn tiêu đề 2 khối song song (Quốc hiệu / Cơ quan ban hành).
- Bảo toàn định dạng căn lề, in đậm và cỡ chữ chuẩn văn bản hành chính.
"""

from __future__ import annotations

from typing import Any
from loguru import logger

from app.core.models import OCRResult, OCRTable, ParallelHeader

__all__ = [
    "export_to_docx",
]

def export_to_docx(result: OCRResult, output_path: str) -> bool:
    """
    Xuất kết quả OCR sang tài liệu Microsoft Word (.docx).
    - Bảo toàn cấu trúc bảng biểu thành Word Table chuẩn khi có bảng.
    - Tuyệt đối KHÔNG in lặp lại nội dung bảng thành văn bản thường.
    - Tự động xoay khổ giấy Ngang (Landscape) cho bảng rộng (>= 8 cột).
    - Bảo toàn tiêu đề 2 khối song song và căn chỉnh chuẩn văn bản hành chính.
    """
    try:
        import docx  # type: ignore[import-untyped,import-not-found]
        from docx.shared import Inches, Pt, RGBColor, Mm  # type: ignore[import-untyped,import-not-found]
        from docx.enum.text import WD_ALIGN_PARAGRAPH  # type: ignore[import-untyped,import-not-found]
        from docx.enum.table import WD_TABLE_ALIGNMENT  # type: ignore[import-untyped,import-not-found]
        from docx.enum.section import WD_ORIENT  # type: ignore[import-untyped,import-not-found]
        from docx.oxml import parse_xml  # type: ignore[import-untyped,import-not-found]
        from docx.oxml.ns import nsdecls  # type: ignore[import-untyped,import-not-found]
    except ImportError:
        logger.error("python-docx is not installed. Cannot export to DOCX.")
        return False

    try:
        import re
        doc: Any = docx.Document()

        # Kiểm tra bảng rộng (num_cols >= 8) để tự động xoay khổ giấy Ngang (Landscape)
        has_wide_table = bool(result.tables and any(tbl.num_cols >= 8 for tbl in result.tables))
        section = doc.sections[0]
        if has_wide_table:
            section.orientation = WD_ORIENT.LANDSCAPE
            section.page_width = Mm(297)
            section.page_height = Mm(210)
            section.left_margin = Mm(12)
            section.right_margin = Mm(12)
            section.top_margin = Mm(15)
            section.bottom_margin = Mm(15)
        else:
            section.top_margin = Inches(0.8)
            section.bottom_margin = Inches(0.8)
            section.left_margin = Inches(1.0)
            section.right_margin = Inches(0.8)

        # Thiết lập style Normal
        style: Any = doc.styles["Normal"]
        font = style.font
        font.name = "Times New Roman"
        font.size = Pt(12)
        font.color.rgb = RGBColor(0, 0, 0)

        # 1. Nếu có parallel_header (Tiêu đề 2 khối song song), tạo bảng 1 hàng 2 cột không viền ở đầu trang Word
        header_rendered = False
        hdr_lines_set = set()
        if result.parallel_header and (result.parallel_header.left_lines or result.parallel_header.right_lines):
            hdr = result.parallel_header
            hdr_lines_set = set(re.sub(r"\s+", " ", l.strip().lower()) for l in (hdr.left_lines + hdr.right_lines) if l.strip())

            hdr_table: Any = doc.add_table(rows=1, cols=2)
            hdr_table.alignment = WD_TABLE_ALIGNMENT.CENTER

            try:
                tblPr: Any = hdr_table._tbl.tblPr
                borders = parse_xml(
                    r'<w:tblBorders %s><w:top w:val="none"/><w:left w:val="none"/>'
                    r'<w:bottom w:val="none"/><w:right w:val="none"/>'
                    r'<w:insideH w:val="none"/><w:insideV w:val="none"/></w:tblBorders>' % nsdecls("w")
                )
                tblPr.append(borders)
            except Exception as e_border:
                logger.debug(f"Could not remove table borders: {e_border}")

            # Khối trái: Cơ quan ban hành, số hiệu
            cell_left = hdr_table.cell(0, 0)
            for idx, line_text in enumerate(hdr.left_lines):
                p = cell_left.paragraphs[0] if idx == 0 else cell_left.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_before = Pt(1)
                p.paragraph_format.space_after = Pt(1)
                run = p.add_run(line_text)
                run.font.name = "Times New Roman"
                run.font.size = Pt(11)
                if idx == 0 or line_text.isupper():
                    run.bold = True

            # Khối phải: Quốc hiệu, tiêu ngữ, ngày tháng
            cell_right = hdr_table.cell(0, 1)
            for idx, line_text in enumerate(hdr.right_lines):
                p = cell_right.paragraphs[0] if idx == 0 else cell_right.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_before = Pt(1)
                p.paragraph_format.space_after = Pt(1)
                run = p.add_run(line_text)
                run.font.name = "Times New Roman"
                run.font.size = Pt(11)
                if any(kw in line_text for kw in ["ĐẢNG CỘNG SẢN", "CỘNG HÒA", "ĐỘC LẬP"]):
                    run.bold = True
                elif "ngày" in line_text.lower() or "tháng" in line_text.lower():
                    run.italic = True

            header_rendered = True
            p_spacer = doc.add_paragraph()
            p_spacer.paragraph_format.space_before = Pt(4)
            p_spacer.paragraph_format.space_after = Pt(4)

        # 2. Xây dựng tập hợp các dòng text thuộc bảng để CHẶN TUYỆT ĐỐI việc in lặp ra paragraph
        # Gom tất cả các cell và hàng của mọi bảng trong result.tables
        table_cell_signatures = set()
        for tbl in result.tables:
            for row in tbl.grid:
                # Toàn bộ nội dung hàng nối lại
                row_joined = " ".join(c.strip() for c in row if c.strip())
                if row_joined:
                    table_cell_signatures.add(re.sub(r"\s+", " ", row_joined.lower()))
                for cell in row:
                    c_clean = cell.strip()
                    if c_clean:
                        table_cell_signatures.add(re.sub(r"\s+", " ", c_clean.lower()))

        def _is_table_first_row(line_s: str, tbl: OCRTable) -> bool:
            """Bảng chỉ có thể bắt đầu bằng dòng có ít nhất 2 cột chứa dấu '|'."""
            if " | " not in line_s and line_s.count("|") < 1:
                return False
            parts = [p.strip() for p in line_s.split("|") if p.strip()]
            return len(parts) >= 2

        def _is_table_row_line(line_s: str, tbl: OCRTable) -> bool:
            """Kiểm tra một dòng text có thuộc bảng hay không."""
            if " | " in line_s or line_s.count("|") >= 1:
                return True
            s_clean = re.sub(r"\s+", " ", line_s.strip().lower())
            if s_clean in table_cell_signatures:
                return True
            # Kiểm tra dòng số thứ tự hoặc tổng cộng lẻ: '1', '2', 'Tổng cộng'
            if re.match(r"^(?:\d{1,2}|Tổng\s+cộng)$", line_s.strip(), re.IGNORECASE):
                return True
            return False

        def _render_word_table(tbl: OCRTable) -> Any:
            """Vẽ đúng 1 bảng Word từ OCRTable."""
            is_wide = tbl.num_cols >= 8
            font_size_pt = Pt(9.0) if is_wide else Pt(11.0)
            doc_table: Any = doc.add_table(rows=len(tbl.grid), cols=tbl.num_cols)
            doc_table.alignment = WD_TABLE_ALIGNMENT.CENTER
            doc_table.style = "Table Grid"

            # Tính độ rộng cột theo tỷ lệ
            total_width_inches = 10.5 if is_wide else 6.5
            col_max_lens = [
                max((len(tbl.grid[r][c].strip()) for r in range(len(tbl.grid))), default=5)
                for c in range(tbl.num_cols)
            ]
            total_len = max(1, sum(col_max_lens))
            col_widths_inches = []
            for c_i, c_len in enumerate(col_max_lens):
                if is_wide:
                    min_w = 0.35 if c_i in {0, 3, 4, 13, 14} else 0.55
                    max_w = 2.0
                else:
                    min_w = 0.6
                    max_w = 4.0
                w_inch = max(min_w, min(max_w, (c_len / total_len) * total_width_inches))
                col_widths_inches.append(w_inch)

            sum_w = sum(col_widths_inches)
            if sum_w > 0:
                col_widths_inches = [(w / sum_w) * total_width_inches for w in col_widths_inches]

            # tblHeader: Lặp tiêu đề bảng ở mỗi trang Word nếu bảng dài
            if len(tbl.grid) > 0:
                try:
                    header_tr: Any = doc_table.rows[0]._tr.get_or_add_trPr()
                    header_tr.append(parse_xml(r'<w:tblHeader %s/>' % nsdecls("w")))
                except Exception:
                    pass

            has_merged_cells = bool(tbl.cells and any(c.is_merged for c in tbl.cells))
            if has_merged_cells:
                # Đổ dữ liệu các ô không gộp
                for c in tbl.cells:
                    if not c.is_merged:
                        try:
                            w_cell = doc_table.cell(c.row_start, c.col_start)
                            w_cell.text = c.text
                        except Exception as e_cell:
                            logger.debug(f"Error populating unmerged cell: {e_cell}")
                # Đổ dữ liệu các ô gộp
                for c in tbl.cells:
                    if c.is_merged:
                        try:
                            c_tl = doc_table.cell(c.row_start, c.col_start)
                            c_br = doc_table.cell(c.row_end - 1, c.col_end - 1)
                            merged_w_cell = c_tl.merge(c_br)
                            merged_w_cell.text = c.text
                        except Exception as e_merge:
                            logger.debug(f"Error merging table cells: {e_merge}")
            else:
                for r_i, row_data in enumerate(tbl.grid):
                    for c_i in range(min(len(row_data), tbl.num_cols)):
                        cell = doc_table.cell(r_i, c_i)
                        cell.text = row_data[c_i].strip()

            # Định dạng font, cỡ chữ, căn lề và độ rộng cột
            for r_i, row in enumerate(doc_table.rows):
                for c_i, cell in enumerate(row.cells):
                    if c_i < len(col_widths_inches):
                        try:
                            cell.width = Inches(col_widths_inches[c_i])
                        except Exception:
                            pass
                    cell_text = cell.text.strip()
                    for p in cell.paragraphs:
                        p.paragraph_format.space_before = Pt(1) if is_wide else Pt(2)
                        p.paragraph_format.space_after = Pt(1) if is_wide else Pt(2)
                        p.paragraph_format.line_spacing = 1.0 if is_wide else 1.15
                        for run in p.runs:
                            run.font.name = "Times New Roman"
                            run.font.size = font_size_pt
                            if r_i == 0 or (r_i <= 2 and is_wide and (any(ch.isdigit() for ch in cell_text) or any(kw in cell_text.lower() for kw in ["tt", "nam", "nữ", "dân tộc"]))):
                                run.bold = True
                        if r_i == 0 or (c_i != 1 and len(cell_text) <= 6) or re.match(r"^\(\d+\)$", cell_text):
                            p.alignment = WD_ALIGN_PARAGRAPH.CENTER

            return doc_table

        # 3. Xuất phần thân văn bản và các bảng biểu
        lines = result.full_text.split("\n")
        table_idx = 0
        in_table_range = False
        last_was_major_title = False
        hit_major_title = False

        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            line_clean = re.sub(r"\s+", " ", line_str.lower())
            if header_rendered and not hit_major_title:
                if line_clean in hdr_lines_set or any(kw in line_str.upper() for kw in ["ỦY BAN", "UY BAN", "CỘNG HÒA", "ĐỘC LẬP", "ĐẢNG CỘNG SẢN"]):
                    continue
                if re.match(r"^\d{1,2}$", line_str):
                    continue

            # Kiểm tra nếu dòng thuộc bảng biểu
            if result.tables and table_idx < len(result.tables):
                current_tbl = result.tables[table_idx]
                if not in_table_range:
                    if _is_table_first_row(line_str, current_tbl):
                        # Vị trí xuất hiện đầu tiên của bảng: VẼ BẢNG ĐÚNG 1 LẦN
                        _render_word_table(current_tbl)
                        in_table_range = True
                        continue
                else:
                    if _is_table_row_line(line_str, current_tbl):
                        # Bỏ qua dòng này, không in ra paragraph!
                        continue
                    else:
                        # Vừa kết thúc vùng bảng hiện tại
                        in_table_range = False
                        table_idx += 1
                        if table_idx < len(result.tables):
                            next_tbl = result.tables[table_idx]
                            if _is_table_first_row(line_str, next_tbl):
                                _render_word_table(next_tbl)
                                in_table_range = True
                                continue

            # Dòng văn bản thông thường ngoài bảng (Pre-table hoặc Post-table)
            in_table_range = False
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.line_spacing = 1.15

            # Quy tắc định dạng tiêu đề, đề mục La Mã, mục con:
            # a. Mục con bắt đầu bằng số thứ tự (^\d+\.\s+ như 1., 2., ..., 6. Đồng chí...):
            # TUYỆT ĐỐI KHÔNG CĂN GIỮA VÀ KHÔNG IN ĐẬM TOÀN DÒNG!
            is_numbered_sub_item = bool(re.match(r"^\d+\.\s+", line_str))

            # b. Đề mục La Mã lớn (I. NỘI DUNG..., II. TIẾN HÀNH..., III. KẾT LUẬN...):
            # BẮT BUỘC IN ĐẬM VÀ CĂN LỀ TRÁI
            is_roman_heading = bool(re.match(r"^(?:I{1,3}|IV|V|VI{0,3}|IX|X)\.\s+", line_str))

            # c. Tiêu đề lớn chính giữa trang (BIÊN BẢN, DANH SÁCH, QUYẾT ĐỊNH,...)
            is_major_title = False
            if not is_numbered_sub_item and not is_roman_heading:
                line_u = line_str.upper()
                major_title_kws = [
                    "BIÊN BẢN", "QUYẾT ĐỊNH", "THÔNG TƯ", "NGHỊ ĐỊNH", "BÁO CÁO",
                    "KẾ HOẠCH", "BẢNG ĐIỂM", "ĐẢNG CỘNG SẢN VIỆT NAM",
                    "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM", "ĐỘC LẬP - TỰ DO - HẠNH PHÚC",
                    "HỢP ĐỒNG", "HÓA ĐƠN", "PHIẾU THU", "PHIẾU CHI", "BÁO GIÁ",
                    "BẢNG KÊ", "GIẤY XÁC NHẬN", "GIẤY CHỨNG NHẬN",
                    "BÀI HỌC", "GIÁO TRÌNH", "ĐỀ THI", "ĐÁP ÁN", "TỔNG KẾT",
                    "PHIẾU KHÁM", "KẾT QUẢ XÉT NGHIỆM", "DANH SÁCH",
                ]
                is_major_title = any(kw == line_u or line_u.startswith(kw + " ") for kw in major_title_kws)

            # d. Tiêu đề phụ nằm ngay dưới tiêu đề chính (sinh hoạt chi bộ định kỳ..., Công nhận người có uy tín...)
            is_subtitle_under_major = False
            if last_was_major_title and not is_numbered_sub_item and not is_roman_heading:
                if not line_str.startswith("(") and len(line_str) <= 120:
                    is_subtitle_under_major = True

            # e. Chú thích đóng mở ngoặc dưới tiêu đề: (Ban hành kèm theo...)
            is_parenthesized_note = bool(line_str.startswith("(") and line_str.endswith(")"))

            if is_numbered_sub_item:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                run = p.add_run(line_str)
                run.font.name = "Times New Roman"
                run.font.size = Pt(12)
                last_was_major_title = False
            elif is_roman_heading:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                p.paragraph_format.space_before = Pt(6)
                run = p.add_run(line_str)
                run.bold = True
                run.font.name = "Times New Roman"
                run.font.size = Pt(12.5)
                last_was_major_title = False
            elif is_major_title:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_before = Pt(6)
                p.paragraph_format.space_after = Pt(2)
                run = p.add_run(line_str)
                run.bold = True
                run.font.name = "Times New Roman"
                run.font.size = Pt(13)
                last_was_major_title = True
                hit_major_title = True
            elif is_subtitle_under_major:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_before = Pt(1)
                p.paragraph_format.space_after = Pt(4)
                run = p.add_run(line_str)
                run.bold = True
                run.font.name = "Times New Roman"
                run.font.size = Pt(12)
                last_was_major_title = False
            elif is_parenthesized_note:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_before = Pt(1)
                p.paragraph_format.space_after = Pt(4)
                run = p.add_run(line_str)
                run.italic = True
                run.font.name = "Times New Roman"
                run.font.size = Pt(11)
                last_was_major_title = False
            elif line_str.startswith("*") or line_str.startswith("Ghi chú:"):
                # Dòng ghi chú chân trang
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                p.paragraph_format.space_before = Pt(6)
                run = p.add_run(line_str)
                run.font.name = "Times New Roman"
                run.font.size = Pt(11)
                if line_str.startswith("Ghi chú:"):
                    run.bold = True
                last_was_major_title = False
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                run = p.add_run(line_str)
                run.font.name = "Times New Roman"
                run.font.size = Pt(12)
                last_was_major_title = False

        doc.save(output_path)
        logger.info(f"DOCX exported successfully: {output_path}")
        return True
    except Exception as exc:
        logger.error(f"Error exporting to DOCX: {exc}")
        return False

