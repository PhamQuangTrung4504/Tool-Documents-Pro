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
    - Bảo toàn tiêu đề, căn lề và các đoạn văn bản.
    """
    try:
        import docx  # type: ignore[import-untyped,import-not-found]
        from docx.shared import Inches, Pt, RGBColor  # type: ignore[import-untyped,import-not-found]
        from docx.enum.text import WD_ALIGN_PARAGRAPH  # type: ignore[import-untyped,import-not-found]
        from docx.enum.table import WD_TABLE_ALIGNMENT  # type: ignore[import-untyped,import-not-found]
    except ImportError:
        logger.error("python-docx is not installed. Cannot export to DOCX.")
        return False

    try:
        doc: Any = docx.Document()

        # Thiết lập lề trang tiêu chuẩn (A4: Top/Bottom 2cm, Left 2.5cm, Right 2cm)
        for section in doc.sections:
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
            hdr_lines_set = set(l.strip() for l in (hdr.left_lines + hdr.right_lines) if l.strip())

            hdr_table: Any = doc.add_table(rows=1, cols=2)
            hdr_table.alignment = WD_TABLE_ALIGNMENT.CENTER

            # Gỡ bỏ đường viền của bảng header
            try:
                from docx.oxml import parse_xml  # type: ignore[import-untyped,import-not-found]
                from docx.oxml.ns import nsdecls  # type: ignore[import-untyped,import-not-found]
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

        # 2. Xuất phần thân văn bản và các bảng biểu
        lines = result.full_text.split("\n")
        skipping_header = header_rendered
        table_idx = 0
        in_table_mode = False

        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            if skipping_header and line_str in hdr_lines_set:
                continue
            skipping_header = False

            # Kiểm tra nếu dòng thuộc bảng
            if result.tables and table_idx < len(result.tables) and " | " in line_str:
                tbl = result.tables[table_idx]
                cells = [c.strip() for c in line_str.split(" | ")]
                if len(cells) == tbl.num_cols or abs(len(cells) - tbl.num_cols) <= 1:
                    if not in_table_mode:
                        doc_table: Any = doc.add_table(rows=len(tbl.grid), cols=tbl.num_cols)
                        doc_table.alignment = WD_TABLE_ALIGNMENT.CENTER
                        doc_table.style = "Table Grid"

                        # Tính độ rộng cột tỷ lệ (Proportional column widths)
                        total_width_inches = 6.5
                        col_max_lens = [
                            max((len(tbl.grid[r][c].strip()) for r in range(len(tbl.grid))), default=5)
                            for c in range(tbl.num_cols)
                        ]
                        total_len = max(1, sum(col_max_lens))
                        col_widths_inches = []
                        for c_len in col_max_lens:
                            w_inch = max(0.6, min(4.0, (c_len / total_len) * total_width_inches))
                            col_widths_inches.append(w_inch)

                        sum_w = sum(col_widths_inches)
                        if sum_w > 0:
                            col_widths_inches = [(w / sum_w) * total_width_inches for w in col_widths_inches]

                        # Lặp tiêu đề bảng ở mỗi trang Word nếu bảng dài (tblHeader)
                        if len(tbl.grid) > 0:
                            try:
                                from docx.oxml import parse_xml  # type: ignore[import-untyped,import-not-found]
                                from docx.oxml.ns import nsdecls  # type: ignore[import-untyped,import-not-found]
                                header_tr: Any = doc_table.rows[0]._tr.get_or_add_trPr()
                                header_tr.append(parse_xml(r'<w:tblHeader %s/>' % nsdecls("w")))
                            except Exception:
                                pass

                        has_merged_cells = bool(tbl.cells and any(c.is_merged for c in tbl.cells))
                        if has_merged_cells:
                            for c in tbl.cells:
                                if not c.is_merged:
                                    try:
                                        w_cell = doc_table.cell(c.row_start, c.col_start)
                                        w_cell.text = c.text
                                    except Exception as e_cell:
                                        logger.debug(f"Error populating unmerged cell: {e_cell}")
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
                                    if c_i < len(col_widths_inches):
                                        cell.width = Inches(col_widths_inches[c_i])
                                    cell.text = row_data[c_i].strip()

                        for r_i, row in enumerate(doc_table.rows):
                            for c_i, cell in enumerate(row.cells):
                                if c_i < len(col_widths_inches):
                                    try:
                                        cell.width = Inches(col_widths_inches[c_i])
                                    except Exception:
                                        pass
                                cell_text = cell.text.strip()
                                for p in cell.paragraphs:
                                    p.paragraph_format.space_before = Pt(2)
                                    p.paragraph_format.space_after = Pt(2)
                                    for run in p.runs:
                                        run.font.name = "Times New Roman"
                                        run.font.size = Pt(11)
                                        if r_i == 0:
                                            run.bold = True
                                    if r_i == 0 or (c_i != 1 and len(cell_text) <= 5):
                                        p.alignment = WD_ALIGN_PARAGRAPH.CENTER

                        in_table_mode = True
                        table_idx += 1
                    continue

            in_table_mode = False
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.line_spacing = 1.15

            is_centered_title = any(kw in line_str.upper() for kw in [
                "BIÊN BẢN", "QUYẾT ĐỊNH", "THÔNG TƯ", "NGHỊ ĐỊNH", "BÁO CÁO",
                "KẾ HOẠCH", "BẢNG ĐIỂM", "ĐẢNG CỘNG SẢN VIỆT NAM",
                "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM", "ĐỘC LẬP - TỰ DO - HẠNH PHÚC",
                "HỢP ĐỒNG", "HÓA ĐƠN", "PHIẾU THU", "PHIẾU CHI", "BÁO GIÁ",
                "BẢNG KÊ", "GIẤY XÁC NHẬN", "GIẤY CHỨNG NHẬN", "CHƯƠNG",
                "BÀI HỌC", "GIÁO TRÌNH", "ĐỀ THI", "ĐÁP ÁN", "TỔNG KẾT",
                "PHIẾU KHÁM", "KẾT QUẢ XÉT NGHIỆM"
            ])
            if is_centered_title:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                run = p.add_run(line_str)
                run.bold = True
                run.font.name = "Times New Roman"
                run.font.size = Pt(13)
            else:
                run = p.add_run(line_str)
                run.font.name = "Times New Roman"
                run.font.size = Pt(12)

        doc.save(output_path)
        logger.info(f"DOCX exported successfully: {output_path}")
        return True
    except Exception as exc:
        logger.error(f"Error exporting to DOCX: {exc}")
        return False

