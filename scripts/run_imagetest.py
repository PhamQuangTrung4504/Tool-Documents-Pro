#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts.run_imagetest
=====================
Script kiểm thử tự động E2E trên 4 ảnh thực tế trong assets/imagetest/:
- bang1.png
- bang2.png
- vanban1.png
- vanban2.png

Thực hiện:
1. OCREngine.recognize()
2. VietnameseSpellChecker.check_text_spelling()
3. export_to_docx()
4. Xuất báo cáo chi tiết ra logs/imagetest_report.txt
"""

import os
import sys
import time
from pathlib import Path

# Cấu hình môi trường Windows & Paddle
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PADDLE_PDX_EAGER_INIT"] = "False"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    import torch  # noqa: F401
except Exception:
    pass

import cv2
from loguru import logger

from app.core.docx_exporter import export_to_docx
from app.core.ocr_engine import OCREngine
from app.core.spell_corrector import VietnameseSpellChecker


def run_all_imagetests() -> None:
    logger.info("=" * 70)
    logger.info("BẮT ĐẦU CHẠY E2E AUTOMATED VERIFICATION TRÊN ASSETS/IMAGETEST/")
    logger.info("=" * 70)

    assets_dir = ROOT_DIR / "assets" / "imagetest"
    image_names = ["bang1.png", "bang2.png", "vanban1.png", "vanban2.png"]

    docx_out_dir = ROOT_DIR / "logs" / "imagetest_docx"
    docx_out_dir.mkdir(parents=True, exist_ok=True)

    report_path = ROOT_DIR / "logs" / "imagetest_report.txt"
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append(f"BÁO CÁO KIỂM THỬ E2E 4 ẢNH THẬT (ASSETS/IMAGETEST/) - {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 80 + "\n")

    logger.info("Khởi tạo OCREngine (lang='vi') …")
    engine = OCREngine(lang="vi", use_gpu=False, det_db_box_thresh=0.25)
    spell_checker = VietnameseSpellChecker()

    total_images = len(image_names)
    success_count = 0

    for idx, name in enumerate(image_names, 1):
        img_path = assets_dir / name
        logger.info(f"\n[{idx}/{total_images}] Đang xử lý: {name} (Đường dẫn: {img_path})")

        if not img_path.exists():
            err_msg = f"LỖI: File không tồn tại: {img_path}"
            logger.error(err_msg)
            report_lines.append(err_msg)
            continue

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            err_msg = f"LỖI: Không thể đọc ảnh bằng OpenCV: {img_path}"
            logger.error(err_msg)
            report_lines.append(err_msg)
            continue

        h, w = img_bgr.shape[:2]
        logger.info(f"Kích thước ảnh: {w}x{h} px")

        t0 = time.perf_counter()
        ocr_res = engine.recognize(img_bgr)
        elapsed = time.perf_counter() - t0

        spell_errors = spell_checker.check_text_spelling(ocr_res.full_text)
        docx_file = docx_out_dir / f"{Path(name).stem}.docx"
        export_success = export_to_docx(ocr_res, docx_file)

        tables = ocr_res.tables
        num_tables = len(tables)
        total_merged = sum(
            sum(1 for c in getattr(tbl, "cells", []) if getattr(c, "is_merged", False))
            for tbl in tables
        )

        logger.info(f"Nhận dạng hoàn tất trong {elapsed:.2f}s | Boxes: {len(ocr_res.boxes)} | Tables: {num_tables}")
        logger.info(f"Lỗi chính tả phát hiện: {len(spell_errors)} | Xuất DOCX: {'Thành công' if export_success else 'Thất bại'}")

        # Thêm vào báo cáo
        report_lines.append("-" * 80)
        report_lines.append(f"ẢNH: {name} ({w}x{h} px)")
        report_lines.append(f"Thời gian xử lý: {elapsed:.2f}s")
        report_lines.append(f"Số lượng OCRBox: {len(ocr_res.boxes)}")
        report_lines.append(f"Số lượng Bảng phát hiện: {num_tables}")
        for t_i, tbl in enumerate(tables, 1):
            n_rows = getattr(tbl, "num_rows", len(tbl.grid))
            n_cols = getattr(tbl, "num_cols", len(tbl.grid[0]) if tbl.grid else 0)
            merged_cells = sum(1 for c in getattr(tbl, "cells", []) if getattr(c, "is_merged", False))
            report_lines.append(f"  * Bảng {t_i}: {n_rows} hàng x {n_cols} cột | Ô gộp (merged cells): {merged_cells}")
        report_lines.append(f"Lỗi chính tả (Spell Checker): {len(spell_errors)}")
        if spell_errors:
            report_lines.append("  Các lỗi chính tả:")
            for se in spell_errors[:10]:
                suggs = ", ".join(se.suggestions[:3]) if getattr(se, "suggestions", None) else "Không có"
                tok = getattr(se, "token", str(se))
                report_lines.append(f"    - Từ: '{tok}' (Gợi ý: {suggs})")
        report_lines.append(f"Xuất DOCX: {docx_file} (Success={export_success})")
        report_lines.append("\n--- NỘI DUNG VĂN BẢN NHẬN DẠNG ĐẦY ĐỦ (FULL_TEXT) ---")
        report_lines.append(ocr_res.full_text)
        report_lines.append("--- HẾT NỘI DUNG VĂN BẢN ---\n")

        success_count += 1

    report_content = "\n".join(report_lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    logger.info("=" * 70)
    logger.info(f"HOÀN THÀNH {success_count}/{total_images} ẢNH. BÁO CÁO ĐÃ ĐƯỢC GHI RA: {report_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    run_all_imagetests()
