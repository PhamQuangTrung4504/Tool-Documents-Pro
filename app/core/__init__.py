"""
app.core — Module lõi chứa OCR engine và các tiện ích xử lý ảnh, kiểm tra chính tả.
"""

from app.core.ocr_engine import OCREngine, OCRResult
from app.core.spell_corrector import check_text_spelling

__all__ = ["OCREngine", "OCRResult", "check_text_spelling"]
