"""
app.core — Module lõi chứa OCR engine và các tiện ích xử lý ảnh, kiểm tra chính tả.
"""

from app.core.ocr_engine import OCREngine, OCRResult
from app.core.spell_corrector import (
    SpellError,
    check_text_spelling,
    check_text_spelling_detailed,
    get_suggestions,
)

__all__ = [
    "OCREngine",
    "OCRResult",
    "SpellError",
    "check_text_spelling",
    "check_text_spelling_detailed",
    "get_suggestions",
]
