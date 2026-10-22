"""
app.core — Module lõi chứa OCR engine và các tiện ích xử lý ảnh, kiểm tra chính tả.
"""

from app.core.ocr_engine import OCREngine, OCRResult, recover_emails_and_urls, format_progress_table_row
from app.core.spell_corrector import (
    SpellError,
    check_text_spelling,
    check_text_spelling_detailed,
    get_suggestions,
)
from app.core.structured_editing import (
    CorrectionRecord,
    EditOwnershipError,
    EditTarget,
    StructuredDocumentEditor,
)

__all__ = [
    "OCREngine",
    "OCRResult",
    "recover_emails_and_urls",
    "format_progress_table_row",
    "SpellError",
    "check_text_spelling",
    "check_text_spelling_detailed",
    "get_suggestions",
    "CorrectionRecord",
    "EditOwnershipError",
    "EditTarget",
    "StructuredDocumentEditor",
]
