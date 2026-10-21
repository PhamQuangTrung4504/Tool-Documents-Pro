from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from app.core.document_renderer import DocumentRenderService, InvalidDocxError, RendererUnavailable
from app.core.document_session import DocumentSession
from app.core.docx_exporter import export_to_docx
from app.core.models import OCRResult


class DocumentPreviewTests(unittest.TestCase):
    def test_session_paths_are_unique_and_owned(self) -> None:
        with TemporaryDirectory() as folder:
            root = Path(folder)
            first = DocumentSession.create(root)
            second = DocumentSession.create(root)

            self.assertNotEqual(first.directory, second.directory)
            self.assertEqual(first.docx_path.parent, first.directory)
            self.assertTrue(first.owns(first.docx_path))
            self.assertFalse(first.owns(root / "unrelated.docx"))
            self.assertTrue(first.cleanup())
            self.assertFalse(first.directory.exists())
            self.assertTrue(second.directory.exists())
            self.assertTrue(second.cleanup())

    def test_exporter_creates_valid_session_docx(self) -> None:
        from docx import Document

        with TemporaryDirectory() as folder:
            session = DocumentSession.create(Path(folder))
            try:
                result = OCRResult(full_text="Test OCR document\nSecond line")
                self.assertTrue(export_to_docx(result, str(session.docx_path)))
                self.assertTrue(session.docx_path.is_file())
                document = Document(session.docx_path)
                self.assertIn("Test OCR document", "\n".join(p.text for p in document.paragraphs))
            finally:
                self.assertTrue(session.cleanup())

    def test_renderer_rejects_missing_docx(self) -> None:
        with TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaisesRegex(InvalidDocxError, "missing or empty"):
                DocumentRenderService().render_docx(root / "missing.docx", root / "preview.pdf")

    def test_renderer_distinguishes_invalid_docx(self) -> None:
        with TemporaryDirectory() as folder:
            root = Path(folder)
            invalid = root / "invalid.docx"
            invalid.write_bytes(b"not an OOXML package")
            with self.assertRaises(InvalidDocxError):
                DocumentRenderService().render_docx(invalid, root / "preview.pdf")

    def test_renderer_reports_unavailable_without_office(self) -> None:
        with TemporaryDirectory() as folder:
            session = DocumentSession.create(Path(folder))
            try:
                self.assertTrue(export_to_docx(OCRResult(full_text="Renderer availability"), str(session.docx_path)))
                with patch("app.core.document_renderer._is_windows", return_value=False), patch(
                    "app.core.document_renderer.shutil.which", return_value=None
                ):
                    with self.assertRaisesRegex(RendererUnavailable, "DOCX remains available to save"):
                        DocumentRenderService().render_docx(session.docx_path, session.pdf_path)
            finally:
                self.assertTrue(session.cleanup())
