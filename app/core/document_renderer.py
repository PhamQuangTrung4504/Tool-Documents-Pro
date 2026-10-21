"""DOCX validation and isolated Word/LibreOffice-to-PDF rendering."""
from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

from loguru import logger


class DocumentRenderError(RuntimeError):
    """Base class for actionable, non-sensitive rendering diagnostics."""


class InvalidDocxError(DocumentRenderError):
    pass


class MissingComDependency(DocumentRenderError):
    pass


class ComImportFailure(DocumentRenderError):
    pass


class WordComInitializationError(DocumentRenderError):
    pass


class WordActivationError(DocumentRenderError):
    pass


class WordConversionError(DocumentRenderError):
    pass


class PdfLibraryUnavailable(DocumentRenderError):
    pass


class PdfValidationError(DocumentRenderError):
    pass


class RendererUnavailable(DocumentRenderError):
    pass


def _hresult(error: BaseException) -> str:
    value = getattr(error, "hresult", None)
    return f"0x{value & 0xFFFFFFFF:08X}" if isinstance(value, int) else "unavailable"


def _is_windows() -> bool:
    return os.name == "nt"


def _word_com_registration() -> str:
    if not _is_windows():
        return "not-applicable"
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, r"Word.Application\CLSID") as key:
            clsid, _ = winreg.QueryValueEx(key, None)
        return f"registered ({clsid})"
    except OSError:
        return "not-registered"


def _validate_docx(docx_path: Path) -> None:
    if not docx_path.is_file() or docx_path.stat().st_size == 0:
        raise InvalidDocxError("The temporary DOCX is missing or empty.")
    try:
        with zipfile.ZipFile(docx_path) as archive:
            members = set(archive.namelist())
            required = {"[Content_Types].xml", "word/document.xml"}
            missing = required - members
            corrupt = archive.testzip()
            if missing or corrupt:
                detail = "required OOXML parts are missing" if missing else "an OOXML part is corrupt"
                raise InvalidDocxError(f"The temporary DOCX is invalid: {detail}.")
    except (OSError, zipfile.BadZipFile) as exc:
        raise InvalidDocxError("The temporary DOCX is not a readable OOXML package.") from exc


def _validate_pdf(pdf_path: Path) -> int:
    try:
        pymupdf = importlib.import_module("pymupdf")
    except ImportError as exc:
        raise PdfLibraryUnavailable("PyMuPDF is unavailable in the application environment.") from exc
    try:
        with pymupdf.open(pdf_path) as document:
            page_count = document.page_count
            if page_count < 1:
                raise PdfValidationError("The renderer produced a PDF with no pages.")
            # Force page-tree access while still on the renderer worker, so the UI
            # does not receive a PDF that only fails when its first page is opened.
            for index in range(page_count):
                document.load_page(index).rect
            return page_count
    except PdfValidationError:
        raise
    except Exception as exc:
        raise PdfValidationError("The renderer produced a PDF that PyMuPDF could not open.") from exc


class DocumentRenderService:
    """Prefer a private Word COM instance on Windows, then optional LibreOffice."""

    def __init__(self) -> None:
        self.last_renderer: str | None = None

    def render_docx(self, docx_path: Path, pdf_path: Path) -> Path:
        docx_path = Path(docx_path).resolve()
        pdf_path = Path(pdf_path).resolve()
        _validate_docx(docx_path)

        office = shutil.which("soffice") or shutil.which("libreoffice")
        if _is_windows():
            try:
                self._render_with_word(docx_path, pdf_path)
                self.last_renderer = "Microsoft Word COM"
                _validate_pdf(pdf_path)
                return pdf_path
            except (MissingComDependency, ComImportFailure, WordActivationError) as exc:
                logger.warning(
                    "Word renderer unavailable at stage {} (code={}, COM registration={}); checking LibreOffice fallback.",
                    type(exc).__name__,
                    getattr(exc, "diagnostic_code", "not-applicable"),
                    _word_com_registration(),
                )
                if not office:
                    raise
            # Word conversion and PDF validation failures deliberately propagate;
            # they must not be hidden by a different renderer's output.

        if office:
            self._render_with_libreoffice(office, docx_path, pdf_path)
            self.last_renderer = "LibreOffice"
            _validate_pdf(pdf_path)
            return pdf_path

        if _is_windows():
            raise RendererUnavailable(
                "Microsoft Word could not be activated in this process and LibreOffice was not found. "
                "See the application log for the Word COM diagnostic. The DOCX remains available to save."
            )
        raise RendererUnavailable(
            "Microsoft Word COM is only available on Windows and LibreOffice was not found. "
            "The DOCX remains available to save."
        )

    @staticmethod
    def _render_with_word(docx_path: Path, pdf_path: Path) -> None:
        try:
            pythoncom = importlib.import_module("pythoncom")
        except ModuleNotFoundError as exc:
            if exc.name == "pythoncom":
                registration = _word_com_registration()
                raise MissingComDependency(
                    f"pywin32 is missing from the active Python environment; Word COM registration is {registration}."
                ) from exc
            raise ComImportFailure("pywin32 could not import its COM runtime module.") from exc
        try:
            win32_client = importlib.import_module("win32com.client")
        except ModuleNotFoundError as exc:
            if exc.name == "win32com" or (exc.name or "").startswith("win32com."):
                raise MissingComDependency(
                    "pywin32 is missing or incomplete in the application's active Python environment."
                ) from exc
            raise ComImportFailure("pywin32 COM client dependencies could not be imported.") from exc
        except Exception as exc:
            raise ComImportFailure("pywin32 COM client failed to import.") from exc

        initialized = False
        word = None
        document = None
        try:
            try:
                pythoncom.CoInitialize()
                initialized = True
            except Exception as exc:
                error = WordComInitializationError(
                    f"Windows COM could not initialize on the document worker ({_hresult(exc)})."
                )
                error.diagnostic_code = _hresult(exc)
                raise error from exc

            try:
                # DispatchEx owns a separate Word process; never attach to or quit
                # a Word instance started by the user.
                word = win32_client.DispatchEx("Word.Application")
                word.Visible = False
                word.DisplayAlerts = 0
            except Exception as exc:
                error = WordActivationError(
                    f"Windows has Word registered, but this process could not start its private Word instance ({_hresult(exc)})."
                )
                error.diagnostic_code = _hresult(exc)
                raise error from exc

            try:
                document = word.Documents.Open(
                    str(docx_path),
                    ReadOnly=True,
                    AddToRecentFiles=False,
                    Visible=False,
                )
                document.ExportAsFixedFormat(str(pdf_path), 17)
                if not pdf_path.is_file() or pdf_path.stat().st_size == 0:
                    raise RuntimeError("Word produced no PDF output.")
            except Exception as exc:
                error = WordConversionError(
                    f"Word started, but DOCX-to-PDF conversion failed ({_hresult(exc)})."
                )
                error.diagnostic_code = _hresult(exc)
                raise error from exc
        finally:
            try:
                if document is not None:
                    document.Close(False)
            except Exception as exc:
                logger.warning("Word document close failed ({}).", _hresult(exc))
            finally:
                # Release the document proxy before shutting down its private
                # application, so no COM reference is finalized after Quit().
                document = None
            try:
                if word is not None:
                    word.Quit(False)
            except Exception as exc:
                logger.warning("Private Word instance quit failed ({}).", _hresult(exc))
            finally:
                # Release the DispatchEx proxy on the COM-initialized worker.
                word = None
                if initialized:
                    try:
                        pythoncom.CoUninitialize()
                    except Exception as exc:
                        logger.warning("COM uninitialization failed ({}).", _hresult(exc))

    @staticmethod
    def _render_with_libreoffice(office: str, docx_path: Path, pdf_path: Path) -> None:
        profile = Path(tempfile.mkdtemp(prefix="tdp-lo-profile-"))
        try:
            subprocess.run(
                [office, "--headless", f"-env:UserInstallation={profile.as_uri()}", "--convert-to", "pdf", "--outdir", str(pdf_path.parent), str(docx_path)],
                check=True,
                timeout=120,
                capture_output=True,
                text=True,
            )
            generated = pdf_path.parent / f"{docx_path.stem}.pdf"
            if generated != pdf_path:
                generated.replace(pdf_path)
            if not pdf_path.is_file() or pdf_path.stat().st_size == 0:
                raise DocumentRenderError("LibreOffice produced no PDF output.")
        except subprocess.TimeoutExpired as exc:
            raise DocumentRenderError("LibreOffice conversion exceeded its 120-second limit.") from exc
        except subprocess.CalledProcessError as exc:
            # Do not log stdout/stderr: office can include document names or text.
            logger.error("LibreOffice conversion failed with exit code {}.", exc.returncode)
            raise DocumentRenderError(
                f"LibreOffice conversion failed with exit code {exc.returncode}."
            ) from exc
        finally:
            try:
                shutil.rmtree(profile)
            except OSError as exc:
                logger.warning("LibreOffice profile cleanup failed ({}).", type(exc).__name__)
