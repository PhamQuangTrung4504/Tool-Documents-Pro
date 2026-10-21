"""Owned temporary document sessions for generated OCR documents."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path


class DocumentSession:
    """A unique, app-owned directory containing one generated DOCX and preview PDF."""

    _MARKER = ".tool-documents-pro-session.json"

    def __init__(self, root: Path, session_id: str) -> None:
        self.root = root.resolve()
        self.session_id = session_id
        self.directory = (self.root / session_id).resolve()
        self.docx_path = self.directory / "document.docx"
        self.pdf_path = self.directory / "preview.pdf"

    @classmethod
    def create(cls, base_dir: Path | None = None) -> "DocumentSession":
        if base_dir is None:
            local_app_data = os.environ.get("LOCALAPPDATA")
            base_dir = Path(local_app_data) / "Tool-Documents-Pro" / "temp" if local_app_data else Path(tempfile.gettempdir()) / "Tool-Documents-Pro" / "temp"
        root = Path(base_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        session_id = uuid.uuid4().hex
        session = cls(root, session_id)
        session.directory.mkdir()
        marker = {"application": "Tool-Documents-Pro", "session_id": session_id}
        (session.directory / cls._MARKER).write_text(json.dumps(marker), encoding="utf-8")
        return session

    def owns(self, path: Path | None = None) -> bool:
        target = (path or self.directory).resolve()
        try:
            target.relative_to(self.directory)
        except ValueError:
            return False
        marker = self.directory / self._MARKER
        try:
            data = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return data.get("application") == "Tool-Documents-Pro" and data.get("session_id") == self.session_id

    def cleanup(self) -> bool:
        if not self.owns():
            return False
        try:
            shutil.rmtree(self.directory)
            return True
        except OSError:
            return False
