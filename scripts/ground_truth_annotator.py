#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Chạy giao diện độc lập để đối chiếu ảnh và nhập ground truth thủ công."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="Manifest benchmark cần đối chiếu")
    args = parser.parse_args()
    try:
        from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox
        from app.ui.ground_truth_annotator import GroundTruthAnnotator

        app = QApplication.instance() or QApplication(sys.argv)
        manifest_path = args.manifest
        if manifest_path is None:
            manifest_path_text, _ = QFileDialog.getOpenFileName(
                None, "Mở manifest benchmark", "", "JSON (*.json)"
            )
            if not manifest_path_text:
                return 0
            manifest_path = Path(manifest_path_text)
        window = GroundTruthAnnotator(manifest_path)
        window.show()
        return app.exec_()
    except Exception as exc:
        try:
            QMessageBox.critical(None, "Không thể mở công cụ annotation", str(exc))
        except Exception:
            print(f"Không thể mở công cụ annotation: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
