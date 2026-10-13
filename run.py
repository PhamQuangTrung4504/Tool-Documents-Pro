#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tool-Documents-Pro — OCR Desktop Application
=============================================
Entry point: Khởi chạy ứng dụng nhận dạng chữ từ ảnh.

Usage:
    python run.py
"""

import sys
import os
import warnings

# Ngăn chặn xung đột thư viện DLL OpenMP giữa PyTorch và PaddlePaddle trên Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Ngăn chặn PaddleX 3.x tự động kích hoạt eager initialization và kiểm tra model nguồn
os.environ["PADDLE_PDX_EAGER_INIT"] = "False"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

# Nạp trước PyTorch để Windows load c10.dll và OpenMP an toàn trước khi Paddle khởi tạo
try:
    import torch  # noqa: F401
except Exception:
    pass

# Lọc bỏ các cảnh báo phụ thuộc không ảnh hưởng runtime
warnings.filterwarnings("ignore", message=".*RequestsDependencyWarning.*")
warnings.filterwarnings("ignore", message=".*No ccache found.*")
warnings.filterwarnings("ignore", message=".*enable_nested_tensor.*")

# ---------------------------------------------------------------------------
# Đảm bảo thư mục gốc nằm trong sys.path
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# ---------------------------------------------------------------------------
# Cấu hình logging với loguru
# ---------------------------------------------------------------------------
from loguru import logger

LOG_DIR = os.path.join(ROOT_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logger.remove()  # Xoá handler mặc định
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    colorize=True,
)
logger.add(
    os.path.join(LOG_DIR, "app_{time:YYYY-MM-DD}.log"),
    level="DEBUG",
    rotation="10 MB",
    retention="7 days",
    encoding="utf-8",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} | {message}",
)

# ---------------------------------------------------------------------------
# Khởi chạy ứng dụng PyQt5
# ---------------------------------------------------------------------------


def main() -> int:
    """Hàm chính — tạo QApplication và hiển thị MainWindow."""
    logger.info("=" * 60)
    logger.info("Tool-Documents-Pro — Starting application …")
    logger.info("=" * 60)

    # Kiểm tra phiên bản Python
    py_ver = sys.version_info
    if py_ver < (3, 9) or py_ver >= (3, 12):
        logger.warning(
            f"Python {py_ver.major}.{py_ver.minor} detected. "
            "Recommended: Python 3.9 — 3.11."
        )

    # Import PyQt5
    try:
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtCore import Qt
        from PyQt5.QtGui import QFont
    except ImportError:
        logger.error(
            "PyQt5 chưa được cài đặt. Chạy: pip install -r requirements.txt"
        )
        return 1

    # Bật DPI scaling cho màn hình độ phân giải cao
    if hasattr(Qt, "AA_EnableHighDpiScaling"):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, "AA_UseHighDpiPixmaps"):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("Tool-Documents-Pro")
    app.setOrganizationName("ToolDocumentsPro")

    # Font mặc định dễ đọc
    default_font = QFont("Segoe UI", 10)
    app.setFont(default_font)

    # Import và hiển thị cửa sổ chính
    from app.ui.main_window import MainWindow

    window = MainWindow()
    window.show()

    logger.info("Application window opened.")
    exit_code = app.exec_()

    logger.info(f"Application exited with code {exit_code}.")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
