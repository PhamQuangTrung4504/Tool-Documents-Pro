"""Khởi tạo các runtime native theo thứ tự an toàn cho OCR.

Module này cố ý không nhập Paddle/PaddleX. Các entrypoint gọi bootstrap trước
khi nạp engine OCR; nếu Torch lỗi, lỗi được truyền nguyên vẹn để không che lỗi DLL.
"""
from __future__ import annotations

import importlib
import sys
from typing import Any

_TORCH_BOOTSTRAPPED = False


def bootstrap_torch_before_paddle() -> Any:
    """Nạp Torch trước Paddle trong từng tiến trình (kể cả tiến trình spawn)."""
    global _TORCH_BOOTSTRAPPED
    if _TORCH_BOOTSTRAPPED:
        return sys.modules["torch"]
    if "paddle" in sys.modules:
        raise RuntimeError(
            "Paddle đã được nạp trước khi bootstrap xác nhận thứ tự Torch; "
            "không thể xác minh thứ tự nạp DLL an toàn."
        )
    torch_module = importlib.import_module("torch")
    _TORCH_BOOTSTRAPPED = True
    return torch_module
