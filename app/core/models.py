#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
app.core.models
===============
Cấu trúc dữ liệu và Enum đại diện cho kết quả nhận dạng OCR, bảng biểu,
tiêu đề hành chính 2 khối song song và quyết định khôi phục ký tự.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "OCRBox",
    "TableCell",
    "OCRTable",
    "ParallelHeader",
    "HeaderDetectionResult",
    "LayoutResult",
    "OCRResult",
    "SymbolDecision",
]

# ---------------------------------------------------------------------------
# Kiểu dữ liệu kết quả OCR
# ---------------------------------------------------------------------------


@dataclass
class OCRBox:
    """Một vùng văn bản đã được nhận dạng."""

    text: str
    confidence: float
    bbox: List[List[int]]  # 4 điểm [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    raw_text: str = ""     # Lưu giữ văn bản nhận dạng ban đầu của OCR engine

    def __post_init__(self):
        if not self.raw_text:
            self.raw_text = self.text

    @property
    def top_left(self) -> Tuple[int, int]:
        return (int(self.bbox[0][0]), int(self.bbox[0][1]))

    @property
    def bottom_right(self) -> Tuple[int, int]:
        return (int(self.bbox[2][0]), int(self.bbox[2][1]))

    def __repr__(self) -> str:
        return (
            f"OCRBox(text={self.text!r}, "
            f"conf={self.confidence:.2f}, "
            f"pos={self.top_left})"
        )


@dataclass
class TableCell:
    """Đại diện cho 1 ô vật lý trong bảng, hỗ trợ ô gộp 2D (rowspan & colspan)."""

    row_start: int
    col_start: int
    row_end: int      # exclusive (row_span = row_end - row_start)
    col_end: int      # exclusive (col_span = col_end - col_start)
    text: str = ""
    bbox: Optional[List[int]] = None  # [x1, y1, x2, y2] của khung ô vật lý
    boxes: List[OCRBox] = field(default_factory=list)

    @property
    def row_span(self) -> int:
        return max(1, self.row_end - self.row_start)

    @property
    def col_span(self) -> int:
        return max(1, self.col_end - self.col_start)

    @property
    def is_merged(self) -> bool:
        return self.row_span > 1 or self.col_span > 1


@dataclass
class OCRTable:
    """Bảng biểu đã được tái dựng cấu trúc hàng và cột."""

    grid: List[List[str]] = field(default_factory=list)  # Ma trận các ô: [hàng][cột]
    bbox: List[int] = field(default_factory=list)        # [left, top, right, bottom]
    num_rows: int = 0
    num_cols: int = 0
    row_boxes: List[List[OCRBox]] = field(default_factory=list)  # Danh sách box cho từng hàng
    cells: List[TableCell] = field(default_factory=list)         # Danh sách ô vật lý (kèm rowspan/colspan)


@dataclass
class ParallelHeader:
    """Cấu trúc dữ liệu đại diện cho Tiêu đề hành chính 2 khối song song."""

    left_lines: List[str] = field(default_factory=list)   # Cơ quan ban hành, số hiệu văn bản
    right_lines: List[str] = field(default_factory=list)  # Quốc hiệu, tiêu ngữ, địa danh, ngày tháng
    left_boxes: List[OCRBox] = field(default_factory=list)
    right_boxes: List[OCRBox] = field(default_factory=list)


class HeaderDetectionResult(Tuple[List[List[OCRBox]], List[dict]]):
    """
    Kết quả phát hiện header 2 khối song song.
    Kế thừa tuple (grouped_lines, remaining) để tương thích 100% với code giải nén 2 phần tử:
    header_lines, rem = detect_parallel_administrative_header(...)
    đồng thời lưu thuộc tính parallel_header.
    """
    parallel_header: Optional[ParallelHeader]

    def __new__(
        cls,
        grouped_lines: List[List[OCRBox]],
        remaining: List[dict],
        parallel_header: Optional[ParallelHeader] = None,
    ) -> HeaderDetectionResult:
        inst = super().__new__(cls, (grouped_lines, remaining))
        inst.parallel_header = parallel_header
        return inst


class LayoutResult(Tuple[List[List[OCRBox]], List[OCRTable]]):
    """
    Kết quả tái dựng bố cục tài liệu.
    Kế thừa tuple (grouped_lines, tables) để tương thích 100% với code giải nén 2 phần tử:
    grouped_lines, tables = reconstruct_document_layout(...)
    đồng thời lưu thuộc tính parallel_header.
    """
    parallel_header: Optional[ParallelHeader]

    def __new__(
        cls,
        grouped_lines: List[List[OCRBox]],
        tables: List[OCRTable],
        parallel_header: Optional[ParallelHeader] = None,
    ) -> LayoutResult:
        inst = super().__new__(cls, (grouped_lines, tables))
        inst.parallel_header = parallel_header
        return inst

    @property
    def grouped_lines(self) -> List[List[OCRBox]]:
        return self[0]

    @property
    def tables(self) -> List[OCRTable]:
        return self[1]


@dataclass
class OCRResult:
    """Kết quả tổng hợp sau khi chạy OCR trên một ảnh."""

    boxes: List[OCRBox] = field(default_factory=list)
    full_text: str = ""
    language: str = "vi"
    elapsed_seconds: float = 0.0
    image_width: int = 0
    image_height: int = 0
    tables: List[OCRTable] = field(default_factory=list)
    raw_full_text: str = ""                             # Văn bản gốc ban đầu trước khi khôi phục ký tự
    audit_trail: List[dict] = field(default_factory=list)  # Nhật ký vết các phép sửa tự động
    parallel_header: Optional[ParallelHeader] = None      # Tiêu đề hành chính 2 khối song song

    @property
    def text(self) -> str:
        return self.full_text

    @property
    def lines(self) -> List[str]:
        if self.full_text:
            return self.full_text.splitlines()
        return [b.text for b in self.boxes] if self.boxes else []

    @property
    def num_lines(self) -> int:
        if self.full_text:
            return len(self.full_text.splitlines())
        return len(self.boxes)

    @property
    def average_confidence(self) -> float:
        if not self.boxes:
            return 0.0
        return sum(b.confidence for b in self.boxes) / len(self.boxes)




class SymbolDecision(str, Enum):
    """Quyết định 3 trạng thái cho phục hồi ký tự OCR."""
    AUTO_CORRECT = "AUTO_CORRECT"
    FLAG_UNCERTAIN = "FLAG_UNCERTAIN"
    PRESERVE = "PRESERVE"


