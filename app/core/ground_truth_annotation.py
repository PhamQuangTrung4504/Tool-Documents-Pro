"""Mô hình nhập, tuần tự hóa và lưu ground truth đã được người dùng đối chiếu."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import date
from pathlib import Path
from typing import Any, Dict, List


class VerifiedGroundTruthOverwriteError(ValueError):
    """Từ chối thay thế ground truth đã xác minh nếu chưa có đồng ý rõ ràng."""


def new_paragraph_block(text: str = "") -> Dict[str, Any]:
    return {"kind": "paragraph", "text": text, "reviewed": False, "bbox": None}


def new_table_block(rows: int = 1, columns: int = 1) -> Dict[str, Any]:
    return {
        "kind": "table",
        "category": "user_annotated",
        "grid": [[""] * max(1, columns) for _ in range(max(1, rows))],
        "cell_spans": [],
        "diagonal_cells": [],
        "cell_bboxes": {},
        "bbox": None,
        "reviewed": False,
    }


def serialize_reading_order(blocks: List[Dict[str, Any]]) -> str:
    """Tuần tự hóa theo thứ tự khối; bảng dùng thứ tự hàng trước, ô trái sang phải.

    Ô rỗng vẫn giữ dấu tab để không làm mất vị trí dữ liệu. Không chuẩn hóa,
    sửa chính tả hoặc tự thêm nội dung vào chuỗi người dùng đã nhập.
    """
    rendered = []
    for block in blocks:
        kind = block.get("kind")
        if kind == "paragraph":
            text = block.get("text", "")
            if not isinstance(text, str):
                raise ValueError("Nội dung đoạn phải là chuỗi")
            rendered.append(text)
        elif kind == "table":
            grid = block.get("grid")
            if not isinstance(grid, list) or not grid:
                raise ValueError("Bảng phải có ít nhất một hàng")
            width = len(grid[0]) if isinstance(grid[0], list) else 0
            if width == 0 or any(not isinstance(row, list) or len(row) != width for row in grid):
                raise ValueError("Lưới bảng phải là ma trận chữ nhật")
            if any(not isinstance(cell, str) for row in grid for cell in row):
                raise ValueError("Nội dung mỗi ô phải là chuỗi")
            rendered.append("\n".join("\t".join(row) for row in grid))
        else:
            raise ValueError(f"Loại khối không hỗ trợ: {kind!r}")
    return "\n".join(rendered)


def make_ground_truth(
    blocks: List[Dict[str, Any]],
    *,
    verified: bool,
    reviewed_by: str,
    reviewed_at: str,
    source_note: str,
) -> Dict[str, Any]:
    """Tạo schema tương thích benchmark; verified chỉ được bật từ thao tác tường minh."""
    if not blocks:
        raise ValueError("Cần ít nhất một khối đã đối chiếu")
    if verified and any(block.get("reviewed") is not True for block in blocks):
        raise ValueError("Không thể xác minh khi còn khối chưa được đánh dấu đã đối chiếu")
    if verified and not all(value.strip() for value in (reviewed_by, reviewed_at, source_note)):
        raise ValueError("Cần người kiểm tra, ngày kiểm tra và ghi chú nguồn trước khi xác minh")

    copied_blocks = json.loads(json.dumps(blocks, ensure_ascii=False))
    table_blocks = [block for block in copied_blocks if block["kind"] == "table"]
    if len(table_blocks) > 1:
        raise ValueError("Benchmark hiện nhận một bảng ground truth cho mỗi tài liệu")
    table = None
    if table_blocks:
        source = table_blocks[0]
        table = {
            "category": source.get("category", "user_annotated"),
            "expected_grid": source["grid"],
            "expected_non_table_lines": [
                line for block in copied_blocks if block["kind"] == "paragraph"
                for line in block.get("text", "").splitlines()
            ],
            "expected_cell_spans": source.get("cell_spans", []),
            "expected_diagonal_cells": source.get("diagonal_cells", []),
            "metadata": {"reading_order": "row_major", "cell_separator": "tab"},
        }
    return {
        "text": serialize_reading_order(copied_blocks),
        "blocks": copied_blocks,
        "verified": bool(verified),
        "reviewed_by": reviewed_by.strip() if verified else "",
        "reviewed_at": reviewed_at.strip() if verified else "",
        "source_note": source_note.strip() if verified else "",
        "table": table,
        "formatting_annotation": None,
    }


def verify_ground_truth(
    blocks: List[Dict[str, Any]], reviewed_by: str, reviewed_at: str, source_note: str
) -> Dict[str, Any]:
    """Tạo Ground Truth xác minh theo yêu cầu gọi tường minh của người dùng."""
    return make_ground_truth(
        blocks, verified=True, reviewed_by=reviewed_by,
        reviewed_at=reviewed_at or date.today().isoformat(), source_note=source_note,
    )


def manifest_has_verified_ground_truth(payload: Dict[str, Any]) -> bool:
    return any(
        isinstance(item, dict)
        and isinstance(item.get("ground_truth"), dict)
        and item["ground_truth"].get("verified") is True
        for item in payload.get("documents", [])
    )


def save_manifest_safely(path: Path, payload: Dict[str, Any], *, allow_replace_verified: bool = False) -> None:
    """Ghi nguyên tử; không thay tệp đang có ground truth xác minh nếu chưa được phép."""
    target = Path(path).expanduser().resolve()
    if target.exists() and not allow_replace_verified:
        try:
            current = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            current = {}
        if manifest_has_verified_ground_truth(current):
            raise VerifiedGroundTruthOverwriteError(
                "Tệp đích chứa ground truth đã xác minh; cần xác nhận rõ ràng trước khi ghi đè"
            )
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=target.parent,
            prefix=f".{target.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temp_path = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temp_path, target)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
