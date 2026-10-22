"""Chỉnh sửa OCR có cấu trúc, có thể hoàn tác và giữ nguyên dữ liệu nguồn."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Optional

from app.core.models import DocumentBlock, OCRResult, OCRTable, TableCell


class EditOwnershipError(ValueError):
    """Không thể xác định duy nhất paragraph hoặc cell sở hữu OCRBox."""


@dataclass(frozen=True)
class EditTarget:
    """Định danh ổn định trong OCRResult, không phụ thuộc id() của Python."""

    kind: str
    box_index: int
    block_index: int
    table_index: Optional[int] = None
    row_start: Optional[int] = None
    col_start: Optional[int] = None
    header_side: Optional[str] = None
    header_line_index: Optional[int] = None


@dataclass(frozen=True)
class CorrectionRecord:
    target: EditTarget
    old_box_text: str
    new_box_text: str
    old_region_text: str
    new_region_text: str
    source: str = "manual"
    candidate_index: Optional[int] = None
    candidate_recognizer: Optional[str] = None


class StructuredDocumentEditor:
    """Áp dụng các sửa đổi đã xác nhận vào một bản sao của OCRResult.

    ``raw_text`` và metadata nhận dạng không bị sửa. Các correction records
    giữ định danh box/block/cell có thể truy vết trong kết quả OCR ban đầu.
    """

    _TEXT_BLOCK_KINDS = {"paragraph", "heading", "caption", "parallel_header"}

    def __init__(self, result: OCRResult) -> None:
        self._original = deepcopy(result)
        self._result = deepcopy(result)
        self._undo: list[CorrectionRecord] = []
        self._redo: list[CorrectionRecord] = []

    @property
    def original_result(self) -> OCRResult:
        return deepcopy(self._original)

    @property
    def result(self) -> OCRResult:
        """Trả về bản sao để worker xuất Word không sửa state của editor."""
        return deepcopy(self._result)

    @property
    def corrections(self) -> tuple[CorrectionRecord, ...]:
        return tuple(self._undo)

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    @property
    def dirty(self) -> bool:
        return bool(self._undo)

    def apply_box_correction(self, box_index: int, new_text: str, source: str = "manual",
                             candidate_index: Optional[int] = None,
                             candidate_recognizer: Optional[str] = None) -> CorrectionRecord:
        if not 0 <= box_index < len(self._result.boxes):
            raise IndexError("OCRBox index is outside the OCR result")
        new_text = str(new_text)
        box = self._result.boxes[box_index]
        target = self._resolve_target(box_index)
        old_region = self._get_region_text(target)
        old_box_text = box.text
        if old_box_text == new_text:
            raise ValueError("The confirmed text is unchanged")
        new_region = self._replace_box_text(old_region, old_box_text, new_text, target)
        record = CorrectionRecord(target, old_box_text, new_text, old_region, new_region,
                                  source, candidate_index, candidate_recognizer)
        self._apply_record(record, forward=True)
        self._undo.append(record)
        self._redo.clear()
        self._sync_full_text()
        self._sync_audit_trail()
        return record

    def apply_candidate(self, box_index: int, candidate_index: int) -> CorrectionRecord:
        """Apply one explicitly chosen candidate; caller action is confirmation."""
        if not 0 <= box_index < len(self._result.boxes):
            raise IndexError("OCRBox index is outside the OCR result")
        candidates = self._result.boxes[box_index].metadata.get("candidates", [])
        if not 0 <= candidate_index < len(candidates):
            raise IndexError("OCR candidate index is outside the candidate list")
        candidate = candidates[candidate_index]
        if not isinstance(candidate, dict) or not isinstance(candidate.get("text"), str):
            raise ValueError("OCR candidate does not contain usable text")
        if candidate_index > 0:
            mapping = candidate.get("mapping", self._result.boxes[box_index].metadata.get("alternate_mapping", {}))
            if not isinstance(mapping, dict) or mapping.get("status") != "matched":
                raise ValueError("Alternate candidate has no verified region mapping")
        return self.apply_box_correction(
            box_index, candidate["text"], source="confirmed_candidate",
            candidate_index=candidate_index,
            candidate_recognizer=candidate.get("recognizer"),
        )

    def undo(self) -> Optional[CorrectionRecord]:
        if not self._undo:
            return None
        record = self._undo.pop()
        self._apply_record(record, forward=False)
        self._redo.append(record)
        self._sync_full_text()
        self._sync_audit_trail()
        return record

    def redo(self) -> Optional[CorrectionRecord]:
        if not self._redo:
            return None
        record = self._redo.pop()
        self._apply_record(record, forward=True)
        self._undo.append(record)
        self._sync_full_text()
        self._sync_audit_trail()
        return record

    def _resolve_target(self, box_index: int) -> EditTarget:
        owners: list[EditTarget] = []
        for block_index, block in enumerate(self._result.document_blocks):
            if block.kind not in self._TEXT_BLOCK_KINDS or box_index not in block.box_indices:
                continue
            if block.kind == "parallel_header":
                header = self._result.parallel_header
                if header is None:
                    raise EditOwnershipError("Parallel header data is unavailable")
                left = any(source is self._result.boxes[box_index] for source in header.left_boxes)
                right = any(source is self._result.boxes[box_index] for source in header.right_boxes)
                if left == right:
                    raise EditOwnershipError("Header side ownership is ambiguous")
                side = "left" if left else "right"
                lines = header.left_lines if left else header.right_lines
                matching_lines = [i for i, line in enumerate(lines)
                                  if line.count(self._result.boxes[box_index].text) == 1]
                if len(matching_lines) != 1:
                    raise EditOwnershipError("Header line ownership is ambiguous")
                owners.append(EditTarget("header", box_index, block_index,
                                         header_side=side, header_line_index=matching_lines[0]))
            else:
                owners.append(EditTarget("block", box_index, block_index))

        for table_index, table in enumerate(self._result.tables):
            for cell in table.cells:
                matching = self._cell_box_indexes(cell)
                if box_index in matching:
                    # A repeated link to one physical cell is still one owner.
                    owners.append(EditTarget("cell", box_index, self._table_block_index(table_index),
                                             table_index, cell.row_start, cell.col_start))

        # De-duplicate structurally identical links but reject distinct owners.
        unique = list(dict.fromkeys(owners))
        if len(unique) != 1:
            label = "unassigned" if not unique else "ambiguous"
            raise EditOwnershipError(f"OCRBox ownership is {label}; no edit was applied")
        return unique[0]

    def _cell_box_indexes(self, cell: TableCell) -> set[int]:
        found: set[int] = set()
        for source in cell.boxes:
            matches = [index for index, box in enumerate(self._result.boxes) if source is box]
            if len(matches) == 1:
                found.add(matches[0])
        return found

    def _table_block_index(self, table_index: int) -> int:
        matches = [index for index, block in enumerate(self._result.document_blocks)
                   if block.kind == "table" and block.table_index == table_index]
        if len(matches) != 1:
            raise EditOwnershipError("Table block ownership is missing or duplicated")
        return matches[0]

    def _get_region_text(self, target: EditTarget) -> str:
        if target.kind == "block":
            return self._result.document_blocks[target.block_index].text
        if target.kind == "header":
            header = self._result.parallel_header
            lines = header.left_lines if target.header_side == "left" else header.right_lines
            return lines[target.header_line_index]
        cell = self._get_cell(target)
        return cell.text if cell.text else self._result.tables[target.table_index].grid[target.row_start][target.col_start]

    def _replace_box_text(self, region: str, old: str, new: str, target: EditTarget) -> str:
        positions: list[int] = []
        if old:
            start = 0
            while (found := region.find(old, start)) >= 0:
                left_ok = not old[0].isalnum() or found == 0 or not region[found - 1].isalnum()
                end = found + len(old)
                right_ok = not old[-1].isalnum() or end == len(region) or not region[end].isalnum()
                if left_ok and right_ok:
                    positions.append(found)
                start = found + 1
        if len(positions) == 1:
            position = positions[0]
            return region[:position] + new + region[position + len(old):]
        if region == old and len(self._owner_box_indexes(target)) == 1:
            return new
        raise EditOwnershipError("OCRBox text does not identify one unique span in its owner")

    def _owner_box_indexes(self, target: EditTarget) -> set[int]:
        if target.kind == "block":
            return set(self._result.document_blocks[target.block_index].box_indices)
        if target.kind == "header":
            header = self._result.parallel_header
            sources = header.left_boxes if target.header_side == "left" else header.right_boxes
            return {index for index, box in enumerate(self._result.boxes)
                    if any(source is box for source in sources)}
        return self._cell_box_indexes(self._get_cell(target))

    def _get_cell(self, target: EditTarget) -> TableCell:
        table = self._result.tables[target.table_index]
        matches = [cell for cell in table.cells
                   if cell.row_start == target.row_start and cell.col_start == target.col_start]
        if len(matches) != 1:
            raise EditOwnershipError("Table cell identity is missing or ambiguous")
        return matches[0]

    def _apply_record(self, record: CorrectionRecord, forward: bool) -> None:
        target = record.target
        region_text = record.new_region_text if forward else record.old_region_text
        box_text = record.new_box_text if forward else record.old_box_text
        if target.kind == "block":
            self._result.document_blocks[target.block_index].text = region_text
        elif target.kind == "header":
            header = self._result.parallel_header
            lines = header.left_lines if target.header_side == "left" else header.right_lines
            updated_lines = list(lines)
            updated_lines[target.header_line_index] = region_text
            if target.header_side == "left":
                header.left_lines = updated_lines
            else:
                header.right_lines = updated_lines
            block = self._result.document_blocks[target.block_index]
            block.text = "\n".join(header.left_lines + header.right_lines)
        else:
            cell = self._get_cell(target)
            cell.text = region_text
            self._result.tables[target.table_index].grid[target.row_start][target.col_start] = region_text
        self._result.boxes[target.box_index].text = box_text

    def _sync_full_text(self) -> None:
        parts: list[str] = []
        for block in sorted(self._result.document_blocks, key=lambda item: item.order):
            if block.kind == "table":
                if block.table_index is None or not 0 <= block.table_index < len(self._result.tables):
                    raise EditOwnershipError("Document block references an invalid table")
                parts.extend(" | ".join(row) for row in self._result.tables[block.table_index].grid)
            elif block.kind == "parallel_header":
                parts.extend(block.text.splitlines())
            else:
                parts.append(block.text)
        self._result.full_text = "\n".join(parts)

    def _sync_audit_trail(self) -> None:
        self._result.audit_trail = deepcopy(self._original.audit_trail)
        self._result.audit_trail.extend({
            "kind": "user_correction",
            "target": asdict(record.target),
            "old_text": record.old_box_text,
            "new_text": record.new_box_text,
            "source": record.source,
            "candidate_index": record.candidate_index,
            "candidate_recognizer": record.candidate_recognizer,
        } for record in self._undo)


__all__ = ["CorrectionRecord", "EditOwnershipError", "EditTarget", "StructuredDocumentEditor"]
