"""Opt-in, source-linked diagnostics for OCR document reconstruction.

Diagnostics can contain recognized document text. Nothing in this module writes
or logs that data unless the caller explicitly requests a report destination.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median
from typing import Any, Optional

import cv2
import numpy as np

from app.core.models import DocumentBlock, OCRBox, OCRResult

__all__ = [
    "build_reconstruction_diagnostic",
    "write_reconstruction_diagnostic",
    "infer_visual_document_styles",
]


def _box_rect(box: OCRBox) -> list[int]:
    xs = [int(point[0]) for point in box.bbox]
    ys = [int(point[1]) for point in box.bbox]
    return [min(xs), min(ys), max(xs), max(ys)]


def _table_box_ownership(result: OCRResult, table_index: int) -> dict[int, list[dict[str, int]]]:
    table = result.tables[table_index]
    indexes = {id(box): index for index, box in enumerate(result.boxes)}
    assigned: dict[int, list[dict[str, int]]] = {}
    for cell in table.cells:
        for box in cell.boxes:
            index = indexes.get(id(box))
            if index is not None:
                assigned.setdefault(index, []).append({
                    "row_start": cell.row_start,
                    "col_start": cell.col_start,
                    "row_end": cell.row_end,
                    "col_end": cell.col_end,
                })
    for row_index, row in enumerate(table.row_boxes):
        for box in row:
            index = indexes.get(id(box))
            if index is not None and index not in assigned:
                assigned.setdefault(index, []).append({"row_start": row_index, "col_start": -1})
    return assigned


def build_reconstruction_diagnostic(
    result: OCRResult,
    source_boxes: Optional[list[OCRBox]] = None,
) -> dict[str, Any]:
    """Build a detailed, non-logging report of box ownership and output blocks.

    ``source_boxes`` can include boxes present immediately before layout
    reconstruction, allowing the report to identify boxes not carried into the
    final OCRResult. Text is included because this is an explicit diagnostic.
    """
    boxes = source_boxes if source_boxes is not None else result.boxes
    result_indexes = {id(box): index for index, box in enumerate(result.boxes)}
    owners: dict[int, list[str]] = {index: [] for index in range(len(boxes))}
    result_to_source = {id(box): index for index, box in enumerate(boxes)}

    for block_index, block in enumerate(result.document_blocks):
        if block.kind == "table" and block.table_index is not None and 0 <= block.table_index < len(result.tables):
            for result_box_index in _table_box_ownership(result, block.table_index):
                box = result.boxes[result_box_index]
                source_index = result_to_source.get(id(box))
                if source_index is not None:
                    owners[source_index].append(f"table:{block.table_index}")
        for box_index in ([] if block.kind == "table" else block.box_indices):
            if 0 <= box_index < len(result.boxes):
                source_index = result_to_source.get(id(result.boxes[box_index]))
                if source_index is not None:
                    owners[source_index].append(f"block:{block_index}:{block.kind}")

    # Legacy blocks can omit box_indices; infer only direct paragraph geometry
    # membership by object identity when available, never by text equality.
    unassigned = [index for index, regions in owners.items() if not regions]
    multi_assigned = {index: regions for index, regions in owners.items() if len(set(regions)) > 1}
    repeated_owner_entries = {
        index: regions for index, regions in owners.items() if len(regions) != len(set(regions))
    }

    recognition_entries = []
    for index, box in enumerate(boxes):
        recognition_entries.append({
            "box_index": index,
            "result_box_index": result_indexes.get(id(box)),
            "bbox": _box_rect(box),
            "raw_text": box.raw_text,
            "text": box.text,
            "recognition": box.metadata,
            "owners": owners[index],
        })

    tables = []
    table_boxes_outside_bbox: dict[int, list[int]] = {}
    for table_index, table in enumerate(result.tables):
        cell_ownership = _table_box_ownership(result, table_index)
        outside_bbox = []
        if len(table.bbox) >= 4:
            left, top, right, bottom = table.bbox[:4]
            for box_index in cell_ownership:
                x1, y1, x2, y2 = _box_rect(result.boxes[box_index])
                center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                if not (left <= center_x <= right and top <= center_y <= bottom):
                    outside_bbox.append(box_index)
        if outside_bbox:
            table_boxes_outside_bbox[table_index] = outside_bbox
        tables.append({
            "table_index": table_index,
            "bbox": list(table.bbox),
            "rows": table.num_rows,
            "columns": table.num_cols,
            "grid": [list(row) for row in table.grid],
            "cells": [
                {
                    "row_start": cell.row_start,
                    "col_start": cell.col_start,
                    "row_end": cell.row_end,
                    "col_end": cell.col_end,
                    "bbox": list(cell.bbox) if cell.bbox else None,
                    "text": cell.text,
                    "box_indices": [
                        result_indexes[id(box)] for box in cell.boxes if id(box) in result_indexes
                    ],
                    "diagonal_border": cell.diagonal_border,
                    "border_edges": cell.border_edges,
                }
                for cell in table.cells
            ],
            "row_box_indices": [
                [result_indexes[id(box)] for box in row if id(box) in result_indexes]
                for row in table.row_boxes
            ],
            "owned_box_indices": sorted(cell_ownership),
            "owned_boxes_outside_bbox": outside_bbox,
        })

    blocks = [
        {
            "block_index": index,
            "kind": block.kind,
            "order": block.order,
            "bbox": block.bbox,
            "box_indices": list(block.box_indices),
            "table_index": block.table_index,
            "raw_text": block.raw_text,
            "text": block.text,
            "metadata": block.metadata,
        }
        for index, block in enumerate(result.document_blocks)
    ]
    paragraph_output = [block.text for block in result.document_blocks if block.kind in {"paragraph", "heading", "caption"}]
    return {
        "schema_version": 1,
        "recognized_boxes": recognition_entries,
        "tables": tables,
        "document_blocks": blocks,
        "output": {
            "paragraphs": paragraph_output,
            "tables": [table["grid"] for table in tables],
            "raw_full_text": result.raw_full_text,
            "full_text": result.full_text,
        },
        "integrity": {
            "unassigned_box_indices": unassigned,
            "multiply_assigned_box_indices": multi_assigned,
            "duplicate_owner_entries": repeated_owner_entries,
            "table_owned_boxes_outside_bbox": table_boxes_outside_bbox,
            "source_box_count": len(boxes),
            "result_box_count": len(result.boxes),
            "recognized_source_boxes_missing_from_result": [
                index for index, box in enumerate(boxes) if id(box) not in result_indexes
            ],
        },
    }


def write_reconstruction_diagnostic(
    result: OCRResult,
    destination: str | Path,
    source_boxes: Optional[list[OCRBox]] = None,
) -> Path:
    """Write an explicit diagnostic JSON to a caller-selected path."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(build_reconstruction_diagnostic(result, source_boxes), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def _stroke_width(gray_crop: np.ndarray) -> Optional[float]:
    if gray_crop.size == 0 or min(gray_crop.shape[:2]) < 5:
        return None
    # OCR polygons can include table rules. Exclude a narrow perimeter before
    # measuring glyph strokes; this also reduces clipping-edge influence.
    inset_y = max(1, int(round(gray_crop.shape[0] * 0.04)))
    inset_x = max(1, int(round(gray_crop.shape[1] * 0.015)))
    if gray_crop.shape[0] > 2 * inset_y and gray_crop.shape[1] > 2 * inset_x:
        gray_crop = gray_crop[inset_y:-inset_y, inset_x:-inset_x]
    _, ink = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    if cv2.countNonZero(ink) < 4:
        return None
    distance = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
    local_max = cv2.dilate(distance, np.ones((3, 3), dtype=np.uint8))
    peaks = distance[(distance > 0) & (distance >= local_max - 1e-4)]
    return float(2.0 * np.median(peaks)) if peaks.size else None


def _stroke_measurement_rect(rect: list[int], width: int, height: int) -> list[int]:
    left, top, right, bottom = rect
    left, top = max(0, left), max(0, top)
    right, bottom = min(width, right), min(height, bottom)
    crop_width, crop_height = max(0, right-left), max(0, bottom-top)
    inset_y = max(1, int(round(crop_height * 0.04)))
    inset_x = max(1, int(round(crop_width * 0.015)))
    if crop_height > 2 * inset_y and crop_width > 2 * inset_x:
        return [left + inset_x, top + inset_y, right - inset_x, bottom - inset_y]
    return [left, top, right, bottom]


def infer_visual_document_styles(
    blocks: list[DocumentBlock],
    boxes: list[OCRBox],
    source_image: np.ndarray,
    diagnostics: Optional[list[dict[str, Any]]] = None,
) -> list[DocumentBlock]:
    """Attach conservative, relative visual style evidence to OCR blocks.

    Only bold and alignment are inferred. Italic is deliberately left unknown
    until a reliable image feature is available; exporters can still honor an
    explicit ``metadata['format']['italic']`` supplied by a verified source.
    """
    if source_image is None or source_image.size == 0 or not boxes:
        return blocks
    gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY) if source_image.ndim == 3 else source_image
    height, width = gray.shape[:2]
    rects = [_box_rect(box) for box in boxes]
    table_box_indices = {
        index for block in blocks if block.kind == "table" for index in block.box_indices
    }
    stroke_widths: list[Optional[float]] = []
    box_heights: list[int] = []
    for left, top, right, bottom in rects:
        left, top = max(0, left), max(0, top)
        right, bottom = min(width, right), min(height, bottom)
        box_heights.append(max(1, bottom - top))
        stroke_widths.append(_stroke_width(gray[top:bottom, left:right]))
    valid_heights = [value for value in box_heights if value > 0]
    median_height = float(median(valid_heights)) if valid_heights else 0.0

    for block_index, block in enumerate(blocks):
        if block.kind not in {"paragraph", "heading", "caption", "parallel_header"}:
            continue
        kind_before_style = block.kind
        if not isinstance(block.metadata, dict):
            block.metadata = {}
        valid_indices = [index for index in block.box_indices if 0 <= index < len(boxes)]
        if not valid_indices:
            continue
        xs = [coordinate for index in valid_indices for coordinate in (rects[index][0], rects[index][2])]
        ys = [coordinate for index in valid_indices for coordinate in (rects[index][1], rects[index][3])]
        left, right, top, bottom = min(xs), max(xs), min(ys), max(ys)
        center = (left + right) / 2.0
        fmt: dict[str, Any] = {}
        tolerance = max(3.0, width * 0.012)
        peers = [rect for rect_index, rect in enumerate(rects)
                 if rect_index not in valid_indices and rect_index not in table_box_indices]
        left_alignment_support = sum(abs(left - rect[0]) <= tolerance for rect in peers)
        right_alignment_support = sum(abs(right - rect[2]) <= tolerance for rect in peers)
        if right_alignment_support >= 2 and right_alignment_support > left_alignment_support:
            fmt["alignment"] = "right"
        elif left_alignment_support >= 2:
            fmt["alignment"] = "left"
        elif abs(center - width / 2.0) <= width * 0.035:
            fmt["alignment"] = "center"
        # No positional evidence means unknown; left alignment is a Word
        # default, not a visual inference.
        ratios = []
        peer_counts = []
        comparable_peer_counts = []
        comparison_records: list[dict[str, Any]] = []
        for index in valid_indices:
            current = stroke_widths[index]
            if current is None:
                continue
            if diagnostics is None:
                peers = [
                    stroke_widths[peer]
                    for peer in range(len(boxes))
                    if peer != index
                    and peer not in table_box_indices
                    and 0.75 <= box_heights[peer] / max(1, box_heights[index]) <= 1.33
                    and stroke_widths[peer] is not None
                ]
                peer_indices: list[int] = []
            else:
                peer_indices = [
                    peer
                    for peer in range(len(boxes))
                    if peer != index
                    and peer not in table_box_indices
                    and 0.75 <= box_heights[peer] / max(1, box_heights[index]) <= 1.33
                    and stroke_widths[peer] is not None
                ]
                peers = [stroke_widths[peer] for peer in peer_indices]
            comparable_peer_counts.append(len(peers))
            if diagnostics is not None:
                peer_median = float(median(peers)) if peers else None
                comparison_records.append({
                    "box_index": index,
                    "bbox": rects[index],
                    "measurement_bbox": _stroke_measurement_rect(rects[index], width, height),
                    "stroke_width": current,
                    "peer_box_indices": peer_indices,
                    "peers": [
                        {"box_index": peer, "bbox": rects[peer],
                         "measurement_bbox": _stroke_measurement_rect(rects[peer], width, height),
                         "stroke_width": stroke_widths[peer], "height": box_heights[peer]}
                        for peer in peer_indices
                    ],
                    "peer_stroke_widths": peers,
                    "peer_median_stroke_width": peer_median,
                    "relative_stroke_width": current / peer_median if peer_median else None,
                })
            if len(peers) >= 3:
                ratios.append(current / max(float(median(peers)), 1e-6))
                peer_counts.append(len(peers))
        fmt["bold_evidence"] = {
            "status": "unknown",
            "comparison_boxes": max(comparable_peer_counts, default=0),
        }
        if ratios:
            bold_ratio = float(median(ratios))
            fmt["bold_evidence"] = {
                "relative_stroke_width": round(bold_ratio, 3),
                "comparison_boxes": min(peer_counts),
                "status": "supported" if bold_ratio >= 1.3 and min(peer_counts) >= 3 else "unknown",
            }
            if bold_ratio >= 1.3 and min(peer_counts) >= 3:
                fmt["bold"] = True
                block.metadata["outline_level"] = 1 if (bottom - top) >= 1.15 * max(median_height, 1.0) else 2
                if block.kind == "paragraph" and (
                    fmt.get("alignment") == "center"
                    or (bottom - top) >= 1.15 * max(median_height, 1.0)
                ):
                    block.kind = "heading"
        if block.bbox and len(block.bbox) >= 4:
            fmt["bbox_source"] = "ocr_geometry"
        block.metadata["format"] = fmt
        if diagnostics is not None:
            diagnostics.append({
                "block_index": block_index,
                "kind_before_style": kind_before_style,
                "kind_after_style": block.kind,
                "box_indices": valid_indices,
                "bbox": [left, top, right, bottom],
                "alignment_evidence": {
                    "left_peer_count": left_alignment_support,
                    "right_peer_count": right_alignment_support,
                    "selected": fmt.get("alignment"),
                },
                "bold_evidence": {
                    "metric": {
                        "binarization": "Otsu inverse threshold",
                        "distance_transform": "L2 mask size 3",
                        "skeleton_proxy": "median local maxima in 3x3 neighborhood",
                        "stroke_width_formula": "2 * median(local_max_distance)",
                        "measurement_inset": {"x_fraction": 0.015, "y_fraction": 0.04,
                                               "minimum_pixels": 1},
                    },
                    "threshold_relative_stroke_width": 1.3,
                    "measurements": comparison_records,
                    "decision": fmt.get("bold_evidence"),
                },
                "result_format": dict(fmt),
            })
    return blocks
