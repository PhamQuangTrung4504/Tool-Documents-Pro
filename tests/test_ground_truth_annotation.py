from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core.ground_truth_annotation import (
    VerifiedGroundTruthOverwriteError,
    make_ground_truth,
    new_paragraph_block,
    new_table_block,
    save_manifest_safely,
    serialize_reading_order,
    verify_ground_truth,
)
from app.core.ocr_benchmark import validate_manifest


def test_reading_order_keeps_paragraphs_empty_cells_rows_and_table_spans():
    paragraph_before = new_paragraph_block("Tiêu đề đã kiểm tra")
    table = new_table_block(2, 3)
    table["grid"] = [["Nhóm", "", "Số liệu"], ["", "Giá trị", "05%"]]
    table["cell_spans"] = [[0, 0, 1, 2]]
    table["diagonal_cells"] = [[0, 2, "tl2br"]]
    paragraph_after = new_paragraph_block("Ghi chú cuối bảng")

    assert serialize_reading_order([paragraph_before, table, paragraph_after]) == (
        "Tiêu đề đã kiểm tra\nNhóm\t\tSố liệu\n\tGiá trị\t05%\nGhi chú cuối bảng"
    )
    truth = make_ground_truth(
        [paragraph_before, table, paragraph_after], verified=False,
        reviewed_by="", reviewed_at="", source_note="",
    )
    assert truth["table"]["expected_grid"] == table["grid"]
    assert truth["table"]["expected_cell_spans"] == [[0, 0, 1, 2]]
    assert truth["table"]["expected_diagonal_cells"] == [[0, 2, "tl2br"]]
    assert truth["table"]["expected_non_table_lines"] == ["Tiêu đề đã kiểm tra", "Ghi chú cuối bảng"]
    assert truth["verified"] is False


def test_verification_requires_every_block_to_be_reviewed_explicitly():
    blocks = [new_paragraph_block("Nội dung đối chiếu ảnh")]
    with pytest.raises(ValueError, match="chưa được đánh dấu"):
        verify_ground_truth(blocks, "người kiểm tra", "2026-10-22", "Ảnh gốc")

    blocks[0]["reviewed"] = True
    truth = verify_ground_truth(blocks, "người kiểm tra", "2026-10-22", "Đã so với ảnh nguồn")
    assert truth["verified"] is True
    document = {"id": "sample", "image": "source.png", "ground_truth": truth}
    assert validate_manifest(document) == []


def test_export_and_reopen_manifest_preserves_verified_annotations(tmp_path: Path):
    blocks = [new_paragraph_block("Văn bản được nhập thủ công")]
    blocks[0]["reviewed"] = True
    truth = verify_ground_truth(blocks, "reviewer", "2026-10-22", "Đối chiếu ảnh nguồn")
    target = tmp_path / "manifest with spaces.json"
    payload = {"schema_version": 1, "documents": [{"id": "doc-1", "image": "source.png", "ground_truth": truth}]}
    save_manifest_safely(target, payload)

    reopened = json.loads(target.read_text(encoding="utf-8"))
    assert reopened["documents"][0]["ground_truth"]["text"] == "Văn bản được nhập thủ công"
    assert reopened["documents"][0]["ground_truth"]["verified"] is True
    assert validate_manifest(reopened["documents"][0]) == []


def test_export_refuses_to_overwrite_verified_ground_truth_without_consent(tmp_path: Path):
    target = tmp_path / "verified.json"
    original = {"documents": [{"ground_truth": {"verified": True, "text": "original"}}]}
    save_manifest_safely(target, original)
    original_bytes = target.read_bytes()
    replacement = {"documents": [{"ground_truth": {"verified": False, "text": "draft"}}]}

    with pytest.raises(VerifiedGroundTruthOverwriteError):
        save_manifest_safely(target, replacement)
    assert target.read_bytes() == original_bytes

    save_manifest_safely(target, replacement, allow_replace_verified=True)
    assert json.loads(target.read_text(encoding="utf-8")) == replacement
