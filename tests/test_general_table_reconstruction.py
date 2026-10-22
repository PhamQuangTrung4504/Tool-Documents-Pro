from __future__ import annotations

import zipfile

import cv2
import numpy as np
from lxml import etree

from app.core.docx_exporter import _estimate_column_widths, export_to_docx
from app.core.layout_analyzer import _group_boxes_sliding_window, _table_candidate_quality, reconstruct_document_layout
from app.core.models import DocumentBlock, OCRBox, OCRResult, OCRTable, TableCell
from app.core.document_diagnostics import (
    build_reconstruction_diagnostic,
    infer_visual_document_styles,
    write_reconstruction_diagnostic,
)
from app.core.table_engines import extract_physical_table_topology
from app.core.table_evaluator import TableGroundTruth, evaluate_table_sample
from app.core.ocr_engine import _resolve_table_row_identity


def _box(text: str, x1: int, y1: int, x2: int, y2: int) -> OCRBox:
    return OCRBox(text, 0.99, [[x1, y1], [x2, y1], [x2, y2], [x1, y2]])


def _ruled_table(
    x_edges: list[int],
    y_edges: list[int],
    boxes: list[OCRBox],
    *,
    missing_vertical: set[tuple[int, int]] | None = None,
    missing_horizontal: set[tuple[int, int]] | None = None,
    diagonal: tuple[int, int] | None = None,
    size: tuple[int, int] = (500, 420),
) -> np.ndarray:
    """Draw rules from independent ground-truth edge lists."""
    image = np.full((size[1], size[0], 3), 255, dtype=np.uint8)
    missing_vertical = missing_vertical or set()
    missing_horizontal = missing_horizontal or set()
    for row, y in enumerate(y_edges):
        for col in range(len(x_edges) - 1):
            if row not in (0, len(y_edges) - 1) and (row - 1, col) in missing_horizontal:
                continue
            cv2.line(image, (x_edges[col], y), (x_edges[col + 1], y), (0, 0, 0), 2)
    for col, x in enumerate(x_edges):
        for row in range(len(y_edges) - 1):
            if col not in (0, len(x_edges) - 1) and (row, col - 1) in missing_vertical:
                continue
            cv2.line(image, (x, y_edges[row]), (x, y_edges[row + 1]), (0, 0, 0), 2)
    if diagonal:
        row, col = diagonal
        cv2.line(image, (x_edges[col], y_edges[row]), (x_edges[col + 1], y_edges[row + 1]), (0, 0, 0), 2)
    return image


def test_physical_grid_recovers_spans_empty_cells_and_diagonal_from_geometry():
    x_edges = [30, 130, 260, 370, 470]
    y_edges = [30, 100, 170, 250, 330]
    boxes = [
        _box("Group heading", 150, 45, 235, 62),
        _box("Child A", 45, 115, 95, 130),
        _box("Child B", 175, 115, 225, 130),
        _box("Vertical group", 45, 190, 105, 208),
        _box("Detail", 175, 270, 220, 290),
        _box("End", 390, 270, 440, 290),
    ]
    # No divider in row 0 between columns 1 and 2 creates a horizontal span.
    # No divider in column 0 between rows 2 and 3 creates a vertical span.
    image = _ruled_table(
        x_edges,
        y_edges,
        boxes,
        missing_vertical={(0, 2)},
        missing_horizontal={(1, 0)},
        diagonal=(0, 1),
    )

    table = extract_physical_table_topology(image, boxes)
    assert table is not None
    assert (table.num_rows, table.num_cols) == (4, 4)
    assert table.column_boundaries == sorted(table.column_boundaries)
    assert len(table.column_boundaries) == table.num_cols + 1
    assert table.topology_issues() == []
    assert any(cell.col_span == 2 and cell.row_start == 0 for cell in table.cells)
    assert any(cell.row_span == 2 and cell.col_start == 0 for cell in table.cells)
    assert any(cell.diagonal_border == "tl2br" for cell in table.cells)
    assert any(cell.border_edges is not None for cell in table.cells)
    assert sum(len(cell.boxes) for cell in table.cells) == len(boxes)

    ground_truth = TableGroundTruth(
        sample_id="synthetic_geometry_spans",
        category="ruled_multilevel",
        expected_grid=[
            ["", "Group heading", "", ""],
            ["Child A Vertical group", "Child B", "", ""],
            ["", "", "", ""],
            ["", "Detail", "", "End"],
        ],
        expected_cell_spans=[(0, 2, 1, 4), (1, 0, 3, 1)],
        expected_diagonal_cells=[(0, 1, "tl2br")],
    )
    trace = evaluate_table_sample(
        "synthetic_geometry_spans",
        boxes,
        ground_truth,
        image=image,
    )
    assert trace.metrics["cell_span_f1"] == 1.0
    assert trace.metrics["diagonal_cell_f1"] == 1.0
    assert trace.metrics["ocr_box_assignment_rate"] == 1.0
    assert trace.metrics["duplicate_cell_box_assignment_count"] == 0.0


def test_cross_divider_box_is_not_forced_into_a_nearest_cell():
    x_edges = [30, 150, 280, 470]
    y_edges = [30, 110, 190, 300]
    boxes = [
        _box("Alpha", 50, 50, 100, 70),
        _box("Beta", 180, 50, 230, 70),
        _box("crosses two cells", 105, 125, 195, 145),
        _box("Omega", 320, 220, 380, 240),
    ]
    image = _ruled_table(x_edges, y_edges, boxes)
    table = extract_physical_table_topology(image, boxes)
    assert table is not None
    assert sum(box is boxes[2] for cell in table.cells for box in cell.boxes) == 0
    assert "crosses two cells" not in " ".join(cell.text for cell in table.cells)
    lines, tables = reconstruct_document_layout(boxes, image=image)
    assert tables and tables[0].topology_issues() == []
    assert sum(box is boxes[2] for line in lines for box in line) == 1


def test_table_model_reports_overlapping_or_out_of_bounds_spans():
    table = OCRTable(
        grid=[["a", "b"], ["c", "d"]],
        num_rows=2,
        num_cols=2,
        cells=[
            TableCell(0, 0, 1, 2),
            TableCell(0, 1, 2, 2),
            TableCell(1, 0, 3, 1),
        ],
    )
    assert set(table.topology_issues()) == {"overlapping_cells", "cell_span_out_of_bounds"}


def test_docx_widths_use_source_geometry_and_preserve_recognized_text(tmp_path):
    table = OCRTable(
        grid=[["Label", "Data", "Notes"], ["0", "O", "untouched-text"]],
        bbox=[0, 0, 500, 220],
        num_rows=2,
        num_cols=3,
        column_boundaries=[0, 60, 300, 500],
    )
    widths = _estimate_column_widths(table, 6.0, 11.0)
    assert widths[0] < widths[2] < widths[1]
    assert abs(sum(widths) - 6.0) < 1e-6

    output = tmp_path / "geometry.docx"
    result = OCRResult(
        full_text="Report\nLabel Data Notes\n0 O untouched-text", tables=[table],
        document_blocks=[DocumentBlock(kind="table", order=0, table_index=0)],
    )
    assert export_to_docx(result, str(output))
    with zipfile.ZipFile(output) as archive:
        document = etree.fromstring(archive.read("word/document.xml"))
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    grid_widths = [
        int(width)
        for width in document.xpath("//w:tbl[1]/w:tblGrid/w:gridCol/@w:w", namespaces=ns)
    ]
    assert len(grid_widths) == 3
    assert abs(grid_widths[0] / grid_widths[1] - 0.25) < 0.04
    assert abs(grid_widths[2] / grid_widths[1] - (200 / 240)) < 0.04
    cells = document.xpath("//w:tbl[1]/w:tr[2]/w:tc", namespaces=ns)
    values = ["".join(cell.xpath(".//w:t/text()", namespaces=ns)) for cell in cells]
    assert values == ["0", "O", "untouched-text"]


def test_physical_grid_detection_scales_across_resolution():
    for scale in (0.5, 1.0, 2.0):
        x_edges = [int(value * scale) for value in (30, 150, 300, 470)]
        y_edges = [int(value * scale) for value in (30, 110, 200, 330)]
        boxes = [
            _box("northwest", int(48 * scale), int(48 * scale), int(118 * scale), int(72 * scale)),
            _box("center value", int(190 * scale), int(135 * scale), int(260 * scale), int(158 * scale)),
            _box("southeast", int(330 * scale), int(260 * scale), int(410 * scale), int(284 * scale)),
            _box("southwest", int(48 * scale), int(260 * scale), int(118 * scale), int(284 * scale)),
        ]
        image = _ruled_table(
            x_edges,
            y_edges,
            boxes,
            size=(int(500 * scale), int(420 * scale)),
        )
        table = extract_physical_table_topology(image, boxes)
        assert table is not None
        assert (table.num_rows, table.num_cols) == (3, 3)
        assert table.topology_issues() == []


def test_same_baseline_text_in_two_distant_blocks_is_not_joined():
    boxes = [
        _box("Left organization", 30, 50, 220, 70),
        _box("Right heading", 760, 50, 970, 70),
    ]
    info = [
        {
            "box": box,
            "text": box.text,
            "left": min(p[0] for p in box.bbox),
            "right": max(p[0] for p in box.bbox),
            "top": min(p[1] for p in box.bbox),
            "bottom": max(p[1] for p in box.bbox),
            "h": 20,
            "w": max(p[0] for p in box.bbox) - min(p[0] for p in box.bbox),
            "cy": 60,
        }
        for box in boxes
    ]
    lines = _group_boxes_sliding_window(info, 0, 1000, 100, 20)
    assert len(lines) == 2
    assert all(len(line) == 1 for line in lines)


def test_docx_xml_preserves_horizontal_vertical_spans_and_diagonal_border(tmp_path):
    table = OCRTable(
        grid=[["group", "", "top-right"], ["left", "middle", "vertical"], ["", "", ""]],
        bbox=[0, 0, 600, 240],
        num_rows=3,
        num_cols=3,
        column_boundaries=[0, 160, 360, 600],
        cells=[
            TableCell(0, 0, 1, 2, text="group", diagonal_border="tl2br"),
            TableCell(0, 2, 1, 3, text="top-right"),
            TableCell(1, 0, 2, 1, text="left"),
            TableCell(1, 1, 2, 2, text="middle"),
            TableCell(1, 2, 3, 3, text="vertical"),
        ],
    )
    output = tmp_path / "spans.docx"
    assert export_to_docx(
        OCRResult(
            full_text="group top-right\nleft middle vertical", tables=[table],
            document_blocks=[DocumentBlock(kind="table", order=0, table_index=0)],
        ), str(output)
    )
    with zipfile.ZipFile(output) as archive:
        document = etree.fromstring(archive.read("word/document.xml"))
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    rows = document.xpath("//w:tbl[1]/w:tr", namespaces=ns)
    first_row_spans = [
        int(value) for value in rows[0].xpath("./w:tc/w:tcPr/w:gridSpan/@w:val", namespaces=ns)
    ]
    assert first_row_spans == [2]
    assert rows[1].xpath("./w:tc[3]/w:tcPr/w:vMerge/@w:val", namespaces=ns) == ["restart"]
    continuation = rows[2].xpath("./w:tc[3]/w:tcPr/w:vMerge", namespaces=ns)
    assert len(continuation) == 1
    assert continuation[0].get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val") in (None, "continue")
    assert document.xpath("//w:tcPr/w:tcBorders/w:tl2br", namespaces=ns)


def test_docx_export_uses_explicit_blocks_and_keeps_duplicate_text_and_pipes(tmp_path):
    first = OCRTable(grid=[["same | content"]], num_rows=1, num_cols=1)
    second = OCRTable(grid=[["same | content"]], num_rows=1, num_cols=1)
    result = OCRResult(
        full_text="same | content\nsame | content\nsame | content",
        tables=[first, second],
        document_blocks=[
            DocumentBlock(kind="paragraph", order=0, text="same | content"),
            DocumentBlock(kind="table", order=1, table_index=0),
            DocumentBlock(kind="paragraph", order=2, text="same | content"),
            DocumentBlock(kind="table", order=3, table_index=1),
        ],
    )
    output = tmp_path / "ordered.docx"
    assert export_to_docx(result, str(output))
    with zipfile.ZipFile(output) as archive:
        document = etree.fromstring(archive.read("word/document.xml"))
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    body = document.xpath("/w:document/w:body/*[self::w:p or self::w:tbl]", namespaces=ns)
    kinds = [etree.QName(node).localname for node in body]
    assert kinds == ["p", "tbl", "p", "tbl"]
    table_values = ["".join(node.xpath(".//w:t/text()", namespaces=ns)) for node in body if etree.QName(node).localname == "tbl"]
    paragraph_values = ["".join(node.xpath(".//w:t/text()", namespaces=ns)) for node in body if etree.QName(node).localname == "p"]
    assert table_values == ["same | content", "same | content"]
    assert paragraph_values == ["same | content", "same | content"]


def test_docx_export_keeps_legacy_text_when_table_position_has_no_geometry(tmp_path):
    table = OCRTable(grid=[["duplicate"]], num_rows=1, num_cols=1)
    result = OCRResult(full_text="duplicate\nbody | pipe", tables=[table])
    output = tmp_path / "legacy.docx"
    assert export_to_docx(result, str(output))
    with zipfile.ZipFile(output) as archive:
        document = etree.fromstring(archive.read("word/document.xml"))
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    assert document.xpath("//w:tbl", namespaces=ns) == []
    text = " ".join(document.xpath("//w:p//w:t/text()", namespaces=ns))
    assert "duplicate" in text
    assert "body | pipe" in text


def test_geometry_fallback_interleaves_table_between_legacy_ocrbox_paragraphs(tmp_path):
    heading = _box("Heading", 30, 10, 110, 25)
    top_left = _box("A", 35, 50, 55, 65)
    top_right = _box("B", 180, 50, 200, 65)
    bottom_left = _box("C", 35, 80, 55, 95)
    bottom_right = _box("D", 180, 80, 200, 95)
    footer = _box("Footer", 30, 130, 100, 145)
    table = OCRTable(
        grid=[["A", "B"], ["C", "D"]], bbox=[25, 40, 220, 105], num_rows=2, num_cols=2,
        row_boxes=[[top_left, top_right], [bottom_left, bottom_right]],
    )
    result = OCRResult(
        boxes=[heading, top_left, top_right, bottom_left, bottom_right, footer],
        full_text="Heading\nA B\nC D\nFooter", tables=[table],
    )
    output = tmp_path / "legacy-geometry.docx"
    assert export_to_docx(result, str(output))
    with zipfile.ZipFile(output) as archive:
        document = etree.fromstring(archive.read("word/document.xml"))
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    children = document.xpath("/w:document/w:body/*[self::w:p or self::w:tbl]", namespaces=ns)
    assert [etree.QName(node).localname for node in children] == ["p", "tbl", "p"]
    assert "Heading" in "".join(children[0].xpath(".//w:t/text()", namespaces=ns))
    assert "Footer" in "".join(children[2].xpath(".//w:t/text()", namespaces=ns))


def test_docx_header_rows_repeat_multiline_cell_and_no_table_fallback(tmp_path):
    table = OCRTable(
        grid=[["Header", "Header B"], ["first\nsecond", ""]],
        num_rows=2, num_cols=2, header_rows=[0],
    )
    result = OCRResult(
        tables=[table],
        document_blocks=[DocumentBlock(kind="table", order=0, table_index=0)],
    )
    output = tmp_path / "repeat-header.docx"
    assert export_to_docx(result, str(output))
    with zipfile.ZipFile(output) as archive:
        document = etree.fromstring(archive.read("word/document.xml"))
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    assert document.xpath("count(//w:tr[1]/w:trPr/w:tblHeader)", namespaces=ns) == 1.0
    assert "first" in document.xpath("string(//w:tr[2])", namespaces=ns)
    assert "second" in document.xpath("string(//w:tr[2])", namespaces=ns)

    no_table = OCRResult(
        full_text="A literal | separator remains text",
        document_blocks=[DocumentBlock(kind="paragraph", order=0, text="A literal | separator remains text")],
    )
    text_output = tmp_path / "no-table.docx"
    assert export_to_docx(no_table, str(text_output))
    with zipfile.ZipFile(text_output) as archive:
        plain_xml = etree.fromstring(archive.read("word/document.xml"))
    assert plain_xml.xpath("//w:tbl", namespaces=ns) == []
    assert "A literal | separator remains text" in plain_xml.xpath("string(//w:p)", namespaces=ns)


def test_layout_grouping_does_not_rewrite_recognized_bullet_marker():
    marker = _box("*", 20, 40, 34, 54)
    text = _box("recognized item", 44, 40, 170, 54)
    info = [
        {"box": item, "text": item.text, "left": item.bbox[0][0], "right": item.bbox[1][0],
         "top": item.bbox[0][1], "bottom": item.bbox[2][1], "h": 14,
         "w": item.bbox[1][0] - item.bbox[0][0], "cy": 47}
        for item in (marker, text)
    ]
    lines = _group_boxes_sliding_window(info, 0, 300, 80, 14)
    assert len(lines) == 1
    assert marker.text == "*"


def _quality_table(boxes: list[OCRBox], rows: list[list[str]] | None = None) -> OCRTable:
    rows = rows if rows is not None else [["A", "B"], ["C", "D"]]
    return OCRTable(
        grid=rows,
        num_rows=len(rows),
        num_cols=max((len(row) for row in rows), default=0),
        row_boxes=[[boxes[0]], [boxes[1]]] if len(boxes) >= 2 and len(rows) == 2 else [],
        cells=[TableCell(0, 0, 1, 1, boxes=[boxes[0]])] if boxes else [],
    )


def test_table_candidate_quality_valid_invalid_empty_and_tie():
    boxes = [_box("A", 1, 1, 10, 10), _box("C", 1, 20, 10, 30)]
    valid = _quality_table(boxes)
    valid_quality = _table_candidate_quality(valid, boxes)
    assert 0 < valid_quality[0] <= 1
    assert valid_quality[1:] == (1.0, 1.0)

    invalid = _quality_table(boxes)
    invalid.cells.append(TableCell(0, 0, 2, 1))
    invalid_quality = _table_candidate_quality(invalid, boxes)
    assert invalid_quality[0] == 0
    assert invalid_quality[1] == valid_quality[1]

    empty = OCRTable()
    assert _table_candidate_quality(empty, boxes) == (0.0, 0.0, 0.0)
    assert _table_candidate_quality(valid, []) == (0.0, 0.0, 0.0)

    same_quality = _table_candidate_quality(_quality_table(boxes), boxes)
    assert same_quality == valid_quality


def test_table_candidate_quality_prefers_more_complete_candidate():
    boxes = [_box("A", 1, 1, 10, 10), _box("B", 20, 1, 30, 10)]
    complete = OCRTable(
        grid=[["A", "B"]], num_rows=1, num_cols=2,
        row_boxes=[boxes], cells=[TableCell(0, 0, 1, 1, boxes=[boxes[0]]), TableCell(0, 1, 1, 2, boxes=[boxes[1]])],
    )
    sparse = OCRTable(
        grid=[["A", ""]], num_rows=1, num_cols=2,
        row_boxes=[[boxes[0]]], cells=[TableCell(0, 0, 1, 1, boxes=[boxes[0]])],
    )
    assert _table_candidate_quality(complete, boxes)[0] > _table_candidate_quality(sparse, boxes)[0]


def test_equal_content_rows_have_independent_logical_identity():
    first_box = _box("repeat", 20, 20, 60, 35)
    second_box = _box("repeat", 20, 50, 60, 65)
    row_map = [
        (frozenset({id(first_box)}), 0, 0, ["repeat"]),
        (frozenset({id(second_box)}), 0, 1, ["repeat"]),
    ]
    rendered: set[tuple[int, int]] = set()
    first = _resolve_table_row_identity(frozenset({id(first_box)}), row_map, rendered)
    second = _resolve_table_row_identity(frozenset({id(second_box)}), row_map, rendered)
    assert first == (0, 0, ["repeat"])
    assert second == (0, 1, ["repeat"])
    assert rendered == {(0, 0), (0, 1)}
    assert _resolve_table_row_identity(frozenset({id(first_box)}), row_map, rendered) is None


def test_layout_engine_branch_uses_candidate_quality(monkeypatch):
    import app.core.layout_analyzer as analyzer
    import app.core.table_engines as engines

    boxes = [_box("one", 20, 20, 60, 35), _box("two", 20, 50, 60, 65)]
    legacy = OCRTable(grid=[["one"], ["two"]], num_rows=2, num_cols=1)
    spatial = OCRTable(grid=[["one"], ["two"]], num_rows=2, num_cols=1)
    monkeypatch.setattr(analyzer, "detect_and_reconstruct_table", lambda *args: (legacy, [], [], []))
    monkeypatch.setattr(engines, "unified_hybrid_engine", lambda *args, **kwargs: ([], [spatial]))
    original = analyzer._table_candidate_quality
    calls: list[OCRTable] = []

    def score_candidate(table: OCRTable, source: list[OCRBox]):
        calls.append(table)
        original(table, source)
        # On each invocation the legacy candidate is scored before spatial.
        score = 0.2 if len(calls) % 2 else 0.9
        return score, 0.8, 1.0

    monkeypatch.setattr(analyzer, "_table_candidate_quality", score_candidate)
    diagnostics: dict = {}
    analyzer.reconstruct_document_layout(boxes, diagnostics=diagnostics)
    assert len(calls) >= 2
    assert diagnostics["selected_engine"] == "unified_hybrid"


def test_layout_engine_uses_stable_legacy_choice_for_exact_score_tie(monkeypatch):
    import app.core.layout_analyzer as analyzer
    import app.core.table_engines as engines

    boxes = [_box("one", 20, 20, 60, 35), _box("two", 20, 50, 60, 65)]
    legacy = OCRTable(grid=[["one"], ["two"]], num_rows=2, num_cols=1)
    spatial = OCRTable(grid=[["one"], ["two"]], num_rows=2, num_cols=1)
    monkeypatch.setattr(analyzer, "detect_and_reconstruct_table", lambda *args: (legacy, [], [], []))
    monkeypatch.setattr(engines, "unified_hybrid_engine", lambda *args, **kwargs: ([], [spatial]))
    monkeypatch.setattr(analyzer, "_table_candidate_quality", lambda *args: (0.5, 0.5, 1.0))
    diagnostics: dict = {}
    analyzer.reconstruct_document_layout(boxes, diagnostics=diagnostics)
    assert diagnostics["selected_engine"] == "legacy_anchor_quality"


def test_reconstruction_diagnostic_reports_source_ownership_and_loss(tmp_path):
    paragraph_box = _box("Outside note", 10, 10, 90, 25)
    table_box = _box("value", 20, 50, 70, 65)
    outside_table_box = _box("nearby caption", 10, 90, 140, 105)
    dropped_box = _box("unassigned source", 10, 120, 140, 135)
    table = OCRTable(
        grid=[["value"]], bbox=[15, 45, 80, 75], num_rows=1, num_cols=1,
        row_boxes=[[table_box, outside_table_box]],
        cells=[TableCell(0, 0, 1, 1, text="value", bbox=[15, 45, 80, 75], boxes=[table_box])],
    )
    result = OCRResult(
        boxes=[paragraph_box, table_box, outside_table_box],
        full_text="Outside note\nvalue",
        raw_full_text="Outside note\nvalue",
        tables=[table],
        document_blocks=[
            DocumentBlock(kind="paragraph", order=0, text="Outside note", raw_text="Outside note", box_indices=[0]),
            DocumentBlock(kind="table", order=1, table_index=0, box_indices=[1, 2]),
        ],
    )

    report = build_reconstruction_diagnostic(result, source_boxes=[paragraph_box, table_box, outside_table_box, dropped_box])
    assert report["recognized_boxes"][0]["owners"] == ["block:0:paragraph"]
    assert report["recognized_boxes"][1]["owners"] == ["table:0"]
    assert report["recognized_boxes"][2]["owners"] == ["table:0"]
    assert report["integrity"]["table_owned_boxes_outside_bbox"] == {0: [2]}
    assert report["integrity"]["unassigned_box_indices"] == [3]
    assert report["integrity"]["recognized_source_boxes_missing_from_result"] == [3]
    assert report["output"]["paragraphs"] == ["Outside note"]
    assert report["output"]["tables"] == [[ ["value"] ]]

    # Diagnostic files are explicitly requested and placed in a caller-owned temp dir.
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory(prefix="tdp-reconstruction-diagnostic-") as external_tmp:
        destination = write_reconstruction_diagnostic(result, Path(external_tmp) / "trace.json")
        assert destination.exists()
        assert "Outside note" in destination.read_text(encoding="utf-8")


def test_visual_style_inference_uses_relative_source_stroke_and_geometry():
    import cv2
    import numpy as np

    image = np.full((250, 520, 3), 255, dtype=np.uint8)
    entries = []
    for baseline, text in zip((38, 76, 114, 152), ("Body line A", "Body line B", "Body line C", "Body line D")):
        cv2.putText(image, text, (24, baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
        (width, height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)
        entries.append((text, [
            [24, baseline - height], [24 + width, baseline - height],
            [24 + width, baseline], [24, baseline],
        ]))
    heading = "Section title"
    heading_width, heading_height = cv2.getTextSize(heading, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
    heading_x, heading_baseline = (520 - heading_width) // 2, 215
    cv2.putText(image, heading, (heading_x, heading_baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
    entries.append((heading, [
        [heading_x, heading_baseline - heading_height],
        [heading_x + heading_width, heading_baseline - heading_height],
        [heading_x + heading_width, heading_baseline], [heading_x, heading_baseline],
    ]))
    boxes = [OCRBox(text, 0.9, bbox) for text, bbox in entries]
    blocks = [
        DocumentBlock(kind="paragraph", order=i, text=text, box_indices=[i], bbox=[
            min(p[0] for p in box.bbox), min(p[1] for p in box.bbox),
            max(p[0] for p in box.bbox), max(p[1] for p in box.bbox),
        ])
        for i, (text, box) in enumerate(zip((entry[0] for entry in entries), boxes))
    ]

    infer_visual_document_styles(blocks, boxes, image)
    heading_block = blocks[-1]
    assert heading_block.kind == "heading"
    assert heading_block.metadata["format"]["bold"] is True
    assert heading_block.metadata["format"]["alignment"] == "center"
    assert all("italic" not in block.metadata.get("format", {}) for block in blocks)


def test_docx_export_honors_explicit_visual_format_metadata(tmp_path):
    import docx
    from app.core.docx_exporter import export_to_docx

    result = OCRResult(
        full_text="Evidence based title",
        document_blocks=[DocumentBlock(
            kind="heading", order=0, text="Evidence based title",
            metadata={"format": {"alignment": "right", "bold": True, "italic": True, "font_size_pt": 14}},
        )],
    )
    output = tmp_path / "style.docx"
    assert export_to_docx(result, str(output))
    paragraph = docx.Document(output).paragraphs[0]
    assert paragraph.alignment == docx.enum.text.WD_ALIGN_PARAGRAPH.RIGHT
    assert paragraph.runs[0].bold is True
    assert paragraph.runs[0].italic is True
    assert paragraph.runs[0].font.size.pt == 14
