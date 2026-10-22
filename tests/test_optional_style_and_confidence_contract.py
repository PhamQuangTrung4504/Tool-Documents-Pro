import cv2
import numpy as np

from app.core import document_diagnostics
from app.core.document_diagnostics import infer_visual_document_styles
from app.core.models import DocumentBlock, OCRBox, OCRResult, format_confidence


def _box(text, left, top, right, bottom):
    return OCRBox(text=text, confidence=0.9, bbox=[
        [left, top], [right, top], [right, bottom], [left, bottom],
    ])


def test_bold_heading_without_alignment_evidence_does_not_raise_keyerror(monkeypatch):
    image = np.full((100, 500, 3), 255, dtype=np.uint8)
    boxes = [
        _box("Title", 10, 10, 100, 40),
        _box("Peer one", 150, 10, 240, 35),
        _box("Peer two", 260, 10, 350, 35),
        _box("Peer three", 370, 10, 460, 35),
    ]
    stroke_values = iter([3.0, 1.0, 1.0, 1.0])
    monkeypatch.setattr(document_diagnostics, "_stroke_width", lambda crop: next(stroke_values))
    blocks = [DocumentBlock("paragraph", i, text=box.text, box_indices=[i]) for i, box in enumerate(boxes)]

    infer_visual_document_styles(blocks, boxes, image)

    assert blocks[0].kind == "heading"
    assert blocks[0].metadata["format"]["bold"] is True
    assert "alignment" not in blocks[0].metadata["format"]


def test_paragraph_without_alignment_or_bold_evidence_keeps_both_unknown():
    image = np.full((80, 300, 3), 255, dtype=np.uint8)
    boxes = [_box("One line", 20, 20, 120, 40)]
    blocks = [DocumentBlock("paragraph", 0, text="One line", box_indices=[0])]

    infer_visual_document_styles(blocks, boxes, image)

    fmt = blocks[0].metadata["format"]
    assert "alignment" not in fmt
    assert fmt["bold_evidence"]["status"] == "unknown"
    assert "bold" not in fmt


def test_supported_left_right_and_center_alignment_are_preserved():
    image = np.full((100, 400, 3), 255, dtype=np.uint8)
    cases = [
        ("left", [_box("target", 40, 10, 100, 25), _box("peer-a", 40, 35, 110, 50),
                   _box("peer-b", 40, 60, 120, 75)]),
        ("right", [_box("target", 200, 10, 260, 25), _box("peer-a", 170, 35, 260, 50),
                    _box("peer-b", 180, 60, 260, 75)]),
        ("center", [_box("target", 170, 10, 230, 25)]),
    ]

    for expected, boxes in cases:
        block = DocumentBlock("paragraph", 0, text="target", box_indices=[0])
        infer_visual_document_styles([block], boxes, image)
        assert block.metadata["format"]["alignment"] == expected


def test_optional_alignment_metadata_is_safe_with_and_without_table():
    image = np.full((100, 400, 3), 255, dtype=np.uint8)
    boxes = [_box("paragraph", 20, 15, 150, 35), _box("table cell", 20, 55, 150, 75)]
    paragraph = DocumentBlock("paragraph", 0, text="paragraph", box_indices=[0], metadata={})
    table = DocumentBlock("table", 1, box_indices=[1])

    infer_visual_document_styles([paragraph, table], boxes, image)
    assert "alignment" not in paragraph.metadata["format"]
    assert paragraph.metadata["format"]["bold_evidence"]["status"] == "unknown"

    unknown = DocumentBlock("paragraph", 0, text="paragraph", box_indices=[0], metadata={
        "format": {"bold_evidence": {"status": "unknown"}},
    })
    infer_visual_document_styles([unknown], boxes[:1], image)
    assert "alignment" not in unknown.metadata["format"]
    assert unknown.metadata["format"]["bold_evidence"]["status"] == "unknown"


def test_style_inference_preserves_independent_format_and_recomputes_old_bold_evidence():
    image = np.full((80, 300, 3), 255, dtype=np.uint8)
    boxes = [_box("One line", 20, 20, 120, 40)]
    block = DocumentBlock("paragraph", 0, text="One line", box_indices=[0], metadata={
        "format": {
            "italic": True,
            "font_size_pt": 15,
            "alignment": "right",
            "bold": True,
            "bold_evidence": {"status": "supported", "relative_stroke_width": 1.5},
        },
    })

    infer_visual_document_styles([block], boxes, image)

    fmt = block.metadata["format"]
    assert fmt["italic"] is True
    assert fmt["font_size_pt"] == 15
    assert fmt["alignment"] == "right"
    assert "bold" not in fmt
    assert fmt["bold_evidence"]["status"] == "unknown"

    explicitly_regular = DocumentBlock("paragraph", 0, text="One line", box_indices=[0], metadata={
        "format": {"bold": False, "bold_evidence": {"status": "unknown"}},
    })
    infer_visual_document_styles([explicitly_regular], boxes, image)
    assert explicitly_regular.metadata["format"]["bold"] is False


def test_average_confidence_distinguishes_missing_partial_and_zero_scores():
    def result(scores):
        return OCRResult(boxes=[
            OCRBox(str(i), score, [[i, 0], [i+1, 0], [i+1, 1], [i, 1]])
            for i, score in enumerate(scores)
        ])

    assert result([None, None]).average_confidence is None
    assert result([None, 0.8]).average_confidence == 0.8
    assert result([0.2, 0.8]).average_confidence == 0.5
    assert result([0.0]).average_confidence == 0.0
    assert format_confidence(None) == "N/A"
    assert format_confidence(None, as_percent=True) == "N/A"
    assert format_confidence(0.0, as_percent=True) == "0.0%"
