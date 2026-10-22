import cv2
import numpy as np

from app.core.document_diagnostics import infer_visual_document_styles
from app.core.models import DocumentBlock, OCRBox
from app.core.ocr_engine import _recognition_evidence


def test_disagreement_keeps_candidates_from_the_same_crop_without_score_voting():
    bbox = [[10, 10], [100, 10], [100, 30], [10, 30]]
    evidence = _recognition_evidence(4, bbox, "vietocr:model", "mẫu (3?445)", 0.82,
                                    "mẫu (3=445)", 0.99)

    assert evidence["arbitration"] == "disagreement_preserved"
    assert evidence["needs_review"] is True
    assert evidence["selected_candidate"] == 0
    assert evidence["confidence_comparable_across_models"] is False
    assert [candidate["crop_id"] for candidate in evidence["candidates"]] == [4, 4]
    assert all(candidate["crop_bbox"] == bbox for candidate in evidence["candidates"])
    assert evidence["candidate_recommendation"] == {
        "status": "human_review_required",
        "candidate_indices": [0, 1],
        "selected_index": None,
        "reason": "verified_candidates_disagree",
    }


def test_candidates_from_distinct_crops_are_not_combined():
    first = _recognition_evidence(2, [[0, 0], [20, 0], [20, 10], [0, 10]],
                                 "vietocr", "a", 0.7, "b", 0.9)
    second = _recognition_evidence(3, [[40, 0], [60, 0], [60, 10], [40, 10]],
                                  "vietocr", "c", 0.7, "d", 0.9)
    assert {candidate["crop_id"] for candidate in first["candidates"]} == {2}
    assert {candidate["crop_id"] for candidate in second["candidates"]} == {3}
    assert first["candidates"][0]["text"] == "a"
    assert second["candidates"][0]["text"] == "c"


def test_uncertain_vietnamese_and_symbol_candidates_are_preserved_verbatim():
    source_candidates = ["Thừ trường hợp", "(3?445)", "0s", "đ/d", "á"]
    for crop_id, candidate_text in enumerate(source_candidates):
        evidence = _recognition_evidence(crop_id, [[0, 0], [30, 0], [30, 12], [0, 12]],
                                        "vietocr", candidate_text, None)
        assert evidence["candidates"][0]["text"] == candidate_text
        assert evidence["needs_review"] is True
        assert evidence["selected_candidate"] == 0


def test_agreement_retains_primary_without_claiming_cross_model_confidence_vote():
    evidence = _recognition_evidence(
        8, [[1, 2], [20, 2], [20, 12], [1, 12]],
        "vietocr", "đối chiếu 123", 0.62,
        alternate_text="đối chiếu 123", alternate_confidence=0.99,
        alternate_mapping={"status": "matched"},
    )
    assert evidence["selected_candidate"] == 0
    assert evidence["confidence_comparable_across_models"] is False
    assert evidence["candidate_recommendation"] == {
        "status": "primary_retained",
        "candidate_indices": [0],
        "selected_index": 0,
        "reason": "candidate_texts_agree",
    }


def test_disagreement_proposal_does_not_replace_primary_for_symbols_or_brackets():
    for primary, alternate in (("(3?445)", "3=4+5"), ("(abc", "abc"), ("12+3", "12-3")):
        evidence = _recognition_evidence(
            9, [[1, 2], [30, 2], [30, 12], [1, 12]],
            "vietocr", primary, None,
            alternate_text=alternate, alternate_confidence=0.0,
            alternate_mapping={"status": "matched"},
        )
        assert evidence["candidates"][0]["text"] == primary
        assert evidence["candidate_recommendation"]["selected_index"] is None
        assert evidence["candidate_recommendation"]["candidate_indices"] == [0, 1]


def test_missing_primary_candidate_does_not_automatically_promote_alternate():
    evidence = _recognition_evidence(
        10, [[1, 2], [30, 2], [30, 12], [1, 12]],
        "none", "", None, alternate_text="recognized alternate",
        alternate_mapping={"status": "matched"},
    )
    assert evidence["needs_review"] is True
    assert evidence["candidate_recommendation"] == {
        "status": "human_review_required",
        "candidate_indices": [1],
        "selected_index": None,
        "reason": "primary_candidate_missing",
    }


def _put_line(image, text, baseline, thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    (width, height), _ = cv2.getTextSize(text, font, scale, thickness)
    left = 24
    cv2.putText(image, text, (left, baseline), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)
    return OCRBox(text, 1.0, [[left, baseline-height], [left+width, baseline-height],
                              [left+width, baseline], [left, baseline]])


def test_caps_are_not_bold_evidence_and_unknown_style_stays_unknown():
    image = np.full((230, 520, 3), 255, dtype=np.uint8)
    boxes = [_put_line(image, text, baseline) for text, baseline in
             [("Body line A", 35), ("Body line B", 75), ("BODY LINE C", 115), ("Body line D", 155)]]
    blocks = [DocumentBlock("paragraph", index, text=box.text, box_indices=[index])
              for index, box in enumerate(boxes)]

    infer_visual_document_styles(blocks, boxes, image)

    caps_format = blocks[2].metadata["format"]
    assert caps_format.get("bold") is not True
    assert caps_format["bold_evidence"]["status"] == "unknown"
    assert "italic" not in caps_format


def test_skeleton_thickness_detects_bold_when_distance_peaks_are_quantized(monkeypatch, tmp_path):
    import app.core.document_diagnostics as style_module
    from docx import Document
    from app.core.docx_exporter import export_to_docx
    from app.core.models import OCRResult

    image = np.full((150, 360, 3), 255, dtype=np.uint8)
    text = "II. SECTION HEADING"
    boxes = []
    blocks = []
    for index, (baseline, thickness) in enumerate(((28, 1), (58, 1), (88, 1), (118, 2))):
        (width, height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, thickness)
        cv2.putText(image, text, (20, baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 0), thickness, cv2.LINE_AA)
        box = OCRBox(text, 1.0, [[20, baseline-height], [20+width, baseline-height],
                                 [20+width, baseline], [20, baseline]])
        boxes.append(box)
        blocks.append(DocumentBlock("paragraph", index, text=text, box_indices=[index]))

    # Emulate the low-resolution condition observed in real scans: the old
    # distance-transform median is identical for regular and bold boxes.
    monkeypatch.setattr(style_module, "_stroke_width", lambda _crop: 1.91)
    infer_visual_document_styles(blocks, boxes, image)

    assert [block.metadata["format"].get("bold") is True for block in blocks] == [False, False, False, True]
    evidence = blocks[-1].metadata["format"]["bold_evidence"]
    assert evidence["signals"]["stroke_width"] == "unknown"
    assert evidence["signals"]["skeleton_thickness"] == "supported"

    output = tmp_path / "skeleton_bold.docx"
    assert export_to_docx(OCRResult(boxes=boxes, document_blocks=blocks), str(output))
    runs = [paragraph.runs[0] for paragraph in Document(output).paragraphs]
    assert [run.bold for run in runs] == [None, None, None, True]


def test_table_edge_boxes_do_not_inflate_paragraph_bold_measurement():
    image = np.full((220, 520, 3), 255, dtype=np.uint8)
    body_boxes = [_put_line(image, f"Normal body {i}", 35 + i*38) for i in range(4)]
    edge_box = _put_line(image, "Grid rule neighbor", 195, thickness=1)
    x1, y1, x2, y2 = edge_box.bbox[0][0], edge_box.bbox[0][1], edge_box.bbox[2][0], edge_box.bbox[2][1]
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), 2)
    boxes = body_boxes + [edge_box]
    blocks = [DocumentBlock("paragraph", i, text=box.text, box_indices=[i]) for i, box in enumerate(body_boxes)]
    blocks.append(DocumentBlock("table", len(blocks), box_indices=[len(boxes)-1]))

    infer_visual_document_styles(blocks, boxes, image)

    assert blocks[0].metadata["format"].get("bold") is not True
