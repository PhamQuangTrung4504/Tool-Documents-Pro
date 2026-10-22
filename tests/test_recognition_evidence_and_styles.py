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
