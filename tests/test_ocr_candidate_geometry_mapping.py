"""Geometry-only regression tests for PaddleOCR candidate ownership."""

import numpy as np

from app.core.ocr_engine import (
    OCREngine,
    _map_candidate_polygons_to_regions,
    _paddle_result_records,
    _recognition_evidence,
    _resolve_region_candidate_source,
)


def rect(x, y, width=40, height=16):
    return [[x, y], [x + width, y], [x + width, y + height], [x, y + height]]


def test_same_order_maps_original_detections_one_to_one():
    paddles = [rect(10, 10), rect(90, 10)]
    result = _map_candidate_polygons_to_regions(paddles, [rect(10, 10), rect(90, 10)])
    assert [item.get("candidate_index") for item in result] == [0, 1]
    assert all(item["status"] == "matched" for item in result)


def test_reordered_regions_map_by_geometry_not_list_index():
    paddles = [rect(10, 10), rect(90, 10)]
    result = _map_candidate_polygons_to_regions(paddles, [rect(90, 10), rect(10, 10)])
    # Index zipping would map right→left and left→right; geometric identity does not.
    assert [item.get("candidate_index") for item in result] == [1, 0]


def test_filtered_detection_does_not_shift_following_candidate_identity():
    paddles = [rect(10, 10), rect(90, 10)]
    result = _map_candidate_polygons_to_regions(paddles, [rect(10, 10), rect(50, 10), rect(90, 10)])
    assert result[0]["candidate_index"] == 0
    assert result[1]["status"] == "unmapped"
    assert result[2]["candidate_index"] == 1


def test_duplicate_overlapping_detections_are_ambiguous():
    overlap = rect(10, 10)
    result = _map_candidate_polygons_to_regions([overlap, overlap], [overlap])
    assert result[0]["status"] == "ambiguous"
    assert result[0]["reason"] == "multiple_paddle_detections"


def test_merged_region_covering_two_paddle_boxes_has_no_single_candidate_owner():
    paddles = [rect(10, 10), rect(55, 10)]
    merged_region = rect(10, 10, 85, 16)
    result = _map_candidate_polygons_to_regions(paddles, [merged_region])
    assert result[0]["status"] == "ambiguous"
    assert result[0]["reason"] == "multiple_paddle_detections"


def test_production_horizontal_merge_does_not_claim_combined_paddle_text_as_one_detection():
    first, second = rect(10, 10), rect(48, 10)
    merged, merged_texts, _ = OCREngine._merge_horizontally_adjacent_subboxes(
        [first, second], ["first-candidate", "second-candidate"], [0.8, 0.8],
        np.full((50, 100, 3), 255, dtype=np.uint8),
    )
    assert len(merged) == 1
    # This is the former failure mode: an invented combined recognizer result
    # was indexed as if it belonged to the fused polygon.
    assert merged_texts == ["first-candidate second-candidate"]
    mapping = _map_candidate_polygons_to_regions([first, second], merged)
    assert mapping[0]["status"] == "ambiguous"
    assert mapping[0]["reason"] == "multiple_paddle_detections"


def test_nearby_non_overlapping_regions_are_not_matched():
    result = _map_candidate_polygons_to_regions([rect(10, 10)], [rect(53, 10)])
    assert result == [{"status": "unmapped", "reason": "no_geometric_overlap", "candidate_indices": []}]


def test_skewed_quadrilateral_matches_same_skewed_detection():
    skewed = [[20, 20], [68, 14], [72, 30], [24, 36]]
    result = _map_candidate_polygons_to_regions([skewed], [skewed])
    assert result[0]["status"] == "matched"
    assert result[0]["candidate_index"] == 0


def test_separate_table_regions_remain_independently_mapped():
    # Two distant table areas share identical local row/column geometry.
    paddles = [rect(20, 40), rect(120, 40), rect(20, 400), rect(120, 400)]
    targets = [rect(120, 400), rect(20, 40), rect(120, 40), rect(20, 400)]
    result = _map_candidate_polygons_to_regions(paddles, targets)
    assert [item.get("candidate_index") for item in result] == [3, 0, 1, 2]


def test_preprocessing_scale_and_detector_padding_are_removed_before_matching():
    # Synthetic source geometry: scale source→processed by 0.5, then detector adds 16 px.
    source = rect(100, 80, 60, 24)
    processed = [[x * 0.5, y * 0.5] for x, y in source]
    padded_detector = [[x + 16, y + 16] for x, y in processed]
    recovered_processed = [[x - 16, y - 16] for x, y in padded_detector]
    result = _map_candidate_polygons_to_regions([recovered_processed], [processed])
    assert result[0]["status"] == "matched"
    assert result[0]["coordinate_space"] == "preprocessed_image"


def test_unmatched_mapping_reason_is_preserved_in_box_metadata_without_candidate_text():
    mapping = {"status": "ambiguous", "reason": "multiple_paddle_detections"}
    evidence = _recognition_evidence(3, rect(10, 10), "vietocr", "recognized", 0.8,
                                    alternate_mapping=mapping)
    assert evidence["alternate_mapping"] == mapping
    assert len(evidence["candidates"]) == 1


def test_paddle_result_uses_rec_polys_after_low_score_recognition_is_filtered():
    first, filtered, third = rect(10, 10), rect(60, 10), rect(110, 10)
    raw = {"dt_polys": [first, filtered, third],
           "rec_polys": [third, first], "rec_texts": ["third", "first"],
           "rec_scores": [0.9, 0.8]}
    parsed = _paddle_result_records(raw)
    assert parsed["recognition_status"] == "verified_rec_polys_text_alignment"
    assert [entry["text"] for entry in parsed["recognitions"]] == ["third", "first"]
    assert [entry["polygon"] for entry in parsed["recognitions"]] == [third, first]
    links = _map_candidate_polygons_to_regions(
        [entry["polygon"] for entry in parsed["detections"]],
        [entry["polygon"] for entry in parsed["recognitions"]],
    )
    assert [entry["candidate_index"] for entry in links] == [2, 0]


def test_filtered_recognition_does_not_shift_later_text_onto_unrecognized_detection():
    first, filtered, third = rect(10, 10), rect(60, 10), rect(110, 10)
    parsed = _paddle_result_records({"dt_polys": [first, filtered, third],
                                     "rec_polys": [first, third],
                                     "rec_texts": ["first", "third"],
                                     "rec_scores": [0.9, 0.8]})
    recognition_to_detection = _map_candidate_polygons_to_regions(
        [entry["polygon"] for entry in parsed["detections"]],
        [entry["polygon"] for entry in parsed["recognitions"]],
    )
    # The former dt_polys[i] + rec_texts[i] loop would attach "third" to
    # `filtered` at index 1. rec_polys proves it belongs to detection index 2.
    assert [entry["candidate_index"] for entry in recognition_to_detection] == [0, 2]
    assert parsed["recognitions"][1]["text"] == "third"


def test_paddle_result_with_missing_score_keeps_verified_text_geometry_and_unknown_score():
    polygon = rect(10, 10)
    parsed = _paddle_result_records({"dt_polys": [polygon], "rec_polys": [polygon],
                                     "rec_texts": ["candidate"], "rec_scores": []})
    assert parsed["recognitions"][0]["text"] == "candidate"
    assert parsed["recognitions"][0]["score"] is None
    assert parsed["recognitions"][0]["score_status"] == "unknown"


def test_paddle_result_without_rec_polys_or_with_count_mismatch_withholds_candidates():
    polygon = rect(10, 10)
    missing = _paddle_result_records({"dt_polys": [polygon], "rec_texts": ["candidate"],
                                      "rec_scores": [0.9]})
    mismatch = _paddle_result_records({"dt_polys": [polygon], "rec_polys": [],
                                       "rec_texts": ["candidate"], "rec_scores": [0.9]})
    assert missing["recognitions"] == []
    assert missing["recognition_status"] == "unknown_missing_rec_polys"
    assert mismatch["recognitions"] == []
    assert mismatch["recognition_status"] == "unknown_rec_polygon_text_count_mismatch"


def test_merged_region_cannot_inherit_one_of_multiple_detection_candidates():
    detections = [{"detection_index": 0, "result_index": 0},
                  {"detection_index": 1, "result_index": 0}]
    recognitions = [{"source_detection_index": 0, "source_detection_result_index": 0,
                     "source_detection_mapping": {"status": "matched"}}]
    resolved, index = _resolve_region_candidate_source(
        {"status": "matched", "candidate_index": 0}, [0, 1], detections, recognitions
    )
    assert resolved["status"] == "ambiguous"
    assert index is None


def test_region_candidate_requires_source_detection_identity_to_match():
    detections = [{"detection_index": 1, "result_index": 0}]
    recognitions = [{"source_detection_index": 0, "source_detection_result_index": 0,
                     "source_detection_mapping": {"status": "matched"}}]
    resolved, index = _resolve_region_candidate_source(
        {"status": "matched", "candidate_index": 0}, [0], detections, recognitions
    )
    assert resolved["status"] == "unmapped"
    assert resolved["reason"] == "recognition_detection_link_not_verified"
    assert index is None
