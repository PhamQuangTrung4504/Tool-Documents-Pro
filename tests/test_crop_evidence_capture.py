import copy
import json
import zipfile

import cv2
import numpy as np

from app.core.document_diagnostics import infer_visual_document_styles
from app.core.models import DocumentBlock, OCRBox
from scripts.capture_ocr_evidence import EvidenceWriter


def test_evidence_writer_links_candidate_to_observed_crop_image(tmp_path):
    page = np.full((80, 120, 3), 245, dtype=np.uint8)
    page[20:40, 30:75] = (15, 30, 45)
    writer = EvidenceWriter(tmp_path / "run", tmp_path / "input.png", "run-unique")
    writer.observe({"event": "source_image", "image_size": [120, 80]}, page)
    writer.observe({"event": "preprocessed_rectified_crop", "crop_id": 9}, page[20:40, 30:75].copy())
    writer.observe({"event": "recognition_candidate", "crop_id": 9, "recognizer": "model-x",
                    "text": "candidate", "recognizer_input_exact": True})

    candidate = writer.records[-1]
    crop_file = candidate["crop_artifact_files"]["preprocessed_rectified_crop"]
    saved_crop = cv2.imread(str(writer.run_dir / crop_file))
    assert saved_crop.shape[:2] == (20, 45)
    assert np.array_equal(saved_crop, page[20:40, 30:75])
    assert candidate["run_id"] == "run-unique"
    assert candidate["crop_id"] == 9


def test_optional_style_diagnostics_capture_measurement_and_peers_without_changing_style(tmp_path):
    image = np.full((180, 420, 3), 255, dtype=np.uint8)
    texts = ["Text one", "Text two", "Text three", "Bold title"]
    boxes = []
    for index, (text, thickness) in enumerate(zip(texts, [1, 1, 1, 2])):
        baseline = 30 + index * 38
        scale = 0.55
        (width, height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        cv2.putText(image, text, (20, baseline), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (0, 0, 0), thickness, cv2.LINE_AA)
        boxes.append(OCRBox(text, 0.9, [[20, baseline-height], [20+width, baseline-height],
                                        [20+width, baseline], [20, baseline]]))
    blocks_without_diagnostics = [
        DocumentBlock("paragraph", index, text=box.text, box_indices=[index])
        for index, box in enumerate(boxes)
    ]
    blocks_with_diagnostics = copy.deepcopy(blocks_without_diagnostics)
    records = []

    infer_visual_document_styles(blocks_without_diagnostics, boxes, image)
    infer_visual_document_styles(blocks_with_diagnostics, boxes, image, diagnostics=records)

    assert [(block.kind, block.metadata) for block in blocks_without_diagnostics] == [
        (block.kind, block.metadata) for block in blocks_with_diagnostics
    ]
    target = records[-1]
    measurement = target["bold_evidence"]["measurements"][0]
    assert measurement["bbox"]
    assert measurement["measurement_bbox"]
    assert measurement["peer_box_indices"]
    assert target["bold_evidence"]["metric"]["stroke_width_formula"] == "2 * median(local_max_distance)"
    assert target["bold_evidence"]["threshold_relative_stroke_width"] == 1.3

    writer = EvidenceWriter(tmp_path / "style", tmp_path / "source.png", "style-run")
    writer.observe({"event": "source_image"}, image)
    writer.observe({"event": "style_analysis", **target})
    target_artifact = writer.records[-1]["target_artifacts"][0]
    peer_artifact = writer.records[-1]["comparison_artifacts"][0]
    assert cv2.imread(str(writer.run_dir / target_artifact["measurement_image"])) is not None
    assert cv2.imread(str(writer.run_dir / peer_artifact["measurement_image"])) is not None


def test_report_json_and_external_zip_package_are_readable(tmp_path):
    writer = EvidenceWriter(tmp_path / "bundle_data" / "one", tmp_path / "source.tif", "run-1")
    writer.observe({"event": "recognition_candidate", "crop_id": 1, "text": "sample"})
    writer.write_report(error=RuntimeError("model unavailable"))
    archive = tmp_path / "evidence.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in (tmp_path / "bundle_data").rglob("*"):
            if path.is_file():
                bundle.write(path, path.relative_to(tmp_path))
    with zipfile.ZipFile(archive) as bundle:
        assert "bundle_data/one/report.json" in bundle.namelist()
        report = json.loads(bundle.read("bundle_data/one/report.json"))
    assert report["status"] == "failed"
    assert report["events"][0]["crop_id"] == 1
