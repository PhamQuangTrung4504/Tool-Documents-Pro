import numpy as np
import json
import sys
import tempfile
import zipfile
from pathlib import Path

from PIL import Image

from scripts import ocr_crop_ablation
from scripts import ocr_selective_rerecognition
from scripts.ocr_crop_ablation import build_crop_variants


def _synthetic_crop(height=24, width=50):
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    image[0:4, :] = 0  # đường ngang sát biên để kiểm tra biến thể không tẩy
    image[8:20, 4:8] = 0
    image[6:8, 5:7] = 0  # nét nhỏ gợi vùng có dấu phía trên
    return image


def test_crop_variants_are_independent_and_keep_expected_geometry():
    raw = _synthetic_crop()
    clean = raw.copy()
    clean[:4, :] = 255
    before_raw = raw.copy()
    before_clean = clean.copy()

    variants = build_crop_variants(raw, clean, resize_scale=2.0)

    assert set(variants) == {
        "current_crop", "skip_border_cleaning", "safe_padding", "aspect_preserving_resize"
    }
    assert np.array_equal(raw, before_raw)
    assert np.array_equal(clean, before_clean)
    assert variants["aspect_preserving_resize"].shape[:2] == (48, 100)
    assert variants["current_crop"].shape[0] == 44  # helper production phóng crop thấp hơn 36 px lên 44 px


def test_skip_border_cleaning_retains_synthetic_edge_line():
    raw = _synthetic_crop()
    cleaned = raw.copy()
    cleaned[:4, :] = 255

    variants = build_crop_variants(raw, cleaned)

    assert np.count_nonzero(variants["skip_border_cleaning"][:4] == 0) > 0
    assert np.count_nonzero(variants["current_crop"][:4] == 0) == 0


def test_narrow_crop_padding_and_resize_preserve_aspect_ratio():
    raw = _synthetic_crop(height=30, width=20)
    variants = build_crop_variants(raw, raw.copy(), resize_scale=1.5)
    resized = variants["aspect_preserving_resize"]

    assert variants["current_crop"].shape[1] > raw.shape[1]
    assert round(resized.shape[1] / resized.shape[0], 2) == round(20 / 30, 2)


def test_safe_padding_adds_margin_when_ink_touches_crop_edges():
    raw = np.full((30, 50, 3), 255, dtype=np.uint8)
    raw[0:2, 5:12] = 0  # mực sát mép trên, không phủ ngang phần lớn dòng
    raw[10:22, 47:50] = 0  # nét sát mép phải
    variants = build_crop_variants(raw, raw.copy())

    assert variants["safe_padding"].shape[0] > raw.shape[0]
    assert variants["safe_padding"].shape[1] > raw.shape[1]


def test_cli_full_flow_writes_crop_predictions_times_json_and_zip(monkeypatch):
    from app.core.vision_preprocess import clean_crop_borders, pad_small_crop_for_vietocr

    class FakeModel:
        def predict(self, image, return_prob=True):
            assert isinstance(image, Image.Image)
            return f"variant-{image.width}x{image.height}", 0.75

    class FakeEngine:
        def __init__(self, lang):
            self._vietocr = FakeModel()

        def recognize(self, image_path, **kwargs):
            observer = kwargs["diagnostic_observer"]
            raw = _synthetic_crop(height=40, width=120)
            cleaned = clean_crop_borders(raw)
            current_input = pad_small_crop_for_vietocr(cleaned)
            common = {"crop_id": 12, "polygon_source": [[10, 10], [130, 10], [130, 50], [10, 50]],
                      "bbox_source": [10, 10, 130, 50]}
            observer({"event": "preprocessed_rectified_crop", **common}, raw)
            observer({"event": "vietocr_border_cleaned_crop", **common}, cleaned)
            observer({"event": "vietocr_recognizer_input", "subcrop_index": None, **common}, current_input)
            observer({"event": "recognition_candidate", "recognizer": "vietocr:test",
                      "text": "production output", **common})
            return object()

    monkeypatch.setattr("app.core.ocr_engine.OCREngine", FakeEngine)
    with tempfile.TemporaryDirectory(prefix="tdp-crop-ablation-cli-") as temporary:
        root = Path(temporary)
        image_path = root / "source image.png"
        image_path.write_bytes(b"test input placeholder")
        output_dir = root / "ablation output"
        monkeypatch.setattr(sys, "argv", [
            "ocr_crop_ablation.py", "--images", str(image_path), "--output-dir", str(output_dir)
        ])

        assert ocr_crop_ablation.main() == 0

        report_path = output_dir / "source_image" / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        region = report["regions"][0]
        assert region["production_vietocr_text"] == "production output"
        assert region["current_crop_matches_observed_input_pixels"] is True
        assert len(region["variants"]) == 4
        assert all(row["text"].startswith("variant-") for row in region["variants"])
        assert all(row["inference_seconds"] >= 0 and row["transform_seconds"] >= 0
                   for row in region["variants"])
        assert all((report_path.parent / row["image"]).is_file() for row in region["variants"])

        archive = output_dir.with_suffix(".zip")
        with zipfile.ZipFile(archive) as bundle:
            names = bundle.namelist()
            assert any(name.endswith("report.json") for name in names)
            assert sum(name.endswith(".png") for name in names) >= 6
            archived_report = next(name for name in names if name.endswith("report.json"))
            zipped = json.loads(bundle.read(archived_report))
            assert zipped["regions"][0]["variants"][0]["text"].startswith("variant-")


def test_selective_rerun_only_flags_suspicious_region_and_obeys_call_budget(tmp_path):
    from app.core.vision_preprocess import pad_small_crop_for_vietocr

    class FakeModel:
        calls = 0

        def predict(self, image, return_prob=True):
            self.calls += 1
            return f"rerun-{self.calls}", 0.6

    class FakeEngine:
        def __init__(self):
            self._vietocr = FakeModel()

        def recognize(self, image_path, diagnostic_observer):
            raw = np.full((32, 100, 3), 255, dtype=np.uint8)
            clean = raw.copy()
            current = pad_small_crop_for_vietocr(clean)
            for crop_id, text, paddle_text in ((3, "ordinary text", "ordinary text"),
                                               (8, "uncertain? text", "other candidate")):
                common = {"crop_id": crop_id, "polygon_source": [[1, 1], [101, 1], [101, 33], [1, 33]],
                          "bbox_source": [1, 1, 101, 33]}
                diagnostic_observer({"event": "preprocessed_rectified_crop", **common}, raw)
                diagnostic_observer({"event": "vietocr_border_cleaned_crop", **common}, clean)
                diagnostic_observer({"event": "vietocr_recognizer_input", "subcrop_index": None, **common}, current)
                diagnostic_observer({"event": "recognition_candidate", "recognizer": "vietocr:test",
                                     "text": text, "confidence": 0.8,
                                     "paddle_candidate_mapping": {"status": "matched"}, **common})
                diagnostic_observer({"event": "recognition_candidate", "recognizer": "paddleocr",
                                     "text": paddle_text, "confidence": 0.95,
                                     "paddle_candidate_mapping": {"status": "matched"}, **common})
            return object()

    engine = FakeEngine()
    source = tmp_path / "source.png"
    report = ocr_selective_rerecognition.run_selective(
        engine, source, tmp_path / "result", max_regions=5, max_extra_calls=2,
        max_extra_seconds=30.0,
    )

    assert report["production_result_unchanged"] is True
    assert report["automatic_candidate_selection"] is False
    assert report["selected_candidate"] is None
    assert report["summary"]["regions_flagged"] == 1
    assert report["summary"]["regions_not_flagged_unchanged"] == 1
    assert report["additional_vietocr_calls"] == 2
    assert report["additional_paddleocr_calls"] == 0
    region = report["regions"][0]
    assert region["crop_id"] == 8
    assert "verified_same_region_model_disagreement" in region["risk_reasons"]
    assert all(candidate["provenance"] == "experimental_vietocr_rerecognition"
               for candidate in region["rerecognition_candidates"])
    assert len(region["rerecognition_candidates"]) <= 3
