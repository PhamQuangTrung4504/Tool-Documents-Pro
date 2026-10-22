from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from app.core.ocr_benchmark import (
    compare_runs,
    compute_symbol_accuracy,
    text_metrics,
    validate_manifest,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = REPOSITORY_ROOT / "scripts" / "ocr_quality_benchmark.py"


def _run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CLI_PATH), *arguments],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_init_run_and_compare_accept_windows_paths_with_spaces(tmp_path: Path):
    image_dir = tmp_path / "input images with spaces"
    image_dir.mkdir()
    image = image_dir / "scan page 01.png"
    image.write_bytes(b"placeholder image bytes; init only records the path")
    data_dir = tmp_path / "benchmark data with spaces"
    manifest_path = data_dir / "manual ground truth.json"

    init = _run_cli(
        "init", "--images", str(image), "--output-manifest", str(manifest_path)
    )
    assert init.returncode == 0, init.stderr
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["documents"][0]["image"] == str(image.resolve())

    # Manifest chưa được xác minh: CLI phải dừng trước khi khởi tạo model OCR.
    # Đồng thời kiểm tra parse và chuẩn hóa --manifest, --output-dir có khoảng trắng.
    run = _run_cli(
        "run", "--manifest", str(manifest_path), "--version", "baseline",
        "--output-dir", str(data_dir / "run output with spaces"),
    )
    assert run.returncode == 2
    assert "verified" in run.stderr
    assert "AttributeError" not in run.stderr

    common_record = {
        "id": "doc_0001",
        "image_sha256": "image-hash",
        "ground_truth_sha256": "verified-ground-truth-hash",
        "error_groups": [],
        "cer": 0.0,
        "wer": 0.0,
        "symbol_accuracy": None,
        "ocr_seconds": 0.5,
        "table_metrics": None,
    }
    baseline = data_dir / "baseline result.json"
    candidate = data_dir / "candidate result.json"
    baseline.write_text(json.dumps({"version": "baseline", "documents": [common_record]}), encoding="utf-8")
    candidate.write_text(json.dumps({"version": "candidate", "documents": [common_record]}), encoding="utf-8")
    comparison_output = data_dir / "comparison output.json"
    compare = _run_cli(
        "compare", "--baseline", str(baseline), "--candidate", str(candidate),
        "--output", str(comparison_output),
    )
    assert compare.returncode == 0, compare.stderr
    report = json.loads(comparison_output.read_text(encoding="utf-8"))
    assert report["document_count"] == 1


def test_text_metrics_use_unicode_normalization_and_preserve_errors():
    metrics = text_metrics("Tôi có 12+3.", "Tôi có 12-3.")
    assert metrics["cer"] > 0
    assert metrics["wer"] > 0
    assert metrics["symbol_accuracy"] < 1


def test_symbol_accuracy_counts_deletions_and_returns_unknown_without_symbols():
    assert compute_symbol_accuracy("Mã 12-A", "Mã 1?-A") == 2 / 3
    assert compute_symbol_accuracy("văn bản tiếng Việt", "văn bản tiếng Việt") is None
    assert compute_symbol_accuracy("05%", "") == 0


def test_manifest_requires_explicit_human_verified_ground_truth():
    document = {
        "id": "sample-a",
        "image": "page.png",
        "error_groups": ["digits"],
        "ground_truth": {
            "text": "Nội dung do người kiểm tra nhập",
            "verified": True,
            "reviewed_by": "reviewer",
            "reviewed_at": "2026-10-22",
            "source_note": "Đối chiếu trực tiếp ảnh gốc",
            "table": None,
        },
    }
    assert validate_manifest(document) == []
    document["ground_truth"]["verified"] = False
    assert any("verified" in error for error in validate_manifest(document))


def test_run_comparison_requires_identical_inputs_and_reports_regressions():
    def record(sample_id, cer, text_hash="same"):
        return {
            "id": sample_id,
            "image_sha256": "image",
            "ground_truth_sha256": text_hash,
            "error_groups": ["table"],
            "cer": cer,
            "wer": cer,
            "symbol_accuracy": 1.0,
            "ocr_seconds": 1.0,
            "table_metrics": {"cell_exact_match_rate": 1.0 - cer},
        }

    baseline = {"version": "baseline", "documents": [record("a", 0.1), record("b", 0.0)]}
    candidate = {"version": "trial", "documents": [record("a", 0.2), record("b", 0.0)]}
    report = compare_runs(baseline, candidate)
    assert report["regressed_documents"] == ["a"]
    assert report["documents"][0]["cer_delta_baseline_minus_candidate"] == -0.1
    assert report["baseline_summary"]["groups"]["table"]["document_count"] == 2

    candidate["documents"][0]["ground_truth_sha256"] = "different"
    try:
        compare_runs(baseline, candidate)
    except ValueError as exc:
        assert "ground truth" in str(exc)
    else:
        raise AssertionError("So sánh phải từ chối ground truth không đồng nhất")
