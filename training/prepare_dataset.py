#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
training/prepare_dataset.py
============================
Script chuẩn bị dữ liệu huấn luyện cho PP-OCRv4 Recognition.

Chức năng:
- Đọc ảnh + label từ thư mục đầu vào (PPOCRLabel format).
- Chia train / val / test theo tỷ lệ tuỳ chỉnh.
- Tạo label file (format: image_path \\t text).
- Tuỳ chọn: Tạo LMDB dataset cho tốc độ I/O cao hơn.
- Kiểm tra và thống kê dữ liệu.

Usage:
    python prepare_dataset.py --input_dir ./raw_images --output_dir ./dataset
    python prepare_dataset.py --input_dir ./raw_images --output_dir ./dataset --format lmdb
    python prepare_dataset.py --help
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Hằng số
# ---------------------------------------------------------------------------

SUPPORTED_IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
DEFAULT_TRAIN_RATIO = 0.85
DEFAULT_VAL_RATIO = 0.10
DEFAULT_TEST_RATIO = 0.05
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Đọc label file (PPOCRLabel format hoặc plain text)
# ---------------------------------------------------------------------------


def load_labels(label_path: str) -> List[Tuple[str, str]]:
    """
    Đọc file label.

    Hỗ trợ 2 format:
      1. PPOCRLabel: image_path \\t [{"transcription": "text", ...}]
      2. Plain text: image_path \\t text

    Returns:
        List of (image_path, text) tuples.
    """
    import json

    samples: List[Tuple[str, str]] = []
    label_dir = os.path.dirname(label_path)

    with open(label_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t", maxsplit=1)
            if len(parts) != 2:
                print(
                    f"  ⚠️  Bỏ qua dòng {line_num}: "
                    f"không đúng format (thiếu tab)"
                )
                continue

            img_path_raw, label_raw = parts

            # Xử lý đường dẫn ảnh
            img_path = img_path_raw.strip()
            if not os.path.isabs(img_path):
                img_path = os.path.join(label_dir, img_path)
            img_path = os.path.normpath(img_path)

            # Xử lý label
            label_raw = label_raw.strip()

            # Thử parse JSON (PPOCRLabel format)
            text = ""
            try:
                annotations = json.loads(label_raw)
                if isinstance(annotations, list):
                    texts = []
                    for ann in annotations:
                        if isinstance(ann, dict) and "transcription" in ann:
                            t = ann["transcription"].strip()
                            if t and t != "###":
                                texts.append(t)
                    text = " ".join(texts)
                elif isinstance(annotations, str):
                    text = annotations.strip()
            except (json.JSONDecodeError, TypeError):
                # Plain text format
                text = label_raw

            if not text:
                continue

            samples.append((img_path, text))

    return samples


# ---------------------------------------------------------------------------
# Chia dữ liệu
# ---------------------------------------------------------------------------


def split_dataset(
    samples: List[Tuple[str, str]],
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    seed: int = RANDOM_SEED,
) -> Dict[str, List[Tuple[str, str]]]:
    """
    Chia dữ liệu thành train / val / test.

    Returns:
        Dict với key "train", "val", "test".
    """
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Tổng tỷ lệ phải bằng 1.0, nhận được {total:.4f}"
        )

    random.seed(seed)
    shuffled = samples.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }

    return splits


# ---------------------------------------------------------------------------
# Tạo dataset (file-based)
# ---------------------------------------------------------------------------


def create_file_dataset(
    splits: Dict[str, List[Tuple[str, str]]],
    output_dir: str,
    copy_images: bool = True,
) -> None:
    """
    Tạo dataset dạng file: mỗi split có thư mục images/ và label.txt.
    """
    for split_name, samples in splits.items():
        split_dir = os.path.join(output_dir, split_name)
        img_dir = os.path.join(split_dir, "images")
        os.makedirs(img_dir, exist_ok=True)

        label_path = os.path.join(split_dir, "label.txt")

        with open(label_path, "w", encoding="utf-8") as f:
            for idx, (src_path, text) in enumerate(
                tqdm(samples, desc=f"  {split_name}", unit="img")
            ):
                if not os.path.isfile(src_path):
                    print(f"  ⚠️  Không tìm thấy: {src_path}")
                    continue

                ext = os.path.splitext(src_path)[1].lower()
                dst_name = f"{idx:06d}{ext}"
                dst_path = os.path.join(img_dir, dst_name)

                if copy_images:
                    shutil.copy2(src_path, dst_path)

                rel_path = os.path.join("images", dst_name)
                f.write(f"{rel_path}\t{text}\n")

        print(f"  ✅ {split_name}: {len(samples)} samples → {split_dir}")


# ---------------------------------------------------------------------------
# Tạo LMDB dataset (tuỳ chọn, I/O nhanh hơn)
# ---------------------------------------------------------------------------


def create_lmdb_dataset(
    splits: Dict[str, List[Tuple[str, str]]],
    output_dir: str,
) -> None:
    """
    Tạo dataset dạng LMDB cho mỗi split.
    """
    try:
        import lmdb
    except ImportError:
        print("  ❌ Thư viện lmdb chưa được cài. Chạy: pip install lmdb")
        print("  → Chuyển sang tạo file-based dataset …")
        create_file_dataset(splits, output_dir)
        return

    for split_name, samples in splits.items():
        lmdb_dir = os.path.join(output_dir, f"{split_name}_lmdb")
        os.makedirs(lmdb_dir, exist_ok=True)

        # Ước tính kích thước LMDB (trung bình 500KB/ảnh)
        map_size = max(len(samples) * 500 * 1024, 10 * 1024 * 1024)

        env = lmdb.open(lmdb_dir, map_size=map_size)
        cache: Dict[bytes, bytes] = {}
        count = 0

        for idx, (src_path, text) in enumerate(
            tqdm(samples, desc=f"  {split_name}_lmdb", unit="img")
        ):
            if not os.path.isfile(src_path):
                continue

            with open(src_path, "rb") as img_f:
                img_bin = img_f.read()

            # Keys theo format PaddleOCR LMDB
            image_key = f"image-{count + 1:09d}".encode()
            label_key = f"label-{count + 1:09d}".encode()

            cache[image_key] = img_bin
            cache[label_key] = text.encode("utf-8")
            count += 1

            # Flush mỗi 1000 records
            if count % 1000 == 0:
                with env.begin(write=True) as txn:
                    for k, v in cache.items():
                        txn.put(k, v)
                cache.clear()

        # Flush remaining
        cache[b"num-samples"] = str(count).encode()
        with env.begin(write=True) as txn:
            for k, v in cache.items():
                txn.put(k, v)

        env.close()
        print(f"  ✅ {split_name}_lmdb: {count} samples → {lmdb_dir}")


# ---------------------------------------------------------------------------
# Thống kê dữ liệu
# ---------------------------------------------------------------------------


def print_statistics(
    splits: Dict[str, List[Tuple[str, str]]],
) -> None:
    """In thống kê chi tiết về dữ liệu."""
    print("\n" + "=" * 60)
    print("📊  THỐNG KÊ DỮ LIỆU")
    print("=" * 60)

    total = 0
    for name, samples in splits.items():
        total += len(samples)

    for name, samples in splits.items():
        pct = len(samples) / total * 100 if total > 0 else 0
        lengths = [len(text) for _, text in samples]
        avg_len = sum(lengths) / len(lengths) if lengths else 0
        max_len = max(lengths) if lengths else 0
        min_len = min(lengths) if lengths else 0

        print(f"\n  📁 {name.upper()}")
        print(f"     Số lượng : {len(samples):,} ({pct:.1f}%)")
        print(f"     Độ dài TB: {avg_len:.1f} ký tự")
        print(f"     Ngắn nhất: {min_len} ký tự")
        print(f"     Dài nhất : {max_len} ký tự")

    print(f"\n  📊 Tổng cộng: {total:,} samples")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Scan thư mục ảnh không có label
# ---------------------------------------------------------------------------


def scan_unlabeled_images(input_dir: str) -> List[str]:
    """
    Quét thư mục tìm tất cả ảnh (dùng khi chưa có file label).
    """
    images: List[str] = []
    for root, dirs, files in os.walk(input_dir):
        for fname in sorted(files):
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_IMG_EXT:
                images.append(os.path.join(root, fname))
    return images


# ---------------------------------------------------------------------------
# Validate dataset
# ---------------------------------------------------------------------------


def validate_dataset(output_dir: str) -> None:
    """Kiểm tra dataset đã tạo."""
    print("\n🔍  Kiểm tra dataset …")

    for split in ["train", "val", "test"]:
        label_path = os.path.join(output_dir, split, "label.txt")
        if not os.path.isfile(label_path):
            print(f"  ❌ Thiếu: {label_path}")
            continue

        count = 0
        missing = 0
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t", maxsplit=1)
                if len(parts) == 2:
                    img_rel = parts[0]
                    img_abs = os.path.join(output_dir, split, img_rel)
                    if not os.path.isfile(img_abs):
                        missing += 1
                    count += 1

        status = "✅" if missing == 0 else "⚠️"
        print(
            f"  {status} {split}: {count} entries, "
            f"{missing} missing images"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chuẩn bị dữ liệu huấn luyện OCR cho PP-OCRv4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python prepare_dataset.py --input_dir ./raw_images --output_dir ./dataset
  python prepare_dataset.py --input_dir ./raw_images --output_dir ./dataset --format lmdb
  python prepare_dataset.py --input_dir ./raw_images --output_dir ./dataset --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1
        """,
    )
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Thư mục chứa ảnh gốc và file label (Label.txt hoặc rec_gt.txt)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./dataset",
        help="Thư mục đầu ra (mặc định: ./dataset)",
    )
    parser.add_argument(
        "--train_ratio", type=float, default=DEFAULT_TRAIN_RATIO,
        help=f"Tỷ lệ train (mặc định: {DEFAULT_TRAIN_RATIO})",
    )
    parser.add_argument(
        "--val_ratio", type=float, default=DEFAULT_VAL_RATIO,
        help=f"Tỷ lệ validation (mặc định: {DEFAULT_VAL_RATIO})",
    )
    parser.add_argument(
        "--test_ratio", type=float, default=DEFAULT_TEST_RATIO,
        help=f"Tỷ lệ test (mặc định: {DEFAULT_TEST_RATIO})",
    )
    parser.add_argument(
        "--format", type=str, choices=["file", "lmdb"], default="file",
        help="Định dạng output: file (mặc định) hoặc lmdb",
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help=f"Random seed (mặc định: {RANDOM_SEED})",
    )
    parser.add_argument(
        "--no_copy", action="store_true",
        help="Không copy ảnh (chỉ tạo label file với đường dẫn tuyệt đối)",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("=" * 60)
    print("🏋️  CHUẨN BỊ DỮ LIỆU HUẤN LUYỆN OCR")
    print("=" * 60)
    print(f"  Input : {os.path.abspath(args.input_dir)}")
    print(f"  Output: {os.path.abspath(args.output_dir)}")
    print(
        f"  Split : train={args.train_ratio:.0%} / "
        f"val={args.val_ratio:.0%} / test={args.test_ratio:.0%}"
    )
    print(f"  Format: {args.format}")
    print()

    # --- Tìm file label ---
    input_dir = os.path.abspath(args.input_dir)
    if not os.path.isdir(input_dir):
        print(f"❌ Thư mục không tồn tại: {input_dir}")
        return 1

    label_candidates = [
        "Label.txt", "label.txt", "rec_gt.txt",
        "labels.txt", "gt.txt", "annotation.txt",
    ]

    label_file: Optional[str] = None
    for candidate in label_candidates:
        path = os.path.join(input_dir, candidate)
        if os.path.isfile(path):
            label_file = path
            break

    if label_file is None:
        print("⚠️  Không tìm thấy file label.")
        print("   Tìm kiếm các file ảnh chưa gán nhãn …")

        images = scan_unlabeled_images(input_dir)
        if not images:
            print(f"❌ Không tìm thấy ảnh nào trong {input_dir}")
            return 1

        print(f"   Tìm thấy {len(images)} ảnh chưa gán nhãn.")
        print("   → Hãy gán nhãn bằng PPOCRLabel trước:")
        print("     pip install PPOCRLabel")
        print(f"     PPOCRLabel --lang vi")
        return 1

    print(f"📄 File label: {label_file}")

    # --- Đọc label ---
    print("\n📖 Đọc dữ liệu …")
    samples = load_labels(label_file)
    print(f"   Tổng: {len(samples)} samples")

    if len(samples) == 0:
        print("❌ Không có dữ liệu hợp lệ!")
        return 1

    # --- Kiểm tra ảnh tồn tại ---
    print("\n🔍 Kiểm tra ảnh …")
    valid_samples: List[Tuple[str, str]] = []
    missing_count = 0

    for img_path, text in tqdm(samples, desc="  Checking", unit="img"):
        if os.path.isfile(img_path):
            valid_samples.append((img_path, text))
        else:
            missing_count += 1

    if missing_count > 0:
        print(f"  ⚠️  {missing_count} ảnh không tìm thấy (đã bỏ qua)")

    print(f"  ✅ {len(valid_samples)} ảnh hợp lệ")

    if len(valid_samples) == 0:
        print("❌ Không có ảnh hợp lệ!")
        return 1

    # --- Chia dữ liệu ---
    print("\n✂️  Chia dữ liệu …")
    splits = split_dataset(
        valid_samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    # --- Tạo dataset ---
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n📦 Tạo dataset ({args.format}) …")

    if args.format == "lmdb":
        create_lmdb_dataset(splits, output_dir)
        # Vẫn tạo file-based để dùng cho eval
        create_file_dataset(splits, output_dir, copy_images=not args.no_copy)
    else:
        create_file_dataset(splits, output_dir, copy_images=not args.no_copy)

    # --- Thống kê ---
    print_statistics(splits)

    # --- Validate ---
    if not args.no_copy:
        validate_dataset(output_dir)

    print("\n🎉 Hoàn thành!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
