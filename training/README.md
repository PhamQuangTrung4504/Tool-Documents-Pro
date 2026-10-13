# 🏋️ Training — Huấn luyện mô hình OCR tuỳ chỉnh

> Hướng dẫn huấn luyện (fine-tune) mô hình PP-OCRv4 với dữ liệu riêng,
> sử dụng PaddleOCR Training Framework.

---

## 📋 Mục lục

1. [Yêu cầu](#-yêu-cầu)
2. [Chuẩn bị dữ liệu](#-chuẩn-bị-dữ-liệu)
3. [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
4. [Huấn luyện](#-huấn-luyện)
5. [Đánh giá](#-đánh-giá)
6. [Xuất mô hình](#-xuất-mô-hình)

---

## 💻 Yêu cầu

| Thành phần          | Yêu cầu                                    |
| ------------------- | ------------------------------------------ |
| GPU                 | NVIDIA GPU ≥ 4 GB VRAM (khuyến nghị 8 GB)  |
| CUDA                | 11.6 / 11.7 / 11.8                         |
| cuDNN               | ≥ 8.4                                      |
| PaddlePaddle-GPU    | ≥ 2.6.0                                    |
| PaddleOCR           | ≥ 2.9.0                                    |

```bash
# Cài PaddlePaddle GPU
pip install paddlepaddle-gpu==2.6.2

# Clone PaddleOCR repo (để dùng tools huấn luyện)
git clone https://github.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR
pip install -r requirements.txt
```

---

## 📂 Chuẩn bị dữ liệu

### Bước 1: Thu thập ảnh

Ảnh huấn luyện nên bao gồm:
- Ảnh tài liệu (hợp đồng, hoá đơn, giấy tờ, …)
- Ảnh chụp từ camera / điện thoại
- Ảnh scan chất lượng cao và thấp
- Ảnh chứa chữ Việt Nam (có dấu đầy đủ)

### Bước 2: Gán nhãn

Sử dụng [PPOCRLabel](https://github.com/PFCCLab/PPOCRLabel) để gán nhãn:

```bash
pip install PPOCRLabel
PPOCRLabel --lang vi
```

### Bước 3: Chạy script chuẩn bị

```bash
python prepare_dataset.py \
    --input_dir ./raw_images \
    --output_dir ./dataset \
    --train_ratio 0.85 \
    --val_ratio 0.10 \
    --test_ratio 0.05 \
    --format lmdb
```

---

## 📁 Cấu trúc thư mục

Sau khi chạy `prepare_dataset.py`, thư mục sẽ có dạng:

```
training/
├── dataset/
│   ├── train/
│   │   ├── images/
│   │   └── label.txt        # image_path \t text
│   ├── val/
│   │   ├── images/
│   │   └── label.txt
│   └── test/
│       ├── images/
│       └── label.txt
├── pretrained/               # Model pretrained PP-OCRv4
│   └── vi_PP-OCRv4_rec_train/
├── output/                   # Kết quả huấn luyện
├── prepare_dataset.py
└── config_rec_v4.yml
```

---

## 🚀 Huấn luyện

### Text Recognition (Nhận dạng chữ)

```bash
python -m paddle.distributed.launch \
    --gpus '0' \
    PaddleOCR/tools/train.py \
    -c config_rec_v4.yml
```

### Text Detection (Phát hiện vùng chữ)

Nếu cần fine-tune detection, tham khảo:
- [PP-OCRv4 Detection](https://github.com/PaddlePaddle/PaddleOCR/blob/main/configs/det/)

---

## 📊 Đánh giá

```bash
python PaddleOCR/tools/eval.py \
    -c config_rec_v4.yml \
    -o Global.checkpoints=./output/rec_ppocr_v4/best_accuracy
```

---

## 📦 Xuất mô hình

```bash
# Xuất inference model
python PaddleOCR/tools/export_model.py \
    -c config_rec_v4.yml \
    -o Global.pretrained_model=./output/rec_ppocr_v4/best_accuracy \
       Global.save_inference_dir=./output/inference/rec

# Copy vào thư mục app
cp -r ./output/inference/rec ../app/models/custom_rec/
```

Sau đó cập nhật `ocr_engine.py` để sử dụng model custom:

```python
engine = PaddleOCR(
    rec_model_dir="app/models/custom_rec",
    ...
)
```

---

## 📝 Ghi chú

- **Epoch khuyến nghị**: 100–500 epoch cho fine-tune, 50–100 nếu dữ liệu ít.
- **Batch size**: Tuỳ VRAM. 4 GB → batch 32; 8 GB → batch 64–128.
- **Learning rate**: Bắt đầu 0.0005, giảm dần (cosine decay).
- **Augmentation**: Đã cấu hình sẵn trong `config_rec_v4.yml`.
