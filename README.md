# 📄 Tool-Documents-Pro — OCR Desktop Application

> Ứng dụng Desktop Windows nhận dạng chữ từ ảnh (OCR) sử dụng PaddleOCR + PyQt5.  
> Hỗ trợ nhận dạng đa ngôn ngữ: **Tiếng Việt**, Tiếng Anh, Tiếng Trung, Tiếng Nhật, Tiếng Hàn.

---

## 🖥️ Yêu cầu hệ thống

| Thành phần     | Yêu cầu                   |
| -------------- | ------------------------- |
| Hệ điều hành   | Windows 10 / 11 (64-bit)  |
| Python         | 3.9 — 3.11                |
| RAM            | ≥ 4 GB (khuyến nghị 8 GB) |
| GPU (tuỳ chọn) | NVIDIA CUDA 11.x trở lên  |

---

## 📦 Cài đặt

```bash
# 1. Clone hoặc tải về project
git clone <repo-url>
cd Tool-Documents-Pro

# 2. Tạo môi trường ảo
python -m venv venv
venv\Scripts\activate

# 3. Cài đặt thư viện
pip install vietocr --no-deps ; pip install "einops>=0.6.0" "albumentations>=1.4.3" "Pillow>=10.4.0" ; pip install --no-deps -r requirements.txt ; pip install paddlepaddle paddleocr torch torchvision pymupdf opencv-python numpy scikit-image PyQt5 PyQt5-sip pyperclip loguru gdown symspellpy pyyaml tqdm lmdb python-docx

pip install -r requirements.txt(ko dùng)

-------

# 4. Thoát khỏi môi trường ảo (Không chọn)
deactivate

# 4. Xóa môi trường ảo hiện tại (Không chọn)
Remove-Item -Recurse -Force venv
```

---

## 🚀 Khởi chạy

```bash
python run.py
```

Ứng dụng sẽ mở cửa sổ chính với giao diện đồ hoạ cho phép:

1. **Mở ảnh** — chọn file ảnh từ máy tính (PNG, JPG, BMP, TIFF, WebP).
2. **Dán ảnh** — dán ảnh từ clipboard (Ctrl+V hoặc nút Paste).
3. **Kéo thả** — kéo thả file ảnh trực tiếp vào cửa sổ.
4. **Nhận dạng** — bấm nút OCR hoặc phím tắt Ctrl+R để nhận dạng.
5. **Sao chép kết quả** — bấm nút Copy hoặc Ctrl+C để sao chép văn bản.
6. **Xuất file** — lưu kết quả ra file TXT.

---

## 🗂️ Cấu trúc dự án

```
Tool-Documents-Pro/
├── requirements.txt          # Danh sách thư viện
├── README.md                 # Tài liệu hướng dẫn
├── run.py                    # Điểm khởi chạy chính
├── app/
│   ├── __init__.py           # Package init
│   ├── core/
│   │   ├── __init__.py
│   │   └── ocr_engine.py    # Engine nhận dạng (PaddleOCR)
│   └── ui/
│       ├── __init__.py
│       └── main_window.py   # Giao diện chính (PyQt5)
└── training/
    ├── README.md             # Hướng dẫn huấn luyện
    ├── prepare_dataset.py    # Script chuẩn bị dữ liệu
    └── config_rec_v4.yml     # Cấu hình huấn luyện PP-OCRv4
```

---

## ⌨️ Phím tắt

| Phím tắt       | Chức năng                        |
| -------------- | -------------------------------- |
| `Ctrl+O`       | Mở file ảnh                      |
| `Ctrl+V`       | Dán ảnh từ clipboard             |
| `Ctrl+R`       | Chạy nhận dạng OCR               |
| `Ctrl+C`       | Sao chép kết quả vào clipboard   |
| `Ctrl+S`       | Lưu kết quả ra file TXT          |
| `Ctrl+Shift+C` | Sao chép kết quả (giữ định dạng) |
| `Ctrl+Q`       | Thoát ứng dụng                   |

---

## 🌐 Ngôn ngữ hỗ trợ

PaddleOCR hỗ trợ **80+ ngôn ngữ**. Ứng dụng mặc định cấu hình:

- `vi` — Tiếng Việt
- `en` — Tiếng Anh
- `ch` — Tiếng Trung (Giản thể)
- `japan` — Tiếng Nhật
- `korean` — Tiếng Hàn

Thay đổi ngôn ngữ qua dropdown trên giao diện.

---

## 🏋️ Huấn luyện tuỳ chỉnh

Xem hướng dẫn chi tiết tại [`training/README.md`](training/README.md).

---

## 📝 Giấy phép

MIT License — Sử dụng tự do cho mục đích cá nhân và thương mại.
