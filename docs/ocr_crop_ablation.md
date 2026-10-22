# Thử nghiệm ablation crop OCR

Công cụ `scripts/ocr_crop_ablation.py` chạy OCR production một lần để lấy crop và metadata qua `diagnostic_observer`, sau đó chạy VietOCR trên bốn biến thể của cùng vùng:

- `current_crop`: `clean_crop_borders()` theo luồng hiện tại, tiếp theo `pad_small_crop_for_vietocr()`.
- `skip_border_cleaning`: crop trước bước tẩy đường viền, tiếp theo cùng padding tỷ lệ hiện tại.
- `safe_padding`: crop đã tẩy viền, áp dụng `adaptive_safe_crop_padding()`, tiếp theo padding tỷ lệ hiện tại.
- `aspect_preserving_resize`: crop đã tẩy viền, resize đồng đều hai chiều theo `--resize-scale`, không kéo méo tỷ lệ.

Đầu ra gồm ảnh trước/sau xử lý, văn bản và confidence do VietOCR trả về, thời gian biến đổi và thời gian `predict`, polygon/bbox nguồn, số crop có khác biệt và tệp ZIP. Confidence chỉ được lưu để truy vết; không dùng để kết luận candidate chính xác hơn. Các dòng production đã tách thành nhiều sub-crop được đánh dấu bỏ qua để tránh so sánh sai vùng.

## Chạy trên Windows PowerShell

```powershell
.\venv\Scripts\python.exe scripts\ocr_crop_ablation.py `
  --images `
  "C:\Users\pqtru\OneDrive\Desktop\Tool-Documents-Pro\assets\imagetest\bang1.png" `
  "C:\Users\pqtru\OneDrive\Desktop\Tool-Documents-Pro\assets\imagetest\bang2.png" `
  "C:\Users\pqtru\OneDrive\Desktop\Tool-Documents-Pro\assets\imagetest\vanban1.png"
```

Kết quả mặc định được ghi vào TEMP của Windows. Có thể truyền `--output-dir "D:\OCR Results\Ablation"`; công cụ từ chối thư mục bên trong repository. `--no-preprocess`, `--no-denoise` và `--resize-scale 2.0` là tùy chọn.

## Ground truth vùng

Manifest benchmark theo tài liệu không xác định ground truth cho từng crop. Để tính CER/WER theo vùng, tạo JSON riêng sau khi đối chiếu ảnh. Mỗi mục cần bbox nguồn `[x1, y1, x2, y2]`, văn bản và nguồn xác minh. Công cụ chỉ tính khi `verified` là `true` và bbox khớp duy nhất với IoU từ 0,8 trở lên; crop ID không được dùng làm định danh bền vững.

```json
{
  "regions": [
    {
      "image": "C:\\path\\document.png",
      "bbox_source": [10, 20, 300, 60],
      "text": "Nội dung được người dùng đối chiếu",
      "verified": true,
      "reviewed_by": "reviewer",
      "reviewed_at": "2026-10-22",
      "source_note": "Đối chiếu trực tiếp với ảnh nguồn"
    }
  ]
}
```

Chạy với `--region-ground-truth "D:\OCR Results\regions.json"`. Không dùng đầu ra OCR làm nhãn. Khi không có nhãn vùng khớp duy nhất, báo cáo ghi nhận trạng thái chưa gán nhãn và không đưa ra CER/WER.

## Giới hạn diễn giải

Khác biệt giữa các chuỗi chỉ cho biết preprocessing làm thay đổi đầu ra recognizer; không cho biết biến thể nào đúng. Phép đo thời gian là thời gian tường trên máy chạy, chịu ảnh hưởng khởi động model, thiết bị và tải hệ thống. Lượt OCR chuẩn mất thời gian riêng; thời gian nhận dạng các biến thể được cộng riêng trong báo cáo. Công cụ không thay đổi `OCRResult`, cấu hình hay hành vi mặc định của ứng dụng.
