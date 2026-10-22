# Tái nhận dạng OCR có chọn lọc

`scripts/ocr_selective_rerecognition.py` chạy OCR production một lần, thu crop qua `diagnostic_observer`, rồi chỉ thử lại các vùng có tín hiệu đáng nghi. Đây là công cụ thử nghiệm riêng; không sửa `OCRResult`, không cập nhật `StructuredDocumentEditor`, và không chọn candidate tự động.

## Tín hiệu ưu tiên

- VietOCR và PaddleOCR khác chuỗi trên vùng có ánh xạ Paddle đã xác minh.
- Ánh xạ polygon Paddle chưa xác minh (chỉ khi có candidate Paddle).
- Ký tự `?` hoặc ký tự thay thế trong kết quả chính.
- Crop có chỉ dấu mực sát mép, crop nhỏ/hẹp, hoặc confidence VietOCR thấp/không có.

Các tín hiệu chỉ xếp thứ tự kiểm tra; không chứng minh ứng viên nào đúng. Confidence PaddleOCR không được so trực tiếp với VietOCR. Báo cáo giữ candidate production, candidate thử lại, crop source, bbox/polygon, lý do chọn, chất lượng crop và ký tự ngoài từ vựng VietOCR.

## Ngân sách

Mặc định tối đa 12 vùng, 36 lượt VietOCR bổ sung, 3 lượt cho mỗi vùng và 30 giây cho giai đoạn thử lại. Giới hạn thời gian được kiểm tra giữa các lần gọi model; một lần `predict` đang chạy không thể bị ngắt giữa chừng. PaddleOCR không được gọi thêm; candidate Paddle đã có từ lượt production được lưu làm bằng chứng. Các vùng đã tách thành nhiều sub-crop được bỏ qua để tránh ghép sai danh tính crop.

## Chạy trên Windows PowerShell

```powershell
.\venv\Scripts\python.exe scripts\ocr_selective_rerecognition.py `
  --images `
  "assets\imagetest\bang1.png" `
  "assets\imagetest\bang2.png" `
  "assets\imagetest\vanban1.png"
```

Kết quả JSON, ảnh crop và ZIP được ghi ngoài repository vào thư mục TEMP. Có thể giới hạn chi phí rõ hơn:

```powershell
.\venv\Scripts\python.exe scripts\ocr_selective_rerecognition.py `
  --images "D:\OCR Samples\document.png" `
  --max-regions 5 --max-extra-calls 10 --max-extra-seconds 15 `
  --output-dir "D:\OCR Results\selective"
```

CER/WER chỉ được tính nếu truyền nhãn vùng đã được người dùng xác minh qua `--region-ground-truth`. Định dạng sidecar giống phần nhãn vùng trong [ocr_crop_ablation.md](ocr_crop_ablation.md): ảnh, `bbox_source`, văn bản, `verified`, người/date/nguồn kiểm chứng. Khớp vùng dựa trên IoU bbox duy nhất, không dựa trên crop ID.

## Diễn giải

Candidate thử lại chỉ là bằng chứng để người dùng rà soát. `selected_candidate` luôn là `null`; mọi khác biệt, kể cả candidate lặp lại giống nhau, không được xem là ground truth. Thời gian được ghi riêng cho OCR production và các lượt bổ sung.
