# Công cụ annotation Ground Truth

Công cụ độc lập này dùng manifest do `scripts/ocr_quality_benchmark.py init` tạo ra. Nó không chạy OCR và không điền nội dung từ VietOCR/PaddleOCR.

## Mở công cụ

Trong PowerShell, dùng venv của dự án:

```powershell
python scripts/ground_truth_annotator.py `
  --manifest "$env:LOCALAPPDATA\Tool-Documents-Pro\benchmark\manifest.json"
```

Bỏ `--manifest` để chọn tệp qua hộp thoại. Ảnh hiển thị ở bên trái; kéo chuột để khoanh vùng và xem crop. Chọn một khối đoạn hoặc ô bảng rồi bấm **Gắn vùng ảnh đã khoanh vào khối/ô** để lưu tọa độ vùng nguồn. Các bbox được lưu theo pixel ảnh gốc.

## Nhập theo thứ tự đọc

- Dùng **+ Đoạn** để thêm tiêu đề, đoạn văn hoặc chú thích; sửa thứ tự bằng **Lên/Xuống**.
- Dùng **+ Bảng** để nhập lưới. Mỗi hàng/cột giữ nguyên ô trống; phân cách hàng là xuống dòng, phân cách ô là tab. Không điền nội dung chưa đọc rõ.
- Chọn vùng ô chữ nhật và bấm **Gộp ô rỗng** để ghi colspan/rowspan. Chỉ gộp khi các ô bị che rỗng để tránh xóa nội dung.
- Chọn ô rồi đặt hướng đường chéo nếu đường kẻ đó thực sự có trong ảnh. Header nhiều tầng được biểu diễn bằng nhiều hàng và các ô gộp tương ứng.
- Bấm **Đánh dấu khối đã đối chiếu** sau khi kiểm tra từng khối với ảnh. Chỉ sau đó, điền người kiểm tra/nguồn và bấm **Xác minh Ground Truth của tài liệu**. Đây là thao tác duy nhất bật `verified=true`.

Chuỗi đo CER/WER được tạo từ thứ tự khối do người dùng sắp xếp. Bên trong bảng, quy ước là hàng từ trên xuống, ô trái sang phải, ô phân cách bằng tab; hàng rỗng và ô rỗng vẫn được giữ. Vì phép đo chuỗi tuyến tính có thể tính lỗi thứ tự vùng thành lỗi ký tự, báo cáo cần xem thêm các khối và lưới bảng. Chưa có chỉ số thứ tự đọc độc lập.

Manifest benchmark hiện hỗ trợ một bảng có ground truth cho mỗi tài liệu. Annotation nhiều bảng trong cùng một tài liệu chưa được hỗ trợ; hãy tách mẫu hoặc chỉ đánh giá bảng chính trong lượt benchmark này. Các nhãn span dùng chỉ số hàng/cột bắt đầu từ 0 và biên cuối không bao gồm, tương thích `TableCell`/`table_evaluator.py`.

## Lưu an toàn và đưa vào benchmark

Nút **Xuất manifest…** ghi manifest theo kiểu nguyên tử. Nếu tệp đích đã chứa Ground Truth xác minh, công cụ sẽ hỏi xác nhận trước khi thay thế. Khi chỉnh sửa một tài liệu đã xác minh, nội dung sẽ trở thành bản nháp cho đến khi người dùng xác minh lại.

Sau khi xác minh, chạy benchmark như thường lệ:

```powershell
python scripts/ocr_quality_benchmark.py run `
  --manifest "$env:LOCALAPPDATA\Tool-Documents-Pro\benchmark\manifest.json" `
  --version baseline
```

`expected_grid`, cell spans, ô chéo và dòng ngoài bảng được ghi vào trường `ground_truth.table`; trường `ground_truth.text` dùng quy ước tuần tự hóa trên. Việc nhập nhãn định dạng chỉ là bước chuẩn bị; benchmark hiện chưa chấm điểm định dạng Word.

Không có nội dung Ground Truth nào được xác minh tự động. Bản manifest đang có chỉ được thay thế nếu người dùng xác nhận rõ trong hộp thoại ghi đè.
