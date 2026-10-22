# Benchmark chất lượng OCR

Công cụ này đo kết quả OCR trên ảnh và ground truth do người dùng kiểm tra. Công cụ không tự tạo ground truth, không sửa nội dung nhận dạng và không thay đổi thuật toán OCR.

## 1. Tạo manifest để nhập ground truth

Trong PowerShell, từ thư mục repository và với môi trường ảo đã kích hoạt:

```powershell
python scripts/ocr_quality_benchmark.py init `
  --images "C:\duong-dan\tap-anh" `
  --output-manifest "$env:LOCALAPPDATA\Tool-Documents-Pro\benchmarks\dataset.json"
```

`--images` nhận một hoặc nhiều tệp ảnh/thư mục. Manifest chỉ tham chiếu ảnh gốc, không sao chép ảnh vào repository. Mở `dataset.json` và tự điền `ground_truth.text` sau khi đối chiếu ảnh, đặt `verified` thành `true`, ghi người kiểm tra, ngày kiểm tra và nguồn đối chiếu. Không chép `prediction_text` từ kết quả OCR vào ground truth.

Các nhãn `error_groups` là nhãn do người dùng chọn, ví dụ `vietnamese_diacritics`, `digits_symbols`, `table`, `small_text`, `blurred` hoặc `mixed`. Có thể tự định nghĩa nhóm theo bộ dữ liệu; báo cáo tổng hợp theo các nhãn này.

Để đánh giá bảng, điền tùy chọn `ground_truth.table`, gồm `category`, `expected_grid` và nếu đã được kiểm chứng `expected_cell_spans`, `expected_diagonal_cells`, `expected_non_table_lines`. Chỉ mục cấu trúc bảng được báo riêng với CER/WER. Trường `formatting_annotation` được dành cho nhãn định dạng Word; hiện công cụ chỉ ghi nhận việc có nhãn, chưa chấm điểm định dạng.

## 2. Chạy baseline và phiên bản thử nghiệm

```powershell
python scripts/ocr_quality_benchmark.py run `
  --manifest "$env:LOCALAPPDATA\Tool-Documents-Pro\benchmarks\dataset.json" `
  --version baseline `
  --revision "commit-hoac-mo-ta-baseline"

python scripts/ocr_quality_benchmark.py run `
  --manifest "$env:LOCALAPPDATA\Tool-Documents-Pro\benchmarks\dataset.json" `
  --version trial `
  --revision "commit-hoac-mo-ta-thu-nghiem"
```

Mặc định kết quả được ghi dưới `%LOCALAPPDATA%\Tool-Documents-Pro\benchmarks\<thoi-gian>`. Có thể chỉ định `--output-dir` ở vị trí dữ liệu được cấu hình; vị trí này bắt buộc nằm ngoài repository. Tệp kết quả chứa nội dung tài liệu để tính toán và phải được bảo vệ như dữ liệu nguồn.

CER/WER và độ chính xác ký hiệu được tính theo từng tài liệu và tổng hợp trung bình macro theo tài liệu/nhóm lỗi. Độ chính xác ký hiệu xét chữ số và ký hiệu Unicode thuộc nhóm dấu câu/ký hiệu; ký tự ground truth bị mất tính là sai. Nếu ground truth không có chữ số/ký hiệu, giá trị là `null` (không áp dụng). Thời gian OCR lấy từ `OCRResult.elapsed_seconds`; thời gian tái dựng bảng, nếu có nhãn bảng, được ghi riêng.

Nếu cần lưu diagnostics chi tiết về OCRBox, ứng viên và metadata, thêm `--include-diagnostics`. Chế độ này có thể ghi nội dung nhạy cảm; mặc định tắt.

## 3. So sánh hai lượt

```powershell
python scripts/ocr_quality_benchmark.py compare `
  --baseline "$env:LOCALAPPDATA\Tool-Documents-Pro\benchmarks\<baseline>\run_baseline.json" `
  --candidate "$env:LOCALAPPDATA\Tool-Documents-Pro\benchmarks\<trial>\run_trial.json" `
  --output "$env:LOCALAPPDATA\Tool-Documents-Pro\benchmarks\comparison.json"
```

Lệnh so sánh từ chối các lượt khác mã tài liệu, ảnh hoặc ground truth. Báo cáo nêu CER/WER và độ chính xác ký hiệu theo từng tài liệu, thời gian, chỉ số bảng, cùng danh sách tài liệu có CER tăng. Giữ nguyên cả lượt baseline và thử nghiệm để báo cáo không ghi đè lịch sử.

## 4. Trạng thái dữ liệu hiện có

Trong repository có ảnh tại `assets/imagetest` và bộ benchmark bảng tổng hợp ở `tests/benchmarks/table_benchmark_samples.py`. Bộ benchmark bảng có ground truth cấu trúc được viết trong mã để kiểm thử tổng hợp; nó không phải bộ ground truth OCR tiếng Việt được người dùng xác nhận trên ảnh tài liệu thật. Chưa có manifest ground truth thủ công đã xác minh được ghi nhận ở repository. Vì vậy chưa có CER/WER hay kết quả cải thiện thực tế để báo cáo.

## 5. Giới hạn

- CER/WER đo nội dung chuỗi; chúng không đánh giá thứ tự đọc theo vùng nếu ground truth tuyến tính hóa khác quy ước.
- Độ chính xác ký hiệu là chỉ số ký tự gắn với căn chỉnh Levenshtein, không phải độ chính xác phân loại theo từng loại ký hiệu.
- Đánh giá cấu trúc bảng tái sử dụng `app.core.table_evaluator`; độ chính xác OCR và chất lượng Word/định dạng vẫn là chiều đo riêng.
- Thời gian thay đổi theo CPU/GPU, cache model, phiên bản thư viện và mức tải máy. So sánh nên chạy trên cùng máy, cùng cấu hình và cùng manifest.
- Cần đủ số tài liệu mỗi nhóm để kết luận có ý nghĩa; các trung bình trên mẫu nhỏ chỉ mang tính mô tả.
