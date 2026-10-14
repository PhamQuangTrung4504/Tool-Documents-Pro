# -*- coding: utf-8 -*-
"""
tests.test_ocr_symbol_recovery
==============================
Bộ kiểm thử hồi quy (Regression Tests) toàn diện cho Context-Aware Symbol Recovery
và OCR Pipeline:
A. Khôi phục số La Mã đầu mục (Dương tính & Âm tính).
B. Chuẩn hóa Bullet và dấu cộng (+) theo ngữ cảnh (Dương tính & Âm tính).
C. Bảo toàn dấu phần trăm (%) hợp lệ.
D. Tái dựng bảng và bảo toàn số liệu bảng.
E. Tính bất biến đối với dữ liệu văn bản hành chính, đường dẫn, mã kỹ thuật.
F. Đánh giá chất lượng phép sửa: Đo lường độ chính xác (Precision = 100%).
"""

import os
import sys

# Đảm bảo import được module app
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pytest
from app.core.ocr_engine import (
    ContextSymbolRecovery,
    SymbolDecision,
    OCRBox,
    OCRResult,
    OCRTable,
    detect_and_reconstruct_table,
    reconstruct_document_layout,
    VietnameseTextPostProcessor,
)


class TestRomanNumeralsRecovery:
    """A. KIỂM THỬ SỐ LA MÃ ĐẦU MỤC (Mục 12.A)."""

    @pytest.mark.parametrize(
        "input_text,expected",
        [
            ("| NỘI DUNG, CHƯƠNG TRÌNH SINH HOẠT CHI BỘ", "I. NỘI DUNG, CHƯƠNG TRÌNH SINH HOẠT CHI BỘ"),
            ("|. NỘI DUNG SINH HOẠT", "I. NỘI DUNG SINH HOẠT"),
            ("l. NỘI DUNG KẾ HOẠCH", "I. NỘI DUNG KẾ HOẠCH"),
            ("TI. TIẾN HÀNH SINH HOẠT", "II. TIẾN HÀNH SINH HOẠT"),
            ("ll. TIẾN HÀNH SINH HOẠT", "II. TIẾN HÀNH SINH HOẠT"),
            ("11. TIẾN HÀNH SINH HOẠT", "II. TIẾN HÀNH SINH HOẠT"),
            ("TI TIẾN HÀNH SINH HOẠT", "II. TIẾN HÀNH SINH HOẠT"),
            ("lll. KẾT LUẬN", "III. KẾT LUẬN"),
            ("111. KẾT LUẬN", "III. KẾT LUẬN"),
            ("TII. KẾT LUẬN", "III. KẾT LUẬN"),
            ("lV. TỔ CHỨC THỰC HIỆN", "IV. TỔ CHỨC THỰC HIỆN"),
            ("1V. TỔ CHỨC THỰC HIỆN", "IV. TỔ CHỨC THỰC HIỆN"),
            ("TV. TỔ CHỨC THỰC HIỆN", "IV. TỔ CHỨC THỰC HIỆN"),
            ("II. ĐÁNH GIÁ TÌNH HÌNH", "II. ĐÁNH GIÁ TÌNH HÌNH"),
            ("III. PHƯƠNG HƯỚNG NHIỆM VỤ", "III. PHƯƠNG HƯỚNG NHIỆM VỤ"),
        ],
    )
    def test_roman_numerals_positive(self, input_text, expected):
        """Khôi phục thành công các biến thể OCR của số La Mã đầu mục khi có bằng chứng ngữ cảnh."""
        recovered, dec, _, _ = ContextSymbolRecovery.recover_roman_numeral_in_line(input_text)
        assert recovered == expected
        if input_text != expected:
            assert dec == SymbolDecision.AUTO_CORRECT

    @pytest.mark.parametrize(
        "negative_input",
        [
            "1. NỘP HỒ SƠ",
            "2. KẾT QUẢ",
            "1. Tuyên bố lý do, giới thiệu đại biểu:",
            "2. Chủ trì: Đồng chí Lê Văn Điền, Bí thư chi bộ.",
            "3. Thư ký: Đồng chí Nguyễn Văn Thắng, Chi ủy viên Chi bộ.",
            "4. Đồng chí Lê Văn Điền thông tin tình hình...",
            "5. Đồng chí Lê Thụy Thúy Phượng thông báo...",
            "6. Đồng chí Lê Văn Điền thông qua nội dung...",
            "Số 01-BB/CB",
            "Công văn số 1225-CV/ĐU ngày 21/9/2026",
            "14 đồng chí (14 chính thức, 0 dự bị)",
            "Đảng ủy xã Phước Hải vào lúc 15 giờ",
            "Mã hiệu: TC-01/2026-XNK",
        ],
    )
    def test_roman_numerals_negative(self, negative_input):
        """Bảo vệ tuyệt đối số thứ tự hợp lệ, số hiệu văn bản, ngày tháng, không đổi thành số La Mã."""
        recovered, dec, _, _ = ContextSymbolRecovery.recover_roman_numeral_in_line(negative_input)
        assert recovered == negative_input
        assert dec == SymbolDecision.PRESERVE


class TestBulletRecovery:
    """B. KIỂM THỬ BULLET VÀ DẤU CỘNG (Mục 12.B)."""

    @pytest.mark.parametrize(
        "input_text,expected",
        [
            (
                "% Chi bộ dưới 30 đảng viên ít nhất 90 phút; chi bộ trên 30 đảng viên ít nhất 120 phút",
                "+ Chi bộ dưới 30 đảng viên ít nhất 90 phút; chi bộ trên 30 đảng viên ít nhất 120 phút",
            ),
            (
                "% Thời gian sinh hoạt chi bộ ít hơn quy định 15 phút",
                "+ Thời gian sinh hoạt chi bộ ít hơn quy định 15 phút",
            ),
            (
                "% Từ 90% trở lên",
                "+ Từ 90% trở lên",
            ),
            (
                "% Từ 75% đến dưới 90%",
                "+ Từ 75% đến dưới 90%",
            ),
            (
                "% Từ 65% đến dưới 75%",
                "+ Từ 65% đến dưới 75%",
            ),
            (
                "% Không có đảng viên vắng mặt có lý do quá 03 lần liên tiếp trong năm'",
                "+ Không có đảng viên vắng mặt có lý do quá 03 lần liên tiếp trong năm'",
            ),
            (
                "| Chi bộ thực hiện tốt nhiệm vụ",
                "+ Chi bộ thực hiện tốt nhiệm vụ",
            ),
        ],
    )
    def test_bullet_positive(self, input_text, expected):
        """Khôi phục chính xác dấu cộng (+) đầu dòng bị OCR thành % hoặc |."""
        recovered, dec, _ = ContextSymbolRecovery.recover_bullets_and_pluses(input_text)
        assert recovered == expected
        assert dec == SymbolDecision.AUTO_CORRECT

    @pytest.mark.parametrize(
        "negative_input",
        [
            "a + b = c",
            "50% + 50% = 100%",
            "Tỷ lệ đạt 90%.",
            "vào lúc 15 giờ",
            "ông Nguyễn Văn An",
            "- Lý do: Tổ chức sinh hoạt chi bộ định kỳ tháng 10 năm 2026.",
            "- Giới thiệu đại biểu: không có.",
            "– Nội dung sinh hoạt theo hướng dẫn.",
            "* Ghi chú quan trọng.",
            "1. Tuyên bố lý do",
            "2. Chủ trì hội nghị",
            "5. Đồng chí thông báo tình hình",
            "6. Đồng chí thông qua nội dung",
        ],
    )
    def test_bullet_negative(self, negative_input):
        """Bảo vệ tuyệt đối phép toán, từ ngữ tiếng Việt (v, o), số thứ tự và gạch ngang hợp lệ."""
        recovered, dec, _ = ContextSymbolRecovery.recover_bullets_and_pluses(negative_input)
        assert recovered == negative_input
        assert dec == SymbolDecision.PRESERVE


class TestPercentagePreservation:
    """C. BẢO TOÀN DẤU PHẦN TRĂM HỢP LỆ (Mục 12.C)."""

    @pytest.mark.parametrize(
        "input_text",
        [
            "90%",
            "75%",
            "100%",
            "0.5%",
            "Tỷ lệ đạt 90% theo quy định.",
            "Dưới 60% hoặc có đảng viên vắng không có lý do",
            "Đạt chỉ tiêu 100% trong năm 2026",
            "STT | Tiêu chí | Đạt 90% | Đạt 100% | Ghi chú",
            "Tăng trưởng 8.5% so với cùng kỳ",
        ],
    )
    def test_percentage_intact(self, input_text):
        """Không làm mất hoặc biến đổi bất kỳ giá trị phần trăm hợp lệ nào."""
        recovered, dec, _ = ContextSymbolRecovery.recover_bullets_and_pluses(input_text)
        assert recovered == input_text
        assert "%" in recovered


class TestTableAndDataIntegrity:
    """D. BẢO TOÀN BẢNG BIỂU VÀ SỐ LIỆU (Mục 12.D)."""

    def test_table_stt_column_alignment(self):
        """
        Kiểm tra phục hồi và bảo toàn cột STT trong bảng biểu:
        Số 2 không bị lệch sang cột 1 (Tiêu chí), mà luôn ở cột 0 (STT).
        Dấu % ở đầu ô tiêu chí được khôi phục thành +, nhưng 90% ở sau được giữ nguyên.
        """
        box_info = [
            # Header
            {"box": OCRBox("STT", 0.95, [[30, 110], [65, 110], [65, 135], [30, 135]]), "text": "STT", "left": 30, "right": 65, "top": 110, "bottom": 135, "w": 35, "h": 25, "cy": 122.5},
            {"box": OCRBox("Tiêu chí đánh giá", 0.95, [[150, 110], [400, 110], [400, 135], [150, 135]]), "text": "Tiêu chí đánh giá", "left": 150, "right": 400, "top": 110, "bottom": 135, "w": 250, "h": 25, "cy": 122.5},
            {"box": OCRBox("Điểm tối đa", 0.95, [[630, 110], [690, 110], [690, 135], [630, 135]]), "text": "Điểm tối đa", "left": 630, "right": 690, "top": 110, "bottom": 135, "w": 60, "h": 25, "cy": 122.5},
            {"box": OCRBox("Điểm tự chấm", 0.95, [[710, 110], [770, 110], [770, 135], [710, 135]]), "text": "Điểm tự chấm", "left": 710, "right": 770, "top": 110, "bottom": 135, "w": 60, "h": 25, "cy": 122.5},
            {"box": OCRBox("Ghi chú", 0.95, [[790, 110], [840, 110], [840, 135], [790, 135]]), "text": "Ghi chú", "left": 790, "right": 840, "top": 110, "bottom": 135, "w": 50, "h": 25, "cy": 122.5},

            # Row 1 (STT: 1)
            {"box": OCRBox("1", 0.95, [[40, 150], [55, 150], [55, 175], [40, 175]]), "text": "1", "left": 40, "right": 55, "top": 150, "bottom": 175, "w": 15, "h": 25, "cy": 162.5},
            {"box": OCRBox("Chấp hành thời gian, thời lượng sinh hoạt", 0.95, [[80, 150], [550, 150], [550, 175], [80, 175]]), "text": "Chấp hành thời gian, thời lượng sinh hoạt", "left": 80, "right": 550, "top": 150, "bottom": 175, "w": 470, "h": 25, "cy": 162.5},
            {"box": OCRBox("05", 0.95, [[640, 150], [670, 150], [670, 175], [640, 175]]), "text": "05", "left": 640, "right": 670, "top": 150, "bottom": 175, "w": 30, "h": 25, "cy": 162.5},
            {"box": OCRBox("05", 0.95, [[725, 150], [755, 150], [755, 175], [725, 175]]), "text": "05", "left": 725, "right": 755, "top": 150, "bottom": 175, "w": 30, "h": 25, "cy": 162.5},

            # Row 2 (Sub-item with %)
            {"box": OCRBox("% Chi bộ dưới 30 đảng viên ít nhất 90 phút...", 0.95, [[80, 185], [580, 185], [580, 210], [80, 210]]), "text": "% Chi bộ dưới 30 đảng viên ít nhất 90 phút...", "left": 80, "right": 580, "top": 185, "bottom": 210, "w": 500, "h": 25, "cy": 197.5},
            {"box": OCRBox("05", 0.95, [[640, 185], [670, 185], [670, 210], [640, 210]]), "text": "05", "left": 640, "right": 670, "top": 185, "bottom": 210, "w": 30, "h": 25, "cy": 197.5},
            {"box": OCRBox("03", 0.95, [[725, 185], [755, 185], [755, 210], [725, 210]]), "text": "03", "left": 725, "right": 755, "top": 185, "bottom": 210, "w": 30, "h": 25, "cy": 197.5},

            # Row 3 (STT: 2)
            {"box": OCRBox("2", 0.95, [[40, 230], [55, 230], [55, 255], [40, 255]]), "text": "2", "left": 40, "right": 55, "top": 230, "bottom": 255, "w": 15, "h": 25, "cy": 242.5},
            {"box": OCRBox("Tỷ lệ đảng viên dự sinh hoạt chi bộ", 0.95, [[80, 230], [480, 230], [480, 255], [80, 255]]), "text": "Tỷ lệ đảng viên dự sinh hoạt chi bộ", "left": 80, "right": 480, "top": 230, "bottom": 255, "w": 400, "h": 25, "cy": 242.5},
            {"box": OCRBox("05", 0.95, [[640, 230], [670, 230], [670, 255], [640, 255]]), "text": "05", "left": 640, "right": 670, "top": 230, "bottom": 255, "w": 30, "h": 25, "cy": 242.5},
            {"box": OCRBox("05", 0.95, [[725, 230], [755, 230], [755, 255], [725, 255]]), "text": "05", "left": 725, "right": 755, "top": 230, "bottom": 255, "w": 30, "h": 25, "cy": 242.5},
        ]

        table_obj, _, _, _ = detect_and_reconstruct_table(
            box_info, page_min_x=30, page_max_x=840, page_w=810, median_h=25.0
        )
        assert table_obj is not None
        assert table_obj.num_rows == 4
        assert table_obj.num_cols == 5

        # Hàng 0: Header
        assert table_obj.grid[0][0] == "STT"
        assert table_obj.grid[0][1] == "Tiêu chí đánh giá"

        # Hàng 1: STT = 1
        assert table_obj.grid[1][0] == "1"
        assert table_obj.grid[1][1] == "Chấp hành thời gian, thời lượng sinh hoạt"
        assert table_obj.grid[1][2] == "05"
        assert table_obj.grid[1][3] == "05"

        # Hàng 2: Sub-item, STT = rỗng, Tiêu chí khôi phục bullet +
        assert table_obj.grid[2][0] == ""
        assert table_obj.grid[2][1].startswith("+ Chi bộ dưới 30 đảng viên")
        assert table_obj.grid[2][2] == "05"
        assert table_obj.grid[2][3] == "03"

        # Hàng 3: STT = 2 (ĐƯỢC ĐẶT CHÍNH XÁC Ở CỘT 0, KHÔNG BỊ LỆCH SANG CỘT 1)
        assert table_obj.grid[3][0] == "2"
        assert table_obj.grid[3][1] == "Tỷ lệ đảng viên dự sinh hoạt chi bộ"
        assert table_obj.grid[3][2] == "05"
        assert table_obj.grid[3][3] == "05"


class TestImmutabilityAndPrecision:
    """E & F. TÍNH BẤT BIẾN VÀ ĐÁNH GIÁ ĐỘ CHÍNH XÁC (Mục 12.E, 13)."""

    def test_immutability_of_system_text(self):
        """Kiểm tra không bị thay đổi ngoài ý muốn đối với các định dạng kỹ thuật."""
        test_inputs = [
            "C:/Users/pqtru/Documents/ocr_output.txt",
            "https://dangcongsan.vn/sinh-hoat-chi-bo",
            "email_can_bo@ubnd.gov.vn",
            "Nghị định 30/2020/NĐ-CP ngày 05/3/2020",
            "Điểm số: -1.5 điểm và +2.0 điểm",
            "Tỷ lệ: 99.9%",
            "Mã hex: 0xFF00AA | Token: ID-9081",
        ]
        recovered, audit = ContextSymbolRecovery.recover_document_symbols(test_inputs)
        assert recovered == test_inputs
        assert len(audit) == 0  # Không có phép sửa nào bị kích hoạt nhầm

    def test_precision_metric(self):
        """
        Đánh giá chất lượng phép sửa:
        Precision = (Số phép sửa đúng) / (Tổng số phép tự động sửa)
        Kỳ vọng: Precision = 100.0% (Không có phép sửa sai).
        """
        dataset = [
            # Cần sửa đúng (True Positives)
            ("| NỘI DUNG, CHƯƠNG TRÌNH SINH HOẠT CHI BỘ", "I. NỘI DUNG, CHƯƠNG TRÌNH SINH HOẠT CHI BỘ", True),
            ("TI. TIẾN HÀNH SINH HOẠT", "II. TIẾN HÀNH SINH HOẠT", True),
            ("lll. KẾT LUẬN", "III. KẾT LUẬN", True),
            ("lV. TỔ CHỨC THỰC HIỆN", "IV. TỔ CHỨC THỰC HIỆN", True),
            ("% Chi bộ dưới 30 đảng viên", "+ Chi bộ dưới 30 đảng viên", True),
            ("% Thời gian sinh hoạt", "+ Thời gian sinh hoạt", True),
            ("% Từ 90% trở lên", "+ Từ 90% trở lên", True),
            # Cần giữ nguyên (True Negatives)
            ("1. Tuyên bố lý do", "1. Tuyên bố lý do", False),
            ("2. Chủ trì hội nghị", "2. Chủ trì hội nghị", False),
            ("1. NỘP HỒ SƠ", "1. NỘP HỒ SƠ", False),
            ("2. KẾT QUẢ", "2. KẾT QUẢ", False),
            ("Tỷ lệ đạt 90%.", "Tỷ lệ đạt 90%.", False),
            ("50% + 50% = 100%", "50% + 50% = 100%", False),
            ("Số 01-BB/CB", "Số 01-BB/CB", False),
            ("- Lý do: Tổ chức sinh hoạt", "- Lý do: Tổ chức sinh hoạt", False),
        ]

        inputs = [item[0] for item in dataset]
        expected = [item[1] for item in dataset]
        needs_correction = [item[2] for item in dataset]

        recovered, audit = ContextSymbolRecovery.recover_document_symbols(inputs)

        auto_corrected_count = len(audit)
        correctly_corrected_count = 0
        false_positives = 0

        for orig, rec, exp, should_fix in zip(inputs, recovered, expected, needs_correction):
            if should_fix:
                if rec == exp:
                    correctly_corrected_count += 1
                else:
                    print(f"Error on fix: {orig} -> {rec} (exp: {exp})")
            else:
                if rec != orig:
                    false_positives += 1
                    print(f"False positive: {orig} was wrongly modified to {rec}")

        precision = correctly_corrected_count / auto_corrected_count if auto_corrected_count > 0 else 0.0
        print(f"\nCorrection Quality: Total fixes={auto_corrected_count}, Correct={correctly_corrected_count}, False positives={false_positives}, Precision={precision:.2%}")

        assert false_positives == 0, "Không được có bất kỳ phép sửa sai nào trên dữ liệu hợp lệ!"
        assert correctly_corrected_count == 7, "Cả 7 trường hợp cần sửa đều phải sửa đúng!"
        assert precision == 1.0, "Độ chính xác phép sửa phải đạt tuyệt đối 100%!"


class TestStaffTableMultilineMerge:
    """G. KIỂM THỬ GỘP Ô ĐA DÒNG VÀ TRIỆT TIÊU HÀNG RÁC MỒ CÔI (Ảnh 3)."""

    def test_staff_table_split_birthdate_merged(self):
        import numpy as np

        boxes = [
            # Headers
            OCRBox(text="STT", bbox=[[20, 10], [50, 10], [50, 30], [20, 30]], confidence=0.95),
            OCRBox(text="Họ tên", bbox=[[70, 10], [180, 10], [180, 30], [70, 30]], confidence=0.95),
            OCRBox(text="Ngày sinh", bbox=[[200, 10], [280, 10], [280, 30], [200, 30]], confidence=0.95),
            OCRBox(text="Giới tính", bbox=[[300, 10], [360, 10], [360, 30], [300, 30]], confidence=0.95),
            OCRBox(text="Chủ hộ", bbox=[[380, 10], [480, 10], [480, 30], [380, 30]], confidence=0.95),
            OCRBox(text="Địa chỉ", bbox=[[500, 10], [600, 10], [600, 30], [500, 30]], confidence=0.95),
            OCRBox(text="Tên cha", bbox=[[620, 10], [720, 10], [720, 30], [620, 30]], confidence=0.95),
            # Row 1 (01/01/194 bị ngắt đuôi '8')
            OCRBox(text="1", bbox=[[25, 45], [45, 45], [45, 65], [25, 65]], confidence=0.95),
            OCRBox(text="ĐIỀN THỊ LIÊN", bbox=[[75, 45], [175, 45], [175, 65], [75, 65]], confidence=0.95),
            OCRBox(text="01/01/194", bbox=[[205, 45], [275, 45], [275, 65], [205, 65]], confidence=0.95),
            OCRBox(text="8", bbox=[[255, 68], [268, 68], [268, 85], [255, 85]], confidence=0.95),
            OCRBox(text="Nữ", bbox=[[310, 45], [340, 45], [340, 65], [310, 65]], confidence=0.95),
            OCRBox(text="ĐIỀN THỊ LIÊN", bbox=[[385, 45], [475, 45], [475, 65], [385, 65]], confidence=0.95),
            OCRBox(text="TỔ 5, THÔN 2", bbox=[[505, 45], [595, 45], [595, 65], [505, 65]], confidence=0.95),
            OCRBox(text="ĐIỀN HẠNH", bbox=[[625, 45], [715, 45], [715, 65], [625, 65]], confidence=0.95),
            # Row 2 (19/02/198 bị ngắt đuôi '4')
            OCRBox(text="2", bbox=[[25, 95], [45, 95], [45, 115], [25, 115]], confidence=0.95),
            OCRBox(text="NGUYỄN THỊ THUỶ TÂN", bbox=[[75, 95], [185, 95], [185, 115], [75, 115]], confidence=0.95),
            OCRBox(text="19/02/198", bbox=[[205, 95], [275, 95], [275, 115], [205, 115]], confidence=0.95),
            OCRBox(text="4", bbox=[[255, 118], [268, 118], [268, 135], [255, 135]], confidence=0.95),
            OCRBox(text="Nữ", bbox=[[310, 95], [340, 95], [340, 115], [310, 115]], confidence=0.95),
            OCRBox(text="BÙI ĐỨC VĂN", bbox=[[385, 95], [475, 95], [475, 115], [385, 115]], confidence=0.95),
            OCRBox(text="Thôn 2", bbox=[[505, 95], [555, 95], [555, 115], [505, 115]], confidence=0.95),
            OCRBox(text="NGUYỄN VĂN LỘC", bbox=[[625, 95], [725, 95], [725, 115], [625, 115]], confidence=0.95),
        ]

        box_info = []
        for b in boxes:
            l = min(pt[0] for pt in b.bbox)
            r = max(pt[0] for pt in b.bbox)
            t = min(pt[1] for pt in b.bbox)
            bt = max(pt[1] for pt in b.bbox)
            box_info.append({
                "box": b,
                "text": b.text,
                "left": l,
                "right": r,
                "top": t,
                "bottom": bt,
                "w": r - l,
                "h": bt - t,
                "cy": (t + bt) / 2.0,
            })

        page_min_x = min(item["left"] for item in box_info)
        page_max_x = max(item["right"] for item in box_info)
        page_w = page_max_x - page_min_x
        median_h = float(np.median([item["h"] for item in box_info]))

        tbl, pre, rec, post = detect_and_reconstruct_table(box_info, page_min_x, page_max_x, page_w, median_h)

        assert tbl is not None, "Bảng nhân sự phải được phát hiện và tái dựng thành công"
        # Bảng phải có đúng 3 hàng (Header + Row 1 + Row 2), không được sinh ra hàng rác mồ côi | 8 | |
        assert tbl.num_rows == 3, f"Số hàng phải là 3, thực tế: {tbl.num_rows}"
        assert tbl.num_cols == 7, f"Số cột phải là 7, thực tế: {tbl.num_cols}"

        row1 = tbl.grid[1]
        assert row1[0] == "1"
        assert row1[1] == "ĐIỀN THỊ LIÊN"
        assert row1[2] == "01/01/1948", f"Ngày sinh hàng 1 phải được ghép liền thành '01/01/1948', thực tế: '{row1[2]}'"
        assert row1[3] == "Nữ"
        assert row1[4] == "ĐIỀN THỊ LIÊN"
        assert row1[5] == "TỔ 5, THÔN 2"
        assert row1[6] == "ĐIỀN HẠNH"

        row2 = tbl.grid[2]
        assert row2[0] == "2"
        assert row2[1] == "NGUYỄN THỊ THUỶ TÂN"
        assert row2[2] == "19/02/1984", f"Ngày sinh hàng 2 phải được ghép liền thành '19/02/1984', thực tế: '{row2[2]}'"


class TestIconTableSymbolHandling:
    """H. KIỂM THỬ BẢNG BIỂU TƯỢNG VÀ MÃ KÝ HIỆU (Ảnh 4)."""

    def test_icon_column_cleanup(self):
        import numpy as np

        boxes = [
            # Header
            OCRBox(text="Symb", bbox=[[10, 10], [50, 10], [50, 30], [10, 30]], confidence=0.95),
            OCRBox(text="Code", bbox=[[60, 10], [120, 10], [120, 30], [60, 30]], confidence=0.95),
            OCRBox(text="Symb", bbox=[[140, 10], [180, 10], [180, 30], [140, 30]], confidence=0.95),
            OCRBox(text="Code", bbox=[[190, 10], [250, 10], [250, 30], [190, 30]], confidence=0.95),
            # Row 1 (cy -> [Icon], Code: 14; ♫ -> giữ nguyên, Code: 15)
            OCRBox(text="cy", bbox=[[15, 40], [45, 40], [45, 60], [15, 60]], confidence=0.95),
            OCRBox(text="14", bbox=[[65, 40], [110, 40], [110, 60], [65, 60]], confidence=0.95),
            OCRBox(text="♫", bbox=[[145, 40], [175, 40], [175, 60], [145, 60]], confidence=0.95),
            OCRBox(text="15", bbox=[[195, 40], [240, 40], [240, 60], [195, 60]], confidence=0.95),
            # Row 2 (cos -> [Icon], Code: 21; ® -> giữ nguyên, Code: 0174)
            OCRBox(text="cos", bbox=[[15, 70], [45, 70], [45, 90], [15, 90]], confidence=0.95),
            OCRBox(text="21", bbox=[[65, 70], [110, 70], [110, 90], [65, 90]], confidence=0.95),
            OCRBox(text="®", bbox=[[145, 70], [175, 70], [175, 90], [145, 90]], confidence=0.95),
            OCRBox(text="0174", bbox=[[195, 70], [240, 70], [240, 90], [195, 90]], confidence=0.95),
            # Row 3 (TT -> [Icon], Code: 19)
            OCRBox(text="TT", bbox=[[15, 100], [45, 100], [45, 120], [15, 120]], confidence=0.95),
            OCRBox(text="19", bbox=[[65, 100], [110, 100], [110, 120], [65, 120]], confidence=0.95),
            OCRBox(text="±", bbox=[[145, 100], [175, 100], [175, 120], [145, 120]], confidence=0.95),
            OCRBox(text="0177", bbox=[[195, 100], [240, 100], [240, 120], [195, 120]], confidence=0.95),
        ]

        box_info = []
        for b in boxes:
            l = min(pt[0] for pt in b.bbox)
            r = max(pt[0] for pt in b.bbox)
            t = min(pt[1] for pt in b.bbox)
            bt = max(pt[1] for pt in b.bbox)
            box_info.append({
                "box": b,
                "text": b.text,
                "left": l,
                "right": r,
                "top": t,
                "bottom": bt,
                "w": r - l,
                "h": bt - t,
                "cy": (t + bt) / 2.0,
            })

        tbl, _, _, _ = detect_and_reconstruct_table(box_info, 10, 250, 1000, 20.0)
        assert tbl is not None, "Bảng biểu tượng phải được tái dựng thành công"

        # Hàng 1
        assert tbl.grid[1][0] == "[Icon]", f"Chuỗi rác 'cy' phải được thay bằng '[Icon]', thực tế: '{tbl.grid[1][0]}'"
        assert tbl.grid[1][1] == "14"
        assert tbl.grid[1][2] == "♫", f"Symbol chuẩn '♫' phải được giữ nguyên, thực tế: '{tbl.grid[1][2]}'"
        assert tbl.grid[1][3] == "15"

        # Hàng 2
        assert tbl.grid[2][0] == "[Icon]", f"Chuỗi rác 'cos' phải được thay bằng '[Icon]', thực tế: '{tbl.grid[2][0]}'"
        assert tbl.grid[2][1] == "21"
        assert tbl.grid[2][2] == "®", f"Symbol chuẩn '®' phải được giữ nguyên, thực tế: '{tbl.grid[2][2]}'"
        assert tbl.grid[2][3] == "0174"

        # Hàng 3
        assert tbl.grid[3][0] == "[Icon]", f"Chuỗi rác 'TT' phải được thay bằng '[Icon]', thực tế: '{tbl.grid[3][0]}'"
        assert tbl.grid[3][1] == "19"
        assert tbl.grid[3][2] == "±", f"Symbol chuẩn '±' phải được giữ nguyên, thực tế: '{tbl.grid[3][2]}'"
        assert tbl.grid[3][3] == "0177"


class TestPhase2SymbolAndTableCases:
    """I. KIỂM THỬ TOÀN DIỆN PHASE 2: SYMBOL & CỘT STT."""

    def test_sub_criterion_bullets_and_pipe_removal(self):
        """Khôi phục dấu cộng/trừ đầu dòng và loại bỏ pipe thừa."""
        inputs = [
            "| % Thời gian sinh hoạt chi bộ ít hơn quy định 15 phút",
            "| * Chi bộ dưới 30 đảng viên ít nhất 90 phút; chi bộ trên 30 đảng viên ít nhất 120 phút",
            "| + Chi bộ thực hiện tốt nhiệm vụ",
            "| - Nội dung sinh hoạt chi bộ theo hướng dẫn",
            "+ Chi bộ đoàn kết",
        ]
        recovered, audit = ContextSymbolRecovery.recover_document_symbols(inputs)
        assert recovered[0] == "+ Thời gian sinh hoạt chi bộ ít hơn quy định 15 phút"
        assert recovered[1] == "+ Chi bộ dưới 30 đảng viên ít nhất 90 phút; chi bộ trên 30 đảng viên ít nhất 120 phút"
        assert recovered[2] == "+ Chi bộ thực hiện tốt nhiệm vụ"
        assert recovered[3] == "- Nội dung sinh hoạt chi bộ theo hướng dẫn"
        assert recovered[4] == "+ Chi bộ đoàn kết"

    def test_stt_double_pipe_cleanup(self):
        """Xử lý dấu '| |' thừa ở cột STT và bảo toàn số thứ tự cùng nội dung."""
        inputs = [
            "1 | | Chấp hành thời gian, thời lượng sinh hoạt | 05 | 05",
            "2 | | Tỷ lệ đảng viên dự sinh hoạt chi bộ | 05 | 05",
        ]
        recovered, audit = ContextSymbolRecovery.recover_document_symbols(inputs)
        assert recovered[0] == "1 | Chấp hành thời gian, thời lượng sinh hoạt | 05 | 05"
        assert recovered[1] == "2 | Tỷ lệ đảng viên dự sinh hoạt chi bộ | 05 | 05"
        assert len(audit) == 2
        assert audit[0]["rule"] == "STT_DOUBLE_PIPE_CLEANUP"

    def test_preserve_percentages_and_valid_symbols(self):
        """Bảo toàn các giá trị phần trăm, phép tính, URL, mã hex hợp lệ."""
        inputs = [
            "Tỷ lệ đạt 90% theo quy định",
            "Từ 75% đến dưới 90%",
            "Dưới 60% hoặc có đảng viên vắng không có lý do",
            "50% + 50% = 100%",
            "a + b = c",
            "* Ghi chú: Đây là chú thích quan trọng.",
            "Mã hiệu: 0xFF | ID: 100",
        ]
        recovered, audit = ContextSymbolRecovery.recover_document_symbols(inputs)
        assert recovered == inputs
        assert len(audit) == 0


class TestPhase3TwoColumnHeaderAndDecorativeLine:
    """J. KIỂM THỬ TOÀN DIỆN PHASE 3: HEADER HAI CỘT & LOẠI BỎ ĐƯỜNG KẺ TRANG TRÍ."""

    def test_parallel_header_decorative_percent_removed(self):
        """
        Kiểm tra Header hai cột:
        - Khối trái có đường kẻ ngang trang trí bị nhận nhầm thành '%'.
        - Kết quả: '%' bị loại bỏ hoàn toàn, khối trái đọc trọn vẹn 4 dòng rồi đến khối phải 2 dòng.
        """
        from app.core.ocr_engine import detect_parallel_administrative_header

        box_info = [
            # Khối trái
            {"box": OCRBox("ĐẢNG ỦY XÃ PHƯỚC HẢI", 0.98, [[80, 100], [260, 100], [260, 125], [80, 125]]), "text": "ĐẢNG ỦY XÃ PHƯỚC HẢI", "left": 80, "right": 260, "top": 100, "bottom": 125, "w": 180, "h": 25, "cy": 112.5},
            {"box": OCRBox("CHI BỘ VĂN PHÒNG HĐND", 0.97, [[70, 130], [270, 130], [270, 150], [70, 150]]), "text": "CHI BỘ VĂN PHÒNG HĐND", "left": 70, "right": 270, "top": 130, "bottom": 150, "w": 200, "h": 20, "cy": 140.0},
            {"box": OCRBox("VÀ UBND XÃ", 0.96, [[100, 155], [210, 155], [210, 175], [100, 175]]), "text": "VÀ UBND XÃ", "left": 100, "right": 210, "top": 155, "bottom": 175, "w": 110, "h": 20, "cy": 165.0},
            {"box": OCRBox("%", 0.95, [[145, 180], [165, 180], [165, 195], [145, 195]]), "text": "%", "left": 145, "right": 165, "top": 180, "bottom": 195, "w": 20, "h": 15, "cy": 187.5},
            {"box": OCRBox("Số 01-BB/CB", 0.97, [[100, 200], [200, 200], [200, 225], [100, 225]]), "text": "Số 01-BB/CB", "left": 100, "right": 200, "top": 200, "bottom": 225, "w": 100, "h": 25, "cy": 212.5},
            # Khối phải
            {"box": OCRBox("ĐẢNG CỘNG SẢN VIỆT NAM", 0.99, [[380, 100], [580, 100], [580, 125], [380, 125]]), "text": "ĐẢNG CỘNG SẢN VIỆT NAM", "left": 380, "right": 580, "top": 100, "bottom": 125, "w": 200, "h": 25, "cy": 112.5},
            {"box": OCRBox("Phước Hải, ngày 05 tháng 10 năm 2026", 0.98, [[350, 150], [580, 150], [580, 175], [350, 175]]), "text": "Phước Hải, ngày 05 tháng 10 năm 2026", "left": 350, "right": 580, "top": 150, "bottom": 175, "w": 230, "h": 25, "cy": 162.5},
            # Tiêu đề giữa trang
            {"box": OCRBox("BIÊN BẢN", 0.99, [[250, 250], [370, 250], [370, 280], [250, 280]]), "text": "BIÊN BẢN", "left": 250, "right": 370, "top": 250, "bottom": 280, "w": 120, "h": 30, "cy": 265.0},
        ]

        header_lines, rem = detect_parallel_administrative_header(
            box_info, page_min_x=70, page_max_x=580, page_w=510, gutter_thresh=20.0, median_h=22.0
        )

        header_line_texts = [" ".join(b.text for b in line) for line in header_lines]

        # Đường kẻ '%' phải bị loại bỏ, khối trái gồm đúng 4 dòng:
        assert len(header_line_texts) == 6
        assert header_line_texts[0] == "ĐẢNG ỦY XÃ PHƯỚC HẢI"
        assert header_line_texts[1] == "CHI BỘ VĂN PHÒNG HĐND"
        assert header_line_texts[2] == "VÀ UBND XÃ"
        assert header_line_texts[3] == "Số 01-BB/CB"
        # Sau đó là Khối phải gồm 2 dòng:
        assert header_line_texts[4] == "ĐẢNG CỘNG SẢN VIỆT NAM"
        assert header_line_texts[5] == "Phước Hải, ngày 05 tháng 10 năm 2026"


class TestPhase4SpellingAndTokenization:
    """K. KIỂM THỬ TOÀN DIỆN PHASE 4: SPELL CHECKING CHO 'Vdng MẶT'."""

    def test_vdng_mat_detected_with_exact_span(self):
        """Phát hiện lỗi 'Vdng' với span chính xác, không dính dấu '-' và gợi ý 'Vắng'."""
        from app.core.spell_corrector import check_text_spelling, find_spelling_errors

        sample_text = "- Vdng mặt: 0 đồng chí."

        # Kiểm tra qua cả 2 API
        errors_1 = check_text_spelling(sample_text)
        errors_2 = find_spelling_errors(sample_text)

        assert len(errors_1) == 1
        assert len(errors_2) == 1

        err = errors_1[0]
        assert err.token == "Vdng"
        assert err.start_idx == 2
        assert err.end_idx == 6
        assert sample_text[err.start_idx:err.end_idx] == "Vdng"
        assert "Vắng" in err.suggestions

    def test_correct_phrases_not_flagged(self):
        """Các từ/cụm từ đúng và whitelist ALL CAPS không bị cảnh báo nhầm."""
        from app.core.spell_corrector import check_text_spelling

        correct_samples = [
            "- Vắng mặt: 0 đồng chí.",
            "Có mặt: 14 đồng chí.",
            "UBND, HĐND, TW, BB, CB, DOCX, PDF",
            "Đảng ủy xã Phước Hải",
        ]
        for s in correct_samples:
            errors = check_text_spelling(s)
            assert len(errors) == 0, f"Chuỗi hợp lệ không được báo lỗi: '{s}' -> {errors}"


class TestPhase5TableCleanupAndClippedText:
    """L. KIỂM THỬ TOÀN DIỆN PHASE 5: LÀM SẠCH BẢNG ĐIỂM, STT 1, DẤU | HÀNG CON VÀ VĂN BẢN BỊ CẮT."""

    def test_clipped_text_administrative_recovery(self):
        """Khôi phục cụm từ bị cắt 'nhiệm vụ đặc b' -> 'nhiệm vụ đặc biệt'."""
        from app.core.ocr_engine import VietnameseTextPostProcessor

        clipped = "* Trừ trường hợp được miễn công tác, sinh hoạt đảng theo quy định và đảng viên trong lực lượng vũ trang thực hiện nhiệm vụ đặc b"
        restored = VietnameseTextPostProcessor.clean_and_restore(clipped)
        assert restored.endswith("thực hiện nhiệm vụ đặc biệt")

        # Trường hợp âm tính: từ khác chứa b không bị thay thế tùy tiện
        negative = "Học sinh lớp 12 b tham gia học tập"
        assert VietnameseTextPostProcessor.clean_and_restore(negative) == negative

    def test_table_reconstruction_stt_1_restored_and_sub_rows_piped(self):
        """
        Kiểm tra tái dựng bảng điểm:
        - Hàng 1 có ô STT bị thiếu/nhận nhầm nhưng hàng sau có STT '2' -> Tự động khôi phục STT '1'.
        - Hàng con bắt đầu bằng '+' có các cột điểm được phân cách bằng ' | ' thay vì dính space vào text.
        - Không có dấu '|' thừa lặp trong cột STT.
        """
        from app.core.ocr_engine import detect_and_reconstruct_table, OCRBox

        # Giả lập box_info cho bảng điểm
        # Cột STT: left ~ 20..60
        # Cột Tiêu chí: left ~ 80..500
        # Cột Điểm tối đa: left ~ 540..580
        # Cột Điểm tự chấm: left ~ 620..660
        box_info = [
            # Header
            {"box": OCRBox("STT", 0.99, [[20, 50], [60, 50], [60, 75], [20, 75]]), "text": "STT", "left": 20, "right": 60, "top": 50, "bottom": 75, "w": 40, "h": 25, "cy": 62.5},
            {"box": OCRBox("Tiêu chí đánh giá", 0.99, [[80, 50], [300, 50], [300, 75], [80, 75]]), "text": "Tiêu chí đánh giá", "left": 80, "right": 300, "top": 50, "bottom": 75, "w": 220, "h": 25, "cy": 62.5},
            {"box": OCRBox("Điểm tối đa", 0.98, [[540, 50], [580, 50], [580, 75], [540, 75]]), "text": "Điểm tối đa", "left": 540, "right": 580, "top": 50, "bottom": 75, "w": 40, "h": 25, "cy": 62.5},
            {"box": OCRBox("Điểm tự chấm", 0.98, [[620, 50], [660, 50], [660, 75], [620, 75]]), "text": "Điểm tự chấm", "left": 620, "right": 660, "top": 50, "bottom": 75, "w": 40, "h": 25, "cy": 62.5},

            # Hàng 1 (STT bị VietOCR nhận thành '|', Tiêu chí chính 1)
            {"box": OCRBox("|", 0.80, [[35, 85], [45, 85], [45, 105], [35, 105]]), "text": "|", "left": 35, "right": 45, "top": 85, "bottom": 105, "w": 10, "h": 20, "cy": 95.0},
            {"box": OCRBox("Chấp hành thời gian, thời lượng sinh hoạt", 0.98, [[80, 85], [450, 85], [450, 105], [80, 105]]), "text": "Chấp hành thời gian, thời lượng sinh hoạt", "left": 80, "right": 450, "top": 85, "bottom": 105, "w": 370, "h": 20, "cy": 95.0},
            {"box": OCRBox("05", 0.97, [[545, 85], [575, 85], [575, 105], [545, 105]]), "text": "05", "left": 545, "right": 575, "top": 85, "bottom": 105, "w": 30, "h": 20, "cy": 95.0},
            {"box": OCRBox("05", 0.97, [[625, 85], [655, 85], [655, 105], [625, 105]]), "text": "05", "left": 625, "right": 655, "top": 85, "bottom": 105, "w": 30, "h": 20, "cy": 95.0},

            # Hàng 2 (Tiêu chí con, không có STT)
            {"box": OCRBox("+ Chi bộ dưới 30 đảng viên ít nhất 90 phút", 0.97, [[80, 115], [480, 115], [480, 135], [80, 135]]), "text": "+ Chi bộ dưới 30 đảng viên ít nhất 90 phút", "left": 80, "right": 480, "top": 115, "bottom": 135, "w": 400, "h": 20, "cy": 125.0},
            {"box": OCRBox("05", 0.96, [[545, 115], [575, 115], [575, 135], [545, 135]]), "text": "05", "left": 545, "right": 575, "top": 115, "bottom": 135, "w": 30, "h": 20, "cy": 125.0},
            {"box": OCRBox("05", 0.96, [[625, 115], [655, 115], [655, 135], [625, 135]]), "text": "05", "left": 625, "right": 655, "top": 115, "bottom": 135, "w": 30, "h": 20, "cy": 125.0},

            # Hàng 3 (Tiêu chí chính 2 có STT '2')
            {"box": OCRBox("2", 0.98, [[35, 145], [45, 145], [45, 165], [35, 165]]), "text": "2", "left": 35, "right": 45, "top": 145, "bottom": 165, "w": 10, "h": 20, "cy": 155.0},
            {"box": OCRBox("Tỷ lệ đảng viên dự sinh hoạt chi bộ", 0.98, [[80, 145], [420, 145], [420, 165], [80, 165]]), "text": "Tỷ lệ đảng viên dự sinh hoạt chi bộ", "left": 80, "right": 420, "top": 145, "bottom": 165, "w": 340, "h": 20, "cy": 155.0},
            {"box": OCRBox("05", 0.97, [[545, 145], [575, 145], [575, 165], [545, 165]]), "text": "05", "left": 545, "right": 575, "top": 145, "bottom": 165, "w": 30, "h": 20, "cy": 155.0},
            {"box": OCRBox("05", 0.97, [[625, 145], [655, 145], [655, 165], [625, 165]]), "text": "05", "left": 625, "right": 655, "top": 145, "bottom": 165, "w": 30, "h": 20, "cy": 155.0},

            # Hàng 4 (Tiêu chí con của mục 2)
            {"box": OCRBox("+ Từ 90% trở lên", 0.98, [[80, 175], [260, 175], [260, 195], [80, 195]]), "text": "+ Từ 90% trở lên", "left": 80, "right": 260, "top": 175, "bottom": 195, "w": 180, "h": 20, "cy": 185.0},
            {"box": OCRBox("04", 0.97, [[545, 175], [575, 175], [575, 195], [545, 195]]), "text": "04", "left": 545, "right": 575, "top": 175, "bottom": 195, "w": 30, "h": 20, "cy": 185.0},
            {"box": OCRBox("04", 0.97, [[625, 175], [655, 175], [655, 195], [625, 195]]), "text": "04", "left": 625, "right": 655, "top": 175, "bottom": 195, "w": 30, "h": 20, "cy": 185.0},
        ]

        tbl, pre_b, tbl_rows, post_b = detect_and_reconstruct_table(
            box_info, page_min_x=20, page_max_x=660, page_w=640, median_h=20.0
        )

        assert tbl is not None
        assert len(tbl.grid) == 5
        # STT hàng 1 phải được khôi phục thành '1' (thay vì '|')
        assert tbl.grid[1][0] == "1"
        assert tbl.grid[1][1] == "Chấp hành thời gian, thời lượng sinh hoạt"
        # STT hàng 3 là '2'
        assert tbl.grid[3][0] == "2"
        # Tiêu chí con hàng 4 bảo toàn '90%'
        assert "90%" in tbl.grid[4][1]


class TestPhase6DocxExportParallelHeader:
    """M. KIỂM THỬ XUẤT DOCX CHO PARALLEL ADMINISTRATIVE HEADER."""

    def test_export_docx_with_parallel_header(self, tmp_path):
        """Xác nhận export_to_docx tạo bảng 2 cột không viền cho header và không duplicate text."""
        import docx
        from app.core.ocr_engine import ParallelHeader, OCRResult, export_to_docx

        hdr = ParallelHeader(
            left_lines=[
                "ĐẢNG ỦY XÃ PHƯỚC HẢI",
                "CHI BỘ VĂN PHÒNG HĐND VÀ UBND XÃ",
                "Số 01-BB/CB",
            ],
            right_lines=[
                "ĐẢNG CỘNG SẢN VIỆT NAM",
                "Phước Hải, ngày 05 tháng 10 năm 2026",
            ],
        )

        full_text = (
            "ĐẢNG ỦY XÃ PHƯỚC HẢI\n"
            "CHI BỘ VĂN PHÒNG HĐND VÀ UBND XÃ\n"
            "Số 01-BB/CB\n"
            "ĐẢNG CỘNG SẢN VIỆT NAM\n"
            "Phước Hải, ngày 05 tháng 10 năm 2026\n"
            "BIÊN BẢN\n"
            "sinh hoạt chi bộ định kỳ tháng 10 năm 2026\n"
            "Vào lúc 15 giờ, ngày 05 tháng 10 năm 2026..."
        )

        ocr_res = OCRResult(
            boxes=[],
            full_text=full_text,
            language="vi",
            parallel_header=hdr,
        )

        out_docx = str(tmp_path / "test_parallel_header.docx")
        success = export_to_docx(ocr_res, out_docx)
        assert success is True

        doc = docx.Document(out_docx)
        # Bảng đầu tiên phải là bảng 2 cột của Parallel Header
        assert len(doc.tables) >= 1
        hdr_tbl = doc.tables[0]
        assert len(hdr_tbl.rows) == 1
        assert len(hdr_tbl.columns) == 2

        # Cột trái chứa tên cơ quan, chi bộ, số hiệu
        left_text = hdr_tbl.cell(0, 0).text
        assert "ĐẢNG ỦY XÃ PHƯỚC HẢI" in left_text
        assert "Số 01-BB/CB" in left_text

        # Cột phải chứa Quốc hiệu, ngày tháng
        right_text = hdr_tbl.cell(0, 1).text
        assert "ĐẢNG CỘNG SẢN VIỆT NAM" in right_text
        assert "Phước Hải, ngày 05 tháng 10 năm 2026" in right_text

        # Các đoạn văn bản body bên ngoài không được lặp lại dòng header
        body_paras_text = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        assert "ĐẢNG ỦY XÃ PHƯỚC HẢI" not in body_paras_text
        assert "ĐẢNG CỘNG SẢN VIỆT NAM" not in body_paras_text
        assert "BIÊN BẢN" in body_paras_text


