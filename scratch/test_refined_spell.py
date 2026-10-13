import os
import sys
import unicodedata

sys.path.insert(0, os.path.abspath("."))

from app.core.spell_corrector import (
    VietnameseSpellChecker,
    weighted_edit_distance,
    SpellError,
    VIETNAMESE_DIACRITICS,
)

c = VietnameseSpellChecker()

PROTECTED_COMMON_WORDS = {
    # Từ chức năng / liên từ / giới từ
    'tại', 'ở', 'qua', 'với', 'cho', 'của', 'từ', 'đến', 'trong', 'trên', 'dưới',
    'vào', 'ra', 'bởi', 'theo', 'về', 'cùng', 'do', 'như', 'sang', 'lên', 'xuống',
    'và', 'hoặc', 'hay', 'nhưng', 'mà', 'thì', 'nếu', 'vì', 'tuy', 'dù', 'song',
    'nên', 'thế', 'vậy', 'để', 'nhằm',
    # Phó từ / lượng từ / hư từ
    'đã', 'đang', 'sẽ', 'vừa', 'mới', 'đều', 'cũng', 'rất', 'quá', 'lắm', 'nhất',
    'không', 'chưa', 'chẳng', 'được', 'bị', 'phải', 'có', 'thể', 'chỉ', 'luôn',
    'ngay', 'liền', 'hết', 'xong', 'rồi', 'lại', 'hơn', 'kém',
    # Đại từ / chỉ từ
    'này', 'đó', 'kia', 'ấy', 'nào', 'gì', 'ai', 'đây', 'nọ', 'đâu', 'sao',
    'các', 'những', 'mỗi', 'mọi', 'từng', 'cả', 'tất', 'mình', 'ta', 'nó', 'họ',
    # Danh từ đơn vị / danh từ phổ thông cơ bản
    'tỉnh', 'thành', 'huyện', 'thị', 'xã', 'phường', 'thôn', 'bản', 'ấp', 'khu',
    'số', 'ngày', 'tháng', 'năm', 'giờ', 'phút', 'giây', 'lần', 'câu', 'từ',
    'người', 'nhà', 'nơi', 'việc', 'cái', 'con', 'chiếc', 'bức', 'tấm', 'khối',
    # Tính từ cơ bản phổ biến
    'đúng', 'sai', 'tốt', 'xấu', 'lớn', 'nhỏ', 'nhiều', 'ít', 'cao', 'thấp',
    'xa', 'gần', 'nhanh', 'chậm', 'mới', 'cũ', 'sớm', 'muộn', 'rõ', 'sâu', 'sát',
    # Động từ cơ bản phổ biến
    'đi', 'làm', 'ăn', 'uống', 'nói', 'cười', 'viết', 'đọc', 'xem', 'nghe', 'thấy',
    'biết', 'hiểu', 'nhớ', 'quên', 'mua', 'bán', 'trả', 'nhận', 'gửi', 'lấy', 'bỏ',
    'đặt', 'mang', 'cầm', 'giữ', 'tìm', 'chọn', 'dùng', 'sống', 'chết', 'hỏi', 'xin',
    'bảo', 'gọi', 'mời', 'chờ', 'đợi', 'gặp', 'kéo', 'ghi', 'nêu',
    # Số đếm
    'một', 'hai', 'ba', 'bốn', 'năm', 'sáu', 'bảy', 'tám', 'chín', 'mười', 'trăm', 'nghìn', 'triệu', 'tỷ',
}

def eval_ctx(prev_token, curr_token, next_token, prev_prev_token='', next_next_token=''):
    curr_l = unicodedata.normalize('NFC', curr_token.lower())
    prev_l = unicodedata.normalize('NFC', prev_token.lower()) if prev_token else ''
    prev_prev_l = unicodedata.normalize('NFC', prev_prev_token.lower()) if prev_prev_token else ''

    # 1. Bảo vệ từ chức năng, từ vựng cơ bản, số đếm
    if curr_l in PROTECTED_COMMON_WORDS:
        return None

    # 2. Bảo vệ tên riêng (viết hoa liên tiếp)
    if prev_token and curr_token and prev_token[0].isupper() and curr_token[0].isupper():
        return None

    # 3. Đã là từ ghép chuẩn trong từ điển
    if prev_l and f'{prev_l} {curr_l}' in c._vietnamese_words:
        return None

    # 4. Ranh giới từ ghép trước
    prev_is_bound = bool(prev_prev_l and f'{prev_prev_l} {prev_l}' in c._vietnamese_words)
    if prev_is_bound or prev_l in PROTECTED_COMMON_WORDS or not prev_l:
        return None

    # 5. So khớp với tiền tố liền trước (chỉ khi prev_token có từ ghép với candidate)
    if prev_l in c._compound_prefixes:
        best_cand, min_dist = None, 999.0
        for cand in c._compound_prefixes[prev_l]:
            dist = weighted_edit_distance(curr_l, cand)
            if dist <= 0.6 and dist < min_dist:
                min_dist, best_cand = dist, cand
        if best_cand:
            sugg_fmt = best_cand.capitalize() if curr_token and curr_token[0].isupper() else best_cand
            return sugg_fmt, f"Nghi vấn nhầm ký tự trong cụm '{prev_token} {curr_token}' (gợi ý: '{prev_token} {sugg_fmt}')"
    return None

c._evaluate_context_confusion = eval_ctx

test_cases = [
    ('kỹ su', 1, 'sư'),
    ('lâu nơm', 1, 'năm'),
    ('Bí thu', 1, 'thư'),
    ('chuyên nghiêp', 1, 'nghiệp'),
    ('Kế họach kiểm tra và ra sóat', 2, 'hoạch'),
    ('Thực hiện nghiêm tuc', 1, 'túc'),
    ('Đang kiêmr tra dữ liệu', 1, 'kiểm'),
    ('Cần bổ sung đúng số hotline', 0, ''),
    ('để người dân được biết', 0, ''),
    ('Ủy ban nhân dân thành phố Hồ Chí Minh', 0, ''),
    ('Bộ Văn hóa, Thể thao và Du lịch', 0, ''),
    ('Liên hệ qua email lúc 08:30', 0, ''),
    ('thảo luận tại diễn đàn quốc tế', 0, ''),
]

print('--- RUNNING TEST CASES ---')
all_passed = True
for txt, expected_err_count, expected_sugg in test_cases:
    errs = c.check_text_spelling(txt)
    cnt = len(errs)
    pass_cnt = (cnt == expected_err_count)
    sugg_ok = True
    if expected_sugg and errs:
        sugg_ok = any(expected_sugg in err.suggestions for err in errs)
    status = 'PASS' if (pass_cnt and sugg_ok) else 'FAIL'
    if status == 'FAIL':
        all_passed = False
    print(f'[{status}] {txt[:40]:<40} -> count={cnt} (exp {expected_err_count}), suggs={[e.suggestions for e in errs]}')

print('\nOVERALL RESULT:', 'ALL PASSED' if all_passed else 'SOME FAILED')
