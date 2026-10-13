import os
import sys
import unicodedata

# 1. Load syllables
syll_path = os.path.join('app', 'core', 'vietnamese_syllables.txt')
with open(syll_path, 'r', encoding='utf-8') as f:
    VALID_SYLLABLES = set(unicodedata.normalize('NFC', l.strip().lower()) for l in f if l.strip())

# Add traditional tone variations
consonants = ['', 'b', 'c', 'ch', 'd', 'đ', 'g', 'gh', 'h', 'k', 'kh', 'l', 'm', 'n', 'ng', 'ngh', 'nh', 'p', 'ph', 'qu', 'r', 's', 't', 'th', 'tr', 'v', 'x']
for c in consonants:
    for old_v, new_v in [
        ('oà', 'òa'), ('oá', 'óa'), ('oả', 'ỏa'), ('oã', 'õa'), ('oạ', 'ọa'),
        ('oè', 'òe'), ('oé', 'óe'), ('oẻ', 'ỏe'), ('oẽ', 'õe'), ('oẹ', 'ọe'),
        ('uỳ', 'ùy'), ('uý', 'úy'), ('uỷ', 'ủy'), ('uỹ', 'ũy'), ('uỵ', 'ụy'),
    ]:
        if unicodedata.normalize('NFC', c + new_v) in VALID_SYLLABLES:
            VALID_SYLLABLES.add(unicodedata.normalize('NFC', c + old_v))

# 2. Load compound words
words_path = os.path.join('app', 'core', 'vietnamese_words.txt')
VIETNAMESE_WORDS = set()
COMPOUND_PREFIXES = {}
COMPOUND_SUFFIXES = {}

with open(words_path, 'r', encoding='utf-8') as f:
    for line in f:
        w = unicodedata.normalize('NFC', line.strip().lower())
        if w:
            VIETNAMESE_WORDS.add(w)
            parts = w.split()
            if len(parts) == 2:
                w1, w2 = parts
                COMPOUND_PREFIXES.setdefault(w1, set()).add(w2)
                COMPOUND_SUFFIXES.setdefault(w2, set()).add(w1)

# Word frequency priority for ranking candidates
WORD_PRIORITY = {
    'được': 100, 'trong': 95, 'và': 95, 'các': 90, 'có': 90, 'là': 90, 'của': 90,
    'người': 85, 'với': 85, 'cho': 85, 'năm': 85, 'này': 80, 'đã': 80, 'khi': 80,
    'đáp': 75, 'kiểm': 75, 'soát': 75, 'hoạch': 75, 'thực': 75, 'sư': 75, 'nghiệp': 75,
}

print(f'Ready! Syllables: {len(VALID_SYLLABLES)}, Words: {len(VIETNAMESE_WORDS)}')
