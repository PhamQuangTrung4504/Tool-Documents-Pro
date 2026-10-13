import os
import unicodedata
from scratch.test_suggestions import (
    VALID_SYLLABLES, VIETNAMESE_WORDS, COMPOUND_PREFIXES, COMPOUND_SUFFIXES,
    weighted_edit_distance, get_suggestions
)

def check_context_error(prev_w: str, curr_w: str, next_w: str):
    """Check if curr_w has a contextual error with prev_w or next_w."""
    curr_norm = unicodedata.normalize('NFC', curr_w.lower())
    prev_norm = unicodedata.normalize('NFC', prev_w.lower()) if prev_w else ""
    next_norm = unicodedata.normalize('NFC', next_w.lower()) if next_w else ""
    
    # If the bigram already exists in VIETNAMESE_WORDS, it's definitely fine
    if prev_norm and f"{prev_norm} {curr_norm}" in VIETNAMESE_WORDS:
        return None
    if next_norm and f"{curr_norm} {next_norm}" in VIETNAMESE_WORDS:
        return None
        
    # Check if prev_norm forms a compound with a candidate c very close to curr_norm
    if prev_norm and prev_norm in COMPOUND_PREFIXES:
        best_cand = None
        min_dist = 999.0
        for c in COMPOUND_PREFIXES[prev_norm]:
            d = weighted_edit_distance(curr_norm, c)
            if d <= 0.7 and d < min_dist:
                min_dist = d
                best_cand = c
        if best_cand:
            return best_cand, f"Nghi vấn sai ngữ cảnh trong cụm '{prev_w} {curr_w}' (gợi ý: '{prev_w} {best_cand}')"
            
    # Check if next_norm forms a compound with a candidate c very close to curr_norm
    if next_norm and next_norm in COMPOUND_SUFFIXES:
        best_cand = None
        min_dist = 999.0
        for c in COMPOUND_SUFFIXES[next_norm]:
            d = weighted_edit_distance(curr_norm, c)
            if d <= 0.7 and d < min_dist:
                min_dist = d
                best_cand = c
        if best_cand:
            return best_cand, f"Nghi vấn sai ngữ cảnh trong cụm '{curr_w} {next_w}' (gợi ý: '{best_cand} {next_w}')"
            
    return None

# Test sentences
sentences = [
    ("kỹ", "su", "giỏi"),
    ("lâu", "nơm", "trong"),
    ("chuyên", "nghiệp", "cao"),
    ("đi", "chợ", "mua"),
    ("chợ", "mua", "hoa"),
    ("cái", "bô", "nhôm"),
    ("phu", "thê", "tình"),
    ("bộc", "lộ", "tình"),
]

for p, c, n in sentences:
    res = check_context_error(p, c, n)
    print(f"'{p} {c} {n}' -> {res}")
