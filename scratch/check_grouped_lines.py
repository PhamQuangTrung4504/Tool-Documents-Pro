import os
import sys
sys.path.insert(0, os.path.abspath("."))
import cv2
from app.core.ocr_engine import OCREngine
from app.core.layout_analyzer import reconstruct_document_layout

engine = OCREngine(lang="vi", use_gpu=False)
img = cv2.imread("assets/imagetest/bang2.png")
engine._ensure_loaded()
# We can load boxes directly using paddle
raw_boxes = engine.recognize(img).boxes
layout_res = reconstruct_document_layout(raw_boxes, image=img)
grouped_lines = layout_res[0]
tables = layout_res[1]
print("Total grouped_lines:", len(grouped_lines))
for idx, line in enumerate(grouped_lines):
    line_txt = " ".join(b.text for b in line)
    print(f"Line {idx}: {line_txt[:80]}")
