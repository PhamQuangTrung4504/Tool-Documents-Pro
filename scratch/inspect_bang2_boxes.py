import os
import sys
sys.path.insert(0, os.path.abspath("."))
import cv2
import numpy as np
from app.core.ocr_engine import OCREngine

engine = OCREngine(lang="vi", use_gpu=False)
img = cv2.imread("assets/imagetest/bang2.png")
res = engine.recognize(img)
for i, box in enumerate(res.boxes):
    cy = (box.bbox[0][1] + box.bbox[2][1]) / 2.0
    cx = (box.bbox[0][0] + box.bbox[1][0]) / 2.0
    if any(kw in box.text for kw in ["UY BAN", "DANH SÁCH", "Mẫu số", "Họ tên", "CỘNG HÒA", "Công nhận", "Ban hành"]):
        print(f"Box {i}: cy={cy:.1f}, cx={cx:.1f}, bbox={box.bbox}, text='{box.text}'")
