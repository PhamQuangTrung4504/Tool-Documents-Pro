import os
import sys
sys.path.insert(0, os.path.abspath("."))
import cv2
import numpy as np
from app.core.ocr_engine import OCREngine
from app.core.table_engines import extract_physical_table_topology

engine = OCREngine(lang="vi", use_gpu=False)
img = cv2.imread("assets/imagetest/bang2.png")
res = engine.recognize(img)
print("Detected tables:", len(res.tables))
if res.tables:
    tbl = res.tables[0]
    print("tbl.bbox:", tbl.bbox)
    print("num_rows:", tbl.num_rows, "num_cols:", tbl.num_cols)
    print("row_boxes lengths:", [len(rb) for rb in tbl.row_boxes])

for i, box in enumerate(res.boxes):
    if any(kw in box.text for kw in ["UY BAN", "DANH SÁCH", "Mẫu số", "Họ tên", "CỘNG HÒA"]):
        print(f"Box {i}: cy={box.cy}, bbox={box.bbox}, text='{box.text}'")
