import os
import sys
sys.path.insert(0, os.path.abspath("."))
import cv2
from app.core.ocr_engine import OCREngine
from app.core.table_engines import extract_physical_table_topology

engine = OCREngine(lang="vi", use_gpu=False)
img = cv2.imread("assets/imagetest/bang2.png")
# Get paddle boxes without vietocr to be fast
engine._ensure_loaded()
# We can just get detector results
results = engine._ocr.predict(img)
polygons = []
for res in results:
    dt_polys = res.get("dt_polys", []) if hasattr(res, "get") else getattr(res, "dt_polys", [])
    rec_texts = res.get("rec_texts", []) if hasattr(res, "get") else getattr(res, "rec_texts", [])
    for i, poly in enumerate(dt_polys):
        polygons.append((poly, rec_texts[i] if i < len(rec_texts) else ""))

from app.core.models import OCRBox
boxes = []
for poly, text in polygons:
    bbox_int = [[int(pt[0]), int(pt[1])] for pt in poly[:4]]
    boxes.append(OCRBox(text=text, confidence=0.9, bbox=bbox_int, raw_text=text))

tbl = extract_physical_table_topology(img, boxes)
print("tbl.bbox:", tbl.bbox if tbl else None)

from app.core.layout_analyzer import reconstruct_document_layout
layout_res = reconstruct_document_layout(boxes, image=img)
grouped_lines = layout_res[0]
print("--- ALL GROUPED LINES (COUNT:", len(grouped_lines), ") ---")
for idx, line in enumerate(grouped_lines):
    line_txt = " ".join(b.text for b in line)
    print(f"Line {idx}: {line_txt[:80]}")
