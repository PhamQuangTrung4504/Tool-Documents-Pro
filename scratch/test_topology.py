import os
import sys
sys.path.insert(0, os.path.abspath("."))
import cv2
from app.core.table_engines import extract_physical_table_topology

img = cv2.imread("assets/imagetest/bang2.png")
# Dummy boxes to see grid and bbox
tbl = extract_physical_table_topology(img, [])
if tbl:
    print("tbl.bbox:", tbl.bbox)
    print("tbl.num_rows:", tbl.num_rows, "tbl.num_cols:", tbl.num_cols)
