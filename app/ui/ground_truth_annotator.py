"""Giao diện độc lập đối chiếu ảnh và nhập ground truth benchmark."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PyQt5.QtCore import QPoint, QRect, Qt
from PyQt5.QtGui import QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QMainWindow, QMessageBox, QPushButton, QPlainTextEdit,
    QScrollArea, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from app.core.ground_truth_annotation import (
    VerifiedGroundTruthOverwriteError,
    make_ground_truth,
    new_paragraph_block,
    new_table_block,
    save_manifest_safely,
    verify_ground_truth,
)


class SelectableImage(QLabel):
    """Hiển thị ảnh nguồn, cho phép khoanh vùng bằng tọa độ ảnh gốc."""

    def __init__(self, selected_callback):
        super().__init__()
        self._callback = selected_callback
        self._source_size = (0, 0)
        self._pixmap_rect = QRect()
        self._drag_start: Optional[QPoint] = None
        self._selection = QRect()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(500, 420)
        self.setMouseTracking(True)

    def show_image(self, path: Path) -> None:
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            raise ValueError(f"Không mở được ảnh: {path}")
        self._source_size = (pixmap.width(), pixmap.height())
        self._source_pixmap = pixmap
        self._fit_pixmap()

    def _fit_pixmap(self) -> None:
        if not hasattr(self, "_source_pixmap"):
            return
        shown = self._source_pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(shown)
        self._pixmap_rect = QRect(
            (self.width() - shown.width()) // 2,
            (self.height() - shown.height()) // 2,
            shown.width(), shown.height(),
        )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._fit_pixmap()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self._pixmap_rect.contains(event.pos()):
            self._drag_start = event.pos()
            self._selection = QRect(self._drag_start, self._drag_start)
            self.update()

    def mouseMoveEvent(self, event) -> None:
        if self._drag_start is not None:
            self._selection = QRect(self._drag_start, event.pos()).normalized().intersected(self._pixmap_rect)
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if self._drag_start is None:
            return
        self._selection = QRect(self._drag_start, event.pos()).normalized().intersected(self._pixmap_rect)
        self._drag_start = None
        self.update()
        if self._selection.width() < 2 or self._selection.height() < 2:
            return
        sx = self._source_size[0] / self._pixmap_rect.width()
        sy = self._source_size[1] / self._pixmap_rect.height()
        x1 = round((self._selection.left() - self._pixmap_rect.left()) * sx)
        y1 = round((self._selection.top() - self._pixmap_rect.top()) * sy)
        x2 = round((self._selection.right() - self._pixmap_rect.left()) * sx)
        y2 = round((self._selection.bottom() - self._pixmap_rect.top()) * sy)
        bbox = [max(0, x1), max(0, y1), min(self._source_size[0], x2), min(self._source_size[1], y2)]
        self._callback(bbox)

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if not self._selection.isNull():
            painter = QPainter(self)
            painter.setPen(QPen(Qt.red, 2, Qt.DashLine))
            painter.drawRect(self._selection)


class GroundTruthAnnotator(QMainWindow):
    def __init__(self, manifest_path: Path):
        super().__init__()
        self.setWindowTitle("Đối chiếu Ground Truth OCR")
        self.resize(1450, 900)
        self.manifest_path = manifest_path.expanduser().resolve()
        self.payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.documents: List[Dict[str, Any]] = self.payload.get("documents", [])
        if not self.documents:
            raise ValueError("Manifest không có documents")
        self.current_index = -1
        self.blocks: List[Dict[str, Any]] = []
        self.dirty_docs: set[int] = set()
        self.explicitly_verified: set[int] = set()
        self.selected_bbox: Optional[List[int]] = None
        self.loading = False
        self._build_ui()
        self._load_doc(0)

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)
        top = QHBoxLayout()
        self.doc_combo = QComboBox()
        for item in self.documents:
            self.doc_combo.addItem(f"{item.get('id', '?')} — {Path(item.get('image', '')).name}")
        self.doc_combo.currentIndexChanged.connect(self._switch_doc)
        top.addWidget(QLabel("Tài liệu:"))
        top.addWidget(self.doc_combo, 1)
        self.status_label = QLabel()
        top.addWidget(self.status_label)
        layout.addLayout(top)

        splitter = QSplitter(Qt.Horizontal)
        image_panel = QWidget()
        image_layout = QVBoxLayout(image_panel)
        self.image_view = SelectableImage(self._set_selected_bbox)
        scroll = QScrollArea()
        scroll.setWidget(self.image_view)
        scroll.setWidgetResizable(False)
        image_layout.addWidget(scroll, 1)
        self.crop_view = QLabel("Kéo chuột trên ảnh để chọn vùng")
        self.crop_view.setAlignment(Qt.AlignCenter)
        self.crop_view.setMinimumHeight(130)
        self.crop_view.setMaximumHeight(190)
        image_layout.addWidget(self.crop_view)
        splitter.addWidget(image_panel)

        edit_panel = QWidget()
        edit_layout = QVBoxLayout(edit_panel)
        block_buttons = QHBoxLayout()
        for title, callback in (("+ Đoạn", self._add_paragraph), ("+ Bảng", self._add_table),
                                ("Lên", self._move_up), ("Xuống", self._move_down), ("Xóa", self._delete_block)):
            button = QPushButton(title)
            button.clicked.connect(callback)
            block_buttons.addWidget(button)
        edit_layout.addLayout(block_buttons)
        self.block_list = QListWidget()
        self.block_list.currentRowChanged.connect(self._select_block)
        edit_layout.addWidget(self.block_list, 1)

        self.block_text = QPlainTextEdit()
        self.block_text.setPlaceholderText("Nhập nội dung đã đối chiếu trực tiếp với ảnh nguồn")
        self.block_text.textChanged.connect(self._text_changed)
        edit_layout.addWidget(self.block_text, 2)

        table_controls = QHBoxLayout()
        for title, callback in (("Thêm hàng", self._add_row), ("Thêm cột", self._add_column),
                                ("Gộp ô rỗng", self._merge_cells), ("Tách ô", self._unmerge_cell)):
            button = QPushButton(title)
            button.clicked.connect(callback)
            table_controls.addWidget(button)
        edit_layout.addLayout(table_controls)
        self.table = QTableWidget(0, 0)
        self.table.itemChanged.connect(self._table_changed)
        edit_layout.addWidget(self.table, 3)
        diagonal_row = QHBoxLayout()
        self.diagonal_combo = QComboBox()
        self.diagonal_combo.addItem("Không có đường chéo", "")
        self.diagonal_combo.addItem("Trên trái đến dưới phải", "tl2br")
        self.diagonal_combo.addItem("Trên phải đến dưới trái", "tr2bl")
        diagonal_row.addWidget(self.diagonal_combo, 1)
        diagonal_button = QPushButton("Gán cho ô đang chọn")
        diagonal_button.clicked.connect(self._set_diagonal)
        diagonal_row.addWidget(diagonal_button)
        edit_layout.addLayout(diagonal_row)
        self.review_block_button = QPushButton("Đánh dấu khối đã đối chiếu")
        self.review_block_button.clicked.connect(self._mark_block_reviewed)
        edit_layout.addWidget(self.review_block_button)

        details = QFormLayout()
        self.reviewer = QLineEdit()
        self.review_date = QLineEdit(date.today().isoformat())
        self.source_note = QLineEdit()
        details.addRow("Người kiểm tra:", self.reviewer)
        details.addRow("Ngày kiểm tra:", self.review_date)
        details.addRow("Nguồn đối chiếu:", self.source_note)
        edit_layout.addLayout(details)
        self.attach_region_button = QPushButton("Gắn vùng ảnh đã khoanh vào khối/ô")
        self.attach_region_button.clicked.connect(self._attach_region)
        edit_layout.addWidget(self.attach_region_button)
        verify_row = QHBoxLayout()
        self.verify_button = QPushButton("Xác minh Ground Truth của tài liệu")
        self.verify_button.clicked.connect(self._verify_document)
        self.export_button = QPushButton("Xuất manifest…")
        self.export_button.clicked.connect(self._export)
        verify_row.addWidget(self.verify_button)
        verify_row.addWidget(self.export_button)
        edit_layout.addLayout(verify_row)
        splitter.addWidget(edit_panel)
        splitter.setSizes([760, 680])
        layout.addWidget(splitter, 1)
        self.setCentralWidget(root)

    def _current_document(self) -> Dict[str, Any]:
        return self.documents[self.current_index]

    def _switch_doc(self, index: int) -> None:
        if index < 0 or index == self.current_index:
            return
        self._commit_current_document()
        self.current_index = index
        self._load_doc(index)

    def _load_doc(self, index: int) -> None:
        self.loading = True
        self.current_index = index
        document = self.documents[index]
        image_path = Path(document["image"])
        if not image_path.is_absolute():
            image_path = self.manifest_path.parent / image_path
        self.image_view.show_image(image_path.resolve())
        truth = document.get("ground_truth") or {}
        raw_blocks = truth.get("blocks")
        if isinstance(raw_blocks, list) and raw_blocks:
            self.blocks = raw_blocks
        else:
            self.blocks = [new_paragraph_block(truth.get("text") or "")]
            if truth.get("table"):
                self.status_label.setText("Ground Truth cũ có bảng nhưng chưa tách khối; hãy thêm bảng và nhập/đối chiếu lưới.")
        self.block_list.clear()
        for block in self.blocks:
            self.block_list.addItem(self._block_label(block))
        self.reviewer.setText(truth.get("reviewed_by", ""))
        self.review_date.setText(truth.get("reviewed_at") or date.today().isoformat())
        self.source_note.setText(truth.get("source_note", ""))
        self.loading = False
        self.block_list.setCurrentRow(0)
        self._refresh_status()

    @staticmethod
    def _block_label(block: Dict[str, Any]) -> str:
        return "Bảng" if block.get("kind") == "table" else "Đoạn văn"

    def _refresh_status(self) -> None:
        truth = self._current_document().get("ground_truth") or {}
        verified = truth.get("verified") is True and (
            self.current_index not in self.dirty_docs or self.current_index in self.explicitly_verified
        )
        done = sum(block.get("reviewed") is True for block in self.blocks)
        self.status_label.setText(f"{'ĐÃ XÁC MINH' if verified else 'BẢN NHÁP'} — đã đối chiếu {done}/{len(self.blocks)} khối")

    def _sync_active_block(self) -> None:
        if self.current_index < 0 or not self.blocks:
            return
        row = self.block_list.currentRow()
        if not 0 <= row < len(self.blocks):
            return
        block = self.blocks[row]
        if block["kind"] == "paragraph":
            block["text"] = self.block_text.toPlainText()
        else:
            grid = []
            for r in range(self.table.rowCount()):
                grid.append([
                    self.table.item(r, c).text() if self.table.item(r, c) else ""
                    for c in range(self.table.columnCount())
                ])
            block["grid"] = grid

    def _commit_current_document(self) -> None:
        if self.current_index < 0 or self.current_index not in self.dirty_docs:
            return
        self._sync_active_block()
        self._current_document()["ground_truth"] = make_ground_truth(
            self.blocks,
            verified=self.current_index in self.explicitly_verified,
            reviewed_by=self.reviewer.text(), reviewed_at=self.review_date.text(),
            source_note=self.source_note.text(),
        )

    def _select_block(self, row: int) -> None:
        if self.loading or row < 0 or row >= len(self.blocks):
            return
        old_row = getattr(self, "_displayed_row", None)
        if old_row is not None and 0 <= old_row < len(self.blocks):
            previous = self.blocks[old_row]
            if previous["kind"] == "paragraph":
                previous["text"] = self.block_text.toPlainText()
            else:
                self._capture_table(previous)
        block = self.blocks[row]
        self._displayed_row = row
        self.block_text.blockSignals(True)
        self.table.blockSignals(True)
        is_table = block["kind"] == "table"
        self.block_text.setVisible(not is_table)
        self.table.setVisible(is_table)
        for button in self.findChildren(QPushButton):
            if button.text() in {"Thêm hàng", "Thêm cột", "Gộp ô rỗng", "Tách ô", "Gán cho ô đang chọn"}:
                button.setVisible(is_table)
        if is_table:
            grid = block["grid"]
            self.table.setRowCount(len(grid))
            self.table.setColumnCount(len(grid[0]) if grid else 0)
            for r, values in enumerate(grid):
                for c, value in enumerate(values):
                    self.table.setItem(r, c, QTableWidgetItem(value))
            for r0, c0, r1, c1 in block.get("cell_spans", []):
                self.table.setSpan(r0, c0, r1 - r0, c1 - c0)
        else:
            self.block_text.setPlainText(block.get("text", ""))
        self.block_text.blockSignals(False)
        self.table.blockSignals(False)
        self._refresh_status()

    def _capture_table(self, block: Dict[str, Any]) -> None:
        grid = []
        for r in range(self.table.rowCount()):
            grid.append([self.table.item(r, c).text() if self.table.item(r, c) else ""
                         for c in range(self.table.columnCount())])
        block["grid"] = grid

    def _mark_changed(self) -> None:
        if self.loading or self.current_index < 0:
            return
        self.dirty_docs.add(self.current_index)
        self.explicitly_verified.discard(self.current_index)
        if self.blocks and 0 <= self.block_list.currentRow() < len(self.blocks):
            self.blocks[self.block_list.currentRow()]["reviewed"] = False
        truth = self._current_document().get("ground_truth")
        if isinstance(truth, dict):
            truth["verified"] = False
        self._refresh_status()

    def _text_changed(self) -> None:
        if not self.loading:
            row = self.block_list.currentRow()
            if 0 <= row < len(self.blocks) and self.blocks[row]["kind"] == "paragraph":
                self.blocks[row]["text"] = self.block_text.toPlainText()
                self._mark_changed()

    def _table_changed(self, _item) -> None:
        if not self.loading:
            row = self.block_list.currentRow()
            if 0 <= row < len(self.blocks) and self.blocks[row]["kind"] == "table":
                self._capture_table(self.blocks[row])
                self._mark_changed()

    def _add_block(self, block: Dict[str, Any]) -> None:
        self._sync_active_block()
        self.blocks.append(block)
        self.block_list.addItem(self._block_label(block))
        self.block_list.setCurrentRow(len(self.blocks) - 1)
        self._mark_changed()

    def _add_paragraph(self) -> None:
        self._add_block(new_paragraph_block())

    def _add_table(self) -> None:
        if any(block.get("kind") == "table" for block in self.blocks):
            QMessageBox.information(self, "Giới hạn manifest", "Benchmark hiện chấm một bảng cho mỗi tài liệu.")
            return
        self._add_block(new_table_block())

    def _delete_block(self) -> None:
        row = self.block_list.currentRow()
        if row < 0:
            return
        self.blocks.pop(row)
        self.block_list.takeItem(row)
        self._displayed_row = None
        self._mark_changed()
        if self.blocks:
            self.block_list.setCurrentRow(min(row, len(self.blocks) - 1))

    def _move_block(self, offset: int) -> None:
        row = self.block_list.currentRow()
        target = row + offset
        if not (0 <= row < len(self.blocks) and 0 <= target < len(self.blocks)):
            return
        self._sync_active_block()
        self.blocks[row], self.blocks[target] = self.blocks[target], self.blocks[row]
        item = self.block_list.takeItem(row)
        self.block_list.insertItem(target, item)
        self._displayed_row = None
        self.block_list.setCurrentRow(target)
        self._mark_changed()

    def _move_up(self) -> None:
        self._move_block(-1)

    def _move_down(self) -> None:
        self._move_block(1)

    def _mark_block_reviewed(self) -> None:
        self._sync_active_block()
        row = self.block_list.currentRow()
        if row >= 0:
            self.blocks[row]["reviewed"] = True
            self.dirty_docs.add(self.current_index)
            self._refresh_status()

    def _add_row(self) -> None:
        self._sync_active_block()
        row = self.table.currentRow()
        self.table.insertRow(row + 1 if row >= 0 else self.table.rowCount())
        for c in range(self.table.columnCount()):
            self.table.setItem(self.table.rowCount() - 1, c, QTableWidgetItem(""))
        self._mark_changed()

    def _add_column(self) -> None:
        self._sync_active_block()
        col = self.table.currentColumn()
        self.table.insertColumn(col + 1 if col >= 0 else self.table.columnCount())
        for r in range(self.table.rowCount()):
            self.table.setItem(r, self.table.columnCount() - 1, QTableWidgetItem(""))
        self._mark_changed()

    def _merge_cells(self) -> None:
        indexes = self.table.selectedIndexes()
        if len(indexes) < 2:
            return
        rows = [item.row() for item in indexes]
        cols = [item.column() for item in indexes]
        r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
        if len(indexes) != (r1 - r0 + 1) * (c1 - c0 + 1):
            QMessageBox.warning(self, "Không thể gộp", "Hãy chọn một vùng hình chữ nhật liền nhau.")
            return
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if (r, c) != (r0, c0) and self.table.item(r, c) and self.table.item(r, c).text():
                    QMessageBox.warning(self, "Không thể gộp", "Các ô bị che phải rỗng để tránh mất nội dung.")
                    return
        self.table.setSpan(r0, c0, r1 - r0 + 1, c1 - c0 + 1)
        block = self.blocks[self.block_list.currentRow()]
        span = [r0, c0, r1 + 1, c1 + 1]
        if span not in block.setdefault("cell_spans", []):
            block["cell_spans"].append(span)
        self._mark_changed()

    def _unmerge_cell(self) -> None:
        row, col = self.table.currentRow(), self.table.currentColumn()
        if row < 0 or col < 0:
            return
        r0, c0 = row, col
        for span in self.blocks[self.block_list.currentRow()].get("cell_spans", []):
            if span[0] <= row < span[2] and span[1] <= col < span[3]:
                r0, c0 = span[0], span[1]
                self.blocks[self.block_list.currentRow()]["cell_spans"].remove(span)
                break
        self.table.setSpan(r0, c0, 1, 1)
        self._mark_changed()

    def _set_diagonal(self) -> None:
        row, col = self.table.currentRow(), self.table.currentColumn()
        if row < 0 or col < 0:
            return
        block = self.blocks[self.block_list.currentRow()]
        block["diagonal_cells"] = [item for item in block.get("diagonal_cells", [])
                                   if item[0] != row or item[1] != col]
        direction = self.diagonal_combo.currentData()
        if direction:
            block["diagonal_cells"].append([row, col, direction])
        self._mark_changed()

    def _set_selected_bbox(self, bbox: List[int]) -> None:
        self.selected_bbox = bbox
        image = self.image_view._source_pixmap.copy(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1])
        self.crop_view.setPixmap(image.scaled(self.crop_view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _attach_region(self) -> None:
        if self.selected_bbox is None:
            QMessageBox.information(self, "Chưa chọn vùng", "Kéo chuột trên ảnh để chọn vùng trước.")
            return
        self._sync_active_block()
        row = self.block_list.currentRow()
        if row < 0:
            return
        block = self.blocks[row]
        if block["kind"] == "table" and self.table.currentRow() >= 0 and self.table.currentColumn() >= 0:
            block.setdefault("cell_bboxes", {})[f"{self.table.currentRow()},{self.table.currentColumn()}"] = self.selected_bbox
        else:
            block["bbox"] = self.selected_bbox
        self._mark_changed()

    def _verify_document(self) -> None:
        self._sync_active_block()
        try:
            truth = verify_ground_truth(self.blocks, self.reviewer.text(), self.review_date.text(), self.source_note.text())
        except ValueError as exc:
            QMessageBox.warning(self, "Chưa thể xác minh", str(exc))
            return
        self._current_document()["ground_truth"] = truth
        self.explicitly_verified.add(self.current_index)
        self.dirty_docs.add(self.current_index)
        self._refresh_status()

    def _export(self) -> None:
        self._commit_current_document()
        path_text, _ = QFileDialog.getSaveFileName(
            self, "Xuất manifest Ground Truth", str(self.manifest_path), "JSON (*.json)"
        )
        if not path_text:
            return
        target = Path(path_text)
        if self.current_index in self.dirty_docs or not self._current_document().get("ground_truth"):
            previous_truth = self._current_document().get("ground_truth") or {}
            updated_truth = make_ground_truth(
                self.blocks,
                verified=self.current_index in self.explicitly_verified,
                reviewed_by=self.reviewer.text(), reviewed_at=self.review_date.text(),
                source_note=self.source_note.text(),
            )
            # Bản manifest cũ có thể chứa nhãn bảng nhưng chưa có blocks. Giữ
            # nguyên nhãn đó khi người dùng chỉ sửa phần văn bản và chưa nhập lưới mới.
            if updated_truth["table"] is None and previous_truth.get("table") is not None:
                updated_truth["table"] = previous_truth["table"]
            self._current_document()["ground_truth"] = updated_truth
        try:
            save_manifest_safely(target, self.payload)
        except VerifiedGroundTruthOverwriteError:
            answer = QMessageBox.question(
                self, "Ghi đè Ground Truth đã xác minh?",
                "Tệp đích chứa Ground Truth đã xác minh. Bạn có đồng ý thay thế tệp này không?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
            save_manifest_safely(target, self.payload, allow_replace_verified=True)
        except Exception as exc:
            QMessageBox.critical(self, "Không thể xuất manifest", str(exc))
            return
        self.manifest_path = target.expanduser().resolve()
        self.dirty_docs.clear()
        self._refresh_status()
        QMessageBox.information(self, "Đã xuất", f"Đã lưu manifest: {self.manifest_path}")


def run_annotator(manifest_path: Path) -> int:
    app = QApplication.instance() or QApplication([])
    window = GroundTruthAnnotator(manifest_path)
    window.show()
    return app.exec_()


__all__ = ["GroundTruthAnnotator", "SelectableImage", "run_annotator"]
