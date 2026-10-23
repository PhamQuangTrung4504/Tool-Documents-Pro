"""Kiểm tra IPC spawn bằng engine giả; không phải kiểm thử OCR model thật."""
from __future__ import annotations

import json
import time
import unittest

import numpy as np
from PyQt5.QtCore import QEventLoop, QTimer
from PyQt5.QtWidgets import QApplication

from app.core.ocr_engine import OCRCancelledError
from app.core.ocr_process_worker import _json_value, ocr_result_fingerprint
from app.core.models import DocumentBlock, OCRBox, OCRResult, OCRTable, TableCell
from app.ui.ocr_process_adapter import OCRProcessClient, OCRProcessWorker


def _fake_spawn_worker(connection, cancel_event):
    """Worker spawn-only để kiểm tra hợp đồng, retry và hủy mà không nạp model."""
    import os
    connection.send(json.dumps({"type": "ready", "pid": os.getpid()}))
    while True:
        try:
            request = json.loads(connection.recv())
        except (EOFError, OSError):
            return
        if request.get("type") == "shutdown":
            return
        if request.get("type") != "run":
            continue
        job_id = request["job_id"]
        image = np.load(request["image_path"], allow_pickle=False)
        if int(image[0, 0, 0]) == 2:
            connection.send(json.dumps({"type": "error", "job_id": job_id,
                                        "cancelled": False, "error_type": "SyntheticFailure",
                                        "error": "synthetic"}))
            continue
        if int(image[0, 0, 0]) == 3:
            # Giả lập sập native/process đột ngột: không gửi thông điệp kết thúc.
            os._exit(23)
        if int(image[0, 0, 0]) == 1:
            while not cancel_event.wait(0.02):
                pass
            connection.send(json.dumps({"type": "error", "job_id": job_id,
                                        "cancelled": True, "error": "cancelled"}))
            continue
        now = time.perf_counter_ns()
        connection.send(json.dumps({"type": "stage", "job_id": job_id, "stage": "detection",
                                    "transition": "start", "timestamp_ns": now}))
        result = OCRResult(
            boxes=[OCRBox(text="sample", raw_text="sample", confidence=0.0,
                          bbox=[[0, 0], [5, 0], [5, 2], [0, 2]],
                          metadata={"evidence": {"source": "fake", "sequence": [1, 2]}})],
            full_text="sample", document_blocks=[DocumentBlock(kind="paragraph", order=0, text="sample", box_indices=[0])],
            tables=[OCRTable(grid=[["sample", ""]], bbox=[0, 0, 10, 4], num_rows=1, num_cols=2,
                             header_rows=[0], cells=[TableCell(row_start=0, col_start=0, row_end=1,
                                                                col_end=2, text="sample", boxes=[])])],
        )
        connection.send(json.dumps({"type": "result", "job_id": job_id,
                                    "elapsed_seconds": 0.01, "result": _json_value(result)}))


class OCRProcessAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Dùng QApplication để tương thích khi suite này chạy cùng các test UI.
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.client = OCRProcessClient(worker_target=_fake_spawn_worker)

    def tearDown(self):
        self.client.shutdown(timeout=1.0)

    def _run_worker(self, marker: int, generation: int):
        worker = OCRProcessWorker(self.client, np.full((4, 5, 3), marker, dtype=np.uint8), {}, generation=generation)
        messages, stages = [], []
        loop = QEventLoop()
        worker.result_ready.connect(lambda result: (messages.append(result), loop.quit()))
        worker.stage_event.connect(lambda *event: stages.append(event))
        timeout = QTimer()
        timeout.setSingleShot(True)
        timeout.timeout.connect(loop.quit)
        timeout.start(5000)
        worker.start()
        loop.exec_()
        timeout.stop()
        if worker.isRunning():
            worker.wait(1000)
        self.assertFalse(worker.isRunning())
        self.assertEqual(len(messages), 1)
        return messages[0], stages

    def test_spawn_ipc_round_trip_persists_worker_and_preserves_result_metadata(self):
        result, stages = self._run_worker(0, 71)
        first_pid = self.client._process.pid
        self.assertIsInstance(result, OCRResult)
        self.assertEqual(result.boxes[0].text, "sample")
        self.assertEqual(result.boxes[0].confidence, 0.0)
        self.assertEqual(result.boxes[0].metadata["evidence"]["sequence"], [1, 2])
        self.assertEqual(result.document_blocks[0].box_indices, [0])
        self.assertEqual(result.tables[0].cells[0].col_span, 2)
        self.assertEqual(result.tables[0].header_rows, [0])
        self.assertRegex(ocr_result_fingerprint(result), r"^[0-9a-f]{64}$")
        self.assertEqual(stages[0][0:3], (71, "detection", "start"))

        retry, _ = self._run_worker(0, 72)
        self.assertIsInstance(retry, OCRResult)
        self.assertEqual(self.client._process.pid, first_pid)

    def test_cancel_and_retry_keep_worker_alive_without_stale_job_id(self):
        worker = OCRProcessWorker(self.client, np.ones((2, 2, 3), dtype=np.uint8), {}, generation=80)
        results = []
        loop = QEventLoop()
        worker.result_ready.connect(lambda result: (results.append(result), loop.quit()))
        worker.start()
        QTimer.singleShot(100, worker.request_cancel)
        QTimer.singleShot(5000, loop.quit)
        loop.exec_()
        if worker.isRunning():
            worker.wait(1000)
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], OCRCancelledError)
        self.assertTrue(self.client.is_running)

        retry, _ = self._run_worker(0, 81)
        self.assertIsInstance(retry, OCRResult)

    def test_job_error_does_not_kill_persistent_worker_and_retry_succeeds(self):
        error, _ = self._run_worker(2, 90)
        self.assertIsInstance(error, RuntimeError)
        self.assertIn("SyntheticFailure", str(error))
        self.assertTrue(self.client.is_running)
        result, _ = self._run_worker(0, 91)
        self.assertIsInstance(result, OCRResult)

    def test_worker_crash_is_reported_and_retry_starts_a_new_child(self):
        error, _ = self._run_worker(3, 95)
        self.assertIsInstance(error, RuntimeError)
        self.assertIn("IPC", str(error))
        self.assertFalse(self.client.is_running)

        result, _ = self._run_worker(0, 96)
        self.assertIsInstance(result, OCRResult)
        self.assertTrue(self.client.is_running)

    def test_timeout_requests_cancel_and_terminates_unresponsive_worker(self):
        self.client = OCRProcessClient(worker_target=_fake_spawn_worker,
                                       job_timeout_seconds=0.1, cancel_grace_seconds=0.2)
        result, _ = self._run_worker(1, 100)
        self.assertIsInstance(result, TimeoutError)
        self.assertFalse(self.client.is_running)

    def test_shutdown_reaps_persistent_child(self):
        result, _ = self._run_worker(0, 110)
        self.assertIsInstance(result, OCRResult)
        self.client.shutdown(timeout=2.0)
        self.assertFalse(self.client.is_running)


if __name__ == "__main__":
    unittest.main()
