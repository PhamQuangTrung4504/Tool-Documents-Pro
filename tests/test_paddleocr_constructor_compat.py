"""Kiểm tra ánh xạ constructor theo API PaddleOCR mà không nạp model native."""
from __future__ import annotations

import unittest

from app.core.ocr_engine import _create_paddleocr_detector


class PaddleOCRConstructorCompatibilityTests(unittest.TestCase):
    def _kwargs(self):
        return dict(
            lang="vi", use_gpu=False, use_angle_cls=True,
            det_db_thresh=0.31, det_db_box_thresh=0.52,
            det_db_unclip_ratio=1.8, drop_score=0.27,
            rec_batch_num=5, ocr_version="PP-OCRv3",
        )

    def test_modern_api_uses_supported_names_and_cpu_device(self):
        class ModernPaddleOCR:
            def __init__(self, *, lang=None, ocr_version=None,
                         use_doc_orientation_classify=None, use_doc_unwarping=None,
                         use_textline_orientation=None, text_det_thresh=None,
                         text_det_box_thresh=None, text_det_unclip_ratio=None,
                         text_rec_score_thresh=None, text_recognition_batch_size=None,
                         **kwargs):
                self.kwargs = {
                    "lang": lang, "ocr_version": ocr_version,
                    "use_doc_orientation_classify": use_doc_orientation_classify,
                    "use_doc_unwarping": use_doc_unwarping,
                    "use_textline_orientation": use_textline_orientation,
                    "text_det_thresh": text_det_thresh,
                    "text_det_box_thresh": text_det_box_thresh,
                    "text_det_unclip_ratio": text_det_unclip_ratio,
                    "text_rec_score_thresh": text_rec_score_thresh,
                    "text_recognition_batch_size": text_recognition_batch_size,
                    **kwargs,
                }

        detector = _create_paddleocr_detector(ModernPaddleOCR, **self._kwargs())
        self.assertEqual(detector.kwargs["device"], "cpu")
        self.assertEqual(detector.kwargs["text_rec_score_thresh"], 0.27)
        self.assertEqual(detector.kwargs["text_recognition_batch_size"], 5)
        self.assertNotIn("drop_score", detector.kwargs)
        self.assertNotIn("use_gpu", detector.kwargs)
        self.assertNotIn("show_log", detector.kwargs)

    def test_legacy_api_uses_only_legacy_parameters_it_declares(self):
        class LegacyPaddleOCR:
            def __init__(self, *, lang=None, use_gpu=None, use_angle_cls=None,
                         det_db_thresh=None, det_db_box_thresh=None,
                         det_db_unclip_ratio=None, drop_score=None,
                         rec_batch_num=None, show_log=None):
                self.kwargs = locals()

        detector = _create_paddleocr_detector(LegacyPaddleOCR, **self._kwargs())
        self.assertEqual(detector.kwargs["use_gpu"], False)
        self.assertEqual(detector.kwargs["drop_score"], 0.27)
        self.assertEqual(detector.kwargs["rec_batch_num"], 5)
        self.assertFalse(detector.kwargs["show_log"])
        self.assertNotIn("device", detector.kwargs)
        self.assertNotIn("ocr_version", detector.kwargs)

    def test_initialization_error_is_propagated_without_retry(self):
        calls = []

        class BrokenPaddleOCR:
            def __init__(self, *, lang=None, **kwargs):
                calls.append((lang, kwargs))
                raise PermissionError("cache access denied")

        with self.assertRaisesRegex(PermissionError, "cache access denied"):
            _create_paddleocr_detector(BrokenPaddleOCR, **self._kwargs())
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
