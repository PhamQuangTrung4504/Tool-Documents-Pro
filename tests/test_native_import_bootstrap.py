"""Kiểm tra thứ tự import native trong Python con để cách ly lỗi DLL."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import unittest


class NativeImportBootstrapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("torch") is None or importlib.util.find_spec("paddle") is None:
            raise unittest.SkipTest("Cần cài Torch và Paddle để chạy kiểm thử import native.")

    def _run_child(self, code: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )

    def test_bootstrap_imports_torch_before_paddle_in_fresh_process(self):
        child = self._run_child(
            "import json, sys\n"
            "from app.native_runtime import bootstrap_torch_before_paddle\n"
            "bootstrap_torch_before_paddle()\n"
            "assert 'torch' in sys.modules and 'paddle' not in sys.modules\n"
            "import paddle\n"
            "print(json.dumps({'torch': sys.modules['torch'].__version__, "
            "'paddle': paddle.__version__, 'torch_first': True}))\n"
        )
        self.assertEqual(child.returncode, 0, child.stdout + child.stderr)
        report = json.loads(child.stdout.strip().splitlines()[-1])
        self.assertTrue(report["torch_first"])

    def test_bootstrap_refuses_already_loaded_paddle_before_torch(self):
        child = self._run_child(
            "import paddle\n"
            "from app.native_runtime import bootstrap_torch_before_paddle\n"
            "try:\n"
            "    bootstrap_torch_before_paddle()\n"
            "except RuntimeError as exc:\n"
            "    print(str(exc))\n"
            "else:\n"
            "    raise AssertionError('bootstrap phải từ chối thứ tự không an toàn')\n"
        )
        self.assertEqual(child.returncode, 0, child.stdout + child.stderr)
        self.assertIn("Paddle đã được nạp trước khi bootstrap", child.stdout)


if __name__ == "__main__":
    unittest.main()
