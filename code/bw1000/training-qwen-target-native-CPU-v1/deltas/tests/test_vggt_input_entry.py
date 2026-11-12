"""Exercise the CPU checker's admission failures before any runtime/data import."""

from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("vggt_input_entry", ROOT / "tools/hygon_vggt_input_check.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)
HIDDEN = {key: "" for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")}


class EntryChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.args = []
        self.paths = {}
        for name in ("reference", "config", "input-root", "upstream-root", "upstream-files", "output"):
            self.paths[name] = self.root / name
            self.args += ["--" + name, str(self.paths[name])]
        for name in ("reference-sha256", "config-sha256", "upstream-files-sha256", "backend-sha256"):
            self.args += ["--" + name, "0" * 64]
        now = time.time()
        self.args += ["--epoch", str(now), "--deadline", str(now + 120)]

    def test_expired_future_and_nonfinite_epochs(self):
        for epoch, deadline in ((1, 2), (time.time() + 20, time.time() + 40),
                                (float("nan"), time.time() + 40), (1, float("inf"))):
            args = self.args[:-4] + ["--epoch", str(epoch), "--deadline", str(deadline)]
            with self.assertRaisesRegex(ValueError, "epoch"):
                entry.main(args)
            self.assertFalse(self.paths["output"].exists())

    def test_visible_devices_are_rejected(self):
        for key in HIDDEN:
            with patch.dict(os.environ, {**HIDDEN, key: "0"}):
                with self.assertRaisesRegex(ValueError, "empty device visibility"):
                    entry.main(self.args)

    def test_original_output_cannot_be_reused(self):
        self.paths["output"].write_bytes(b"existing")
        with patch.dict(os.environ, HIDDEN):
            with self.assertRaisesRegex(ValueError, "already exists"):
                entry.main(self.args)
        self.assertEqual(self.paths["output"].read_bytes(), b"existing")

    def test_bad_reference_pin_saves_failed_outcome_without_runtime_import(self):
        self.paths["reference"].write_bytes(b"{}")
        out = io.StringIO()
        with patch.dict(os.environ, HIDDEN), redirect_stdout(out):
            self.assertEqual(entry.main(self.args), 1)
        result = json.loads(self.paths["output"].read_bytes())
        self.assertFalse(result["passed"])
        self.assertFalse(result["CPU_preprocessing_qualified"])
        self.assertEqual(result["error_type"], "ValueError")
        self.assertEqual(result["weight_bytes_read"], 0)
        self.assertNotIn("versions", result)
        events = [json.loads(row) for row in self.paths["output"].with_suffix(".progress.jsonl").read_text().splitlines()]
        self.assertEqual(events, [{"event": "failed_terminal", "error_type": "ValueError"}])
        self.assertEqual(json.loads(out.getvalue())["passed"], False)


if __name__ == "__main__":
    unittest.main()
