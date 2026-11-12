"""Reject wrong platform or visible-card count before fixed replay weight reads."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "generation/src"))
from gencuda.runtime import RuntimeInfo
SPEC = importlib.util.spec_from_file_location("fixed_sdxl_replay", ROOT / "tools/hygon_sdxl_first_step.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class ReachedWeights(Exception):
    pass


class FixedReplayPlatformTests(unittest.TestCase):
    def invoke(self, actual, count, requested=None):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            runtime = RuntimeInfo("qualified-fixture", True, count, "fixture", actual, "version")
            with patch("gencuda.runtime.inspect_runtime", return_value=runtime), \
                    patch("gencuda.runtime.get_torch", side_effect=AssertionError("device must not be loaded")), \
                    patch.object(module, "sha", side_effect=ReachedWeights()) as weight_read:
                kwargs = {} if requested is None else {"accelerator": requested}
                expected = ReachedWeights if count == 1 and actual == (requested or "hip") else RuntimeError
                with self.assertRaises(expected):
                    module.replay(root / "reference.pt", root / "weights", root / "output", {}, **kwargs)
                self.assertEqual(weight_read.called, expected is ReachedWeights)

    def test_default_hip_keeps_existing_route(self):
        self.invoke("hip", 1)

    def test_explicit_cuda_reaches_weight_identity_gate(self):
        self.invoke("cuda", 1, "cuda")

    def test_cuda_request_rejects_hip_before_weight_or_device_reads(self):
        self.invoke("hip", 1, "cuda")

    def test_cuda_request_rejects_two_visible_cards(self):
        self.invoke("cuda", 2, "cuda")

    def test_unknown_platform_rejected_before_namespace_creation(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / "output"
            with self.assertRaises(ValueError):
                module.replay(Path("reference.pt"), Path("weights"), output, {}, accelerator="unknown")
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
