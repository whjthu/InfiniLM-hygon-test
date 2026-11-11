"""Check that diagnostic observers preserve the frozen VLM recipe and native callback."""
import copy
import hashlib
import json
import sys
import unittest
from pathlib import Path, PurePosixPath
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import hygon_vlm_recipe as recipe


class RecipeDiagnosticTest(unittest.TestCase):
    def setUp(self):
        self.original = {
            "command": ["/source/swift", "sft", "--model", "/source/MiniCPM-V-4.6",
                        "--dataset", "/source/coco64.jsonl", "--output_dir", "/source/run",
                        "--max_steps", "25", "--gradient_accumulation_steps", "4",
                        "--warmup_steps", "5", "--external_plugins", "/source/native.py",
                        "--callbacks", "native", "--attn_impl", "sdpa"],
            "environment": {"PATH": "/source/bin", "NPROC_PER_NODE": "8",
                            "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"},
        }
        self.raw = json.dumps(self.original).encode()
        self.patch = patch.dict(recipe.A100_LAUNCH_SHA256,
                                {"T05.sft": hashlib.sha256(self.raw).hexdigest()})
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def build(self, **kwargs):
        return recipe.build_launch("T05.sft", self.raw,
                                   *[PurePosixPath(x) for x in
                                     ("/runtime/swift", "/models", "/input/coco.jsonl",
                                      "/plugins", "/output/train")], 29873, **kwargs)

    def test_normal_recipe_remains_without_an_observer(self):
        row = self.build()
        self.assertIsNone(row["diagnostic_observer"])
        self.assertEqual(row["required_assets"]["plugins"], ["/plugins/native.py"])
        self.assertEqual(row["command"][-4:], ["--callbacks", "native", "--attn_impl", "sdpa"])

    def test_diagnostic_adds_only_its_plugin_and_callback(self):
        baseline = self.build()
        row = self.build(diagnostic_plugin=PurePosixPath("/diagnostics/observe.py"),
                         diagnostic_callback="capture_updates")
        reconstructed = copy.deepcopy(row["command"])
        reconstructed.remove("/diagnostics/observe.py")
        reconstructed.remove("capture_updates")
        self.assertEqual(reconstructed, baseline["command"])
        self.assertEqual(row["environment_overrides"], baseline["environment_overrides"])
        self.assertFalse(row["diagnostic_observer"]["formal_performance"])
        self.assertEqual(row["required_assets"]["plugins"],
                         ["/plugins/native.py", "/diagnostics/observe.py"])
        self.assertEqual(json.loads(self.raw), self.original)

    def test_incomplete_or_duplicate_diagnostic_is_rejected(self):
        invalid = [dict(diagnostic_callback="capture"),
                   dict(diagnostic_plugin=PurePosixPath("/observe.py")),
                   dict(diagnostic_plugin=PurePosixPath("/observe.py"), diagnostic_callback="native"),
                   dict(diagnostic_plugin=PurePosixPath("/observe.py"), diagnostic_callback="invalid-callback")]
        for case in invalid:
            with self.subTest(case=case), self.assertRaises(ValueError):
                self.build(**case)

    def test_original_launch_identity_is_still_required(self):
        changed = self.raw + b" "
        with self.assertRaisesRegex(ValueError, "SHA256"):
            recipe.build_launch("T05.sft", changed,
                                *[PurePosixPath("/x") for _ in range(5)], 29873,
                                diagnostic_plugin=PurePosixPath("/observe.py"),
                                diagnostic_callback="capture")


if __name__ == "__main__":
    unittest.main()
