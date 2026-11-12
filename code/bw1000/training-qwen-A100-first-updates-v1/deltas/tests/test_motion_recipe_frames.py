"""A 128-sample, 60-frame cache must not enter the 128-frame training case."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import hygon_training_recipe as recipe


class MotionFrames(unittest.TestCase):
    def fixture(self, root, *, frames=128, features=201):
        manifest = {"schema_version": 2, "format": "hy-motion-training-shards-v1",
                    "frames": frames, "feature_dim": features, "sample_count": 128}
        raw = json.dumps(manifest).encode()
        path = root / "manifest.json"
        path.write_bytes(raw)
        return {"case": "T08.flow-matching-lora", "config": {
            "training_data_manifest": str(path),
            "training_data_manifest_sha256": hashlib.sha256(raw).hexdigest()}}

    def test_pinned_128_frame_header_without_tensor_access(self):
        with tempfile.TemporaryDirectory() as temp:
            plan = self.fixture(Path(temp))
            before = json.dumps(plan, sort_keys=True)
            found = recipe.verify_motion_manifest(plan)
            self.assertEqual((found["frames"], found["feature_dim"]), (128, 201))
            self.assertEqual(json.dumps(plan, sort_keys=True), before)
            self.assertEqual(len(list(Path(temp).iterdir())), 1)

    def test_128_samples_do_not_substitute_for_128_frames(self):
        with tempfile.TemporaryDirectory() as temp:
            plan = self.fixture(Path(temp), frames=60)
            with self.assertRaisesRegex(ValueError, "128 frames.*60"):
                recipe.verify_motion_manifest(plan)

    def test_correct_shape_with_wrong_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            plan = self.fixture(Path(temp))
            plan["config"]["training_data_manifest_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "SHA256"):
                recipe.verify_motion_manifest(plan)

    def test_truncated_motion_target_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            plan = self.fixture(Path(temp), features=200)
            with self.assertRaisesRegex(ValueError, "201-dimensional"):
                recipe.verify_motion_manifest(plan)

    def test_real_run_rejects_before_device_backend_import(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = self.fixture(root, frames=60)
            options = argparse.Namespace(asset_map=root / "map.json", cards="0",
                                         output_root=root, run_name="fixture", lock_root=root,
                                         lock_host="fixture", dry_run=False)
            with patch.dict(sys.modules, {"gencuda.cli": None}):
                with self.assertRaisesRegex(ValueError, "128 frames.*60"):
                    recipe.run_generation(plan, options)

    def test_plan_records_requirement_without_changing_original_config(self):
        original = {"config": {"training_data_manifest": "/inputs/manifest.json",
                               "training_data_manifest_sha256": "1" * 64},
                    "weights": "/models/motion", "source": "/sources/motion"}
        raw = json.dumps(original).encode()
        identity = hashlib.sha256(raw).hexdigest()
        with patch.dict(recipe.GENERATION, {"T08.flow-matching-lora":
                        (identity, "hy-motion-1.0", "train", 1)}):
            plan = recipe.build_generation_plan("T08.flow-matching-lora", raw, {
                "/inputs/manifest.json": "/target/manifest.json",
                "/models/motion": "/target/model", "/sources/motion": "/target/source"})
        self.assertEqual(plan["required_training_frames"], 128)
        self.assertEqual(plan["config"], {**original["config"],
                                         "training_data_manifest": "/target/manifest.json"})


if __name__ == "__main__":
    unittest.main()
