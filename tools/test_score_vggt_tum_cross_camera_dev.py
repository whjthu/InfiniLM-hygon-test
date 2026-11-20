import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from bind_vggt_tum_dev_rgb import calibration, sha
from score_vggt_tum_cross_camera_dev import ARMS, WINDOWS, canonical_scorer, read_cases, read_heads, summarize


class CrossCameraScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scorer = canonical_scorer()

    def test_real_pose_math_and_planned_denominator(self):
        pose = np.stack([np.eye(4)[:3], np.eye(4)[:3]])
        pose[1, 0, 3] = -.05
        reverse = pose.copy()
        reverse[1, 0, 3] *= -1
        good = self.scorer["angular_errors"](pose, pose)
        wrong = self.scorer["angular_errors"](reverse, pose)
        self.assertAlmostEqual(good["pair_error_deg"], 0)
        self.assertAlmostEqual(wrong["pair_error_deg"], 180)
        fail = {"pair_error_deg": None}
        rows = [dict(sequence=s, window=n, base=fail, alternate25=fail, lowlr25=fail)
                for s, n in sorted(WINDOWS)]
        rows[0].update(base=good, alternate25=wrong)
        report = summarize(rows, self.scorer["summarize_development"])
        self.assertEqual(report["all_windows"]["base"]["failed_pairs"], 23)
        self.assertAlmostEqual(report["all_windows"]["base"]["development_AUC_at_30"], 1 / 24)
        self.assertEqual(report["all_windows"]["alternate25"]["failed_pairs"], 23)
        self.assertEqual(report["all_windows"]["alternate25"]["development_AUC_at_30"], 0)
        self.assertAlmostEqual(report["by_sequence"][rows[0]["sequence"]]["base"]["development_AUC_at_30"], 1 / 12)

    def test_manifest_reads_real_numpy_layout_and_keeps_missing_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sequence, window = min(WINDOWS)
            rows = [dict(sequence=s, window=n, state="RGB224_PRECONDITION_FAILED") for s, n in sorted(WINDOWS)]
            image = np.zeros((2, 224, 224, 3), dtype=np.uint8)
            name = f"{sequence}-window-{window:02d}-rgb224.npy"
            np.save(root / name, image, allow_pickle=False)
            truth = np.stack([np.eye(4)[:3], np.eye(4)[:3]])
            truth[1, 0, 3] = -.05
            raw_image = (root / name).read_bytes()
            rows[0].update(state="RGB224_AND_RELATIVE_POSE_BOUND", rgb_file=name, rgb_file_bytes=len(raw_image),
                           rgb_file_SHA256=sha(raw_image), world_to_camera=truth.tolist(),
                           intrinsics=calibration(sequence)["output_K"])
            manifest = root / "inputs-private.json"
            frozen = json.dumps(dict(windows=rows)).encode()
            manifest.write_bytes(frozen)
            cases, planned = read_cases(manifest, sha(frozen), self.scorer["relative_pose"])
            self.assertEqual((len(cases), len(planned)), (1, 24))
            self.assertEqual(planned[1]["base"]["state"], "MISSING_INPUT")
            np.testing.assert_array_equal(cases[0][1], image)
            (root / name).write_bytes(raw_image[:-1])
            with self.assertRaisesRegex(ValueError, "RGB224 bytes"):
                read_cases(manifest, sha(frozen), self.scorer["relative_pose"])
            rows.pop()
            frozen = json.dumps(dict(windows=rows)).encode()
            manifest.write_bytes(frozen)
            with self.assertRaisesRegex(ValueError, "all 24"):
                read_cases(manifest, sha(frozen), self.scorer["relative_pose"])

    def test_wrong_original_training_request_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "heads.json"
            pins = {arm: dict(gpu_step=job, request=request, clean_post=post, training_receipt_SHA256=receipt,
                              global_step=25, file_SHA256="1" * 64, module_SHA256="2" * 64,
                              original_outcome_SHA256="3" * 64, original_combined_native_SHA256="4" * 64,
                              original_continuous_result_SHA256="5" * 64,
                              actual_checkpoint_bytes_read=0, head_bytes_stat=128)
                    for arm, (job, request, post, receipt) in ARMS.items()}
            raw = json.dumps(pins).encode()
            path.write_bytes(raw)
            self.assertEqual(read_heads(path, sha(raw)), pins)
            pins["lowlr25"]["training_receipt_SHA256"] = "3" * 64
            raw = json.dumps(pins).encode()
            path.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "original training"):
                read_heads(path, sha(raw))


if __name__ == "__main__":
    unittest.main()
