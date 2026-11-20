import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from bind_vggt_tum_dev_rgb import CALIBRATION, calibration, sha
from join_vggt_tum_cross_camera_dev import COMPLETE, FAILED, join
from score_vggt_tum_cross_camera_dev import canonical_scorer, read_cases


def scene(root, sequence, shared=None):
    root.mkdir()
    name = f"{sequence}-window-01-rgb224.npy"
    np.save(root / name, np.zeros((2, 224, 224, 3), dtype=np.uint8), allow_pickle=False)
    artifact = (root / name).read_bytes()
    pose = np.stack([np.eye(4)[:3], np.eye(4)[:3]])
    pose[1, 0, 3] = -.05
    row = dict(sequence=sequence, window=1, state=COMPLETE, rgb_file=name,
               rgb_file_bytes=len(artifact), rgb_file_SHA256=sha(artifact),
               original_frames=[dict(SHA256=shared or sha((sequence + "1").encode())),
                               dict(SHA256=sha((sequence + "2").encode()))],
               intrinsics=calibration(sequence)["output_K"], world_to_camera=pose.tolist())
    rows = [row] + [dict(sequence=sequence, window=n, state=FAILED, reason="RGB_TIMESTAMP_GAP") for n in range(2, 13)]
    raw = json.dumps(dict(binding_SHA256="1" * 64, windows=rows)).encode()
    manifest = root / "inputs-private.json"
    manifest.write_bytes(raw)
    return manifest, sha(raw)


class DevJoinTests(unittest.TestCase):
    def test_combined_inputs_enter_exact_24_window_scorer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            groups = {sequence: scene(root / sequence, sequence) for sequence in CALIBRATION}
            out = root / "combined"
            report = join(groups, out)
            self.assertEqual((report["planned_windows"], report["input_pairs"], report["failed_windows"]), (24, 2, 22))
            cases, rows = read_cases(out / "inputs-private.json", report["cohort_manifest_SHA256"],
                                     canonical_scorer()["relative_pose"])
            self.assertEqual((len(cases), len(rows)), (2, 24))

    def test_wrong_scene_digest_or_shared_frame_fails_before_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            groups = {sequence: scene(root / sequence, sequence, shared="a" * 64) for sequence in CALIBRATION}
            out = root / "combined"
            with self.assertRaisesRegex(ValueError, "share an original RGB"):
                join(groups, out)
            self.assertFalse(out.exists())
            group = next(iter(groups))
            groups[group] = groups[group][0], "e" * 64
            with self.assertRaisesRegex(ValueError, "manifest SHA"):
                join(groups, out)
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
