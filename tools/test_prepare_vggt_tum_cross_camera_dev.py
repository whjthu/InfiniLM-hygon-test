import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from bind_vggt_tum_dev_rgb import sha
from prepare_vggt_tum_cross_camera_dev import prepare_pair, synchronized_pose


class CrossCameraInputTests(unittest.TestCase):
    poses = [(100.0, 0, 0, 0, 0, 0, 0, 1), (100.1, 0.05, 0, 0, 0, 0, 0, 1)]
    times = [100.0, 100.1]

    def test_exposure_time_pose_and_direction(self):
        result = synchronized_pose(self.poses, self.times, [100.003, 100.099])
        self.assertEqual(result["original_mocap_times"], self.times)
        np.testing.assert_allclose(result["rgb_pose_time_error_seconds"], [.003, .001])
        np.testing.assert_allclose(result["world_to_camera"][0], np.eye(4)[:3])
        np.testing.assert_allclose(np.asarray(result["world_to_camera"][1])[:, 3], [-.05, 0, 0])

    def test_reject_gap_reuse_and_low_translation(self):
        for exposure in ([100.03, 100.1], [100.0, 100.001]):
            with self.subTest(exposure=exposure), self.assertRaisesRegex(ValueError, "distinct nearby"):
                synchronized_pose(self.poses, self.times, exposure)
        static = [self.poses[0], (100.1, .01, 0, 0, 0, 0, 0, 1)]
        with self.assertRaisesRegex(ValueError, "translation"):
            synchronized_pose(static, self.times, self.times)

    def test_original_png_pins_before_rgb224_use(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work).resolve()
            (root / "rgb").mkdir()
            names = ["rgb/100.003.png", "rgb/100.099.png"]
            pins = []
            for index, name in enumerate(names):
                buffer = io.BytesIO()
                Image.new("RGB", (640, 480), (100 + index, 80, 60)).save(buffer, format="PNG")
                raw = buffer.getvalue()
                (root / name).write_bytes(raw)
                pins.append(dict(path=name, bytes=len(raw), SHA256=sha(raw)))
            row = dict(media=pins, frames=names, rgb_times=[100.003, 100.099])
            result, geometry = prepare_pair("freiburg1_plant", row, self.poses, self.times, root)
            self.assertEqual((result.shape, result.dtype), ((2, 224, 224, 3), np.dtype("uint8")))
            np.testing.assert_array_equal(result[:, 112, 112, 0], [100, 101])
            self.assertTrue(all(value >= .95 for value in geometry["valid_pixel_fraction"]))
            with self.assertRaisesRegex(ValueError, "timestamps"):
                prepare_pair("freiburg1_plant", dict(row, rgb_times=self.times), self.poses, self.times, root)
            (root / names[1]).write_bytes(b"altered")
            with self.assertRaisesRegex(ValueError, "content changed"):
                prepare_pair("freiburg1_plant", row, self.poses, self.times, root)


if __name__ == "__main__":
    unittest.main()
