from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from bind_vggt_tum_dev_rgb import (
    ANCHOR_SHA, bind_pairs, calibration, parse_rgb_index, pin_frames, rectify_rgb, sha, validate_archive_receipt,
)


class TumRgbBinding(unittest.TestCase):
    def test_preserves_motion_failure_and_refuses_timing_gap(self):
        rows = [dict(window=i + 1, state="LOW_TRANSLATION") for i in range(12)]
        rows[0].update(state="POSE_PAIR_CANDIDATE", original_pose_times=(100.0, 100.15))
        rows[1].update(state="POSE_PAIR_CANDIDATE", original_pose_times=(102.0, 102.15))
        rgb = parse_rgb_index(b"100.001 rgb/100.001.png\n100.15 rgb/100.15.png\n"
                              b"102.030 rgb/102.030.png\n102.15 rgb/102.15.png\n")
        bound = bind_pairs(rows, rgb)
        self.assertEqual([row["state"] for row in bound[:3]],
                         ["RGB_INDEX_MATCHED", "RGB_TIMESTAMP_GAP", "LOW_TRANSLATION"])
        self.assertEqual(len(bound), 12)

    def test_refuses_a_duplicate_or_noncanonical_original_frame_name(self):
        for data in (b"1.2 ../1.2.png\n", b"1.2 rgb/1.3.png\n", b"1.2 rgb/1.2.png\n1.2 rgb/1.2.png\n"):
            with self.subTest(data=data), self.assertRaises(ValueError):
                parse_rgb_index(data)

    def test_rectification_uses_distinct_original_calibrations_and_preserves_color_shape(self):
        x, y = np.meshgrid(np.arange(640), np.arange(480))
        pixel = np.stack((x % 251, y % 253, (x+y) % 239), axis=-1).astype(np.uint8)
        stream = BytesIO()
        Image.fromarray(pixel).save(stream, format="PNG")
        bindings = [rectify_rgb(stream.getvalue(), group) for group in ("freiburg1_plant", "freiburg2_dishes")]
        self.assertTrue(all(output.shape == (224, 224, 3) and f > .9 for output, f in bindings))
        self.assertFalse(np.array_equal(bindings[0][0], bindings[1][0]))
        fr1 = calibration("freiburg1_plant")
        self.assertAlmostEqual(fr1["output_K"][2], (318.6 - 80 + .5) * 224 / 480 - .5)

    def test_missing_media_keeps_the_planned_pair_in_the_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [{"state": "RGB_INDEX_MATCHED", "frames": ["rgb/1.0.png", "rgb/1.2.png"]}]
            pin_frames(rows, Path(tmp))
            self.assertEqual(rows[0]["state"], "RGB_MEDIA_MISSING_OR_INVALID")

    def test_archive_binding_requires_a_completed_matching_stream_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = b"100.1 rgb/100.1.png\n"
            expected = {"rgb/100.1.png": {"SHA256": "0" * 64, "bytes": 1}}
            record = dict(passed=True, sequence="freiburg1_plant",
                          archive="https://webshare.cvg.cit.tum.de/g/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_plant.tgz",
                          frozen_anchor_SHA256=ANCHOR_SHA, rgb_index_sha256=sha(index), selected_frames=expected)
            path = root / "official-stream-receipt-private.json"
            path.write_text(json.dumps(record), encoding="utf-8")
            self.assertEqual(validate_archive_receipt("freiburg1_plant", root, index), expected)
            record["passed"] = False
            path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "archive stream"):
                validate_archive_receipt("freiburg1_plant", root, index)


if __name__ == "__main__":
    unittest.main()
