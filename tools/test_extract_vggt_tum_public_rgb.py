import io
from pathlib import Path
import tarfile
import tempfile
import unittest

from PIL import Image

from extract_vggt_tum_public_rgb import scan_archive


def fixture(index_first, omit_frame=False):
    prefix = "rgbd_dataset_freiburg1_plant/"
    rgb = b"100.001 rgb/100.001.png\n100.300 rgb/100.300.png\n"
    buffer = io.BytesIO()
    image = io.BytesIO()
    Image.new("RGB", (640, 480), (137, 71, 0)).save(image, "PNG")
    png = image.getvalue()
    frames = [("rgb/100.001.png", png), ("depth/100.001.png", b"ignored")]
    if not omit_frame:
        frames.append(("rgb/100.300.png", png))
    index = [("rgb.txt", rgb)]
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, raw in (index + frames if index_first else frames + index):
            member = tarfile.TarInfo(prefix + name)
            member.size = len(raw)
            archive.addfile(member, io.BytesIO(raw))
    buffer.seek(0)
    return buffer


class TumOfficialArchiveExtraction(unittest.TestCase):
    def setUp(self):
        self.anchors = [dict(window=i+1, state="LOW_TRANSLATION") for i in range(12)]
        self.anchors[0].update(state="POSE_PAIR_CANDIDATE", original_pose_times=(100.0,100.3))

    def test_retains_exact_rgb_even_when_archive_index_order_differs(self):
        for index_first in (True, False):
            with self.subTest(index_first=index_first), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "rgb").mkdir()
                result = scan_archive(fixture(index_first), "freiburg1_plant", self.anchors, root)
                self.assertEqual(result["indexed_required_frames"], 2)
                self.assertEqual(result["official_rgb_index_rows"], 2)
                self.assertFalse((root / "depth").exists())

    def test_rejects_missing_indexed_rgb_and_keeps_original_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "rgb").mkdir()
            with self.assertRaisesRegex(ValueError, "indexed RGB frames unavailable"):
                scan_archive(fixture(False, True), "freiburg1_plant", self.anchors, root)
            self.assertFalse((root / "rgb.txt").exists())


if __name__ == "__main__":
    unittest.main()
