import io
from pathlib import Path
import re
import tarfile
import tempfile
import unittest

from PIL import Image

from extract_vggt_tum_public_rgb import RangedArchive, scan_archive


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

    def test_ranged_gzip_reconnect_keeps_correct_offsets(self):
        raw = fixture(False).read()
        requests = []
        url, etag = "https://example.invalid/archive.tgz", '"fixture-archive"'

        class Response(io.BytesIO):
            status = 206

            def geturl(self):
                return url

            def read(self, amount):
                if len(requests) == 2:
                    raise TimeoutError("injected socket idle limit")
                return super().read(amount)

        def open_range(request, *, timeout):
            headers = dict(request.header_items())
            self.assertEqual(headers["If-match"], etag)
            start, end = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", headers["Range"]).groups())
            requests.append((start, end))
            response = Response(raw[start:end+1])
            response.headers = {"ETag": etag, "Content-Range": f"bytes {start}-{end}/{len(raw)}",
                                "Content-Length": str(end-start+1)}
            return response

        reader = RangedArchive(url, len(raw), etag, 50, opener=open_range, clock=lambda: 0,
                               segment_bytes=36, retry_limit=1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "rgb").mkdir()
            result = scan_archive(reader, "freiburg1_plant", self.anchors, root)
        reader.close()
        self.assertEqual(result["indexed_required_frames"], 2)
        self.assertEqual(requests[:3], [(0, 35), (36, 71), (36, 71)])
        self.assertEqual(reader.retries, 1)

    def test_ranged_refuses_changed_object_and_bounds_timeouts(self):
        url = "https://example.invalid/archive.tgz"

        class BadIdentity(io.BytesIO):
            status = 206
            headers = {"ETag": '"replacement"', "Content-Range": "bytes 0-3/4", "Content-Length": "4"}

            def geturl(self):
                return url

        reader = RangedArchive(url, 4, '"fixed"', 30, opener=lambda *a, **kw: BadIdentity(b"abcd"),
                               clock=lambda: 0)
        with self.assertRaisesRegex(ValueError, "immutable object identity"):
            reader.read(4)
        self.assertEqual(reader.retries, 0)

        def open_timeout(*args, **kwargs):
            raise TimeoutError("test")

        reader = RangedArchive(url, 4, '"fixed"', 30, opener=open_timeout, clock=lambda: 0, retry_limit=1)
        with self.assertRaisesRegex(TimeoutError, "bounded range reconnects"):
            reader.read(4)
        self.assertEqual(reader.requests, 2)


if __name__ == "__main__":
    unittest.main()
