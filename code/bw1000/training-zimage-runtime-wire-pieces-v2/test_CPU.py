from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from wire import PIECE, Pieces, load_legacy

LEGACY = Path(__file__).resolve().parent.parent / "training-zimage-publisher-runtime-transfer-v1"


class TestWire(unittest.TestCase):
    def test_reconciles_and_commits_original_range_only_after_all_pieces(self):
        receiver, _, stage = load_legacy(LEGACY)
        first, tail = b"x" * PIECE, b"second piece"
        joined = first + tail
        row = {"asset_id": 1, "filename": "fixture.safetensors", "bytes": len(joined),
               "segment_bytes": 32 << 20, "SHA256": sha256(joined).hexdigest()}
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve() / "target"
            pieces = Pieces(root, receiver, stage)
            self.assertEqual(pieces.state(row)["wire_bytes"], 0)
            pieces.append(row, 0, 0, sha256(first).hexdigest(), BytesIO(first))
            self.assertEqual(pieces.state(row)["wire_bytes"], PIECE)
            with self.assertRaisesRegex(ValueError, "already committed"):
                pieces.append(row, 0, 0, sha256(first).hexdigest(), BytesIO(first))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                pieces.commit(row, 0)
            with self.assertRaisesRegex(ValueError, "digest"):
                pieces.append(row, 0, PIECE, sha256(first).hexdigest(), BytesIO(tail))
            pieces.append(row, 0, PIECE, sha256(tail).hexdigest(), BytesIO(tail))
            result = pieces.commit(row, 0)
            self.assertEqual(result["confirmed_bytes"], len(joined))
            self.assertEqual(result["segment_SHA256"], row["SHA256"])
            self.assertEqual(pieces.state(row)["wire_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
