from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

import receiver
import relay


PAYLOAD = b"exact pinned asset"


def member(asset_id=1, filename="config.json"):
    return {"asset_id": asset_id, "filename": filename, "bytes": len(PAYLOAD), "tensor_payload": False,
            "segment_bytes": 32 << 20, "SHA256": sha256(PAYLOAD).hexdigest(),
            "target_relative_path": "base/" + filename,
            "publisher_URL": "https://huggingface.co/fixed/base/resolve/revision/" + filename}


class Response(BytesIO):
    def __init__(self, status, headers):
        super().__init__(PAYLOAD)
        self.status = status
        self.headers = headers

    def geturl(self):
        return "https://cdn.hf.co/fixed"


class ProvisioningTest(unittest.TestCase):
    def test_target_remains_private_until_sealed_and_atomic_publish_is_idempotent(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "target"
            rows = [member(), member(2, "second.json")]
            for row in rows:
                self.assertEqual(receiver.inspect(root, row)["state"], "ABSENT")
                receiver.receive(root, row, 0, row["SHA256"], BytesIO(PAYLOAD))
                with self.assertRaises(ValueError):
                    receiver.receive(root, row, 0, row["SHA256"], BytesIO(PAYLOAD))
            receiver.seal(root, rows[0])
            self.assertFalse((root / "runtime").exists())
            with self.assertRaises(ValueError):
                receiver.publish_runtime(root, rows)
            receiver.seal(root, rows[1])
            result = receiver.publish_runtime(root, rows)
            self.assertEqual(result["files"], 2)
            self.assertEqual((root / "runtime/base/config.json").read_bytes(), PAYLOAD)
            self.assertEqual(receiver.publish_runtime(root, rows), result)
            self.assertTrue(receiver.inspect(root, rows[0])["whole_file_qualified_at_seal"])

    def test_publisher_200_is_accepted_only_for_a_complete_pinned_small_metadata_body(self):
        row = member()
        headers = {"Content-Length": str(len(PAYLOAD))}
        deadline = time.monotonic() + 10
        self.assertEqual(relay.fetch(row, 0, deadline, open_url=lambda *a, **k: Response(200, headers)), PAYLOAD)
        with self.assertRaises(ValueError):
            relay.fetch({**row, "tensor_payload": True}, 0, deadline,
                        open_url=lambda *a, **k: Response(200, headers))
        with self.assertRaises(ValueError):
            relay.fetch(row, 0, deadline, open_url=lambda *a, **k: Response(
                206, {**headers, "Content-Range": f"bytes 1-{len(PAYLOAD)-1}/{len(PAYLOAD)}"}))
        self.assertFalse(relay.official_endpoint("https://huggingface.co.evil.invalid/archive"))

    def test_unknown_terminates_window_without_retransmission(self):
        calls = []

        def transport(args, payload, _timeout):
            calls.append(args[0])
            if args[0] == "inspect":
                return {"file_id": 1, "confirmed_bytes": 0}
            raise TimeoutError("unknown outcomes must be inspected externally")

        with patch.object(relay, "assets", return_value=[member()]), patch.object(relay, "fetch", return_value=PAYLOAD):
            with self.assertRaises(TimeoutError):
                relay.advance(1, transport, max_segments=2, deadline=time.monotonic()+30,
                              record=lambda *_: None)
        self.assertEqual(calls, ["inspect", "append"])


if __name__ == "__main__":
    unittest.main()
