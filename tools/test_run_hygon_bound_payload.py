import ast
from base64 import b64decode
import json
from pathlib import Path
import shlex
import sys
from tempfile import TemporaryDirectory
import unittest

from run_hygon_bound_payload import digest, send


REQUEST = "0" * 32


def stub(mode, command):
    program = shlex.split(command)[-1]
    binding = ast.literal_eval(program.splitlines()[0].removeprefix("BINDING="))
    if mode == "no-authority":
        return 255
    print(json.dumps({"event": "BOUND", **binding}), flush=True)
    actual = b64decode(sys.stdin.buffer.readline().strip(), validate=True)
    assert digest(actual) == binding["payload_sha256"]
    if mode == "after-send-disconnect":
        return 255
    assert b64decode(json.loads(actual)["data"]) == b"x" * 65536
    print(json.dumps({"event": "RECEIVED", **binding}), flush=True)
    print(json.dumps({"request": binding["request"], "ok": True}), flush=True)
    return 0


class BindingTest(unittest.TestCase):
    def call(self, mode, token):
        return send([sys.executable, __file__, "--stub", mode], request=REQUEST,
                    program="pass", data=b"x" * 65536, send_token=token,
                    authority_seconds=4, outcome_seconds=4)

    def test_disconnect_before_authority_preserves_retry(self):
        with TemporaryDirectory() as d:
            path = Path(d) / "token.json"
            value = self.call("no-authority", path)
            self.assertEqual(value["remote_outcome"], "PAYLOAD_NOT_SENT")
            self.assertFalse(value["payload_send_attempted"])
            self.assertFalse(path.exists())

    def test_disconnect_after_write_commits_original_identity(self):
        with TemporaryDirectory() as d:
            path = Path(d) / "token.json"
            value = self.call("after-send-disconnect", path)
            self.assertEqual(value["remote_outcome"], "UNKNOWN")
            self.assertTrue(value["payload_send_attempted"] and value["payload_flushed"])
            self.assertFalse(value["remote_received"])
            self.assertTrue(path.exists())
            with self.assertRaises(ValueError):
                self.call("received", path)

    def test_response_requires_matching_authenticated_receipt(self):
        with TemporaryDirectory() as d:
            value = self.call("received", Path(d) / "token.json")
            self.assertEqual(value["remote_outcome"], "RESPONSE_OBSERVED")
            self.assertEqual(value["response"], {"request": REQUEST, "ok": True})


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--stub":
        sys.exit(stub(sys.argv[2], sys.argv[3]))
    unittest.main()
