import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from reconcile_llava_shard01_missing_ranges import FILENAME, REQUEST, WHOLE_BYTES, WHOLE_SHA, target_snapshot


class OriginalShardSnapshotTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.binding = dict(request=REQUEST, file=FILENAME, bytes=WHOLE_BYTES, SHA256=WHOLE_SHA)
        raw = json.dumps(self.binding).encode()
        self.intent = self.root / "intent.json"
        self.intent.write_bytes(raw)
        patcher = patch("reconcile_llava_shard01_missing_ranges.RECEIVER_INTENT_SHA", hashlib.sha256(raw).hexdigest())
        patcher.start()
        self.addCleanup(patcher.stop)

    def terminal(self, **update):
        state = dict(self.binding, role="receiver", passed=False, durable_bytes=12, written_bytes=13)
        state.update(update)
        (self.root / "terminal.json").write_text(json.dumps(state), encoding="utf8")

    def test_missing_terminal_preserves_unknown(self):
        result = target_snapshot(self.root)
        self.assertEqual(result["state"], "TARGET_TERMINAL_NOT_OBSERVED")
        self.assertIsNone(result["target_durable_bytes"])

    def test_read_exact_durable_suffix_receipt_without_reading_partial(self):
        self.terminal()
        (self.root / "shard.incomplete").write_bytes(b"x" * 13)
        data = dict(self.binding, durable_bytes=12, prefix_SHA256="3" * 64)
        (self.root / "prefix-0000000012.json").write_text(json.dumps(data), encoding="utf8")
        result = target_snapshot(self.root)
        self.assertEqual(result["state"], "TARGET_RECEIVER_FAILED")
        self.assertEqual(result["target_durable_bytes"], 12)
        self.assertEqual(result["actual_model_weight_file_reads"], 0)
        self.assertFalse(result["independent_target_whole_SHA_verified"])

    def test_rejects_unbound_terminal_or_conflicting_published_shard(self):
        self.terminal(request="other")
        with self.assertRaisesRegex(ValueError, "different request"):
            target_snapshot(self.root)
        self.terminal()
        (self.root / FILENAME).touch()
        with self.assertRaisesRegex(ValueError, "unsent commit"):
            target_snapshot(self.root)


if __name__ == "__main__":
    unittest.main()
