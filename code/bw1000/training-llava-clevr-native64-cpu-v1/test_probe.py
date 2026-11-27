import argparse
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import probe


class GateTests(unittest.TestCase):
    def test_supervision_has_shifted_answer_eos_and_masked_image(self):
        values, result = probe.supervision([10, 99, 12, 2], [-100, -100, 12, 2],
                                           [1, 1, 1, 1], {2}, 99)
        self.assertEqual(values, [12, 2])
        self.assertEqual(result["shifted_supervised_tokens"], 2)

    def test_supervision_rejects_empty_shifted_eos_only_wrong_ids_and_visible_image(self):
        for labels in ([-100] * 4, [10, -100, 12, 2], [-100, -100, -100, 2],
                       [-100, -100, 22, 2], [-100, 99, 12, 2], [-100, -100, 12, -100]):
            with self.subTest(labels=labels), self.assertRaises(probe.Gate):
                probe.supervision([10, 99, 12, 2], labels, [1] * 4, {2}, 99)

    def test_reordered_substituted_and_duplicate_image_bindings(self):
        rows = [({"images": ["images/CLEVR_train_%06d.png" % number]}, str(number))
                for number in range(probe.ROWS)]
        pins = [dict(ordinal=index + 1, row_sha256=row_sha, path=row["images"][0],
                     bytes=32, sha256=hashlib.sha256(row_sha.encode()).hexdigest())
                for index, (row, row_sha) in enumerate(rows)]
        header = dict(train_sha256=probe.TRAIN, stage_receipt_sha256=probe.STAGE)
        probe.check_binding(rows, dict(header, rows=pins))
        for altered in ([pins[1], pins[0], *pins[2:]],
                        [dict(pins[0], path="images/CLEVR_val_000000.png"), *pins[1:]],
                        [dict(pins[0], sha256=pins[1]["sha256"]), *pins[1:]]):
            with self.assertRaises(probe.Gate):
                probe.check_binding(rows, dict(header, rows=altered))

    def test_pinned_bytes_and_exclusive_receipt(self):
        with tempfile.TemporaryDirectory() as name:
            pathname = Path(name).resolve() / "receipt.json"
            probe.persist(pathname, {"status": "first"})
            with self.assertRaises(FileExistsError):
                probe.persist(pathname, {"status": "replay"})
            self.assertEqual(json.loads(probe.read(pathname, 128)), {"status": "first"})
            with self.assertRaisesRegex(probe.Gate, "INPUT_SHA256"):
                probe.read(pathname, 128, "0" * 64)

    def test_setup_failure_leaves_all_64_missing(self):
        result = {"rows": []}
        with patch.object(probe, "admission", side_effect=probe.Gate("EXACT_ADMISSION")):
            probe.run(argparse.Namespace(), result)
        self.assertEqual(len(result["rows"]), 0)
        self.assertEqual(result["failure"]["phase"], "admission")
        self.assertEqual(result["failure"]["code"], "EXACT_ADMISSION")


if __name__ == "__main__":
    unittest.main()
