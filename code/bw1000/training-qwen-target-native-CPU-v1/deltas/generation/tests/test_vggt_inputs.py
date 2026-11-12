"""Pure CPU protocol fixtures; these do not qualify actual CO3D preprocessing."""

import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gencuda.vggt_inputs import FIELDS, checked_files, check_batches


class Dataset:
    sequence_list = ["b", "a"]
    data_store = {"a": range(24), "b": range(25)}

    def __init__(self, fail_at=None):
        self.calls = 0
        self.fail_at = fail_at

    def get_data(self, *, seq_name, ids, aspect_ratio):
        if self.calls == self.fail_at:
            raise OSError("fixture decode failure")
        index = self.calls
        self.calls += 1
        assert aspect_ratio == 1.0
        return {"seq_name": seq_name, "ids": ids, "point_masks": index % 2 == 0}


def reference():
    return {
        "attempts": [{"candidate": i, "valid": i % 2 == 0, "seq_name": "ab"[i % 2],
                      "ids": [(i * 17 + j * 37) % (24 + i % 2) for j in range(2)]}
                     for i in range(7)],
        "expected_batches": [{"batch": i, "fields": {key: {"SHA256": "f" * 64} for key in FIELDS},
                              "content_SHA256": "b" * 64} for i in range(4)],
        "source_sequence_count": 2, "source_valid_batches": 4, "source_rejected_batches": 3,
        "expected_selected_batch_content_SHA256": "c" * 64,
    }


def run(ref=None, dataset=None, **overrides):
    events = []
    options = dict(make_ids=list, make_batch=lambda x: x, valid_depth=bool,
                   describe=lambda x: {key: {"SHA256": "f" * 64} for key in FIELDS},
                   batch_hash=lambda x: "b" * 64, selected_hash=lambda x: "c" * 64,
                   deadline=lambda: None, progress=events.append)
    options.update(overrides)
    return check_batches(dataset or Dataset(), {"cached_batches": 4, "frames_per_step": 2},
                         ref or reference(), **options), events


class InputChecks(unittest.TestCase):
    def test_full_attempt_and_field_denominators(self):
        result, events = run()
        self.assertTrue(result["passed"])
        self.assertEqual((result["candidate_batches"], result["valid_batches"],
                          result["rejected_batches"], result["tensor_fields_checked"]), (7, 4, 3, 20))
        self.assertEqual(len(events), 8)

    def test_all_five_tensor_fields_are_compared(self):
        for field in FIELDS:
            ref = reference()
            ref["expected_batches"][2]["fields"][field]["SHA256"] = "0" * 64
            result, _ = run(ref)
            self.assertFalse(result["passed"])
            self.assertIn({"batch": 2, "field": field, "kind": "tensor_identity"}, result["mismatches"])
            self.assertEqual(result["candidate_batches"], 7)

    def test_rejected_attempt_identity_cannot_be_omitted(self):
        ref = reference()
        ref["attempts"][1]["ids"][0] += 1
        result, _ = run(ref)
        self.assertFalse(result["passed"])
        self.assertEqual(result["mismatches"], [{"candidate": 1, "kind": "candidate_identity"}])

    def test_incomplete_reference_is_rejected(self):
        for change in (lambda r: r["expected_batches"][0]["fields"].pop("depths"),
                       lambda r: r["attempts"].pop(1)):
            ref = reference()
            change(ref)
            with self.assertRaisesRegex(ValueError, "denominator"):
                run(ref)

    def test_batch_and_combined_hashes_are_compared(self):
        result, _ = run(batch_hash=lambda _: "0" * 64, selected_hash=lambda _: "0" * 64)
        self.assertFalse(result["passed"])
        self.assertEqual(len(result["mismatches"]), 5)

    def test_decode_failure_retains_completed_and_planned_counts(self):
        events = []
        with self.assertRaises(OSError):
            run(dataset=Dataset(fail_at=3), progress=events.append)
        self.assertEqual(events[-1], {"event": "failed", "error_type": "OSError",
                         "completed_attempts": 3, "completed_valid_batches": 2,
                         "planned_attempts": 7, "planned_valid_batches": 4})

    def test_deadline_failure_retains_original_denominator(self):
        events, calls = [], []
        def deadline():
            calls.append(None)
            if len(calls) == 5:
                raise TimeoutError()
        with self.assertRaises(TimeoutError):
            run(deadline=deadline, progress=events.append)
        self.assertEqual(events[-1]["completed_attempts"], 2)
        self.assertEqual(events[-1]["planned_attempts"], 7)

    def test_manifest_checks_bytes_names_and_pins(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "member").write_bytes(b"fixture")
            row = {"name": "member", "bytes": 7, "SHA256": hashlib.sha256(b"fixture").hexdigest()}
            self.assertEqual(checked_files(root, [row], lambda: None), {"files": 1, "bytes": 7})
            for name in ("../member", "/member", "a/../member", "./member", "a\\member", "C:member"):
                with self.assertRaises(ValueError):
                    checked_files(root, [{**row, "name": name}], lambda: None)
            for rows in ([row, row], [{**row, "bytes": 8}], [{**row, "SHA256": "0" * 64}]):
                with self.assertRaises(ValueError):
                    checked_files(root, rows, lambda: None)

    def test_member_changed_after_read_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            member = root / "member"
            member.write_bytes(b"fixture")
            row = {"name": "member", "bytes": 7, "SHA256": hashlib.sha256(b"fixture").hexdigest()}
            calls = []
            def mutate():
                calls.append(None)
                if len(calls) == 2:
                    member.write_bytes(b"changed length")
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                checked_files(root, [row], mutate)


if __name__ == "__main__":
    unittest.main()
