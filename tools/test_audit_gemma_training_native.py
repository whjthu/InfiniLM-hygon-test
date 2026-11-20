import copy
import json
import unittest
from unittest.mock import Mock

from audit_gemma_training_native import audit_rows, check_sample, digest


class BatchValue:
    def __init__(self, value):
        self.value = value

    def tolist(self):
        return self.value


def observation():
    encoded = {"input_ids": [2, 40, 102, 108], "labels": [-100, 40, 102, 108], "pixels": [0.5]}
    expected_batch = {name: BatchValue([encoded[name]]) for name in ("input_ids", "labels")}
    expected_batch["attention_mask"] = BatchValue([[1, 1, 1, 1]])
    return encoded, expected_batch


def fingerprint(data):
    raw = json.dumps(data, default=lambda v: v.tolist(), sort_keys=True).encode()
    return {"SHA256": digest(raw)}


class GemmaNativeInputBoundaries(unittest.TestCase):
    def test_supervised_count_excludes_shifted_position(self):
        self.assertEqual(check_sample([2, 40, 102, 108], [-100, 40, 102, 108], [102, 108], {1, 102}), 3)

    def test_mask_token_and_suffix_rejections(self):
        ids = [2, 40, 102, 108]
        for labels, suffix, eos in (
            ([2, 40, 102, 108], [102, 108], {102}),
            ([-100, 41, 102, 108], [102, 108], {102}),
            ([-100, 40, -100, 108], [102, 108], {102}),
            ([-100, 40, 102, 108], [102, 108], {1}),
            ([-100, 40, 102, 108], [108, 102], {102}),
        ):
            with self.subTest(labels=labels, suffix=suffix), self.assertRaises(ValueError):
                check_sample(ids, labels, suffix, eos)

    def test_sequence_over_limit_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "sequence length"):
            check_sample([40] * 512 + [102], [-100] * 512 + [102], [102], {102})

    def test_native_path_keeps_all_fields_and_does_not_reuse_mutated_rows(self):
        encoded, batch = observation()
        full, capped, check, progress = Mock(), Mock(), Mock(), Mock()

        def mutate(row):
            row["messages"].clear()
            return encoded

        full.encode.side_effect = mutate
        capped.encode.side_effect = lambda row: encoded if row["messages"] else None
        capped.data_collator.return_value = batch
        audit_rows([b'{"messages":[1]}'], full, capped, fingerprint, [102, 108], {102}, check, progress)
        self.assertEqual(check.call_count, 2)
        self.assertEqual(progress.call_args.args[0]["shifted_supervised_tokens"], 3)

    def test_rejects_modified_visual_fields_and_collator_mask(self):
        encoded, batch = observation()
        full, capped, check, progress = Mock(), Mock(), Mock(), Mock()
        full.encode.return_value = encoded
        capped.encode.return_value = {**encoded, "pixels": [0.25]}
        capped.data_collator.return_value = batch
        args = ([b"{}"], full, capped, fingerprint, [102, 108], {102}, check, progress)
        with self.assertRaisesRegex(ValueError, "production limit"):
            audit_rows(*args)
        capped.encode.return_value = copy.deepcopy(encoded)
        batch["labels"] = BatchValue([[-100, -100, 102, 108]])
        with self.assertRaisesRegex(ValueError, "collator changed"):
            audit_rows(*args)
        batch["labels"] = BatchValue([encoded["labels"]])
        batch["attention_mask"] = BatchValue([[1, 1, 0, 1]])
        with self.assertRaisesRegex(ValueError, "non-padding"):
            audit_rows(*args)
        progress.assert_not_called()


if __name__ == "__main__":
    unittest.main()
