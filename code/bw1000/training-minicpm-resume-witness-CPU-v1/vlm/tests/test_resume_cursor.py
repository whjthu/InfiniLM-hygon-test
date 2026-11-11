"""Reject misleading resume plans before model or optimizer loading."""

import contextlib
import copy
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "vlm" / "runtime"))
sys.path.insert(0, str(ROOT / "tools"))
from resume_cursor import plan_epoch_boundary_resume, read_pinned_json
from inspect_vlm_resume import main


class ResumeCursorTest(unittest.TestCase):
    def setUp(self):
        self.state = dict(global_step=20, max_steps=25, epoch=10.0, train_batch_size=1)
        self.saved = dict(max_steps=25, per_device_train_batch_size=1,
                          gradient_accumulation_steps=4, ignore_data_skip=False,
                          resume_only_model=False, callbacks=["nvidia_baseline"])
        self.options = dict(checkpoint_step=20, max_steps=25, world_size=8,
                            per_device_batch=1, gradient_accumulation_steps=4,
                            dataloader_batches_per_rank=8)

    def plan(self):
        return plan_epoch_boundary_resume(self.state, self.saved, **self.options)

    def test_unchanged_schedule_and_explicit_unverified_state(self):
        before = copy.deepcopy((self.state, self.saved))
        row = self.plan()
        self.assertEqual((self.state, self.saved), before)
        self.assertEqual(row["expected_epochs_trained"], 10)
        self.assertEqual(row["remaining_optimizer_updates"], 5)
        self.assertEqual(row["planned_micro_forwards_per_rank"], 20)
        self.assertEqual(row["effective_batch"], 32)
        self.assertEqual(row["fresh_only_callbacks_requiring_replacement"], ["nvidia_baseline"])
        for key in ("launch_ready", "recovery_qualified", "subsequent_input_equality_verified"):
            self.assertIs(row[key], False)

    def test_skip_rng_or_optimizer_state_cannot_be_called_full_resume(self):
        for key in ("ignore_data_skip", "resume_only_model"):
            for value in (True, 0, None):
                with self.subTest(key=key, value=value):
                    self.saved[key] = value
                    with self.assertRaises(ValueError):
                        self.plan()
            self.saved[key] = False

    def test_wrong_saved_state_and_restart_as_five_steps_rejected(self):
        for key, value in (("global_step", 0), ("global_step", True),
                           ("epoch", 0.0), ("max_steps", 5), ("train_batch_size", 2)):
            original = self.state[key]
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.state[key] = value
                self.plan()
            self.state[key] = original
        self.options["max_steps"] = 5
        with self.assertRaises(ValueError):
            self.plan()

    def test_partial_epoch_and_incomplete_accumulation_rejected(self):
        for batches in (9, 12):
            self.options["dataloader_batches_per_rank"] = batches
            with self.subTest(batches=batches), self.assertRaises(ValueError):
                self.plan()

    def test_changed_batch_or_accumulation_rejected(self):
        for key in ("per_device_train_batch_size", "gradient_accumulation_steps", "max_steps"):
            previous = self.saved[key]
            self.saved[key] += 1
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.plan()
            self.saved[key] = previous
        for value in (0, -1, True, 8.0):
            self.options["world_size"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.plan()

    def test_saved_json_identity_size_and_callback_ambiguity_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            raw = json.dumps(self.state).encode()
            path.write_bytes(raw)
            pin = hashlib.sha256(raw).hexdigest()
            self.assertEqual(read_pinned_json(path, pin, maximum_bytes=len(raw)), self.state)
            for digest, bound in (("0" * 64, len(raw)), (pin, len(raw) - 1), ("", 1000)):
                with self.subTest(digest=digest, bound=bound), self.assertRaises(ValueError):
                    read_pinned_json(path, digest, maximum_bytes=bound)
        for value in (None, "nvidia_baseline", ["x", "x"], [1]):
            self.saved["callbacks"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.plan()

    def test_actual_cli_uses_both_pinned_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            argv = []
            for name, value in (("trainer-state", self.state), ("saved-args", self.saved)):
                path = Path(directory) / (name + ".json")
                raw = json.dumps(value).encode()
                path.write_bytes(raw)
                argv += ["--" + name, str(path), "--" + name + "-sha256",
                         hashlib.sha256(raw).hexdigest()]
            for key, value in self.options.items():
                argv += ["--" + key.replace("_", "-"), str(value)]
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(main(argv), 0)
            self.assertEqual(json.loads(output.getvalue())["remaining_optimizer_updates"], 5)


if __name__ == "__main__":
    unittest.main()
