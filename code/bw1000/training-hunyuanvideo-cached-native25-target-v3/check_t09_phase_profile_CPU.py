"""Check profile selection, clock reporting and incomplete-event behavior."""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hunyuanvideo_phase_profile import ENV, PHASES, profile_from_environment


class Events:
    def __init__(self):
        self.stream = object()
        self.recorded = []
        self.complete = True

    def current_stream(self):
        return self.stream

    def device_count(self):
        return 1

    def Event(self, *, enable_timing):
        if not enable_timing:
            raise AssertionError("Timing disabled")
        entry = SimpleNamespace()
        entry.ordinal = None

        def record(stream):
            if stream != self.stream:
                raise AssertionError("Wrong occurrence stream")
            entry.ordinal = len(self.recorded)
            self.recorded.append(stream)

        entry.record = record
        entry.query = lambda: self.complete
        entry.elapsed_time = lambda other: 2 * (other.ordinal - entry.ordinal)
        return entry


class PhaseProfileTest(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.events = Events()
        self.runtime = SimpleNamespace(cuda=self.events, version=SimpleNamespace(hip="fixture"))
        self.recipe = dict(warmup=5, measured=20, restored_step=0, recovery=False)

    def open(self, **changes):
        with patch.dict(os.environ, {ENV: "6,7"}):
            return profile_from_environment(self.runtime, self.root, **{**self.recipe, **changes})

    def test_disabled_does_not_touch_the_runtime(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(profile_from_environment(None, self.root, **self.recipe))
            self.assertEqual(list(self.root.iterdir()), [])

    def test_exact_two_steps_have_three_intervals(self):
        profile = self.open()
        self.assertIsNone(profile.begin(5))
        for index in (6, 7):
            step = profile.begin(index)
            for _ in range(3):
                step.mark()
            step.finish(4)
            row = json.loads((profile.directory / f"step-{index:02d}.json").read_bytes())
            self.assertEqual(tuple(row["phase_seconds"]), PHASES)
            self.assertTrue(all(v["device_stream_between_marks"] == .002 for v in row["phase_seconds"].values()))
            self.assertFalse(row["formal_performance_eligible"])
        self.assertEqual(len(self.events.recorded), 8)
        self.assertIsNone(profile.begin(8))
        with self.assertRaisesRegex(ValueError, "repeated"):
            profile.begin(6)

    def test_resume_and_recovery_remain_ineligible(self):
        for case in ({"restored_step": 20}, {"recovery": True}, {"measured": 2}):
            with self.assertRaisesRegex(ValueError, "original fresh"):
                self.open(**case)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_incomplete_or_wrong_stream_has_no_data_receipt(self):
        profile = self.open()
        step = profile.begin(6)
        self.events.stream = object()
        with self.assertRaisesRegex(ValueError, "one submission stream"):
            step.mark()
        self.events.stream = step.stream
        for _ in range(3):
            step.mark()
        self.events.complete = False
        with self.assertRaisesRegex(ValueError, "synchronization"):
            step.finish(4)
        self.assertFalse(step.output.exists())


if __name__ == "__main__":
    unittest.main()
