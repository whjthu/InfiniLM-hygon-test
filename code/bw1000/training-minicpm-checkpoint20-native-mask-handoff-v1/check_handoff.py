"""Test the rendered handoff against the original transition fixtures and a stale-counter refusal."""

import sys
from types import ModuleType
import unittest

import render_handoff as render


handoff = ModuleType("handoff")
exec(compile(render.build(), "<rendered-handoff>", "exec"), vars(handoff))
sys.modules["handoff"] = handoff
prior = render.PRIOR / "check_local.py"
fixture = ModuleType("prior_handoff_fixtures")
fixture.__file__ = str(prior)
exec(compile(prior.read_bytes(), str(prior), "exec"), vars(fixture))
old_sample = fixture.sample


def sample(*args, **kwargs):
    return dict(old_sample(*args, **kwargs), live_standby_recent_cards=8)


fixture.sample = sample


class RecentCountersTest(fixture.TestHandoff):
    def test_missing_or_stale_counter_refuses_signal(self):
        for count in (None, 0, 7):
            trace = []
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "eight recent compute counters"):
                self.run_transition([dict(sample(), live_standby_recent_cards=count)], [], trace)
            self.assertEqual(trace, [])

    def test_standby_counter_may_stop_during_graceful_release(self):
        trace = []
        self.run_transition([sample(), dict(sample(), live_standby_recent_cards=0), sample("COMPLETED", False)],
                            [None, None, fixture.TERMINAL], trace)
        self.assertEqual(trace.count("SIGTERM"), 1)
        self.assertEqual(trace.count("launch"), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
