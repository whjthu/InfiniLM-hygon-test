"""Fixtures for the actual single-signal and launch transitions."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import handoff


PCI = [f"0000:{x:02x}:00.0" for x in range(8)]
EXPECTED = dict(request=handoff.STANDBY_REQUEST, parent=handoff.PARENT, node=handoff.NODE,
                label=handoff.STANDBY_NAME, pci=PCI)
WORKER = dict(node=handoff.NODE, step=handoff.STANDBY_STEP, synthetic=True, worker_pid=50, pci=PCI)
TERMINAL = dict(node=handoff.NODE, step=handoff.STANDBY_STEP, state="WORKER_RETURNED")
PARENT = dict(state="RUNNING", nodes=40, allocated_DCU=320,
              ledger_exact=True, step_locations_exact=True, owned_by_observer=True)


def sample(state="RUNNING", active=True):
    row = [handoff.STANDBY_STEP, handoff.STANDBY_NAME, state, "0:0", "20", handoff.NODE, handoff.STANDBY_TRES]
    steps = {handoff.STANDBY_STEP: dict(state=state, member_nodes=[handoff.NODE],
                                      allocated_TRES=handoff.STANDBY_TRES)} if active else {}
    return dict(parent=PARENT.copy(), original_diagnostic_unstarted=True, accounting=row, active_node_steps=steps)


class TestHandoff(unittest.TestCase):
    def run_transition(self, observations, terminals, trace, signal_rc=0, worker=None):
        timeline = iter(observations)
        receipts = iter(terminals)
        clock = [0]

        def call(action, result):
            trace.append(action)
            return result

        def sleep(seconds):
            clock[0] += seconds

        return handoff.bounded_transition(lambda: next(timeline), lambda: next(receipts),
            lambda proof: handoff.verify_candidate(proof, EXPECTED, WORKER if worker is None else worker),
            lambda: call("SIGTERM", signal_rc), lambda: call("launch", 7),
            lambda name, _: trace.append(name), monotonic=lambda: clock[0], sleep=sleep)

    def test_one_exact_signal_before_independent_release(self):
        trace = []
        self.assertEqual(self.run_transition([sample(), sample(), sample("COMPLETED", False)],
                                             [None, None, TERMINAL], trace), 7)
        self.assertEqual(trace, ["before-signal.json", "SIGTERM", "signal-return.json",
                                 "standby-released.json", "launch"])

    def test_natural_completion_does_not_signal(self):
        trace = []
        self.run_transition([sample("COMPLETING"), sample("COMPLETED", False)], [TERMINAL], trace)
        self.assertEqual(trace, ["standby-released.json", "launch"])

    def test_lost_signal_race_requires_terminal(self):
        trace = []
        self.run_transition([sample(), sample("COMPLETED", False)], [None, TERMINAL], trace, signal_rc=1)
        self.assertEqual(trace.count("SIGTERM"), 1)
        self.assertEqual(trace.count("launch"), 1)

    def test_foreign_worker_owner_or_step_cannot_signal(self):
        trace = []
        other = sample()
        other["active_node_steps"]["1103606.999"] = {}
        foreign = sample()
        foreign["parent"]["owned_by_observer"] = False
        for row, worker in ((other, WORKER), (foreign, WORKER), (sample(), dict(WORKER, synthetic=False)),
                            (sample(), dict(WORKER, pci=PCI[:-1]))):
            with self.subTest(row=row, worker=worker), self.assertRaises(ValueError):
                self.run_transition([row], [], trace, worker=worker)
        self.assertEqual(trace, [])

    def test_prior_diagnostic_and_ambiguous_terminal_cannot_launch(self):
        trace = []
        started = sample()
        started["original_diagnostic_unstarted"] = False
        with self.assertRaises(ValueError):
            self.run_transition([started], [], trace)
        self.assertEqual(trace, [])
        terminal = sample("COMPLETED", False)
        self.assertFalse(handoff.released(terminal, dict(TERMINAL, step="1103606.999")))
        self.assertFalse(handoff.released(terminal, None))
        self.assertFalse(handoff.released(sample("COMPLETED"), TERMINAL))

    def test_no_replay_when_standby_still_active(self):
        trace = []
        with self.assertRaises(TimeoutError):
            self.run_transition([deepcopy(sample()) for _ in range(50)], [None] * 50, trace)
        self.assertEqual(trace.count("SIGTERM"), 1)
        self.assertNotIn("launch", trace)


if __name__ == "__main__":
    unittest.main(verbosity=2)
