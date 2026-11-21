from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from functools import partial
import json
from pathlib import Path
import sys
from threading import Event
import time
import unittest

import controller
import package
from setup_gate import ARM_LEAD_NS, SETUP_LEAD_NS, PhaseClosed, PhaseGate
from scope import ALLOWED_NODES, REQUIRED_NODES, REQUEST


sys.path.insert(0, str(package.REPO / "tools"))
import test_report_training80_backlog as fixture


class MockClock:
    anchor = 100_000_000_000
    now = 0

    def monotonic(self):
        return self.now

    def wall_from_mono(self, value):
        return self.anchor + value

    def verify(self):
        pass


def wait_arrival(gate, card):
    deadline = time.monotonic() + 1
    while card not in gate.summary()["arrived_epoch_ns"]:
        if time.monotonic() >= deadline:
            raise TimeoutError("fixture owner did not enter the gate")
        time.sleep(0.002)


class SetupTest(unittest.TestCase):
    def test_first_owner_can_wait_past_old_twenty_five_seconds(self):
        clock = MockClock()
        setup = PhaseGate(clock, clock.anchor+50*10**9, 8, Event().is_set)
        with ThreadPoolExecutor(8) as pool:
            first = pool.submit(setup.wait, 0, "setup")
            wait_arrival(setup, 0)
            clock.now = 35*10**9
            rest = [pool.submit(setup.wait, card, "setup") for card in range(1, 8)]
            releases = [future.result(timeout=1) for future in (first, *rest)]
        self.assertEqual({item.released_epoch_ns for item in releases}, {clock.anchor + clock.now})
        self.assertEqual(releases[0].earliest_arrival_epoch_ns, clock.anchor)
        self.assertEqual(releases[0].latest_arrival_epoch_ns, clock.anchor+35*10**9)

    def test_setup_primary_cause_preserved_and_arm_cannot_slide(self):
        clock = MockClock()
        setup = PhaseGate(clock, clock.anchor+50*10**9, 8, Event().is_set)
        with ThreadPoolExecutor(2) as pool:
            owner = pool.submit(setup.wait, 0, "setup_gate")
            wait_arrival(setup, 0)
            setup.abort(4, "capture", "ValueError")
            setup.abort(0, "setup_gate", "PhaseClosed")
            with self.assertRaisesRegex(PhaseClosed, "capture: ValueError"):
                owner.result(timeout=1)
        self.assertEqual(setup.summary()["first_failure"]["card"], 4)
        original_start = clock.anchor+60*10**9
        arm = PhaseGate(clock, original_start-ARM_LEAD_NS, 8, Event().is_set)
        clock.now = 60*10**9-ARM_LEAD_NS
        with self.assertRaisesRegex(PhaseClosed, "PHASE_CUTOFF"):
            arm.wait(7, "eager_arm_0_gate")
        self.assertIsNone(arm.summary()["released_epoch_ns"])
        self.assertEqual(arm.summary()["cutoff_epoch_ns"], original_start-ARM_LEAD_NS)

    def test_cooperative_abort_never_releases_lone_owner(self):
        clock = MockClock()
        stopped = Event()
        gate = PhaseGate(clock, clock.anchor+50*10**9, 8, stopped.is_set)
        with ThreadPoolExecutor(1) as pool:
            owner = pool.submit(gate.wait, 0, "setup")
            wait_arrival(gate, 0)
            stopped.set()
            with self.assertRaisesRegex(PhaseClosed, "COOPERATIVE_STOP"):
                owner.result(timeout=1)
        self.assertEqual(gate.summary()["arrived_epoch_ns"], {0: clock.anchor})

    def test_controller_requires_fixed_boundaries_and_eight_owners(self):
        start = 150_000_000_000
        node, step = "f13r3n13", "1103606.4"

        def gate(cutoff):
            return dict(cutoff_epoch_ns=cutoff, expected_cards=8,
                        arrived_epoch_ns={str(card): cutoff-100+card for card in range(8)},
                        released_epoch_ns=cutoff-93, first_failure=None)

        grid = [[mode, start+offset*10**9, start+(offset+8)*10**9]
                for mode, offset in zip(("eager", "graph", "graph", "eager"), (0, 12, 24, 36), strict=True)]
        report = dict(request=REQUEST, node=node, step=step, process_enter_epoch_ns=start-10, grid=grid,
                      cards=[{"card": card} for card in range(8)], setup_gate=gate(start-SETUP_LEAD_NS),
                      arm_gates=[gate(row[1]-ARM_LEAD_NS) for row in grid])
        read = partial(controller.phase_diagnostic, node=node, row=[step], start=start)
        self.assertTrue(read(json.dumps(report).encode())["all_released"])
        report["arm_gates"][1]["arrived_epoch_ns"].pop("7")
        with self.assertRaisesRegex(ValueError, "without eight"):
            read(json.dumps(report).encode())

    def test_real_packager_default_three_pairs_and_optional_clearance(self):
        scope = fixture.BacklogTest()
        scope.setUp()
        self.addCleanup(scope.doCleanups)
        scope.now = time.monotonic()
        scope.anchor = time.time() - scope.now
        for row in scope.snapshot["nodes"].values():
            row.update(PCIs=[f"0000:{index:02x}:00.0" for index in range(1, 9)],
                       device_probe_step=f"{row['parent']}.900", kfd_process_count=0,
                       own_device_fd_process_count=0, probe_error=None)

        def render(requested=None, protected=()):
            return package.prepare("a"*40, time.time(), scope.verified_snapshot(), set(protected), requested)

        plan, body, sources = render()
        self.assertEqual(set(plan["selected"]), REQUIRED_NODES)
        self.assertEqual(len(plan["shards"]), 3)
        self.assertTrue(all(len(item["nodes"]) == 2 for item in plan["shards"].values()))
        self.assertTrue(plan["shared_start_epoch"] == plan["campaign_epoch"]+60 and len(body) <= 131072)
        self.assertIn(package.V2_TERMINAL.as_posix(), sources)
        optional = "f12r2n12"
        self.assertIn(optional, ALLOWED_NODES - REQUIRED_NODES)
        expanded, _, _ = render(REQUIRED_NODES | {optional})
        self.assertEqual(len(expanded["selected"]), 7)
        with self.assertRaisesRegex(ValueError, "lacks clearance"):
            render(REQUIRED_NODES | {optional}, {optional})
        with self.assertRaisesRegex(ValueError, "comparison scope"):
            render(REQUIRED_NODES - {"f13r3n13"})


if __name__ == "__main__":
    unittest.main()
