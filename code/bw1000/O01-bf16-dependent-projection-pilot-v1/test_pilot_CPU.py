from contextlib import ExitStack, contextmanager
from functools import partial
import json
from pathlib import Path
import runpy
import sqlite3
import sys
from tempfile import TemporaryDirectory
from threading import Event
import time
from types import SimpleNamespace
import unittest

import dependent
import launch
import package
from contract import BYTES_PER_STAGE, FIRST_ACTIVE, LANES, M, NANO, PHASES, PROBES, amortization, counters, grid, validate_arm
sys.path.insert(0, str(package.REPO / "code/bw1000/fleet-bf16-projection-graph64-setup-v3"))
import report
from scope import EXCLUDED_NODES, PILOT_NODE, PILOT_PARENT, REQUEST
from setup_gate import ARM_LEAD_NS, SETUP_LEAD_NS


sys.path.insert(0, str(package.REPO / "tools"))
import test_report_training80_backlog as dispatch_fixture


def planned_arm(index, window, *, groups=60, finished_seconds=60):
    mode, start, end = window
    finished = start+finished_seconds*NANO
    quarter = [max(0, min(15, groups-index*15)) for index in range(4)]
    return dict(index=index, mode=mode, scheduled_start_epoch_ns=start, scheduled_end_epoch_ns=end,
                actual_start_epoch_ns=start, actual_end_epoch_ns=finished, collection_end_epoch_ns=finished,
                wall_ns=finished-start, state="COMPLETE" if finished_seconds == 60 else "COOPERATIVE_STOPPED",
                passed=finished_seconds == 60, projection_full_BF16=True, projection_CPU_first16=True, consumer_full_CPU=True,
                compute_stream_ms=2., chains_per_submission=LANES, h2d_payload_bytes=0, d2h_payload_bytes=0,
                complete_groups_by_quarter=quarter, **counters(mode, groups))


class ProfileTest(unittest.TestCase):
    def setUp(self):
        self.plan = dict(campaign_epoch=1000.0, shared_start_epoch=1082.0, deadline_epoch=1350.0)
        self.windows = grid(self.plan, 0)

    def test_fixed_O01_packager_rejects_protected_stale_busy_and_foreign_nodes(self):
        fleet = dispatch_fixture.BacklogTest()
        fleet.setUp()
        self.addCleanup(fleet.doCleanups)
        fleet.now = time.monotonic()
        fleet.anchor = time.time()-fleet.now
        for item in fleet.snapshot["nodes"].values():
            item.update(PCIs=[f"0000:{index:02x}:00.0" for index in range(1, 9)],
                        device_probe_step=f"{item['parent']}.900", kfd_process_count=0,
                        own_device_fd_process_count=0, probe_error=None)
        build = partial(package.prepare, "a"*40, self.plan["campaign_epoch"], fleet.verified_snapshot())
        plan, body, members = build(set())
        (label, unit), = plan["shards"].items()
        self.assertEqual(plan["cohort_by_shard"], {label: 0})
        self.assertEqual((unit["parent"], unit["zone"], unit["nodes"]), (PILOT_PARENT, "O01", [PILOT_NODE]))
        self.assertEqual(set(plan["selected"]), {PILOT_NODE})
        self.assertEqual([end-start for _,start,end in grid(plan,0)], [60*NANO]*4)
        self.assertEqual(grid(plan,0)[0][1], round((plan["campaign_epoch"]+FIRST_ACTIVE[0])*NANO))
        self.assertLessEqual(len(body), 131072)
        compile(body, "isolated wire payload", "exec")
        self.assertIn(package.T01_PROVIDER.as_posix(), members)
        self.assertIn(package.V5_TERMINAL.as_posix(), members)
        queue = launch.Queue(fleet.repo/"pilot-fixture.sqlite")
        self.addCleanup(queue.db.close)
        receipt = fleet.verified_snapshot()["local_receipt"]
        with queue.db:
            claimed = launch.reserve_wave(queue, plan, "a"*40, receipt, time.monotonic()+10, "f"*64,
                                          launch.DIRECTORY)
        self.assertEqual(claimed[label]["handle"]["placement_by_parent"], {PILOT_PARENT: {PILOT_NODE: 8}})
        self.assertEqual(queue.summary()["runs"], {"DISPATCHING": 1})
        with self.assertRaises(sqlite3.IntegrityError), queue.db:
            launch.reserve_wave(queue, plan, "a"*40, receipt, time.monotonic()+10, "f"*64, launch.DIRECTORY)
        self.assertEqual(queue.summary()["runs"], {"DISPATCHING": 1})
        with self.assertRaisesRegex(ValueError, "explicit requested nodes"):
            build(set(), [next(iter(EXCLUDED_NODES))])
        with self.assertRaisesRegex(ValueError, "fixed one-node"):
            build(set(), ["f12r2n11"])
        with self.assertRaisesRegex(ValueError, "no freshly clear"):
            build({PILOT_NODE}, [PILOT_NODE])
        fleet.snapshot["nodes"][PILOT_NODE]["kfd_process_count"] = 1
        with self.assertRaisesRegex(ValueError, "no freshly clear"):
            package.select(fleet.verified_snapshot(), set())
        with self.assertRaisesRegex(ValueError, "new CPU-only dispatch snapshot"):
            package.select(fleet.verified_snapshot(age=91), set())
        self.assertEqual(launch.priority_protection(dict(queue_held_nodes=[], original_launch_holds={},
            rows=[dict(dispatch_state="proposed_for_root", kind="training", placement={PILOT_NODE: 1})])), {PILOT_NODE})

    def test_measurement_cooperative_stop_commits_only_completed_graphs(self):
        _, src = package.source(package.REPO)
        with TemporaryDirectory() as folder:
            staged = Path(folder)
            for name, raw in src.items():
                (staged / name).write_bytes(raw)
            measure = runpy.run_path(str(staged / "worker.py"))["measure"]
        mode, start, end = self.windows[1]

        class Clock:
            anchor = 0
            position = start

            def toward(self, target, halt):
                self.position = target
                return self.position

            def monotonic(self):
                return self.position

            def verify(self):
                pass

            def wall_from_mono(self, instant):
                return instant

        clock, stop, verified = Clock(), Event(), []

        def burst(*args):
            clock.position += NANO
            if clock.position == start+19*NANO:
                stop.set()
            return dict(compute_host_ns=5, compute_stream_ms=1., end_ns=clock.position)

        arm = dict(index=1, mode=mode, scheduled_start_epoch_ns=start, scheduled_end_epoch_ns=end)
        measure(None, None, None, lambda: verified.append("checked"), mode, clock, stop, start, end, arm, burst=burst)
        self.assertEqual(validate_arm(arm, 1, self.windows[1]), "COOPERATIVE_STOPPED")
        self.assertEqual((arm["projection_calls"], arm["consumer_calls"], arm["useful_rows"], arm["input_d2d_bytes"]),
                         (38, 38, 38*M, 38*BYTES_PER_STAGE))
        self.assertEqual(counters("graph", 19), counters("eager", 19))
        self.assertEqual(arm["complete_groups_by_quarter"], [14,5,0,0])
        self.assertEqual(verified, ["checked"])
        arm["consumer_calls"] += 1
        with self.assertRaisesRegex(ValueError, "work counters"):
            validate_arm(arm, 1, self.windows[1])

    def test_independent_receipts_keep_complete_and_cooperative_denominators(self):
        addresses = [f"0000:{index:02d}:00.0" for index in range(1,9)]

        def gate(cutoff):
            return dict(expected_cards=8, cutoff_epoch_ns=cutoff, first_failure=None,
                        arrived_epoch_ns={str(card): cutoff-1 for card in range(8)}, released_epoch_ns=cutoff-1)

        for stopped in (False, True):
            with self.subTest(stopped=stopped), TemporaryDirectory() as folder:
                folder = Path(folder).resolve()
                arms = [planned_arm(index, window) for index, window in enumerate(self.windows)]
                if stopped:
                    arms = [arms[0], planned_arm(1, self.windows[1], finished_seconds=17, groups=17)]
                intent = dict(request=REQUEST, node="f12r3n04", step="1087656.841", grid=self.windows,
                              selected_PCI=addresses, planned_cards=8, planned_arms=32, planned_busy_card_samples=384,
                              cohort=0, process_enter_epoch_ns=1000*NANO, setup_open_epoch_ns=1000*NANO,
                              setup_enter_epoch_ns=1001*NANO, torch_import_begin_epoch_ns=1002*NANO,
                              torch_import_end_epoch_ns=1003*NANO, hip_ready_epoch_ns=1004*NANO)
                finished = 8 if not stopped else 0
                terminal = dict(intent, cooperative_stop=stopped, completed_arms=32 if not stopped else 8,
                                valid_cards=finished, setup_gate=gate(self.windows[0][1]-SETUP_LEAD_NS),
                                arm_gates=[gate(item[1]-ARM_LEAD_NS) for item in self.windows])

                def write(name, document):
                    (folder / name).write_text(json.dumps(document), encoding="utf8")

                write("intent.json", intent)
                write("terminal.json", terminal)
                for card, pci in enumerate(addresses):
                    setup_marks = dict(owner_start_epoch_ns=1005*NANO, projection_ready_epoch_ns=1006*NANO,
                                       pair_capture_begin_epoch_ns=1007*NANO, pair_capture_proved_epoch_ns=1008*NANO)
                    prep = dict(capture_ns=100, eager_first_pair_ns=102, graph_first_pair_ns=103,
                                diagnostic_stage_ms_by_lane=[dict.fromkeys(PHASES, .1)]*LANES,
                                graph_changed_input_cpu_proof=True)
                    memory = dict.fromkeys(report.MEMORY_PHASES, dict(free_bytes=9*1024**3,total_bytes=12*1024**3))
                    write(f"setup-card-{card}.json", dict(request=REQUEST, step=intent["step"], node=intent["node"],
                                        card=card, setup_marks=setup_marks, preparation=prep, memory=memory))
                    write(f"card-{card}.json", dict(card=card, PCI=pci, provider_SHA256=report.HIP_SHA, arms=arms,
                            preparation=prep, setup_marks=setup_marks, normal_release=True, error=None, **memory))
                    for index, arm in enumerate(arms):
                        write(f"card-{card}-arm-{index}.json", arm)
                with (folder / "busy.jsonl").open("w", encoding="utf8") as samples:
                    for arm in arms:
                        for offset in PROBES:
                            start = arm["scheduled_start_epoch_ns"]+offset*NANO
                            if start > arm["actual_end_epoch_ns"]:
                                continue
                            samples.write(json.dumps(dict(arm=arm["index"], mode=arm["mode"],
                                planned_epoch_ns=start, observed_begin_epoch_ns=start, observed_end_epoch_ns=start,
                                busy_percent=dict.fromkeys(addresses, 90)))+"\n")
                result = report.compare(folder)
                self.assertEqual(result["valid_cards"], finished)
                self.assertIs(result["passed"], not stopped)
                self.assertEqual(result["completed_arms"], 8 if stopped else 32)
                self.assertEqual(result["actual_busy_card_samples"], 128 if stopped else 384)
                self.assertEqual(len(result["card_setup"]), 8)
                if not stopped:
                    self.assertEqual(result["capture_break_even_pairs"][addresses[0]], [None,None])
                if stopped:
                    partial_arm = result["per_arm"][1]
                    self.assertEqual(partial_arm["measured_partial_cards"], 8)
                    self.assertEqual(partial_arm["useful_rows"], 8*17*LANES*M)



    def test_graph_preparation_replays_two_current_lanes_and_proves_changed_input(self):
        outputs = [None]*LANES
        captures, state, calls = [], dict(sign=1, capture=None), []

        class Graph:
            def replay(self):
                for lane in self.lanes:
                    direct(lane)

            def reset(self):
                calls.append("release")

        class Marker:
            def record(self):
                return None

            def elapsed_time(self, other):
                return .1

        @contextmanager
        def capture(graph, **unused):
            graph.lanes = []
            state["capture"] = graph.lanes
            try:
                yield
            finally:
                captures.append(tuple(graph.lanes))
                state["capture"] = None

        cuda = SimpleNamespace(Event=lambda **kw: Marker(), CUDAGraph=Graph, graph=capture)
        torch = SimpleNamespace(cuda=cuda)
        stream = SimpleNamespace(synchronize=lambda: None)

        def direct(lane, marks=None):
            if state["capture"] is not None:
                state["capture"].append(lane)
            else:
                outputs[lane] = (lane+1)*state["sign"]
            if marks:
                self.assertEqual(len(marks), 4)

        def pair():
            for lane in range(LANES):
                direct(lane)

        def poison():
            outputs[:] = [None]*LANES

        def flip():
            state["sign"] *= -1

        def verify(negative=False):
            self.assertEqual(outputs, [(-1 if negative else 1)*(lane+1) for lane in range(LANES)])
            calls.append("changed" if negative else "normal")

        with ExitStack() as cleanup:
            methods, proof = dependent.prepare(torch, stream, (direct, pair, poison, flip, verify), cleanup)
            for method in methods.values():
                poison()
                method()
                verify()
        self.assertEqual(captures, [(0,1)])
        self.assertIn("changed", calls)
        self.assertEqual(calls[-1], "release")
        self.assertIs(proof["graph_changed_input_cpu_proof"], True)
        self.assertEqual(set(proof["diagnostic_stage_ms_by_lane"][0]), set(PHASES))
        self.assertEqual(amortization(20, 11, 6), 5)
        self.assertIsNone(amortization(20, 6, 6))


if __name__ == "__main__":
    unittest.main()
