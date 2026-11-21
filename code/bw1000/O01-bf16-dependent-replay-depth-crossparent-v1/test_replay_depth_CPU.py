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
from contract import BYTES_PER_STAGE, DEPTH, FIRST_ACTIVE, HOST_COUNTERS, LANES, M, NANO, PHASES, PROBES, amortization, counters, grid, polarity_order, validate_arm
sys.path.insert(0, str(package.REPO / "code/bw1000/fleet-bf16-projection-graph64-setup-v3"))
import report
from scope import EXCLUDED_NODES, REPRESENTATIVE_NODES, REQUEST
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
                compute_stream_ms=2., chains_per_submission=LANES*DEPTH[mode], h2d_payload_bytes=0, d2h_payload_bytes=0,
                complete_groups_by_quarter=quarter, terminal_polarity=(groups*DEPTH[mode]-1)%2,
                polarity_changes_between_replays=groups-1, **dict.fromkeys(HOST_COUNTERS, 1),
                **counters(mode, groups))


class ProfileTest(unittest.TestCase):
    def setUp(self):
        self.plan = dict(campaign_epoch=1000.0, shared_start_epoch=1082.0, deadline_epoch=1390.0)
        self.windows = grid(self.plan, 0)

    def test_both_exact_O01_parents_and_fresh_shard_reservations(self):
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
        self.assertEqual(set(plan["selected"]), set(REPRESENTATIVE_NODES))
        self.assertEqual(len(plan["shards"]), 2)
        self.assertEqual(sorted(plan["cohort_by_shard"].values()), [0,1])
        for index,first in enumerate(FIRST_ACTIVE):
            windows = grid(plan,index)
            self.assertEqual(windows[0][1], round((plan["campaign_epoch"]+first)*NANO))
            self.assertEqual([end-start for _,start,end in windows], [60*NANO]*4)
        self.assertLessEqual(len(body), 131072)
        compile(body, "isolated wire payload", "exec")
        self.assertIn(package.ORIGINAL.as_posix(), members)
        self.assertIn(package.V5_TERMINAL.as_posix(), members)
        queue = launch.Queue(fleet.repo/"depth-fixture.sqlite")
        self.addCleanup(queue.db.close)
        receipt = fleet.verified_snapshot()["local_receipt"]
        with queue.db:
            claimed = launch.reserve_wave(queue, plan, "a"*40, receipt, time.monotonic()+10, "f"*64, launch.DIRECTORY)
        self.assertEqual({entry["handle"]["parent"] for entry in claimed.values()}, set(REPRESENTATIVE_NODES.values()))
        self.assertEqual(queue.summary()["runs"], {"DISPATCHING": 2})
        with self.assertRaises(sqlite3.IntegrityError), queue.db:
            launch.reserve_wave(queue, plan, "a"*40, receipt, time.monotonic()+10, "f"*64, launch.DIRECTORY)
        self.assertEqual(queue.summary()["runs"], {"DISPATCHING": 2})
        with self.assertRaisesRegex(ValueError, "explicit requested nodes"):
            build(set(), [next(iter(EXCLUDED_NODES))])
        with self.assertRaisesRegex(ValueError, "fixed O01"):
            build(set(), ["f13r1n01"])
        with self.assertRaisesRegex(ValueError, "both O01 parents"):
            build({"f12r2n10","f12r2n13"})
        smaller, _, _ = build({"f12r2n13"})
        self.assertEqual(set(smaller["selected"]), set(REPRESENTATIVE_NODES)-{"f12r2n13"})
        with self.assertRaisesRegex(ValueError, "lacks clearance"):
            build({"f12r2n13"}, list(REPRESENTATIVE_NODES))
        fleet.snapshot["nodes"]["f12r2n13"]["kfd_process_count"] = 1
        self.assertNotIn("f12r2n13", package.select(fleet.verified_snapshot(), set()))
        with self.assertRaisesRegex(ValueError, "new CPU-only dispatch snapshot"):
            package.select(fleet.verified_snapshot(age=91), set())

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

        actual_replay_phases = []
        variants = tuple(lambda polarity=polarity: actual_replay_phases.append(polarity)
                         for polarity in (0,1))

        def burst(_, streams, variant, _clock):
            variant()
            clock.position += NANO
            if clock.position == start+19*NANO:
                stop.set()
            return dict(**dict.fromkeys(HOST_COUNTERS,5),compute_stream_ms=1.,end_ns=clock.position)

        arm = dict(index=1, mode=mode, scheduled_start_epoch_ns=start, scheduled_end_epoch_ns=end)
        measure(None, None, variants, verified.append, mode, clock, stop, start, end, arm, burst=burst)
        self.assertEqual(validate_arm(arm, 1, self.windows[1]), "COOPERATIVE_STOPPED")
        self.assertEqual((arm["projection_calls"], arm["consumer_calls"], arm["useful_rows"], arm["input_d2d_bytes"]),
                         (114, 114, 114*M, 114*BYTES_PER_STAGE))
        self.assertEqual(counters("depth3",19)["useful_rows"], counters("depth1",57)["useful_rows"])
        self.assertEqual(actual_replay_phases, [i%2 for i in range(19)])
        self.assertEqual(arm["polarity_changes_between_replays"], 18)
        self.assertEqual(arm["terminal_polarity"], 0)
        self.assertEqual(arm["complete_groups_by_quarter"], [14,5,0,0])
        self.assertEqual(verified, [(0,1)])
        bad_consumer = dict(arm, consumer_calls=arm["consumer_calls"]+1)
        with self.assertRaisesRegex(ValueError, "work counters"):
            validate_arm(bad_consumer, 1, self.windows[1])
        bad_polarity = dict(arm, polarity_changes_between_replays=0)
        with self.assertRaisesRegex(ValueError, "proof"):
            validate_arm(bad_polarity, 1, self.windows[1])
        bad_sync = dict(arm, host_sync_wait_ns=arm["wall_ns"])
        with self.assertRaisesRegex(ValueError, "host submit"):
            validate_arm(bad_sync, 1, self.windows[1])

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
                    prep = dict(by_depth={mode:dict(capture_ns=[100,101],first_replay_ns=[102,103]) for mode in DEPTH},
                                diagnostic_stage_ms_by_lane=[dict.fromkeys(PHASES, .1)]*LANES,
                                graph_changed_input_across_replays_cpu_proof=True)
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
                    self.assertEqual(result["capture_break_even_pairs"][addresses[0]], [1,1])
                if stopped:
                    partial_arm = result["per_arm"][1]
                    self.assertEqual(partial_arm["measured_partial_cards"], 8)
                    self.assertEqual(partial_arm["useful_rows"], 8*17*3*LANES*M)



    def test_both_graph_variants_follow_one_logical_input_sequence(self):
        captures, output, state, calls = [], {}, dict(sign=1, capture=None), []

        class Graph:
            def replay(self):
                for polarity in self.phases:
                    pair(polarity)

            def reset(self):
                calls.append("released")

        class Marker:
            def record(self):
                return None

            def elapsed_time(self, other):
                return .1

        @contextmanager
        def capture(graph, **unused):
            graph.phases = []
            state["capture"] = graph.phases
            try:
                yield
            finally:
                captures.append(tuple(graph.phases))
                state["capture"] = None

        torch = SimpleNamespace(cuda=SimpleNamespace(Event=lambda **kw: Marker(), CUDAGraph=Graph, graph=capture))
        stream = SimpleNamespace(synchronize=lambda: None)

        def direct(polarity, lane, marks=None):
            output[polarity,lane] = (lane+1)*state["sign"]*(-1 if polarity else 1)

        def pair(polarity):
            if state["capture"] is not None:
                state["capture"].append(polarity)
            else:
                for lane in range(LANES):
                    direct(polarity,lane)

        def poison():
            output.clear()

        def flip():
            state["sign"] *= -1

        def verify(polarities, flipped=False):
            for polarity in polarities:
                for lane in range(LANES):
                    self.assertEqual(output[polarity,lane], (lane+1)*(-1 if bool(polarity)^flipped else 1))
            calls.append("flipped" if flipped else "known")

        with ExitStack() as cleanup:
            methods, proof = dependent.prepare(torch, stream, (direct,pair,poison,flip,verify), cleanup)
            for variants in methods.values():
                poison()
                for method in variants:
                    method()
                verify((0,1))
        self.assertEqual(captures, [(0,),(1,),(0,1,0),(1,0,1)])
        for mode in DEPTH:
            sequence = [phase for replay in range(4) for phase in polarity_order(mode,replay%2)]
            self.assertEqual(sequence, [i%2 for i in range(DEPTH[mode]*4)])
        self.assertEqual(calls.count("flipped"), 4)
        self.assertEqual(calls.count("released"), 4)
        self.assertIs(proof["graph_changed_input_across_replays_cpu_proof"], True)
        self.assertEqual(set(proof["diagnostic_stage_ms_by_lane"][0]), set(PHASES))
        self.assertEqual(amortization(20, 11, 6), 5)
        self.assertIsNone(amortization(20, 6, 6))


if __name__ == "__main__":
    unittest.main()
