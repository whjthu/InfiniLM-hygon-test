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
from unittest.mock import patch
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent))
import dependent
import decision
import controller
import launch
import package
from contract import BYTES_PER_STAGE, DEPTH, FIRST_ACTIVE, HOST_COUNTERS, LANES, M, NANO, PHASES, PROBES, amortization, counters, grid, polarity_order, validate_arm
sys.path.insert(0, str(package.REPO / "code/bw1000/fleet-bf16-projection-graph64-setup-v3"))
import report
from scope import EXCLUDED_NODES, REPRESENTATIVE_NODES, REQUEST
from setup_gate import ARM_LEAD_NS, SETUP_LEAD_NS


sys.path.insert(0, str(package.REPO / "tools"))
import test_report_training80_backlog as dispatch_fixture


def planned_arm(index, window, *, groups=60, finished_seconds=30):
    mode, start, end = window
    finished = start+finished_seconds*NANO
    quarter = [max(0, min(15, groups-index*15)) for index in range(4)]
    return dict(index=index, mode=mode, scheduled_start_epoch_ns=start, scheduled_end_epoch_ns=end,
                actual_start_epoch_ns=start, actual_end_epoch_ns=finished, collection_end_epoch_ns=finished,
                wall_ns=finished-start, state="COMPLETE" if finished_seconds == 30 else "COOPERATIVE_STOPPED",
                passed=finished_seconds == 30, projection_full_BF16=True, projection_CPU_first16=True, consumer_full_CPU=True,
                compute_stream_ms=2., chains_per_submission=LANES*DEPTH[mode], h2d_payload_bytes=0, d2h_payload_bytes=0,
                complete_groups_by_quarter=quarter, terminal_polarity=(groups*DEPTH[mode]-1)%2,
                polarity_changes_between_replays=groups-1, **dict.fromkeys(HOST_COUNTERS, 1),
                **counters(mode, groups))


class ProfileTest(unittest.TestCase):
    def setUp(self):
        self.plan = dict(campaign_epoch=1000.0, shared_start_epoch=1082.0, deadline_epoch=1260.0)
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
            self.assertEqual([end-start for _,start,end in windows], [30*NANO]*4)
        self.assertLessEqual(len(body), 131072)
        compile(body, "isolated wire payload", "exec")
        self.assertIn(package.ORIGINAL.as_posix(), members)
        self.assertIn(package.V5_TERMINAL.as_posix(), members)
        self.assertIn(package.FLEET_TERMINAL.as_posix(), members)
        self.assertIn(package.FOUR_TERMINAL.as_posix(), members)
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
        with self.assertRaisesRegex(ValueError, "all four original O01"):
            build({"f12r2n13"})
        with self.assertRaisesRegex(ValueError, "fixed O01"):
            build(set(), ["f12r2n10", "f12r3n17"])
        fleet.snapshot["nodes"]["f12r2n13"]["kfd_process_count"] = 1
        with self.assertRaisesRegex(ValueError, "all four original O01"):
            package.select(fleet.verified_snapshot(), set())
        with self.assertRaisesRegex(ValueError, "new CPU-only dispatch snapshot"):
            package.select(fleet.verified_snapshot(age=91), set())

    def test_remote_known_conflict_spawns_neither_parent(self):
        with TemporaryDirectory() as folder:
            workspace = Path(folder).resolve(strict=True)
            (workspace / 'containers/dtk26-sandbox-r1').mkdir(parents=True)
            (workspace / 'results').mkdir()
            owners = {node:(parent,'O01') for node,parent in REPRESENTATIVE_NODES.items()}
            shards = {parent:dict(request='a'*32,parent=parent,zone='O01',
                                  nodes=[n for n,p in REPRESENTATIVE_NODES.items() if p==parent])
                      for parent in set(REPRESENTATIVE_NODES.values())}
            plan = dict(campaign_epoch=time.time(), shards=shards, selected=owners,
                        cohort_by_shard={parent:idx for idx,parent in enumerate(sorted(shards))})
            scheduler = SimpleNamespace(account=lambda _:{}, allocation=lambda parent,_:
                                        {'f12r2n13'} if parent=='1103606' else set())
            def save(path, document):
                path.write_text(json.dumps(document), encoding='utf8')
            with (patch.object(controller,'ROOT',workspace), patch.object(controller.os,'access',return_value=True),
                  patch.object(controller.subprocess,'Popen') as popen):
                result = controller.dispatch(workspace,plan,owners,scheduler,save)
                popen.assert_not_called()
            self.assertEqual(result['planned_cards'],32)
            self.assertEqual(result['client_count'],0)

    def test_candidate_gate_uses_both_orders_all_nodes_and_capture_payback(self):
        summary = dict(device_qualified=True, planned_cards=32, valid_cards=32, planned_arms=128, completed_arms=128,
                       planned_busy_card_samples=768, actual_busy_card_samples=768,
                       paired_comparison=dict(planned_pairs=64, measured_pairs=64, qualified_nodes=4), nodes={})
        for node in decision.NODES:
            ratios = {str(card):[1.04,1.04] for card in range(8)}
            factors = {str(card):[.02,.03] for card in range(8)}
            summary['nodes'][node] = dict(passed=True, capture_window_fraction=factors,
                                          **{decision.PER_CARD_PAIRS:ratios})
        self.assertEqual(decision.classify(summary)['qualification'],'KEEP_COMPONENT_CANDIDATE')
        for node,row in summary['nodes'].items():
            for values in row[decision.PER_CARD_PAIRS].values():
                values[1]=.99
        self.assertEqual(decision.classify(summary)['qualification'],'STOP_AT_DEPTH9')
        for row in summary['nodes'].values():
            for values in row[decision.PER_CARD_PAIRS].values():
                values[1]=1.04
            for values in row['capture_window_fraction'].values():
                values[1]=None
        self.assertEqual(decision.classify(summary)['qualification'],'STOP_AT_DEPTH9')
        summary['valid_cards']=31
        self.assertEqual(decision.classify(summary)['qualification'],'INCOMPLETE')

    def test_reconcile_actual_labels_and_keep_unknowns(self):
        shards = {parent:dict(parent=parent, nodes=[n for n,p in REPRESENTATIVE_NODES.items() if p==parent])
                  for parent in set(REPRESENTATIVE_NODES.values())}
        plan = dict(campaign_epoch=1000., selected=REPRESENTATIVE_NODES, shards=shards,
                    cohort_by_shard={'1087656':0,'1103606':1})
        rows = {label:[item['parent']+'.999', label, 'COMPLETED', '0:0', '246',
                       ','.join(item['nodes']), 'node=2,cpu=24,gres/dcu=16,mem=128G']
                for label,item in shards.items()}
        scheduler = SimpleNamespace(account=lambda _:dict(rows), hosts=lambda text:text.split(','), TERMINAL={'COMPLETED'})
        def audit(_path, node, _row, _plan, _read, _compare):
            ratios = {f'{node}-{card}':[1.04,1.04] for card in range(8)}
            return dict(valid_cards=8, completed_arms=32, actual_busy_card_samples=192, passed=True,
                        capture_window_fraction={card:[.25,.25] for card in ratios},
                        **{decision.PER_CARD_PAIRS:ratios})
        with TemporaryDirectory() as folder:
            root = Path(folder).resolve(strict=True)
            with patch.object(controller,'ROOT',root), patch.object(controller,'bound_node',side_effect=audit):
                first = controller.reconcile(plan, scheduler, Path.read_bytes, None)
                self.assertEqual(first['depth15_decision']['qualification'],'KEEP_COMPONENT_CANDIDATE')
                rows.pop('1103606')
                observed = controller.reconcile(plan, scheduler, Path.read_bytes, None)
            self.assertEqual(observed['planned_cards'],32)
            self.assertEqual(observed['paired_comparison']['measured_pairs'],32)
            self.assertEqual(observed['depth15_decision']['qualification'],'INCOMPLETE')
            self.assertEqual(observed['shards']['1103606']['state'],'UNKNOWN')

    def test_measurement_cooperative_stop_commits_only_completed_graphs(self):
        _, src = package.source(package.REPO)
        with TemporaryDirectory() as folder:
            staged = Path(folder).resolve(strict=True)
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
                         (570, 570, 570*M, 570*BYTES_PER_STAGE))
        self.assertEqual(counters("depth15",3)["useful_rows"], counters("depth9",5)["useful_rows"])
        self.assertEqual(actual_replay_phases, [i%2 for i in range(19)])
        self.assertEqual(arm["polarity_changes_between_replays"], 18)
        self.assertEqual(arm["terminal_polarity"], 0)
        self.assertEqual(arm["complete_groups_by_quarter"], [7,7,5,0])
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
                              selected_PCI=addresses, planned_cards=8, planned_arms=32, planned_busy_card_samples=192,
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
                self.assertEqual(result["actual_busy_card_samples"], 80 if stopped else 192)
                self.assertEqual(len(result["card_setup"]), 8)
                if not stopped:
                    self.assertEqual(result["capture_break_even_pairs"][addresses[0]], [1,1])
                if stopped:
                    partial_arm = result["per_arm"][1]
                    self.assertEqual(partial_arm["measured_partial_cards"], 8)
                    self.assertEqual(partial_arm["useful_rows"], 8*17*15*LANES*M)



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
        self.assertEqual(captures, [polarity_order(mode, start) for mode in DEPTH for start in (0,1)])
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
