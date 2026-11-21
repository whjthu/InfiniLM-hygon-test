from functools import partial
from hashlib import sha256
import json
from pathlib import Path
import runpy
import sys
from tempfile import TemporaryDirectory
from threading import Event
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import controller
import overlap
import package
from contract import DEPTH, EARLY, FIRST_ACTIVE, LATE, NANO, PROBES, cohort_for, counters, grid, validate_arm
sys.path.insert(0, str(package.REPO / "code/bw1000/fleet-bf16-projection-graph64-setup-v3"))
import report
from scope import EXCLUDED_NODES, REQUEST
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
                passed=finished_seconds == 60, full_BF16_equality=True, CPU_first16=True,
                compute_stream_ms=2., captured_gemms_per_replay=DEPTH[mode], h2d_payload_bytes=0, d2h_payload_bytes=0,
                complete_groups_by_quarter=quarter, **counters(mode, groups))


class ProfileTest(unittest.TestCase):
    def setUp(self):
        self.plan = dict(campaign_epoch=1000.0, shared_start_epoch=1048.0, deadline_epoch=1455.0)
        self.windows = grid(self.plan, 0)

    def test_packager_preserves_seven_exclusions_and_training_first(self):
        fleet = dispatch_fixture.BacklogTest()
        fleet.setUp()
        self.addCleanup(fleet.doCleanups)
        fleet.now = time.monotonic()
        fleet.anchor = time.time()-fleet.now
        for item in fleet.snapshot["nodes"].values():
            item.update(PCIs=[f"0000:{index:02x}:00.0" for index in range(1, 9)],
                        device_probe_step=f"{item['parent']}.900", kfd_process_count=0,
                        own_device_fd_process_count=0, probe_error=None)
        build = partial(package.prepare, "a"*40, time.time(), fleet.verified_snapshot())
        plan, body, members = build(set())
        self.assertEqual((len(plan["selected"]), len(plan["shards"])), (73, 31))
        self.assertEqual(set(fleet.snapshot["nodes"])-set(plan["selected"]), EXCLUDED_NODES)
        self.assertEqual([end-start for _,start,end in grid(plan, 0)], [60*NANO]*5)
        self.assertLessEqual(len(body), 131072)
        self.assertIn(package.V3_TERMINAL.as_posix(), members)
        self.assertIn(package.V4_TERMINAL.as_posix(), members)
        for late in LATE:
            self.assertEqual(cohort_for(plan, late), 2)
        early_nodes = {node for node in plan["selected"] if cohort_for(plan, node) == 0}
        self.assertTrue(early_nodes)
        self.assertLessEqual(early_nodes, EARLY)
        for index, expected in enumerate(FIRST_ACTIVE):
            windows = grid(plan, index)
            self.assertEqual(windows[0][1], round((plan["campaign_epoch"]+expected)*NANO))
            self.assertLess(windows[3][2], windows[4][1])
        self.assertEqual({grid(plan, index)[4] for index in range(len(FIRST_ACTIVE))},
                         {grid(plan, 0)[4]})
        with self.assertRaisesRegex(ValueError, "derived group"):
            package.validate_plan(dict(plan, cohort_by_shard=dict.fromkeys(plan["shards"], 0)),
                                  package.ledger(package.source(package.REPO)[1]["ledger.json"]))
        protected = sorted(plan["selected"])[0]
        smaller, _, _ = build({protected})
        self.assertEqual(set(smaller["selected"]), set(plan["selected"])-{protected})
        t09 = "f11r4n17"
        subset = set(plan["selected"])-{t09}
        exact, _, _ = build({t09}, subset)
        self.assertEqual(set(exact["selected"]), subset)
        self.assertNotIn(t09, {node for shard in exact["shards"].values() for node in shard["nodes"]})
        with self.assertRaisesRegex(ValueError, "original blocked"):
            build(set(), [next(iter(EXCLUDED_NODES))])
        with self.assertRaisesRegex(ValueError, "lacks clearance|no freshly clear"):
            build({protected}, [protected])

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
        self.assertEqual((arm["gemm_calls"], arm["useful_rows"]), (19*32, 19*32*186))
        self.assertEqual(counters("graph32", 38)["useful_rows"], counters("graph64", 19)["useful_rows"])
        self.assertEqual(arm["complete_groups_by_quarter"], [14,5,0,0])
        self.assertEqual(verified, ["checked"])
        arm["useful_rows"] += 1
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
                              selected_PCI=addresses, planned_cards=8, planned_arms=40, planned_busy_card_samples=480,
                              cohort=0, process_enter_epoch_ns=1000*NANO, setup_open_epoch_ns=1000*NANO,
                              setup_enter_epoch_ns=1001*NANO, torch_import_begin_epoch_ns=1002*NANO,
                              torch_import_end_epoch_ns=1003*NANO, hip_ready_epoch_ns=1004*NANO)
                finished = 8 if not stopped else 0
                terminal = dict(intent, cooperative_stop=stopped, completed_arms=40 if not stopped else 8,
                                valid_cards=finished, setup_gate=gate(self.windows[0][1]-SETUP_LEAD_NS),
                                arm_gates=[gate(item[1]-ARM_LEAD_NS) for item in self.windows])

                def write(name, document):
                    (folder / name).write_text(json.dumps(document), encoding="utf8")

                write("intent.json", intent)
                write("terminal.json", terminal)
                for card, pci in enumerate(addresses):
                    setup_marks = dict(owner_start_epoch_ns=1005*NANO, projection_ready_epoch_ns=1006*NANO,
                                       pair_capture_begin_epoch_ns=1007*NANO, pair_capture_proved_epoch_ns=1008*NANO)
                    capture_ns = {"graph64": 100, "graph32": 101}
                    write(f"setup-card-{card}.json", dict(request=REQUEST, step=intent["step"], node=intent["node"],
                                                        card=card, setup_marks=setup_marks, capture_ns=capture_ns))
                    write(f"card-{card}.json", dict(card=card, PCI=pci, provider_SHA256=report.HIP_SHA, arms=arms,
                                                   capture_ns=capture_ns, setup_marks=setup_marks,
                                                   normal_release=True, error=None))
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
                self.assertEqual(result["completed_arms"], 8 if stopped else 40)
                self.assertEqual(result["actual_busy_card_samples"], 128 if stopped else 480)
                self.assertEqual(len(result["card_setup"]), 8)
                expected_common = None if stopped else list(self.windows[-1][1:])
                self.assertEqual(result["common_window_epoch_ns"], expected_common)
                if stopped:
                    partial_arm = result["per_arm"][1]
                    self.assertEqual(partial_arm["measured_partial_cards"], 8)
                    self.assertEqual(partial_arm["useful_rows"], 8*17*32*186)


    def test_launch_and_capture_have_separate_startup_markers(self):
        node, step = "f12r3n04", "1087656.841"
        plan = dict(shards={"fixture": {"nodes": [node]}}, cohort_by_shard={"fixture": 2})
        with TemporaryDirectory() as folder:
            path = Path(folder)
            for num, phase in enumerate(("process-enter", "stop-handler-ready", "setup-enter", "torch-import-begin"), 1):
                (path/(phase+".json")).write_text(json.dumps(dict(
                    request=REQUEST, node=node, step=step, epoch_ns=num*NANO, cohort=2)), encoding="utf8")
            marks = controller.startup_observation(path, node, (step,), plan, Path.read_bytes)
            self.assertEqual(list(marks), ["process-enter", "stop-handler-ready", "setup-enter", "torch-import-begin"])
            self.assertEqual(marks["torch-import-begin"]["epoch_ns"], 4*NANO)
            with self.assertRaisesRegex(ValueError, "phase differs"):
                controller.startup_observation(path, node, ("1087656.840",), plan, Path.read_bytes)

    def test_early_stop_requires_original_worker_handler(self):
        node, parent, step = "f12r3n04", "1087656", "1087656.841"
        label, shard_request = "fixture", "1"*32
        unit = dict(request=shard_request, parent=parent, zone="I01", nodes=[node])
        campaign = dict(shards={label: unit}, cohort_by_shard={label: 2})
        argv = ["srun", "--job-name="+label]
        owner = dict(pid=123, start_ticks=9, uid=1, argv=argv)
        accounting = [step, label, "RUNNING", "0:0", "60", node, "cpu=12,node=1,gres/dcu=8,mem=64G"]
        live = dict(StepId=step, NodeList=node)
        scheduler = SimpleNamespace(allocation=lambda job, nodes: set(nodes), hosts=lambda expression: [expression],
                                    query=lambda *args: [live], account=lambda labels: {label: accounting})
        with TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            run = root/"results"/controller.RUN
            attempt = run/"attempts"/label
            worker = run/"nodes"/node
            attempt.mkdir(parents=True)
            worker.mkdir(parents=True)
            common = dict(unit, label=label)
            (attempt/"owner.json").write_text(json.dumps(dict(common, **owner)), encoding="utf8")
            (attempt/"intent.json").write_text(json.dumps(dict(common, attempted=True, argv=argv)), encoding="utf8")
            before = dict(request=REQUEST, node=node, step=step, cohort=2, epoch_ns=100*NANO)
            (worker/"process-enter.json").write_text(json.dumps(before), encoding="utf8")
            with patch.object(controller, "ROOT", root), patch.object(controller, "proc_identity", return_value=owner):
                inspect = partial(controller.inspect_stop, campaign, label, shard_request, step,
                                  {node: (parent, "I01")}, scheduler, Path.read_bytes, lambda row: row)
                with self.assertRaisesRegex(ValueError, "handler is not confirmed"):
                    inspect()
                (worker/"stop-handler-ready.json").write_text(json.dumps(dict(before, epoch_ns=101*NANO)),
                                                              encoding="utf8")
                receipt = inspect()
                self.assertEqual(receipt["worker_stop_readiness"][node]["kind"], "stop_handler_ready")
                self.assertEqual(receipt["signal_argv"], ["scancel", "--signal=USR1", step])
                self.assertTrue(receipt["independent_terminal_and_post_required"])

    def test_common_requires_every_exact_independent_clock(self):
        start, end = self.windows[-1][1:]
        nodes = {"a": dict(passed=True, original_step="1087656.841", common_window_epoch_ns=[start, end]),
                 "b": dict(passed=True, original_step="1103606.800", common_window_epoch_ns=[start, end])}
        terminal = json.dumps(dict(request=REQUEST, campaign_epoch=1000.0, planned_nodes=2, planned_cards=16,
                                   nodes=nodes)).encode()
        bound = dict(max_abs_error_ns=1000, valid_from_epoch_ns=start-1, valid_until_epoch_ns=end+1,
                     raw_clock_evidence_SHA256="a"*64)

        def observed(evidence):
            raw = json.dumps(dict(request=REQUEST, source_role="INDEPENDENT_BOUNDED_TARGET_CLOCKS",
                                  terminal_SHA256=sha256(terminal).hexdigest(), nodes=evidence)).encode()
            return raw, sha256(raw).hexdigest()

        self.assertIsNone(overlap.audit(terminal)["simultaneous_sustained_ns"])
        evidence = {name: dict(bound, step=node["original_step"]) for name, node in nodes.items()}
        self.assertIsNone(overlap.audit(terminal, *observed({"a": evidence["a"]}))["simultaneous_sustained_ns"])
        accepted = overlap.audit(terminal, *observed(evidence))
        self.assertEqual(accepted["simultaneous_sustained_ns"], 60*NANO-2000)
        raw, _ = observed(evidence)
        with self.assertRaisesRegex(ValueError, "SHA differs"):
            overlap.audit(terminal, raw, "0"*64)
        evidence["b"]["step"] = "1103606.801"
        with self.assertRaisesRegex(ValueError, "exact bounded"):
            overlap.audit(terminal, *observed(evidence))


if __name__ == "__main__":
    unittest.main()
