from functools import partial
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
import package
from contract import NANO, PROBES, counters, grid, validate_arm
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
                passed=finished_seconds == 60, full_BF16_equality=True, CPU_first16=True, padding_zero=True,
                compute_stream_ms=2., captured_gemms_per_replay=64, h2d_payload_bytes=0, d2h_payload_bytes=0,
                complete_groups_by_quarter=quarter, **counters(mode, groups))


class ProfileTest(unittest.TestCase):
    def setUp(self):
        self.plan = dict(campaign_epoch=1000.0, shared_start_epoch=1060.0, deadline_epoch=1330.0)
        self.windows = grid(self.plan)

    def test_packager_preserves_five_exclusions_and_training_first(self):
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
        self.assertEqual((len(plan["selected"]), len(plan["shards"])), (75, 31))
        self.assertEqual(set(fleet.snapshot["nodes"])-set(plan["selected"]), EXCLUDED_NODES)
        self.assertEqual([end-start for _,start,end in grid(plan)], [60*NANO]*4)
        self.assertLessEqual(len(body), 131072)
        self.assertIn(package.V3_TERMINAL.as_posix(), members)
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
        measure(None, None, None, verified.append, mode, clock, stop, start, end, arm, burst=burst)
        self.assertEqual(validate_arm(arm, 1, self.windows[1]), "COOPERATIVE_STOPPED")
        self.assertEqual((arm["gemm_calls"], arm["useful_rows"], arm["padded_rows"]),
                         (19*64, 19*64*186, 19*64*6))
        self.assertEqual(arm["complete_groups_by_quarter"], [14,5,0,0])
        self.assertEqual(verified, [mode])
        arm["useful_rows"] = arm["physical_rows"]
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
                              selected_PCI=addresses, planned_cards=8, planned_arms=32, planned_busy_card_samples=384)
                finished = 8 if not stopped else 0
                terminal = dict(intent, cooperative_stop=stopped, completed_arms=32 if not stopped else 8,
                                valid_cards=finished, setup_gate=gate(self.windows[0][1]-SETUP_LEAD_NS),
                                arm_gates=[gate(item[1]-ARM_LEAD_NS) for item in self.windows])

                def write(name, document):
                    (folder / name).write_text(json.dumps(document), encoding="utf8")

                write("intent.json", intent)
                write("terminal.json", terminal)
                for card, pci in enumerate(addresses):
                    write(f"card-{card}.json", dict(card=card, PCI=pci, provider_SHA256=report.HIP_SHA, arms=arms,
                                                   capture_ns={"native186": 100, "padded192": 101},
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
                self.assertEqual(result["completed_arms"], 8 if stopped else 32)
                self.assertEqual(result["actual_busy_card_samples"], 128 if stopped else 384)
                if stopped:
                    partial_arm = result["per_arm"][1]
                    self.assertEqual(partial_arm["measured_partial_cards"], 8)
                    self.assertEqual(partial_arm["useful_rows"], 8*17*64*186)

    def test_inspect_stop_reads_and_binds_one_original_numbered_step(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            node, parent, step = "f12r3n04", "1087656", "1087656.841"
            label, request = "frozen-shard", "1"*32
            common = dict(request=request, parent=parent, zone="I01", nodes=[node], label=label)
            argv = ["srun", "--job-name="+label]
            owner = dict(pid=123, start_ticks=9, uid=1, argv=argv)
            run = root / "results" / controller.RUN
            attempt = run / "attempts" / label
            worker = run / "nodes" / node
            attempt.mkdir(parents=True)
            worker.mkdir(parents=True)
            (attempt / "owner.json").write_text(json.dumps(dict(common, **owner)), encoding="utf8")
            (attempt / "intent.json").write_text(json.dumps(dict(common, attempted=True, argv=argv)), encoding="utf8")
            pcis = [f"0000:{index:02d}:00.0" for index in range(1,9)]
            (worker / "intent.json").write_text(json.dumps(dict(request=REQUEST, node=node, step=step,
                                                                 grid=self.windows, selected_PCI=pcis)), encoding="utf8")
            plan = dict(self.plan, shards={label: common}, selected={node: {"PCI": pcis}})
            accounting = [step, label, "RUNNING", "0:0", "60", node, "cpu=12,node=1,gres/dcu=8,mem=64G"]
            live = dict(StepId=step, NodeList=node)
            scheduler = SimpleNamespace(allocation=lambda job, nodes: set(nodes), hosts=lambda expression: [expression],
                                        query=lambda *args: [live], account=lambda labels: {label: accounting})
            invoke = partial(controller.inspect_stop, plan, label, request, step, {node: (parent, "I01")},
                             scheduler, Path.read_bytes, lambda entry: entry)
            with patch.object(controller, "ROOT", root), patch.object(controller, "proc_identity", return_value=owner):
                result = invoke()
                self.assertEqual(result["signal_argv"], ["scancel", "--signal=USR1", step])
                self.assertTrue(result["read_only"] and result["independent_terminal_and_post_required"])
                accounting[2] = "COMPLETED"
                with self.assertRaisesRegex(ValueError, "owned running shard"):
                    invoke()


if __name__ == "__main__":
    unittest.main()
