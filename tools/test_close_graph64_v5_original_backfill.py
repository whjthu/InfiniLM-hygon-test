import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import close_graph64_v5_original_backfill as v5
from prepare_training80_dispatch_handoff import reserve_handoff
import test_close_graph64_original_backfill as fixture
from tuning_queue import DIRECT_STAGE, digest


class OriginalV5Test(unittest.TestCase):
    def setUp(self):
        test = self.fixture = fixture.OriginalGraph64Test()
        test.setUp()
        self.addCleanup(test.doCleanups)
        self.source_files = v5.frozen_source(v5.REPO)
        scope = {"__name__": "fixture_v5_scope"}
        path = v5.REPO / v5.PACKAGE / "scope.py"
        v5.scope_loader(self.source_files["scope.py"], str(path), scope)
        ledger = scope["ledger"](v5.TRAINING80_LEDGER.read_bytes())
        selected = {key: value for key, value in test.plan["selected"].items() if key not in {"f11r4n17", "f13r1n01"}}
        shards = scope["placement"](selected, ledger)
        test.plan.update(request=v5.REQUEST, publication=v5.PUBLICATION, selected=selected, shards=shards,
                         campaign_epoch=v5.EPOCH, shared_start_epoch=v5.EPOCH+48, deadline_epoch=v5.EPOCH+455,
                         cohort_by_shard=scope["assign_cohorts"](shards), source_SHA256=v5.ENVELOPE_SHA)
        self.assertEqual((len(selected), len(shards)), (73, 31))
        test.labels, test.handles = list(shards), {}
        for label, unit in shards.items():
            packet = dict(candidate=label, case=v5.CASE, kind="operator", request=unit["request"],
                          placement_by_parent={unit["parent"]: dict.fromkeys(unit["nodes"], 8)},
                          publication_commit=v5.PUBLICATION, workload_SHA256=v5.CLAIM_SHA, campaign_request=v5.REQUEST,
                          launcher=(v5.PACKAGE / "controller.py").as_posix(), invocation={"kind": "pinned_finite_code"},
                          snapshot_payload_SHA256=test.plan["before_snapshot_SHA256"], snapshot_raw_SHA256="c"*64,
                          snapshot_request="d"*32, expires_local_monotonic=100., replay_allowed=False)
            packet["handoff_SHA256"] = digest(packet)
            run = digest(dict(stage=DIRECT_STAGE, request=unit["request"]))
            intent = v5.DIRECTORY / "shards" / (label + ".json")
            handle = reserve_handoff(test.queue, run, packet, intent.as_posix())["handle"]
            test.queue.submission(run, handle, unknown=True)
            test.handles[label] = dict(run=run, handle=handle, packet=packet)
        test.markers = {key: fixture.encode(value) for key, value in test.handles.items()}
        test.verdict.update(request=v5.REQUEST, campaign_epoch=v5.EPOCH, clock_inter_node_bounded=False,
                            common_window_inter_node_overlap_ns=None, **v5.DENOMINATOR)

    def terminal(self, label, ordinal, *, partial=False, valid_post=True):
        test = self.fixture
        row = test.terminal(label, ordinal, state="FAILED" if partial else "COMPLETED",
                            post="clean" if valid_post else "busy")
        row[4] = "180" if partial else "445"
        for node in test.plan["shards"][label]["nodes"]:
            arms = []
            for index, mode in enumerate(v5.ORDER):
                if partial:
                    complete, stopped, samples = [(8, 0, 96), (0, 8, 24), (0, 0, 0), (0, 0, 0), (0, 0, 0)][index]
                else:
                    complete, stopped, samples = 8, 0, 96
                count = complete+stopped
                groups = 4*count
                gemms = groups*(32 if mode == "graph32" else 64)
                arms.append(dict(mode=mode, planned_cards=8, complete_cards=complete, measured_partial_cards=stopped,
                                 measured_card_samples=samples, groups=groups, gemm_calls=gemms,
                                 useful_rows=186*gemms,
                                 complete_groups_by_quarter=[count]*4))
            test.verdict["nodes"][node] = dict(passed=not partial, original_step=row[0], valid_cards=0 if partial else 8,
                                              completed_arms=8 if partial else 40,
                                              actual_busy_card_samples=120 if partial else 480, per_arm=arms,
                                              common_window_epoch_ns=None if partial else [
                                                  round((v5.EPOCH+t)*v5.NANO) for t in (375,435)],
                                              per_card_graph32_over_graph64_pairs={str(i): [.95, 1.05] for i in range(8)},
                                              cooperative_stopped=partial,
                                              raw_SHA256={"intent.json": "1"*64, "terminal.json": "2"*64})
        return row

    def invoke(self, **kwargs):
        with patch.object(fixture, "graph", SimpleNamespace(**v5._checker)):
            return self.fixture.invoke(**kwargs)

    def test_real_caps_and_cooperative_partial_only_close_individually_proven(self):
        full, partial, blocked = self.fixture.labels[:3]
        self.terminal(full, 1)
        self.terminal(partial, 2, partial=True)
        self.terminal(blocked, 3, valid_post=False)
        report = self.invoke(apply=True)
        self.assertEqual((report["planned_shards"], report["planned_nodes"], report["planned_cards"],
                          report["planned_arms"], report["planned_busy_card_samples"]), (31, 73, 584, 2920, 35040))
        self.assertEqual((report["closed_shards"], report["held_shards"]), (2, 29))
        self.assertEqual(report["shards"][blocked]["state"], "HOLD_POST_UNVERIFIED")
        stopped = report["shards"][partial]["result"]
        nodes = self.fixture.plan["shards"][partial]["nodes"]
        self.assertFalse(stopped["passed"])
        self.assertEqual(stopped["cooperative_stop_nodes"], nodes)
        self.assertEqual(stopped["incomplete_worker_nodes"], nodes)
        self.assertEqual(report["paired_comparison"]["planned_pairs"], 1168)
        self.assertGreater(report["paired_comparison"]["measured_pairs"], 0)
        self.assertIsNone(report["common_window_inter_node_overlap_ns"])
        self.assertFalse(report["common_clock_inter_node_bounded"])
        self.assertEqual(stopped["per_arm_work"][1]["measured_partial_cards"], 8*len(nodes))
        self.assertEqual(stopped["per_arm_work"][1]["useful_rows"], 4*len(nodes)*8*32*186)
        self.assertEqual(self.invoke(apply=True)["shards"][full]["state"], "ALREADY_CLOSED")
        self.assertEqual(self.fixture.queue.summary()["runs"], {"TERMINAL": 2, "UNKNOWN": 60})

    def test_unauthorized_gpu_shape_wrong_post_order_and_bad_worker_stats_do_not_close(self):
        first, second = self.fixture.labels[:2]
        row = self.terminal(first, 1)
        row[6] = row[6].rsplit(",", 1)[0] + ",gres/dcu=4"
        self.terminal(second, 2, partial=True)
        self.assertEqual(self.invoke(apply=True, post_before_terminal=True)["closed_shards"], 0)
        node = self.fixture.plan["shards"][second]["nodes"][0]
        self.fixture.verdict["nodes"][node]["per_arm"][1]["gemm_calls"] *= 2
        report = self.invoke(apply=True)
        self.assertEqual(report["shards"][first]["state"], "HOLD_TERMINAL_UNVERIFIED")
        self.assertEqual(report["shards"][second]["state"], "HOLD_TERMINAL_UNVERIFIED")
        self.assertEqual(self.fixture.queue.summary()["runs"], {"UNKNOWN": 62})

    def test_bad_common_window_and_unverified_clock_do_not_release(self):
        first, second = self.fixture.labels[:2]
        self.terminal(first, 1)
        self.fixture.verdict["shards"][second] = {"state": "RUNNING", "terminal": False}
        node = self.fixture.plan["shards"][first]["nodes"][0]
        original_endpoint = self.fixture.verdict["nodes"][node]["common_window_epoch_ns"][1]
        self.fixture.verdict["nodes"][node]["common_window_epoch_ns"][1] -= 2*v5.NANO
        receipt = self.invoke(apply=True)
        self.assertEqual(receipt["shards"][first]["state"], "HOLD_TERMINAL_UNVERIFIED")
        self.assertEqual(receipt["shards"][second]["state"], "HOLD_TERMINAL_UNKNOWN")
        self.fixture.verdict["nodes"][node]["common_window_epoch_ns"][1] = original_endpoint
        self.fixture.verdict["clock_inter_node_bounded"] = True
        with self.assertRaisesRegex(ValueError, "unbounded clock"):
            self.invoke(apply=True)
        self.assertEqual(self.fixture.queue.summary()["runs"], {"UNKNOWN": 62})

    def test_plan_pin_and_actual_source_envelope_are_required(self):
        manifest = v5.PACKAGE / "SOURCE_FILES.json"
        refs = json.loads(self.source_files["SOURCE_FILES.json"])["files"]
        paths = [manifest, v5.PACKAGE / "ROOT_CLAIM.json", *(Path(spec["path"]) for spec in refs.values())]
        for relative in paths:
            dst = self.fixture.repo / relative
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes((v5.REPO / relative).read_bytes())
        directory = self.fixture.repo / v5.DIRECTORY
        (directory / "shards").mkdir(parents=True)
        plan = directory / "original-plan.json"
        raw = fixture.encode(self.fixture.plan)
        plan.write_bytes(raw)
        (directory / "original-shards.json").write_bytes(fixture.encode(self.fixture.handles))
        for label, data in self.fixture.markers.items():
            (directory / "shards" / (label + ".json")).write_bytes(data)
        with self.assertRaisesRegex(ValueError, "plan bytes differ"):
            v5.original_inputs(self.fixture.repo)
        with patch.dict(v5._checker, {"PLAN_SHA": v5.sha(raw)}):
            self.assertEqual(v5.original_inputs(self.fixture.repo)[0], self.fixture.plan)
        broken = fixture.encode(dict(self.fixture.plan, source_SHA256="f"*64))
        plan.write_bytes(broken)
        with patch.dict(v5._checker, {"PLAN_SHA": v5.sha(broken)}):
            with self.assertRaisesRegex(ValueError, "deployed envelope"):
                v5.original_inputs(self.fixture.repo)


if __name__ == "__main__":
    unittest.main()
