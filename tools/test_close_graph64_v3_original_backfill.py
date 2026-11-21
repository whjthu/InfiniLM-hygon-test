from pathlib import Path
from types import SimpleNamespace
import json
import unittest
from unittest.mock import patch

import close_graph64_v3_original_backfill as v3
from prepare_training80_dispatch_handoff import reserve_handoff
import test_close_graph64_original_backfill as fixture
from tuning_queue import DIRECT_STAGE, digest


class Graph64V3OriginalTest(unittest.TestCase):
    def setUp(self):
        harness = self.harness = fixture.OriginalGraph64Test()
        harness.setUp()
        self.addCleanup(harness.doCleanups)
        path = v3.REPO / v3.PACKAGE / "scope.py"
        scope = {"__name__": "fixture_v3_scope"}
        exec(compile(path.read_bytes(), str(path), "exec"), scope)
        ledger = v3._checker["TRAINING80_LEDGER"]
        owners = scope["ledger"](ledger.read_bytes())
        selected = {node: harness.plan["selected"][node] for node in sorted(scope["REQUIRED_NODES"])}
        shards = scope["placement"](selected, owners)
        harness.plan.update(request=v3.REQUEST, publication=v3.PUBLICATION, selected=selected, shards=shards,
                            source_SHA256=v3.frozen_source(v3.REPO))
        harness.labels = list(shards)
        harness.handles = {}
        for label, unit in shards.items():
            packet = dict(candidate=label, case=v3.CASE, kind="operator", request=unit["request"],
                          placement_by_parent={unit["parent"]: dict.fromkeys(unit["nodes"], 8)},
                          publication_commit=v3.PUBLICATION, workload_SHA256=v3.CLAIM_SHA, campaign_request=v3.REQUEST,
                          snapshot_payload_SHA256=harness.plan["before_snapshot_SHA256"], snapshot_raw_SHA256="c"*64,
                          snapshot_request="d"*32, expires_local_monotonic=100.0,
                          launcher=(v3.PACKAGE / "controller.py").as_posix(), invocation={"kind": "pinned_finite_code"},
                          replay_allowed=False)
            packet["handoff_SHA256"] = digest(packet)
            run = digest(dict(stage=DIRECT_STAGE, request=unit["request"]))
            intent = v3.DIRECTORY / "shards" / (label + ".json")
            handle = reserve_handoff(harness.queue, run, packet, intent.as_posix())["handle"]
            harness.queue.submission(run, handle, unknown=True)
            harness.handles[label] = dict(run=run, handle=handle, packet=packet)
        harness.markers = {label: fixture.encode(value) for label, value in harness.handles.items()}
        harness.verdict.update(request=v3.REQUEST, **v3.DENOMINATOR)

    def terminal_with_setup(self, label, ordinal, **kwargs):
        row = self.harness.terminal(label, ordinal, **kwargs)
        for node in self.harness.plan["shards"][label]["nodes"]:
            if node in self.harness.verdict["nodes"]:
                self.harness.verdict["nodes"][node]["setup_phase"] = dict(
                    request=v3.REQUEST, node=node, step=row[0], all_released=True, raw_SHA256="1"*64)

    def invoke(self, apply=False):
        with patch.object(fixture, "graph", SimpleNamespace(**v3._checker)):
            return self.harness.invoke(apply=apply)

    def test_v3_actual_denominator_closes_only_its_proven_pair(self):
        successful, failed, held = self.harness.labels
        self.terminal_with_setup(successful, 1)
        self.harness.terminal(failed, 2, state="FAILED")
        self.terminal_with_setup(held, 3, post="busy")
        report = self.invoke(apply=True)
        self.assertEqual((report["planned_shards"], report["planned_nodes"], report["planned_cards"],
                          report["planned_arms"], report["planned_busy_card_samples"]), (3, 6, 48, 192, 768))
        self.assertEqual(report["closed_shards"], 2)
        self.assertEqual(report["shards"][held]["state"], "HOLD_POST_UNVERIFIED")
        self.assertFalse(report["shards"][failed]["result"]["passed"])
        evidence = report["shards"][successful]["result"]["evidence"]
        self.assertEqual((evidence["campaign_request"], evidence["local_plan_SHA256"], evidence["controller_SHA256"]),
                         (v3.REQUEST, v3.PLAN_SHA, v3.CONTROLLER_SHA))
        self.assertEqual(self.harness.queue.summary()["runs"], {"TERMINAL": 2, "UNKNOWN": 32})
        self.assertEqual(self.invoke(apply=True)["shards"][successful]["state"], "ALREADY_CLOSED")

    def test_wrong_campaign_denominator_and_setup_step_rejected(self):
        with self.assertRaises(ValueError):
            v3.terminal_receipt(fixture.encode(dict(request=fixture.graph.REQUEST, replay_allowed=False)))
        label = self.harness.labels[0]
        self.terminal_with_setup(label, 1)
        self.harness.verdict["planned_busy_card_samples"] = 9600
        with self.assertRaisesRegex(ValueError, "v3 three-shard result"):
            self.invoke()
        self.harness.verdict.update(v3.DENOMINATOR)
        node = self.harness.plan["shards"][label]["nodes"][0]
        self.harness.verdict["nodes"][node]["setup_phase"]["step"] = "1103606.999"
        report = self.invoke(apply=True)
        self.assertEqual(report["shards"][label]["state"], "HOLD_TERMINAL_UNVERIFIED")
        self.assertEqual(self.harness.queue.summary()["runs"], {"UNKNOWN": 34})

    def test_pinned_plan_and_source_envelope_reject_change(self):
        root, repo = v3.REPO, self.harness.repo
        manifest = v3.PACKAGE / "SOURCE_FILES.json"
        refs = json.loads((root / manifest).read_bytes())["files"]
        paths = [manifest, v3.PACKAGE / "ROOT_CLAIM.json", *(Path(spec["path"]) for spec in refs.values())]
        for relative in paths:
            target = repo / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((root / relative).read_bytes())
        directory = repo / v3.DIRECTORY
        (directory / "shards").mkdir(parents=True)
        plan_file = directory / "original-plan.json"
        (directory / "original-shards.json").write_bytes(fixture.encode(self.harness.handles))
        for label, raw in self.harness.markers.items():
            (directory / "shards" / (label + ".json")).write_bytes(raw)
        correct = fixture.encode(self.harness.plan)
        plan_file.write_bytes(correct)
        with self.assertRaisesRegex(ValueError, "plan bytes differ"):
            v3.original_inputs(repo)
        with patch.dict(v3._checker, {"PLAN_SHA": v3.sha(correct)}):
            self.assertEqual(v3.original_inputs(repo)[0], self.harness.plan)
            plan_file.write_bytes(correct + b"\n")
            with self.assertRaisesRegex(ValueError, "plan bytes differ"):
                v3.original_inputs(repo)
        altered_source = fixture.encode(dict(self.harness.plan, source_SHA256="f"*64))
        plan_file.write_bytes(altered_source)
        with patch.dict(v3._checker, {"PLAN_SHA": v3.sha(altered_source)}):
            with self.assertRaisesRegex(ValueError, "original source envelope"):
                v3.original_inputs(repo)


if __name__ == "__main__":
    unittest.main()
