import json
import time
import unittest
from unittest.mock import patch

import close_graph64_original_backfill as graph
from prepare_training80_dispatch_handoff import reserve_handoff
import test_report_training80_backlog as fixture
from tuning_queue import DIRECT_STAGE, Queue, digest


def encode(value):
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


class OriginalGraph64Test(unittest.TestCase):
    def setUp(self):
        self.fleet = fixture.BacklogTest()
        self.fleet.setUp()
        self.addCleanup(self.fleet.doCleanups)
        self.repo = self.fleet.repo.resolve()
        self.now = time.monotonic()
        self.anchor = time.time() - self.now
        scope = {"__name__": "fixture_scope"}
        path = graph.REPO / graph.PACKAGE / "scope.py"
        exec(compile(path.read_bytes(), str(path), "exec"), scope)
        owners = scope["ledger"](graph.TRAINING80_LEDGER.read_bytes())
        excluded = {"f11r2n15", "f11r4n09", "f11r4n10", "f11r4n19", "f12r2n12"}
        selected = {name: dict(parent=parent, zone=zone, before_CPU_step=f"{parent}.1",
                              PCI=[f"0000:{number:02x}:00.0" for number in range(1, 9)])
                    for name, (parent, zone) in owners.items() if name not in excluded}
        self.plan = dict(request=graph.REQUEST, publication=graph.PUBLICATION, campaign_epoch=100.0,
                         shared_start_epoch=160.0, deadline_epoch=225.0, pilot_SHA256=scope["PILOT_SHA"],
                         source_SHA256="f"*64, before_snapshot_SHA256="b"*64, selected=selected,
                         shards=scope["placement"](selected, owners))
        self.labels = list(self.plan["shards"])
        self.assertEqual(len(self.labels), 31)
        self.queue = Queue(self.repo / "fixture.sqlite")
        self.addCleanup(self.queue.db.close)
        self.handles = {}
        for label, unit in self.plan["shards"].items():
            parent = unit["parent"]
            packet = dict(candidate=label, case=graph.CASE, kind="operator", request=unit["request"],
                          placement_by_parent={parent: dict.fromkeys(unit["nodes"], 8)}, publication_commit=graph.PUBLICATION,
                          workload_SHA256=graph.CLAIM_SHA, campaign_request=graph.REQUEST,
                          snapshot_payload_SHA256="b"*64, snapshot_raw_SHA256="c"*64, snapshot_request="d"*32,
                          expires_local_monotonic=100.0, launcher=(graph.PACKAGE / "controller.py").as_posix(),
                          invocation={"kind": "pinned_finite_code"}, replay_allowed=False)
            packet["handoff_SHA256"] = digest(packet)
            run = digest(dict(stage=DIRECT_STAGE, request=unit["request"]))
            path = graph.DIRECTORY / "shards" / (label + ".json")
            handle = reserve_handoff(self.queue, run, packet, path.as_posix())["handle"]
            self.queue.submission(run, handle, unknown=True)
            self.handles[label] = dict(run=run, handle=handle, packet=packet)
        self.markers = {label: encode(value) for label, value in self.handles.items()}
        self.verdict = dict(request=graph.REQUEST, replay_allowed=False, shards={}, nodes={}, **graph.DENOMINATOR)
        self.post = self.fleet.snapshot
        self.post["request_id"] = "e"*32
        self.post["probes"] = dict.fromkeys(("1087656", "1103606"))
        for parent in self.post["probes"]:
            self.post["probes"][parent] = {"batches": []}
        for probe in self.post["nodes"].values():
            probe.update(device_probe_observed=False, KFD_clean=False, PID_clean=False, device_probe_step=None,
                         probe_error="DEFERRED_BY_PRIORITY_ONLY", kfd_process_count=None,
                         own_device_fd_process_count=None)

    def terminal(self, label, ordinal, *, state="COMPLETED", post="clean"):
        unit = self.plan["shards"][label]
        parent, nodes = unit["parent"], unit["nodes"]
        step, count = f"{parent}.{300 + ordinal}", len(nodes)
        exit_code = "0:0" if state == "COMPLETED" else "1:0"
        row = [step, label, state, exit_code, "60", ",".join(nodes),
               f"cpu={12*count},mem={64*count}G,node={count},gres/dcu={8*count}"]
        passed = state == "COMPLETED"
        self.verdict["shards"][label] = dict(unit, state=state, terminal=True, device_qualified=passed,
                                            independent_post_required=True, planned_cards=8*count, attempted=True,
                                            attempt_SHA256="1"*64, resources_match=True, accounting=row)
        for index, name in enumerate(nodes):
            if passed:
                self.verdict["nodes"][name] = dict(passed=True, original_step=step, valid_cards=8,
                                                   completed_arms=32, actual_busy_card_samples=128)
            if post == "missing":
                continue
            busy = post == "busy" and index == 0
            probe = self.post["nodes"][name]
            probe.update(PCIs=self.plan["selected"][name]["PCI"], device_probe_step=f"{parent}.{900 + ordinal}",
                         device_probe_observed=True, KFD_clean=not busy, PID_clean=not busy, probe_error=None,
                         kfd_process_count=int(busy), own_device_fd_process_count=int(busy))
        if post != "missing":
            self.post["probes"][parent]["batches"].append(dict(
                requested_nodes=nodes, received_nodes=nodes, state="RETURNED", returncode=0,
                request_id=f"{ordinal:032x}", stdout_SHA256="2"*64))
        return row

    def invoke(self, *, apply=False, age=1, post_before_terminal=False, change_post_raw=False):
        originals = (self.plan, self.handles, self.markers, graph.sha(encode(self.handles)), graph.TRAINING80_LEDGER)
        reconcile_raw = encode(self.verdict)
        terminal_received = self.now - (0.1 if post_before_terminal else 100)
        capture = graph.terminal_receipt(reconcile_raw, monotonic=lambda: terminal_received,
                                         wall=lambda: self.anchor + terminal_received)
        post_raw = encode(self.post)
        receipt = dict(request_id=self.post["request_id"], received_monotonic=self.now-age,
                       roundtrip_s=0.5, boot_epoch_anchor=self.anchor, raw_response_SHA256=graph.sha(post_raw),
                       payload_SHA256=digest(self.post))
        snapshot = dict(self.post, local_receipt=receipt)
        if change_post_raw:
            post_raw += b"\n"
        return graph.close(self.queue.db, originals, reconcile_raw, encode(capture), encode(snapshot), post_raw,
                           now=self.now, boot_anchor=self.anchor, queue=self.queue if apply else None)

    def test_mixed_outcomes_close_only_independently_proven_shards(self):
        successful, failed, held, running, missed = self.labels[:5]
        self.terminal(successful, 1)
        self.terminal(failed, 2, state="FAILED")
        self.terminal(held, 3, post="busy")
        self.verdict["shards"][running] = {"state": "RUNNING", "terminal": False}
        dry = self.invoke()
        self.assertEqual(dry["ready_to_close_shards"], 2)
        self.assertEqual(self.queue.summary()["runs"], {"UNKNOWN": 31})
        actual = self.invoke(apply=True)
        self.assertEqual((actual["planned_shards"], actual["planned_cards"], actual["held_shards"]), (31, 600, 29))
        self.assertEqual(self.queue.summary()["runs"], {"TERMINAL": 2, "UNKNOWN": 29})
        self.assertEqual(actual["shards"][successful]["state"], "CLOSED")
        failed_result = actual["shards"][failed]["result"]
        self.assertFalse(failed_result["passed"])
        self.assertEqual(failed_result["failed"], failed_result["planned"])
        self.assertEqual(actual["shards"][held]["state"], "HOLD_POST_UNVERIFIED")
        self.assertEqual(actual["shards"][running]["state"], "HOLD_TERMINAL_UNKNOWN")
        self.assertEqual(actual["shards"][missed]["state"], "HOLD_TERMINAL_UNKNOWN")
        expected_valid = sum(actual["shards"][label]["planned_cards"] for label in (successful, held))
        self.assertEqual(actual["verified_worker_counters"]["valid_cards"], expected_valid)
        second = self.invoke(apply=True)
        self.assertEqual(second["shards"][successful]["state"], "ALREADY_CLOSED")
        self.assertEqual(second["closed_shards"], 2)

    def test_stale_preterminal_and_altered_post_keep_the_hold(self):
        label = self.labels[0]
        self.terminal(label, 1)
        for kwargs in (dict(age=90), dict(post_before_terminal=True), dict(change_post_raw=True)):
            with self.subTest(kwargs=kwargs):
                result = self.invoke(apply=True, **kwargs)
                self.assertEqual(result["shards"][label]["state"], "HOLD_POST_UNVERIFIED")
                self.assertEqual(result["closed_shards"], 0)
        self.assertEqual(self.queue.summary()["runs"], {"UNKNOWN": 31})

    def test_missing_marker_or_changed_terminal_identity_does_not_release(self):
        first, second, third = self.labels[:3]
        self.terminal(first, 1)
        self.markers.pop(first)
        self.terminal(second, 2)[1] = "other-label"
        self.terminal(third, 3, post="missing")
        result = self.invoke(apply=True)
        self.assertEqual(result["shards"][first]["state"], "HOLD_CLAIM_UNVERIFIED")
        self.assertEqual(result["shards"][second]["state"], "HOLD_TERMINAL_UNVERIFIED")
        self.assertEqual(result["shards"][third]["state"], "HOLD_POST_UNVERIFIED")
        self.assertEqual(result["closed_shards"], 0)

    def test_original_plan_bytes_and_per_shard_index(self):
        for name in ("scope.py", "controller.py"):
            target = self.repo / graph.PACKAGE / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((graph.REPO / graph.PACKAGE / name).read_bytes())
        ledger = self.repo / graph.TRAINING80_LEDGER.relative_to(graph.REPO)
        ledger.parent.mkdir(parents=True, exist_ok=True)
        ledger.write_bytes(graph.TRAINING80_LEDGER.read_bytes())
        directory = self.repo / graph.DIRECTORY
        (directory / "shards").mkdir(parents=True)
        raw = encode(self.plan)
        path = directory / "original-plan.json"
        path.write_bytes(raw)
        (directory / "original-shards.json").write_bytes(encode(self.handles))
        for label, data in self.markers.items():
            (directory / "shards" / (label + ".json")).write_bytes(data)
        with patch.object(graph, "PLAN_SHA", graph.sha(raw)):
            plan, handles, markers, _, _ = graph.original_inputs(self.repo)
            self.assertEqual((plan, handles, markers), (self.plan, self.handles, self.markers))
            path.write_bytes(raw + b"\n")
            with self.assertRaisesRegex(ValueError, "plan bytes differ"):
                graph.original_inputs(self.repo)

    def test_original_slurm_hostlist_expansion(self):
        self.assertEqual(graph.hosts("f13r1n[01-03,05],a14r3n04"),
                         {"f13r1n01", "f13r1n02", "f13r1n03", "f13r1n05", "a14r3n04"})
        for malformed in ("f13r1n[01,],a14r3n04", "f13r1n[01-03,03]", "f13r1n[03-01]", "f13r1n1"):
            with self.subTest(malformed=malformed), self.assertRaises(ValueError):
                graph.hosts(malformed)


if __name__ == "__main__":
    unittest.main()
