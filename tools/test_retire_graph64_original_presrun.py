from contextlib import ExitStack
import json
import unittest
from unittest.mock import patch

import retire_graph64_original_presrun as retire
import test_close_graph64_original_backfill as fixture
from tuning_queue import digest


class OriginalPreexecRetireTest(unittest.TestCase):
    def setUp(self):
        self.scope = fixture.OriginalGraph64Test()
        self.scope.setUp()
        self.addCleanup(self.scope.doCleanups)
        self.graph = fixture.graph
        self.queue = self.scope.queue
        self.first, self.second = self.scope.labels[:2]
        self.scope.terminal(self.first, 1)
        self.scope.terminal(self.second, 2, post="busy")
        self.absence = (json.dumps(dict(epoch=retire.OBSERVED_EPOCH, stage_plan=False, result_intent=False,
                                        stage_dir=False, result_dir=False, steps=[])) + "\n").encode()
        self.transport = fixture.encode(dict(request=self.graph.REQUEST, planned_nodes=75, planned_cards=600,
                                              planned_shards=31, transport_attempted=True, remote_outcome="UNKNOWN",
                                              transport_returncode=255, terminal_verified=False, replay_allowed=False))
        self.stderr = b"Connection timed out during banner exchange\n"
        self.failure = (self.absence, self.transport, self.stderr, b"")
        self.assertEqual(self.graph.sha(self.absence), retire.ABSENCE_SHA)
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(retire, "TRANSPORT_SHA", self.graph.sha(self.transport)))
        stack.enter_context(patch.object(retire, "STDERR_SHA", self.graph.sha(self.stderr)))
        epoch = retire.OBSERVED_EPOCH + 20
        self.scope.post["observed_epoch"] = self.scope.post["membership"]["observed_epoch"] = epoch

    def invoke(self, *, apply=False, age=2, failure=None, raw_suffix=b""):
        original = (self.scope.plan, self.scope.handles, self.scope.markers,
                    self.graph.sha(fixture.encode(self.scope.handles)), self.graph.TRAINING80_LEDGER)
        raw = fixture.encode(self.scope.post)
        receipt = dict(request_id=self.scope.post["request_id"], received_monotonic=self.scope.now-age,
                       roundtrip_s=0.5, boot_epoch_anchor=self.scope.anchor, raw_response_SHA256=self.graph.sha(raw),
                       payload_SHA256=digest(self.scope.post))
        snapshot = fixture.encode(dict(self.scope.post, local_receipt=receipt))
        return retire.retire(self.queue.db, original, self.failure if failure is None else failure, snapshot,
                             raw + raw_suffix, now=self.scope.now, anchor=self.scope.anchor,
                             queue=self.queue if apply else None)

    def test_one_shard_retires_once_and_blocked_shards_keep_holds(self):
        dry = self.invoke()
        self.assertEqual(dry["ready_to_retire_shards"], 1)
        self.assertEqual(self.queue.summary()["runs"], {"UNKNOWN": 31})
        applied = self.invoke(apply=True)
        self.assertEqual((applied["planned_shards"], applied["planned_cards"]), (31, 600))
        self.assertEqual((applied["retired_shards"], applied["held_shards"], applied["actual_graph_GPU_steps"]), (1, 30, 0))
        row = applied["shards"][self.first]
        self.assertEqual(row["state"], "RETIRED_PRELAUNCH")
        self.assertEqual(applied["shards"][self.second]["state"], "HOLD_CLEARANCE_UNVERIFIED")
        state, nodes, raw = self.queue.db.execute("SELECT state,nodes,result FROM runs WHERE id=?", (row["run"],)).fetchone()
        saved = json.loads(raw)
        self.assertEqual((state, nodes), ("RETIRED_PRELAUNCH", "[]"))
        self.assertTrue(saved["transport_attempted"])
        self.assertFalse(saved["measurement_terminal"])
        repeated = self.invoke(apply=True, age=300)
        self.assertEqual(repeated["shards"][self.first]["state"], "ALREADY_RETIRED")
        self.assertEqual(self.queue.summary()["runs"], {"RETIRED_PRELAUNCH": 1, "UNKNOWN": 30})
        self.assertEqual(self.queue.db.execute("SELECT count(*) FROM events WHERE kind=?", (retire.EVENT,)).fetchone()[0], 1)

    def test_stale_or_tampered_clearance_keeps_all_shards(self):
        for inputs in (dict(age=300), dict(raw_suffix=b"\n")):
            with self.subTest(inputs=inputs):
                summary = self.invoke(apply=True, **inputs)
                self.assertEqual((summary["retired_shards"], summary["held_shards"]), (0, 31))
        self.scope.post["observed_epoch"] = self.scope.post["membership"]["observed_epoch"] = retire.OBSERVED_EPOCH - 1
        self.assertEqual(self.invoke(apply=True)["retired_shards"], 0)
        self.assertEqual(self.queue.summary()["runs"], {"UNKNOWN": 31})

    def test_altered_initial_evidence_or_unexpired_campaign_cannot_retire(self):
        for failure in ((self.absence + b"\n", self.transport, self.stderr, b""),
                        (self.absence, self.transport, self.stderr, b"unexpected"),
                        (self.absence, self.transport, b"different SSH error", b"")):
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                self.invoke(apply=True, failure=failure)
        self.scope.plan["campaign_epoch"] = retire.OBSERVED_EPOCH - 180
        with self.assertRaisesRegex(ValueError, "after controller expiry"):
            self.invoke(apply=True)
        self.assertEqual(self.queue.summary()["runs"], {"UNKNOWN": 31})

    def test_changed_marker_or_unexpected_queue_state_keeps_shard(self):
        self.scope.terminal(self.scope.labels[2], 3)
        self.scope.markers[self.first] = b"{}"
        run = self.scope.handles[self.scope.labels[2]]["run"]
        self.queue.db.execute('UPDATE runs SET state="SUBMITTED" WHERE id=?', (run,))
        summary = self.invoke(apply=True)
        self.assertEqual((summary["retired_shards"], summary["held_shards"]), (0, 31))
        self.assertEqual(summary["shards"][self.first]["state"], "HOLD_ORIGINAL_UNVERIFIED")
        self.assertEqual(summary["shards"][self.scope.labels[2]]["state"], "HOLD_ORIGINAL_UNVERIFIED")


if __name__ == "__main__":
    unittest.main()
