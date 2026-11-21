"""Exercise the paired compile's reservation, transport and target receipt checks."""

from copy import deepcopy
from hashlib import sha256
import json
import subprocess
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from collect_training80_dispatch_snapshot import receive
import control_deep_rope_prefix_width_pair as host
import dispatch_deep_rope_prefix_width_pair_once as dispatcher
from report_training80_backlog import read_queue_holds
import test_report_training80_backlog as fleet_fixtures
import test_run_deep_rope_prefix_width_compile_pair as compile_fixtures
import tuning_queue
from tuning_queue import Queue


class PairDispatchTest(TestCase):
    def setUp(self):
        fixture = fleet_fixtures.BacklogTest()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.repo = fixture.repo.resolve()
        _, self.members = dispatcher.scope(dispatcher.REPO)
        for name in self.members:
            dest = self.repo / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes((dispatcher.REPO / name).read_bytes())
        self.database = self.repo / dispatcher.QUEUE_DB.relative_to(dispatcher.REPO)
        Queue(self.database).db.close()
        self.board = self.repo / dispatcher.BACKLOG.relative_to(dispatcher.REPO)
        candidate = dict(dispatcher.backlog_identity(), priority=925, next_action="fixture")
        self.board.write_text(json.dumps(dict(minimum_ready_training=5, items=[candidate])))
        payload = deepcopy(fixture.snapshot)
        payload["nodes"][dispatcher.NODE].update(
            PCIs=[f"0000:{index:02x}:00.0" for index in range(1, 9)], kfd_process_count=0,
            own_device_fd_process_count=0, device_probe_step=f"{dispatcher.PARENT}.703", probe_error=None,
        )
        with patch.object(tuning_queue, "time", SimpleNamespace(monotonic=lambda: 997., time=lambda: 3997.)):
            self.snapshot = receive(json.dumps(payload).encode(), 996., payload["request_id"])

    def dispatch(self, transport):
        def published(repo, commit, files):
            self.assertEqual((repo, commit), (self.repo, "a" * 40))
            self.assertEqual(set(files), self.members | {self.board.relative_to(repo).as_posix()})

        clock = SimpleNamespace(monotonic=lambda: 1000., time=lambda: 4000.)
        with patch.dict(dispatcher._FROZEN, {"time": clock}):
            return dispatcher.dispatch("a" * 40, str(dispatcher.PROJECT), transport, snapshot=self.snapshot,
                                       repo=self.repo, backlog=self.board, database=self.database, check_publication=published)

    def test_timeout_preserves_request_and_does_not_replay(self):
        attempts = []

        def callback(argv, *_):
            attempts.append(argv)
            original, = read_queue_holds(self.database)
            self.assertEqual(original["state"], "UNKNOWN")
            self.assertEqual(original["direct_request"], dispatcher.REQUEST)
            self.assertEqual(original["nodes"], [dispatcher.NODE])
            raise subprocess.TimeoutExpired(argv[:1], 1)

        result = self.dispatch(callback)
        self.assertEqual(result["transport_error"], "TimeoutExpired")
        with self.assertRaisesRegex(ValueError, "already has a local scope"):
            self.dispatch(callback)
        self.assertEqual(len(attempts), 1)
        intent = json.loads((self.repo / dispatcher.INTENT).read_bytes())
        self.assertEqual(intent["target_intent"], str(dispatcher.TARGET / "intent.json"))
        self.assertIn(dispatcher.WORKER_SHA, intent["argv"])

    def test_argv_refuses_checkout_outside_existing_project(self):
        with self.assertRaisesRegex(ValueError, "canonical target checkout"):
            dispatcher.original_argv("/tmp/elsewhere")
        args = dispatcher.original_argv(str(dispatcher.PROJECT / "published"))
        self.assertEqual(args[-1], dispatcher.CLAIM["SHA256"])
        read_only = dispatcher.original_argv(str(dispatcher.PROJECT / "published"), reconcile=True)
        self.assertEqual(read_only[-2:], (dispatcher.CLAIM["SHA256"], "--reconcile-existing"))


class PairReceiptTest(TestCase):
    def setUp(self):
        fixture = compile_fixtures.CompilePairTest()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.node = dispatcher.NODE
        self.outputs, self.epoch, self.step = fixture.outputs, fixture.epoch, fixture.step
        self.result = fixture.execute()
        self.root = self.outputs / dispatcher.PAIR
        self.row = [self.step, dispatcher.PAIR, "COMPLETED", "0:0", "20", dispatcher.NODE,
                    "cpu=4,mem=8192M,node=1"]
        for arm, evidence in zip(host.VARIANTS, self.result["variant_results"], strict=True):
            path = self.outputs / arm.label / "CPU-result.json"
            worker = json.loads(path.read_bytes())
            worker["object"] = f"/refactor/candidates/{arm.label}/inverse_output_rope.o"
            raw = json.dumps(worker).encode()
            path.write_bytes(raw)
            evidence["worker_receipt_SHA256"] = sha256(raw).hexdigest()
        (self.root / "CPU-pair-result.json").write_text(json.dumps(self.result))

    def test_requires_exact_parent_resources_and_both_frozen_objects(self):
        result = host.check_result(self.root, self.row, self.epoch)
        self.assertEqual(set(result["object_SHA256"]), {"pack32", "pack64"})
        foreign = self.row.copy()
        foreign[0] = "32768.700"
        with self.assertRaisesRegex(ValueError, "accounting"):
            host.check_result(self.root, foreign, self.epoch)
        malformed = self.row.copy()
        malformed[-1] += ",gres/dcu=1"
        with self.assertRaisesRegex(ValueError, "accounting"):
            host.check_result(self.root, malformed, self.epoch)

    def test_changed_native_object_remains_unverified(self):
        (self.outputs / host.VARIANTS[1].label / "inverse_output_rope.o").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "SHA differs"):
            host.check_result(self.root, self.row, self.epoch)

    def test_read_only_reconciliation_keeps_running_and_checks_terminal(self):
        run = self.outputs / "controller"
        run.mkdir()
        intent = dict(request=host.REQUEST, parent=host.PARENT, node=host.NODE,
                      claim_SHA256=dispatcher.CLAIM["SHA256"], source_manifest_SHA256=host.WORKLOAD,
                      replay_allowed=False, epoch=self.epoch, deadline=self.epoch+300)
        (run / "intent.json").write_text(json.dumps(intent))
        running = self.row.copy()
        running[2] = "RUNNING"

        def observed(row):
            return host.reconcile(run, self.root, dispatcher.CLAIM["SHA256"],
                                  lambda _: row, lambda original: original[2] == "COMPLETED")

        self.assertEqual(observed(running)["state"], "UNKNOWN")
        self.assertEqual(observed(self.row)["state"], "COMPILE_VERIFIED")
        self.assertEqual({path.name for path in run.iterdir()}, {"intent.json"})
