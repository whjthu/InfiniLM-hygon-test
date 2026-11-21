"""Qualify the Deep inverse RoPE direct reservation using local fixtures."""

from concurrent.futures import ThreadPoolExecutor
import json
import subprocess
import sys
from threading import Barrier, Lock
from types import SimpleNamespace
import unittest

import dispatch_deep_inverse_rope_compile_once as launcher
from prepare_training80_dispatch_handoff import build_handoff, claim_handoff
from report_training80_backlog import read_queue_holds
import test_report_training80_backlog as fixtures
from tuning_queue import Queue, digest


class DeepInverseRoPEDispatchTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.BacklogTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.repo = self.fixture.repo.resolve()
        self.members = launcher.scope(launcher.REPO)[1]
        for name in self.members:
            target = self.repo / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((launcher.REPO / name).read_bytes())
        self.board_path = self.repo / launcher.BACKLOG.relative_to(launcher.REPO)
        self.job = self.fixture.trial(1, "operator", node=launcher.NODE)
        row = dict(launcher.backlog_identity(), priority=924, next_action="fixture original compile")
        self.board = dict(minimum_ready_training=5, items=[row, self.job])
        self.save_board()
        self.database = self.repo / launcher.QUEUE_DB.relative_to(launcher.REPO)
        Queue(self.database).db.close()
        self.intent = self.repo / launcher.INTENT
        self.publication = "a" * 40

    def save_board(self):
        self.board_path.write_bytes(json.dumps(self.board).encode())

    def check_publication(self, repo, publication, members):
        self.assertEqual((repo, publication), (self.repo, self.publication))
        self.assertEqual(set(members), self.members | {self.board_path.relative_to(repo).as_posix()})

    def invoke(self, transport, *, check=None, checkout=None):
        return launcher.dispatch(self.publication, str(launcher.PROJECT) if checkout is None else checkout,
                                 transport, repo=self.repo, backlog=self.board_path, database=self.database,
                                 check_publication=self.check_publication if check is None else check)

    def forbid(self, *_):
        self.fail("unsafe transport")

    def test_shared_hold_blocks_an_existing_gpu_handoff_before_transport(self):
        snap = self.fixture.verified_snapshot()
        clock = dict(now=self.fixture.now, boot_anchor=self.fixture.anchor)
        [prior] = build_handoff(snap, self.board, repo=self.repo, **clock)["coordinator_handoffs"]

        def transport(argv, stdout, _, limit):
            self.assertEqual(limit, 320)
            local = json.loads(self.intent.read_bytes())
            self.assertEqual(local["argv_SHA256"], digest(argv))
            self.assertEqual(local["target_intent"], str(launcher.TARGET / "intent.json"))
            self.assertEqual(local["placement"], launcher.PLACEMENT)
            self.assertEqual(local["terminal_reader_manifest_SHA256"], launcher.AUDIT["SHA256"])
            [hold] = read_queue_holds(self.database)
            self.assertEqual((hold["state"], hold["direct_request"], hold["nodes"]),
                             ("DISPATCHING", launcher.REQUEST, [launcher.NODE]))
            queue = Queue(self.database)
            try:
                with self.assertRaisesRegex(ValueError, "no current device-admitted"):
                    claim_handoff(queue, snap, self.board, prior, self.job["launch"]["attempt_receipts"][0],
                                  repo=self.repo, **clock)
            finally:
                queue.db.close()
            stdout.write(b"recorded success-like controller output")
            return SimpleNamespace(returncode=0)

        result = self.invoke(transport)
        self.assertEqual((result["transport_returncode"], result["remote_outcome"]), (0, "UNKNOWN"))
        self.assertFalse(result["terminal_verified"])
        self.assertEqual(read_queue_holds(self.database)[0]["state"], "UNKNOWN")

    def test_timeout_and_duplicate_never_invoke_original_twice(self):
        def timeout(*_):
            raise subprocess.TimeoutExpired("recorded", 1)

        result = self.invoke(timeout)
        self.assertEqual(result["transport_error"], "TimeoutExpired")
        self.assertEqual(json.loads(self.intent.with_name("transport.json").read_bytes()), result)
        with self.assertRaisesRegex(ValueError, "already has a local scope"):
            self.invoke(self.forbid)

    def test_simultaneous_workers_only_one_finite_callback(self):
        barrier, lock = Barrier(2), Lock()
        hits = []

        def publication(*args):
            self.check_publication(*args)
            barrier.wait(timeout=5)

        def request():
            def transport(*_):
                with lock:
                    hits.append(1)
                return SimpleNamespace(returncode=255)

            try:
                return self.invoke(transport, check=publication)["remote_outcome"]
            except ValueError as error:
                self.assertIn("already has a local scope", str(error))
                return "REFUSED"

        with ThreadPoolExecutor(max_workers=2) as pool:
            a, b = pool.submit(request), pool.submit(request)
            self.assertEqual({a.result(timeout=10), b.result(timeout=10)}, {"UNKNOWN", "REFUSED"})
        self.assertEqual(hits, [1])

    def test_missing_backlog_marker_and_legacy_original_scope_refused(self):
        del self.board["items"][0]["attempt_receipts"]
        self.save_board()
        with self.assertRaisesRegex(ValueError, "register the exact"):
            self.invoke(self.forbid)
        self.board["items"][0]["attempt_receipts"] = [launcher.INTENT]
        self.save_board()
        (self.repo / "local-only" / (launcher.LABEL + "-controller")).mkdir()
        with self.assertRaisesRegex(ValueError, "already has a local scope"):
            self.invoke(self.forbid)
        self.assertFalse(self.intent.exists())

    def test_other_existing_scope_and_alternate_database_refused(self):
        self.board["items"].append(dict(id="prior-node-scope", state="reconcile", placement={launcher.NODE: 1}))
        self.save_board()
        with self.assertRaisesRegex(ValueError, "another original request"):
            self.invoke(self.forbid)
        alternate = self.repo / "separate.sqlite3"
        Queue(alternate).db.close()
        with self.assertRaisesRegex(ValueError, "shared planning database"):
            launcher.dispatch(self.publication, str(launcher.PROJECT), self.forbid, repo=self.repo,
                              backlog=self.board_path, database=alternate, check_publication=self.check_publication)
        self.assertFalse(self.intent.exists())

    def test_unrelated_checkpoint_and_frozen_worker_drift_refused(self):
        with self.assertRaisesRegex(ValueError, "canonical target"):
            self.invoke(self.forbid, checkout=str(launcher.PROJECT / "../other"))
        (self.repo / launcher.MANIFEST["path"]).with_name("compile_once.py").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "source member bytes differ"):
            self.invoke(self.forbid)
        self.assertFalse(self.intent.exists())

    def test_published_board_change_refused(self):
        def change(*args):
            self.check_publication(*args)
            self.board["items"][0]["next_action"] = "concurrent edit"
            self.save_board()

        with self.assertRaisesRegex(ValueError, "backlog changed"):
            self.invoke(self.forbid, check=change)
        self.assertEqual(read_queue_holds(self.database), [])

    def test_isolated_frozen_controller_import_and_target_sibling_pin(self):
        cmd = launcher.original_argv(str(launcher.PROJECT))
        bootstrap = cmd[6]
        self.assertIn("runpy.run_path(str(entry),run_name='__main__')", bootstrap)
        help_only = bootstrap.replace("'--claim-sha256',sys.argv[5]", "'--help'")
        argv = [sys.executable, *cmd[1:6], help_only, str(self.repo / launcher.HOST), str(self.repo / launcher.MEMBER),
                *cmd[9:]]
        completed = subprocess.run(argv, cwd=self.repo, capture_output=True, timeout=10, check=True)
        self.assertIn(b"--claim-sha256", completed.stdout)
        (self.repo / launcher.MEMBER).write_bytes(b"raise RuntimeError('unreviewed helper executed')\n")
        failure = subprocess.run(argv, cwd=self.repo, capture_output=True, timeout=10, check=False)
        self.assertNotEqual(failure.returncode, 0)
        self.assertIn(b"frozen Deep controller or membership helper differs", failure.stderr)
        self.assertNotIn(b"unreviewed helper executed", failure.stderr)


if __name__ == "__main__":
    unittest.main()
