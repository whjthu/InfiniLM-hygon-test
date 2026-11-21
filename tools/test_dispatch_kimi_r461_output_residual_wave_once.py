"""Local qualification of the Kimi zero-GRES direct reservation adapter."""

from concurrent.futures import ThreadPoolExecutor
import json
import subprocess
from threading import Barrier, Lock
from types import SimpleNamespace
import unittest

import dispatch_kimi_r461_output_residual_wave_once as launcher
from prepare_training80_dispatch_handoff import build_handoff, claim_handoff
from report_training80_backlog import read_original_holds, read_queue_holds
import test_report_training80_backlog as fixtures
from tuning_queue import Queue, digest


class KimiOriginalScoreDispatchTest(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.BacklogTest()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.repo = fixture.repo.resolve()
        self.members = launcher.frozen_inputs(launcher.REPO)
        self.approved = "a" * 40
        for name in self.members:
            target = self.repo / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((launcher.REPO / name).read_bytes())
        self.board_path = self.repo / launcher.BACKLOG.relative_to(launcher.REPO)
        self.job = fixture.trial(1, "operator", node=launcher.NODE)
        candidate = dict(launcher.backlog_identity(), priority=923, next_action="fixture original request")
        self.board = dict(minimum_ready_training=5, items=[candidate, self.job])
        self.write_board()
        self.database = self.repo / launcher.QUEUE_DB.relative_to(launcher.REPO)
        queue = Queue(self.database)
        queue.db.close()
        self.intent = self.repo / launcher.INTENT

    def write_board(self):
        self.board_path.write_bytes(json.dumps(self.board).encode())

    def check_publication(self, repo, publication, members):
        self.assertEqual((repo, publication), (self.repo, self.approved))
        self.assertEqual(set(members), self.members | {self.board_path.relative_to(repo).as_posix()})

    def dispatch(self, transport, *, check=None, checkout=None):
        return launcher.dispatch(self.approved, str(launcher.PROJECT) if checkout is None else checkout, transport,
                                 repo=self.repo, backlog=self.board_path, database=self.database,
                                 check_publication=self.check_publication if check is None else check)

    def never_transport(self, *_):
        self.fail("unsafe transport invocation")

    def test_original_hold_committed_before_callback_even_for_existing_gpu_handoff(self):
        snapshot = self.fixture.verified_snapshot()
        local_clock = dict(now=self.fixture.now, boot_anchor=self.fixture.anchor)
        [scheduled] = build_handoff(snapshot, self.board, repo=self.repo, **local_clock)["coordinator_handoffs"]

        def transport(argv, stdout, stderr, limit):
            self.assertEqual(argv[-2:], ("--claim-sha256", launcher.CLAIM["SHA256"]))
            self.assertEqual(limit, 320)
            original = json.loads(self.intent.read_bytes())
            self.assertEqual(original["argv_SHA256"], digest(argv))
            self.assertEqual(original["target_intent"], str(launcher.TARGET / "intent.json"))
            self.assertEqual(original["placement"], launcher.PLACEMENT)
            [held] = read_queue_holds(self.database)
            self.assertEqual(held["state"], "DISPATCHING")
            self.assertEqual(held["direct_request"], launcher.REQUEST)
            self.assertEqual(held["nodes"], [launcher.NODE])
            competitor = Queue(self.database)
            try:
                with self.assertRaisesRegex(ValueError, "no current device-admitted"):
                    claim_handoff(competitor, snapshot, self.board, scheduled, self.job["launch"]["attempt_receipts"][0],
                                  repo=self.repo, **local_clock)
            finally:
                competitor.db.close()
            stdout.write(b"fixture successful transport with unverified terminal")
            return SimpleNamespace(returncode=0)

        result = self.dispatch(transport)
        self.assertEqual((result["remote_outcome"], result["transport_returncode"]), ("UNKNOWN", 0))
        self.assertFalse(result["terminal_verified"])
        self.assertEqual(read_queue_holds(self.database)[0]["state"], "UNKNOWN")

    def test_loss_preserves_both_markers_without_replay(self):
        def timeout(*_):
            raise subprocess.TimeoutExpired("recorded", 1)

        outcome = self.dispatch(timeout)
        self.assertEqual(outcome["transport_error"], "TimeoutExpired")
        self.assertEqual(json.loads(self.intent.with_name("transport.json").read_bytes()), outcome)
        self.assertEqual(read_queue_holds(self.database)[0]["nodes"], [launcher.NODE])
        with self.assertRaisesRegex(ValueError, "already claimed"):
            self.dispatch(self.never_transport)

    def test_parallel_same_scope_only_one_callback(self):
        gate, guard = Barrier(2), Lock()
        calls = []

        def verify(*args):
            self.check_publication(*args)
            gate.wait(timeout=5)

        def dispatch():
            def callback(*_):
                with guard:
                    calls.append("only")
                return SimpleNamespace(returncode=255)

            try:
                return self.dispatch(callback, check=verify)["remote_outcome"]
            except ValueError as exc:
                self.assertIn("already claimed", str(exc))
                return "REFUSED"

        with ThreadPoolExecutor(max_workers=2) as workers:
            a, b = workers.submit(dispatch), workers.submit(dispatch)
            self.assertEqual({a.result(timeout=10), b.result(timeout=10)}, {"UNKNOWN", "REFUSED"})
        self.assertEqual(calls, ["only"])

    def test_crash_survivor_marker_enters_normal_planner_and_blocks_dispatch(self):
        self.intent.parent.mkdir()
        self.intent.write_bytes(b"")
        holds, _, _ = read_original_holds(self.repo, self.board["items"], {launcher.NODE})
        self.assertEqual(holds[launcher.IDENT], dict(nodes=[launcher.NODE], state="reconcile"))
        self.assertEqual(read_queue_holds(self.database), [])
        with self.assertRaisesRegex(ValueError, "already claimed"):
            self.dispatch(self.never_transport)

    def test_legacy_hold_and_sqlite_hold_both_block(self):
        self.board["items"].append(dict(id="older-authorization", state="reconcile", placement={launcher.NODE: 1}))
        self.write_board()
        with self.assertRaisesRegex(ValueError, "another original request"):
            self.dispatch(self.never_transport)
        self.board["items"].pop()
        self.write_board()
        queue = Queue(self.database)
        queue.db.execute("INSERT INTO candidates VALUES (?,?)", ("prior", "{}"))
        queue.db.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?)",
                         ("prior", "prior", "model", "UNKNOWN", "{}", json.dumps([launcher.NODE]), None, None, 1))
        queue.db.close()
        with self.assertRaisesRegex(ValueError, "another original request"):
            self.dispatch(self.never_transport)
        self.assertFalse(self.intent.exists())

    def test_missing_shared_marker_is_refused(self):
        del self.board["items"][0]["attempt_receipts"]
        self.write_board()
        with self.assertRaisesRegex(ValueError, "register the exact"):
            self.dispatch(self.never_transport)
        self.assertFalse(self.intent.exists())

    def test_stale_published_board_or_ordinary_publication_error_cannot_claim(self):
        def changed(*args):
            self.check_publication(*args)
            self.board["items"][0]["next_action"] = "changed while comparing publication"
            self.write_board()

        def unpublished(*_):
            raise ValueError("published tip changed")

        with self.assertRaisesRegex(ValueError, "backlog changed"):
            self.dispatch(self.never_transport, check=changed)
        with self.assertRaisesRegex(ValueError, "published tip changed"):
            self.dispatch(self.never_transport, check=unpublished)
        self.assertEqual(read_queue_holds(self.database), [])

    def test_invalid_target_and_source_drift_cannot_claim(self):
        with self.assertRaisesRegex(ValueError, "canonical target"):
            self.dispatch(self.never_transport, checkout=str(launcher.PROJECT / "../other"))
        (self.repo / launcher.WORKER / "compile_once.py").write_bytes(b"drift")
        with self.assertRaisesRegex(ValueError, "source member bytes differ"):
            self.dispatch(self.never_transport)
        self.assertFalse(self.intent.exists())

    def test_separate_database_cannot_bypass_shared_node_reservations(self):
        alternate = self.repo / "isolated.sqlite3"
        Queue(alternate).db.close()
        with self.assertRaisesRegex(ValueError, "original shared planning database"):
            launcher.dispatch(self.approved, str(launcher.PROJECT), self.never_transport, repo=self.repo,
                              backlog=self.board_path, database=alternate, check_publication=self.check_publication)
        self.assertFalse(self.intent.exists())


if __name__ == "__main__":
    unittest.main()
