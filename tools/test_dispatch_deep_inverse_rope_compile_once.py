"""Qualify the Deep inverse RoPE direct reservation using local fixtures."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import subprocess
import sys
from threading import Barrier, Lock
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from collect_training80_dispatch_snapshot import receive
import dispatch_deep_inverse_rope_compile_once as launcher
from prepare_training80_dispatch_handoff import build_handoff, claim_handoff
from report_training80_backlog import read_queue_holds
import test_report_training80_backlog as fixtures
from tuning_queue import Queue, digest
import tuning_queue


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
        self.now = 1000.
        clock = patch.object(launcher, "time", SimpleNamespace(monotonic=lambda: self.now, time=lambda: self.now + 3000.))
        clock.start()
        self.addCleanup(clock.stop)
        self.snapshot = self.collector_receipt()

    def collector_receipt(self, change=lambda _: None, *, anchor=3000.):
        payload = deepcopy(self.fixture.snapshot)
        payload["nodes"][launcher.NODE].update(
            PCIs=[f"0000:{n:02x}:00.0" for n in range(1, 9)], kfd_process_count=0, own_device_fd_process_count=0,
            device_probe_step=f"{launcher.PARENT}.707", probe_error=None,
        )
        change(payload)
        with patch.object(tuning_queue, "time", SimpleNamespace(monotonic=lambda: 997., time=lambda: 997. + anchor)):
            return receive(json.dumps(payload).encode(), 996., payload["request_id"])

    def save_board(self):
        self.board_path.write_bytes(json.dumps(self.board).encode())

    def check_publication(self, repo, publication, members):
        self.assertEqual((repo, publication), (self.repo, self.publication))
        self.assertEqual(set(members), self.members | {self.board_path.relative_to(repo).as_posix()})

    def invoke(self, transport, *, check=None, checkout=None, snapshot=None):
        return launcher.dispatch(self.publication, str(launcher.PROJECT) if checkout is None else checkout,
                                 transport, snapshot=self.snapshot if snapshot is None else snapshot,
                                 repo=self.repo, backlog=self.board_path, database=self.database,
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
            self.assertEqual(local["device_admission"]["raw_response_SHA256"],
                             self.snapshot["local_receipt"]["raw_response_SHA256"])
            self.assertEqual(local["device_admission"]["device_probe_step"], f"{launcher.PARENT}.707")
            self.assertEqual(local["device_admission"]["PCIs"], self.snapshot["nodes"][launcher.NODE]["PCIs"])
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
        self.assertTrue(result["transport_attempted"])
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
                              backlog=self.board_path, database=alternate, check_publication=self.check_publication,
                              snapshot=self.snapshot)
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

    def test_device_receipt_required_and_mutated_payload_refused(self):
        with self.assertRaisesRegex(ValueError, "original received device snapshot"):
            self.invoke(self.forbid, snapshot=self.fixture.membership)
        self.snapshot["nodes"][launcher.NODE]["device_probe_step"] = f"{launcher.PARENT}.708"
        with self.assertRaisesRegex(ValueError, "collector identity or payload differs"):
            self.invoke(self.forbid)
        self.assertFalse(self.intent.exists())

    def test_foreign_parent_bad_pci_kfd_and_pid_refuse(self):
        scope = launcher.NODE
        for fields in (
            dict(device_probe_step="1087656.707"), dict(PCIs=[str(n) for n in range(8)]),
            dict(KFD_clean=False, kfd_process_count=1),
            dict(PID_clean=False, own_device_fd_process_count=1),
        ):
            with self.subTest(fields=fields):
                snapshot = self.collector_receipt(lambda value: value["nodes"][scope].update(fields))
                with self.assertRaisesRegex(ValueError, "same-parent PCI, KFD and PID"):
                    self.invoke(self.forbid, snapshot=snapshot)
        self.assertFalse(self.intent.exists())

    def test_reboot_anchor_and_long_publication_refused_before_claim(self):
        with self.assertRaisesRegex(ValueError, "expired or belongs to a different local clock"):
            self.invoke(self.forbid, snapshot=self.collector_receipt(anchor=2990.))

        def long_publication(*args):
            self.check_publication(*args)
            self.now = 1057.

        with self.assertRaisesRegex(ValueError, "device receipt expired"):
            self.invoke(self.forbid, check=long_publication)
        self.assertFalse(self.intent.exists())
        self.assertEqual(read_queue_holds(self.database), [])

    def test_receipt_expiring_under_the_transaction_does_not_leave_an_intent(self):
        shared_read = launcher.read_original_holds

        def delayed(*args):
            output = shared_read(*args)
            self.now = 1057.
            return output

        with patch.object(launcher, "read_original_holds", delayed):
            with self.assertRaisesRegex(ValueError, "device receipt expired"):
                self.invoke(self.forbid)
        self.assertFalse(self.intent.exists())
        self.assertEqual(read_queue_holds(self.database), [])

    def test_receipt_expiring_after_intent_commit_does_not_invoke_transport(self):
        original_write = launcher.write_once

        def delayed(path, value):
            original_write(path, value)
            if path == self.intent:
                self.now = 1057.

        with patch.object(launcher, "write_once", delayed):
            result = self.invoke(self.forbid)
        self.assertEqual(result["remote_outcome"], "NOT_ATTEMPTED")
        self.assertFalse(result["transport_attempted"])
        self.assertTrue(self.intent.is_file())
        self.assertEqual(read_queue_holds(self.database)[0]["state"], "DISPATCHING")


if __name__ == "__main__":
    unittest.main()
