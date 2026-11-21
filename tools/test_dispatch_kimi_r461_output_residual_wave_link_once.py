from hashlib import sha256
import json
from pathlib import Path
import runpy
import subprocess
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
import unittest

import dispatch_kimi_r461_output_residual_wave_link_once as link
import test_report_training80_backlog as fixture
from tuning_queue import Queue


class LinkDispatchTest(unittest.TestCase):
    def setUp(self):
        self.fleet = fixture.BacklogTest()
        self.fleet.setUp()
        self.addCleanup(self.fleet.doCleanups)
        self.fleet.now = time.monotonic()
        self.fleet.anchor = time.time() - self.fleet.now
        self.repo = self.fleet.repo.resolve()
        self.task, self.members = link.frozen_inputs(link.REPO)
        for name in self.members:
            target = self.repo / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((link.REPO / name).read_bytes())
        self.backlog = self.repo / link.BACKLOG.relative_to(link.REPO)
        self.database = self.repo / link.QUEUE_DB.relative_to(link.REPO)
        self.backlog.parent.mkdir(parents=True, exist_ok=True)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        Queue(self.database).db.close()
        self.row = dict(link.backlog_identity(self.task), priority=900, next_action="fixture")
        self.backlog.write_text(json.dumps(dict(minimum_ready_training=5, items=[self.row])), encoding="utf-8")

    def local_dispatch(self, snapshot, transport):
        def check_publication(repo, revision, members):
            self.assertEqual((repo, revision), (self.repo, "a"*40))
            self.assertEqual(set(members), self.members | {self.backlog.relative_to(self.repo).as_posix()})

        return link.dispatch("a"*40, str(link.PROJECT / "approved-checkout"), snapshot, transport,
                             repo=self.repo, backlog=self.backlog, database=self.database,
                             check_publication=check_publication)

    def test_unknown_transport_stays_reserved_and_cannot_repeat(self):
        calls = []

        def transport(argv, stdout, stderr, cap):
            calls.append(argv)
            queue = Queue(self.database)
            try:
                state = queue.db.execute("SELECT state FROM runs").fetchone()[0]
            finally:
                queue.db.close()
            self.assertEqual(state, "DISPATCHING")
            original = json.loads((self.repo / link.INTENT).read_bytes())
            self.assertEqual(original["request"], link.REQUEST)
            self.assertEqual(original["remote_outcome"], "UNKNOWN")
            self.assertEqual(original["placement_by_parent"], {link.PARENT: {link.NODE: 0}})
            self.assertEqual(argv[-1], self.task["SHA256"])
            self.assertEqual(cap, 320)
            raise subprocess.TimeoutExpired(argv, cap)

        snapshot = self.fleet.verified_snapshot()
        result = self.local_dispatch(snapshot, transport)
        self.assertEqual(result["transport_error"], "TimeoutExpired")
        queue = Queue(self.database)
        try:
            self.assertEqual(queue.dispatch_state(result["queue_run"])["state"], "UNKNOWN")
        finally:
            queue.db.close()
        with self.assertRaisesRegex(ValueError, "already claimed"):
            self.local_dispatch(snapshot, transport)
        self.assertEqual(len(calls), 1)

    def test_stale_or_kfd_occupied_cannot_claim(self):
        def no_transport(*args):
            self.fail("transport ran despite blocked admission")

        with self.assertRaisesRegex(ValueError, "fresh full-device-clear"):
            self.local_dispatch(self.fleet.verified_snapshot(age=100), no_transport)
        self.fleet.snapshot["nodes"][link.NODE]["KFD_clean"] = False
        with self.assertRaisesRegex(ValueError, "fresh full-device-clear"):
            self.local_dispatch(self.fleet.verified_snapshot(), no_transport)
        self.assertFalse((self.repo / link.INTENT).parent.exists())

    def test_other_admitted_work_protects_the_node(self):
        trial = self.fleet.trial(1, node=link.NODE)
        self.backlog.write_text(json.dumps(dict(minimum_ready_training=5, items=[self.row, trial])), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "fresh full-device-clear"):
            self.local_dispatch(self.fleet.verified_snapshot(), lambda *args: SimpleNamespace(returncode=0))
        self.assertFalse((self.repo / link.INTENT).parent.exists())


class HostLinkTest(unittest.TestCase):
    def test_frozen_score_and_exact_command_are_bound(self):
        host = runpy.run_path(str(link.REPO / link.ENTRY))
        claim = sha256((link.REPO / link.TASK).read_bytes()).hexdigest()
        _, original, _, terminal, groups, score, prefix = host["frozen"](link.REPO, claim)
        self.assertEqual(score["row"][:4], ["1103606.637", host["SCORE_LABEL"], "COMPLETED", "0:0"])
        self.assertIn(link.NODE, groups[link.PARENT][link.ZONE])
        with self.assertRaisesRegex(ValueError, "wrong original Kimi compile evidence"):
            host["score_accounting"](dict(state="UNKNOWN", row=score["row"]), original)
        command = host["build_command"](host["ROOT"], Path("/tmp/scope"), Path("/tmp/generated"), 8.0)
        for flag in ("--gres=none", "--cpus-per-task=4", "--mem=8192M", "--time=00:05:00"):
            self.assertIn(flag, command)
        self.assertEqual(command[-2:], ["8.0", "308.0"])
        script = host["inside_script"](prefix)
        self.assertTrue(script.startswith(prefix))
        self.assertIn(b"CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1", script)
        self.assertIn(link.SCORE_PIN.encode(), script)
        running = ["1103606.638", link.LABEL, "RUNNING", "0:0", "1", link.NODE, "cpu=4,mem=8G,node=1"]
        self.assertEqual(host["result"](running, 8.0, None, original, {}, terminal), {"state": "UNKNOWN"})

    def test_terminal_receipt_and_both_libraries_bind_the_actual_accounting(self):
        host = runpy.run_path(str(link.REPO / link.ENTRY))
        claim = sha256((link.REPO / link.TASK).read_bytes()).hexdigest()
        _, original, membership, terminal, _, _, _ = host["frozen"](link.REPO, claim)
        row = ["1103606.638", link.LABEL, "COMPLETED", "0:0", "1", link.NODE, "cpu=4,mem=8G,node=1"]
        with TemporaryDirectory() as work:
            output = Path(work).resolve()
            arms = {}
            for name in ("full-tree", "wave32"):
                data = name.encode()
                library = output / (name + ".so")
                library.write_bytes(data)
                arms[name] = dict(path="/refactor/candidates/" + link.LABEL + "/" + library.name,
                                 SHA256=sha256(data).hexdigest(), exports=["bound-cabi"], providers=["provider"])
            receipt = dict(step=row[0], node=link.NODE, epoch=8.0, deadline=308.0,
                           score_compile_step="1103606.637", score_receipt_SHA256=link.SCORE_PIN,
                           generated_manifest_SHA256=host["MANIFEST"], device_calls=0, weights_read=0,
                           completed=True, passed=True, arms=arms)
            (output / "CPU-result.json").write_text(json.dumps(receipt), encoding="utf-8")
            result = host["result"](row, 8.0, output, original, membership, terminal)
            self.assertEqual(result["state"], "LINK_VERIFIED")
            (output / "wave32.so").write_bytes(b"drift")
            result = host["result"](row, 8.0, output, original, membership, terminal)
            self.assertEqual(result["state"], "TERMINAL_UNVERIFIED")


if __name__ == "__main__":
    unittest.main()
