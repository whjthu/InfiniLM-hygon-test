"""Local fixtures for zero-GRES exact-scope and procfs visibility behavior."""

import errno
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe


class Visibility(unittest.TestCase):
    def make_rows(self, observer_tres="cpu=1,mem=256M,node=1"):
        observer = "1103606.3210"
        return observer, [
            {"StepId": probe.TARGET, "Name": probe.TARGET_LABEL, "NodeList": probe.NODE,
             "State": "RUNNING", "TRES": "cpu=8,gres/dcu=8,mem=16G,node=1"},
            {"StepId": observer, "Name": probe.LABEL, "NodeList": probe.NODE,
             "State": "RUNNING", "TRES": observer_tres},
            {"StepId": "1103606.3211", "Name": "unrelated", "NodeList": "f12r2n13",
             "State": "RUNNING", "TRES": "cpu=2,mem=2G,node=1"},
        ]

    def test_exact_live_resources_and_wrong_or_duplicate_observer(self):
        step, rows = self.make_rows()
        with patch.object(probe, "rows", return_value=rows):
            probe.live_bindings(step)
        for pattern in ("cpu=1,mem=256M,node=1,gres/dcu=1", "cpu=2,mem=256M,node=1"):
            _, altered = self.make_rows(pattern)
            with patch.object(probe, "rows", return_value=altered):
                with self.assertRaisesRegex(ValueError, "ACTUAL_STEP_BINDING"):
                    probe.live_bindings(step)
        with patch.object(probe, "rows", return_value=rows + rows[1:2]):
            with self.assertRaisesRegex(ValueError, "NUMBERED_STEP_NOT_UNIQUE"):
                probe.live_bindings(step)

    def test_enumeration_and_link_denial_and_only_whitelisted_device_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            process = Path(temporary)
            fd = process / "fd"
            fd.mkdir()
            for i in range(3):
                (fd / str(i)).touch()
            render = {"renderD128": "0000:09:00.0"}
            targets = {"0": "/dev/kfd", "1": "/dev/dri/renderD128", "2": "/private/unrelated"}
            with patch.object(probe.os, "readlink", side_effect=lambda p: targets[p.name]) as readlink:
                result = probe.fd_visibility(process, render)
                self.assertEqual((result["worker_KFD_FDs"], result["mapped_DRM_PCI_count"]), (1, 1))
                self.assertEqual(readlink.call_count, 3)
                self.assertNotIn("/private", str(result))
            original = Path.iterdir

            def denied(path):
                if path == fd:
                    raise PermissionError(errno.EACCES, "denied", str(fd))
                yield from original(path)

            with patch.object(Path, "iterdir", denied), patch.object(probe.os, "readlink") as readlink:
                result = probe.fd_visibility(process, render)
                self.assertEqual(result["fd_enumeration"], "permission_denied")
                self.assertEqual(result["fd_enumeration_errno"], errno.EACCES)
                readlink.assert_not_called()

            with patch.object(probe.os, "readlink", side_effect=PermissionError(errno.EPERM, "denied")):
                result = probe.fd_visibility(process, render)
                self.assertEqual(result["fd_enumeration"], "observed")
                self.assertEqual(result["fd_readlink_permission_denied"], 3)
                self.assertEqual(result["fd_readlink_errno"], [errno.EPERM])

    def test_cgroup_record_and_reused_worker(self):
        target_cgroup = b"6:memory:/slurm/job_1103606/step_2296/task_0\n"
        self.assertTrue(probe.names_step(target_cgroup))
        self.assertFalse(probe.names_step(target_cgroup.replace(b"step_2296", b"step_22960")))
        expected = {"worker_start_ticks": 123, "worker_cgroup_SHA256": probe.digest(target_cgroup),
                    "boot_id_SHA256": probe.digest(b"boot")}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            boot = root / "boot"
            boot.write_bytes(b"boot")
            (root / "cgroup").write_bytes(target_cgroup)

            def stat(ticks):
                (root / "stat").write_text("101 (with ) bracket) S " + " ".join(["0"] * 18 + [str(ticks)]))

            stat(123)
            with patch.object(probe, "BOOT", boot), patch.object(probe.os, "getuid", return_value=root.stat().st_uid, create=True):
                probe.identity(root, expected)
                stat(124)
                with self.assertRaisesRegex(ValueError, "WORKER_IDENTITY_CHANGED_OR_NOT_VISIBLE"):
                    probe.identity(root, expected)

    def test_render_job_and_container_remain_bounded(self):
        self.assertRegex(probe.REQUEST, r"^[a-f0-9]{32}$")
        source = probe.STAGE / "probe.py"
        argv = probe.srun_argv(source, "a" * 64)
        self.assertIn("--gres=none", argv)
        self.assertIn("--time=00:01:00", argv)
        self.assertIn("--mem=256M", argv)
        self.assertIn("--cpus-per-task=1", argv)
        self.assertNotIn("--gres=dcu", " ".join(argv))
        self.assertFalse(any("\\" in item for item in argv))
        inside = probe.container_argv(source, "a" * 64)
        self.assertIn("--rocm", inside)
        self.assertIn("--no-home", inside)
        self.assertIn("/usr/local/hyhal:/opt/hyhal:ro", inside)
        self.assertNotIn("env.sh", " ".join(inside))
        with self.assertRaisesRegex(ValueError, "SOURCE_OR_STAGE_DIFFERS"):
            probe.srun_argv(source.with_name("other.py"), "a" * 64)

    def test_host_checks_live_owner_before_and_after_and_whitelists(self):
        source, pin = probe.STAGE / "probe.py", "a" * 64
        view = {"namespace": [0, 71], "identity_bound": True, "devices": {"a": True}, "fd": {"b": True}}
        expected = {"worker_PID": 123, "worker_start_ticks": 124, "worker_cgroup_SHA256": "b" * 64,
                    "boot_id_SHA256": "c" * 64}
        command_result = SimpleNamespace(returncode=0, stdout=probe.encode(view), stderr=b"")
        with patch.object(Path, "samefile", return_value=True), patch.object(probe, "step_from_env", return_value="1103606.3210"), \
             patch.object(probe, "live_bindings") as live, patch.object(probe, "belongs_to", return_value=True) as membership, \
             patch.object(probe, "original", return_value=expected), patch.object(probe, "view", return_value=view), \
             patch.object(probe, "identity") as identity, patch.object(probe.subprocess, "run", return_value=command_result) as run:
            result = probe.host(source, pin)
            self.assertTrue(result["same_worker_identity"])
            self.assertFalse(result["target_clean"])
            self.assertTrue(result["host"]["identity_bound"])
            self.assertNotIn("namespace", result["host"])
            self.assertNotIn("worker_PID", str(result))
            self.assertNotIn("boot_id", str(result))
            self.assertEqual(live.call_count, 2)
            self.assertEqual(membership.call_count, 3)
            identity.assert_called_once()
            self.assertEqual(run.call_args.kwargs["timeout"], 20)
            self.assertEqual(run.call_args.kwargs["cwd"], "/")

        other = probe.public(view, view | {"namespace": [0, 72]})
        self.assertFalse(other["same_worker_identity"])
        self.assertFalse(other["holder_audit_complete"])

    def test_cli_runs_with_isolated_python(self):
        result = subprocess.run([sys.executable, "-I", "-S", "-B", probe.__file__, "--help"],
                                check=True, capture_output=True, text=True)
        self.assertIn("{host,inside}", result.stdout)


if __name__ == "__main__":
    unittest.main()
