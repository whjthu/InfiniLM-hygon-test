"""Exercise the exact derived audit against synthetic procfs and driver inventory."""

import errno
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import derive
import fd_permission


def load(name, path):
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class HolderDiagnostic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.work.cleanup)
        work = Path(cls.work.name)
        source = derive.HERE.with_name("zimage-o01-eight-card-memory48-launch-v1") / "render.py"
        renderer = load("fd_diagnostic_render_baseline", source)
        baseline = renderer.baseline()
        renderer.render_runtime(baseline)
        cls.overlay = derive.overlay(baseline)
        for name, raw in (baseline | cls.overlay).items():
            if name.startswith("runtime/"):
                path = work / name
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(raw)
        sys.path.insert(0, str(work / "runtime"))
        cls.addClassCleanup(sys.path.remove, str(work / "runtime"))
        cls.owner = load("linux_owner", work / "runtime/linux_owner.py")
        cls.supervisor = load("fd_diagnostic_test_supervisor", work / "runtime/supervisor.py")

    def setUp(self):
        work = tempfile.TemporaryDirectory()
        self.addCleanup(work.cleanup)
        root = Path(work.name)
        proc, kfd = root / "proc", root / "kfd"
        proc.mkdir()
        kfd.mkdir()
        self.proc, self.kfd = proc, kfd
        self.step = "200.7"
        self.pcis = {"0000:09:00.0", "0000:36:00.0"}
        self.worker = "0000:36:00.0"
        entries = []
        for index, pci in enumerate(sorted(self.pcis)):
            entry = MagicMock()
            entry.name = f"renderD{128 + index}"
            entry.__truediv__.return_value.resolve.return_value.name = pci
            entries.append(entry)
        render = MagicMock()
        render.iterdir.side_effect = lambda: iter(entries)
        self.children = {}
        self.rows = {}
        self.targets = {}
        for pid, group in ((100, "/slurm/job_200/step_7"), (101, "/slurm/job_200/step_7/task_0")):
            self.make_process(pid, group)
        (kfd / "101").mkdir()
        child = self.rows["101"]
        self.children[self.worker] = SimpleNamespace(identity=child)
        self.add_fd(101, 1, "/dev/kfd")
        self.add_fd(101, 2, "/dev/dri/renderD129")
        changes = {
            "DRM": render, "PROC": proc, "KFD": kfd,
            "identity": lambda path: self.rows.get(path.name),
            "audit_no_other_children": MagicMock(),
        }
        self.apply(patch.multiple(self.owner, **changes))
        self.apply(patch.object(self.owner.os, "getuid", return_value=proc.stat().st_uid, create=True))
        self.apply(patch.object(self.owner.os, "getpid", return_value=100))
        self.readlink = self.apply(patch.object(self.owner.os, "readlink", side_effect=self.targets.__getitem__))

    def apply(self, fixture):
        applied = fixture.start()
        self.addCleanup(fixture.stop)
        return applied

    def make_process(self, pid, group):
        path = self.proc / str(pid)
        (path / "fd").mkdir(parents=True)
        (path / "cgroup").write_bytes(f"0::{group}\n".encode())
        self.rows[str(pid)] = self.owner.Identity(pid, "S", 100, pid, pid, 999 + pid)

    def add_fd(self, pid, fd, value):
        path = self.proc / str(pid) / "fd" / str(fd)
        path.touch()
        self.targets[path] = value

    def audit(self):
        return self.owner.audit_device_holders(self.children, self.pcis, step=self.step)

    def deny_directory(self, pid):
        original = Path.iterdir

        def list_path(path):
            if path == self.proc / str(pid) / "fd":
                raise PermissionError(errno.EACCES, "PRIVATE", str(path))
            yield from original(path)

        return patch.object(Path, "iterdir", list_path)

    def denied(self, operation, relationship, in_kfd, code):
        with self.assertRaises(fd_permission.HolderAccessDenied) as caught:
            self.audit()
        self.assertEqual(caught.exception.diagnostic, {
            "operation": operation, "errno": code,
            "process_relationship": relationship, "process_in_KFD_listing": in_kfd,
            "holder_audit_complete": False, "target_clean": False,
        })
        self.assertNotIn("PRIVATE", str(caught.exception))

    def test_valid_scan_preserves_full_device_result(self):
        result = self.audit()
        self.assertEqual(result, {"sibling_driver_PID_to_PCI": {"101": self.worker},
                                  "KFD_PIDs": ["101"], "DRM_PCIs": sorted(self.pcis)})

    def test_distinguishes_enumeration_and_readlink_for_bound_worker(self):
        with self.deny_directory(101):
            self.denied("fd_enumeration", "owned_surviving_standby", True, errno.EACCES)

        def readlink(path):
            if path.name == "2":
                raise PermissionError(errno.EPERM, "PRIVATE", str(path))
            return self.targets[path]

        self.readlink.side_effect = readlink
        self.denied("fd_readlink", "owned_surviving_standby", True, errno.EPERM)

    def test_same_numbered_step_and_external_same_uid_are_both_denials(self):
        self.make_process(102, "/slurm/job_200/step_7/task_1")
        with self.deny_directory(102):
            self.denied("fd_enumeration", "same_numbered_step_unowned", False, errno.EACCES)
            (self.proc / "102" / "cgroup").write_bytes(b"0::/slurm/job_200/step_70\n")
            self.denied("fd_enumeration", "same_uid_outside_numbered_step", False, errno.EACCES)
            (self.proc / "102" / "cgroup").write_bytes(b"expected unavailable")
            self.denied("fd_enumeration", "same_uid_step_unresolved", False, errno.EACCES)

    def test_reused_worker_is_never_counted_as_owned(self):
        row = self.rows["101"]
        self.rows["101"] = self.owner.Identity(row.pid, row.state, row.parent, row.group, row.session, 99221)
        with self.deny_directory(101):
            self.denied("fd_enumeration", "process_identity_unresolved", True, errno.EACCES)

    def test_original_holder_rejections_preserved(self):
        (self.kfd / "102").mkdir()
        with self.assertRaisesRegex(ValueError, "UNOWNED_KFD_HOLDER"):
            self.audit()
        (self.kfd / "102").rmdir()
        self.add_fd(101, 2, "/dev/dri/renderD128")
        with self.assertRaisesRegex(ValueError, "UNOWNED_OR_CROSS_PCI_RENDER_FD"):
            self.audit()
        self.make_process(102, "/unrelated")
        self.add_fd(102, 2, "/dev/kfd")
        self.add_fd(101, 2, "/dev/dri/renderD129")
        with self.assertRaisesRegex(ValueError, "UNOWNED_PROJECT_KFD_FD"):
            self.audit()

    def test_production_caller_blocks_both_handoffs_and_preserves_diagnostic(self):
        for denied_phase in ("pre_witness", "pre_restore"):
            with self.subTest(denied_phase=denied_phase):
                report = {}
                fake = SimpleNamespace(step=self.step, results=report)
                fake.phase = MagicMock()
                fake.inventory = MagicMock()
                fake.start_standbys = MagicMock(return_value={
                    self.worker: self.children[self.worker], self.supervisor.TARGET: object(),
                })
                fake.growth = MagicMock()
                fake.stop_target = MagicMock()
                fake.witness = MagicMock()
                fake.restore = MagicMock()
                fake.hold = MagicMock()
                blocker = fd_permission.HolderAccessDenied(
                    PermissionError(errno.EACCES, "PRIVATE"), "fd_enumeration",
                    "same_uid_outside_numbered_step", False)
                checks = [blocker] if denied_phase == "pre_witness" else [{}, blocker]

                def audit(siblings, boundary):
                    return self.supervisor.Step.clean_target(fake, siblings, boundary)

                fake.clean_target = audit
                with patch.object(self.owner, "audit_device_holders", side_effect=checks):
                    with self.assertRaises(fd_permission.HolderAccessDenied):
                        self.supervisor.cycle(fake, report)
                self.assertEqual(report["holder_audit_denial"]["boundary"], denied_phase)
                self.assertIs(report["holder_audit_denial"]["target_clean"], False)
                self.assertEqual(fake.witness.call_count, int(denied_phase == "pre_restore"))
                fake.restore.assert_not_called()

    def test_source_pin_refuses_changed_baseline(self):
        with self.assertRaisesRegex(ValueError, "frozen runtime bytes differ"):
            derive.overlay({"runtime/linux_owner.py": b"x", "runtime/supervisor.py": b"y"})

    def test_host_whitelists_only_scope_bound_negative_diagnostic(self):
        gate = fd_permission.HolderAccessDenied(
            PermissionError(errno.EACCES, "PRIVATE"), "fd_readlink", "same_uid_outside_numbered_step", False)
        request, scope_sha, runtime_sha = "a" * 32, "b" * 64, "c" * 64
        terminal = {"request": request, "step": self.step, "scope_SHA256": scope_sha, "runtime_SHA256": runtime_sha,
                    "error": {"type": "HolderAccessDenied"}, "restored_eight_proved": False,
                    "holder_audit_denial": {"boundary": "pre_witness", **gate.diagnostic, "test_private_path": "junk"}}
        args = dict(request=request, step=self.step, scope_sha=scope_sha, runtime_sha=runtime_sha)
        verified = fd_permission.terminal_diagnostic(terminal, **args)
        self.assertEqual(verified, {"boundary": "pre_witness", **gate.diagnostic})
        with self.assertRaisesRegex(ValueError, "FD_AUDIT_TERMINAL_IDENTITY"):
            fd_permission.terminal_diagnostic(terminal, **(args | {"step": "200.70"}))
        with self.assertRaisesRegex(ValueError, "FD_AUDIT_TERMINAL_DIAGNOSTIC"):
            fd_permission.terminal_diagnostic(terminal | {"witness": {}}, **args)


if __name__ == "__main__":
    unittest.main()
