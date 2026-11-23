"""Exercise the source renderer and handoff ordering off cluster."""

import ast
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import linux_owner
import prepare
import supervisor


class RecordingStep:
    def __init__(self, blocked=None, qualified=True):
        self.blocked, self.qualified = blocked, qualified
        self.calls = []
        self.clean_checks = 0
        self.initial = {pci: "first-" + pci for pci in supervisor.GPU_SET}

    def inventory(self):
        self.calls.append("inventory")

    def start_standbys(self):
        self.calls.append("start-eight")
        return dict(self.initial)

    def growth(self, cards, seconds, latest):
        phase = "seven" if set(cards) == supervisor.STANDBY_SET else (
            "restored" if cards[supervisor.TARGET] == "new-nine" else "initial"
        )
        self.calls.append(phase)
        return dict(cards)

    def stop_target(self, target):
        assert target == self.initial[supervisor.TARGET]
        self.calls.append("stop-nine")
        return True

    def clean_target(self, siblings):
        assert siblings == {pci: self.initial[pci] for pci in supervisor.STANDBY_SET}
        self.clean_checks += 1
        self.calls.append("clean-" + str(self.clean_checks))
        if self.blocked == self.clean_checks:
            raise ValueError("UNKNOWN_TARGET_HOLDER")
        return True

    def witness(self, siblings):
        assert set(siblings) == supervisor.STANDBY_SET
        self.calls.append("witness")
        return {"qualified_operator_route": self.qualified}

    def restore(self):
        self.calls.append("restart-nine")
        return "new-nine"

    def hold(self, all_cards):
        assert all_cards[supervisor.TARGET] == "new-nine" and len(all_cards) == 8
        self.calls.append("hold-eight")


class Checks(unittest.TestCase):
    def test_witness_tail_and_positive_refill(self):
        recording = RecordingStep()
        report = {}
        supervisor.cycle(recording, report)
        self.assertEqual(recording.calls, ["inventory", "start-eight", "initial", "stop-nine", "seven", "clean-1",
                                           "witness", "clean-2", "restart-nine", "restored", "hold-eight"])
        self.assertTrue(report["restored_eight_proved"])
        self.assertEqual(len(report["restored_eight_growth"]), 8)
        self.assertEqual(len(report["seven_growth_before_witness"]), 7)

    def test_failed_route_with_reconciled_owner_still_restores(self):
        report = {}
        supervisor.cycle(RecordingStep(qualified=False), report)
        self.assertFalse(report["witness"]["qualified_operator_route"])
        self.assertTrue(report["restored_eight_proved"])

    def test_unknown_holder_on_either_handoff_never_refills(self):
        for blocked in (1, 2):
            with self.subTest(blocked=blocked):
                report = {}
                recording = RecordingStep(blocked=blocked)
                with self.assertRaisesRegex(ValueError, "UNKNOWN_TARGET_HOLDER"):
                    supervisor.cycle(recording, report)
                self.assertNotIn("restart-nine", recording.calls)
                self.assertIs("witness" in recording.calls, blocked == 2)
                self.assertNotIn("restored_eight_proved", report)

    def test_full_runtime_and_unmodified_numerical_path(self):
        prior = ModuleType("prior_for_test")
        old = prepare.ORIGINAL / "prepare.py"
        prior.__file__ = str(old)
        exec(compile(prepare.exact(old, prepare.ORIGINAL_RENDERER_SHA), str(old), "exec"), vars(prior))
        before = prior.frozen_renderer().source_files()
        after = prepare.render()

        def function(source, name):
            return next(ast.dump(node) for node in ast.parse(source).body
                        if isinstance(node, ast.FunctionDef) and node.name == name)

        for name in ("native_probe", "fixed_inputs", "witness", "audit", "trace_size"):
            self.assertEqual(function(before["worker.py"], name), function(after["witness.py"], name), name)
        self.assertEqual(before["contract.py"], after["contract.py"])
        self.assertNotIn(b"start_new_session=True", after["witness.py"])
        self.assertIn(b"WITNESS_MASKED_HIP_COUNT", after["witness.py"])
        with tempfile.TemporaryDirectory() as temporary:
            stage = Path(temporary) / "stage"
            runtime = prepare.stage(stage)
            paths = sorted(stage.glob("*"))
            self.assertEqual({x.name for x in paths}, supervisor.RUNTIME_MEMBERS | {"RUNTIME_FILES.json"})
            with patch.object(supervisor, "HERE", stage):
                supervisor.verify_runtime(runtime["sha256"])
                (stage / "witness.py").write_bytes(b"x=3")
                with self.assertRaisesRegex(ValueError, "INPUT_SHA256"):
                    supervisor.verify_runtime(runtime["sha256"])
            result = subprocess.run([sys.executable, "-I", "-B", str(stage / "supervisor.py"), "--help"],
                                    capture_output=True, text=True, check=True)
            self.assertIn("--runtime-sha256", result.stdout)

    def test_driver_holders_exact_or_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proc, kfd = root / "proc", root / "kfd"
            proc.mkdir()
            kfd.mkdir()
            devices = {}
            fd_targets = {}
            render = MagicMock()
            render_entries = []
            for index, pci in enumerate(sorted(supervisor.GPU_SET)):
                entry = MagicMock()
                entry.name = f"renderD{128 + index}"
                entry.__truediv__.return_value.resolve.return_value.name = pci
                render_entries.append(entry)
                if pci == supervisor.TARGET:
                    continue
                pid = 200 + index
                fd = proc / str(pid) / "fd"
                fd.mkdir(parents=True)
                for name, link in (("1", "/dev/kfd"), ("2", "/dev/dri/" + entry.name)):
                    path = fd / name
                    path.touch()
                    fd_targets[path] = link
                (kfd / str(pid)).mkdir()
                devices[pci] = SimpleNamespace(identity=SimpleNamespace(pid=pid))
            render.iterdir.return_value = iter(render_entries)
            with patch.object(linux_owner, "DRM", render), patch.object(linux_owner, "PROC", proc), \
                 patch.object(linux_owner, "KFD", kfd), patch.object(linux_owner, "audit_no_other_children"), \
                 patch.object(linux_owner.os, "getuid", return_value=proc.stat().st_uid, create=True), \
                 patch.object(linux_owner.os, "readlink", side_effect=fd_targets.__getitem__):
                result = linux_owner.audit_device_holders(devices, supervisor.GPU_SET)
                self.assertEqual(len(result["sibling_driver_PID_to_PCI"]), 7)
                (kfd / "999").mkdir()
                render.iterdir.return_value = iter(render_entries)
                with self.assertRaisesRegex(ValueError, "UNOWNED_KFD_HOLDER"):
                    linux_owner.audit_device_holders(devices, supervisor.GPU_SET)
                (kfd / "999").rmdir()
                fd_targets[proc / "201" / "fd" / "2"] = "/dev/dri/renderD128"
                render.iterdir.return_value = iter(render_entries)
                with self.assertRaisesRegex(ValueError, "UNOWNED_OR_CROSS_PCI_RENDER_FD"):
                    linux_owner.audit_device_holders(devices, supervisor.GPU_SET)

    def test_group_signal_rejects_reused_pid(self):
        with tempfile.TemporaryDirectory() as temporary:
            proc = Path(temporary)
            pid = proc / "901"
            pid.mkdir()
            stat = pid / "stat"

            def write(start_ticks):
                fields = ["S", "55", "901", "901", *["0"] * 15, str(start_ticks)]
                stat.write_text("901 (test ) name) " + " ".join(fields))

            write(12345)
            with patch.object(linux_owner, "PROC", proc), patch.object(linux_owner.os, "getpid", return_value=55), \
                 patch.object(linux_owner.os, "killpg", create=True) as kill:
                child = linux_owner.Child(SimpleNamespace(pid=901), proc, "witness", supervisor.TARGET, {})
                child.signal(signal.SIGTERM)
                write(12346)
                with self.assertRaisesRegex(ValueError, "OWNED_SESSION_CHANGED"):
                    child.signal(signal.SIGTERM)
                kill.assert_called_once_with(901, signal.SIGTERM)


if __name__ == "__main__":
    unittest.main()
