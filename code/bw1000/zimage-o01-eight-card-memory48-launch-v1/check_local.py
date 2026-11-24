"""Exercise the derived HOST and device entries without target access."""

import ast
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import memory_link as link
import render
import transport


SIDE_STEP = "1103606.2601"
GPU_STEP = "1103606.2602"
PRE_STEP = "1103606.2600"
UID = 42


def functions(raw):
    return {node.name: node for node in ast.parse(raw).body if isinstance(node, ast.FunctionDef)}


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def historical_step(spec):
    return [link.PREDECESSOR_STEP, link.PREDECESSOR_LABEL, "COMPLETED", "0:0", "10500", spec.NODE,
            "cpu=8,gres/dcu=8,mem=16G,node=1"]


def schedule(spec, rows, active=()):
    return {"epoch": time.time(), "parent": {"JobId": spec.PARENT, "JobState": "RUNNING", "UserId": "fixture(42)",
                                           "AllocTRES": "node=40,gres/dcu=320"},
            "parent_nodes": [spec.NODE, *[f"f13r9n{n:02}" for n in range(1, 40)]],
            "accounting_columns": spec.COLUMNS, "accounting_complete": True, "accounting_rows": rows,
            "numbered_steps_complete": True, "numbered_steps": list(active)}


def scope(spec):
    epoch = time.time()
    return {"request": link.REQUEST, "label": link.LABEL, "parent": spec.PARENT, "node": spec.NODE,
            "epoch": epoch, "deadline_epoch": epoch + 240, "allocated_DCU": 8, "host_memory_MiB": 49152,
            "memory_observer_label": link.OBSERVER_LABEL, "fresh_root_admitted": True, "boot_id_SHA256": "1" * 64,
            "attested_entry_SHA256": spec.ATTESTED_SHA, "runtime_SHA256": spec.RUNTIME_SHA,
            "independent_pre_step": PRE_STEP}


def local_write(path, payload):
    with path.open("xb") as handle:
        handle.write((json.dumps(payload, allow_nan=False, sort_keys=True) + "\n").encode())


class LinkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = render.baseline()
        cls.members = render.render()

    @contextmanager
    def host(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(sys, "path", list(sys.path)):
            path = Path(temporary)
            (path / "runtime").mkdir()
            for name, data in self.members.items():
                (path / name).write_bytes(data)
            spec = load(path / "host_scope.py", "derived_host_scope")
            spec.RUN = path
            spec.STAGE = path
            spec.save_new = local_write
            yield path, spec

    def test_derivation_preserves_math_and_original_cycle_operations(self):
        orig = functions(self.base["runtime/witness.py"])
        new = functions(self.members["runtime/witness.py"])
        for name in ("witness", "native_probe", "fixed_inputs", "guarded", "open_layout", "audit", "same_bits"):
            self.assertEqual(ast.dump(orig[name]), ast.dump(new[name]))
        self.assertEqual(self.base["runtime/contract.py"], self.members["runtime/contract.py"])
        self.assertEqual(self.base["runtime/standby_worker.py"], self.members["runtime/standby_worker.py"])
        previous = functions(self.base["runtime/supervisor.py"])["cycle"]
        revised = functions(self.members["runtime/supervisor.py"])["cycle"]
        phases = [node for node in revised.body if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                  and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "phase"]
        self.assertEqual(len(phases), 8)
        revised.body = [node for node in revised.body if node not in phases]
        self.assertEqual(ast.dump(previous), ast.dump(revised))
        self.assertEqual(self.base["node_entry.py"], self.members["identity_original.py"])
        self.assertIn(b'pending_namespace["install"]', self.members["guarded-standby-bridge.py"])
        self.assertIn(b'memory_namespace["install"]', self.members["guarded-standby-bridge.py"])

    def test_rendered_supervisor_does_not_construct_runtime_before_sample(self):
        with self.host() as (path, _), patch.object(sys, "path", list(sys.path)):
            runtime = path / "runtime"
            supervisor = load(runtime / "supervisor.py", "derived_supervisor")
            run = path / "supervised"
            argv = ["supervisor.py", "--scope", str(path / "scope.json"), "--scope-sha256", "a" * 64,
                    "--runtime-sha256", "b" * 64, "--out", str(run)]
            with patch.object(sys, "argv", argv), patch.object(supervisor, "verify_runtime"), \
                 patch.object(supervisor, "load_pinned", return_value=b"{}"), \
                 patch.object(supervisor, "physical_admission", return_value=GPU_STEP), \
                 patch.object(supervisor.signal, "signal"), patch.object(supervisor, "Step") as step, \
                 patch.object(supervisor.ram_trace, "await_baseline", side_effect=TimeoutError("test")):
                self.assertEqual(supervisor.main(), 1)
                step.assert_not_called()
                self.assertEqual(json.loads((run / "terminal.json").read_bytes())["error"]["type"], "TimeoutError")

    def test_two_exact_argv_share_epoch_and_gpu_waits_for_numbered_observer(self):
        with self.host() as (path, spec):
            plan = scope(spec)
            key = spec.digest(spec.encoded(plan))
            gpu = spec.render_gpu(plan, key, spec.ATTESTED_SHA, spec.DEVICE_HOST_SHA)
            sidecar = link.observer_argv(spec, plan, key)
            self.assertIn("--gres=dcu:8", gpu)
            self.assertIn("--cpus-per-task=12", gpu)
            self.assertIn("--mem=49152M", gpu)
            self.assertIn("--gres=none", sidecar)
            self.assertIn("--cpus-per-task=1", sidecar)
            self.assertIn("--mem=256M", sidecar)
            self.assertIn(key, gpu)
            self.assertIn(key, sidecar)
            self.assertNotIn("--overlap", gpu + sidecar)
            self.assertIn("--output=" + str(path / "ram-slurm.stdout"), sidecar)
            order = []
            row = (f"StepId={SIDE_STEP} Name={link.OBSERVER_LABEL} NodeList={spec.NODE} State=RUNNING "
                   "TRES=cpu=1,mem=256M,node=1")

            def start(argv, **kwargs):
                self.assertEqual(argv, sidecar)
                self.assertTrue((path / "ram-observer-attempt.json").exists())
                order.append("sidecar_cli")
                return SimpleNamespace(pid=123, poll=lambda: None)

            def launch(scope, argv):
                self.assertEqual(order, ["sidecar_cli", "sidecar_accounting"])
                self.assertTrue((path / "ram-observer-adopted.json").exists())
                self.assertEqual(argv, gpu)
                return {"step": GPU_STEP}

            scheduler = SimpleNamespace(collect=lambda: schedule(spec, [historical_step(spec)]),
                                        command=lambda *args: order.append("sidecar_accounting") or row,
                                        single=lambda line, identity: dict(word.split("=", 1) for word in line.split()))
            with patch.object(link.subprocess, "Popen", side_effect=start), \
                 patch.object(link.os, "getuid", return_value=UID, create=True):
                result = link.launch_pair(plan, gpu, spec, scheduler, launch, lambda: {})
            self.assertEqual(result["gpu"]["step"], GPU_STEP)
            self.assertEqual(json.loads((path / "ram-observer-attempt.json").read_bytes())["argv_SHA256"],
                             spec.digest(spec.encoded(sidecar)))

    def test_no_gpu_refusal_requires_exact_sidecar_terminal_for_post(self):
        with self.host() as (path, spec):
            old = historical_step(spec)
            old_running = list(old)
            old_running[2] = "RUNNING"
            with self.assertRaisesRegex(ValueError, "PREDECESSOR_STILL_PROTECTED"):
                spec.old_standby([old_running])
            plan = scope(spec)
            scheduler = SimpleNamespace(collect=lambda: schedule(spec, [old]), command=lambda *args: "")
            proc = SimpleNamespace(pid=110, poll=lambda: 3)
            with patch.object(link.subprocess, "Popen", return_value=proc), \
                 patch.object(link.os, "getuid", return_value=UID, create=True):
                refused = link.launch_pair(plan, ["fake", "gpu"], spec, scheduler,
                                           lambda *args: self.fail("GPU must never be sent"), lambda: {})
            self.assertFalse(refused["gpu_exec_attempted"])
            side = [SIDE_STEP, link.OBSERVER_LABEL, "FAILED", "1:0", "3", spec.NODE, "cpu=1,mem=256M,node=1"]
            with self.assertRaises(ValueError):
                spec.post_after_terminal(schedule(spec, [old]), plan, time.time(), UID)
            row, role, argv = spec.post_after_terminal(schedule(spec, [old, side]), plan, time.time(), UID)
            self.assertEqual(row, side)
            self.assertEqual(role, "REAL_RAM_OBSERVER_TERMINAL_WITH_PROVED_NO_GPU_EXEC")
            self.assertIn("--gres=none", argv)
            (path / "launcher-observation.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "RAM_PRE_GPU_REFUSAL_NOT_PROVED"):
                spec.post_after_terminal(schedule(spec, [old, side]), plan, time.time(), UID)

    def test_pinned_installer_accepts_exact_stage_members(self):
        with self.host() as (path, _):
            dispatch = transport.adapter()
            self.assertIn(link.LABEL, dispatch.STAGE)
            self.assertEqual(dispatch.REQUEST, link.REQUEST)
            installer = ast.parse(dispatch.INSTALLER)
            sets = {node.targets[0].id: ast.literal_eval(node.value) for node in installer.body
                    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id in ("root", "member")}
            files = set(self.members) - {"HOST_STAGE_FILES.json"}
            self.assertEqual(files, sets["root"] | {"runtime/" + name for name in sets["member"]})
            payload = dispatch.payload(path, render.pin(self.members["HOST_STAGE_FILES.json"])["sha256"])
            self.assertLessEqual(len(payload), 524288)


if __name__ == "__main__":
    unittest.main()
