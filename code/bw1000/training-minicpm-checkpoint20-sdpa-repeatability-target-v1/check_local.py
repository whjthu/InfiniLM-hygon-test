"""Verify the rendered HOST controller with the actual archived dependencies."""

import argparse
from contextlib import redirect_stdout
from hashlib import sha256
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


TARGET = "training-minicpm-checkpoint20-sdpa-repeatability-target-v1"


class TestController(unittest.TestCase):
    def assembly(self):
        previous, bundle = host.verified(host.REPO)
        old = host.bind(previous, previous.controller(host.REPO))
        frozen = scope.frozen(directory, source_sha, scope.TARGET_FILES)
        names = scope.STAGED | {"original-observer/" + name for name in scope.OLD_OBSERVER}
        bundle.update({name: frozen[name] for name in names - bundle.keys()})
        self.assertEqual(set(bundle), names)
        self.assertEqual(len(bundle), 15)
        return previous, old, bundle

    def original_plan(self):
        source = host.REPO / "code/bw1000/training-minicpm-usgs-exposure25-resume20-hip-v1"
        translator = host.REPO / "code/bw1000/training-qwen-checkpointing-lifecycle-CPU-v1/framework/tools/hygon_vlm_recipe.py"
        recipe = scope.import_file("fixture_original_recipe", source / "minicpm_domain_recipe.py")
        original = recipe.command_plan(translator, (source / "original-launch.json").read_bytes(),
                                       str(scope.ORIGINAL), "a" * 64, "resumed", 29929)
        original_worker = scope.import_file("fixture_original_native_plan",
                                            host.REPO / "code/bw1000" / scope.ORIGINAL.name / "worker.py")
        return original_worker.normalized_expected(original, scope.ORIGINAL)

    def test_registered_command_and_four_scoped_arms(self):
        previous, old, _ = self.assembly()
        old.RUN, old.RESULTS = scope.RUN, scope.RESULTS
        config = dict(path=str(scope.RUN / "worker-config.json"), SHA256="f" * 64)
        apptainer = previous.gpu_command(old, config, 1000)
        self.assertIn(f"{scope.ORIGINAL}:{scope.ORIGINAL}:ro", apptainer)
        self.assertIn(f"{scope.RUN}:{scope.RUN}:rw", apptainer)
        self.assertIn(str(1000 + scope.CLEANUP), apptainer[-1])
        commands = [old.step_command(phase, apptainer) for phase in scope.PHASES]
        self.assertEqual([cmd[cmd.index("-t") + 1] for cmd in commands], ["1", "31", "1"])
        self.assertEqual([next(arg for arg in cmd if arg.startswith("--gres=")) for cmd in commands],
                         ["--gres=none", "--gres=dcu:8", "--gres=none"])

        original = self.original_plan()
        previous.render = importlib.import_module("contract").render
        self.assertEqual(list(scope.AB), ["a1", "b1", "b2", "a2"])
        for arm in scope.AB:
            rendered = scope.render(previous, original, arm)
            changed = {new for prior, new in zip(original["command"], rendered["command"], strict=True)
                       if prior != new}
            self.assertEqual(changed, {str(scope.RUN / arm / "train"),
                                       str(scope.RUN / "target-worker/native_route_observer.py")})
            expected_env = dict(original["environment_overrides"], MINICPM_ROUTE_ARM=arm,
                                MASTER_PORT=str(scope.PORT))
            self.assertEqual(rendered["environment_overrides"], expected_env)

    def test_asset_preflight_without_device_idle_and_refuses_prior_label(self):
        previous, bundle = host.verified(host.REPO)
        old = previous.controller(host.REPO)
        native = json.dumps(self.original_plan()).encode()
        with tempfile.TemporaryDirectory() as path:
            root = Path(path).resolve(strict=True)
            plan = root / "original-plan.json"
            plan.write_bytes(native)
            stage = {"plans": {"resumed": dict(path=str(plan), bytes=len(native),
                                               SHA256=sha256(native).hexdigest())}}
            with (patch.object(host, "verified", side_effect=lambda _: (previous, bundle.copy())),
                  patch.object(previous, "controller", return_value=old),
                  patch.object(old, "accepted_stage", return_value=stage),
                  patch.object(old, "accounting", return_value=None) as accounting,
                  patch.object(importlib.import_module("contract"), "checkpoint", return_value={}),
                  patch.object(host, "RUN", root / "unstarted"),
                  patch.object(host, "idle", side_effect=AssertionError("device admission ran unexpectedly")),
                  patch.object(sys, "argv", ["host_once.py", "--asset-preflight", "--source-sha256", source_sha]),
                  redirect_stdout(io.StringIO()) as stdout):
                self.assertEqual(host.main(), 0)
                result = json.loads(stdout.getvalue())
                self.assertEqual((result["original_asset_gate"], result["installed_worker_files"]), ("PASSED", 15))
                self.assertIs(result["device_admission_checked"], False)
                self.assertEqual(accounting.call_count, 3)
                self.assertFalse((root / "unstarted").exists())
                accounting.return_value = ["original"]
                with self.assertRaisesRegex(ValueError, "namespace or Slurm label"):
                    host.main()

    def test_current_parent_owner_and_empty_node(self):
        adm = host.current_admission
        ledger = json.loads((directory / "current_ledger.json").read_bytes())
        nodes = sum(adm.plan(ledger)[scope.PARENT].values(), [])

        def query_for(uid=123, target_step=None, live_nodes=None):
            def query(*argv):
                if argv[0] == "squeue":
                    self.assertEqual(argv[3], scope.PARENT)
                    return [scope.PARENT + "|RUNNING|40|all-nodes"]
                if argv[2] == "hostnames":
                    return (nodes if live_nodes is None else live_nodes) if argv[3] == "all-nodes" else [scope.NODE]
                if argv[2] == "job":
                    return [f"JobId={scope.PARENT} JobState=RUNNING UserId=owner({uid}) AllocTRES=gres/dcu=320"]
                self.assertEqual(argv[2], "step")
                rows = [f"StepId={scope.PARENT}.batch"]
                if target_step:
                    rows.append(f"StepId={scope.PARENT}.12345 NodeList=node Nodes=1 State=RUNNING TRES={target_step}")
                return rows
            return query

        self.assertEqual(adm.check(123, query_for())["allocated_DCU"], 320)
        for query in (query_for(uid=124), query_for(target_step="cpu=2"),
                      query_for(target_step="gres/dcu=8"), query_for(live_nodes=[*nodes[:-1], "other"])):
            with self.assertRaises(ValueError):
                adm.check(123, query)

    def test_one_epoch_and_missing_tail_budget(self):
        _, old, _ = self.assembly()
        archived = scope.import_file("fixture_original_worker", host.REPO / "code/bw1000" / scope.ORIGINAL.name / "worker.py")
        archived.WHOLE, archived.NATIVE = scope.WHOLE, scope.NATIVE
        clock = {"wall": 1010, "mono": 4}
        budget = archived.Budget(1000, 1000 + scope.WHOLE, 1000 + scope.NATIVE,
                                 wall=lambda: clock["wall"], monotonic=lambda: clock["mono"])
        self.assertEqual(budget.left(scope.AB["a1"], reserve=scope.ARM_FLOOR), 270)
        clock["mono"] += 300
        with self.assertRaises(ValueError):
            budget.left(scope.AB["a1"], reserve=scope.ARM_FLOOR)
        self.assertEqual(budget.left(scope.AB["b1"], reserve=scope.ARM_FLOOR), 350)
        clock["wall"] = 1000 + scope.AB["a2"] - 179
        with self.assertRaises(ValueError):
            budget.left(scope.AB["a2"], reserve=scope.ARM_FLOOR)
        self.assertEqual(old.WORKER_CLEANUP, scope.CLEANUP)
        self.assertGreater(scope.WHOLE - scope.GPU_CLIENT, 90 + 26)

    def test_native_child_receipt_and_policy(self):
        with tempfile.TemporaryDirectory() as path:
            root = Path(path)
            intents = []
            deadline_calls = []

            def save(dest, row):
                if dest.name == "native-intent.json":
                    intents.append(row)

            def left(offset, reserve=0):
                deadline_calls.append((offset, reserve))
                return 20

            def popen(*_args, **kwargs):
                env = kwargs["env"]
                self.assertTrue(kwargs["start_new_session"])
                self.assertEqual(env["MINICPM_ROUTE_ARM"], "b2")
                self.assertEqual(env["MINICPM_USGS_CONFIG_SHA256"], "d" * 64)
                return SimpleNamespace(pid=42, poll=lambda: 0, returncode=0, wait=lambda **_: 0)

            owned = SimpleNamespace(snapshot=lambda: {42: dict(PID=42, PPID=41, PGID=42, startticks=21, state="S")},
                                    refresh=lambda *_: {}, cleanup=lambda *_args, **_kw:
                                    dict(owned_processes_remaining=0, termination_needed=False))
            fake_port = SimpleNamespace(bind=lambda addr: self.assertEqual(addr, ("127.0.0.1", scope.PORT)))
            with (patch.object(worker, "os", SimpleNamespace(environ={})),
                  patch.object(worker.subprocess, "Popen", side_effect=popen), patch.object(worker.signal, "signal"),
                  patch.object(worker.socket, "socket") as sock,
                  patch.object(worker, "observed_ranks", return_value=list(range(8)))):
                sock.return_value.__enter__.return_value = fake_port
                plan = dict(command=["frozen-swift.py"],
                            environment_overrides=dict(MASTER_PORT=str(scope.PORT), MINICPM_ROUTE_ARM="b2"))
                result = worker.run_arm(root, plan, dict(epoch=100, native_deadline=1740, boot_id="boot"),
                                        dict(path="witness.json", SHA256="d" * 64), "b2",
                                        SimpleNamespace(left=left), owned, save)
            self.assertTrue(result["passed"])
            self.assertEqual(result["owners"][42]["startticks"], 21)
            self.assertEqual(deadline_calls[0], (scope.AB["b2"], scope.ARM_FLOOR))
            self.assertEqual(intents[0]["arm_deadline"], 1320)
            self.assertEqual(intents[0]["planned_optimizer_updates"], 0)

    def workflow(self, kind):
        previous, old, bundle = self.assembly()
        with tempfile.TemporaryDirectory() as path:
            old.RUN = Path(path).resolve(strict=True) / "new"
            old.original_membership = lambda: None
            old.accounting = lambda _: None
            before = dict(boot_id="boot", pci=list(range(8)), pci_to_kfd=None, master_port=scope.PORT)
            old.read_cpu_inventory = lambda phase, _: before if phase == "pre" else dict(
                before, owned_processes_remaining=[], owned_registry_complete=True)
            calls = []

            def execute(phase, command, elapsed, epoch):
                self.assertGreaterEqual(elapsed(), 0)
                self.assertTrue(command)
                calls.append(phase)
                step = scope.PARENT + ".1234"
                if phase == "gpu" and kind != "unknown":
                    names = list(scope.AB) if kind == "pass" else ["a1"]
                    arms = []
                    for arm in names:
                        (old.RUN / arm).mkdir()
                        old.write(old.RUN / arm / "native-intent.json", {})
                        arms.append(dict(arm=arm, cleanup_known=True, cleanup=dict(owned_processes_remaining=0)))
                    if kind == "cleanup_missing":
                        arms.clear()
                    old.write(old.RUN / "worker-outcome.json",
                              dict(request=scope.REQUEST, epoch=epoch, passed=kind == "pass", arms=arms,
                                   planned_forwards=32, planned_optimizer_updates=0))
                    if kind == "pass":
                        old.write(old.RUN / "worker-assigned.json", dict(request=scope.REQUEST, epoch=epoch,
                                  actual_step=step, boot_id="boot", visible_PCI_by_index=before["pci"]))
                        old.write(old.RUN / "sdpa-repeatability-comparison.json",
                                  dict(actual_step=step, complete=True, planned_forwards=32,
                                       verified_forwards=32, comparable_ranks=8))
                return dict(passed=phase != "gpu" or kind == "pass", row=[step],
                            terminal_visible=kind != "unknown" or phase != "gpu")

            old.execute_step = execute
            with patch.object(previous, "gpu_command", return_value=["fixture-apptainer"]):
                result = host.workflow(old, previous, dict(publication="a" * 40, stage={}), bundle, source_sha)
                with self.assertRaises(ValueError):
                    host.workflow(old, previous, dict(publication="a" * 40, stage={}), bundle, source_sha)
            return result, calls

    def test_terminal_and_unknown_preserve_all_32(self):
        for kind in ("pass", "failed", "cleanup_missing", "unknown"):
            with self.subTest(kind=kind):
                result, calls = self.workflow(kind)
                self.assertEqual(calls, ["pre", "gpu"] if kind == "unknown" else ["pre", "gpu", "post"])
                self.assertIs(result["complete"], kind == "pass", result.get("error"))
                self.assertIs(result["clean_post"], kind in ("pass", "failed"), result.get("error"))
                self.assertIs(result.get("terminal_outcome_unknown", False), kind == "unknown")
                self.assertEqual(result["planned_forwards"], 32)
                self.assertEqual(result["planned_optimizer_updates"], 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rendered_root", type=Path)
    directory = parser.parse_args().rendered_root.resolve(strict=True) / "code/bw1000" / TARGET
    source_sha = sha256((directory / "SOURCE_FILES.json").read_bytes()).hexdigest()
    sys.path.insert(0, str(directory))
    host = importlib.import_module("host_once")
    scope = importlib.import_module("bracket_contract")
    worker = importlib.import_module("worker")
    unittest.main(argv=[__file__], verbosity=2)
