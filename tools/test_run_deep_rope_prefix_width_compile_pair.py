"""Exercise the paired dispatcher on the exact published packages and fixture receipts."""

from hashlib import sha256
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from unittest import TestCase

import run_deep_rope_prefix_width_compile_pair as pair


SOURCE = Path(__file__).resolve().parents[1] / "code/bw1000/Deep-C4-causal-consumer-r483"


class CompilePairTest(TestCase):
    def setUp(self):
        local = TemporaryDirectory()
        self.addCleanup(local.cleanup)
        self.root = Path(local.name).resolve()
        self.packages = self.root / "incoming"
        self.outputs = self.root / "candidates"
        self.outputs.mkdir()
        for arm in pair.VARIANTS:
            shutil.copytree(SOURCE / f"inverse-output-rope-prefix-pack{arm.width}-target-v1",
                            self.packages / arm.label)
        self.epoch = 1795000000.0
        self.step = "1103606.700"
        self.node = "test-node"
        self.calls = []

    def emit(self, arm, *, corrupt=False):
        output = self.outputs / arm.label
        output.mkdir()
        obj = output / "inverse_output_rope.o"
        obj_data, fresh, actual = f"test only {arm.width}".encode(), b"test M", b"test MD"
        obj.write_bytes(obj_data + (b"changed" if corrupt else b""))
        (output / "inverse_output_rope.before.d").write_bytes(fresh)
        (output / "inverse_output_rope.d").write_bytes(actual)
        pair.save_once(output / "CPU-result.json", dict(label=arm.label, original_step=self.step, node=self.node,
            epoch=self.epoch, deadline=self.epoch+300, manifest_SHA256=arm.manifest, GRES="none",
            planned_device_TUs=1, actual_device_TUs=1, actual_GPU_steps=0, explicit_Runtime_calls=0,
            completed=True, passed=True, device_qualified=False, exports=[arm.export], object=str(obj),
            object_SHA256=sha256(obj_data).hexdigest(), fresh_M_SHA256=sha256(fresh).hexdigest(),
            actual_MD_SHA256=sha256(actual).hexdigest()))

    def fake(self, command, env, log, seconds, failed=0, corrupt=0):
        arm, = [arm for arm in pair.VARIANTS if arm.label == command.parent.name]
        self.assertEqual(env["TUNING_CPU_MANIFEST_SHA"], arm.manifest)
        self.assertEqual(log.name, f"pack{arm.width}.log")
        self.calls.append((arm.width, seconds))
        if arm.width == failed:
            return 2
        self.emit(arm, corrupt=arm.width == corrupt)
        return 0

    def execute(self, invoke=None, clock=lambda: 150):
        return pair.run_pair(self.packages, self.outputs, self.step, self.node, self.epoch,
                             clock, invoke=self.fake if invoke is None else invoke)

    def test_both_distinct_objects_pass_in_their_shared_step(self):
        result = self.execute()
        self.assertEqual(self.calls, [(32, 138), (64, 138)])
        self.assertEqual(result["verified_device_TUs"], 2)
        self.assertTrue(result["passed"])
        self.assertNotEqual(result["variant_results"][0]["object_SHA256"],
                            result["variant_results"][1]["object_SHA256"])

    def test_existing_variant_blocks_both_invocations(self):
        (self.outputs / pair.VARIANTS[1].label).mkdir()
        with self.assertRaisesRegex(ValueError, "reconcile"):
            self.execute()
        self.assertFalse(self.calls)

    def test_drift_in_the_second_package_blocks_both_invocations(self):
        (self.packages / pair.VARIANTS[1].label / "compile_once.py").write_text("altered")
        with self.assertRaisesRegex(ValueError, "SHA differs"):
            self.execute()
        self.assertFalse(self.calls)
        self.assertFalse((self.outputs / pair.PAIR).exists())

    def test_worker_failure_and_budget_exhaustion_retain_both_arms(self):
        budgets = iter([70, 21])
        result = self.execute(lambda *args: self.fake(*args, failed=32), clock=lambda: next(budgets))
        self.assertEqual(self.calls, [(32, 58)])
        self.assertEqual(len(result["variant_results"]), 2)
        self.assertEqual(result["variant_results"][0]["returncode"], 2)
        self.assertFalse(result["variant_results"][1]["invoked"])
        self.assertTrue((self.outputs / pair.PAIR / "pack64-result.json").exists())
        self.assertFalse(result["passed"])

    def test_bad_object_receipt_is_not_promoted(self):
        result = self.execute(lambda *args: self.fake(*args, corrupt=64))
        self.assertEqual(result["verified_device_TUs"], 1)
        self.assertTrue(result["variant_results"][0]["verified"])
        self.assertFalse(result["variant_results"][1]["verified"])
        self.assertFalse(result["passed"])

    def test_fixed_host_command_mounts_both_frozen_variants_read_only(self):
        repo = pair.PROJECT / "published"
        argv = pair.host_step_command(repo, "1103606", "test-node", self.epoch)
        self.assertIn("--gres=none", argv)
        self.assertIn("--mem=8192M", argv)
        self.assertIn("--cpus-per-task=4", argv)
        self.assertIn("--job-name=" + pair.PAIR, argv)
        for arm in pair.VARIANTS:
            source = repo / pair.SOURCE / f"inverse-output-rope-prefix-pack{arm.width}-target-v1"
            self.assertIn(f"{source}:/refactor/incoming/{arm.label}:ro", argv)
        self.assertIn(f"TUNING_CPU_DEADLINE={self.epoch+300}", argv[-1])
        with self.assertRaisesRegex(ValueError, "placement"):
            pair.host_step_command(repo, "1103606", "test-node;abort", self.epoch)
