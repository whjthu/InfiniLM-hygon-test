from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import preflight_gemma_render72_native_cpu as preflight


class GemmaPreflightTest(TestCase):
    def setUp(self):
        self.calls = []
        self.framework = Path("framework")
        self.stage_report = dict(state="STAGING_VERIFIED_CPU_PREREQUISITES_PENDING",
                                 verified_training_input_files=16)
        self.final_report = dict(stage_namespace_present=True, cpu_namespace_present=False,
                                 cpu_accounting_row=None)
        self.slot_available = True

    def stage(self):
        self.calls.append("stage")
        return self.stage_report

    def assets(self, root, check_deadline):
        self.assertEqual(root, self.framework)
        self.calls.append("assets")
        check_deadline()
        return {"training_rows":72}

    def slot(self):
        self.calls.append("slot")
        return self.slot_available

    def final(self):
        self.calls.append("final")
        return self.final_report

    def inspect(self, *, check_assets=None, check_deadline=lambda: None):
        return preflight.inspect(self.framework, stage_status=self.stage, final_status=self.final,
                                 cpu_slot=self.slot, check_deadline=check_deadline,
                                 check_assets=check_assets or self.assets)

    def test_unstarted_or_unknown_stage_ends_without_touching_cpu_inputs(self):
        for state in ("ORIGINAL_NAMESPACES_CLEAR", "RECONCILE_ORIGINAL"):
            with self.subTest(state=state):
                self.stage_report["state"] = state
                self.assertEqual(self.inspect()["state"], state)
                self.assertEqual(self.calls, ["stage"])
                self.calls.clear()

    def test_success_uses_inputs_before_fresh_slot_and_rechecks_original(self):
        report = self.inspect()
        self.assertEqual(self.calls, ["stage", "assets", "slot", "final"])
        self.assertEqual(report["state"], "TARGET_ASSETS_VERIFIED_CPU_DISPATCH_PENDING")
        self.assertEqual(report["assets"]["training_rows"], 72)
        self.assertEqual(report["cpu_resources"], dict(cpus=4, memory_MiB=8192, GRES=0))
        self.assertEqual((report["actual_Slurm_submissions"], report["optimizer_updates"]), (0, 0))
        self.assertFalse(report["training_READY"])

    def test_exact_worker_asset_failure_keeps_cpu_slot_unclaimed(self):
        def bad_assets(*_):
            raise ValueError("processor differs")
        with self.assertRaisesRegex(ValueError, "processor differs"):
            self.inspect(check_assets=bad_assets)
        self.assertEqual(self.calls, ["stage"])

    def test_no_dispatch_when_ownership_is_unavailable(self):
        self.slot_available = False
        self.assertEqual(self.inspect()["state"], "CPU_SLOT_UNCONFIRMED_OR_OCCUPIED")
        self.assertEqual(self.calls, ["stage", "assets", "slot"])

    def test_final_original_or_missing_stage_blocks_preflight_success(self):
        mutations = [("cpu_namespace_present", True), ("cpu_accounting_row", ["1103606.991"]),
                     ("stage_namespace_present", False)]
        for key, value in mutations:
            with self.subTest(key=key), patch.dict(self.final_report, {key:value}):
                self.assertEqual(self.inspect()["state"], "RECONCILE_ORIGINAL")
                self.assertEqual(self.calls, ["stage", "assets", "slot", "final"])
                self.calls.clear()

    def test_expired_boundary_avoids_fresh_slot_query(self):
        checks = iter([None, None, TimeoutError("expired")])

        def deadline():
            error = next(checks)
            if error:
                raise error

        with self.assertRaises(TimeoutError):
            self.inspect(check_deadline=deadline)
        self.assertEqual(self.calls, ["stage", "assets"])
