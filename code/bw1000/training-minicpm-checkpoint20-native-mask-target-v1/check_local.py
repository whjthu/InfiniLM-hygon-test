"""Exercise the rendered control path and failure denominators on the local CPU."""

import argparse
from contextlib import redirect_stdout
from hashlib import sha256
import importlib
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


TARGET = Path(__file__).parent.name
STEP = "1103606.901234"


def corpus():
    old_records, history = old_fixture.corpus()
    records = {}
    for rank in range(8):
        for arm, old_arm in zip(compare.ARMS, ("a1", "b1", "b2", "a2"), strict=True):
            record = old_records[rank, old_arm]
            record["route_owner"].update(arm=arm, declared_policy="NATIVE_AUTO")
            loss = 0.31 if arm in ("u1", "c1") else 0.32
            record["actual_loss"].update(loss_scalar=loss, loss=loss.hex())
            record["RNG_after_forward"] = f"end-{rank}"
            record["root_before_forward"]["context"] = dict(autocast_enabled=False)
            record["language_before_forward"]["context"] = dict(autocast_enabled=True, autocast_dtype="torch.bfloat16")
            record["inner_linear"] = dict(module="fixture.linear", context=dict(autocast_enabled=True))
            record["operators"].update(selected_SDPA={route.FLASH: route.NATIVE_CALLS},
                                       native_dropout={"aten::dropout": 186, "aten::native_dropout": 186})
            inventory = None
            if arm.startswith("c"):
                masks = [dict(ordinal=i, module=f"layer.{i}.dropout", shape=[8], mask_stride=[1],
                              mask_bytes=8, provider="aten::native_dropout", mask_SHA256="a" * 64)
                         for i in range(186)]
                inventory = dict(native_mask_calls=186, mask_fingerprints=masks,
                                 logical_mask_bytes_moved_to_host=186*8, charged_device_storage_bytes=186*8,
                                 cap_device_storage_bytes=compare.CAP, post_forward_copy_and_digest_seconds=0.01)
            record["native_mask_observation"] = dict(arm=arm, storage_cap=compare.CAP, fingerprints=inventory)
            records[rank, arm] = record
    return records, history


class MaskControlTest(unittest.TestCase):
    def test_repeated_loss_with_matching_masks_keeps_all_32_forwards(self):
        records, history = corpus()
        result = compare.comparison(records, STEP, history)
        self.assertTrue(result["complete"], result["errors"])
        self.assertEqual((result["verified_forwards"], result["comparable_ranks"]), (32, 8))
        self.assertTrue(all(row["repeated_loss_diff_with_equal_observed_masks"] for row in result["rows"]))
        self.assertFalse(result["causal_source_proven"])
        self.assertFalse(result["formal_recovery_qualified"])

    def test_masks_and_observer_interference_are_independent(self):
        records, history = corpus()
        records[0, "c2"]["native_mask_observation"]["fingerprints"]["mask_fingerprints"][5]["mask_SHA256"] = "b" * 64
        records[1, "c2"]["actual_loss"] = records[1, "c1"]["actual_loss"]
        result = compare.comparison(records, STEP, history)
        first, second = result["rows"][:2]
        self.assertTrue(result["complete"])
        self.assertEqual(first["mask_comparison"]["different_native_masks"], 1)
        self.assertFalse(first["repeated_loss_diff_with_equal_observed_masks"])
        self.assertTrue(second["control_variation_not_reproduced_with_observer"])

    def test_bad_mask_cap_and_missing_rank_stay_in_denominator(self):
        records, history = corpus()
        records[0, "c1"]["native_mask_observation"]["storage_cap"] += 1
        del records[7, "u2"]
        result = compare.comparison(records, STEP, history)
        self.assertFalse(result["complete"])
        self.assertEqual((result["planned_forwards"], result["verified_forwards"], result["missing_forwards"]), (32, 30, 1))

    def test_changed_initial_rng_duplicate_process_and_old_step_fail(self):
        records, history = corpus()
        records[0, "c2"]["route_owner"]["pid"] = records[0, "c1"]["route_owner"]["pid"]
        changed = records[1, "u2"]
        changed["RNG_after_native_restore"] = changed["RNG_before_compute_loss"] = "drift"
        changed["root_before_forward"]["RNG"] = "drift"
        result = compare.comparison(records, STEP, history)
        self.assertFalse(result["complete"])
        self.assertEqual(result["comparable_ranks"], 7)
        self.assertIn({"reason": "RANK_PROCESS_REUSED_OR_BOOT_CHANGED"}, result["errors"])
        with self.assertRaisesRegex(ValueError, "new numbered mask diagnostic"):
            compare.comparison(records, "1103606.2303", history)

    def test_capture_with_changed_autocast_cannot_be_compared(self):
        records, history = corpus()
        records[0, "c2"]["inner_linear"]["context"]["autocast_enabled"] = False
        result = compare.comparison(records, STEP, history)
        self.assertFalse(result["complete"])
        self.assertEqual((result["planned_forwards"], result["comparable_ranks"]), (32, 7))


class RenderedHostTest(unittest.TestCase):
    def plan(self):
        source = host.REPO / "code/bw1000/training-minicpm-usgs-exposure25-resume20-hip-v1"
        translator = host.REPO / "code/bw1000/training-qwen-checkpointing-lifecycle-CPU-v1/framework/tools/hygon_vlm_recipe.py"
        recipe = scope.import_file("fixture_original_recipe", source / "minicpm_domain_recipe.py")
        original = recipe.command_plan(translator, (source / "original-launch.json").read_bytes(),
                                       str(scope.ORIGINAL), "a" * 64, "resumed", 29929)
        worker = scope.import_file("fixture_original_worker", host.REPO / "code/bw1000" / scope.ORIGINAL.name / "worker.py")
        return worker.normalized_expected(original, scope.ORIGINAL)

    def test_asset_preflight_uses_original_archive_and_no_device_command(self):
        previous, bundle = host.verified(host.REPO)
        old = previous.controller(host.REPO)
        native = json.dumps(self.plan()).encode()
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(strict=True)
            plan = root / "original-plan.json"
            plan.write_bytes(native)
            stage = {"plans": {"resumed": dict(path=str(plan), bytes=len(native), SHA256=sha256(native).hexdigest())}}
            with (patch.object(host, "verified", side_effect=lambda _: (previous, bundle.copy())),
                  patch.object(previous, "controller", return_value=old),
                  patch.object(old, "accepted_stage", return_value=stage),
                  patch.object(old, "accounting", return_value=None) as accounting,
                  patch.object(importlib.import_module("contract"), "checkpoint", return_value={}),
                  patch.object(host, "RUN", root / "unstarted"),
                  patch.object(host, "idle", side_effect=AssertionError("unexpected device admission")),
                  patch.object(sys, "argv", ["host_once.py", "--asset-preflight", "--source-sha256", source_sha]),
                  redirect_stdout(io.StringIO()) as stdout):
                self.assertEqual(host.main(), 0)
                result = json.loads(stdout.getvalue())
                self.assertEqual((result["installed_worker_files"], result["actual_GPU_steps"]), (18, 0))
                self.assertFalse(result["device_admission_checked"])
                self.assertFalse((root / "unstarted").exists())
                self.assertEqual(accounting.call_count, 3)
                accounting.return_value = ["prior"]
                with self.assertRaisesRegex(ValueError, "namespace or Slurm label"):
                    host.main()

    def test_new_callback_four_arms_and_real_slurm_resources(self):
        previous, _ = host.verified(host.REPO)
        old = host.bind(previous, previous.controller(host.REPO))
        old.RUN, old.RESULTS = scope.RUN, scope.RESULTS
        config = dict(path=str(scope.RUN / "worker-config.json"), SHA256="f" * 64)
        apptainer = previous.gpu_command(old, config, 1000)
        self.assertIn(f"{scope.ORIGINAL}:{scope.ORIGINAL}:ro", apptainer)
        self.assertIn(str(1000 + scope.CLEANUP), apptainer[-1])
        commands = [old.step_command(phase, apptainer) for phase in scope.PHASES]
        self.assertEqual([cmd[cmd.index("-t") + 1] for cmd in commands], ["1", "31", "1"])
        self.assertEqual([next(arg for arg in cmd if arg.startswith("--gres=")) for cmd in commands],
                         ["--gres=none", "--gres=dcu:8", "--gres=none"])
        original = self.plan()
        previous.render = importlib.import_module("contract").render
        self.assertEqual(tuple(scope.AB), compare.ARMS)
        for arm in scope.AB:
            rendered = scope.render(previous, original, arm)
            changed = {new for prior, new in zip(original["command"], rendered["command"], strict=True) if prior != new}
            self.assertEqual(changed, {str(scope.RUN / arm / "train"), str(scope.RUN / "target-worker/native_plugin.py")})
            self.assertEqual(rendered["environment_overrides"], dict(original["environment_overrides"],
                             MINICPM_NATIVE_MASK_ARM=arm, MINICPM_NATIVE_MASK_STORAGE_LIMIT=str(compare.CAP),
                             MASTER_PORT=str(scope.PORT)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rendered_root", type=Path)
    directory = parser.parse_args().rendered_root.resolve(strict=True) / "code/bw1000" / TARGET
    source_sha = sha256((directory / "SOURCE_FILES.json").read_bytes()).hexdigest()
    sys.path.insert(0, str(directory))
    host = importlib.import_module("host_once")
    scope = importlib.import_module("bracket_contract")
    compare = importlib.import_module("mask_compare")
    route = importlib.import_module("route_control")
    old_fixture = scope.import_file("prior_fixture", Path(__file__).resolve().parent.with_name(
        "training-minicpm-checkpoint20-sdpa-repeatability-v1") / "check_local.py")
    unittest.main(argv=[__file__], verbosity=2)
