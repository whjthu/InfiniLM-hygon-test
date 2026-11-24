"""Exercise the no-update SDPA route boundary with local recording fixtures."""

from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from read_completed import read_arms
from route_control import ARMS, FLASH, MATH, NATIVE_CALLS, WRAPPER, comparison, profile_with_route


STEP = "1103606.901234"


def corpus():
    records, original = {}, {}
    for rank in range(8):
        original[rank] = {"input": f"input-{rank}", "state": {"adapter": f"adapter-{rank}"}, "RNG": f"rng-{rank}"}
        for index, arm in enumerate(ARMS):
            math = arm.startswith("b")
            loss = 0.3 if math else 0.31 if arm == "a1" else 0.32
            records[(rank, arm)] = {
                "rank": rank, "world_size": 8, "state": "STOP_BEFORE_BACKWARD",
                "checkpoint_step": 20, "global_step": 20, "planned_forwards": 1, "completed_forwards": 1,
                "backward_calls": 0, "optimizer_updates": 0,
                "route_owner": {"arm": arm, "step": STEP, "declared_policy": "MATH_ONLY" if math else "NATIVE_AUTO",
                                "pid": 1000+rank*4+index, "startticks": 1, "boot_id": "fixture"},
                "actual_input": {"global_step": 20, "ordinal": 1, "runtime_cursor": {"micro_in_update": 1},
                                 "full_input": original[rank]["input"]},
                "actual_loss": {"ordinal": 1, "loss_scalar": loss, "loss": loss.hex()},
                "RNG_after_native_restore": original[rank]["RNG"],
                "RNG_before_compute_loss": original[rank]["RNG"], "RNG_after_forward": f"end-{math}-{rank}",
                "root_before_forward": {"RNG": original[rank]["RNG"], "model": {"training": True}},
                "restored_state": original[rank]["state"], "ddp": {"sync_gradients": False},
                "compute_loss_source": {"source": "fixture"},
                "language_before_forward": {"training": True, "attention_config": "sdpa"},
                "operators": {"SDPA_route_observed": True, "SDPA_wrappers": {WRAPPER: NATIVE_CALLS},
                              "selected_SDPA": {(MATH if math else FLASH): NATIVE_CALLS}},
            }
    return records, original


class RouteControlTest(unittest.TestCase):
    def test_actual_route_changes_only_in_math_phase_and_restores_on_exit(self):
        before = {"flash": True, "math": True, "mem_efficient": True, "cudnn": False}
        state = dict(before)
        math_only = {"flash": False, "math": True, "mem_efficient": False, "cudnn": False}
        selected = []

        @contextmanager
        def kernel(backends):
            self.assertEqual(backends, ["MATH"])
            selected.append(backends)
            state.update(math_only)
            try:
                yield
            finally:
                state.update(before)

        @contextmanager
        def profiler():
            yield dict(state)

        cuda = SimpleNamespace(**{name + "_sdp_enabled": lambda name=name: state[name] for name in before})
        attention = SimpleNamespace(SDPBackend=SimpleNamespace(MATH="MATH"), sdpa_kernel=kernel)
        torch = SimpleNamespace(backends=SimpleNamespace(cuda=cuda), nn=SimpleNamespace(attention=attention))
        with profile_with_route(torch, profiler, "a1") as actual:
            self.assertEqual(actual, before)
        with self.assertRaises(SystemExit), profile_with_route(torch, profiler, "b2") as actual:
            self.assertEqual(actual, math_only)
            raise SystemExit(0)
        self.assertEqual(state, before)
        self.assertEqual(selected, [["MATH"]])

    def test_scoped_policy_can_support_repeatability_without_claiming_historical_cause(self):
        result = comparison(*self.arguments())
        self.assertTrue(result["complete"])
        self.assertEqual(result["verified_forwards"], 32)
        self.assertEqual(result["comparable_ranks"], 8)
        self.assertTrue(all(row["math_loss_exact"] and not row["flash_loss_exact"] for row in result["rows"]))
        self.assertFalse(result["historical_cause_proven"])
        self.assertFalse(result["formal_recovery_qualified"])

    def test_route_fallback_and_missing_arm_keep_all_32_planned(self):
        records, _, original = self.arguments()
        records[(0, "b1")]["operators"]["selected_SDPA"] = {FLASH: NATIVE_CALLS}
        del records[(7, "a2")]
        result = comparison(records, STEP, original)
        self.assertFalse(result["complete"])
        self.assertEqual((result["planned_forwards"], result["verified_forwards"], result["missing_forwards"]), (32, 30, 1))

    def test_seed_change_and_reused_pid_are_rejected(self):
        records, _, original = self.arguments()
        records[(0, "a2")]["route_owner"]["pid"] = records[(0, "a1")]["route_owner"]["pid"]
        changed = records[(1, "b2")]
        changed["RNG_after_native_restore"] = changed["RNG_before_compute_loss"] = "different"
        changed["root_before_forward"]["RNG"] = "different"
        result = comparison(records, STEP, original)
        errors = {row["error"] for row in result["errors"]}
        self.assertEqual(errors, {"INPUT_RNG_STATE_OR_CONFIG_CHANGED", "ARM_PROCESS_REUSE_OR_BOOT_CHANGED"})

    def test_reader_requires_complete_state_ready_and_stop_receipts(self):
        records, _, original = self.arguments()
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(strict=True)
            directory = root / "a1/train/minicpm-usgs-witness/rank-00"
            directory.mkdir(parents=True)
            packet = {"state": records[(0, "a1")], "ready": {"run": {"actual_step": STEP},
                      "rank": 0, "world_size": 8, "checkpoint_step": 20, "state": "FORWARD_RETURNED"},
                      "stop": {"rank": 0, "state": "ALL_RANKS_STOPPED", "global_step": 20,
                               "backward_calls": 0, "optimizer_updates": 0}}
            for name, value in packet.items():
                (directory / f"first-forward-{name}.json").write_text(json.dumps(value), encoding="utf-8")
            passed, hashes, failed = read_arms(root, STEP)
            self.assertEqual((len(passed), len(hashes), len(failed)), (1, 3, 31))
            result = comparison(passed, STEP, original)
            self.assertEqual(result["planned_forwards"], 32)
            self.assertEqual(result["verified_forwards"], 1)
            (directory / "first-forward-stop.json").unlink()
            passed, _, _ = read_arms(root, STEP)
            self.assertFalse(passed)

    @staticmethod
    def arguments():
        records, original = corpus()
        return deepcopy(records), STEP, original


if __name__ == "__main__":
    unittest.main()
