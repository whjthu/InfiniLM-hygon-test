"""Read same-input and mask coverage evidence against the frozen native prefix scope."""

from hashlib import sha256
import json
import math
import re

from contract import load


def read_prefix_update(output):
    root = output / "native-prefix"
    result = {"passed": False, "expected_updates": 25, "verified_updates": 0, "formal_A100": False}
    try:
        intent, _ = load(root / "intent.json", 8192)
        if intent != {"shape": [1, 16, 5326, 128], "stride": [10907648, 128, 2048, 1],
                      "valid": 4785, "planned_updates": 25, "expected_mask_checks_per_phase": 54,
                      "same_input_update": 1, "formal_A100": False}:
            raise ValueError("Native prefix intent differs")
        pair, pair_pin = load(root / "first-update.json", 16384)
        result["first_update"] = pair
        result["first_update_SHA256"] = pair_pin["SHA256"]
        if (pair["actual_global_step"] != 1 or pair["baseline_reverted"] is not True
                or pair["passed"] is not True or pair["optimizer_scalar_entries_equal"] <= 0
                or pair["formal_A100"] is not False
                or pair["tolerance"] != {"relative_l2": 0.01, "max_tensor_relative_l2": 0.05}
                or set(pair["identical_inputs"]) != {"noise", "timestep", "hidden_states", "target"}
                or any(re.fullmatch(r"[0-9a-f]{64}", digest) is None
                       for digest in pair["identical_inputs"].values())):
            raise ValueError("Original same-input first update did not pass")
        groups = pair["groups"]
        if set(groups) != {"forward", "unclipped_gradients", "clipped_gradients",
                           "optimizer_tensors", "parameter_updates"}:
            raise ValueError("Native comparison tensor groups differ")
        trainables = groups["parameter_updates"]["tensors"]
        if (type(trainables) is not int or trainables <= 0 or groups["forward"]["tensors"] != 2
                or groups["unclipped_gradients"]["tensors"] != trainables
                or groups["clipped_gradients"]["tensors"] != trainables):
            raise ValueError("Native paired tensor denominators differ")
        for group in groups.values():
            if (group["tensors"] <= 0 or group["passed"] is not True
                    or not math.isfinite(group["relative_l2"]) or not 0 <= group["relative_l2"] <= 0.01
                    or not math.isfinite(group["maximum_per_tensor_relative_l2"])
                    or not 0 <= group["maximum_per_tensor_relative_l2"] <= 0.05):
                raise ValueError("Original same-input group failed")
        checks = {}
        for step in range(1, 26):
            observed, ob_pin = load(root / f"step-{step:02d}.json", 8192)
            if observed != {"global_step": step,
                            "actual_mask_checks": {"forward": 54, "backward_recompute": 54}, "passed": True}:
                raise ValueError("Native prefix actual mask check is incomplete")
            checks[str(step)] = ob_pin["SHA256"]
            result["verified_updates"] += 1
        result["mask_checks"] = result["verified_updates"] * 108
        result["step_receipts_SHA256"] = sha256(json.dumps(checks, sort_keys=True).encode()).hexdigest()
        result["passed"] = True
    except (OSError, KeyError, TypeError, ValueError) as error:
        result["error_type"] = type(error).__name__
    return result
