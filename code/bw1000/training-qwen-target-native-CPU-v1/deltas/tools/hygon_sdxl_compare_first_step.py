"""Independently compare a fixed SDXL UNet update against the A100 r2 policy."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from pickle import UnpicklingError

from hygon_sdxl_first_step import UNET_FILES, require, sha, verify_reference


BOUNDARY = "fixed_realized_inputs"


def check_tensor(torch, actual, expected, name):
    require(isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor)
            and actual.shape == expected.shape and actual.dtype == expected.dtype,
            f"{name}: tensor shape or dtype differs")
    require(bool(torch.isfinite(actual).all()) and bool(torch.isfinite(expected).all()),
            f"{name}: nonfinite tensor")


def exact(torch, actual, expected, name):
    if isinstance(expected, torch.Tensor):
        check_tensor(torch, actual, expected, name)
        require(torch.equal(actual, expected), f"{name}: tensor values differ")
    elif isinstance(expected, dict):
        require(isinstance(actual, dict) and list(actual) == list(expected), f"{name}: keys or order differ")
        for key, value in expected.items():
            exact(torch, actual[key], value, f"{name}/{key}")
    elif isinstance(expected, (list, tuple)):
        require(type(actual) is type(expected) and len(actual) == len(expected), f"{name}: sequence differs")
        for index, (left, right) in enumerate(zip(actual, expected)):
            exact(torch, left, right, f"{name}/{index}")
    else:
        require(type(actual) is type(expected) and actual == expected, f"{name}: scalar differs")
        require(not isinstance(actual, float) or math.isfinite(actual), f"{name}: nonfinite scalar")


def relative(error, reference):
    return error / reference if reference else (0.0 if error == 0 else None)


def vector_metrics(torch, actual, expected, name, initial=None):
    require(isinstance(actual, dict) and list(actual) == list(expected), f"{name}: names or order differ")
    require(bool(expected), f"{name}: missing tensors")
    per_tensor = {}
    for key, right in expected.items():
        left = actual[key]
        check_tensor(torch, left, right, f"{name}/{key}")
        require(right.is_floating_point() and right.numel() > 0, f"{name}/{key}: expected floating data")
        if initial is not None:
            check_tensor(torch, right, initial[key], f"{name}/{key}/initial")
        a, b = left.double().reshape(-1), right.double().reshape(-1)
        if initial is not None:
            start = initial[key].double().reshape(-1)
            a, b = a - start, b - start
        delta = a - b
        a2, b2 = float(torch.dot(a, a)), float(torch.dot(b, b))
        delta2, dot = float(torch.dot(delta, delta)), float(torch.dot(a, b))
        require(all(math.isfinite(value) for value in (a2, b2, delta2, dot)),
                f"{name}/{key}: nonfinite reduction")
        per_tensor[str(key)] = {
            "reference_l2_squared": b2, "observed_l2_squared": a2,
            "difference_l2_squared": delta2, "dot": dot,
            "relative_l2": relative(math.sqrt(delta2), math.sqrt(b2)),
        }
    rows = per_tensor.values()
    b_norm = math.sqrt(math.fsum(row["reference_l2_squared"] for row in rows))
    a_norm = math.sqrt(math.fsum(row["observed_l2_squared"] for row in rows))
    error = math.sqrt(math.fsum(row["difference_l2_squared"] for row in rows))
    dot = math.fsum(row["dot"] for row in rows)
    return {
        "reference_l2": b_norm, "observed_l2": a_norm, "relative_l2": relative(error, b_norm),
        "cosine": dot / (a_norm * b_norm) if a_norm and b_norm else None,
        "per_tensor": per_tensor,
        "worst_relative_l2": sorted(per_tensor, key=lambda key: (
            per_tensor[key]["relative_l2"] is None, per_tensor[key]["relative_l2"]
        ), reverse=True)[:10],
    }


def compare(torch, actual, reference, policy):
    limits = policy["metrics"]
    for key in ("initial_adapter", "initial_optimizer", "inputs", "batch", "target"):
        exact(torch, actual[key], reference[key], key)
    initial = reference["initial_adapter"]
    initial_groups = reference["initial_optimizer"]["param_groups"]
    require(len(initial_groups) == 1 and initial_groups[0]["params"] == list(range(len(initial))),
            "optimizer IDs do not map to the ordered trainables")
    optimizer = actual["final_optimizer"]
    expected_optimizer = reference["final_optimizer"]
    exact(torch, optimizer["param_groups"], expected_optimizer["param_groups"], "final optimizer groups")
    state, expected_state = optimizer["state"], expected_optimizer["state"]
    require(list(state) == list(expected_state), "optimizer state IDs or order differ")
    moments = policy["optimizer_state_tensor_keys"]
    for param_id, values in expected_state.items():
        require(list(state[param_id]) == list(values), f"optimizer {param_id}: fields or order differ")
        for key in values.keys() - set(moments):
            exact(torch, state[param_id][key], values[key], f"optimizer/{param_id}/{key}")

    metrics = {}
    checks = {"exact_inputs_and_state_schema": True}
    for name, limit in (("loss", "loss_relative_error_max"), ("preclip_norm", "gradient_preclip_norm_relative_error_max")):
        check_tensor(torch, actual[name], reference[name], name)
        require(reference[name].numel() == 1 and reference[name].is_floating_point(), f"{name}: expected scalar")
        a, b = float(actual[name]), float(reference[name])
        error = relative(abs(a - b), abs(b))
        metrics[name] = {"observed": a, "reference": b, "relative_error": error}
        checks[name] = error is not None and error <= limits[limit]

    def vectors(name, left, right, limit, initial=None):
        value = vector_metrics(torch, left, right, name, initial)
        metrics[name] = value
        error = value["relative_l2"]
        checks[name] = error is not None and error <= limits[limit]

    vectors("prediction", {"output": actual["prediction"]}, {"output": reference["prediction"]},
            "prediction_relative_l2_max")
    for name, limit in (("gradients_before_clip", "gradient_preclip_relative_l2_max"),
                        ("gradients_after_clip", "gradient_postclip_relative_l2_max")):
        require(list(reference[name]) == list(initial), f"{name}: trainable order differs")
        vectors(name, actual[name], reference[name], limit)
    cosine = metrics["gradients_before_clip"]["cosine"]
    checks["gradient_direction"] = cosine is not None and cosine >= limits["gradient_preclip_cosine_min"]
    require(list(reference["final_adapter"]) == list(initial), "final trainable order differs")
    vectors("parameter_update", actual["final_adapter"], reference["final_adapter"],
            "parameter_update_relative_l2_max", initial)
    names = list(initial)
    for key in moments:
        vectors("optimizer_" + key, {names[i]: state[i][key] for i in expected_state},
                {names[i]: expected_state[i][key] for i in expected_state}, "optimizer_moment_relative_l2_max")
    checks["nonzero_gradient_and_update"] = all(
        metrics[name][side] > 0 for name in ("gradients_before_clip", "parameter_update")
        for side in ("reference_l2", "observed_l2")
    )
    checks["all_required_floating_values_finite"] = True
    return {"passed": all(checks.values()), "checks": checks, "metrics": metrics}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "policy", "observed", "receipt", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--accelerator", choices=("hip", "cuda"), default="hip")
    args = parser.parse_args()
    binding = verify_reference(args.reference, args.policy)
    receipt = json.loads(args.receipt.read_bytes())
    observed_sha = sha(args.observed)
    require(all(receipt.get(key) == value for key, value in binding.items())
            and receipt.get("boundary") == BOUNDARY
            and receipt.get("observed_sha256") == observed_sha
            and receipt.get("unet_files") == UNET_FILES
            and receipt.get("replay_source_sha256") == sha(Path(__file__).with_name("hygon_sdxl_first_step.py")),
            "observed file or replay receipt differs from the frozen binding")
    runtime = receipt["runtime"]
    require(runtime["accelerator"] == args.accelerator and runtime["cuda_available"] is True
            and type(runtime["cuda_count"]) is int and runtime["cuda_count"] == 1,
            "receipt must describe one visible device on the requested accelerator")
    policy = json.loads(args.policy.read_bytes())
    import torch

    with args.output.open("x", encoding="utf-8") as handle:
        torch.set_num_threads(1)
        try:
            reference = torch.load(args.reference, map_location="cpu", weights_only=True)
            observed = torch.load(args.observed, map_location="cpu", weights_only=True)
            result = compare(torch, observed, reference, policy)
        except (ValueError, TypeError, KeyError, IndexError, RuntimeError, ArithmeticError, UnpicklingError) as error:
            result = {"passed": False, "structural_failure": str(error)}
        json.dump({**binding, **result, "accelerator": args.accelerator, "boundary": BOUNDARY, "observed_sha256": observed_sha,
                   "replay_receipt_sha256": sha(args.receipt), "comparator_sha256": sha(Path(__file__))},
                  handle, indent=2, allow_nan=False)
        handle.write("\n")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
