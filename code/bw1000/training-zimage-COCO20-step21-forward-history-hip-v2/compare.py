"""Classify the first native forward divergence from hashed records."""
import json
from math import isfinite


def need(condition, message):
    if not condition:
        raise ValueError(message)


def rows(root, role):
    worker = json.loads((root / role / "worker-result.json").read_bytes())
    trace = json.loads((root / role / "forward-trace.json").read_bytes())
    expected = list(range(11, 21)) if role == "continuous" else []
    need(worker["passed"] and worker["training"]["warm_updates"] == trace["warm_updates"] == expected,
         "Missing native worker or warm-update denominator")
    need(worker["actual_optimizer_updates"] == len(expected) and trace["measured_optimizer_updates"] == 0,
         "Measured forward crossed the optimizer boundary")
    need(trace["mode"] == role and isfinite(trace["loss"]) and trace["trace"], "Incomplete native trace")
    return worker, trace


def compare(root):
    warm_worker, warm = rows(root, "continuous")
    cold_worker, cold = rows(root, "resume20")
    need(warm_worker["physical_PCI"] == cold_worker["physical_PCI"]
         and warm_worker["worker_identity"] != cold_worker["worker_identity"]
         and warm_worker["step"] != cold_worker["step"], "Paired device or process identity differs")
    need(warm["restored_integrity_SHA256"] == cold["restored_integrity_SHA256"]
         and warm["input_identity"] == cold["input_identity"], "Paired original checkpoint or input differs")
    left_order = [(row["boundary"], row["occurrence"]) for row in warm["trace"]]
    right_order = [(row["boundary"], row["occurrence"]) for row in cold["trace"]]
    need(left_order == right_order,
         "Paired native execution sequences differ")
    first_input = first_output = None
    for left, right in zip(warm["trace"], cold["trace"], strict=True):
        boundary = {"name": left["boundary"], "occurrence": left["occurrence"]}
        if first_input is None and (left["args"], left["kwargs"]) != (right["args"], right["kwargs"]):
            first_input = boundary
        if first_output is None and left["output"] != right["output"]:
            first_output = boundary
    return {"passed": True, "verified_diagnostic_only": True, "formal_recovery": False,
            "worker_steps": {"continuous": warm_worker["step"], "resume20": cold_worker["step"]},
            "warm_updates": len(warm["warm_updates"]), "measured_forwards": 2,
            "first_different_boundary_input": first_input, "first_different_boundary_output": first_output,
            "boundaries_per_arm": len(warm["trace"]),
            "step21_prediction_matches_between_arms":
                warm["operands"]["prediction_SHA256"] == cold["operands"]["prediction_SHA256"],
            "matches_original_continuous_prediction": {key: arm["matches_original_continuous_prediction"]
                                                        for key, arm in (("warm", warm), ("cold", cold))},
            "loss_absolute_difference": abs(warm["loss"] - cold["loss"])}
