"""Read original resident-versus-reload evidence on the producing compute node."""
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
import socket
import sys


def need(condition, message):
    if not condition:
        raise ValueError(message)


def rows(root, role):
    worker = json.loads((root / role / "worker-result.json").read_bytes())
    trace = json.loads((root / role / "forward-trace.json").read_bytes())
    expected = list(range(1, 21)) if role == "continuous" else []
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
    need(warm["checkpoint_pin"] == cold["checkpoint_pin"] and
         warm["input_identity"] == cold["input_identity"], "Paired candidate checkpoint or input differs")
    need(warm["restored_before_forward"] is False and cold["restored_before_forward"] is True,
         "Resident arm must not restore its checkpoint")
    need(len(warm["prefix_matches_original"]) == 20 and not cold["prefix_matches_original"],
         "Original training prefix evidence differs")
    comparable = set(warm["operands"]) - {"prediction_SHA256"}
    need(set(cold["operands"]) - {"prediction_SHA256"} == comparable and comparable
         and all(warm["operands"][k] == cold["operands"][k] for k in comparable),
         "Measured forward operands differ")
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
            "checkpoint_pin": warm["checkpoint_pin"],
            "live_state_match": {key: warm["current_live_state_SHA256"][key] == cold["current_live_state_SHA256"][key]
                                 for key in ("optimizer", "trainables")},
            "first_prefix_step_different_from_original":
                next((index for index, same in enumerate(warm["prefix_matches_original"], 1) if not same), None),
            "first_different_boundary_input": first_input, "first_different_boundary_output": first_output,
            "boundaries_per_arm": len(warm["trace"]),
            "step21_prediction_matches_between_arms":
                warm["operands"]["prediction_SHA256"] == cold["operands"]["prediction_SHA256"],
            "matches_original_continuous_prediction": {key: arm["matches_original_continuous_prediction"]
                                                        for key, arm in (("warm", warm), ("cold", cold))},
            "loss_absolute_difference": abs(warm["loss"] - cold["loss"])}

def main():
    root = Path(__file__).resolve().parent
    claim = json.loads((root / "CLAIM.json").read_bytes())
    if (socket.gethostname().split(".")[0] != claim["node"] or
            os.environ.get("SLURM_JOB_ID") != claim["parent"]):
        raise ValueError("Comparison must execute on the original compute node")
    for key, step in zip(("continuous", "resume20"), sys.argv[1:], strict=True):
        worker = json.loads((root / key / "worker-result.json").read_bytes())
        need(worker["step"] == step and worker["request"] == claim["request"], "Unexpected worker identity")
    result = compare(root)
    result["reader_step"] = claim["parent"] + "." + os.environ["SLURM_STEP_ID"]
    for role in ("continuous", "resume20"):
        folder = root / role
        trace = json.loads((folder / "forward-trace.json").read_bytes())
        progress = (folder / "forward-boundary-progress.jsonl").read_bytes()
        need(trace["trace"] == [json.loads(line) for line in progress.splitlines()],
             "Durable forward trace differs")
    members = ("worker-result.json", "forward-trace.json", "forward-boundary-progress.jsonl", "optimizer-journal.jsonl")
    result["evidence"] = {role + "/" + member: sha256((root / role / member).read_bytes()).hexdigest()
                          for role in ("continuous", "resume20") for member in members}
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
