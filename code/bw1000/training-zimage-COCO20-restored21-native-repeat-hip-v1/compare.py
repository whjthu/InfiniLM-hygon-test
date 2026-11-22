"""Consume the completed repeat arms in native execution order."""

import json
import math
import re


FILES = ("worker-result.json", "repeat-receipt.json", "steps.jsonl", "optimizer-journal.jsonl")
DIRECT_STAGES = ("restored_integrity_SHA256", "input_identity", "operands", "loss")
FINAL_STAGES = ("adapter_SHA256", "optimizer_SHA256", "rng_SHA256", "diffusion_scheduler_SHA256")


def need(condition, message):
    if not condition:
        raise ValueError(message)


def difference(left, right):
    need(len(left) == 476 and set(left) == set(right), "LoRA tensor coverage differs")
    changed = sorted(name for name in left if left[name] != right[name])
    return {"changed_tensors": len(changed), "first_changed_names": changed[:8]}


def compare(root):
    claim = json.loads((root / "CLAIM.json").read_bytes())
    rows = {}
    workers = {}
    for mode, expected in claim["arms"].items():
        path = root / mode
        worker = json.loads((path / "worker-result.json").read_bytes())
        receipt = json.loads((path / "repeat-receipt.json").read_bytes())
        need(worker["passed"] and worker["request"] == claim["request"]
             and worker["physical_PCI"] == receipt["original_reference_pci"]
             and worker["optimizer_returned_steps"] == worker["optimizer_intended_steps"] == [21] * len(expected)
             and worker["actual_optimizer_updates"] == len(expected) and receipt["arms"] == expected,
             "Original-device worker or optimizer completion differs")
        workers[mode] = worker["step"], worker["worker_identity"], worker["physical_PCI"], receipt
        journal = [json.loads(line) for line in (path / "optimizer-journal.jsonl").read_bytes().splitlines()]
        need(journal == [{"global_step": 21, "arm": arm, "event": event} for arm in expected
                         for event in ("optimizer_intent", "optimizer_returned")], "Optimizer journal differs")
        measured = [json.loads(line) for line in (path / "steps.jsonl").read_bytes().splitlines()]
        need([row["arm"] for row in measured] == expected, "Missing or duplicated native repeat arm")
        for row in measured:
            need(row["global_step"] == 21 and math.isfinite(row["preclip_norm"]) and row["preclip_norm"] > 0
                 and row["original_checkpoint_SHA256"] == claim["original_files"]["continuous/checkpoint-20.private.pt"]["SHA256"],
                 "Native repeat row identity differs")
            need(set(row["gradients"]) == {"preclip", "postclip"}, "Native gradient boundary coverage differs")
            for hashes in (*row["gradients"].values(), row["adapter_tensor_SHA256"]):
                need(len(hashes) == 476 and all(re.fullmatch("[0-9a-f]{64}", value) for value in hashes.values()),
                     "Native tensor hashes incomplete")
            rows[row["arm"]] = row
    first_step, first_pid, first_pci, first_receipt = workers["continuous"]
    new_step, new_pid, new_pci, new_receipt = workers["resume20"]
    need(first_step != new_step and first_pid != new_pid and first_pci == new_pci
         and first_receipt["flash2_has_deterministic_argument"] == new_receipt["flash2_has_deterministic_argument"],
         "Independent native process or provider differs")
    result = {}
    base = rows["resident_first"]
    for arm in ("resident_repeat", "fresh_process"):
        right = rows[arm]
        need(all(base[field] == right[field] for field in DIRECT_STAGES), "Restored state or forward differs")
        gradients = {stage: difference(base["gradients"][stage], right["gradients"][stage])
                     for stage in ("preclip", "postclip")}
        equal = {field: base[field] == right[field] for field in FINAL_STAGES}
        stages = {"preclip": gradients["preclip"]["changed_tensors"] == 0,
                  "preclip_norm": base["preclip_norm"] == right["preclip_norm"],
                  "postclip": gradients["postclip"]["changed_tensors"] == 0, **equal}
        result[arm] = {
            "first_difference": next((stage for stage, same in stages.items() if not same), None),
            "same_by_stage": stages,
            "gradients": gradients,
            "adapter": difference(base["adapter_tensor_SHA256"], right["adapter_tensor_SHA256"]),
            "preclip_norms": [base["preclip_norm"], right["preclip_norm"]],
        }
    return dict(passed=True, completed_updates=len(rows), original_step21_forward_exact=True,
                arms=list(rows), comparisons=result, worker_steps={mode: record[0] for mode, record in workers.items()}, flash2_has_deterministic_argument=first_receipt["flash2_has_deterministic_argument"],
                formal_recovery=False, formal_A100=False, formal_performance=False)
