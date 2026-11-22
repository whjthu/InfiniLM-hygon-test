"""Independently reconcile the exact Z-Image COCO20 terminal records."""

from datetime import datetime, timezone
from hashlib import file_digest, sha256
import json
import math
from pathlib import Path
import re
import subprocess


ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/"
            "training-zimage-COCO20-recovery-boundaries-hip-v2")
REQUEST = "33f344e7da364b9883dd5345dfcd8c16"
NODE = "f13r2n07"
IDS = {"admission": "1103606.1386", "pre": "1103606.1387", "continuous": "1103606.1388",
       "between": "1103606.1389", "resume20": "1103606.1390", "post": "1103606.1391"}
COLUMNS = ["JobIDRaw", "JobName%100", "State", "ExitCode", "ElapsedRaw", "NodeList", "AllocTRES"]
BOUNDARIES = ("preclip", "postclip", "LoRA", "step", "exp_avg", "exp_avg_sq")
FILE_PINS = {}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def raw(name, limit=2**21):
    path = ROOT / name
    need(path.resolve(strict=True) == path and path.is_file(), "Missing or symlinked original record")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    need(0 < len(data) <= limit, "Original record exceeds its bound")
    FILE_PINS[name] = {"bytes": len(data), "SHA256": sha256(data).hexdigest()}
    return data


def read(name):
    return json.loads(raw(name))


def lines(name):
    return [json.loads(line) for line in raw(name).splitlines()]


def verify_step(role, row):
    ident = IDS[role]
    expected_cards = role in ("continuous", "resume20")
    need(row[:4] == [ident, ROOT.name + "-" + role, "COMPLETED", "0:0"] and row[5] == NODE
         and ("gres/dcu=1" in row[6].split(",")) == expected_cards, "Slurm identity or resource shape differs")


def verify_arm(role, first):
    worker = read(role + "/worker-result.json")
    steps = list(range(first, 26))
    need(worker["passed"] is True and worker["request"] == REQUEST and worker["step"] == IDS[role]
         and worker["optimizer_intended_steps"] == worker["optimizer_returned_steps"] == steps
         and worker["actual_optimizer_updates"] == len(steps)
         and worker["optimizer_completion_uncertain"] is False, "Real optimizer denominators differ")
    journal = lines(role + "/optimizer-journal.jsonl")
    need(journal == [{"global_step": step, "event": event} for step in steps
                     for event in ("optimizer_intent", "optimizer_returned")], "Actual optimizer journal differs")
    metrics = lines(role + "/steps.jsonl")
    need([item["global_step"] for item in metrics] == steps and all(item["ordinal"] == (item["global_step"]-1) % 20
         for item in metrics), "Physical capture drops or changes a training step")
    return worker, metrics


def verify_pairs(a, b):
    names = set(a[0]["tensor_boundaries"]["preclip"])
    need(len(names) == 476, "Named trainable parameter denominator differs")
    for lhs, rhs in zip(a, b, strict=True):
        for field in ("global_step", "ordinal", "input_identity", "operands", "loss", "preclip_norm"):
            need(lhs[field] == rhs[field], "A paired input, scalar, or forward differs")
        need(math.isfinite(lhs["loss"]) and math.isfinite(lhs["preclip_norm"]) and lhs["preclip_norm"] > 0,
             "Invalid original loss or gradient norm")
        left, right = lhs["tensor_boundaries"], rhs["tensor_boundaries"]
        need(set(left) == set(right) == set(BOUNDARIES), "Missing captured training state")
        for boundary in BOUNDARIES:
            need(left[boundary] == right[boundary] and set(left[boundary]) == names
                 and all(re.fullmatch(r"[0-9a-f]{64}", digest) for digest in left[boundary].values()),
                 "Captured named gradients, adapters or AdamW states differ")
    return len(names)


def main():
    outcome = read("outcome.json")
    need(outcome["request"] == REQUEST and outcome["passed"] is True and outcome["post_clean"] is True
         and outcome["accounting_columns"] == COLUMNS, "Original controller terminal differs")
    listing = subprocess.check_output(
        ["sacct", "-nP", "-j", ",".join(IDS.values()), "--format", ",".join(COLUMNS)], timeout=12, text=True,
    )
    exact = {}
    for line in listing.splitlines():
        row = line.split("|")
        if len(row) == 7 and row[0] in IDS.values():
            need(row[0] not in exact, "Duplicate original Slurm accounting")
            exact[row[0]] = row
    need(set(exact) == set(IDS.values()), "Missing actual Slurm step")
    for role, ident in IDS.items():
        row = exact[ident]
        verify_step(role, row)
        saved = (read("admission-accounting.json")["row"] if role == "admission"
                 else outcome["observations"][role]["accounting"])
        need(row == saved, "Recorded and actual Slurm steps differ")
    before = read("pre.json")
    between, after = read("between.json"), read("post.json")
    for role, observed in (("pre", before), ("between", between), ("post", after)):
        need(observed["node"] == NODE and observed["step"] == IDS[role]
             and len(set(observed["PCIs"])) == 8 and observed["PCIs"] == before["PCIs"]
             and observed["boot_id_SHA256"] == before["boot_id_SHA256"]
             and observed["kfd_process_count"] == observed["own_device_fd_process_count"] == 0,
             "Independent cleanliness or device identity changed")
    continuous, a = verify_arm("continuous", 1)
    resumed, b = verify_arm("resume20", 21)
    need(continuous["worker_identity"] != resumed["worker_identity"]
         and continuous["physical_PCI"] == resumed["physical_PCI"] and continuous["physical_PCI"] in before["PCIs"],
         "Fresh process or same-card preservation differs")
    count = verify_pairs(a[20:], b)
    left, right = read("continuous/checkpoint-receipts.json"), read("resume20/checkpoint-receipts.json")
    restore = read("resume20/restore.json")
    need([x["global_step"] for x in left] == [10, 20, 25] and [x["global_step"] for x in right] == [25]
         and [x["next_data_cursor"] for x in left] == [10, 0, 5] and right[0]["next_data_cursor"] == 5,
         "Checkpoint denominator or data cursor changed")
    need(restore["restored_boundary_exact"] is True and restore["metadata"]["global_step"] == 20 and
         restore["next"]["global_step"] == 21 and restore["next"]["sample_cursor"] == 0
         and restore["checkpoint_SHA256"] == left[1]["checkpoint_SHA256"], "Fresh worker used another checkpoint")
    checkpoint = ROOT / "continuous/checkpoint-20.private.pt"
    need(checkpoint.resolve(strict=True) == checkpoint and checkpoint.stat().st_size == left[1]["bytes"]
         and 0 < left[1]["bytes"] < 128 * 1024**2, "Unexpected original checkpoint path or extent")
    with checkpoint.open("rb") as stream:
        actual_cp_sha = file_digest(stream, "sha256").hexdigest()
    need(actual_cp_sha == restore["checkpoint_SHA256"], "Resume checkpoint on disk differs")
    fields = ("adapter_SHA256", "state_SHA256", "optimizer_SHA256", "diffusion_scheduler_SHA256", "rng_SHA256",
              "integrity_SHA256")
    need(all(left[-1][key] == right[0][key] for key in fields), "Final checkpoint state identities differ")
    original_comparison = read("comparison.json")
    need(FILE_PINS["comparison.json"]["SHA256"] == outcome["comparison_SHA256"]
         and original_comparison["passed"] is True, "Original comparison receipt differs")
    result = {
        "observed_utc": datetime.now(timezone.utc).isoformat(), "request": REQUEST, "node": NODE,
        "state": "NATIVE_HIP_25_PLUS_5_AND_ALL_NAMED_BOUNDARIES_EXACT",
        "accounting_columns": COLUMNS, "accounting": {role: exact[ident] for role, ident in IDS.items()},
        "continuous_updates": len(a), "fresh_process_updates": len(b), "per_boundary_named_tensors": count,
        "paired_update_steps": [21, 22, 23, 24, 25], "exact_boundaries": list(BOUNDARIES),
        "changed_tensor_identities": 0, "same_platform_restart_passed": True,
        "checkpoint_20": {"bytes": left[1]["bytes"], "SHA256": actual_cp_sha},
        "checkpoint_state_fields_exact": list(fields), "files": FILE_PINS, "post_clean": True,
        "formal_stage3": False, "formal_performance": False, "matched_A100_and_quality_qualified": False,
        "tensor_payloads_transferred": False,
    }
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
