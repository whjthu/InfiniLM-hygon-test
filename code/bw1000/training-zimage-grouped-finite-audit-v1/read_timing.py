"""Check real paired finite-audit timings and all matched training boundary records."""

from hashlib import sha256
import json
import math
from statistics import median


BASELINE = "training-zimage-COCO20-recovery-boundaries-hip-v2"
BASELINE_STEPS_SHA = "37826ff6283ffb39afac1f7aaadb834af7f5bf855b56e47abd6488ee39ca78f0"
BASELINE_WORKER_SHA = "b27f88a13bbaa64912d29c700f00d527860f3b86f55c7b0c648a4c3b45b43c19"
UPDATE_STEPS = list(range(1, 26))
PAIRED_STEPS = (6, 7, 8)
STAGES = ("preclip", "postclip", "updated")


def read(path, limit, expected=None):
    if path.is_symlink() or not 0 < path.stat().st_size <= limit:
        raise ValueError("Timing evidence is outside the fixed file scope")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if not 0 < len(raw) <= limit or (expected is not None and sha256(raw).hexdigest() != expected):
        raise ValueError("Original timing evidence hash differs")
    return raw


def lines(raw):
    return [json.loads(line) for line in raw.splitlines()]


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def verify_continuous(root, request, step):
    before = root.parent / BASELINE / "continuous"
    current = root / "continuous"
    original_raw = read(before / "steps.jsonl", 2**21, BASELINE_STEPS_SHA)
    current_raw = read(current / "steps.jsonl", 2**21)
    original_rows = lines(original_raw)
    actual_rows = lines(current_raw)
    if [r["global_step"] for r in actual_rows] != UPDATE_STEPS:
        raise ValueError("Actual native trajectory is incomplete")
    original = json.loads(read(before / "worker-result.json", 16384, BASELINE_WORKER_SHA))
    worker = json.loads(read(current / "worker-result.json", 16384))
    if (worker["request"] != request or worker["step"] != step or worker["optimizer_returned_steps"] != UPDATE_STEPS
            or worker["optimizer_intended_steps"] != UPDATE_STEPS or worker["actual_optimizer_updates"] != 25
            or worker["passed"] is not True or worker["initial_adapter_hashes_SHA256"] != original["initial_adapter_hashes_SHA256"]
            or worker["trainable_tensors"] != 476):
        raise ValueError("Original optimizer, input and trainable scope differs")
    if [r["global_step"] for r in original_rows] != UPDATE_STEPS:
        raise ValueError("Baseline native trajectory differs")
    differences = [a["global_step"] for a, b in zip(actual_rows, original_rows, strict=True) if a != b]
    timing_raw = read(current / "timing.jsonl", 65536)
    timing = lines(timing_raw)
    if [item["global_step"] for item in timing] != UPDATE_STEPS:
        raise ValueError("Native timing row denominator differs")
    pair_rows, checkpoints = [], []
    for row in timing:
        step_id = row["global_step"]
        pairs = row["paired_finite_checks"]
        if not positive(row["native_update_seconds_including_witness"]) or not positive(row["worker_step_seconds"]):
            raise ValueError("Native measurement is not finite and positive")
        checkpoint = row["checkpoint_seconds"]
        if step_id in (10, 20, 25):
            if not positive(checkpoint):
                raise ValueError("Checkpoint timing is missing")
            checkpoints.append({"global_step": step_id, "seconds": checkpoint})
        elif checkpoint is not None:
            raise ValueError("Unexpected checkpoint measurement")
        if step_id in PAIRED_STEPS:
            if set(pairs) != set(STAGES):
                raise ValueError("Paired finite checking stages differ")
            order = ["grouped", "original"] if step_id == 7 else ["original", "grouped"]
            for stage, item in sorted(pairs.items()):
                costs = item["seconds"]
                if (item["count"] != 476 or item["identical"] is not True or item["order"] != order
                        or set(costs) != {"grouped", "original"} or not all(positive(t) for t in costs.values())):
                    raise ValueError("Paired finite verdict or scope differs")
                pair_rows.append({"global_step": step_id, "phase": stage, **costs})
        elif pairs:
            raise ValueError("Unexpected duplicate original audit arm")
    if len(pair_rows) != 9:
        raise ValueError("Actual paired audit denominator differs")
    averages = {stage: median(r["grouped"] / r["original"] for r in pair_rows if r["phase"] == stage)
                for stage in STAGES}
    return {
        "passed": not differences, "formal_performance": False, "matched_optimizer_steps": 25 - len(differences),
        "trajectory_difference_steps": differences, "paired_finite_checks": pair_rows,
        "paired_median_grouped_over_original": averages, "checkpoints": checkpoints,
        "timing_SHA256": sha256(timing_raw).hexdigest(), "native_steps_SHA256": sha256(current_raw).hexdigest(),
        "original_steps_SHA256": BASELINE_STEPS_SHA,
        "worker_training_section_seconds": worker["training"]["worker_training_section_seconds"],
        "original_worker_training_section_seconds": original["training"]["worker_training_section_seconds"],
    }
