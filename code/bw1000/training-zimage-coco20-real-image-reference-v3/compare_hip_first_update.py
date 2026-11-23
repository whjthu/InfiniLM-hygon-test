"""Independently pair the corrected and original full Hygon Z-Image first updates."""

import argparse
from hashlib import sha256
from io import BytesIO
import json
from math import fsum, isfinite, sqrt
import os
from pathlib import Path

ORIGINAL_PROGRESS = "a1694d2ffc3491d5c89d896cd73260f9ded215bd83e6ea5eba78e5b14403d2d0"
CAPTURE_FILES = {
    "initial.pt", "initial_optimizer.pt", "input.pt", "prepared.pt", "target.pt", "forward.pt",
    "preclip.pt", "postclip.pt", "clip_norm.pt", "updated.pt", "optimizer.pt", "optimizer_groups.pt",
}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def read(path, digest, limit, expected_size=None):
    require(path.is_file() and not path.is_symlink(), f"Missing archive member {path.name}")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    require(0 < len(data) <= limit and (expected_size is None or len(data) == expected_size)
            and sha256(data).hexdigest() == digest, f"Archive member differs at {path.name}")
    return data


def captured(torch, path, digest, phase):
    progress = json.loads(read(path / "progress.json", digest, 64 << 10))
    require(progress["case"] == "T07.sft" and progress["phase"] == phase
            and progress["first_update_complete"] is True and progress["failed"] is False
            and progress["planned_global_steps"] == progress["completed_global_steps"] == 1,
            "Actual native first update has not completed")
    pins = progress["files"]
    total = sum(pin["bytes"] for pin in pins)
    require(len(pins) == len(CAPTURE_FILES) and {pin["name"] for pin in pins} == CAPTURE_FILES
            and total == progress["artifact_bytes"] and total <= 256 << 20, "Capture inventory differs")
    return {pin["name"][:-3]: torch.load(BytesIO(read(path / pin["name"], pin["SHA256"], total, pin["bytes"])),
                                         map_location="cpu", weights_only=True) for pin in pins}


def summarize(torch, left, right):
    count = 0
    exact = 0
    errors = []
    reference = []
    max_errors = []
    for a, b in zip(left, right, strict=True):
        require(a.shape == b.shape and a.dtype == b.dtype and a.device.type == b.device.type == "cpu",
                "Paired tensor type or layout differs")
        require(bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()), "Non-finite paired tensor")
        count += a.numel()
        exact += int((a == b).sum())
        a64, b64 = a.to(torch.float64), b.to(torch.float64)
        diff = a64 - b64
        errors.append(float(diff.square().sum()))
        reference.append(float(b64.square().sum()))
        max_errors.append(float(diff.abs().max()))
    scale = sqrt(fsum(reference))
    error = sqrt(fsum(errors))
    return {"elements": count, "equal_elements": exact, "difference_l2": error,
            "original_l2": scale, "relative_l2": error / scale if scale else None,
            "max_absolute_error": max(max_errors)}


def compare(torch, tree_sha, corrected, original):
    fixed = {key: tree_sha(corrected[key]) == tree_sha(original[key])
             for key in ("initial", "initial_optimizer", "prepared", "target", "optimizer_groups")}
    fixed["raw_batch"] = tree_sha(corrected["input"]["batch"]) == tree_sha(original["input"]["batch"])
    require(all(fixed.values()), "Paired initial state or realized fixed operands differ")
    names = list(original["initial"])
    require(len(names) == 476 and sum(value.numel() for value in original["initial"].values()) == 9922560
            and names == list(corrected["initial"]), "Native trainable inventory differs")
    for source in (corrected, original):
        for group in ("preclip", "postclip", "updated", "optimizer"):
            require(list(source[group]) == names, f"Wrong trainable inventory in {group}")
        require(all(float(source["optimizer"][name]["step"]) == 1 for name in names),
                "Expected exactly one AdamW update for each parameter")

    values = {}
    values["prediction"] = summarize(torch, [corrected["forward"]["noise_prediction"]],
                                     [original["forward"]["noise_prediction"]])
    for key in ("preclip", "postclip", "updated"):
        values[key] = summarize(torch, corrected[key].values(), original[key].values())
    for key in ("exp_avg", "exp_avg_sq"):
        values[key] = summarize(torch, (corrected["optimizer"][name][key] for name in names),
                                (original["optimizer"][name][key] for name in names))
    def delta(arm):
        for name in names:
            yield arm["updated"][name].to(torch.float64) - arm["initial"][name].to(torch.float64)

    values["parameter_delta"] = summarize(torch, delta(corrected), delta(original))
    for key, a, b in (("loss", corrected["forward"]["loss"], original["forward"]["loss"]),
                      ("clip_norm", corrected["clip_norm"], original["clip_norm"])):
        values[key] = {"corrected": float(a), "original": float(b), "absolute_error": abs(float(a) - float(b))}
        require(all(isfinite(x) for x in values[key].values()), "Non-finite scalar comparison")
    return {"fixed_operands": fixed, "rng_snapshot_equal": tree_sha(corrected["input"]["RNG"]) ==
            tree_sha(original["input"]["RNG"]), "paired": values, "formal_cross_platform": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corrected", type=Path)
    parser.add_argument("corrected_progress_sha256")
    parser.add_argument("original", type=Path)
    args = parser.parse_args()
    require(all(os.environ.get(name) == "" for name in
                ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")), "Must use zero GRES")
    import torch
    from gencuda.hy_resume_checkpoint import tree_sha256
    torch.set_num_threads(4)
    a = captured(torch, args.corrected, args.corrected_progress_sha256, "FIXED_FIRST_UPDATE_RETURNED")
    b = captured(torch, args.original, ORIGINAL_PROGRESS, "HIP_COCO20_FIRST_UPDATE_RETURNED")
    report = compare(torch, tree_sha256, a, b)
    require(not torch.cuda.is_initialized(), "Unexpected accelerator initialization")
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
