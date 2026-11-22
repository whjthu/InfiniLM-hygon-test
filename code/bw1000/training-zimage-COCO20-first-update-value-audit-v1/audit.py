"""Audit Z-Image COCO20 first-update captures on CPU, using original bytes in place."""

import argparse
import hashlib
import importlib
import io
import json
import math
import os
from pathlib import Path
import re


FILES = frozenset({
    "initial.pt", "initial_optimizer.pt", "input.pt", "prepared.pt", "target.pt",
    "forward.pt", "preclip.pt", "postclip.pt", "clip_norm.pt", "updated.pt",
    "optimizer.pt", "optimizer_groups.pt",
})
PARAMETERS = 476
ELEMENTS = 9_922_560
LATENT_SHAPE = (1, 16, 32, 32)
MAX_CAPTURE_BYTES = 256 << 20
A100_PROGRESS = "bb9021ac253361bf3cfd396d1bc491ac5d034be493f8f695dec45afaef439af8"
PROMPT_SHA256 = "2cc36f2ff7893972c8f75fcf128df6e7d22e2296e162ab176a61f7e0a8808e96"
FRAMEWORK_SHA256 = "aeb21195bb7bfb64f36ea4299e52b359a695d986d4556a70f0b1c5c3915de8d7"


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def pinned_bytes(path, expected, limit, *, size=None):
    require(bool(re.fullmatch(r"[0-9a-f]{64}", expected))
            and path.is_file() and not path.is_symlink() and path.resolve() == path,
            f"{path.name}: expected a canonical file with a SHA-256 pin")
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        require(0 < before.st_size <= limit and (size is None or before.st_size == size),
                f"{path.name}: invalid file extent")
        raw = stream.read(before.st_size + 1)
        after = os.fstat(stream.fileno())
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    require(identity(before) == identity(after) == identity(path.stat()) and len(raw) == before.st_size
            and digest(raw) == expected, f"{path.name}: file bytes differ")
    return raw


def fields(value, names, stage):
    require(type(value) is dict and value.keys() == names, f"{stage}: unexpected fields")
    return value


def tensor(torch, value, stage, *, shape=None, dtype=None):
    require(isinstance(value, torch.Tensor) and value.device.type == "cpu"
            and value.layout == torch.strided and value.numel() > 0
            and (shape is None or tuple(value.shape) == tuple(shape))
            and (dtype is None or value.dtype == dtype) and bool(torch.isfinite(value).all()),
            f"{stage}: tensor extent, dtype, device or finite values differ")
    return value


def scalar(torch, value, stage):
    return float(tensor(torch, value, stage).reshape(()))


def norm(torch, values):
    terms = []
    for value in values:
        x = value.to(torch.float64).reshape(-1)
        terms.append(float(torch.dot(x, x)))
    result = math.sqrt(math.fsum(terms))
    require(math.isfinite(result), "CPU norm reduction overflowed")
    return result


def audit(torch, tree_sha, root, progress_sha, arm):
    expected = {"HIP": ("HIP_COCO20_FIRST_UPDATE_RETURNED", 1), "A100": ("NATIVE25_RETURNED", 25)}
    phase, steps = expected[arm]
    progress = json.loads(pinned_bytes(root / "progress.json", progress_sha, 64 << 10))
    require(progress["phase"] == phase and progress["first_update_complete"] is True
            and progress["planned_global_steps"] == progress["completed_global_steps"] == steps
            and progress["failed"] is False, f"{arm}: original first-update phase is incomplete")
    pins = progress["files"]
    require({p["name"] for p in pins} == FILES and len(pins) == len(FILES)
            and all(type(f["bytes"]) is int and f["bytes"] > 0 for f in pins),
            f"{arm}: twelve distinct capture files required")
    total = sum(f["bytes"] for f in pins)
    require(total == progress["artifact_bytes"] and total <= MAX_CAPTURE_BYTES, f"{arm}: capture bound exceeded")
    saved = {}
    for pin in pins:
        raw = pinned_bytes(root / pin["name"], pin["SHA256"], MAX_CAPTURE_BYTES, size=pin["bytes"])
        saved[pin["name"][:-3]] = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)

    initial = saved["initial"]
    names = list(initial)
    require(type(initial) is dict and len(names) == PARAMETERS
            and all(type(n) is str and n.startswith("pipe.dit.") and ".lora_" in n for n in names),
            f"{arm}: unexpected trainable names")
    for name, value in initial.items():
        tensor(torch, value, "initial", dtype=torch.bfloat16)
    require(sum(v.numel() for v in initial.values()) == ELEMENTS, f"{arm}: parameter element count differs")
    for stage in ("updated", "preclip", "postclip", "optimizer"):
        require(type(saved[stage]) is dict and list(saved[stage]) == names, f"{arm}: {stage} names or order differs")
    state = fields(saved["initial_optimizer"], {"state", "param_groups"}, "initial optimizer")
    require(type(state["state"]) is dict and not state["state"] and type(state["param_groups"]) is list
            and len(state["param_groups"]) == 1, f"{arm}: optimizer was not fresh")
    group = state["param_groups"][0]
    require(group["params"] == list(range(PARAMETERS)) and group["lr"] == 1e-4 and group["weight_decay"] == .01
            and group["betas"] == (.9, .999) and group["eps"] == 1e-8 and group["amsgrad"] is False
            and tree_sha(state["param_groups"]) == tree_sha(saved["optimizer_groups"]),
            f"{arm}: AdamW group identity or trainable order differs")
    deltas, first_moment, second_moment = {}, {}, {}
    changed = 0
    for name in names:
        p = initial[name]
        for stage in ("updated", "preclip", "postclip"):
            tensor(torch, saved[stage][name], stage, shape=p.shape, dtype=p.dtype)
        moment = fields(saved["optimizer"][name], {"step", "exp_avg", "exp_avg_sq"}, "AdamW state")
        require(scalar(torch, moment["step"], "AdamW step") == 1, f"{arm}: expected exactly one optimizer update")
        first_moment[name] = tensor(torch, moment["exp_avg"], "AdamW first moment", shape=p.shape, dtype=p.dtype)
        second_moment[name] = tensor(torch, moment["exp_avg_sq"], "AdamW second moment", shape=p.shape, dtype=p.dtype)
        require(bool((second_moment[name] >= 0).all()), f"{arm}: AdamW squared moment became negative")
        delta = saved["updated"][name].to(torch.float64) - p.to(torch.float64)
        deltas[name] = delta
        changed += not torch.equal(saved["updated"][name], p)

    observation = fields(saved["input"], {"batch", "RNG", "input_SHA256"}, "raw input")
    batch = observation["batch"]
    require(isinstance(batch, dict) and type(batch.get("prompt")) is str and digest(batch["prompt"].encode()) == PROMPT_SHA256
            and tree_sha(batch) == observation["input_SHA256"], f"{arm}: original row or input hash differs")
    image = fields(batch["image"], {"kind", "mode", "size", "decoded_pixels_SHA256"}, "decoded RGB")
    require(image["kind"] == "PIL.Image" and image["mode"] == "RGB" and image["size"] == [256, 256]
            and re.fullmatch(r"[a-f0-9]{64}", image["decoded_pixels_SHA256"]),
            f"{arm}: original native image extent differs")
    prepared = fields(saved["prepared"], {"inputs", "noise", "timestep_id", "timestep"}, "prepared")
    values = prepared["inputs"]
    require(type(values) is dict, f"{arm}: original native prepared input map missing")
    for tag, value in (("noise", prepared["noise"]), ("clean latents", values["input_latents"]),
                       ("noised latents", values["latents"])):
        tensor(torch, value, tag, shape=LATENT_SHAPE)
    index = tensor(torch, prepared["timestep_id"], "timestep id", shape=(1,), dtype=torch.int64)
    require(int(index.item()) == 726 and 0 <= int(index.item()) < 1000,
            f"{arm}: original COCO20 timestep differs")
    tensor(torch, prepared["timestep"], "cast timestep", shape=(1,))
    target = fields(saved["target"], {"training_target"}, "training target")["training_target"]
    forward = fields(saved["forward"], {"noise_prediction", "loss"}, "forward")
    prediction = tensor(torch, forward["noise_prediction"], "prediction", shape=LATENT_SHAPE)
    tensor(torch, target, "target", shape=LATENT_SHAPE)
    loss = scalar(torch, forward["loss"], "loss")
    clip_norm = scalar(torch, saved["clip_norm"], "gradient clip input norm")
    norms = {name: norm(torch, group.values()) for name, group in (
        ("gradient_before_clip", saved["preclip"]), ("gradient_after_clip", saved["postclip"]),
        ("parameter_update", deltas), ("adamw_exp_avg", first_moment), ("adamw_exp_avg_sq", second_moment),
    )}
    require(loss >= 0 and clip_norm > 0 and changed > 0 and all(value > 0 for value in norms.values()),
            f"{arm}: first-step training state failed finite/nonzero boundary")
    exact_before = {
        "batch": tree_sha(batch), "decoded_RGB256": image["decoded_pixels_SHA256"],
        "initial_parameters": tree_sha(initial), "initial_optimizer": tree_sha(state),
        "trainable_order": tree_sha(names), "prepared_inputs": tree_sha(values),
        "clean_latents": tree_sha(values["input_latents"]), "noised_latents": tree_sha(values["latents"]),
        "noise": tree_sha(prepared["noise"]), "timestep_index": tree_sha(index),
        "timestep_value": tree_sha(prepared["timestep"]), "training_target": tree_sha(target),
    }
    summary = {"arm": arm, "capture_progress_SHA256": progress_sha, "files_decoded": len(saved),
               "capture_bytes": total, "trainable_tensors": PARAMETERS, "trainable_elements": ELEMENTS,
               "changed_trainable_tensors": changed, "optimizer_steps": 1, "optimizer_LR": group["lr"],
               "clip_norm_saved": clip_norm, "loss": loss, "timestep_id": int(index.item()), "norms": norms,
               "exact_before": exact_before, "rng_snapshot_SHA256": tree_sha(observation["RNG"]),
               "prediction_SHA256": tree_sha(prediction), "updated_parameters_SHA256": tree_sha(saved["updated"]),
               "optimizer_state_SHA256": tree_sha(saved["optimizer"])}
    vectors = {"prediction": {"prediction": prediction}, "gradient_before_clip": saved["preclip"],
               "gradient_after_clip": saved["postclip"], "parameter_update": deltas,
               "adamw_exp_avg": first_moment, "adamw_exp_avg_sq": second_moment}
    return summary, vectors


def compare(torch, left, right):
    current, current_vectors = left
    reference, reference_vectors = right
    checks = {k: value == reference["exact_before"][k] for k, value in current["exact_before"].items()}
    report = {"exact_checks": checks, "same_realized_inputs_and_initial_state": all(checks.values()),
              "original_numerical_thresholds_applied": False, "formal_numerics": False}
    if not report["same_realized_inputs_and_initial_state"]:
        return report
    metrics = {}
    for stage, expected in reference_vectors.items():
        actual = current_vectors[stage]
        require(list(actual) == list(expected), f"{stage}: paired inventory differs")
        for name, t in expected.items():
            require(actual[name].shape == t.shape and actual[name].dtype == t.dtype, f"{stage}: paired shape or dtype differs")
        error = norm(torch, (actual[k].to(torch.float64) - expected[k].to(torch.float64) for k in expected))
        scale = norm(torch, expected.values())
        metrics[stage] = {"difference_l2": error, "A100_l2": scale,
                          "relative_l2": error / scale if scale else None}
    for stage in ("loss", "clip_norm_saved"):
        a, b = current[stage], reference[stage]
        metrics[stage] = {"HIP": a, "A100": b, "absolute_error": abs(a-b),
                          "relative_error": abs(a-b) / abs(b) if b else None}
    report["metrics"] = metrics
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hip", type=Path, required=True)
    p.add_argument("--hip-progress-sha256", required=True)
    p.add_argument("--a100", type=Path)
    p.add_argument("--framework-file", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    require(not args.output.exists(), "Existing audit output")
    require(all(os.environ.get(k) == "" for k in
                ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")),
            "CPU audit requires all accelerator visibility masks empty")
    framework = args.framework_file.resolve(strict=True)
    pinned_bytes(framework, FRAMEWORK_SHA256, 64 << 10, size=27542)
    provider = importlib.import_module("gencuda.hy_resume_checkpoint")
    require(Path(provider.__file__).resolve() == framework, "Tensor fingerprint provider escaped the approved source")
    import torch

    require(not torch.cuda.is_initialized(), "An accelerator initialized during CPU audit import")
    torch.set_num_threads(4)
    actual = audit(torch, provider.tree_sha256, args.hip.resolve(strict=True), args.hip_progress_sha256, "HIP")
    report = {"case": "T07.sft", "HIP": actual[0], "A100": None, "paired": None,
              "formal_numerics": False, "torch": str(torch.__version__)}
    if args.a100 is not None:
        expected = audit(torch, provider.tree_sha256, args.a100.resolve(strict=True), A100_PROGRESS, "A100")
        report.update(A100=expected[0], paired=compare(torch, actual, expected))
    require(not torch.cuda.is_initialized(), "An accelerator initialized during CPU audit")
    report["accelerator_initialized"] = False
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(report, out, indent=2, allow_nan=False)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())
    print(json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    main()
