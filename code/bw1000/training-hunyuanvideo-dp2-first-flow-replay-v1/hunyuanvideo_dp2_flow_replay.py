"""Sealed first-flow capture on CUDA and rank-local replay for DP2 diagnostics."""

import argparse
from hashlib import sha256
import io
import json
from pathlib import Path
import re

from . import hunyuanvideo_resume_checkpoint as state


LAUNCH_SHA256 = "181b1ca706a7d62741298192c3391a0336acf0c60624aa9f2645e90a55934a6f"
SCHEMA = "hunyuanvideo-dp2-original-first-flow-v1"
MAX_TENSOR_BYTES = 32 << 20


def need(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def pinned(path, expected, maximum):
    path = Path(path)
    need(not path.is_symlink(), f"unexpected link: {path.name}")
    with path.open("rb") as handle:
        raw = handle.read(maximum + 1)
    need(0 < len(raw) <= maximum and digest(expected) and sha256(raw).hexdigest() == expected,
         f"sealed bytes differ: {path.name}")
    return raw


def validate_plan(plan):
    need(plan["launch_SHA256"] == LAUNCH_SHA256 and digest(plan["cache_SHA256"])
         and digest(plan["initial_adapter_SHA256"])
         and type(plan["seed"]) is int and 0 <= plan["seed"] < 2**32 - 1,
         "original A100 launch or initial state binding differs")
    ranks = plan["expected_flow_SHA256"]
    need(set(ranks) == {"0", "1"} and all(digest(value) for value in ranks.values())
         and ranks["0"] != ranks["1"], "expected both distinct original rank inputs")
    return plan


def validate_binding(config):
    binding = config["first_flow_replay"]
    need(config.get("resume_from_checkpoint") is None and isinstance(binding, dict)
         and set(binding) == {"manifest", "manifest_SHA256", "plan", "plan_SHA256"},
         "DP2 first-flow replay requires a fresh run with a sealed input binding")
    need(all(isinstance(binding[key], str) and Path(binding[key]).is_absolute()
             for key in ("manifest", "plan")) and all(digest(binding[key]) for key in ("manifest_SHA256", "plan_SHA256")),
         "DP2 first-flow binding paths or hashes differ")


def replay_inputs(torch, config, rank, original_draw, device, initial_adapter):
    validate_binding(config)
    needs = config["first_flow_replay"]
    manifest_path = Path(needs["manifest"])
    plan = validate_plan(json.loads(pinned(needs["plan"], needs["plan_SHA256"], 16384)))
    receipt = json.loads(pinned(manifest_path, needs["manifest_SHA256"], 16384))
    need(receipt["schema"] == SCHEMA and receipt["case"] == "T09.dp2"
         and receipt["plan_SHA256"] == needs["plan_SHA256"] and receipt["launch_SHA256"] == LAUNCH_SHA256
         and receipt["optimizer_updates"] == 0
         and receipt["cache_SHA256"] == plan["cache_SHA256"] == config["training_cache_sha"]
         and receipt["seed"] == plan["seed"] == config["seed"]
         and initial_adapter == plan["initial_adapter_SHA256"],
         "capture provenance or live DP2 input prerequisites differ")
    need(type(rank) is int and rank in (0,1) and set(receipt["ranks"]) == {"0", "1"}, "unexpected replay rank set")
    entry = receipt["ranks"][str(rank)]
    file = f"rank-{rank}-first-flow.pt"
    need(entry["file"] == file and entry["input_SHA256"] == plan["expected_flow_SHA256"][str(rank)],
         "rank replay receipt differs from the original A100 fingerprint")
    raw = pinned(manifest_path.parent / file, entry["SHA256"], MAX_TENSOR_BYTES)
    need(len(raw) == entry["bytes"], "rank replay byte count differs")
    record = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    need(type(record) is tuple and len(record) == len(original_draw) == 3
         and all(torch.is_tensor(actual) and actual.shape == expected.shape and actual.dtype == expected.dtype
                 and actual.device.type == "cpu" and bool(torch.isfinite(actual).all())
                 for actual, expected in zip(record, original_draw, strict=True)),
         "rank replay tensor metadata or values differ")
    need(state.tree_sha256(record) == entry["input_SHA256"], "rank input tensor content differs")
    return tuple(item.to(device=device) for item in record)


def write_update_artifact(torch, directory, phase, payload):
    need(phase in ("before", "forward", "preclip", "postclip", "updated"), "unexpected DP2 update phase")
    buffer = io.BytesIO()
    torch.save(state._cpu_tree(payload), buffer)
    raw = buffer.getvalue()
    need(0 < len(raw) <= (64 << 20), "DP2 update artifact extent differs")
    output = directory / ("first-update-" + phase + ".pt")
    with output.open("xb") as handle:
        handle.write(raw)
    return {"path": str(output), "sha256": sha256(raw).hexdigest(), "bytes": len(raw)}


def capture(torch, native, plan, launch, cache_path, output, *, device_index, plan_sha):
    config = launch["config"]
    need(config["parallel_mode"] == "data_parallel" and config["world_size"] == 2 and config["sp"] == 1
         and config["seed"] == plan["seed"] and config["training_cache_sha"] == plan["cache_SHA256"],
         "original DP2 recipe differs")
    need(torch.version.cuda is not None and torch.version.hip is None and torch.cuda.is_available()
         and 0 <= device_index < torch.cuda.device_count(), "capture requires a visible original-class CUDA device")
    device = torch.device(f"cuda:{device_index}")
    name = torch.cuda.get_device_properties(device).name
    need(name == "NVIDIA A100-SXM4-40GB", "capture requires the A100 reference device class")
    need(not output.exists(), "capture output already exists")
    cache_raw = pinned(cache_path, plan["cache_SHA256"], 64 << 20)
    cache = torch.load(io.BytesIO(cache_raw), map_location="cpu", weights_only=True)
    native.validate_hunyuanvideo_cache(cache)
    need(all(bool(torch.isfinite(cache[key].float()).all()) for key in native.REQUIRED_CACHE_SHAPES),
         "non-finite original training cache")
    torch.cuda.set_device(device)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    values = native.prepare_native_cache(torch, cache, device=device)
    payloads, ranks = {}, {}
    for rank in (0, 1):
        seed = plan["seed"] + rank
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        data = state._cpu_tree(native.sample_native_flow_inputs(torch, values, config, device=device))
        actual = state.tree_sha256(data)
        need(actual == plan["expected_flow_SHA256"][str(rank)],
             f"rank {rank} regenerated input differs from the archived original")
        buffer = io.BytesIO()
        torch.save(data, buffer)
        raw = buffer.getvalue()
        need(0 < len(raw) <= MAX_TENSOR_BYTES, "capture tensor extent differs")
        file = f"rank-{rank}-first-flow.pt"
        payloads[file] = raw
        ranks[str(rank)] = {"file": file, "bytes": len(raw), "SHA256": sha256(raw).hexdigest(), "input_SHA256": actual}
    output.mkdir(mode=0o700)
    for file, raw in payloads.items():
        with (output / file).open("xb") as handle:
            handle.write(raw)
    receipt = {"schema": SCHEMA, "case": "T09.dp2", "plan_SHA256": plan_sha, "launch_SHA256": LAUNCH_SHA256,
               "cache_SHA256": plan["cache_SHA256"], "seed": plan["seed"], "device_class": name, "ranks": ranks,
               "model_loads": 0, "optimizer_updates": 0}
    with (output / "manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--original-launch", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-index", type=int, required=True)
    args = parser.parse_args()
    plan = validate_plan(json.loads(pinned(args.plan, args.plan_sha256, 16384)))
    launch = json.loads(pinned(args.original_launch, LAUNCH_SHA256, 8192))
    import torch
    from .backends import hunyuanvideo_training as native

    capture(torch, native, plan, launch, args.cache, args.output,
            device_index=args.device_index, plan_sha=args.plan_sha256)


if __name__ == "__main__":
    main()
