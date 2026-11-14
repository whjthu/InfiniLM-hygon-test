"""Validate the staged Qwen framework within the target CPU-only step."""
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    stage = Path("/stage")
    framework = Path("/framework")
    out = Path("/evidence")
    sys.path.insert(0, str(stage))
    sys.path.insert(0, str(framework))
    deadline = float(os.environ["QWEN_CPU_DEADLINE"])
    require(0 < deadline - time.time() <= 210, "Invalid CPU deadline")
    signal.setitimer(signal.ITIMER_REAL, deadline - time.time())
    require(os.environ["SLURM_JOB_ID"] == "1087656" and
            socket.gethostname().split(".")[0] == "a01r4n16", "Unexpected Slurm placement")
    require(all(os.environ[x] == "-1" for x in (
        "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")), "Devices visible")
    from vlm.runtime.swift_entry import resolve_cli
    from recipe import prepare
    import torch
    from vlm.runtime.initial_trainables import restore_initial_trainables

    ports = 29895
    original = (stage / "original-launch.json").read_bytes()
    binding_path = stage / "QWEN_TARGET_NATIVE_BINDING.json"
    binding = json.loads(binding_path.read_bytes())
    entry = framework / "vlm/runtime/swift_entry.py"
    launch = prepare(original, binding_path, entry, framework, out, ports,
                     epoch=deadline - 230, deadline=deadline, max_seconds=230)
    assets = launch["required_assets"]
    model = Path(assets["model_directory"])
    require(len(json.loads((model / "model.safetensors.index.json").read_bytes())["weight_map"]) == 750,
            "Unexpected model index")
    require(Path(assets["dataset"]).is_file() and all(Path(x).is_file() for x in assets["plugins"]),
            "Training input or plugin missing inside container")
    require(resolve_cli().__module__ == "swift.cli.main" and torch.version.hip is not None and
            torch.cuda.device_count() == 0, "Unexpected Swift or HIP provider")

    env = os.environ.copy()
    for name in ("NPROC_PER_NODE", "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
                 "GROUP_RANK", "NNODES", "NODE_RANK", "MASTER_ADDR"):
        env.pop(name, None)
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", WANDB_MODE="offline")
    help_run = subprocess.run([sys.executable, "-B", str(entry), "sft", "--help"],
                              env=env, capture_output=True, timeout=min(90, deadline-time.time()-20))
    (out / "swift-help.stdout").write_bytes(help_run.stdout)
    (out / "swift-help.stderr").write_bytes(help_run.stderr)
    require(help_run.returncode == 0 and all(flag in help_run.stdout for flag in (
        b"--model", b"--max_steps", b"--lora_dropout", b"--gradient_accumulation_steps",
        b"--external_plugins", b"--callbacks")), "Canonical Swift SFT help check failed")

    manifest = binding["trainable_manifest"]
    require(len(manifest) == 504, "Unexpected trainable inventory")
    named = [(row["name"], torch.nn.Parameter(torch.empty(
        row["shape"], dtype=getattr(torch, row["dtype"].removeprefix("torch.")), device="cpu")))
        for row in manifest]
    optimizer = torch.optim.AdamW([param for _, param in named])
    initial = binding["initial_state"]
    proof = restore_initial_trainables(named, optimizer, global_step=0, path=initial["path"],
        file_bytes=initial["bytes"], file_SHA256=initial["SHA256"],
        named_SHA256=binding["initial_trainables_SHA256"])
    result = dict(passed=True, step="1087656."+os.environ["SLURM_STEP_ID"],
        node=socket.gethostname().split(".")[0], torch_version=str(torch.__version__), hip=torch.version.hip,
        swift_help_SHA256=sha(help_run.stdout), model_index_tensors=750, initial_cpu_restore=proof,
        actual_model_loaded=False, training_updates=0, GPU_cards=0)
    with (out / "compute.json").open("x") as stream:
        json.dump(result, stream, indent=2)


if __name__ == "__main__":
    main()
