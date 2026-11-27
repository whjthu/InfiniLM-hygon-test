"""One independent CPU readback of the generated checkpoint and saved trainables."""
import argparse
import json
import os
from pathlib import Path
import re
import socket
import sys

from scope_contract import audit_ranks, need
from render_worker import check_scope


def verify(run, scope_sha, file_pin, source_read, loader):
    need(file_pin(run / "off-scope.json")["SHA256"] == scope_sha, "OFF_RELOAD_SCOPE_SHA")
    scope = check_scope(source_read(run / "off-scope.json"))
    need(run.name == scope["label"], "OFF_RELOAD_SCOPE_PATH")
    result = source_read(run / "off-first-two.json")
    records = [source_read(run / "continuous/train/minicpm-usgs-witness" / f"rank-{i:02d}" / "result.json") for i in range(8)]
    expected = audit_ranks(records, scope["request"])
    need(all(result[name] == value for name, value in expected.items()), "OFF_RELOAD_RANK_AGREEMENT")
    inventory = run / "continuous-checkpoint-2-files.json"
    need(file_pin(inventory) == result["checkpoint_inventory"], "OFF_RELOAD_INVENTORY_FILE")
    record = source_read(inventory)
    names = {"trainer_state.json", "args.json", "adapter_model.safetensors", "optimizer.pt", "scheduler.pt", "adapter_config.json"}
    names |= {f"rng_state_{i}.pth" for i in range(8)}
    root = run / "continuous/train/checkpoint-2"
    need(root.resolve(strict=True) == root and not root.is_symlink() and record["global_step"] == 2
         and record["member_count"] == len(names) == 14 and set(record["files"]) == names, "OFF_CP2_INVENTORY")
    for name, pin in record["files"].items():
        need(file_pin(root / name) == pin, "OFF_CP2_MEMBER")
    state = source_read(root / "trainer_state.json")
    need(state["global_step"] == 2 and state["max_steps"] == 25 and state["epoch"] == 2/25, "OFF_CP2_TRAINER_BOUNDARY")
    adapter, optimizer, scheduler = loader(root)
    need(adapter == expected["adapter"] and optimizer["slots"] == expected["adapter"]["tensors"]
         and optimizer["steps"] == [2] and scheduler["last_epoch"] == 2, "OFF_CP2_RELOAD_STATE")
    for name in ("adapter_model.safetensors", "optimizer.pt", "scheduler.pt"):
        need(file_pin(root / name) == record["files"][name], "OFF_CPU_RELOAD_FILE_CHANGED")
    return dict(request=scope["request"], planned_native_forwards=64, native_forwards=64,
                planned_optimizer_updates=2, actual_optimizer_updates=2, adapter=adapter, optimizer=optimizer,
                scheduler=scheduler, checkpoint_members=14, rng_members_hashed=8,
                full_training_resume_run=False, device_restore_verified=False, actual_GPU=0,
                state="OFF_NATIVE_CP2_INDEPENDENT_CPU_RELOAD_PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--scope-sha", required=True)
    args = parser.parse_args()
    need(re.fullmatch(r"[0-9a-f]{64}", args.scope_sha) and args.run.resolve(strict=True) == args.run, "OFF_CP2_ARGS")
    raw_scope = (args.run / "off-scope.json").read_bytes()
    from scope_contract import sha
    need(sha(raw_scope) == args.scope_sha, "OFF_CP2_SCOPE_DIGEST")
    scope = check_scope(json.loads(raw_scope))
    need(os.environ.get("SLURM_JOB_ID") == scope["parent"] and re.fullmatch(r"[0-9]+", os.environ.get("SLURM_STEP_ID", ""))
         and socket.gethostname().split(".")[0] == scope["node"], "OFF_CPU_RELOAD_ACTUAL_STEP")
    need(all(os.environ.get(k) == "-1" for k in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"))
         and os.environ.get("SLURM_STEP_GPUS") in (None, "", "(null)")
         and os.environ.get("SLURM_GPUS_ON_NODE") in (None, "", "0"), "OFF_CP2_ZERO_GRES_VISIBILITY")
    sys.path.insert(0, str(args.run / "target-worker"))
    from worker import file_pin, read
    sys.path.insert(0, str(args.run / "framework"))

    def loader(root):
        import torch
        from safetensors.torch import load_file
        from vlm.runtime.first_update_audit import fingerprint

        def no_gpu(*unused, **ignored):
            raise RuntimeError("OFF_CP2_GPU_INITIALIZATION")

        torch.cuda._lazy_init = no_gpu
        adapter = fingerprint(load_file(str(root / "adapter_model.safetensors"), device="cpu"))
        opt = torch.load(root / "optimizer.pt", map_location="cpu", weights_only=True)
        steps = []
        for item in opt["state"].values():
            step = item["step"]
            if isinstance(step, torch.Tensor):
                need(step.device.type == "cpu" and step.numel() == 1, "OFF_CP2_OPTIMIZER_DEVICE")
                step = float(step.item())
            need(step == 2, "OFF_CP2_OPTIMIZER_STEP")
            for value in item.values():
                if isinstance(value, torch.Tensor):
                    need(value.device.type == "cpu" and (not value.is_floating_point() or bool(torch.isfinite(value).all())),
                         "OFF_CP2_OPTIMIZER_TENSOR")
            steps.append(2)
        schedule = torch.load(root / "scheduler.pt", map_location="cpu", weights_only=True)
        return adapter, dict(slots=len(steps), steps=sorted(set(steps))), dict(last_epoch=schedule["last_epoch"])

    import signal
    signal.signal(signal.SIGALRM, lambda *_: sys.exit("OFF_CP2_RELOAD_DEADLINE"))
    signal.alarm(120)
    print(json.dumps(verify(args.run, args.scope_sha, file_pin, read, loader), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
