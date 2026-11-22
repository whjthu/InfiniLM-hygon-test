"""Exec the common Swift entry in a ROOT-admitted eight-HIP-rank worker.

ROOT's qualified controller must own this PID/tree, RSS, deadline, Slurm
accounting and independent post. This module never launches Slurm/SSH.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import sys
import time

from fresh_contract import clock, fresh_config, need
from stable_io import read_exact, identity, pin, write_json
from stage import ROOT, D, PINS, TRAINING_REQUEST, verify_package

NODE = "f13r1n12"
PARENT = "1103606"
CHECKPOINT_FILES = {"adapter_model.safetensors", "adapter_config.json", "trainer_state.json", "args.json",
    "optimizer.pt", "scheduler.pt", *("rng_state_%s.pth" % i for i in range(8))}


def authorize_resume(gate):
    need(gate.get("passed") is True and gate.get("request_id") == TRAINING_REQUEST
         and gate.get("independent_passed") is True and gate.get("accounting_C0") is True
         and gate.get("owned_cleanup_passed") is True
         and gate.get("arm") == "continuous" and gate.get("actual_updates") == 25
         and gate.get("ranks_completed") == 8 and gate.get("microbatches_per_rank") == 100,
         "resume requires independently consumed original complete continuous25")
    need(set(gate.get("checkpoint20_files", {})) == CHECKPOINT_FILES,
         "checkpoint20 must include all14 native members")
    return gate["checkpoint20_files"]


def admission_gate(before, run, step, now_ns):
    need(before["request_id"] == TRAINING_REQUEST and before["node"] == NODE
         and re.fullmatch(PARENT + r"\.\d+", before["step"]) and before["step"] != step,
         "fresh admission belongs to another scope")
    need(before["kfd_holder_count"] == 0 and before["full_fd_inventory_performed"] is True
         and before["master_port"] == 29932 and before["master_port_available"] is True
         and len(set(before["pci"])) == 8 and before["boot_id"] == run["boot_id"]
         and 0 <= now_ns - before["monotonic_ns"] <= 45_000_000_000,
         "eight-PCI/KFD/port admission missing or stale")


def execute(arm, run_path, run_sha, admission_path, admission_sha):
    source = verify_package("GPU")
    raw = run_path.read_bytes()
    need(hashlib.sha256(raw).hexdigest() == run_sha, "ROOT run intent SHA differs")
    run = json.loads(raw)
    need(run["request_id"] == TRAINING_REQUEST and run["root"] == str(ROOT)
         and run["parent"] == PARENT and run["node"] == NODE
         and run["source_manifest"] == source and "initial_trainables" not in run,
         "fresh intent scope/reference differs")
    clock(run["epoch"], run["deadline"], time.time(), 2400)
    need(run["epoch"] < run["native_deadline"] <= run["deadline"] - 120,
         "native work leaves no independent cleanup/accounting reserve")
    mono_end = time.monotonic() + clock(run["epoch"], run["native_deadline"], time.time(), 2400)
    def check():
        clock(run["epoch"], run["native_deadline"], time.time(), 2400)
        need(time.monotonic() < mono_end, "original native monotonic deadline exhausted")
    step = os.environ.get("SLURM_JOB_ID", "") + "." + os.environ.get("SLURM_STEP_ID", "")
    need(re.fullmatch(PARENT + r"\.\d+", step) and socket.gethostname().split(".")[0] == NODE,
         "worker is outside root's exact target step")
    need(not any(n in os.environ for n in ("RANK", "LOCAL_RANK", "WORLD_SIZE")), "entry must start unranked")
    before_raw = admission_path.read_bytes()
    need(hashlib.sha256(before_raw).hexdigest() == admission_sha, "fresh admission SHA differs")
    before = json.loads(before_raw)
    admission_gate(before, run, step, time.monotonic_ns())
    need(Path("/proc/sys/kernel/random/boot_id").read_text().strip() == run["boot_id"], "worker boot changed")
    need(pin(ROOT / "stage-complete.json") == run["stage_complete"], "zero-GRES stage not bound")
    stage = json.loads((ROOT / "stage-complete.json").read_bytes())
    need(stage["passed"] is True and stage["training_request_id"] == TRAINING_REQUEST
         and stage["source_manifest"] == source, "actual target asset stage missing")
    for path, expected in stage["reads"].items():
        need(identity(Path(path).lstat()) == expected["file_identity"], "staged exact source/model identity changed")
    manifest_raw, _ = read_exact(ROOT / "fresh-framework-files.json", stage["fresh_framework_manifest"], check)
    for name, expected in json.loads(manifest_raw)["files"].items():
        read_exact(ROOT / "framework" / name, expected, check, False)
    read_exact(ROOT / "view/train800-swift.jsonl", PINS["view"], check, False)
    plan_raw, _ = read_exact(ROOT / (arm + "-plan.json"), pin(D / (arm + "-plan.json")), check)
    plan = json.loads(plan_raw)
    read_exact(ROOT / "structural-binding.json", pin(D / "STRUCTURAL_BINDING.json"), check, False)
    runtime_raw, _ = read_exact(ROOT / "runtime-binding.json", stage["runtime_binding"], check)
    runtime = json.loads(runtime_raw)
    read_exact(Path(runtime["trainer"]["path"]), runtime["trainer"], check, False)
    config = dict(case="T02.sft", dataset_SHA256=PINS["view"]["SHA256"],
                  binding=str(ROOT / "structural-binding.json"))
    if arm == "continuous":
        config.update(initialization_mode="TARGET_FRESH_DOMAIN_ONLY", seed=42, data_seed=42)
        fresh_config(config)
    else:
        gate_raw, _ = read_exact(ROOT / "continuous-independent-terminal.json", run["continuous_independent_terminal"], check)
        files = authorize_resume(json.loads(gate_raw))
        config.update(checkpoint_step=20, checkpoint=str(ROOT / "continuous/train/checkpoint-20"),
                      checkpoint_files=files, trainer_SHA256=runtime["trainer"]["SHA256"])
    # Repeat the kernel PCI/KFD check inside the actual allocated container.
    scope = {"__name__": "gemma_fresh_snapshot"}
    exec(compile((D / "snapshot.py").read_bytes(), str(D / "snapshot.py"), "exec"), scope)
    scope["NODE"] = NODE
    current = scope["snapshot"](True)
    need(current["pci"] == before["pci"] and current["boot_id"] == before["boot_id"]
         and current["kfd_holder_count"] == 0, "actual allocated kernel devices changed")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 29932))
    hip = ctypes.CDLL("libamdhip64.so")
    count = ctypes.c_int()
    need(hip.hipGetDeviceCount(ctypes.byref(count)) == 0 and count.value == 8, "actual HIP allocation differs")
    pci = []
    for index in range(8):
        bus = ctypes.create_string_buffer(32)
        need(hip.hipDeviceGetPCIBusId(bus, 32, index) == 0, "HIP PCI query failed")
        value = bus.value.decode().lower()
        pci.append("0000:" + value if len(value) == 7 else value)
    need(sorted(pci) == current["pci"], "actual HIP PCI differs from kernel inventory")
    arm_root = ROOT / arm
    arm_root.mkdir(mode=0o700)  # both arms exclusive; resumed never writes continuous
    config_path = arm_root / "witness-config.json"
    write_json(config_path, config)
    prefix = "NATIVE_CONTINUOUS" if arm == "continuous" else "NATIVE_RESUME"
    env = dict(os.environ, **plan["environment_overrides"])
    env.update(MASTER_ADDR="127.0.0.1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    env.update({prefix + "_CONFIG": str(config_path), prefix + "_CONFIG_SHA256": pin(config_path)["SHA256"],
        prefix + "_EPOCH": str(run["epoch"]), prefix + "_DEADLINE": str(run["native_deadline"]),
        prefix + "_MAX_SECONDS": "2400"})
    write_json(arm_root / "entry.json", dict(request_id=TRAINING_REQUEST, arm=arm, step=step,
        epoch=run["epoch"], deadline=run["deadline"], native_deadline=run["native_deadline"],
        source_manifest=source, plan=pin(ROOT / (arm + "-plan.json")), actual_updates=None,
        output_directory_initially_absent=True, admission_SHA256=admission_sha, actual_visible_PCI=pci))
    check()
    need(time.monotonic_ns() - before["monotonic_ns"] <= 45_000_000_000, "admission expired before exec")
    os.execve(sys.executable, [sys.executable, "-B", *plan["command"]], env)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("arm", choices=("continuous", "resumed"))
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--run-sha", required=True)
    p.add_argument("--admission", required=True, type=Path)
    p.add_argument("--admission-sha", required=True)
    a = p.parse_args()
    execute(a.arm, a.run, a.run_sha, a.admission, a.admission_sha)
