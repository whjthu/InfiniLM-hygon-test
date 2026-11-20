"""Run both Gemma arms inside their original admitted eight-card Slurm step."""

import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import socket
import stat
import subprocess
import sys
import time

from compare_gemma_domain_recovery import compare, read_receipt, require
from install_gemma_domain_framework import BASE_MANIFEST, base_files, candidate_files, exact_source
from prepare_gemma_domain_pair import candidate
from read_gemma_domain_terminal import (
    DATASET_SHA, LABEL, NODE, REQUIRED_CHECKPOINT_FILES, bound_step, numbered_step, token,
)


OWNED_HELPER = (
    "code/bw1000/training-qwen-usgs-exposure25-resume20-hip-v1/owned_processes.py",
    "f09a3071ae6d574fc4492d9a3589e6465d702fb5ea7c53cbcd9b468bf5aed0cc",
)
DEVICE_HELPER = (
    "code/bw1000/training-qwen-usgs-exposure25-resume20-hip-v1/snapshot.py",
    "98b45c22130c1e39a14ebb7393d85f2a763ffc2417de35a4f57fa303f3f05b8a",
)
MODEL_BINDING = (
    "coordination/training/gemma-target-original-model16-v1.json",
    "46428e251757e386835e2832a32308a6a07178d9ef776a3ea1b70062c5cf77b3",
)
CHECKPOINTS = {"continuous": (10, 20, 25), "resumed": (25,)}
CALLBACKS = {"continuous": b"gemma_domain_continuous_witness", "resumed": b"gemma_domain_resume_witness"}


def sha_file(path, check, *, limit=256 * 1024**2):
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= limit, "unsafe or oversized binary artifact")
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as stream:
        for raw in iter(lambda: stream.read(1024**2), b""):
            check()
            total += len(raw)
            require(total <= before.st_size, "artifact changed while being hashed")
            digest.update(raw)
    after = path.lstat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    require(total == before.st_size and all(getattr(before, k) == getattr(after, k) for k in fields),
            "artifact changed while being hashed")
    return {"bytes": total, "SHA256": digest.hexdigest()}


def write_json(path, data):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, allow_nan=False, sort_keys=True)
        stream.write("\n")


def inventory(root, arm, step, check):
    require(arm in CHECKPOINTS and step in CHECKPOINTS[arm], "unexpected Gemma checkpoint scope")
    checkpoint = root / arm / "train" / f"checkpoint-{step}"
    require(checkpoint.resolve(strict=True) == checkpoint, "checkpoint aliases another directory")
    files = {name: sha_file(checkpoint / name, check) for name in sorted(REQUIRED_CHECKPOINT_FILES)}
    require(sum(pin["bytes"] for pin in files.values()) <= 512 * 1024**2, "checkpoint exceeds native restore budget")
    trainer = read_receipt(checkpoint, Path("trainer_state.json"), {})
    args = read_receipt(checkpoint, Path("args.json"), {})
    require(trainer["global_step"] == step and trainer["max_steps"] == args["max_steps"] == 25
            and trainer["train_batch_size"] == args["per_device_train_batch_size"] == 1
            and args["gradient_accumulation_steps"] == 4, "native checkpoint step or accumulation differs")
    write_json(root / f"{arm}-checkpoint-{step}-files.json", dict(global_step=step, files=files))
    return files


def command_owned(argv, arm_root):
    try:
        index = argv.index(b"--output_dir")
        return argv[index + 1] == os.fsencode(arm_root / "train") and CALLBACKS[arm_root.name] in argv
    except (ValueError, IndexError, KeyError):
        return False


def pinned_module(repo, pin, module_name):
    path = repo / pin[0]
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == pin[1], "pinned runtime helper differs")
    scope = {"__name__": module_name}
    exec(compile(raw, str(path), "exec"), scope)
    return scope


def ownership(repo):
    scope = pinned_module(repo, OWNED_HELPER, "gemma_owned_processes")
    scope["native_command"] = command_owned
    return scope


def verify_devices(repo, root, run, before):
    require(before["step"] == run["steps"]["pre"] and before["node"] == NODE
            and before["kfd_holder_count"] == 0 and before["full_fd_inventory_performed"] is True
            and before["master_port"] == run["master_port"] and before["master_port_available"] is True,
            "original independent admission snapshot differs")
    scope = pinned_module(repo, DEVICE_HELPER, "gemma_device_snapshot")
    scope["NODE"] = NODE
    current = scope["snapshot"](True)
    require(current["pci"] == before["pci"] and current["boot_id"] == before["boot_id"]
            and current["kfd_holder_count"] == 0 and len(current["pci"]) == 8
            and 0 <= current["monotonic_ns"] - before["monotonic_ns"] <= 45_000_000_000,
            "assigned devices do not match the fresh admission snapshot")
    with socket.socket() as check:
        check.bind(("127.0.0.1", run["master_port"]))
    hip = ctypes.CDLL("libamdhip64.so")
    count = ctypes.c_int()
    require(hip.hipGetDeviceCount(ctypes.byref(count)) == 0 and count.value == 8, "wrong visible HIP device count")
    visible = []
    for index in range(count.value):
        bus = ctypes.create_string_buffer(32)
        require(hip.hipDeviceGetPCIBusId(bus, len(bus), index) == 0, "HIP PCI query failed")
        serial = bus.value.decode().lower()
        visible.append("0000:" + serial if len(serial) == 7 else serial)
    require(sorted(visible) == current["pci"], "actual HIP PCI allocation differs")
    write_json(root / "assigned.json", dict(current, visible_pci=visible))


def verify_model(repo, plans):
    raw = (repo / MODEL_BINDING[0]).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == MODEL_BINDING[1], "original target Gemma model receipt differs")
    saved = json.loads(raw)
    model_dir = Path(saved["target_model_directory"])
    require(saved["actual_full_target_weight_SHA_verified"] is True and len(saved["files"]) == 16
            and all(plan["required_assets"]["model_directory"] == str(model_dir) for plan in plans.values()),
            "training arms are not bound to the already-qualified original Gemma weights")
    for name, pin in saved["files"].items():
        path = model_dir / name
        require(Path(name).name == name and stat.S_ISREG(path.lstat().st_mode)
                and path.stat().st_size == pin["bytes"], "original Gemma asset path or size changed")


def rank_receipts(arm_root):
    rows = []
    kind = "continuous-witness" if arm_root.name == "continuous" else "resume-witness"
    for rank in range(8):
        folder = arm_root / "train" / kind / f"rank-{rank:02d}"
        require(folder.resolve(strict=True) == folder, "native rank receipt directory aliases another path")
        pins = {}
        result = read_receipt(folder.resolve(strict=True), Path("result.json"), pins)
        owner = read_receipt(folder.resolve(strict=True), Path("owner.json"), pins)
        require((result["rank"], result["world_size"], result["completed_global_step"]) == (rank, 8, 25)
                and owner["rank"] == owner["local_rank"] == rank, "native rank receipt is incomplete")
        rows.append(dict(rank=rank, result_SHA256=pins["result.json"], owner_SHA256=pins["owner.json"]))
    return rows


def run_arm(root, arm, plan, config, run, deadline, checker, owned):
    started = time.monotonic()
    arm_root = root / arm
    arm_root.mkdir(mode=0o700)
    cfg_path = arm_root / "witness-config.json"
    write_json(cfg_path, config)
    prefix = "NATIVE_CONTINUOUS" if arm == "continuous" else "NATIVE_RESUME"
    env = dict(os.environ, **plan["environment_overrides"])
    require(not {"RANK", "LOCAL_RANK", "WORLD_SIZE"}.intersection(env), "native worker must start unranked")
    env.update({"MASTER_ADDR": "127.0.0.1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                f"{prefix}_CONFIG": str(cfg_path), f"{prefix}_CONFIG_SHA256": sha_file(cfg_path, checker)["SHA256"],
                f"{prefix}_EPOCH": str(run["epoch"]), f"{prefix}_DEADLINE": str(deadline),
                f"{prefix}_MAX_SECONDS": str(run["maximum_seconds"])})
    registry, rc, failure, ranks = {}, None, None, []
    child = None
    checker()
    try:
        with (arm_root / "native.log").open("xb") as log:
            child = subprocess.Popen([sys.executable, "-B", *plan["command"]], stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            leader = owned["snapshot"]().get(child.pid)
            if leader is not None:
                registry[child.pid] = {k: leader[k] for k in ("PID", "PPID", "PGID", "startticks", "state")}
            while child.poll() is None:
                checker()
                owned["refresh"](registry, arm_root)
                time.sleep(0.25)
            rc = child.returncode
        require(rc == 0, "native Gemma process returned failure")
        ranks = rank_receipts(arm_root)
    except (OSError, ValueError, KeyError, TimeoutError, subprocess.SubprocessError) as error:
        failure = type(error).__name__
    finally:
        cleanup = owned["cleanup"](registry, arm_root, grace=5)
        if child is not None:
            try:
                child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                failure = "OwnedNativeProcessUnreaped"
        passed = (rc == 0 and failure is None and len(ranks) == 8 and not cleanup["termination_needed"]
                  and cleanup["owned_processes_remaining"] == 0)
        write_json(arm_root / "native-outcome.json", dict(request_id=run["request_id"], step=run["steps"]["gpu"],
            rc=rc, failure=failure, ranks=ranks, passed=passed, wall_seconds=time.monotonic() - started, **cleanup))
    return passed


def execute(repo, root, intent_sha):
    root = root.resolve(strict=True)
    receipts = {}
    run = read_receipt(root, Path("intent.json"), receipts)
    step = os.environ.get("SLURM_JOB_ID", "") + "." + os.environ.get("SLURM_STEP_ID", "")
    require(receipts["intent.json"] == intent_sha and token(run["request_id"], 32)
            and run["label"] == LABEL and "steps" not in run and numbered_step(step)
            and socket.gethostname().split(".")[0] == NODE,
            "worker is outside the original Gemma Slurm step")
    before = read_receipt(root, Path("before.json"), receipts)
    pre = bound_step(before, run["request_id"], intent_sha)
    require(step != pre, "GPU worker and preflight may not share a Slurm step")
    write_json(root / "gpu-step-binding.json", dict(step=step, node=NODE, pre_step=pre,
        request_id=run["request_id"], intent_SHA256=intent_sha, before_SHA256=receipts["before.json"]))
    now = time.time()
    epoch, deadline, maximum = (run[key] for key in ("epoch", "native_deadline", "maximum_seconds"))
    require(all(math.isfinite(v) for v in (epoch, deadline, maximum)) and epoch <= now
            and 30 < deadline - now <= deadline - epoch <= maximum <= 2400, "native run budget is invalid")
    stop = time.monotonic() + (deadline - now)

    def check():
        if time.monotonic() >= stop or time.time() >= deadline:
            raise TimeoutError("Gemma native deadline exhausted")

    installed = read_receipt(root, Path("installation-complete.json"), receipts)
    require(receipts["installation-complete.json"] == run["installation_receipt_SHA256"],
            "installed framework receipt differs")
    target = PurePosixPath(str(root))
    overlay, plans = candidate(repo, target, PurePosixPath(run["models_root"]),
                               PurePosixPath(run["plugins_root"]), run["master_port"])
    base = base_files((repo / BASE_MANIFEST[0]).read_bytes())
    files, tree = candidate_files(base, overlay)
    require(installed["candidate_framework_tree"] == tree and installed["installed_files_verified"] == len(files),
            "installed framework tree differs")
    for name, pin in files.items():
        check()
        exact_source(root / "framework", name, pin)
    for arm, expected in plans.items():
        require(read_receipt(root, Path(f"{arm}-private.json"), receipts) == expected, "native arm command differs")
    verify_model(repo, plans)
    cpu = read_receipt(root, Path("native-input-cpu-result.json"), receipts)
    require(receipts["native-input-cpu-result.json"] == run["native_cpu_receipt_SHA256"] and cpu["passed"] is True
            and (cpu["unique_rows"], cpu["encode_calls"], cpu["collator_calls"]) == (72, 144, 72),
            "native CPU input audit is missing or incomplete")
    require(sha_file(root / "view/train800-swift.jsonl", check)["SHA256"] == DATASET_SHA,
            "frozen Gemma domain training file differs")
    require(sha_file(Path(run["initial_trainables"]), check)["SHA256"]
            == "b85fdfc7e3a7866cbc85575793f8621a314229aa6f9dcc471a75fdc028dee94b",
            "actual original Gemma initialization is absent")
    run = dict(run, steps={"pre": pre, "gpu": step})
    verify_devices(repo, root, run, before)
    owned = ownership(repo)
    shared = dict(case="T02.sft", dataset_SHA256=DATASET_SHA, binding=run["gemma_binding"])
    cfg = dict(shared, initial_trainables=run["initial_trainables"])
    require(run_arm(root, "continuous", plans["continuous"], cfg, run, deadline, check, owned),
            "native continuous arm failed")
    continuous = {number: inventory(root, "continuous", number, check) for number in CHECKPOINTS["continuous"]}
    resume = dict(shared, checkpoint_step=20, checkpoint=str(root / "continuous/train/checkpoint-20"),
                  checkpoint_files=continuous[20], trainer_SHA256=run["trainer_SHA256"])
    require(run_arm(root, "resumed", plans["resumed"], resume, run, deadline, check, owned), "native resumed arm failed")
    inventory(root, "resumed", 25, check)
    paired = compare(root)
    write_json(root / "paired-recovery.json", paired)
    return paired["exact_paired_recovery"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--intent-sha", required=True)
    opts = parser.parse_args()
    raise SystemExit(0 if execute(opts.repo, opts.root, opts.intent_sha) else 1)


if __name__ == "__main__":
    main()
