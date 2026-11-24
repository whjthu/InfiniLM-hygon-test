"""Connect the new disk-backed RAM observer to the existing one-shot HOST launcher."""

from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import time


REQUEST = "274608fc6e0d410a8f545287843cb887"
LABEL = "O01-zimage-lt-eight-card-memory48-v1"
PRE = "dea95c27aecb4389a30e1e7f48ee1940"
POST = "651ba2519f6a40e69e0bf434b50fd2b1"
OBSERVER = "247473e5e57b472daf49bf48f64d6d8e"
OBSERVER_LABEL = "O01-zimage-hybrid-ram-" + OBSERVER[:12]
PREDECESSOR_STEP = "1103606.2179"
PREDECESSOR_REQUEST = "8bfbd56e71bf424da2133e9ab0e482ef"
PREDECESSOR_LABEL = "training80-dcu-standby-8bfbd56e71bf-f12r2n11"
GPU_MIB = 49152
SIDE_MIB = 256


def observer_argv(spec, scope, scope_sha):
    spec.require(scope["request"] == REQUEST and scope["host_memory_MiB"] == GPU_MIB
                 and scope["memory_observer_label"] == OBSERVER_LABEL and scope["deadline_epoch"] == scope["epoch"] + 240,
                 "RAM_SCOPE_IDENTITY")
    run = spec.RUN
    stage = spec.STAGE
    head = spec.srun_base(OBSERVER_LABEL, 1, "none", "256M", 260, run)
    head[head.index("--output=" + str(run / "slurm.stdout"))] = "--output=" + str(run / "ram-slurm.stdout")
    head[head.index("--error=" + str(run / "slurm.stderr"))] = "--error=" + str(run / "ram-slurm.stderr")
    return [*head,
            spec.PYTHON, "-I", "-S", "-B", "-u", str(stage / "runtime/ram_trace.py"),
            "--scope", str(run / "scope.json"), "--scope-sha256", scope_sha,
            "--owner", str(run / "attested-host-owner.json"),
            "--reader", str(stage / "memory_reader.py"),
            "--attested-source", str(stage / "identity_original.py"),
            "--phases", str(run / "supervised/memory-phases"),
            "--out", str(run / "ram-observer")]


def sidecar_live(spec, scheduler):
    rows = []
    for line in scheduler.command("scontrol", "-o", "show", "step", spec.PARENT).splitlines():
        if line.strip():
            row = scheduler.single(line, "StepId")
            if row.get("Name") == OBSERVER_LABEL:
                rows.append(row)
    spec.require(len(rows) <= 1, "RAM_OBSERVER_STEP_NOT_UNIQUE")
    if not rows:
        return None
    [row] = rows
    spec.require(spec.STEP.fullmatch(row["StepId"]) and row["StepId"] != PREDECESSOR_STEP
                 and row["NodeList"] == spec.NODE, "RAM_OBSERVER_PLACEMENT")
    spec.check_resources(row["TRES"], 1, 0, {"256M"})
    return row["StepId"] if row["State"] == "RUNNING" else None


def launch_pair(scope, gpu_argv, spec, scheduler, run_gpu, safe_environment):
    """Keep the original shared lock at the caller across both one-shot attempts."""
    run = Path(spec.RUN)
    scope_sha = spec.digest(spec.encoded(scope))
    argv = observer_argv(spec, scope, scope_sha)
    observation = scheduler.collect()
    rows, active = spec.scheduler_scope(observation, time.time(), os.getuid())
    spec.require(not active and not any(row[1] in {OBSERVER_LABEL, LABEL} for row in rows),
                 "ORIGINAL_RAM_OR_GPU_ALREADY_VISIBLE")
    spec.require(not any(row["Name"] in {OBSERVER_LABEL, LABEL} for row in observation["numbered_steps"]),
                 "ORIGINAL_RAM_OR_GPU_LABEL_ALREADY_VISIBLE")
    remaining = scope["deadline_epoch"] - time.time() - 185
    spec.require(remaining >= 20, "INSUFFICIENT_ORIGINAL_RAM_EPOCH")
    spec.save_new(run / "ram-observer-attempt.json", {"request": OBSERVER, "label": OBSERVER_LABEL,
                  "gpu_request": REQUEST, "scope_SHA256": scope_sha,
                  "argv_SHA256": spec.digest(spec.encoded(argv)), "replay_allowed": False,
                  "deadline_epoch": scope["deadline_epoch"]})
    process = None
    side_step = None
    refusal = None
    try:
        with (run / "ram-launcher.stderr").open("xb") as errors:
            process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=errors,
                                       env=safe_environment(), start_new_session=True)
        until = time.monotonic() + min(8, remaining - 10)
        while time.monotonic() < until:
            side_step = sidecar_live(spec, scheduler)
            if side_step is not None or process.poll() is not None:
                break
            time.sleep(0.25)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        refusal = type(error).__name__
    if side_step is None:
        result = {"request": REQUEST, "scope_SHA256": scope_sha, "gpu_exec_attempted": False,
                  "observer_request": OBSERVER, "observer_launcher_PID": process.pid if process else None,
                  "observer_returncode": process.poll() if process else None,
                  "reason": refusal or "RAM_OBSERVER_NUMBERED_STEP_UNOBSERVED", "replay_allowed": False}
        spec.save_new(run / "ram-pre-gpu-refusal.json", result)
        return result
    spec.save_new(run / "ram-observer-adopted.json", {"request": OBSERVER, "scope_SHA256": scope_sha,
                  "observer_step": side_step, "launcher_PID": process.pid, "replay_allowed": False})
    return {"observer_step": side_step, "observer_request": OBSERVER, "gpu": run_gpu(scope, gpu_argv)}


def observer_status(run):
    root = Path(run) / "ram-observer"
    result = {}
    for name, cap in (("ready.json", 4096), ("observer-terminal.json", 16384), ("samples.jsonl", 1024 * 1024)):
        path = root / name
        try:
            with path.open("rb") as handle:
                data = handle.read(cap + 1)
        except FileNotFoundError:
            continue
        if not 0 < len(data) <= cap or path.is_symlink():
            continue
        digest = sha256(data).hexdigest()
        if name == "samples.jsonl":
            result["samples"] = {"observed_bytes": len(data), "SHA256": digest, "complete": False}
        else:
            row = json.loads(data)
            if row["request"] != REQUEST:
                raise ValueError("RAM_EVIDENCE_REQUEST")
            keys = ("step", "actual_host_limit_bytes", "observer", "target", "samples", "end_reason",
                    "failure_type", "failure_gate")
            result[name] = {"SHA256": digest, **{key: row[key] for key in keys if key in row}}
    refusal = Path(run) / "ram-pre-gpu-refusal.json"
    try:
        with refusal.open("rb") as handle:
            data = handle.read(4097)
    except FileNotFoundError:
        return result
    if 0 < len(data) <= 4096 and not refusal.is_symlink():
        row = json.loads(data)
        if row.get("request") == REQUEST and row.get("gpu_exec_attempted") is False:
            result["pre_gpu_refusal"] = {"SHA256": sha256(data).hexdigest(), "gpu_exec_attempted": False,
                                         "reason": row.get("reason")}
    return result


def terminal_reference(records, scope, run, exact_row, check_resources):
    """Return actual GPU accounting or retire a proved pre-GPU refusal after sidecar terminal."""
    if any(row[1] == LABEL for row in records):
        gpu = exact_row(records, LABEL)
        check_resources(gpu[6], 12, 8, {"48G", "49152M"})
        return gpu, "GPU_STEP_TERMINAL"
    run = Path(run)
    scope_sha = sha256((json.dumps(scope, allow_nan=False, sort_keys=True) + "\n").encode()).hexdigest()
    path = run / "ram-pre-gpu-refusal.json"
    if path.is_symlink():
        raise ValueError("RAM_REFUSAL_RECEIPT_SYMLINK")
    with path.open("rb") as handle:
        raw = handle.read(4097)
    if not 0 < len(raw) <= 4096:
        raise ValueError("RAM_REFUSAL_RECEIPT_BOUND")
    receipt = json.loads(raw)
    if not (receipt["request"] == REQUEST and receipt["scope_SHA256"] == scope_sha
            and receipt["observer_request"] == OBSERVER and receipt["gpu_exec_attempted"] is False
            and receipt["replay_allowed"] is False and
            not any(os.path.lexists(run / leaf) for leaf in ("ram-observer-adopted.json", "launcher-pid.json",
                "launcher-observation.json", "attested-host-owner.json", "attested-container-owner.json"))):
        raise ValueError("RAM_PRE_GPU_REFUSAL_NOT_PROVED")
    sidecar = exact_row(records, OBSERVER_LABEL)
    check_resources(sidecar[6], 1, 0, {"256M"})
    return sidecar, "REAL_RAM_OBSERVER_TERMINAL_WITH_PROVED_NO_GPU_EXEC"
