"""Capture host-RAM cgroup samples outside the timed eight-card step."""

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import tempfile
import time
from types import ModuleType


PARENT, NODE = "1103606", "f12r2n11"
OLD_GPU_REQUEST = "e23d24ea0c9c4d00af06154d82dc22e8"
OLD_GPU_LABEL = "O01-zimage-lt-eight-card-supervisor-v1"
OLD_OBSERVER_LABEL = "O01-refill-memory-observe-d97b876aac66"
OLD_STEPS = {"1103606.2175", "1103606.2179", "1103606.2185"}
MEMORY_READER_SHA = "bdff67d465583fef518b304dc50979c9bf03f05238f428d325e8ff152ce10c0e"
PROPOSED_MIB = 49152
MIB = 1 << 20
PHASES = ("inventory_enter", "cold_standbys_enter", "eight_standbys_ready", "seven_standbys_ready",
          "witness_enter", "witness_return", "restore_enter", "eight_restored", "cleanup_enter", "cleanup_return")
STEP = re.compile(PARENT + r"\.[0-9]+")


def need(ok, gate):
    if not ok:
        raise ValueError(gate)


def digest(raw):
    return sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, allow_nan=False, sort_keys=True) + "\n").encode()


def read(path, maximum=16384):
    with path.open("rb") as handle:
        raw = handle.read(maximum + 1)
    need(len(raw) <= maximum, "INPUT_TOO_LARGE")
    return raw


def pinned(path, expected, maximum=16384):
    need(not path.is_symlink(), "BOUND_FILE_SYMLINK")
    raw = read(path, maximum)
    need(raw and digest(raw) == expected, "PIN_MISMATCH")
    return raw


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def commit(path, value):
    """Publish complete bytes once, even while the other step reads this directory."""
    handle, temporary = tempfile.mkstemp(prefix=".ram-", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as output:
            output.write(encoded(value))
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary, path)
        sync_directory(path.parent)
    finally:
        os.unlink(temporary)


def valid_step(step):
    return bool(STEP.fullmatch(step) and step not in OLD_STEPS)


def scope_checks(scope):
    first, last = scope["epoch"], scope["deadline_epoch"]
    planned = scope["host_memory_MiB"]
    need(scope["parent"] == PARENT and scope["node"] == NODE and scope["allocated_DCU"] == 8
         and re.fullmatch("[0-9a-f]{32}", scope["request"]) and scope["request"] != OLD_GPU_REQUEST
         and re.fullmatch("[A-Za-z0-9_-]{1,100}", scope["label"]) and scope["label"] != OLD_GPU_LABEL
         and all(re.fullmatch("[0-9a-f]{64}", scope[field])
                 for field in ("boot_id_SHA256", "runtime_SHA256", "attested_entry_SHA256"))
         and re.fullmatch("[A-Za-z0-9_-]{1,100}", scope["memory_observer_label"])
         and scope["memory_observer_label"] not in {scope["label"], OLD_OBSERVER_LABEL, OLD_GPU_LABEL}
         and type(planned) is int and planned > 24576
         and all(type(t) in (int, float) and math.isfinite(t) for t in (first, last))
         and last == first + 240, "ORIGINAL_EPOCH_OR_SCOPE")
    return planned * MIB


def phase_path(directory, name):
    return directory / f"{PHASES.index(name):02}-{name}.json"


def record_phase(directory, scope, scope_sha, step, name):
    """Supervisor hook at an existing phase boundary, outside its kernel timings."""
    need(valid_step(step), "PHASE_STEP_IDENTITY")
    commit(phase_path(directory, name), {"request": scope["request"], "step": step, "scope_SHA256": scope_sha,
          "phase": name, "epoch_ns": time.time_ns(), "monotonic_ns": time.monotonic_ns()})


def latest_phase(directory, scope, scope_sha, step):
    candidates = []
    for name in PHASES:
        try:
            event = json.loads(read(phase_path(directory, name), 2048))
        except FileNotFoundError:
            continue
        need((event["request"], event["step"], event["scope_SHA256"], event["phase"])
             == (scope["request"], step, scope_sha, name) and type(event["monotonic_ns"]) is int
             and 0 < event["monotonic_ns"] <= time.monotonic_ns(), "PHASE_RECEIPT_IDENTITY")
        candidates.append(event)
    values = [event["monotonic_ns"] for event in candidates]
    need(values == sorted(set(values)), "PHASE_TIME_ORDER")
    return candidates[-1] if candidates else None


def await_baseline(ready_path, scope, scope_sha, step):
    """Wait before HIP inventory or child creation, keeping the original epoch."""
    limit = scope_checks(scope)
    deadline = time.monotonic() + min(10, scope["deadline_epoch"] - time.time() - 185)
    while time.monotonic() < deadline:
        try:
            ready = json.loads(read(ready_path, 4096))
        except FileNotFoundError:
            time.sleep(0.1)
            continue
        need(valid_step(step) and ready["request"] == scope["request"] and ready["step"] == step
             and ready["scope_SHA256"] == scope_sha and ready["actual_host_limit_bytes"] == limit
             and 0 <= time.monotonic_ns() - ready["monotonic_ns"] <= 15_000_000_000
             and re.fullmatch("[0-9a-f]{64}", ready["first_sample_SHA256"]), "BASELINE_READY_IDENTITY")
        return ready
    raise TimeoutError("HOST_RAM_BASELINE_NOT_READY")


def memory_module(path):
    module = ModuleType("original_o01_cgroup_memory_reader")
    module.__file__ = str(path)
    exec(compile(pinned(path, MEMORY_READER_SHA, 65536), str(path), "exec"), vars(module))
    return module


def inspect_resources(text, scope, step, label, cpus, dcus, memory):
    """Use exact step lookup on the real multi-row scontrol response."""
    expected = {"cpu": str(cpus), "node": "1", "mem": str(memory) + "M"}
    if dcus:
        expected["gres/dcu"] = str(dcus)
    rows = [dict(word.split("=", 1) for word in line.split() if "=" in word) for line in text.splitlines()]
    matching = [row for row in rows if row.get("StepId") == step]
    need(len(matching) == 1, "SCHEDULER_STEP_NOT_UNIQUE")
    row = matching[0]
    tres = row["TRES"].split(",")
    resources = dict(field.split("=", 1) for field in tres)
    if memory % 1024 == 0 and resources.get("mem") == str(memory // 1024) + "G":
        resources["mem"] = str(memory) + "M"
    need(row["Name"] == label and row["NodeList"] == scope["node"] and row["State"] == "RUNNING"
         and len(tres) == len(resources) and resources == expected, "ACTUAL_STEP_RESOURCES")


def target_binding(observer, identity, evidence, scope, scope_sha, mine):
    owner = evidence["owner"]
    target, pid = owner["step"], owner["pid"]
    need(valid_step(target) and target != mine and type(pid) is int and pid > 1
         and evidence["request"] == scope["request"] and evidence["scope_SHA256"] == scope_sha
         and evidence["entry_SHA256"] == scope["attested_entry_SHA256"]
         and evidence["runtime_SHA256"] == scope["runtime_SHA256"]
         and evidence["slurm_listpids_matched"] is True and evidence["live_slurm"]["StepId"] == target
         and owner["cgroup_names_step"] is True and owner["node"] == NODE and owner["uid"] == os.getuid()
         and owner["boot_SHA256"] == scope["boot_id_SHA256"]
         and scope["epoch"] <= owner["epoch"] < scope["deadline_epoch"] - 185,
         "TARGET_ATTESTED_OWNER")
    affinity = set(owner["affinity"])
    need(len(owner["affinity"]) == len(affinity) == 12 and all(type(cpu) is int and cpu >= 0 for cpu in affinity)
         and affinity.isdisjoint(os.sched_getaffinity(0)), "OBSERVER_GPU_CPUSETS_OVERLAP")
    same, raw = observer.process_binding(identity, pid, target, scope["boot_id_SHA256"])
    need(same["start_ticks"] == owner["start_ticks"] and same["cgroup_sha256"] == owner["cgroup_SHA256"],
         "TARGET_PROCESS_REUSED_OR_MOVED")
    observer.TARGET = target  # Existing reader selector only. Frozen bytes and parsers are unchanged.
    version, _, group = observer.memory_controller(raw.decode())
    base, path = observer.resolve_step(observer.read(Path("/proc/self/mountinfo"), 262144).decode(), version, group)
    return target, version, base, path


def sample(observer, base, path, version, directory_id):
    first = path.lstat()
    need(stat.S_ISDIR(first.st_mode) and (first.st_dev, first.st_ino) == directory_id, "TARGET_CGROUP_REPLACED")
    values = observer.memory_counters(base, path, version)
    last = path.lstat()
    need((last.st_dev, last.st_ino) == directory_id, "TARGET_CGROUP_REPLACED")
    return values


def budget(values, expected):
    step_limit = values["step_memory"]["limit_bytes"]
    need(step_limit["value"] == expected and values["minimum_visible_hard_limit_bytes"] == expected,
         "STEP_OR_ANCESTOR_LIMIT_DIFFERS")
    return step_limit["value"]


def observed_phase(directory, scope, scope_sha, target):
    try:
        event = latest_phase(directory, scope, scope_sha, target)
    except (OSError, ValueError, KeyError, TypeError):
        return {"phase_receipt_invalid": True}
    return {"phase": event["phase"], "phase_monotonic_ns": event["monotonic_ns"]} if event else {}


def await_owner(path, deadline):
    # The already-qualified host writes its owner receipt exclusively but the reader can see in-flight bytes.
    while time.monotonic() < deadline:
        try:
            need(not path.is_symlink(), "OWNER_RECEIPT_SYMLINK")
            return json.loads(read(path))
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(0.1)
    raise TimeoutError("ORIGINAL_ATTESTED_OWNER_NOT_VISIBLE")


def run(args, scope, expected):
    observer = memory_module(args.reader)
    identity = observer.load_identity(args.attested_source)
    mine = os.environ["SLURM_JOB_ID"] + "." + os.environ["SLURM_STEP_ID"]
    need(valid_step(mine) and socket.gethostname().split(".", 1)[0] == NODE
         and os.environ.get("SLURM_STEP_GPUS") in {None, "", "(null)"}
         and os.environ.get("SLURM_CPUS_PER_TASK") == "1" and len(os.sched_getaffinity(0)) == 1,
         "ZERO_GRES_OBSERVER_SCOPE")
    boot = scope["boot_id_SHA256"]
    observer.process_binding(identity, os.getpid(), mine, boot)
    records = observer.control("show", "step", PARENT, "--oneliner")
    inspect_resources(records, scope, mine, scope["memory_observer_label"], 1, 0, 256)
    remaining = scope["deadline_epoch"] - time.time()
    need(185 < remaining <= 240, "OBSERVER_ORIGINAL_EPOCH_TOO_LATE")
    stop_at = time.monotonic() + remaining + 12
    evidence = await_owner(args.owner, time.monotonic() + min(35, remaining - 185))
    target, version, base, path = target_binding(observer, identity, evidence, scope, args.scope_sha256, mine)
    records = observer.control("show", "step", PARENT, "--oneliner")
    inspect_resources(records, scope, target, scope["label"], 12, 8, scope["host_memory_MiB"])
    found = path.lstat()
    directory_id = (found.st_dev, found.st_ino)
    result = {"request": scope["request"], "observer": mine, "target": target, "samples": 0,
              "evidence_type": "host_RAM_step_cgroup", "HBM_measured": False, "step_terminal_independent": False}
    with (args.out / "samples.jsonl").open("xb") as log:
        while time.monotonic() < stop_at and result["samples"] < 256:
            try:
                left = observed_phase(args.phases, scope, args.scope_sha256, target)
                started = time.monotonic_ns()
                row = sample(observer, base, path, version, directory_id)
                finished = time.monotonic_ns()
                right = observed_phase(args.phases, scope, args.scope_sha256, target)
            except FileNotFoundError:
                result["end_reason"] = "ORIGINAL_CGROUP_OR_COUNTER_DISAPPEARED"
                break
            record = {"sample": result["samples"], "step": target, "scope_SHA256": args.scope_sha256,
                      "start_monotonic_ns": started, "end_monotonic_ns": finished, "epoch_ns": time.time_ns(),
                      "phase_before": left, "phase_after": right, "memory": row}
            raw = encoded(record)
            log.write(raw)
            log.flush()
            os.fsync(log.fileno())
            result["samples"] += 1
            if result["samples"] == 1:
                commit(args.out / "ready.json", {"request": scope["request"], "step": target,
                       "scope_SHA256": args.scope_sha256, "actual_host_limit_bytes": budget(row, expected),
                       "first_sample_SHA256": digest(raw), "monotonic_ns": finished})
            if left.get("phase") == right.get("phase") == "cleanup_return":
                result["cleanup_phase_seen"] = True
            time.sleep(min(1.0, max(0.0, stop_at - time.monotonic())))
    result.setdefault("end_reason", "SAMPLING_BOUND")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("scope", "owner", "reader", "attested-source", "phases", "out"):
        parser.add_argument("--" + key, required=True, type=Path)
    parser.add_argument("--scope-sha256", required=True)
    args = parser.parse_args()
    scope = json.loads(pinned(args.scope, args.scope_sha256))
    expected = scope_checks(scope)
    args.out.mkdir(mode=0o700)
    try:
        result = run(args, scope, expected)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        gate = str(error)
        result = {"request": scope["request"], "failure_type": type(error).__name__,
                  "failure_gate": gate if re.fullmatch("[A-Z0-9_]+", gate) else None,
                  "partial_samples_file_present": (args.out / "samples.jsonl").is_file(),
                  "evidence_type": "host_RAM_step_cgroup", "HBM_measured": False, "step_terminal_independent": False}
    commit(args.out / "observer-terminal.json", result)
    return int(not (args.out / "ready.json").is_file() or "failure_type" in result)


if __name__ == "__main__":
    raise SystemExit(main())
