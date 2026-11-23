"""Read host-RAM cgroup counters of the actual .2179 standby from one zero-GRES step."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import socket
import subprocess
import time
from types import ModuleType


REQUEST = "d97b876aac6643c581a7ef130ea728c0"
LABEL = "O01-refill-memory-observe-d97b876aac66"
PARENT, NODE = "1103606", "f12r2n11"
TARGET = "1103606.2179"
TARGET_REQUEST = "8bfbd56e71bf424da2133e9ab0e482ef"
TARGET_LABEL = "training80-dcu-standby-8bfbd56e71bf-f12r2n11"
WORKER_PID, WORKER_START = 982368, 356482943
RESULTS = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
WORKER = RESULTS / f"training80-dcu-standby-{TARGET_REQUEST}" / NODE
PROC = Path("/proc")
BOOT = PROC / "sys/kernel/random/boot_id"
ENTRY_SHA = "c7f13669918f720ea195cb49d0be843681d3155b025520caa8b1887babbd6ee6"
SHA = re.compile("[0-9a-f]{64}")
PCIS = frozenset(f"0000:{bus}:00.0" for bus in ("09", "36", "55", "77", "85", "b5", "d5", "f5"))
V2_CLASSES = {"anon", "file", "shmem", "kernel", "kernel_stack", "pagetables", "percpu", "slab",
              "slab_reclaimable", "slab_unreclaimable", "sock", "vmalloc", "file_mapped", "file_dirty", "anon_thp"}
V1_CLASSES = {"rss", "cache", "mapped_file", "swap", "shmem", "kmem", "total_rss", "total_cache",
              "total_mapped_file", "total_swap", "total_shmem", "hierarchical_memory_limit"}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(raw):
    return sha256(raw).hexdigest()


def read(path, maximum=16384):
    with path.open("rb") as handle:
        raw = handle.read(maximum + 1)
    require(0 < len(raw) <= maximum, "INPUT_FILE_BOUND")
    return raw


def exact(path, expected, maximum):
    require(path.is_file() and not path.is_symlink(), "SOURCE_REGULAR_FILE")
    raw = read(path, maximum)
    require(digest(raw) == expected, "SOURCE_SHA256_DIFFERS")
    return raw


def load_identity(path):
    module = ModuleType("original_attested_slurm_identity")
    module.__file__ = str(path)
    exec(compile(exact(path, ENTRY_SHA, 65536), str(path), "exec"), vars(module))
    return module


def numeric(text):
    require(re.fullmatch("[0-9]+", text), "INVALID_COUNTER")
    return int(text)


def table(text):
    result = {}
    for line in text.splitlines():
        key, value = line.split()
        require(key not in result, "DUPLICATE_COUNTER")
        result[key] = numeric(value)
    return result


def safe_absolute(text):
    path = PurePosixPath(text)
    require(path.is_absolute() and path.anchor == "/" and ".." not in path.parts, "INVALID_CGROUP_PATH")
    return path


def memory_controller(text):
    rows = [line.split(":", 2) for line in text.splitlines()]
    require(rows and all(len(row) == 3 and row[0].isdecimal() for row in rows), "INVALID_CGROUP_ROWS")
    v1 = [row[2] for row in rows if "memory" in row[1].split(",")]
    v2 = [row[2] for row in rows if row[:2] == ["0", ""]]
    version, matching = (1, v1) if v1 else (2, v2)
    require(len(matching) == 1, "MEMORY_CONTROLLER_UNRESOLVED")
    task = safe_absolute(matching[0])
    _, ordinal = TARGET.split(".")
    parts = task.parts
    matches = [index for index in range(len(parts) - 1) if parts[index:index + 2] ==
               (f"job_{PARENT}", f"step_{ordinal}")]
    require(len(matches) == 1, "TARGET_MEMORY_CGROUP_STEP")
    return version, task, PurePosixPath(*parts[:matches[0] + 2])


def unescape_mount(text):
    return re.sub(r"\\(040|011|012|134)", lambda match: chr(int(match[1], 8)), text)


def mount_candidates(mountinfo, version, step):
    matching = []
    for line in mountinfo.splitlines():
        before, after = line.split(" - ", 1)
        fields, fs = before.split(), after.split()
        require(len(fields) >= 6 and len(fs) >= 3, "INVALID_MOUNTINFO")
        if version == 2 and fs[0] != "cgroup2":
            continue
        if version == 1 and (fs[0] != "cgroup" or "memory" not in fs[2].split(",") + fields[5].split(",")
                             and fs[1] != "memory"):
            continue
        root = safe_absolute(unescape_mount(fields[3]))
        mount = safe_absolute(unescape_mount(fields[4]))
        if step.is_relative_to(root):
            matching.append((root, mount))
    require(matching, "CGROUP_MEMORY_MOUNT_NOT_FOUND")
    depth = max(len(root.parts) for root, _ in matching)
    return {(root, mount) for root, mount in matching if len(root.parts) == depth}


def resolve_step(mountinfo, version, step):
    locations = set()
    for root, mount in mount_candidates(mountinfo, version, step):
        base = Path(str(mount)).resolve(strict=True)
        actual = (base / step.relative_to(root).as_posix()).resolve(strict=True)
        require(actual.is_relative_to(base), "CGROUP_PATH_ESCAPED_MOUNT")
        locations.add((base, actual))
    require(len(locations) == 1, "AMBIGUOUS_CGROUP_MEMORY_MOUNT")
    return locations.pop()


def counter(directory, leaf, *, kind="number", optional=False):
    try:
        data = read(directory / leaf, 65536).decode("ascii").strip()
    except FileNotFoundError:
        if optional:
            return {"available": False}
        raise
    if kind == "pairs":
        return {"available": True, "value": table(data)}
    if kind == "v2_limit" and data == "max":
        return {"available": True, "value": None, "unlimited": True}
    value = numeric(data)
    if kind == "v1_limit" and value >= (1 << 63) - (1 << 20):
        return {"available": True, "value": None, "unlimited": True, "raw": value}
    return {"available": True, "value": value}


def memory_counters(base, step, version):
    if version == 2:
        limit, usage, peak, classes = "memory.max", "memory.current", "memory.peak", V2_CLASSES
        extras = {"events": ("memory.events", "pairs", False),
                  "events_local": ("memory.events.local", "pairs", True),
                  "high_bytes": ("memory.high", "v2_limit", True),
                  "swap_current_bytes": ("memory.swap.current", "number", True),
                  "swap_limit_bytes": ("memory.swap.max", "v2_limit", True)}
    else:
        limit, usage, peak, classes = ("memory.limit_in_bytes", "memory.usage_in_bytes",
                                       "memory.max_usage_in_bytes", V1_CLASSES)
        extras = {"limit_exceed_count": ("memory.failcnt", "number", False),
                  "oom_control": ("memory.oom_control", "pairs", True)}
    limit_kind = f"v{version}_limit"
    values = {"limit_bytes": counter(step, limit, kind=limit_kind), "current_bytes": counter(step, usage),
              "peak_bytes": counter(step, peak, optional=True)}
    values.update({key: counter(step, leaf, kind=kind, optional=optional)
                   for key, (leaf, kind, optional) in extras.items()})
    stat = counter(step, "memory.stat", kind="pairs")["value"]
    values["classified_bytes"] = {key: value for key, value in stat.items() if key in classes}
    ancestors = []
    cursor = step
    while cursor != base:
        cursor = cursor.parent
        ancestors.append({"limit_bytes": counter(cursor, limit, kind=limit_kind, optional=True),
                          "current_bytes": counter(cursor, usage, optional=True)})
    finite = [row["value"] for row in [values["limit_bytes"], *(p["limit_bytes"] for p in ancestors)]
              if row.get("value") is not None]
    return {"cgroup_version": version, "step_memory": values, "visible_ancestor_memory": ancestors,
            "minimum_visible_hard_limit_bytes": min(finite) if finite else None}


def control(*arguments):
    raw = subprocess.check_output(["scontrol", *arguments], timeout=5)
    require(0 < len(raw) <= 131072, "SLURM_RESPONSE_BOUND")
    return raw.decode("utf8")


def known_step(text, step, label, cpu, dcu, mem):
    target = []
    for line in text.splitlines():
        tokens = dict(word.split("=", 1) for word in line.split() if "=" in word)
        if tokens.get("StepId") == step:
            target.append(tokens)
    require(len(target) == 1, "STEP_NOT_UNIQUE_OR_MISSING")
    row = target[0]
    entries = row["TRES"].split(",")
    resources = dict(word.split("=", 1) for word in entries)
    expected = {"cpu": str(cpu), "node": "1", "mem": mem}
    if dcu:
        expected["gres/dcu"] = str(dcu)
    require(row["Name"] == label and row["State"] == "RUNNING" and row["NodeList"] == NODE
            and resources == expected and len(resources) == len(entries), "STEP_RESOURCES_CHANGED")
    return {name: row[name] for name in ("StepId", "Name", "State", "NodeList", "TRES")}


def process_binding(identity, pid, step, boot_sha):
    process = PROC / str(pid)
    require(process.stat().st_uid == os.getuid(), "PID_UID_CHANGED")
    stat = read(process / "stat").decode()
    require(stat.split(" ", 1)[0] == str(pid)
            and stat.rsplit(") ", 1)[1].split()[0] not in {"Z", "X"}, "PID_NOT_LIVE")
    cgroup = read(process / "cgroup")
    require(identity.step_cgroup(cgroup.decode(), step) and digest(read(BOOT, 128)) == boot_sha,
            "PID_CGROUP_OR_BOOT_CHANGED")
    require(identity.task_pid(control("listpids", step), step, pid), "EXACT_PROCESS_NOT_IN_SLURM_STEP")
    return {"pid": pid, "start_ticks": identity.start_ticks(stat), "cgroup_sha256": digest(cgroup)}, cgroup


def worker_receipt():
    expected = json.loads(read(WORKER / "expected.json"))
    raw = read(WORKER / "worker.json")
    current = json.loads(raw)
    require((expected["node"], expected["parent"], expected["request"], expected["label"])
            == (NODE, PARENT, TARGET_REQUEST, TARGET_LABEL) and set(expected["pci"]) == PCIS
            and current["node"] == NODE and current["step"] == TARGET and current["worker_pid"] == WORKER_PID
            and len(current["pci"]) == 8 and set(current["pci"]) == PCIS, "ORIGINAL_STANDBY_WORKER_BINDING")
    return digest(raw)


def observe(args):
    identity = load_identity(args.attested_source)
    step = os.environ["SLURM_JOB_ID"] + "." + os.environ["SLURM_STEP_ID"]
    require(re.fullmatch(PARENT + r"\.[0-9]+", step) and step not in {TARGET, "1103606.2180"}
            and socket.gethostname().split(".", 1)[0] == NODE
            and os.environ.get("SLURM_STEP_GPUS") in {None, "", "(null)"}
            and os.environ.get("SLURM_CPUS_PER_TASK") == "1" and len(os.sched_getaffinity(0)) == 1,
            "ZERO_GRES_OBSERVER_SCOPE")
    boot = args.boot_sha256
    require(SHA.fullmatch(boot) and digest(read(BOOT, 128)) == boot, "ORIGINAL_BOOT_DIFFERS")
    require((args.previous_worker_pid, args.previous_worker_startticks) == (WORKER_PID, WORKER_START),
            "ORIGINAL_2180_OWNER_DIFFERS")
    before, before_mono = time.time(), time.monotonic()
    records = control("show", "step", PARENT, "--oneliner")
    target = known_step(records, TARGET, TARGET_LABEL, 8, 8, "16G")
    mine = known_step(records, step, LABEL, 1, 0, "256M")
    observer, _ = process_binding(identity, os.getpid(), step, boot)
    worker_sha = worker_receipt()
    owner, raw = process_binding(identity, WORKER_PID, TARGET, boot)
    require(owner["start_ticks"] == WORKER_START, "ORIGINAL_WORKER_START_DIFFERS")
    version, _, group = memory_controller(raw.decode())
    base, folder = resolve_step(read(PROC / "self/mountinfo", 262144).decode(), version, group)
    memory = memory_counters(base, folder, version)
    again, _ = process_binding(identity, WORKER_PID, TARGET, boot)
    records = control("show", "step", PARENT, "--oneliner")
    require(owner == again and worker_sha == worker_receipt()
            and target == known_step(records, TARGET, TARGET_LABEL, 8, 8, "16G")
            and mine == known_step(records, step, LABEL, 1, 0, "256M"), "ORIGINAL_WORKER_OR_STEP_DRIFT")
    return {"request": REQUEST, "passed": True, "target": target, "observer": mine,
            "worker_receipt_sha256": worker_sha, "target_owner": owner, "observer_owner": observer,
            "same_verified_boot": True, "start_epoch": before, "end_epoch": time.time(),
            "elapsed_monotonic_s": time.monotonic() - before_mono, "memory": memory,
            "memory_domain": "host_RAM_cgroup", "GPU_calls": 0, "HBM_measured": False,
            "historical_2175_peak_or_OOM_cause_measured": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--attested-source", required=True, type=Path)
    parser.add_argument("--boot-sha256", required=True)
    parser.add_argument("--previous-worker-pid", required=True, type=int)
    parser.add_argument("--previous-worker-startticks", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    exact(Path(__file__).resolve(strict=True), args.source_sha256, 65536)
    with args.out.open("xb") as handle:
        try:
            result = observe(args)
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
            gate = str(error)
            result = {"request": REQUEST, "passed": False, "target_step": TARGET, "GPU_calls": 0,
                      "failure_type": type(error).__name__,
                      "failure_gate": gate if re.fullmatch("[A-Z0-9_]+", gate) else None}
        handle.write((json.dumps(result, allow_nan=False, sort_keys=True) + "\n").encode())
        handle.flush()
        os.fsync(handle.fileno())
    return int(not result["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
