"""Bind the Z-Image supervisor to its actual Slurm task before any GPU calls."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import socket
import subprocess
import sys
import time


PROJECT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
PARENT, NODE, OLD_STEP = "1103606", "f12r2n11", "1103606.2075"
REQUEST = "e23d24ea0c9c4d00af06154d82dc22e8"
LABEL = "O01-zimage-lt-eight-card-supervisor-v1"
PROBE_REQUEST = "af580d6d58004e5ebfe582d5d53d55ed"
PROBE_LABEL = "O01-zimage-hybrid-pid-none-af580d6d5800"
RUNTIME_SHA = "7ba1d9355838c09d41176f67708d892528832971f83f036be63d6ddacbbdb35d"
NATIVE_SHA = "f735cc99e10d0c958b3736fde852bbf9e1f9747827225f1375f0214726279a07"
BOOT = Path("/proc/sys/kernel/random/boot_id")
APPTAINER = "/public/software/apps/apptainer/1.3.4/bin/apptainer"
DTK = PROJECT / "containers/dtk26-sandbox-r1"
BUILD = PROJECT / "results/training-zimage-lt-native-build-fd31777395fb474f86dedacef84dcce9"
SITE = PROJECT / "results/training-t06-sdxl-runtime-v1"
SHELL = (
    "set -euo pipefail; source /opt/dtk/env.sh; "
    "export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 "
    "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/training-runtime/site; "
    "unset ROCBLAS_LAYER HIPBLASLT_LOG_LEVEL HIPBLASLT_LOG_MASK; "
    'exec python3 -B /pair-entry.py "$@"'
)


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def read(path, size=16 << 10):
    require(not path.is_symlink(), f"unexpected symbolic link: {path.name}")
    with path.open("rb") as handle:
        data = handle.read(size + 1)
    require(0 < len(data) <= size, f"invalid file extent: {path.name}")
    return data


def bound(path, expected, size=16 << 10):
    data = read(path, size)
    require(sha256(data).hexdigest() == expected, f"source or receipt hash differs: {path.name}")
    return data


def save(path, values):
    raw = (json.dumps(values, sort_keys=True, allow_nan=False) + "\n").encode()
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return sha256(raw).hexdigest()


def live_step(output, step, label=LABEL, cards=8):
    matches = []
    for line in output.splitlines():
        fields = dict(token.split("=", 1) for token in line.split() if "=" in token)
        if fields.get("StepId") == step:
            matches.append(fields)
    require(len(matches) == 1, "exact Slurm step absent or duplicated")
    row = matches[0]
    resources = set(row.get("TRES", "").split(","))
    expected_gres = {f"gres/dcu={cards}"} if cards else set()
    require(row.get("NodeList") == NODE and row.get("Name") == label and row.get("State") == "RUNNING"
            and {"cpu=12", "node=1"} <= resources
            and {item for item in resources if item.startswith("gres/")} == expected_gres,
            "actual Slurm step resources differ")
    return {key: row[key] for key in ("StepId", "Name", "NodeList", "State", "TRES")}


def start_ticks(stat_line):
    tail = stat_line.rsplit(") ", 1)[1].split()
    ticks = int(tail[19])
    require(ticks > 0, "invalid process start identity")
    return ticks


def step_cgroup(lines, step):
    job, ordinal = step.split(".")
    for line in lines.splitlines():
        parts = PurePosixPath(line.split(":", 2)[-1]).parts
        if any(parts[index:index + 2] == (f"job_{job}", f"step_{ordinal}") for index in range(len(parts))):
            return True
    return False


def process_identity():
    step = os.environ["SLURM_JOB_ID"] + "." + os.environ["SLURM_STEP_ID"]
    now = time.time()
    affinity = sorted(os.sched_getaffinity(0))
    boot = sha256(read(BOOT, 128)).hexdigest()
    cgroup = read(Path("/proc/self/cgroup"))
    require(re.fullmatch(PARENT + r"\.[0-9]+", step) and step != OLD_STEP
            and socket.gethostname().split(".")[0] == NODE
            and os.environ.get("SLURM_CPUS_PER_TASK") == "12" and len(affinity) == 12,
            "task identity or CPU affinity differs")
    return {"step": step, "node": NODE, "affinity": affinity, "boot_SHA256": boot, "uid": os.getuid(),
            "cgroup_SHA256": sha256(cgroup).hexdigest(), "cgroup_names_step": step_cgroup(cgroup.decode(), step),
            "pid": os.getpid(), "start_ticks": start_ticks(Path("/proc/self/stat").read_text()), "epoch": now}


def local_identity(scope):
    identity = process_identity()
    require(scope["request"] == REQUEST and scope["label"] == LABEL and scope["fresh_root_admitted"] is True
            and scope["parent"] == PARENT and scope["node"] == NODE
            and scope["boot_id_SHA256"] == identity["boot_SHA256"], "scope or boot differs")
    require(scope["epoch"] <= identity["epoch"] < scope["deadline_epoch"] - 185
            and scope["deadline_epoch"] == scope["epoch"] + 240, "insufficient original epoch remaining")
    return identity


def task_pid(output, step, pid):
    job, ordinal = step.split(".")
    return any(row.split()[:3] == [str(pid), job, ordinal] for row in output.splitlines())


def attest_step(identity, label, cards):
    step = identity["step"]
    control = subprocess.check_output(["scontrol", "show", "step", PARENT, "--oneliner"], text=True, timeout=5)
    row = live_step(control, step, label, cards)
    pids = subprocess.check_output(["scontrol", "listpids", step], text=True, timeout=5)
    return row, task_pid(pids, step, identity["pid"])


def probe(args):
    require(os.environ.get("SLURM_STEP_GPUS") in (None, "", "(null)"), "zero-GRES probe has a GPU binding")
    identity = process_identity()
    row, matched = attest_step(identity, PROBE_LABEL, 0)
    save(args.run / "zero-GRES-owner.json", {"request": PROBE_REQUEST, "owner": identity,
                                            "slurm_listpids_matched": matched, "live_slurm": row})
    return int(not matched)


def host(args):
    output = args.run.resolve(strict=True)
    runtime = args.runtime.resolve(strict=True)
    entry = Path(__file__).resolve(strict=True)
    bound(entry, args.entry_sha256, 64 << 10)
    scope = json.loads(bound(output / "scope.json", args.scope_sha256))
    identity = local_identity(scope)
    row, matched = attest_step(identity, LABEL, 8)
    require(matched, "actual Slurm task-process membership differs")
    bound(runtime / "RUNTIME_FILES.json", RUNTIME_SHA)
    bound(BUILD / "libzimage_lt_probe.so", NATIVE_SHA, 17760)
    require(Path("/opt/hyhal").samefile("/usr/local/hyhal"), "target hyhal alias identity changed")
    private_receipt = output / "attested-host-owner.json"
    owner_pin = save(private_receipt, {"request": REQUEST, "scope_SHA256": args.scope_sha256,
                                     "runtime_SHA256": RUNTIME_SHA, "entry_SHA256": args.entry_sha256,
                                     "owner": identity, "live_slurm": row, "slurm_listpids_matched": True})
    cmd = [APPTAINER, "exec", "--rocm", "--no-home", "--pwd", "/",
           "--bind", f"{entry}:/pair-entry.py:ro", "--bind", f"{output}:/pair-run:rw",
           "--bind", f"{runtime}:/pair-hybrid:ro", "--bind", f"{BUILD}:/pair-build:ro",
           "--bind", f"{SITE}:/training-runtime:ro", "--bind", "/usr/local/hyhal:/opt/hyhal:ro",
           str(DTK), "/bin/bash", "--noprofile", "--norc", "-c", SHELL, "hybrid", "inside",
           "--run", "/pair-run", "--runtime", "/pair-hybrid", "--scope-sha256", args.scope_sha256,
           "--owner-sha256", owner_pin]
    os.execv(APPTAINER, cmd)


def inside(args):
    scope_file = args.run / "scope.json"
    scope = json.loads(bound(scope_file, args.scope_sha256))
    current = local_identity(scope)
    attest = json.loads(bound(args.run / "attested-host-owner.json", args.owner_sha256))
    previous = attest["owner"]
    bound(Path(__file__).resolve(strict=True), attest["entry_SHA256"], 64 << 10)
    require(attest["request"] == REQUEST and attest["scope_SHA256"] == args.scope_sha256
            and attest["runtime_SHA256"] == RUNTIME_SHA and attest["slurm_listpids_matched"] is True
            and attest["live_slurm"]["StepId"] == current["step"]
            and all(current[key] == previous[key] for key in
                    ("step", "node", "boot_SHA256", "uid", "affinity", "cgroup_SHA256", "cgroup_names_step")),
            "container changed the attested task identity or device cgroup")
    owner_stat = Path(f"/proc/{previous['pid']}/stat").read_text()
    require(start_ticks(owner_stat) == previous["start_ticks"], "host task process was replaced")
    bound(args.runtime / "RUNTIME_FILES.json", RUNTIME_SHA)
    save(args.run / "attested-container-owner.json",
         {"request": REQUEST, "host_owner_SHA256": args.owner_sha256, "owner": current})
    os.execv(sys.executable, [sys.executable, "-B", str(args.runtime / "supervisor.py"),
                             "--scope", str(scope_file), "--scope-sha256", args.scope_sha256,
                             "--runtime-sha256", RUNTIME_SHA, "--out", str(args.run / "supervised")])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("host", "inside", "probe"))
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--scope-sha256")
    parser.add_argument("--entry-sha256")
    parser.add_argument("--owner-sha256")
    args = parser.parse_args()
    if args.action == "probe":
        return probe(args)
    if args.runtime is None or args.scope_sha256 is None:
        parser.error("host and inside need runtime and scope inputs")
    return host(args) if args.action == "host" else inside(args)


if __name__ == "__main__":
    raise SystemExit(main())
