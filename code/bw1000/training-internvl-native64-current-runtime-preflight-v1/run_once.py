"""Run the claimed T03 CPU prerequisite once and preserve the original step evidence."""

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
HERE = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
APPTAINER = "/public/software/apps/apptainer/1.3.4/bin/apptainer"
ACCOUNTING = "JobIDRaw,JobName%100,State%28,ExitCode,ElapsedRaw,NodeList%150,AllocTRES%150"
SHELL = (
    "set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh; "
    "export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 "
    "OPENBLAS_NUM_THREADS=1 TMPDIR=/tmp PYTHONPATH=/swift-runtime/site "
    "CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 "
    "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; exec /usr/bin/python3 -B -u \"$@\""
)


def write_once(path, record):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def query(*command):
    return subprocess.run(command, check=True, capture_output=True, text=True, timeout=7).stdout


def accounting(parent):
    start = "2026-11-01T00:00:00"
    output = query("sacct", "-n", "-P", "--starttime", start, "-j", parent, "-o", ACCOUNTING)
    return [row.split("|") for row in output.splitlines() if row.strip()]


def main():
    claim = json.loads((HERE / "CLAIM.json").read_bytes())
    parent, node, label = (claim[key] for key in ("parent", "node", "label"))
    if HERE != RESULTS / label or claim["gpus"] != 0:
        raise ValueError("Unexpected scope or GPU request")
    job = dict(
        part.split("=", 1)
        for part in query("scontrol", "show", "job", parent, "-o").split()
        if "=" in part
    )
    if job.get("JobState") != "RUNNING" or not job.get("UserId", "").endswith(f"({os.getuid()})"):
        raise ValueError("Original owned parent is not running")
    resources = dict(item.split("=", 1) for item in job["AllocTRES"].split(","))
    if resources.get("gres/dcu") != "320" or resources.get("node") != "40":
        raise ValueError("Parent allocation differs")
    if node not in query("scontrol", "show", "hostnames", job["NodeList"]).splitlines():
        raise ValueError("Claimed node is outside the parent allocation")

    raw_mount = (RESULTS / claim["mount_outcome"]).read_bytes()
    if sha256(raw_mount).hexdigest() != claim["mount_outcome_SHA256"]:
        raise ValueError("Original current-node mount receipt differs")
    mount = json.loads(raw_mount)
    if (mount["step"], mount["node"], mount["read_only_members"], mount["current_compute_mount_verified"]) != (
            claim["mount_step"], node, 17, True):
        raise ValueError("Read-only mounts are unverified for the claimed node")
    rows = accounting(parent)
    prior = [row for row in rows if row[0] == claim["mount_step"]]
    if len(prior) != 1 or prior[0][2:4] != ["COMPLETED", "0:0"] or prior[0][5] != node:
        raise ValueError("Original mount step accounting differs")
    if any(row[1] == label for row in rows):
        raise ValueError("Exact preflight label already has accounting")
    steps = query("scontrol", "show", "step", parent, "-o")
    for line in steps.splitlines():
        step = re.search(r"\bStepId=" + parent + r"\.\d+\b", line)
        nodes = re.search(r"\bNodes=(\S+)", line)
        if step and not nodes:
            raise ValueError("Numbered step lacks a node list")
        if step and node in query("scontrol", "show", "hostnames", nodes[1]).splitlines():
            raise ValueError("Claimed node already hosts a numbered step")

    args = [
        "srun", "--jobid=" + parent, "--job-name=" + label, "--exact", "--immediate=5",
        "--nodes=1", "--ntasks=1", "--nodelist=" + node, "--cpus-per-task=2",
        "--mem=4096M", "--gres=none", "--time=1", APPTAINER, "exec",
        "--rocm", "--contain", "--no-home", "--pwd", "/",
    ]
    binds = [
        (HERE, HERE, "ro"), (HERE / "tmp", "/tmp", "rw"),
        ("/usr/local/hyhal", "/opt/hyhal", "ro"),
        (RESULTS / "training-vlm-swift-offline-runtime-v1", "/swift-runtime", "ro"),
    ]
    for name in (
        "training-qwen-a68952c-swift-initial-CPU-v1",
        "training-internvl-target-native64-CPU-v5",
        "training-vlm-original-assets-v1",
        "training-internvl-rank00-reference-provision-v1/payload",
    ):
        source = RESULTS / name
        binds.append((source, source, "ro"))
    for source, destination, mode in binds:
        args.extend(("--bind", f"{source}:{destination}:{mode}"))
    args.extend((
        str(ROOT / "containers/dtk26-sandbox-r1"), "/bin/bash", "--noprofile", "--norc",
        "-c", SHELL, "preflight", str(HERE / "preflight.py"),
    ))

    begin = time.time()
    write_once(HERE / "intent.json", {
        "request": claim["request"], "parent": parent, "node": node, "epoch": begin,
        "label": label, "cpu": 2, "memory_mib": 4096, "gpus": 0, "slurm_seconds": 60,
    })
    with (HERE / "step.stdout").open("xb") as out, (HERE / "step.stderr").open("xb") as err:
        client = subprocess.run(args, stdout=out, stderr=err, timeout=75)
    matches = []
    for _ in range(4):
        matches = [row for row in accounting(parent) if row[1] == label]
        if len(matches) == 1 and matches[0][2] not in ("RUNNING", "PENDING", "COMPLETING"):
            break
        time.sleep(1)
    raw = (HERE / "step.stdout").read_bytes()
    err = (HERE / "step.stderr").read_bytes()
    report = None
    if client.returncode == 0 and len(raw) <= 16384:
        report = json.loads(raw)
    row = matches[0] if len(matches) == 1 else None
    success = bool(row and report and row[0] == report["step"] and row[2:4] == ["COMPLETED", "0:0"]
                   and row[5] == node and "gres" not in row[6] and report["request"] == claim["request"]
                   and report["node"] == node)
    outcome = {
        "request": claim["request"], "runtime_preflight_pass": success, "training_ready": False,
        "client_rc": client.returncode, "accounting": matches, "elapsed_seconds": time.time() - begin,
        "stdout_SHA256": sha256(raw).hexdigest(), "stderr_SHA256": sha256(err).hexdigest(),
        "compute_report": report,
    }
    if client.returncode != 0:
        known_error = re.search(rb"(?m)^(?:ValueError|ModuleNotFoundError|ImportError|KeyError): [A-Za-z0-9 _.:'-]{1,180}$", err)
        outcome["diagnostic"] = known_error[0].decode("ascii") if known_error else None
    write_once(HERE / "outcome.json", outcome)
    print(json.dumps(outcome, sort_keys=True, allow_nan=False))
    return int(not success)


if __name__ == "__main__":
    raise SystemExit(main())
