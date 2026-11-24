"""Run one pinned zero-GRES public geometry check beside an existing standby."""

from base64 import b64decode
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import signal
import subprocess
import sys
import time


ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
ACCOUNTING = ("JobIDRaw", "JobName%100", "State", "ExitCode", "ElapsedRaw", "NodeList", "AllocTRES")
CONTAINER = ROOT / "containers/dtk26-sandbox-r1"
RUNTIME = ROOT / "results/training-t06-sdxl-runtime-v1"
APPTAINER = "/public/software/apps/apptainer/1.3.4/bin/apptainer"


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")


def get_files(expected, offered):
    if set(expected) != set(offered):
        raise ValueError("staged file set differs")
    data = {}
    for name, pin in expected.items():
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or path.as_posix() != name:
            raise ValueError("scope has an invalid staged path")
        raw = b64decode(offered[name], validate=True)
        if len(raw) != pin["bytes"] or sha256(raw).hexdigest() != pin["SHA256"]:
            raise ValueError("staged file bytes differ")
        data[name] = raw
    return data


def fields(row):
    return dict(part.split("=", 1) for part in row.split() if "=" in part)


def verify_live(ask, claim):
    parent, node = claim["parent"], claim["node"]
    job = fields(ask("scontrol", "show", "job", parent, "--oneliner"))
    if (job.get("JobState") != "RUNNING" or job.get("NumNodes") != "40"
            or not job.get("UserId", "").startswith("aclsylqidf(")
            or "gres/dcu=320" not in job.get("AllocTRES", "").split(",")):
        raise ValueError("published 320-card parent is unavailable")
    hosts = ask("scontrol", "show", "hostnames", job["NodeList"]).splitlines()
    if len(set(hosts)) != 40 or node not in hosts:
        raise ValueError("CPU node absent from the original parent")
    rows = [fields(row) for row in ask("scontrol", "show", "step", parent, "--oneliner").splitlines()]
    at_node = []
    for row in rows:
        if not row.get("StepId") or row["StepId"] in (parent + ".batch", parent + ".extern"):
            continue
        spec = row.get("NodeList")
        if not spec or spec in ("(null)", "None", "Unknown", "N/A"):
            raise ValueError("unresolved live step placement")
        if node in (spec.split(",") if "[" not in spec else ask("scontrol", "show", "hostnames", spec).splitlines()):
            at_node.append(row)
    standby, = at_node
    if ([standby.get(k) for k in ("StepId", "Name", "State", "NodeList")]
            != [claim["standby_step"], claim["standby_label"], "RUNNING", node]
            or "gres/dcu=8" not in standby.get("TRES", "").split(",")):
        raise ValueError("current standby or node co-residency differs")
    return {k: standby[k] for k in ("StepId", "Name", "State", "NodeList", "TRES")}


def controller(claim, files):
    epoch, anchor = time.time(), time.monotonic()
    deadline = epoch + claim["total_seconds"]
    worker_deadline = epoch + claim["work_seconds"]
    label, parent, node = (claim[k] for k in ("label", "parent", "node"))
    work = ROOT / "results" / label
    if (label, parent, node, claim["gpu_cards"], claim["cpu_cores"], claim["memory_MiB"],
        claim["total_seconds"], claim["work_seconds"], claim["slurm_minutes"]) != (
            "training-hymotion-native-public-geometry-cpu-v1", "1103606", "f13r2n09", 0, 4, 8192, 300, 220, 4):
        raise ValueError("CPU resource scope differs")

    def left(reserve=0):
        remaining = min(deadline - time.time(), claim["total_seconds"] - (time.monotonic() - anchor)) - reserve
        if remaining <= 0:
            raise TimeoutError("the original scope expired")
        return remaining

    def ask(*argv):
        return subprocess.check_output(argv, text=True, timeout=min(9, left(4))).strip()

    def original_row():
        rows = [line.split("|") for line in ask("sacct", "-nP", "-j", parent, "--format=" + ",".join(ACCOUNTING)).splitlines()]
        matched = [row for row in rows if len(row) == len(ACCOUNTING) and row[1] == label]
        if len(matched) > 1:
            raise ValueError("multiple target accounting rows")
        return matched[0] if matched else None

    if work.exists() or original_row():
        raise ValueError("original namespace or label exists")
    standby_before = verify_live(ask, claim)
    for path in (CONTAINER, RUNTIME / "site", Path(APPTAINER), Path("/opt/hyhal")):
        if not path.exists():
            raise ValueError("previously bound target runtime path is absent")
    data = get_files(claim["payload_files"], files)
    work.mkdir(mode=0o700)
    source, evidence = work / "source", work / "evidence"
    source.mkdir()
    evidence.mkdir()
    for name, raw in data.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(raw)
    write_json(source / "CLAIM.json", claim)
    write_json(work / "intent.json", dict(request=claim["request"], epoch=epoch, deadline=deadline,
                                          claim_SHA256=CLAIM_SHA256, node=node, parent=parent,
                                          standby_before=standby_before))
    print(json.dumps(dict(request=claim["request"], event="REMOTE_INTENT_CREATED", epoch=epoch)), flush=True)
    sys.path.insert(0, str(source))
    from slurm_accounting import wait_terminal

    shell = (
        "set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh; "
        "export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 "
        "PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 "
        "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/training-runtime/site; "
        f"export INPUT_CPU_EPOCH={epoch} INPUT_CPU_WORK_DEADLINE={worker_deadline}; "
        "LEFT=$(python3 -B -c " + shlex.quote(f"import time;print(int({worker_deadline}-time.time()))") + "); "
        'test "$LEFT" -gt 20; exec /usr/bin/timeout --kill-after=5s "$LEFT" python3 -B -u /input-stage/inside.py'
    )
    apptainer = [APPTAINER, "exec", "--no-home", "--pwd", "/",
                 "--bind", f"{source}:/input-stage:ro", "--bind", f"{evidence}:/evidence:rw",
                 "--bind", f"{RUNTIME}:/training-runtime:ro", "--bind", "/opt/hyhal:/opt/hyhal:ro",
                 str(CONTAINER), "/bin/bash", "--noprofile", "--norc", "-c", shell]
    argv = ["srun", "--jobid=" + parent, "--job-name=" + label, "--exact", "--nodes=1", "--ntasks=1",
            "--nodelist=" + node, "--cpus-per-task=4", "--mem=8192M", "--gres=none", "--immediate=5", "--time=4", *apptainer]

    with (work / "CPU.log").open("xb") as log:
        child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        timed_out = False
        try:
            child.wait(timeout=left(38))
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait(timeout=5)
    terminal = wait_terminal(original_row, time.monotonic() + min(17, left(12)), interval=.6, max_observations=25)
    row = terminal["row"]
    try:
        standby_after = verify_live(ask, claim)
    except (OSError, ValueError, subprocess.SubprocessError, TimeoutError):
        standby_after = None
    value = None
    p = evidence / "CPU-result.json"
    if p.is_file():
        with p.open("rb") as stream:
            raw = stream.read(16385)
        if len(raw) <= 16384:
            value = dict(SHA256=sha256(raw).hexdigest(), value=json.loads(raw))
    passed = bool(row and row[2:4] == ["COMPLETED", "0:0"] and row[5] == node
                  and "gres/" not in row[6] and child.returncode == 0 and not timed_out
                  and standby_after == standby_before
                  and value and value["value"]["step"] == row[0] and value["value"]["passed"])
    result = dict(request=claim["request"], passed=passed, epoch=epoch, deadline=deadline,
                  child_returncode=child.returncode, timed_out=timed_out, accounting_columns=ACCOUNTING,
                  terminal=terminal, native_result=value, standby_before=standby_before,
                  standby_after=standby_after, GPU_cards=0)
    write_json(work / "outcome.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return int(not passed)


if __name__ == "__main__":
    raise SystemExit(controller(CLAIM, FILES))
