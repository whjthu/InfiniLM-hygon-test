"""Own one metadata-only Gemma question ablation in the T02 CPU slot."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
if SCRIPT_DIR.name == "tools":
    sys.path.append(str(SCRIPT_DIR.parent / "code/bw1000/training-qwen-native25-hip-v3"))
import control_gemma_render72_native_cpu as prior
from read_training_slurm_membership import allocated_dcu, snapshot
from slurm_accounting import wait_terminal


LABEL = "training-gemma-usgs72-question-only-native-CPU-v1"
REQUEST = "eef6ad53dab54c3994d70b3adf25a4fd"
PARENT, NODE = prior.PARENT, prior.NODE
RUN = prior.RESULTS / LABEL
STAGE = prior.TARGET / "incoming" / (LABEL + "-reviewed")
TRAIN = prior.DATA / "train-swift.jsonl"
FRAMEWORK = prior.RESULTS / "training-qwen-a68952c-swift-initial-CPU-v1/framework"
LEDGER_PIN = "36361b92773f8eda412132ea176c29e814e4620ea8392ff287e68d854d67f5c9"


def scoped_binds(src, output, train=TRAIN, framework=FRAMEWORK):
    return [(src, "/audit-src", "ro"), (output, "/audit-run", "rw"),
            (train, train.as_posix(), "ro"), (framework, "/input-framework", "ro"),
            (prior.PROCESSOR, "/gemma-processor", "ro"), (prior.RUNTIME, "/swift-runtime", "ro"),
            (prior.TRANSFORMERS, "/minicpm-transformers", "ro"),
            (Path("/usr/local/hyhal"), "/opt/hyhal", "ro")]


def command(src, output, epoch, minutes, publication, manifest):
    env = dict(prior.MASK, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               HF_HUB_DISABLE_TELEMETRY="1", PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1",
               PYTHONPATH="/minicpm-transformers:/swift-runtime/site:/input-framework")
    worker = ["python3", "-B", "-u", "/audit-src/worker.py", "--epoch", str(epoch),
              "--deadline", str(epoch + 390), "--publication", publication, "--source-manifest-sha", manifest]
    shell = ("set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh; export "
             + " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items()) + "; exec " + shlex.join(worker))
    args = ["srun", "--jobid=" + PARENT, "--job-name=" + LABEL, "--exact", "--immediate=5", "-N1", "-n1",
            "--nodelist=" + NODE, "--cpus-per-task=4", "--mem=8192M", "--gres=none", "--time=" + str(minutes),
            prior.APPTAINER, "exec", "--contain", "--no-home", "--pwd", "/"]
    for source, target, mode in scoped_binds(src, output):
        args.extend(("--bind", f"{source}:{target}:{mode}"))
    return [*args, str(prior.TARGET / "containers/dtk26-sandbox-r1"), "/bin/bash",
            "--noprofile", "--norc", "-c", shell]


def qualified(row):
    if not row or not re.fullmatch(PARENT + r"\.\d+", row[0]):
        return False
    return (row[1:4] == [LABEL, "COMPLETED", "0:0"] and row[5] == NODE and row[4].isdecimal()
            and int(row[4]) <= 420 and allocated_dcu(row[6]) == 0
            and row[6] in {"cpu=4,mem=8G,node=1", "cpu=4,mem=8192M,node=1"})


def run(publication, manifest):
    need = prior.require
    need(re.fullmatch(r"[0-9a-f]{40}", publication) and re.fullmatch(r"[0-9a-f]{64}", manifest)
         and Path(__file__).resolve().parent == STAGE, "published scope or staged executable differs")
    epoch, deadline = time.time(), time.monotonic() + 480
    src = STAGE / "src"
    pins = json.loads(prior.pinned(src / "SOURCE_FILES.json", manifest))["files"]
    need(all(Path(n).name == n for n in pins), "nonflat source manifest")
    for name, item in pins.items():
        need(len(prior.pinned(src / name, item["SHA256"])) == item["bytes"], "source length differs")
    claim = json.loads((src / "CLAIM.json").read_bytes())
    need(claim.get("request") == REQUEST and claim["root_GO"] is claim["CPU_GO"] is True
         and claim["GPU_GO"] is claim["training_GO"] is False, "published CPU-only adoption required")
    ledger = json.loads(prior.pinned(STAGE / "ledger.json", LEDGER_PIN))
    roster = {PARENT: {group: item["allocations"][PARENT] for group, item in ledger["groups"].items()}}

    def query(*args):
        left = deadline - time.monotonic()
        need(left > 4, "CPU scope time exhausted")
        return subprocess.check_output(args, text=True, timeout=min(8, left - 2)).strip().splitlines()

    def accounting():
        rows = [line.split("|") for line in query("sacct", "-nP", "-j", PARENT,
                "-S", "2026-11-19T00:00:00", "-o", ",".join(prior.COLUMNS))]
        matched = [row for row in rows if len(row) == 7 and row[1] == LABEL]
        need(len(matched) <= 1, "nonunique original CPU step")
        return matched[0] if matched else None

    before = snapshot(roster, owner_uid=os.getuid(), run_query=query)["parents"][PARENT]
    need(before["owned_by_observer"] and before["ledger_exact"] and before["step_locations_exact"]
         and before["allocated_DCU"] == 320
         and before["state"] == "RUNNING" and NODE in roster[PARENT]["T02"]
         and all(NODE not in step["member_nodes"] for step in before["active_steps"].values()),
         "Gemma T02 node unavailable")
    need(not RUN.exists() and accounting() is None, "original scope already exists")
    for source, _, _ in scoped_binds(src, RUN):
        if source != RUN:
            need(source.resolve(strict=True) == source, "read-only scope has an alias")
    need(TRAIN.stat().st_size == claim["source_train"]["bytes"], "original train extent differs")
    RUN.mkdir(mode=0o700)
    minutes = int(min(420, deadline - time.monotonic() - 38) // 60)
    need(1 <= minutes <= 7 and time.time() < epoch + 60, "scope budget exhausted")
    cmd = command(src, RUN, epoch, minutes, publication, manifest)
    prior.json_once(RUN / "host-intent.json", dict(request=REQUEST, epoch=epoch, publication=publication,
                    source_manifest_SHA256=manifest, deadline=epoch+480, command=cmd, gpu=0,
                    planned_rows=72, replay_allowed=False))
    code = None
    failed = None
    try:
        with (RUN / "step.log").open("xb") as log:
            code = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                  timeout=min(minutes * 60 + 4, deadline - time.monotonic() - 30),
                                  env=dict(os.environ, **prior.MASK)).returncode
    except (OSError, subprocess.SubprocessError) as error:
        failed = type(error).__name__
    try:
        terminal = wait_terminal(accounting, min(time.monotonic()+22, deadline-4), interval=1, max_observations=12)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        terminal = dict(terminal_visible=False, row=None, error_type=type(error).__name__)
    worker_path = RUN / "audit/result.json"
    worker = None
    worker_sha = None
    if worker_path.is_file() and worker_path.stat().st_size < 65536:
        raw = worker_path.read_bytes()
        j = json.loads(raw)
        need(j["request"] == REQUEST and j["epoch"] == epoch and j["planned"] == 72, "different worker output")
        keys = ("passed", "phase", "planned", "attempted", "completed", "failed", "missing", "model_loaded",
                "actual_updates", "actual_media_body_reads", "GPU_init_attempts_blocked", "pixel_fields_absent",
                "failure_type", "maximum_input_tokens", "total_shifted_supervised_tokens", "derived_train")
        worker = {k: j[k] for k in keys if k in j}
        worker_sha = sha256(raw).hexdigest()
    passed = bool(code == 0 and terminal["terminal_visible"] and qualified(terminal["row"])
                  and worker and worker["passed"] and worker["completed"] == 72
                  and worker["model_loaded"] is False and worker["actual_updates"] == 0)
    record = dict(request=REQUEST, publication=publication, manifest_SHA256=manifest, epoch=epoch, client_rc=code,
                  client_error=failed, accounting=terminal, passed=passed, worker=worker, worker_SHA256=worker_sha)
    prior.json_once(RUN / "host-outcome.json", record)
    return record


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--publication", required=True)
    p.add_argument("--manifest", required=True)
    args = p.parse_args()
    result = run(args.publication, args.manifest)
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result["passed"] else 1)
