"""Execute one bounded CPU-only native preflight after publication and installation."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import time

from input_contract import NEW_SHA, verify_revision
from slurm_accounting import is_terminal, wait_terminal

ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
LABEL = "training-qwen-treasury-explicit-native-CPU-v2"
RUN = ROOT / "results" / LABEL
PARENT, NODE = "1087656", "a01r4n14"
COLUMNS = "JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def fields(line):
    return dict(word.split("=", 1) for word in line.split() if "=" in word)


def remaining(deadline, monotonic_end, reserve=0):
    value = min(deadline - time.time(), monotonic_end - time.monotonic()) - reserve
    require(math.isfinite(value) and value > 0, "original CPU epoch exhausted")
    return value


def runtime_bind_source(host):
    if host == Path("/opt/hyhal"):
        target = Path("/usr/local/hyhal")
        require(host.resolve(strict=True) == target and target.is_dir() and not target.is_symlink(),
                "system hyhal alias identity changed")
        return target
    require(host.is_dir() and not host.is_symlink(), "required read-only runtime bind missing")
    return host


def main(publication, expected_claim):
    epoch, started = time.time(), time.monotonic()
    deadline, monotonic_end = epoch + 600, started + 600
    source = Path(__file__).resolve().parent
    require(source == RUN / "source" and not RUN.is_symlink(), "unexpected installed namespace")
    require(re.fullmatch(r"[0-9a-f]{40}", publication) is not None, "invalid publication")
    require(re.fullmatch(r"[0-9a-f]{64}", expected_claim) is not None, "invalid claim pin")
    require(digest(source / "CLAIM.json") == expected_claim, "claim differs from publication")
    claim = json.loads((source / "CLAIM.json").read_bytes())
    require((claim["parent"], claim["node"], claim["label"], claim["whole_epoch_seconds"], claim["GPU_cards"])
            == (PARENT, NODE, LABEL, 600, 0), "resource claim changed")
    subprocess.run(["/usr/local/python3.12/bin/python3", "-I", "-S", "-B", str(source / "verify_package.py")],
                   check=True, capture_output=True, timeout=min(15, remaining(deadline, monotonic_end)))
    require(not any((RUN / name).exists() for name in ("intent.json", "outcome.json", "owned-srun.json", "step.log")),
            "original launch namespace exists")
    data = RUN / "inputs.jsonl"
    old_data = ROOT / "results/training-qwen-treasury-pilot-data-v1/data"
    verify_revision((old_data / "inputs.jsonl").read_bytes(), data.read_bytes())
    images = json.loads((source / "IMAGE_PINS.json").read_bytes())
    require(len(images) == 72, "image inventory changed")
    for name, expected in images.items():
        require(re.fullmatch(r"[0-9-]+_[0-9-]+\.png", name) is not None, "invalid image name")
        require(digest(old_data / "train" / name) == expected, "training image changed")

    def query(*args):
        return subprocess.check_output(args, text=True,
                                       timeout=min(12, remaining(deadline, monotonic_end))).strip().splitlines()

    def accounting():
        rows = [line.split("|") for line in query("sacct", "-n", "-P", "-j", PARENT, "-o", COLUMNS)]
        require(all(len(row) == 7 for row in rows), "accounting schema changed")
        matches = [row for row in rows if row[1] == LABEL]
        require(len(matches) <= 1, "duplicate original step")
        return matches[0] if matches else None

    require(accounting() is None, "original step already exists")
    jobs = query("scontrol", "show", "job", PARENT, "--oneliner")
    require(len(jobs) == 1, "parent allocation unavailable")
    job = fields(jobs[0])
    require(job.get("JobState") == "RUNNING" and job.get("NumNodes") == "40"
            and "gres/dcu=320" in job.get("AllocTRES", "").split(","), "parent allocation changed")
    require(NODE in query("scontrol", "show", "hostnames", job["NodeList"]), "node outside parent")
    for line in query("scontrol", "show", "step", PARENT, "--oneliner"):
        row = fields(line)
        step = row.get("StepId", "")
        if step in (PARENT + ".extern", PARENT + ".batch"):
            continue
        require(re.fullmatch(PARENT + r"\.[0-9]+", step) is not None, "unknown active step")
        require(NODE not in query("scontrol", "show", "hostnames", row["NodeList"]), "node carries an active step")

    model = ROOT / "results/training-qwen-original-public-model-v1/model"
    framework = ROOT / "results/training-qwen-a68952c-swift-initial-CPU-v1/framework"
    binds = ((RUN, RUN), (old_data.parent, "/refactor/results/training-qwen-treasury-pilot-data-v1"),
             (model, model), (framework, framework),
             (ROOT / "results/training-vlm-swift-offline-runtime-v1", "/swift-runtime"),
             (ROOT / "results/training-qwen-vision014-CPU-v2/vision", "/vision"),
             (ROOT / "results/training-minicpm-hygon-processor-r2/cpu/runtime-site", "/minicpm-transformers"),
             (Path("/opt/hyhal"), "/opt/hyhal"))
    container = ["/public/software/apps/apptainer/1.3.4/bin/apptainer", "exec", "--contain", "--no-home", "--pwd", "/"]
    for host, target in binds:
        host = runtime_bind_source(host)
        container.extend(("--bind", f"{host}:{target}:ro"))
    seconds = min(450, int(remaining(deadline, monotonic_end, 75)))
    require(seconds > 0, "CPU work budget exhausted")
    process = ["python3", "-B", "-u", str(source / "check_explicit_native_inputs.py"),
               "--dataset", str(data), "--model-dir", str(model)]
    shell = ("set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh; "
             "export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 "
             "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=; "
             "export PYTHONPATH=/vision:/minicpm-transformers:/swift-runtime/site:" + str(framework) + "; "
             f"exec /usr/bin/timeout --kill-after=10s {seconds} " + shlex.join(process))
    container.extend((str(ROOT / "containers/dtk26-sandbox-r1"), "/bin/bash", "--noprofile", "--norc", "-c", shell))
    command = ["srun", "--jobid=" + PARENT, "--job-name=" + LABEL, "--exact", "--nodes=1", "--ntasks=1",
               "--nodelist=" + NODE, "--cpus-per-task=8", "--mem=12288M", "--gres=none",
               "--immediate=5", "--time=8", *container]
    save(RUN / "intent.json", dict(publication=publication, claim_SHA256=expected_claim, epoch=epoch,
         deadline=deadline, command=command, input_SHA256=NEW_SHA, maximum_invocations=1, GPU_cards=0))
    rc, timed_out = None, False
    log_path = RUN / "step.log"
    with log_path.open("xb") as stream:
        try:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            try:
                proc = Path("/proc", str(child.pid), "stat").read_text().rsplit(") ", 1)[1].split()
                save(RUN / "owned-srun.json", dict(PID=child.pid, PGID=int(proc[2]), startticks=int(proc[19])))
            except FileNotFoundError:
                save(RUN / "owned-srun.json", dict(PID=child.pid, exited_before_identity_observation=True))
            try:
                rc = child.wait(timeout=min(seconds + 20, remaining(deadline, monotonic_end, 40)))
            except subprocess.TimeoutExpired:
                timed_out = True
        except OSError:
            timed_out = True
    accounting_errors = []

    def observe_accounting():
        try:
            return accounting()
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            accounting_errors.append(type(error).__name__)
            return None

    visibility = wait_terminal(observe_accounting, min(monotonic_end - 10, time.monotonic() + 35),
                               interval=2, max_observations=18)
    row = visibility["row"]
    metrics = None
    log_size = log_path.stat().st_size
    log_oversized = log_size > 4 * 1024**2
    parse_error = False
    with log_path.open("rb") as stream:
        bounded_log = stream.read(4 * 1024**2)
    for line in bounded_log.splitlines():
        if line.startswith(b"{"):
            try:
                value = json.loads(line)
            except (ValueError, UnicodeError):
                continue
            if isinstance(value, dict) and value.get("state") == "NATIVE_CPU_PREPROCESSING_AND_COLLATION_PASS":
                if metrics is not None:
                    parse_error = True
                metrics = value
    passed = bool(rc == 0 and not timed_out and not parse_error and not log_oversized
                  and row and is_terminal(row) and row[2:4] == ["COMPLETED", "0:0"]
                  and row[5] == NODE and "gres/dcu=" not in row[6] and metrics
                  and metrics["rows"] == 160 and metrics["dataset_SHA256"] == NEW_SHA
                  and metrics["rope_model_parameters_on_meta"] is True and metrics["actual_GPU_cards"] == 0
                  and metrics["actual_updates"] == 0)
    result = dict(passed=passed, epoch=epoch, deadline=deadline, observed_epoch=time.time(), rc=rc,
                  client_timeout=timed_out, accounting_columns=COLUMNS, accounting=visibility,
                  accounting_errors=accounting_errors, metrics=metrics,
                  bounded_log_SHA256=hashlib.sha256(bounded_log).hexdigest(),
                  log_bytes=log_size, log_oversized=log_oversized, duplicate_result=parse_error, GPU_cards=0,
                  current_step_outcome="TERMINAL" if row and is_terminal(row) else "UNKNOWN", formal_training_acceptance=False)
    save(RUN / "outcome.json", result)
    print(json.dumps(result, allow_nan=False))
    return 0 if passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication", required=True)
    parser.add_argument("--claim-sha256", required=True)
    args = parser.parse_args()
    raise SystemExit(main(args.publication, args.claim_sha256))
