"""Run the pinned development-input preflight in one CPU-only Slurm step."""

import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time


ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
RESULTS = ROOT / "results"
RUN = RESULTS / "training-qwen-treasury-dev-native-CPU-v1"
PARENT, NODE, LABEL = "1087656", "a01r4n14", RUN.name
COLUMNS = "JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES"


def query(*args):
    return subprocess.check_output(args, text=True, timeout=12).strip().splitlines()


def accounting():
    rows = [line.split("|") for line in query("sacct", "-n", "-P", "-j", PARENT, "-o", COLUMNS)]
    return [row for row in rows if len(row) == len(COLUMNS.split(",")) and row[1] == LABEL]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    intent = json.loads((RUN / "intent.json").read_bytes())
    claim_path = RUN / "claim.json"
    claim = json.loads(claim_path.read_bytes())
    if sha(claim_path) != intent["claim_SHA256"] or (claim["parent"], claim["node"], claim["label"]) != (PARENT, NODE, LABEL):
        raise ValueError("CPU intent differs from published scope")
    for name, expected in claim["source_SHA256"].items():
        if sha(RUN / "source" / name) != expected:
            raise ValueError("staged source differs")
    if sha(RUN / "inventory.json") != claim["inventory_SHA256"] or accounting():
        raise ValueError("input inventory changed or original accounting already exists")
    alloc = query("squeue", "-h", "-j", PARENT, "-o", "%T|%D|%N")
    state, count, nodes = alloc[0].split("|")
    if len(alloc) != 1 or (state, count) != ("RUNNING", "40") or NODE not in query("scontrol", "show", "hostnames", nodes):
        raise ValueError("allocated node changed")
    for line in query("scontrol", "show", "step", PARENT, "--oneliner"):
        fields = dict(word.split("=", 1) for word in line.split() if "=" in word)
        step = fields.get("StepId", "")
        if step.startswith(PARENT + ".") and step.rsplit(".", 1)[1].isdigit():
            if NODE in query("scontrol", "show", "hostnames", fields["NodeList"]):
                raise ValueError("development node already carries a compute step")

    model = RESULTS / "training-qwen-original-public-model-v1/model"
    runtime = ["/public/software/apps/apptainer/1.3.4/bin/apptainer", "exec", "--contain", "--no-home", "--pwd", "/"]
    binds = ((RUN, RUN), (model, model), (RESULTS / "training-vlm-swift-offline-runtime-v1", "/swift-runtime"),
             (RESULTS / "training-qwen-vision014-CPU-v2/vision", "/vision"),
             (RESULTS / "training-minicpm-hygon-processor-r2/cpu/runtime-site", "/minicpm-transformers"),
             (Path("/opt/hyhal"), "/opt/hyhal"))
    for host, target in binds:
        runtime.extend(("--bind", f"{host}:{target}:ro"))
    process = ["python3", "-B", "-u", str(RUN / "source/check_qwen_holdout_native.py"),
               "--root", str(RUN / "dev"), "--inventory", str(RUN / "inventory.json"), "--model-dir", str(model)]
    seconds = min(440, int(intent["deadline"] - time.time()) - 65)
    if seconds <= 0:
        raise TimeoutError("CPU scope expired before submission")
    shell = ("set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh; "
             "export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 "
             "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=; "
             "export PYTHONPATH=/vision:/minicpm-transformers:/swift-runtime/site; "
             f"exec /usr/bin/timeout --kill-after=10s {seconds} " + shlex.join(process))
    runtime.extend((str(ROOT / "containers/dtk26-sandbox-r1"), "/bin/bash", "--noprofile", "--norc", "-c", shell))
    cmd = ["srun", "--jobid=" + PARENT, "--job-name=" + LABEL, "--exact", "--nodes=1", "--ntasks=1",
           "--nodelist=" + NODE, "--cpus-per-task=8", "--mem=12288M", "--gres=none", "--immediate=5", "--time=8", *runtime]
    log = RUN / "step.log"
    rc = None
    with log.open("xb") as stream:
        try:
            rc = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, timeout=seconds + 30).returncode
        except subprocess.TimeoutExpired:
            pass
    rows = []
    until = min(intent["deadline"], time.time() + 25)
    while time.time() < until:
        rows = accounting()
        if len(rows) == 1 and rows[0][2].split()[0].rstrip("+") not in {"RUNNING", "PENDING", "COMPLETING", "CONFIGURING"}:
            break
        time.sleep(1)
    metrics = None
    for line in log.read_bytes().splitlines():
        if line.startswith(b'{"actual_GPU_cards":'):
            metrics = json.loads(line)
    complete = len(rows) == 1 and rows[0][2:4] == ["COMPLETED", "0:0"] and rows[0][5] == NODE
    passed = bool(rc == 0 and complete and "gres/dcu=" not in rows[0][6] and metrics
                  and metrics["state"] == "NATIVE_CPU_DEVELOPMENT_INFERENCE_INPUTS_PASS"
                  and [metrics["arms"][arm]["rows"] for arm in ("visual", "text_only")] == [108, 108])
    with (RUN / "outcome.json").open("x") as stream:
        json.dump({"passed": passed, "accounting_columns": COLUMNS, "rows": rows,
                   "rc": rc, "metrics": metrics, "log_SHA256": sha(log)}, stream)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
