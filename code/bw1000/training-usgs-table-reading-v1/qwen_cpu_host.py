"""Allocate one bounded CPU-only Slurm step for the published Qwen input check."""
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time


ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
RESULTS = ROOT / "results"
LABEL = "training-qwen-usgs-table-native-input-CPU-v1"
PARENT = "1103606"
NODE = "f13r1n10"
RUN = RESULTS / LABEL
DATA = RESULTS / "training-domain-usgs-table-reading-v1"


def query(*args):
    return subprocess.check_output(args, timeout=9, text=True).splitlines()


def fields(line):
    return dict(part.split("=", 1) for part in line.split() if "=" in part)


def write(path, payload):
    with path.open("x", encoding="utf8") as handle:
        json.dump(payload, handle, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())


def present(path):
    return path.is_dir() and not path.is_symlink()


def main(payload):
    epoch = time.time()
    claim = payload["claim"]
    assert (claim["label"], claim["parent"], claim["node"], claim["allocated_DCU"]) == (LABEL, PARENT, NODE, 0)
    assert claim["CPU_GO"] and not claim["GPU_GO"] and re.fullmatch("[0-9a-f]{32}", claim["request"])
    assert not RUN.exists()
    assert not any(row.split("|")[1:2] == [LABEL]
                   for row in query("sacct", "-nP", "-j", PARENT, "--format=JobIDRaw,JobName%100"))
    [raw] = query("scontrol", "show", "job", PARENT, "-o")
    job = fields(raw)
    assert (job["JobState"], job["NumNodes"]) == ("RUNNING", "40")
    assert job["UserId"].endswith(f"({os.getuid()})") and "gres/dcu=320" in job["AllocTRES"].split(",")
    assert NODE in query("scontrol", "show", "hostnames", job["NodeList"])
    steps = query("scontrol", "show", "step", PARENT, "-o")
    assert steps
    for line in steps:
        step = fields(line)
        suffix = step["StepId"].removeprefix(PARENT + ".")
        if suffix in ("extern", "batch"):
            continue
        assert suffix.isdecimal() and NODE not in query("scontrol", "show", "hostnames", step["NodeList"])
    staged = json.loads((DATA / "stage-receipt.json").read_bytes())
    assert staged["request"] == claim["data_placement_request"] and staged["staged"]
    assert staged["output_identity_SHA256"] == claim["output_identity_SHA256"]
    train = DATA / "data/train/swift.jsonl"
    assert hashlib.sha256(train.read_bytes()).hexdigest() == claim["dataset_SHA256"]
    worker = base64.b64decode(payload["worker"], validate=True)
    assert hashlib.sha256(worker).hexdigest() == claim["source_body_SHA256"]

    model = RESULTS / "training-qwen-original-public-model-v1/model"
    framework = RESULTS / "training-qwen-a68952c-swift-initial-CPU-v1/framework"
    binds = ((RUN, RUN), (DATA / "data/train", DATA / "data/train"), (model, model), (framework, framework),
             (RESULTS / "training-vlm-swift-offline-runtime-v1", "/swift-runtime"),
             (RESULTS / "training-qwen-vision014-CPU-v2/vision", "/vision"),
             (RESULTS / "training-minicpm-hygon-processor-r2/cpu/runtime-site", "/minicpm-transformers"),
             (Path("/usr/local/hyhal"), "/opt/hyhal"))
    assert all(present(host) for host, _ in binds[1:])
    assert Path("/opt/hyhal").resolve(strict=True) == Path("/usr/local/hyhal")
    container = ["/public/software/apps/apptainer/1.3.4/bin/apptainer", "exec", "--contain", "--no-home", "--pwd", "/"]
    for host, target in binds:
        container.extend(("--bind", f"{host}:{target}:ro"))
    RUN.mkdir(mode=0o700)
    with (RUN / "check_qwen_native_train_cpu.py").open("xb") as handle:
        handle.write(worker)
    remaining = int(epoch + claim["max_deadline_s"] - time.time())
    assert remaining >= 120
    process = ["python3", "-B", "-u", str(RUN / "check_qwen_native_train_cpu.py"),
               "--dataset", str(train), "--model-dir", str(model)]
    shell = ("set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh; "
             "export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 "
             "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=; "
             f"export PYTHONPATH=/vision:/minicpm-transformers:/swift-runtime/site:{framework}; "
             f"exec /usr/bin/timeout --kill-after=10s {remaining - 40} " + shlex.join(process))
    container.extend((str(ROOT / "containers/dtk26-sandbox-r1"), "/bin/bash", "--noprofile", "--norc", "-c", shell))
    command = ["srun", "--jobid=" + PARENT, "--job-name=" + LABEL, "--exact", "--nodes=1", "--ntasks=1",
               "--nodelist=" + NODE, "--cpus-per-task=8", "--mem=12288M", "--gres=none", "--immediate=5",
               "--time=" + str(remaining // 60), *container]
    receipt = dict(request=claim["request"], publication=payload["publication"],
                   epoch=epoch, deadline=epoch + claim["max_deadline_s"], dataset_SHA256=claim["dataset_SHA256"],
                   node=NODE, parent=PARENT, GPU=0)
    write(RUN / "intent.json", receipt)
    with (RUN / "step.log").open("xb") as log:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
    receipt["owned_srun_pid"] = child.pid
    write(RUN / "launch.json", receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main(PAYLOAD)
