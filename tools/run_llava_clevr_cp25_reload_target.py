"""Run the independent LLaVA CP25 reload once, under the existing owned parent."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time


ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
NODE, PARENT = "f13r1n14", "1103606"
PYTHON = "/usr/local/python3.12/bin/python3"
APPTAINER = "/public/software/apps/apptainer/1.3.4/bin/apptainer"
RESULTS = ROOT / "results"
FRAMEWORK = RESULTS / "training-qwen-a68952c-swift-initial-CPU-v1/framework"
MODEL = RESULTS / "training-llava-original-model18-v1"
SOURCE = ROOT / "incoming/training-llava-clevr-first2-rank-cache-0637f8090c664d27a8a2684b8900fafe/helpers"
TRAINED = RESULTS / "training-llava-clevr-resume25-0e4a27af103c"
LABEL = "llava-original5000-independent-cp25-reload"
COLS = "JobIDRaw,JobName%128,State,ExitCode,ElapsedRaw,NodeList,AllocTRES%160,Start,End"
ENV = {"LC_ALL": "C", "TZ": "UTC", "SLURM_TIME_FORMAT": "standard"}


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def save(path, record):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(record, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


GUEST = r"""
set -eo pipefail
umask 077
ulimit -c 0
source /opt/dtk/env.sh >&2
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MODELSCOPE_OFFLINE=1 PYTHONNOUSERSITE=1
export HOME=/tmp/home TMPDIR=/tmp HF_HOME=/tmp/hf HF_MODULES_CACHE=/tmp/hf-modules
export PYTHONPATH=/swift-runtime/site:/llava-framework:/vision:/llava-helpers OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1
mkdir -p "$HOME" /tmp/hf-modules
exec /usr/bin/python3 -B /reload-stage/llava_clevr_cp25_reload.py \
    --native-loader /llava-helpers/llava_recovered_native_cpu.py \
    --vision-manifest /reload-stage/VISION_FILES.json \
    --native-binding /llava-helpers/NATIVE_BINDING.json \
    --model /llava-base/model --model-receipt /llava-base/receipt.json \
    --resume-receipt /llava-trained/independent-resume25.json \
    --checkpoint /llava-trained/train/checkpoint-25 --out /reload-output/reload.json
"""


def compute(work, request, deadline):
    need(os.environ.get("SLURM_JOB_ID") == PARENT and os.environ.get("SLURMD_NODENAME") == NODE
         and os.environ.get("SLURM_STEP_GPUS") in (None, "", "(null)"), "Exact node and zero GRES required")
    scratch = Path("/tmp") / f"{LABEL}-{request}"
    scratch.mkdir(mode=0o700)
    try:
        mounts = [(work / "source", "/reload-stage", "ro"), (work / "output", "/reload-output", "rw"),
                  (MODEL, "/llava-base", "ro"), (TRAINED, "/llava-trained", "ro"),
                  (SOURCE, "/llava-helpers", "ro"), (FRAMEWORK, "/llava-framework", "ro"),
                  (RESULTS / "training-vlm-swift-offline-runtime-v1", "/swift-runtime", "ro"),
                  (RESULTS / "training-qwen-vision014-CPU-v2/vision", "/vision", "ro"), (scratch, "/tmp", "rw")]
        argv = [APPTAINER, "exec", "--contain", "--no-mount", "home,cwd", "--pwd", "/"]
        for source, target, mode in mounts:
            need(source.is_dir(), f"Missing required mount {target}")
            argv.extend(("--bind", f"{source}:{target}:{mode}"))
        argv += [str(ROOT / "containers/dtk26-sandbox-r1"), "/bin/bash", "-c", GUEST]
        remaining = deadline - time.time() - 40
        need(0 < remaining <= 780, "Original compute deadline exhausted")
        return subprocess.run(["timeout", "--kill-after=5", str(remaining), *argv], check=False).returncode
    finally:
        shutil.rmtree(scratch)


def accounting(label, origin):
    start = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(origin - 120))
    result = subprocess.run(["sacct", "-j", PARENT, "--starttime", start, "-n", "-P", "-o", COLS],
                            capture_output=True, text=True, env={**os.environ, **ENV}, timeout=10, check=True)
    return [row for line in result.stdout.splitlines() if len(row := line.split("|")) == 9 and row[1] == label
            and re.fullmatch(PARENT + r"\.\d+", row[0])]


def launch(work, request, deadline, worker_sha):
    epoch = time.time()
    need(700 <= deadline - epoch <= 820, "Original launch budget is missing or expired")
    worker = work / "source" / "llava_clevr_cp25_reload.py"
    need(worker.resolve(strict=True) == worker and sha256(worker.read_bytes()).hexdigest() == worker_sha,
         "Exact published source required")
    label = f"{LABEL}-{request[:12]}"
    need(not accounting(label, epoch), "This exact scope is already in accounting")
    save(work / "intent.json", dict(epoch=epoch, deadline=deadline, label=label, request=request, worker_sha=worker_sha))
    step = ["srun", "--exact", "--immediate=5", f"--jobid={PARENT}", "-N1", "-n1", "-c6", "--mem=64G",
            "--gres=none", f"--nodelist={NODE}", "--time=00:12:00", f"--job-name={label}",
            PYTHON, "-B", str(Path(__file__).resolve()), "compute", str(work), request, str(deadline), worker_sha]
    with (work / "step.log").open("xb") as stream:
        rc = subprocess.run(["timeout", "--kill-after=5", str(deadline - time.time() - 24), *step],
                            stdout=stream, stderr=subprocess.STDOUT, check=False).returncode
    rows = []
    for _ in range(5):
        if time.time() + 12 >= deadline:
            break
        rows = accounting(label, epoch)
        if rows and len(rows) == 1 and rows[0][2] not in ("RUNNING", "PENDING", "COMPLETING"):
            break
        time.sleep(1)
    path = work / "output" / "reload.json"
    report = None
    if path.is_file() and 0 < path.stat().st_size <= 32768:
        raw = path.read_bytes()
        report = {"SHA256": sha256(raw).hexdigest(), "summary": json.loads(raw)}
    passed = (rc == 0 and len(rows) == 1 and rows[0][2:4] == ["COMPLETED", "0:0"]
              and "gres/" not in rows[0][6] and report is not None and report["summary"]["step"] == rows[0][0]
              and report["summary"]["trained_adapter"]["loaded_tensors"] == 504)
    save(work / "terminal.json", dict(passed=passed, returncode=rc, request=request, accounting=rows, reload=report))
    return 0 if passed else 1


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("mode", choices=("launch", "compute"))
    cli.add_argument("work", type=Path)
    cli.add_argument("request")
    cli.add_argument("deadline", type=float)
    cli.add_argument("worker_sha")
    args = cli.parse_args()
    need(re.fullmatch(r"[0-9a-f]{32}", args.request) and re.fullmatch(r"[0-9a-f]{64}", args.worker_sha)
         and args.work == RESULTS / f"{LABEL}-{args.request}" and args.work.resolve(strict=True) == args.work
         and args.work.stat().st_uid == os.getuid(), "Exact independently owned launch scope required")
    if args.mode == "compute":
        return compute(args.work, args.request, args.deadline)
    return launch(args.work, args.request, args.deadline, args.worker_sha)


if __name__ == "__main__":
    raise SystemExit(main())
