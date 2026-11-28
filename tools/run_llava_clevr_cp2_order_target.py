"""Launch and account for one zero-GRES native Swift continuation-order check."""

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
APPTAINER = "/public/software/apps/apptainer/1.3.4/bin/apptainer"
HOST_PYTHON = "/usr/local/python3.12/bin/python3"
PARENT = "1103606"
NODE = "f13r1n14"
COLS = "JobIDRaw,JobName%128,State,ExitCode,ElapsedRaw,NodeList,AllocTRES%160,Start,End"
SOURCE_DIGEST = "7d4d493e2d9a46bb5b0457201d744e53af97a3f0c45882a3aec99c584aa75c4e"
TRAIN = ROOT / "incoming/training-llava-clevr-native64-cpu-v1-d14768cf8bac/input"
CP2 = ROOT / "results/training-llava-clevr-first2-c59c90a2356d/train/checkpoint-2"
RUNTIME = ROOT / "results/training-vlm-swift-offline-runtime-v1"
VISION = ROOT / "results/training-qwen-vision014-CPU-v2/vision"
IMAGE = ROOT / "containers/dtk26-sandbox-r1"
ENV = {"LC_ALL": "C", "TZ": "UTC", "SLURM_TIME_FORMAT": "standard"}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def save(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def verified_stage(work, request):
    require(re.fullmatch(r"[a-f0-9]{32}", request) and work.resolve(strict=True) == work
            and work.parent == ROOT / "results" and work.name == f"llava-clevr-cp2-order-{request}"
            and work.stat().st_uid == os.getuid(), "Exact private execution directory required")
    path = work / "source" / "SOURCE_FILES.json"
    manifest = path.read_bytes()
    require(sha256(manifest).hexdigest() == SOURCE_DIGEST, "Frozen CPU source manifest differs")
    members = json.loads(manifest)["files"]
    require(set(members) == {"check_local.py", "contract.py", "cpu_order.py", "HANDOFF.md"}, "Exact CPU source members required")
    for name, config in members.items():
        src = path.parent / name
        body = src.read_bytes()
        require(src.resolve(strict=True) == src and len(body) == config["bytes"]
                and sha256(body).hexdigest() == config["SHA256"], "Frozen CPU source file differs")


GUEST = r"""
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh >&2
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MODELSCOPE_OFFLINE=1 PYTHONNOUSERSITE=1
export HF_HOME=/tmp/hf MODELSCOPE_CACHE=/tmp/modelscope HOME=/tmp/home TMPDIR=/tmp
export GLOO_SOCKET_IFNAME=lo PYTHONPATH=/swift-runtime/site:/vision OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
umask 077
mkdir -p "$HOME" /tmp/elastic /tmp/cpu-ranks
test "$(stat -f -c %T /dev/shm)" = tmpfs
test "$(stat -c %a /dev/shm)" = 1777
shm_owner=$(stat -c %u /dev/shm)
test "$shm_owner" = 0 || test "$shm_owner" = "$(id -u)"
shm=$(mktemp -d "/dev/shm/llava-cp2-order-$1.XXXXXXXX")
test "$(stat -c %a "$shm")" = 700
test ! -e "$shm/store"
/usr/bin/python3 -B /llava-resume/cpu_order.py prepare --output /llava-order --checkpoint /llava-cp2
cat > /tmp/cpu-ranks/rank_bootstrap.py <<'PYTHON'
import os
from pathlib import Path
import runpy
import sys

rank = int(os.environ["LOCAL_RANK"])
if not 0 <= rank < 8 or "transformers" in sys.modules:
    raise ValueError("Expected a new native CPU rank")
cache = Path("/tmp/cpu-ranks") / f"hf-modules-rank-{rank}"
cache.mkdir(mode=0o700)
os.environ["HF_MODULES_CACHE"] = str(cache)
sys.path.insert(0, "/llava-resume")
runpy.run_path("/llava-resume/cpu_order.py", run_name="__main__")
PYTHON
/usr/bin/python3 -B -m torch.distributed.run --nnodes 1 --nproc-per-node 8 --master-addr 127.0.0.1 \
  --master-port "$2" --log-dir /tmp/elastic /tmp/cpu-ranks/rank_bootstrap.py rank \
  --output /llava-order --processor /llava-processor --scratch /tmp/cpu-ranks --rendezvous "$shm/store"
/usr/bin/python3 -B /llava-resume/cpu_order.py merge --output /llava-order
if test -f "$shm/store"; then rm -- "$shm/store"; fi
rmdir -- "$shm"
"""


def compute(work, request):
    require(os.environ["SLURM_JOB_ID"] == PARENT and os.environ["SLURMD_NODENAME"] == NODE
            and os.environ.get("SLURM_STEP_GPUS") in (None, "", "(null)"), "Unexpected allocated CPU node or GRES")
    tmp = Path("/tmp") / f"llava-cp2-order-{request}"
    tmp.mkdir(mode=0o700)
    rc = None
    try:
        binds = [(work / "source", "/llava-resume", "ro"), (work / "authority", "/llava-cpu-authority", "ro"),
                 (work / "processor", "/llava-processor", "ro"), (work / "output", "/llava-order", "rw"),
                 (TRAIN, "/llava-train-only", "ro"), (CP2, "/llava-cp2", "ro"), (RUNTIME, "/swift-runtime", "ro"),
                 (VISION, "/vision", "ro"), (tmp, "/tmp", "rw")]
        argv = [APPTAINER, "exec", "--contain", "--no-mount", "home,cwd", "--pwd", "/"]
        for src, dest, mode in binds:
            argv.extend(("--bind", f"{src}:{dest}:{mode}"))
        port = 30500 + int(request[:4], 16) % 2048
        argv.extend((str(IMAGE), "/usr/bin/timeout", "-k", "12s", "285s", "/bin/bash", "--noprofile", "--norc",
                     "-c", GUEST, "llava-cp2-order", request, str(port)))
        rc = subprocess.run(argv, stdin=subprocess.DEVNULL, check=False).returncode
        return rc
    finally:
        if rc == 0:
            shutil.rmtree(tmp)


def observe(label):
    argv = ["sacct", "-j", PARENT, "-nP", "-o", COLS, "-S", "2026-11-28T00:00:00"]
    until = time.monotonic() + 24
    while time.monotonic() < until:
        p = subprocess.run(argv, env={**os.environ, **ENV}, capture_output=True, text=True, timeout=10)
        if p.returncode == 0:
            rows = [line.split("|") for line in p.stdout.splitlines() if len(line.split("|")) == 9]
            matching = [row for row in rows if row[1] == label]
            if len(matching) == 1 and matching[0][2] not in {"RUNNING", "COMPLETING", "PENDING"}:
                return matching[0]
        time.sleep(1)
    return None


def launch(work, request):
    label = f"llava-clevr-cp2-order-{request[:12]}"
    save(work / "intent.json", dict(request=request, label=label, gpu=0, mode="native Swift order only"))
    argv = ["srun", "--exact", "--immediate=5", "--threads=1", f"--jobid={PARENT}", "--nodes=1", "--ntasks=1",
            f"--nodelist={NODE}", "--cpus-per-task=16", "--mem=32G", "--gres=none", "--time=00:06:00",
            f"--job-name={label}", HOST_PYTHON, "-I", "-S", "-B", str(Path(__file__).resolve()), "compute",
            str(work), request]
    with (work / "slurm.stdout").open("xb") as out, (work / "slurm.stderr").open("xb") as err:
        try:
            rc = subprocess.run(argv, env={**os.environ, **ENV}, stdin=subprocess.DEVNULL,
                                stdout=out, stderr=err, timeout=350).returncode
        except subprocess.TimeoutExpired:
            rc = 124
    row = observe(label)
    logs = {}
    for name in ("slurm.stdout", "slurm.stderr"):
        with (work / name).open("rb") as stream:
            raw = stream.read((1 << 20) + 1)
        require(len(raw) <= 1 << 20, "Bounded CPU diagnostics exceeded")
        signals = ("torch.distributed.elastic", "EXACT_NATIVE_LLAVA_TRAIN_OPTIONS", "FileNotFoundError",
                   "ModuleNotFoundError", "ValueError", "PermissionError", "FileStore", "llava_onevision1_5")
        logs[name] = dict(bytes=len(raw), SHA256=sha256(raw).hexdigest(),
                          signals=[signal for signal in signals if signal.encode() in raw])
    passed = bool(rc == 0 and row and row[2:4] == ["COMPLETED", "0:0"] and row[5] == NODE
                  and row[6] == "cpu=16,mem=32G,node=1")
    result = dict(request=request, slurm_rc=rc, accounting=row, logs=logs, passed=passed,
                  target_GPU=0, updates=0, training_ready=False)
    if passed:
        target = work / "output" / "CP2_NATIVE_RESUME_PLAN.private.json"
        with target.open("rb") as stream:
            plan = stream.read((2 << 20) + 1)
        require(0 < len(plan) <= 2 << 20, "Native order plan extent differs")
        summary = json.loads((work / "slurm.stdout").read_bytes().splitlines()[-1])
        require(summary == dict(planned_forward_slots=736, first_update=3, start_epoch=1, intra_epoch_skip=0,
                                planned_updates=23, native_CPU_metadata_only=True,
                                plan_SHA256=sha256(plan).hexdigest()), "Native finder terminal differs from its plan")
        result["summary"] = summary
    result["rank_errors"] = {i: json.loads(file.read_bytes())["error_type"] for i in range(8)
                             if (file := work / "output" / f"rank-{i}-error.json").is_file()}
    save(work / "terminal.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("launch", "compute"))
    parser.add_argument("work", type=Path)
    parser.add_argument("request")
    args = parser.parse_args()
    verified_stage(args.work, args.request)
    raise SystemExit(compute(args.work, args.request) if args.role == "compute" else launch(args.work, args.request))
