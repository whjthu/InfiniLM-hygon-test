"""Construct the one-DCU T09 native replay container command for the owned Slurm step."""
import math
from pathlib import PurePosixPath
import re

from device_entry import NODE, PARENT
from framework_binding import DEPLOYMENT
from run_replay import NATIVE_SOURCE

ROOT = PurePosixPath("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
APPTAINER = "/public/software/apps/apptainer/1.3.4/bin/apptainer"
NATIVE = ROOT / "results/training-hunyuanvideo-original-source-v1/source"
RUNTIME = ROOT / "results/training-t06-sdxl-runtime-v1"
FRAMEWORK = ROOT / "results" / DEPLOYMENT / "framework"
IMAGE = ROOT / "containers/dtk26-sandbox-r1"
GPU_ENV = """set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/training-runtime/site
exec python3 -B -u /t09-run/src/device_entry.py "$@"
"""


def command(*, workspace, audit_sha256, deadline, execution_authority_sha256):
    if not re.fullmatch(r"[0-9a-f]{64}", audit_sha256) or not re.fullmatch(r"[0-9a-f]{64}", execution_authority_sha256) or not math.isfinite(deadline):
        raise ValueError("Invalid audit SHA-256 or native deadline")
    run = PurePosixPath(workspace)
    if run.parent != ROOT / "results" or not run.name.startswith("training-hunyuanvideo-sft-"):
        raise ValueError("Unexpected T09 private workspace")
    apptainer = [
        APPTAINER, "exec", "--rocm", "--no-home", "--pwd", "/",
        "--bind", f"{ROOT}:/refactor:ro",
        "--bind", f"{NATIVE}:{NATIVE_SOURCE.as_posix()}:ro",
        "--bind", f"{RUNTIME}:/training-runtime:ro",
        "--bind", f"{run}:/t09-run:rw",
        "--bind", f"{run}/src:/t09-run/src:ro",
        "--bind", f"{FRAMEWORK}:/t09-framework:ro",
        "--bind", "/opt/hyhal:/opt/hyhal:ro", str(IMAGE),
        "/bin/bash", "--noprofile", "--norc", "-c", GPU_ENV, "t09-replay",
        "--results", "/refactor/results",
        "--workspace", "/t09-run",
        "--framework", "/t09-framework",
        "--launch", "/t09-run/src/ORIGINAL_LAUNCH.json",
        "--audit", "/t09-run/src/transformer-audit.json",
        "--audit-sha256", audit_sha256,
        "--execution-authority", "/t09-run/src/execution-authority.json",
        "--execution-authority-sha256", execution_authority_sha256,
        "--gate", f"/t09-run/gpu-gate/{NODE}.json",
        "--lock-root", "/t09-run/locks",
        "--deadline", str(deadline),
    ]
    return [
        "srun", f"--jobid={PARENT}", f"--job-name={run.name}-gpu", "--exact", "--nodes=1",
        "--ntasks=1", f"--nodelist={NODE}", "--immediate=5", "--gres=dcu:1",
        "--cpus-per-task=8", "--mem=128000M", "--time=20",
        "/usr/local/python3.12/bin/python3", "-B", str(run / "src/host_entry.py"),
        str(run), "gpu-gate", *apptainer,
    ]
