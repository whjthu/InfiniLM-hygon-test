"""Exact scoped argv; actual Slurm limits shrink with the original epoch."""
import math
from contract import LABEL, NATIVE, NATIVE_ALIAS, NODE, PARENT, ROOT, RUN, RUNTIME, WORK, need
from flash_compat import ALIAS, BACKING

ENV = """set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/training-runtime/site
exec python3 -B -u "$@"
"""


def allowance(budget, phase):
    if phase == "gpu":
        seconds = math.floor(min(budget.left(WORK) - 10, budget.left() - 180, 1200))
    else:
        need(phase in ("pre", "post"), "Unknown phase")
        seconds = math.floor(min(45, budget.left() - 35))
    need(seconds > 0, "No original phase budget remains")
    return seconds


def argv(phase, seconds, origin, preflight_sha, manifest_sha):
    need(type(seconds) is int and 0 < seconds <= (1200 if phase == "gpu" else 45), "Invalid actual Slurm budget")
    time_limit = f"{seconds // 3600:02}:{seconds // 60 % 60:02}:{seconds % 60:02}"
    base = ["srun", f"--jobid={PARENT}", f"--job-name={LABEL}-{phase}", "--exact", "--nodes=1",
            "--ntasks=1", f"--nodelist={NODE}", "--immediate=5", "--kill-on-bad-exit=1",
            "--gres=dcu:1" if phase == "gpu" else "--gres=none",
            "--cpus-per-task=8" if phase == "gpu" else "--cpus-per-task=1",
            "--mem=128000M" if phase == "gpu" else "--mem=1024M", "--time=" + time_limit,
            "/usr/local/python3.12/bin/python3", "-B", str(RUN / "src/host_entry.py"),
            "gpu-gate" if phase == "gpu" else phase]
    if phase == "gpu":
        base += ["/public/software/apps/apptainer/1.3.4/bin/apptainer", "exec", "--rocm", "--no-home", "--pwd", "/",
                 "--bind", f"{ROOT}:{ROOT}:ro", "--bind", f"{RUNTIME}:/training-runtime:ro",
                 "--bind", f"{NATIVE}:{NATIVE_ALIAS}:ro",
                 "--bind", f"{BACKING}:{ALIAS}:ro",
                 "--bind", f"{RUN}:{RUN}:rw", "--bind", f"{RUN}/src:{RUN}/src:ro",
                 "--bind", "/opt/hyhal:/opt/hyhal:ro", str(ROOT / "containers/dtk26-sandbox-r1"),
                 "/bin/bash", "--noprofile", "--norc", "-c", ENV, LABEL,
                 str(RUN / "src/native_worker.py"), "--epoch", str(origin),
                 "--preflight-sha256", preflight_sha, "--source-manifest-sha256", manifest_sha]
    return base
