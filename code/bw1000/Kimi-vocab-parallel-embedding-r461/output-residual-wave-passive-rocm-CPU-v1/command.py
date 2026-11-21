"""Render one CPU-only ROCm mount audit in the original DTK environment."""

from hashlib import sha256
import math
from pathlib import Path

from verify import LABEL, NODE, PARENT


ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
OLD_ROOT = ROOT.with_name("hygon-inference-20261013")
ENV_PATH = Path("code/bw1000/Kimi-vocab-parallel-embedding-r461/"
                "output-residual-stage-diagnostic-CPU-v1/CPU_inside.sh")
ENV_SHA256 = "46df727b996ce227b6dc52ffea7c9154e596abc5ea2cced814bbbb88c2122292"
ENV_PREFIX_SHA256 = "456e3ab53c263e6d01857974c2c0abdef21a0ea1aea06f0dfa588b13fe9e11b3"


def command(published_env, staged, out, epoch):
    prefix, marker, _ = published_env.partition(b"left=$(python")
    if (not marker or sha256(published_env).hexdigest() != ENV_SHA256
            or sha256(prefix).hexdigest() != ENV_PREFIX_SHA256):
        raise ValueError("original DTK environment differs")
    if staged != ROOT / "incoming" / LABEL or out != ROOT / "results" / LABEL:
        raise ValueError("repair requires the unique original scope paths")
    if isinstance(epoch, bool) or not math.isfinite(epoch) or epoch <= 0:
        raise ValueError("invalid request epoch")
    inside = prefix.decode("utf8") + '''
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
umask 077
ulimit -f 128
budget=$(python -I -S -B -c 'import math,sys,time
epoch=float(sys.argv[1]); now=time.time()
if not math.isfinite(epoch) or not epoch <= now < epoch+240:
    raise SystemExit("passive ROCm audit epoch expired")
print(int(epoch+280-now))' "$2")
exec /usr/bin/timeout --kill-after=5s "${budget}s" python -I -S -B "$@"
'''
    return ["srun", f"--jobid={PARENT}", f"--job-name={LABEL}", f"--nodelist={NODE}",
            "--exact", "--nodes=1", "--ntasks=1", "--cpus-per-task=4", "--cpu-bind=threads",
            "--mem=8192M", "--gres=none", "--time=00:05:00", "--immediate=5",
            "/public/software/apps/apptainer/1.3.4/bin/apptainer", "exec", "--rocm", "--pwd", "/",
            "--home", str(OLD_ROOT / "build-deps/root") + ":/root",
            "--bind", f"{ROOT}:/refactor:ro",
            "--bind", f"{OLD_ROOT}:/old:ro",
            "--bind", "/opt/hyhal:/opt/hyhal:ro",
            "--bind", f"{out}:{out}:rw",
            "--bind", f"{staged}:{staged}:ro",
            str(ROOT / "containers/dtk26-sandbox-r1"), "/bin/bash", "--noprofile", "--norc",
            "-c", inside, "original678-passive-rocm", str(staged / "verify.py"), repr(epoch)]
