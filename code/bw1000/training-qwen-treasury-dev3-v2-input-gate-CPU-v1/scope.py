"""Three CPU-only filesystem-entry checks; no GPU or subsequent stage submission."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
RESULTS = ROOT / "results"
LABEL = "training-qwen-treasury-dev3-v2-input-gate-CPU-v1"
RUN = RESULTS / LABEL
SRC = RUN / "source"
PREP = RESULTS / "training-qwen-treasury-explicit-dev-CPU-v1"
MEDIA = RESULTS / "training-qwen-treasury-dev-native-CPU-v1/dev/media"
MODEL = RESULTS / "training-qwen-original-public-model-v1/model"
ARMS = {"base": ("1087656", "a01r4n14"), "old_domain": ("1087656", "a01r4n16"), "domain": ("1103606", "f13r1n10")}
ADAPTERS = {"base": None,
 "old_domain": "/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-qwen-treasury-sft-v3/continuous/train/checkpoint-5",
 "domain": "/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-qwen-treasury-explicit-sft-v1/continuous/train/checkpoint-5"}
COLS = "JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES"
INVENTORY_SHA = "07d906e3ed840d828ee41509cae21b45a754ddef0825d411ca3d8216f606b0d8"


def require(ok, why):
    if not ok:
        raise ValueError(why)


def read(path):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 1024**2, "unsafe metadata")
    return json.loads(path.read_bytes())


def sha(path):
    require(path.is_file() and not path.is_symlink(), "missing or aliased input")
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024**2), b""):
            h.update(data)
    return h.hexdigest()


def remaining(deadline, mono_end, wall_now, mono_now, requested, reserve=0):
    values = (deadline, mono_end, wall_now, mono_now, requested, reserve)
    require(all(math.isfinite(x) for x in values) and requested > 0 and reserve >= 0, "invalid original clock")
    value = min(deadline-wall_now, mono_end-mono_now)-reserve
    require(value > 0, "original CPU epoch exhausted")
    return min(requested, value)


def stage_budget(deadline, mono_end, wall_now, mono_now):
    slurm = 60 * (int(remaining(deadline, mono_end, wall_now, mono_now, 120, 45)) // 60)
    require(slurm >= 120, "insufficient original budget for fixed90second input gate")
    return dict(inside_seconds=90, kill_grace_seconds=5, Slurm_seconds=120, Slurm_time="00:02:00",
                client_seconds=125)


def runtime_bind_source(host):
    host = Path(host)
    if host == Path("/opt/hyhal"):
        target = Path("/usr/local/hyhal")
        require(host.resolve(strict=True) == target and target.is_dir() and not target.is_symlink(), "system hyhal alias changed")
        return target
    require(host.is_dir() and not host.is_symlink(), "runtime bind missing or linked")
    return host
