"""One independent zero-GPU metadata gate; original GPU-v3 runtime paths stay frozen."""
from pathlib import Path, PurePosixPath
import math
import json
import hashlib

ROOT = PurePosixPath("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
RESULTS = ROOT / "results"
LABEL = "training-qwen-treasury-capacity-host-gate-v1"
RUN = Path(str(RESULTS / LABEL))
SRC = RUN / "source"
ORIGINAL = RESULTS / "training-qwen-treasury-explicit-dev3-gpu-v3"
PREP = RESULTS / "training-qwen-treasury-explicit-dev-CPU-v1"
MEDIA = RESULTS / "training-qwen-treasury-dev-native-CPU-v1/dev/media"
MODEL = RESULTS / "training-qwen-original-public-model-v1/model"
IMAGE = ROOT / "containers/dtk26-sandbox-r1"
FLAGS = ["/public/software/apps/apptainer/1.3.4/bin/apptainer", "exec", "--contain", "--rocm", "--no-home", "--pwd", "/"]
ARMS = {"base": dict(parent="1087656", node="a01r4n14", adapter=None),
        "old_domain": dict(parent="1087656", node="a01r4n16", adapter=str(RESULTS/"training-qwen-treasury-sft-v3/continuous/train/checkpoint-5")),
        "domain": dict(parent="1103606", node="f13r1n10", adapter=str(RESULTS/"training-qwen-treasury-explicit-sft-v1/continuous/train/checkpoint-5"))}
COLS = "JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES"


def require(ok, why):
    if not ok:
        raise ValueError(why)


def original_binds(arm):
    old_run, old_src = ORIGINAL/"arms"/arm, ORIGINAL/"source"
    binds = [(old_run, old_run, "rw"), (old_src, old_src, "ro"), (MODEL, MODEL, "ro"), (PREP, PREP, "ro"),
             (MEDIA, PREP/"dev/media", "ro"),
             (RESULTS/"training-vlm-swift-offline-runtime-v1", "/swift-runtime", "ro"),
             (RESULTS/"training-qwen-vision014-CPU-v2/vision", "/vision", "ro"),
             (RESULTS/"training-minicpm-hygon-processor-r2/cpu/runtime-site", "/minicpm-transformers", "ro"),
             (PurePosixPath("/opt/hyhal"), "/opt/hyhal", "ro")]
    if ARMS[arm]["adapter"] is not None:
        binds.append((ARMS[arm]["adapter"], ARMS[arm]["adapter"], "ro"))
    return [[str(a), str(b), c] for a,b,c in binds]


def runtime_bind_source(host):
    host = Path(host)
    if host == Path("/opt/hyhal"):
        target = Path("/usr/local/hyhal")
        require(host.resolve(strict=True) == target and target.is_dir() and not target.is_symlink(), "system hyhal alias differs")
        return str(target)
    require(host.is_dir() and not host.is_symlink(), "original runtime bind missing or linked")
    return str(host)


def budgets(deadline, mono_end, now, mono):
    require(all(math.isfinite(v) for v in (deadline,mono_end,now,mono)), "invalid original epoch")
    remaining = min(deadline-now, mono_end-mono)
    require(remaining >= 90, "not enough original CPU epoch for one60second stage and accounting")
    return dict(Slurm_seconds=60,stage_work_seconds=50,kill_grace_seconds=5,client_seconds=min(65,remaining-20),remaining_seconds=remaining)


def pin(path):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size<=1024**2,"unsafe source metadata")
    b=path.read_bytes()
    return dict(bytes=len(b),SHA256=hashlib.sha256(b).hexdigest())


def read(path):
    return json.loads(path.read_bytes())
