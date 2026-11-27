"""Start one published zero-GRES HunyuanVideo cache build on the owned compute node."""

from base64 import b64decode
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


PARENT, NODE = "1103606", "f13r3n14"
LABEL = "training-hunyuan-abo-native-text-cache-v1"
PROJECT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
RESULTS = PROJECT / "results"
NATIVE = RESULTS / "training-hunyuanvideo-original-source-v1/source"
MEDIA = RESULTS / "training-hunyuanvideo-abo-vae-inspect-v1/a6b6685a35a52543d3f6eb19ff141d31/media"
VAE = RESULTS / "training-hunyuanvideo-abo-vae-native-encode-v1/f633434ac0b2ad40f98b4fd4e4ec7a59/worker"
QWEN = Path("/public/home/aclsylqidf/shared/Qwen2.5-VL-7B-Instruct")
RUNTIME = RESULTS / "training-t06-sdxl-runtime-v1"
IMAGE = PROJECT / "containers/dtk26-sandbox-r1"
APPTAINER = "/public/software/apps/apptainer/1.3.4/bin/apptainer"
FILES = {"build_cache.py", "native_text.py"}


def require(test, reason):
    if not test:
        raise ValueError(reason)


def hash_of(data):
    return sha256(data).hexdigest()


def write(path, data):
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def encoded(data):
    return (json.dumps(data, sort_keys=True, allow_nan=False) + "\n").encode()


def small_json(path, maximum=16_384):
    if not path.is_file():
        return None
    with path.open("rb") as stream:
        raw = stream.read(maximum + 1)
    require(0 < len(raw) <= maximum, "UNEXPECTED_RECEIPT_SIZE")
    return dict(SHA256=hash_of(raw), value=json.loads(raw))


def query(start, jobname):
    child = subprocess.run(
        ["env", "TZ=UTC", "LC_ALL=C", "SLURM_TIME_FORMAT=%Y-%m-%dT%H:%M:%S", "sacct",
         "-S", start, "-j", PARENT, "-nP", "-o", "JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES,Start,End"],
        stdin=subprocess.DEVNULL, capture_output=True, timeout=10,
    )
    if child.returncode != 0 or len(child.stdout) > 1 << 20:
        return dict(rc=child.returncode, rows=[], stderr_SHA256=hash_of(child.stderr))
    lines = [line.split("|") for line in child.stdout.decode("ascii").splitlines()]
    return dict(rc=0, rows=[row for row in lines if len(row) == 9 and row[1] == jobname])


def run(request):
    require(os.getuid() == 154459 and re.fullmatch("[a-f0-9]{32}", request["request"])
            and re.fullmatch("[a-f0-9]{40}", request["publication"])
            and re.fullmatch("[a-f0-9]{64}", request["claim_SHA256"]), "EXACT_PROJECT_OWNER_AND_PUBLICATION")
    files, pins = request["files"], request["pins"]
    require(set(files) == set(pins) == FILES, "EXACT_WORKER_FILE_SET")
    sources = {name: b64decode(data, validate=True) for name, data in files.items()}
    for name, body in sources.items():
        require(0 < len(body) <= 24_000 and hash_of(body) == pins[name], "EXACT_PUBLISHED_WORKER")
        compile(body, name, "exec")
    require(all(path.is_dir() for path in (NATIVE, MEDIA, VAE, QWEN, RUNTIME, IMAGE)) and os.access(APPTAINER, os.X_OK),
            "EXACT_INSTALLED_READ_ONLY_ASSETS")
    start = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    jobname = LABEL + "-" + request["request"][:12]
    original = query(start, jobname)
    require(original["rc"] == 0 and not original["rows"], "NO_ORIGINAL_SLURM_STEP")
    namespace = RESULTS / LABEL
    root = namespace / request["request"]
    os.umask(0o077)
    namespace.mkdir(mode=0o700, exist_ok=True)
    require(not namespace.is_symlink(), "EXACT_PRIVATE_NAMESPACE")
    root.mkdir(mode=0o700)
    for name in ("home", "cache", "tmp", "src"):
        (root / name).mkdir(mode=0o700)
    for name, body in sources.items():
        write(root / "src" / name, body)
    epoch = time.time()
    intent = dict(request=request["request"], publication=request["publication"], claim_SHA256=request["claim_SHA256"],
                  pins=pins, epoch=epoch, deadline=epoch+900, jobname=jobname, parent=PARENT, node=NODE,
                  CPUs=12, memory_MiB=114688, DCU=0, replay_allowed=False)
    write(root / "intent.json", encoded(intent))
    command = [APPTAINER, "exec", "--no-home", "--no-mount", "cwd", "--pwd", "/"]
    for source, target, mode in ((root, root, "rw"), (root / "src", root / "src", "ro"), (NATIVE, NATIVE, "ro"),
                                 (MEDIA, MEDIA, "ro"), (VAE, VAE, "ro"), (QWEN, QWEN, "ro"),
                                 (RUNTIME, Path("/training-runtime"), "ro")):
        command += ["--bind", f"{source}:{target}:{mode}"]
    env = """set -e
ulimit -c 0
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=1 PYTHONPATH=/training-runtime/site
export HOME="$1/home" XDG_CACHE_HOME="$1/cache" TMPDIR="$1/tmp"
shift
exec python3 -B -u "$@"
"""
    command += [str(IMAGE), "/bin/bash", "--noprofile", "--norc", "-c", env, LABEL, str(root),
                str(root / "src/build_cache.py"), "--source", str(NATIVE), "--qwen", str(QWEN),
                "--render", str(MEDIA / "render.json"), "--latent", str(VAE / "latent.private.f32le"),
                "--request", request["request"], "--out", str(root / "cache-output")]
    srun = ["srun", "--jobid="+PARENT, "--job-name="+jobname, "--exact", "--immediate=6", "--nodes=1",
            "--ntasks=1", "--nodelist="+NODE, "--cpus-per-task=12", "--mem=112G", "--gres=none",
            "--time=00:14:00", "--kill-on-bad-exit=1"]
    with (root / "stdout.private").open("xb") as out, (root / "stderr.private").open("xb") as err:
        try:
            launcher_rc = subprocess.run([*srun, *command], stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                         timeout=max(1, epoch + 883 - time.time())).returncode
        except subprocess.TimeoutExpired:
            launcher_rc = None
    accounting = query(start, jobname)
    worker = {name: small_json(root / "cache-output" / (name+".json")) for name in ("start", "loaded", "terminal")}
    with (root / "stderr.private").open("rb") as stream:
        stream.seek(max(0, (root / "stderr.private").stat().st_size - 4096))
        error_tail = stream.read().decode(errors="replace")
    error = next((line.strip()[-220:] for line in reversed(error_tail.splitlines())
                  if line.startswith(("RuntimeError:", "ValueError:", "ImportError:", "ModuleNotFoundError:",
                                      "FileNotFoundError:", "srun: error:", "TypeError:", "AttributeError:"))), None)
    receipt = dict(request=request["request"], launcher_rc=launcher_rc, intent_SHA256=hash_of(encoded(intent)),
                   accounting=accounting, worker=worker, bounded_error=error)
    write(root / "controller-receipt.json", encoded(receipt))
    return receipt


if __name__ == "__main__":
    data = sys.stdin.buffer.read(100_001)
    require(0 < len(data) <= 100_000, "BOUNDED_SCOPE_PAYLOAD")
    print(json.dumps(run(json.loads(data)), sort_keys=True))
