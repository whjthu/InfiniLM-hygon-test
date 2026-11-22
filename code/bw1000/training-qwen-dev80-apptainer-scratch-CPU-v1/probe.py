"""Measure the actual container scratch and reproduce the Swift import at zero GRES."""

from hashlib import sha256
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import time


ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
CONTAINER = ROOT / "containers/dtk26-sandbox-r1"
APPTAINER = "/public/software/apps/apptainer/1.3.4/bin/apptainer"

INSIDE = """
import importlib.metadata, json, os, tempfile
from pathlib import Path

def inspect_filesystem(path):
    info = os.statvfs(path)
    dev = os.stat(path).st_dev
    return dict(available_bytes=info.f_bavail * info.f_frsize, total_bytes=info.f_blocks * info.f_frsize,
                available_inodes=info.f_favail, device=[os.major(dev), os.minor(dev)])

scratch = tempfile.gettempdir()
facts = {name:inspect_filesystem(path) for name,path in
         {'python_tmp':scratch,'tmp':'/tmp','var_tmp':'/var/tmp','dev_shm':'/dev/shm',
          'home':str(Path.home()),'results':str(Path.cwd())}.items()}
print('QWEN_SCRATCH_JSON ' + json.dumps(dict(event='container_filesystems', facts=facts)), flush=True)
with tempfile.TemporaryFile(dir=scratch) as stream:
    stream.write(b'qwen scratch probe')
    stream.flush()
    os.fsync(stream.fileno())
print('QWEN_SCRATCH_JSON ' + json.dumps(dict(event='small_tmp_write_passed')), flush=True)
import torch
print('QWEN_SCRATCH_JSON ' + json.dumps(dict(event='torch_imported',HIP=torch.version.hip,
     torch=importlib.metadata.version('torch'))), flush=True)
from swift.infer_engine import AdapterRequest, InferRequest, RequestConfig, TransformersEngine
from swift.model import get_model_processor
from swift.template import get_template
print('QWEN_SCRATCH_JSON ' + json.dumps(dict(event='swift_imported',facts=inspect_filesystem(scratch))), flush=True)
"""

ENV = (
    "set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh; "
    "export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "
    "OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 ROCR_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES=; "
    "export PYTHONPATH=/vision:/minicpm-transformers:/swift-runtime/site; exec "
)


def trial(root, scratch):
    command = [APPTAINER, "exec", "--rocm", "--contain", "--no-home", "--pwd", str(root)]
    if scratch is not None:
        command.extend(("--workdir", str(scratch)))
    for name, target in (
        ("training-vlm-swift-offline-runtime-v1", "/swift-runtime"),
        ("training-qwen-vision014-CPU-v2/vision", "/vision"),
        ("training-minicpm-hygon-processor-r2/cpu/runtime-site", "/minicpm-transformers"),
    ):
        command.extend(("--bind", f"{ROOT / 'results' / name}:{target}:ro"))
    command.extend(("--bind", f"{root}:{root}:rw", "--bind", "/usr/local/hyhal:/opt/hyhal:ro"))
    command += [str(CONTAINER), "/bin/bash", "--noprofile", "--norc", "-c",
                ENV + shlex.join(["python3", "-B", "-u", "-c", INSIDE])]
    started = time.monotonic()
    try:
        child = subprocess.run(command, capture_output=True, timeout=42)
        rc, stdout, stderr = child.returncode, child.stdout, child.stderr
    except subprocess.TimeoutExpired as error:
        rc, stdout, stderr = None, error.stdout or b"", error.stderr or b""
    records = [json.loads(line.removeprefix(b"QWEN_SCRATCH_JSON ")) for line in stdout.splitlines()
               if line.startswith(b"QWEN_SCRATCH_JSON ")]
    return dict(rc=rc, records=records, llvm_enospc=b"No space left on device" in stdout+stderr,
                llvm_fatal=b"LLVM ERROR:" in stdout+stderr, timed_out=rc is None,
                stdout_SHA256=sha256(stdout).hexdigest(), stderr_SHA256=sha256(stderr).hexdigest(),
                elapsed_seconds=time.monotonic()-started)


def main():
    path = Path(__file__).resolve().parent
    claim = json.loads((path / "CLAIM.json").read_bytes())
    node = socket.gethostname().split(".")[0]
    if node not in claim["nodes"] or os.environ["SLURM_JOB_ID"] != claim["parent"]:
        raise ValueError("Unexpected Slurm placement")
    step = claim["parent"]+"."+os.environ["SLURM_STEP_ID"]
    scope = path / node
    scope.mkdir(mode=0o700)
    scratch = scope / "private-workdir"
    scratch.mkdir(mode=0o700)
    results = {}
    for mode, directory in (("default", None), ("private_workdir", scratch)):
        results[mode] = trial(path, directory)
        with (scope / (mode+".json")).open("x", encoding="utf8") as stream:
            json.dump(dict(node=node, step=step, request=claim["request"], mode=mode, result=results[mode]),
                      stream, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
    print(json.dumps(dict(node=node, step=step, rc={mode: result["rc"] for mode,result in results.items()})))


if __name__ == "__main__":
    main()
