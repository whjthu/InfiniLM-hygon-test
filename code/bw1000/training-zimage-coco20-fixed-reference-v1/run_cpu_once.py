"""Stage the pinned diagnostic and hand the existing owned Slurm allocation a CPU test."""

import base64
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time


ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
LABEL = "training-zimage-coco20-fixed-reference-CPU-v1"
REQUEST = "9825ada38231459ba6d97952c141d258"
PACKAGE = {"zimage_fixed_reference.py", "test_zimage_fixed_reference.py", "loss.py"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    line = sys.stdin.buffer.readline(65000)
    require(line.endswith(b"\n") and 0 < len(line) < 65000, "Invalid bounded Z-Image source package")
    payload = json.loads(base64.b64decode(line, validate=False))
    publication = payload["publication"]
    require(re.fullmatch(r"[a-f0-9]{40}", publication) is not None, "Source publication is missing")
    claim_raw = base64.b64decode(payload["claim"], validate=True)
    require(hashlib.sha256(claim_raw).hexdigest() == payload["claim_sha256"], "Source claim differs")
    claim = json.loads(claim_raw)
    require((claim["request"], claim["label"], claim["parent"], claim["node"], claim["GPU_cards"]) ==
            (REQUEST, LABEL, "1103606", "f13r2n07", 0), "Owned diagnostic scope differs")
    files = {name: base64.b64decode(b64, validate=True) for name, b64 in payload["files"].items()}
    require(set(files) == PACKAGE == set(claim["files"]), "Unexpected staged source")
    for name, data in files.items():
        require(claim["files"][name]["bytes"] == len(data)
                and claim["files"][name]["SHA256"] == hashlib.sha256(data).hexdigest(), "Source bytes differ")
    runtime = ROOT / "results/training-t06-sdxl-runtime-v1"
    image = ROOT / "containers/dtk26-sandbox-r1"
    require(runtime.is_dir() and image.is_dir(), "Verified diagnostic runtime is not present")

    os.umask(0o077)
    stage = ROOT / "results" / LABEL
    stage.mkdir()
    package = stage / "gencuda"
    package.mkdir()
    for name, raw in files.items():
        path = (package if name == "zimage_fixed_reference.py" else stage) / name
        with path.open("xb") as out:
            out.write(raw)
    epoch = time.time()
    with (stage / "intent.json").open("x") as out:
        json.dump(dict(request=REQUEST, publication=publication, parent="1103606", node="f13r2n07",
                       epoch=epoch, claim_SHA256=payload["claim_sha256"], GPU_cards=0,
                       local_srun_PID=os.getpid()), out, sort_keys=True)
    sys.stderr.flush()
    sys.stdout.flush()

    test = ("set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh; "
            "export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=/replay:/training-runtime/site "
            "CUDA_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 "
            "OPENBLAS_NUM_THREADS=1; "
            "exec /usr/bin/timeout --kill-after=3s 35s python3 -B -u "
            "/replay/test_zimage_fixed_reference.py /replay/loss.py")
    argv = [
        "srun", "--jobid=1103606", "--job-name=" + LABEL, "--exact", "--nodes=1", "--ntasks=1",
        "--nodelist=f13r2n07", "--cpus-per-task=2", "--mem=4096M", "--gres=none", "--immediate=5",
        "--threads=1", "--time=00:01:00", "--kill-on-bad-exit=1", "--output=" + str(stage / "stdout.private"),
        "--error=" + str(stage / "stderr.private"),
        "/public/software/apps/apptainer/1.3.4/bin/apptainer", "exec", "--no-home", "--pwd", "/",
        "--bind", f"{stage}:/replay:ro", "--bind", f"{runtime}:/training-runtime:ro",
        "--bind", "/usr/local/hyhal:/opt/hyhal:ro", str(image),
        "/bin/bash", "--noprofile", "--norc", "-c", test,
    ]
    os.execvp("srun", argv)


if __name__ == "__main__":
    main()
