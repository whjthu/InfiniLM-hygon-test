"""Submit one bounded zero-GRES profiler metadata step without stopping standby."""

from base64 import b64decode
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
REQUEST = "f329b5e79624463fa97854b93df386d3"
NAME = "zimage-lt-native-build-" + REQUEST[:12]
CPP_SHA = '2eae6ce6e77b89f94e82e9bec39400671c13a712b6af2fe4be887ff4ecaf180c'
PARENT, NODE = "1103606", "f12r2n13"
SOURCE_SHA = "2b9405e770a1d28d3905fbd89dd8be24485710d7280e5840d77581891fd91374"
SHELL = ("set -euo pipefail; source /opt/dtk/env.sh; "
         "export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
         "PYTHONPATH=/training-runtime/site HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES=; "
         "exec timeout --kill-after=5s 90s python3 -B /smoke/cpu_build.py")


def write(path, data):
    with path.open("xb") as handle:
        handle.write((json.dumps(data, sort_keys=True, allow_nan=False) + "\n").encode())
        handle.flush()
        os.fsync(handle.fileno())


def query(*args):
    return subprocess.check_output(args, text=True, timeout=7).splitlines()


def require(value, reason):
    if not value:
        raise ValueError(reason)


def main(data):
    program = b64decode(data["source_base64"], validate=True)
    cpp = b64decode(data["cpp_base64"], validate=True)
    require(sha256(cpp).hexdigest() == CPP_SHA, "Frozen C++ source bytes changed")
    require(data["request"] == REQUEST and sha256(program).hexdigest() == SOURCE_SHA, "CPU source changed")
    published = data["publication"]
    require(len(published) == 40 and all(ch in "0123456789abcdef" for ch in published),
            "missing source publication")
    [line] = query("scontrol", "show", "job", PARENT, "--oneliner")
    fields = dict(token.split("=", 1) for token in line.split() if "=" in token)
    nodes = query("scontrol", "show", "hostnames", fields["NodeList"])
    require(fields["JobState"] == "RUNNING" and fields["UserId"].endswith(f"({os.getuid()})")
            and len(set(nodes)) == 40 and NODE in nodes
            and "gres/dcu=320" in fields["AllocTRES"].split(","), "CPU probe parent ownership changed")
    rows = query("sacct", "-nP", "-j", PARENT, "--format", "JobIDRaw,JobName%100,State")
    require(not any(len(row.split("|")) >= 2 and row.split("|")[1] == NAME for row in rows),
            "previous CPU probe request must be reconciled")
    runtime = ROOT / "results/training-t06-sdxl-runtime-v1"
    image = ROOT / "containers/dtk26-sandbox-r1"
    require(runtime.is_dir() and image.is_dir(), "existing runtime path missing")
    os.umask(0o077)
    dest = ROOT / "results" / ("training-zimage-lt-native-build-" + REQUEST)
    dest.mkdir()
    with (dest / "cpu_build.py").open("xb") as output:
        output.write(program)
    with (dest / "probe.cpp").open("xb") as output:
        output.write(cpp)
    epoch = time.time()
    write(dest / "intent.json", dict(request=REQUEST, publication=published, parent=PARENT, node=NODE,
         source_SHA256=SOURCE_SHA, cpp_SHA256=CPP_SHA, epoch=epoch, deadline=epoch + 130,
         planned_DCU=0, planned_CPUs=2, memory_MiB=4096, replay_allowed=False))
    argv = ["srun", "--jobid=" + PARENT, "--job-name=" + NAME, "--exact", "--immediate=6",
            "--nodes=1", "--ntasks=1", "--nodelist=" + NODE, "--cpus-per-task=2", "--gres=none",
            "--mem=4G", "--time=00:02:00", "--kill-on-bad-exit=1",
            "/public/software/apps/apptainer/1.3.4/bin/apptainer", "exec", "--no-home", "--pwd", "/",
            "--bind", str(dest) + ":/smoke:rw", "--bind", str(runtime) + ":/training-runtime:ro",
            "--bind", "/usr/local/hyhal:/opt/hyhal:ro", str(image), "/bin/bash", "--noprofile", "--norc", "-c", SHELL]
    with (dest / "stdout.jsonl").open("xb") as stdout, (dest / "stderr.private").open("xb") as stderr:
        child = subprocess.Popen(argv, stdout=stdout, stderr=stderr, stdin=subprocess.DEVNULL, start_new_session=True)
    record = {"request": REQUEST, "phase": "launcher_created", "pid": child.pid, "actual_GPU_submitted": 0}
    write(dest / "launcher.json", record)
    print(json.dumps(record, sort_keys=True), flush=True)


if __name__ == "__main__":
    main(BOUND_PAYLOAD)
