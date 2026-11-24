"""Run the frozen adapter tests in one zero-GRES step of the owned allocation."""

import base64
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
LABEL = "trellis-shape-adapter-cpu-v1"
PARENT = "1103606"
NODE = "f13r4n05"
STAGE = ROOT / "results" / LABEL
PROOF_SHA = "c9aa2ffd43b0e1d5658fe18148a22f1104518a0f12cede996ffb5908b15d132c"
COLUMNS = "JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES"
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}


def save(name, value):
    with (STAGE / name).open("x", encoding="utf8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def query(*args):
    return subprocess.check_output(args, text=True, timeout=5).strip()


def accounting():
    output = query("sacct", "-n", "-P", "-j", PARENT, "--format=" + COLUMNS)
    return [row for line in output.splitlines() if len(row := line.split("|")) == 7 and row[1] == LABEL]


def main():
    if len(sys.argv) != 2 or len(sys.argv[1]) != 40 or set(sys.argv[1]) - set("0123456789abcdef"):
        raise ValueError("publication commit identity required")
    epoch = time.time()
    end = time.monotonic() + 120
    data = json.loads(sys.stdin.buffer.read(512 * 1024 + 1))
    proof = base64.b64decode(data["proof"], validate=True)
    if sha256(proof).hexdigest() != PROOF_SHA:
        raise ValueError("CPU source proof differs")
    pins = json.loads(proof)["cpu_payload_sources"]
    if set(data["sources"]) != set(pins):
        raise ValueError("CPU source set differs")
    sources = {name: base64.b64decode(encoded, validate=True) for name, encoded in data["sources"].items()}
    for name, body in sources.items():
        if (len(body), sha256(body).hexdigest()) != (pins[name]["bytes"], pins[name]["SHA256"]):
            raise ValueError("CPU source identity differs")
    current = query("squeue", "-h", "-j", PARENT, "-o", "%i|%T|%D|%N").split("|")
    job_info = query("scontrol", "show", "job", PARENT, "--oneliner")
    owner = re.search(r"(?:^|\s)UserId=[^()\s]+\(([0-9]+)\)", job_info)
    if current[:3] != [PARENT, "RUNNING", "40"] or owner is None or int(owner[1]) != os.geteuid():
        raise RuntimeError("owned allocation differs")
    if NODE not in query("scontrol", "show", "hostnames", current[3]).splitlines():
        raise RuntimeError("node is outside the owned allocation")
    if accounting():
        raise RuntimeError("original test already has an accounting record")
    STAGE.mkdir(mode=0o700)
    save("intent.json", {"publication": sys.argv[1], "epoch": epoch, "deadline": epoch + 120,
                         "parent": PARENT, "node": NODE, "source_proof_sha256": PROOF_SHA, "GPU": 0})
    for name, body in sources.items():
        path = STAGE / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(body)
    script = (
        "set -euo pipefail; source /opt/dtk/env.sh; "
        "export PYTHONNOUSERSITE=1 PYTHONPATH=/candidate/src CUDA_VISIBLE_DEVICES=-1 "
        "HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; "
        "python3 -B -c 'import torch, safetensors; print(torch.__version__, safetensors.__version__)'; "
        "exec python3 -B -m unittest discover -v -s /candidate/tests -p test_trellis_shape_adapter.py"
    )
    command = [
        "srun", "--jobid=" + PARENT, "--job-name=" + LABEL, "--nodes=1", "--ntasks=1",
        "--nodelist=" + NODE, "--cpus-per-task=2", "--mem=2048M", "--gres=none", "--exact",
        "--overlap", "--immediate=5", "--time=00:02:00",
        "/public/software/apps/apptainer/1.3.4/bin/apptainer", "exec", "--no-home", "--pwd", "/",
        "--bind", str(STAGE) + ":/candidate:ro", "--bind", "/opt/hyhal:/opt/hyhal:ro",
        str(ROOT / "containers/dtk26-sandbox-r1"), "/usr/bin/timeout", "-k", "5", "60",
        "/bin/bash", "--noprofile", "--norc", "-c", script,
    ]
    forced = False
    with (STAGE / "CPU.log").open("xb") as log:
        worker = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        save("host-srun.json", {"PID": worker.pid})
        try:
            worker.wait(timeout=max(0, min(72, end - time.monotonic() - 24)))
        except subprocess.TimeoutExpired:
            forced = True
            if worker.poll() is None:
                os.killpg(worker.pid, signal.SIGTERM)
            try:
                worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(worker.pid, signal.SIGKILL)
                worker.wait(timeout=5)
    observations = []
    for _ in range(4):
        if time.monotonic() + 6 >= end:
            break
        try:
            rows = accounting()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            rows = None
        observations.append(rows)
        if rows is not None and len(rows) == 1 and rows[0][2].split()[0] in TERMINAL:
            break
        time.sleep(1)
    rows = observations[-1] if observations else None
    body = (STAGE / "CPU.log").read_bytes()
    tail = body[-3500:].decode(errors="replace")
    passed = bool(not forced and worker.returncode == 0 and rows and len(rows) == 1
                  and rows[0][2:4] == ["COMPLETED", "0:0"] and rows[0][5] == NODE
                  and "gres/" not in rows[0][6] and "Ran 4 tests " in tail and "\nOK\n" in tail)
    result = {"publication": sys.argv[1], "source_proof_sha256": PROOF_SHA, "epoch": epoch,
              "cpu_rc": worker.returncode, "forced": forced, "columns": COLUMNS, "observations": observations,
              "log_SHA256": sha256(body).hexdigest(), "test_log_tail": tail, "passed": passed}
    save("outcome.json", result)
    print(json.dumps(result))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
