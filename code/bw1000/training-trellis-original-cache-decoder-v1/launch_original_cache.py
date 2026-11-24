"""Run the pinned TRELLIS.2 decoder on the complete original training cache."""

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


BASE = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
RESULTS = BASE / "results"
LABEL = "trellis-original-cache-cpu-decode512-v1"
STAGE = RESULTS / LABEL
SOURCE_STAGE = RESULTS / "trellis-upstream75fb-cpu-decode-source-v1"
WEIGHTS = RESULTS / "trellis-native-generation-microsoft-v1/model"
CACHE = RESULTS / "training-t10-trellis-reference-cache-preflight-v1/train-cache.npz"
INPUT_SHA = "d6c6370b340a21aca16568a1c9ed56e1ca6e26c1ecb458386c8ee9b905ee1a88"
SHAPE_SHA = "e3b718d3e43e4f8780e9a24ac6fff231811a67e3b058e336e10fe654c911d581"
MATERIAL_SHA = "97ea69addea2ecd9312910f5f548234665eef51c088386180b7cd5b258645e3c"
PARENT, NODE = "1103606", "f13r4n05"
COLUMNS = "JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES"
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL"}


def require(test, reason):
    if not test:
        raise RuntimeError(reason)


def query(*args):
    return subprocess.check_output(args, text=True, timeout=8).strip()


def accounting():
    rows = query("sacct", "-n", "-P", "-j", PARENT, "--format=" + COLUMNS)
    return [row for line in rows.splitlines() if len(row := line.split("|")) == 7 and row[1] == LABEL]


def save(name, data):
    with (STAGE / name).open("x", encoding="utf8") as stream:
        json.dump(data, stream, sort_keys=True, indent=2)
        stream.write("\n")


def exact(path, size, digest):
    require(path.stat().st_size == size and sha256(path.read_bytes()).hexdigest() == digest,
            f"frozen artifact differs: {path.name}")


def main():
    start = time.monotonic()
    epoch = time.time()
    end = start + 940
    require(len(sys.argv) == 2 and bool(re.fullmatch(r"[0-9a-f]{40}", sys.argv[1])), "publication is required")
    exact(CACHE, 2184979, INPUT_SHA)
    exact(SOURCE_STAGE / "outcome.json", 3032, "9711385116482b5321a90b54db3177686caae9cdf861bc9dcce7497405eb48b7")
    exact(SOURCE_STAGE / "source-verified.json", 461, "060c5688d9d24735020729d4a1dff5a072f8c727d155016437764e4377bf183e")
    proof_bytes = sys.stdin.buffer.read(32 * 1024 + 1)
    require(sha256(proof_bytes).hexdigest() == "a518b396b411c54aabb59905f0faf0782a79b6455b5495d5c22ac2c7dcec12d1",
            "framework proof differs")
    source_proof = json.loads(proof_bytes)
    require(source_proof["source_commit"] == "75fbf0183001ed9876c8dbb35de6b68552ee08bd", "source revision differs")
    for name, pin in source_proof["framework_source_pins"].items():
        exact(SOURCE_STAGE / "src" / name, pin["bytes"], pin["SHA256"])
    checkout = SOURCE_STAGE / "checkout"
    require(query("git", "-C", str(checkout), "rev-parse", "HEAD") == source_proof["source_commit"] and
            not query("git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=normal"),
            "official decoder checkout differs")
    for name, pin in source_proof["files"].items():
        exact(checkout / name, pin["bytes"], pin["SHA256"])
    current = query("squeue", "-h", "-j", PARENT, "-o", "%i|%T|%D|%N").split("|")
    owner = re.search(r"(?:^|\s)UserId=[^()\s]+\(([0-9]+)\)", query("scontrol", "show", "job", PARENT, "--oneliner"))
    require(len(current) == 4 and current[:3] == [PARENT, "RUNNING", "40"] and owner is not None
            and int(owner[1]) == os.geteuid(), "owned parent differs")
    require(NODE in query("scontrol", "show", "hostnames", current[3]).splitlines() and not accounting(),
            "node membership or prior step differs")
    STAGE.mkdir(mode=0o700)
    save("intent.json", {"epoch": epoch, "deadline_epoch": epoch + 940, "publication": sys.argv[1],
                         "input_SHA256": INPUT_SHA, "input_points": 1024, "threads": 4, "memory_MiB": 65536,
                         "GPU": 0, "geometry_resolution": 512, "point_cap": 1000000})
    inside = min(840, int(end - time.monotonic()) - 50)
    require(inside >= 600, "insufficient decode budget")
    script = (
        "set -euo pipefail; source /opt/dtk/env.sh; "
        "export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONNOUSERSITE=1 PYTHONPATH=/candidate/src "
        "CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 "
        "OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_SYSTEM=/dev/null; "
        "exec python3 -B -u -m gencuda.trellis_cpu_decode --source-root /checkout --weights-root /weights "
        "--input /original-cache.npz --input-sha256 " + INPUT_SHA +
        " --output /output/decoded512.npz --resolution 512 --max-points 1000000 --threads 4 "
        "--shape-weights-sha256 " + SHAPE_SHA + " --material-weights-sha256 " + MATERIAL_SHA
    )
    command = [
        "srun", "--jobid=" + PARENT, "--job-name=" + LABEL, "--nodes=1", "--ntasks=1",
        "--nodelist=" + NODE, "--cpus-per-task=4", "--mem=65536M", "--gres=none", "--exact",
        "--overlap", "--immediate=5", "--time=00:15:00", "/public/software/apps/apptainer/1.3.4/bin/apptainer",
        "exec", "--no-home", "--pwd", "/", "--bind", str(SOURCE_STAGE) + ":/candidate:ro",
        "--bind", str(checkout) + ":/checkout:ro", "--bind", str(WEIGHTS) + ":/weights:ro",
        "--bind", str(CACHE) + ":/original-cache.npz:ro", "--bind", str(STAGE) + ":/output:rw",
        "--bind", "/opt/hyhal:/opt/hyhal:ro", str(BASE / "containers/dtk26-sandbox-r1"),
        "/usr/bin/timeout", "-k", "5", str(inside), "/bin/bash", "--noprofile", "--norc", "-c", script,
    ]
    forced = False
    with (STAGE / "CPU.log").open("xb") as log:
        worker = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        proc = Path("/proc", str(worker.pid), "stat").read_text().split(") ", 1)[1].split()
        save("srun.json", {"PID": worker.pid, "startticks": int(proc[19]), "node": NODE, "parent": PARENT})
        try:
            worker.wait(timeout=min(inside + 12, end - time.monotonic() - 28))
        except subprocess.TimeoutExpired:
            forced = True
            os.killpg(worker.pid, signal.SIGTERM)
            try:
                worker.wait(timeout=4)
            except subprocess.TimeoutExpired:
                os.killpg(worker.pid, signal.SIGKILL)
                worker.wait(timeout=4)
    observations = []
    while time.monotonic() + 9 < end and len(observations) < 5:
        try:
            rows = accounting()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            rows = []
        observations.append(rows)
        if len(rows) == 1 and rows[0][2].split()[0] in TERMINAL:
            break
        time.sleep(1)
    prefix = b"TRELLIS2_CPU_DECODE_RESULT="
    log = (STAGE / "CPU.log").read_bytes()
    bodies = [line[len(prefix):] for line in log.splitlines() if line.startswith(prefix)]
    receipt = json.loads(bodies[0]) if len(bodies) == 1 else None
    actual = observations[-1] if observations else []
    output = STAGE / "decoded512.npz"
    passed = bool(worker.returncode == 0 and not forced and len(actual) == 1 and actual[0][2:4] == ["COMPLETED", "0:0"]
                  and actual[0][5] == NODE and "gres/" not in actual[0][6] and receipt and output.is_file())
    result = {"publication": sys.argv[1], "passed": passed, "cpu_rc": worker.returncode, "forced": forced,
              "accounting_columns": COLUMNS, "observations": observations, "receipt": receipt,
              "log_SHA256": sha256(log).hexdigest(), "failure_tail": None if passed else log[-2500:].decode(errors="replace"),
              "end_UTC": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.monotonic() - start}
    save("outcome.json", result)
    print(json.dumps(result))
    return int(not passed)


if __name__ == "__main__":
    raise SystemExit(main())
