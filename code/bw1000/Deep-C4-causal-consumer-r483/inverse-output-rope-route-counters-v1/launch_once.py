"""Run the claimed TU15 CPU-only step once under the existing Slurm allocation."""

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


PROJECT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
LABEL = "deep-rope-route-tu15-cpu-v1"
REQUEST = "deep-rope-route-tu15-build-20261123-v1"
PARENT, NODE = "1103606", "f12r2n13"
SCOPE = PROJECT / "results" / LABEL
STAGE, OUTPUT = SCOPE / "stage", SCOPE / "artifacts"
HOST_PYTHON = "/usr/local/python3.12/bin/python3"


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def query(*args):
    return subprocess.run(args, capture_output=True, text=True, check=True, timeout=12).stdout


def field(line, key):
    match = re.search(r"(?:^|\s)" + re.escape(key) + r"=(\S+)", line)
    if not match:
        raise ValueError(f"Missing Slurm field {key}")
    return match[1]


def hosts(expression):
    return set(query("scontrol", "show", "hostnames", expression).splitlines())


def admit():
    job = query("scontrol", "show", "job", PARENT, "-o")
    nodes = hosts(field(job, "NodeList"))
    if (field(job, "JobId") != PARENT or field(job, "JobState") != "RUNNING"
            or not field(job, "UserId").endswith(f"({os.getuid()})")
            or "gres/dcu=320" not in field(job, "AllocTRES").split(",")
            or len(nodes) != 40 or NODE not in nodes):
        raise ValueError("Parent ownership or expected allocation changed")
    steps = query("scontrol", "show", "step", PARENT, "--oneliner").splitlines()
    if not steps:
        raise ValueError("No active step inventory")
    for line in steps:
        step = field(line, "StepId")
        if step in (PARENT + ".batch", PARENT + ".extern"):
            continue
        if not re.fullmatch(re.escape(PARENT) + r"\.\d+", step) or NODE in hosts(field(line, "NodeList")):
            raise ValueError("Ambiguous or busy claimed node")
    past = query("sacct", "-nP", "-j", PARENT, "--format", "JobIDRaw,JobName%100")
    if any(row.endswith("|" + LABEL) for row in past.splitlines()):
        raise ValueError("Exact label already has accounting")


def read_claim():
    if Path(__file__).resolve() != STAGE / "launch_once.py":
        raise ValueError("Unexpected original build directory")
    raw = (STAGE / "CLAIM.json").read_bytes()
    claim = json.loads(raw)
    if (claim["label"], claim["request"], claim["parent"], claim["node"], claim["cpus"],
            claim["memory_mib"], claim["slurm_seconds"], claim["GPU_steps"]) != (
            LABEL, REQUEST, PARENT, NODE, 4, 8192, 300, 0):
        raise ValueError("Claimed scope differs")
    for name, digest in claim["materials"].items():
        if Path(name).name != name or sha256((STAGE / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Staged build material differs")
    return sha256(raw).hexdigest()


def launch():
    digest = read_claim()
    admit()
    epoch = time.time()
    intent = {"request": REQUEST, "label": LABEL, "parent": PARENT, "node": NODE,
              "claim_SHA256": digest, "epoch": epoch, "deadline": epoch + 300}
    save(SCOPE / "intent.json", intent)
    with (SCOPE / "worker.log").open("xb") as log:
        child = subprocess.Popen(
            [HOST_PYTHON, "-I", "-S", "-B", __file__, "--worker"],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
    print(json.dumps({**intent, "worker_pid": child.pid}, sort_keys=True))


def worker():
    intent = json.loads((SCOPE / "intent.json").read_bytes())
    deadline = intent["deadline"]
    if (intent["claim_SHA256"] != read_claim() or intent["request"] != REQUEST
            or deadline - intent["epoch"] != 300 or not 0 < deadline - time.time() <= 300):
        raise ValueError("Original launch intent or deadline differs")
    argv = [
        "srun", f"--jobid={PARENT}", "--exact", "--immediate=5", "--nodes=1", "--ntasks=1",
        f"--nodelist={NODE}", "--cpus-per-task=4", "--mem=8192M", "--gres=none", "--time=5",
        f"--job-name={LABEL}", "/public/software/apps/apptainer/1.3.4/bin/apptainer",
        "exec", "--contain", "--no-home", "--pwd", "/", "--env", f"TUNING_CPU_DEADLINE={deadline}",
        "--bind", f"{PROJECT}:/refactor:ro", "--bind", f"{STAGE}:/audit:ro",
        "--bind", f"{OUTPUT}:/audit-out:rw", "--bind", "/opt/hyhal:/opt/hyhal:ro",
        str(PROJECT / "containers/dtk26-sandbox-r1"), "/bin/bash", "--noprofile", "--norc", "/audit/inside.sh",
    ]
    result = {"request": REQUEST, "client_rc": None, "timed_out": False}
    try:
        result["client_rc"] = subprocess.run(argv, timeout=max(1, deadline - time.time() - 2)).returncode
    except subprocess.TimeoutExpired:
        result["timed_out"] = True
    finally:
        save(SCOPE / "client-result.json", result)


if __name__ == "__main__":
    if sys.argv[1:] == ["--worker"]:
        worker()
    elif not sys.argv[1:]:
        launch()
    else:
        raise SystemExit("Unexpected launch arguments")
