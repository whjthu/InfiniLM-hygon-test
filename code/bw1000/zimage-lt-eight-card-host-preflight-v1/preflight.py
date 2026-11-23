"""Check the Z-Image compute node's CPU affinity and hyhal roots without GPUs."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import time


REQUEST = "00ba290bf5844476a5b84ceaa9930f5c"
LABEL = "O01-zimage-hybrid-host-none-00ba290bf584"
PARENT, NODE, STANDBY = "1103606", "f12r2n11", "1103606.2075"
ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results") / LABEL
PYTHON = "/usr/local/python3.12/bin/python3"


def write(name, value):
    with (ROOT / name).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def query(*argv):
    return subprocess.check_output(argv, timeout=8, text=True)


def fields(row):
    return dict(token.split("=", 1) for token in row.split() if "=" in token)


def mount(path):
    path = Path(path)
    try:
        actual = path.resolve(strict=True)
        status = actual.stat()
    except FileNotFoundError:
        return None
    return {"resolved": str(actual), "is_dir": stat.S_ISDIR(status.st_mode),
            "device": status.st_dev, "inode": status.st_ino}


def node():
    step = os.environ["SLURM_JOB_ID"] + "." + os.environ["SLURM_STEP_ID"]
    if (socket.gethostname().split(".")[0] != NODE
            or not re.fullmatch(PARENT + r"\.[0-9]+", step) or step == STANDBY
            or os.environ.get("SLURM_CPUS_PER_TASK") != "12"
            or os.environ.get("SLURM_STEP_GPUS") not in (None, "", "(null)")):
        raise RuntimeError("unexpected CPU-only scope")
    affinity = sorted(os.sched_getaffinity(0))
    left, right = (mount(p) for p in ("/opt/hyhal", "/usr/local/hyhal"))
    same = bool(left and right and left["is_dir"] and right["is_dir"]
                and (left["device"], left["inode"]) == (right["device"], right["inode"]))
    kfd = "readable"
    try:
        with os.scandir("/sys/class/kfd/kfd/proc") as entries:
            next(entries, None)
    except OSError as error:
        kfd = type(error).__name__
    write("node.json", {"request": REQUEST, "step": step, "node": NODE, "affinity": affinity,
                       "affinity_matches_requested": len(affinity) == 12,
                       "hyhal": {"/opt/hyhal": left, "/usr/local/hyhal": right}, "same_hyhal_directory": same,
                       "KFD_listing_access_in_zero_GRES_step": kfd, "observed_epoch": time.time()})


def start(source_sha):
    source = Path(__file__).read_bytes()
    if sha256(source).hexdigest() != source_sha:
        raise RuntimeError("published source does not match payload")
    job = fields(query("scontrol", "show", "job", PARENT, "--oneliner"))
    standby = fields(query("scontrol", "show", "step", STANDBY, "--oneliner"))
    if (job["JobState"] != "RUNNING" or not job["UserId"].endswith(f"({os.getuid()})")
            or NODE not in query("scontrol", "show", "hostnames", job["NodeList"]).splitlines()
            or (standby["StepId"], standby["NodeList"], standby["State"]) != (STANDBY, NODE, "RUNNING")
            or "gres/dcu=8" not in standby["TRES"].split(",")):
        raise RuntimeError("expected active eight-card predecessor")
    ROOT.mkdir(mode=0o700)
    with (ROOT / "preflight.py").open("xb") as stream:
        stream.write(source)
        stream.flush()
        os.fsync(stream.fileno())
    start_epoch = time.time()
    write("intent.json", {"request": REQUEST, "label": LABEL, "source_SHA256": source_sha,
                          "started_epoch": start_epoch, "planned_CPU": 12, "planned_DCU": 0})
    cmd = ["srun", "--jobid=" + PARENT, "--job-name=" + LABEL, "--exact", "--immediate=5",
           "--nodes=1", "--ntasks=1", "--nodelist=" + NODE, "--cpus-per-task=12",
           "--threads-per-core=1", "--cpu-bind=threads", "--gres=none", "--mem=1024M", "--time=00:01:00",
           "--output=" + str(ROOT / "slurm.stdout"), "--error=" + str(ROOT / "slurm.stderr"),
           PYTHON, "-I", "-S", "-B", str(ROOT / "preflight.py"), "node"]
    rc = subprocess.run(cmd, stdin=subprocess.DEVNULL).returncode
    write("return.json", {"request": REQUEST, "srun_rc": rc, "returned_epoch": time.time()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "node"))
    parser.add_argument("--source-sha256")
    args = parser.parse_args()
    if args.action == "node":
        node()
    else:
        start(args.source_sha256)


if __name__ == "__main__":
    main()
