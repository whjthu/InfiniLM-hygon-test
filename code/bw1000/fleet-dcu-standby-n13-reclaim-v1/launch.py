"""Restore preemptible work on the eight cards of f12r2n13."""

import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time


RESULTS = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
NODE, PARENT = "f12r2n13", "1103606"
REQUEST = "fa8bf44a79134d75867954655518380e"
SOURCE = RESULTS / "training80-n03-reclaim-bridge-a5362d962be24077a8a025544464e409"
PROBE = SOURCE / "collect_training80_dispatch_snapshot.py"
PROBE_SHA = "3e1b069f93219a165cc19431a54b47bc6044cf54c557e941f5bde797f3404078"
CONTROLLER_SHA = "be504e7670b40d77e503c60c93d76bf2a75003768d764d54884c5f8272976bba"
WORKER_SHA = "dc8820e17b8d4d2886325c6093ad8884e6a5a1085e0eedd40eb53cfb12235059"
OLD = "1103606.3083"
OLD_LABEL = "training80-dcu-standby-bdcdd347d525-f12r2n13"
PINNED_PCI = ("0000:09:00.0", "0000:36:00.0", "0000:55:00.0", "0000:77:00.0",
              "0000:85:00.0", "0000:b5:00.0", "0000:d5:00.0", "0000:f5:00.0")
PROTECTED = {
    "1103606.1691": ("training80-syncbusy-5af4008d8703-control", "cpu=2,mem=512M,node=1"),
    "1103606.3794": ("training80-standby-compute-admit-b15114920600", "cpu=2,mem=768M,node=1"),
}
CONTROLS = {
    "1103606.4735": "training80-standby-compute-e769c4f87da5",
    "1103606.4744": "training80-standby-compute-13956bd9299d",
    "1103606.4762": "training80-standby-compute-d92f5c416729",
    "1103606.4788": "training80-standby-compute-9279177feb3b",
    "1103606.4795": "training80-reclaim-n03-bridge-a5362d962be2",
    "1103606.4799": "training80-standby-compute-7b73a59400c6",
    "1103606.4803": "training80-standby-compute-225e342494af",
}
OBSERVER = "1103606.4808"
OBSERVER_NAME = "training80-long-busy-0445c58f141f"
COLS = "JobIDRaw,JobName%120,State,ExitCode,ElapsedRaw,NodeList,AllocTRES%120,Start,End"
LABEL = f"training80-dcu-standby-{REQUEST[:12]}-{NODE}"


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def read_source(path, pin):
    require(path.resolve(strict=True) == path and stat.S_ISREG(path.lstat().st_mode),
            "Existing source path differs")
    with path.open("rb") as file:
        raw = file.read(262145)
    require(0 < len(raw) <= 262144 and sha256(raw).hexdigest() == pin, "Existing source differs")
    return raw


def query(*argv):
    env = dict(os.environ, LC_ALL="C", TZ="UTC", SLURM_TIME_FORMAT="standard")
    result = subprocess.run(argv, capture_output=True, check=True, timeout=8, env=env, stdin=subprocess.DEVNULL)
    require(not result.stderr and len(result.stdout) < 2 << 20, "Slurm query output differs")
    return result.stdout.decode().splitlines()


def fields(row):
    return dict(value.split("=", 1) for value in row.split() if "=" in value)


def account(handle):
    return [r.split("|") for r in query("sacct", "-nP", "-j", handle, "-o", COLS)
            if r.split("|", 1)[0] == handle]


def placement():
    [job] = map(fields, query("scontrol", "show", "job", "--oneliner", PARENT))
    require(job["JobState"] == "RUNNING" and job["UserId"].endswith(f"({os.getuid()})")
            and "gres/dcu=320" in job["AllocTRES"].split(","),
            "Parent is no longer the original held allocation")
    expanded = {}

    def on_node(node_list):
        if re.fullmatch(r"f\d+r\d+n\d+", node_list):
            return node_list == NODE
        if node_list not in expanded:
            expanded[node_list] = query("scontrol", "show", "hostnames", node_list)
        require(expanded[node_list], "Unresolved placement")
        return NODE in expanded[node_list]

    require(on_node(job["NodeList"]), "Node absent from parent")
    active = {}
    for line in query("scontrol", "show", "step", PARENT, "--oneliner"):
        row = fields(line)
        handle = row["StepId"]
        if not re.fullmatch(PARENT + r"\.\d+", handle) or not on_node(row["NodeList"]):
            continue
        name, state, tres = row["Name"], row["State"], row["TRES"]
        if handle in PROTECTED:
            require((name, tres) == PROTECTED[handle] and state == "TIMEOUT",
                    "Protected historical CPU control changed")
        elif handle in CONTROLS:
            require((name, state, tres) == (CONTROLS[handle], "RUNNING", "cpu=2,mem=2G,node=1"),
                    "Known CPU controller changed")
        elif handle == OBSERVER:
            require(name.startswith(OBSERVER_NAME) and state == "RUNNING"
                    and tres == "cpu=40,mem=10G,node=40", "Full-fleet CPU observer changed")
        else:
            raise ValueError("New placement requires reconciliation")
        active[handle] = name
    for line in query("squeue", "--steps", "-h", "-j", PARENT, "-o", "%i|%N"):
        handle, nodes = line.split("|")
        if re.fullmatch(PARENT + r"\.\d+", handle):
            require(not on_node(nodes) or handle in active, "Target has an unresolved queued step")
    return active


def probe():
    pre = sha256((REQUEST + ":n13-pre").encode()).hexdigest()[:32]
    label = "training80-admit-" + pre
    argv = ["srun", "--threads=1", "--jobid=" + PARENT, "--job-name=" + label, "--exact", "--immediate=5",
            "-N1", "-n1", "-w", NODE, "-c1", "--mem=256M", "--gres=none", "--time=00:01:00",
            "/usr/local/python3.12/bin/python3", "-B", str(PROBE), "node", pre, PARENT, "[43843,43844]"]
    output = subprocess.run(argv, capture_output=True, check=True, timeout=12, stdin=subprocess.DEVNULL)
    require(not output.stderr and 0 < len(output.stdout) <= 4096, "New device probe transport differs")
    row = json.loads(output.stdout)
    require((row["request_id"], row["parent"], row["node"], row["observer_uid"]) ==
            (pre, PARENT, NODE, os.getuid()) and re.fullmatch(PARENT + r"\.\d+", row["step"])
            and tuple(sorted(row["PCIs"])) == PINNED_PCI and "probe_error" not in row
            and row["kfd_process_count"] == row["own_device_fd_process_count"] == 0
            and row["occupied_requested_ports"] == [], "Physical cards are no longer clear")
    deadline = time.monotonic() + 7
    while time.monotonic() < deadline:
        records = account(row["step"])
        if records and records[0][2] not in {"RUNNING", "COMPLETING", "PENDING"}:
            [terminal] = records
            require(terminal[1:4] == [label, "COMPLETED", "0:0"]
                    and terminal[5:7] == [NODE, "cpu=1,mem=256M,node=1"]
                    and terminal[8] not in ("", "Unknown"), "Physical post accounting differs")
            return dict(step=row["step"], accounting=terminal, stdout_SHA256=sha256(output.stdout).hexdigest())
        time.sleep(.25)
    raise ValueError("Physical probe accounting was unavailable")


def run(publication):
    require(re.fullmatch("[0-9a-f]{40}", publication), "Published one-node scope required")
    read_source(PROBE, PROBE_SHA)
    controller = {}
    exec(compile(read_source(SOURCE / "standby_controller.py", CONTROLLER_SHA),
                 "existing-standby-controller", "exec"), controller)
    worker = read_source(SOURCE / "standby_worker.py", WORKER_SHA)
    work = RESULTS / ("training80-dcu-standby-" + REQUEST)
    with os.fdopen(os.open(RESULTS / ".training80-dcu-standby.lock", os.O_RDWR | os.O_NOFOLLOW), "rb") as lock:
        require(stat.S_ISREG(os.fstat(lock.fileno()).st_mode), "Original dispatch lock differs")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        [previous] = account(OLD)
        require(previous[1:4] == [OLD_LABEL, "COMPLETED", "0:0"]
                and previous[5:7] == [NODE, "cpu=8,gres/dcu=8,mem=16G,node=1"]
                and previous[8] not in ("", "Unknown"), "Original standby terminal differs")
        placement()
        start = time.monotonic()
        witness = probe()
        preserved = placement()
        require(time.monotonic() - start <= 30, "Physical proof expired")
        os.umask(0o077)
        work.mkdir(mode=0o700)
        with (work / "worker.py").open("xb") as f:
            f.write(worker)
            f.flush()
            os.fsync(f.fileno())
        controller["write"](work, "intent.json", dict(request=REQUEST, publication=publication, epoch=time.time(),
                             previous=previous, preserved=preserved, probe=witness, synthetic=True))
        submission = controller["spawn"](work, NODE, {"parent": PARENT, "pci": PINNED_PCI}, 10800, REQUEST)
        require(submission["label"] == LABEL, "Unexpected original standby label")
        result = dict(rc=0, request=REQUEST, submission=submission, previous_terminal=OLD, probe=witness,
                      preserved=list(sorted(preserved)), synthetic=True, running_confirmed=False)
        controller["write"](work, "submission-receipt.json", result)
        return result


if __name__ == "__main__":
    try:
        print(json.dumps(run(sys.argv[1]), sort_keys=True), flush=True)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(json.dumps(dict(rc=1, request=REQUEST, error_type=type(error).__name__,
                              error_SHA256=sha256(str(error).encode()).hexdigest())), flush=True)
        raise SystemExit(1)
