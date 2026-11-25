"""Bind one T09 zero-GRES independent PCI, KFD, and rendezvous port observation."""
import base64
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
PARENT = "1103606"
NODE = "f13r3n13"
LABEL = "training-t09-r42-dp2-physical-cpu-66ab6110d01a"
PHYSICAL = "66ab6110d01a49a691c81646b7520869"
SOURCE_SHA = "cc74e48494f902480a4cf1a9a08e757d128ccd143a68d1ad635460b46d36d3fb"
COLUMNS = "JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES"


def require(value, code):
    if not value:
        raise ValueError(code)


def query(*argv):
    result = subprocess.run(argv, capture_output=True, stdin=subprocess.DEVNULL, timeout=5, check=True)
    require(not result.stderr and len(result.stdout) <= 1 << 20, "T09_HOST_SCHEDULER_QUERY_BOUND")
    return result.stdout.decode("ascii").splitlines()


def rows(job):
    result = [line.split("|") for line in query("sacct", "-nP", "-j", job, "-o", COLUMNS)]
    require(all(len(row) == 7 for row in result), "T09_HOST_ACCOUNTING_COLUMNS")
    return result


def nodes(value):
    if re.fullmatch(r"f[0-9]{2}r[0-9]+n[0-9]{2}", value):
        return [value]
    return query("scontrol", "show", "hostnames", value)


def write(path, doc):
    raw = (json.dumps(doc, sort_keys=True, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return sha256(raw).hexdigest()


def other_records():
    for line in query("squeue", "--steps", "-h", "-j", PARENT, "-o", "%i|%N"):
        handle, hostlist = line.split("|")
        require(re.fullmatch(PARENT + r"\.(?:[0-9]+|batch|extern|interactive)", handle), "T09_HOST_UNKNOWN_STEP")
        if handle.rsplit(".", 1)[1].isdecimal():
            require(NODE not in nodes(hostlist), "T09_HOST_NODE_HAS_AN_ACTIVE_STEP")
    for line in query("scontrol", "show", "step", PARENT, "-o"):
        entry = dict(token.split("=", 1) for token in line.split() if "=" in token)
        handle = entry.get("StepId", "")
        require(re.fullmatch(PARENT + r"\.(?:[0-9]+|batch|extern|interactive)", handle), "T09_HOST_UNKNOWN_CONTROLLER_STEP")
        if handle.rsplit(".", 1)[1].isdecimal() and entry.get("State") not in (
                "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"):
            require(NODE not in nodes(entry["NodeList"]), "T09_HOST_NODE_HAS_AN_ACTIVE_CONTROLLER_STEP")


def main(data):
    require(set(data) == {"request", "publication", "source"}
            and re.fullmatch("[0-9a-f]{32}", data["request"])
            and re.fullmatch("[0-9a-f]{40}", data["publication"]), "T09_HOST_INVALID_ENVELOPE")
    source = base64.b64decode(data["source"], validate=True)
    require(len(source) == 6455 and sha256(source).hexdigest() == SOURCE_SHA, "T09_HOST_FROZEN_SOURCE")
    directory = ROOT / LABEL
    with (ROOT / ".training80-dcu-standby.lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(ROOT.resolve(strict=True) == ROOT and not os.path.lexists(directory), "T09_HOST_EXACT_NEW_NAMESPACE")
        parent = [x.split("|") for x in query("squeue", "-h", "-w", NODE, "-o", "%i|%T|%N")]
        require(len(parent) == 1 and parent[0][:2] == [PARENT, "RUNNING"]
                and NODE in nodes(parent[0][2]), "T09_HOST_CURRENT_OWNED_PARENT")
        prior = [row for row in rows("1103606.1614") if row[0] == "1103606.1614"]
        require(len(prior) == 1 and prior[0][2:4] == ["COMPLETED", "0:0"] and prior[0][5] == NODE
                and "gres/" not in prior[0][6], "T09_HOST_PRIOR_INDEPENDENT_POST")
        require(not any(row[1] == LABEL for row in rows(PARENT)), "T09_HOST_PRIOR_CURRENT_LABEL")
        other_records()
        directory.mkdir(mode=0o700)
        member = directory / "physical.py"
        with member.open("xb") as output:
            output.write(source)
        epoch = time.time()
        identity = {key: data[key] for key in ("request", "publication")} | {
            "label": LABEL, "epoch": epoch, "GPU_requested": 0, "replay_allowed": False}
        argv = ["srun", "--jobid=" + PARENT, "--exact", "--immediate=5", "--nodes=1", "--ntasks=1",
                "--nodelist=" + NODE, "--cpus-per-task=1", "--mem=1024M", "--gres=none", "--time=00:01:30",
                "--job-name=" + LABEL, "env", "HIP_VISIBLE_DEVICES=-1", "ROCR_VISIBLE_DEVICES=-1",
                "CUDA_VISIBLE_DEVICES=-1", "/usr/local/python3.12/bin/python3", "-I", "-S", "-B", str(member),
                "--mode", "pre", "--epoch", str(epoch), "--request", PHYSICAL]
        write(directory / "intent.json", identity | {"source_SHA256": SOURCE_SHA, "argv": argv})
        with (directory / "stdout").open("xb") as out, (directory / "stderr").open("xb") as err:
            child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=out, stderr=err)
        write(directory / "client.json", identity | {"client_PID": child.pid})
    try:
        rc = child.wait(timeout=31)
    except subprocess.TimeoutExpired:
        return identity | {"state": "ORIGINAL_OUTCOME_UNKNOWN"}
    match = []
    for _ in range(4):
        match = [row for row in rows(PARENT) if row[1] == LABEL]
        require(len(match) <= 1, "T09_HOST_DUPLICATE_ACCOUNTING_LABEL")
        if match and match[0][2] not in ("PENDING", "RUNNING", "COMPLETING"):
            break
        time.sleep(.4)
    with (directory / "stdout").open("rb") as out, (directory / "stderr").open("rb") as err:
        stdout, stderr = out.read(8193), err.read(8193)
    require(max(len(stdout), len(stderr)) <= 8192, "T09_HOST_PROBE_OUTPUT_BOUND")
    try:
        probe = json.loads(stdout) if stdout else None
    except ValueError:
        probe = None
    allowed = {"request", "epoch", "mode", "passed", "target_GPU", "optimizer_updates", "all_KFD_plus_same_UID_FD",
               "node", "step", "pci", "boot_id", "monotonic_ns", "port", "port_available", "kfd_holder_count",
               "physical_samples", "error_class", "predicate"}
    require(probe is None or set(probe) <= allowed, "T09_HOST_PROBE_OUTPUT_FIELDS")
    row = match[0] if match else None
    passed = bool(rc == 0 and probe and probe["passed"] is True and row and row[0] == probe["step"]
                  and row[1] == LABEL and row[2:4] == ["COMPLETED", "0:0"] and row[5] == NODE
                  and row[4].isdigit() and int(row[4]) <= 90
                  and set(row[6].split(",")) in ({"cpu=1", "mem=1G", "node=1"}, {"cpu=1", "mem=1024M", "node=1"})
                  and probe["request"] == PHYSICAL and probe["target_GPU"] == probe["optimizer_updates"] == 0
                  and probe["kfd_holder_count"] == 0 and probe["physical_samples"] == 2
                  and probe["port"] == 29642 and probe["port_available"] is True
                  and probe["all_KFD_plus_same_UID_FD"] is True and len(set(probe["pci"])) == 8)
    result = identity | {"state": "ACTUAL_ZERO_GRES_DEVICE_AND_PORT_PROBE_PASSED" if passed else "ORIGINAL_CHECK_UNQUALIFIED",
                         "rc": rc, "probe": probe, "accounting": row, "stdout_SHA256": sha256(stdout).hexdigest(),
                         "stderr_SHA256": sha256(stderr).hexdigest(), "lease": False, "READY_added": 0}
    return result | {"target_result_SHA256": write(directory / "result.json", result)}


if __name__ == "__main__":
    print(json.dumps(main(json.load(sys.stdin)), sort_keys=True))
