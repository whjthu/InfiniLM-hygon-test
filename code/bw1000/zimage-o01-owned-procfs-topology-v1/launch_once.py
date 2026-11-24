"""Submit the zero-card procfs diagnostic alongside one verified BF16 standby."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
REQUEST = "2fe5248f9841494ea268191cec2a840c"
LABEL = "O01-owned-procfs-topology-2fe5248f9841"
SOURCE_SHA = "1ee72c6bad9a4098852e256f9c4ccad63f1196ea1565ac1c378dad74f61d25d3"
PARENT, NODE = "1103606", "f12r2n11"
COLUMNS = "JobIDRaw,JobName%120,State,ExitCode,ElapsedRaw,NodeList,AllocTRES,Start,End"


def need(ok, detail):
    if not ok:
        raise ValueError(detail)


def read(argv, seconds=8):
    result = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=seconds, check=True)
    need(len(result.stdout) <= 2 * 1024**2, "Slurm output bound exceeded")
    return result.stdout.decode("utf-8").splitlines()


def fields(line):
    return dict(item.split("=", 1) for item in line.split() if "=" in item)


def node_owned_step(standby, query=read):
    jobs = query(["scontrol", "show", "job", PARENT, "--oneliner"])
    need(len(jobs) == 1, "parent job is ambiguous")
    parent = fields(jobs[0])
    need(parent["JobId"] == PARENT and parent["JobState"] == "RUNNING"
         and parent["UserId"].endswith(f"({os.getuid()})") and parent["NumNodes"] == "40"
         and "gres/dcu=320" in parent["AllocTRES"].split(","), "parent job ownership differs")
    need(NODE in query(["scontrol", "show", "hostnames", parent["NodeList"]]), "node outside owned parent")
    active = []
    for line in query(["scontrol", "show", "step", PARENT, "--oneliner"]):
        row = fields(line)
        if not re.fullmatch(PARENT + r"\.[0-9]+", row.get("StepId", "")):
            continue
        nodes = ([row["NodeList"]] if row.get("Nodes") == "1"
                 else query(["scontrol", "show", "hostnames", row["NodeList"]]))
        if NODE in nodes:
            active.append(row)
    need(len(active) == 1 and active[0]["StepId"] == standby and active[0]["State"] == "RUNNING"
         and active[0]["Name"].startswith("training80-dcu-standby-")
         and set(active[0]["TRES"].split(",")) == {"cpu=8", "gres/dcu=8", "mem=16G", "node=1"},
         "expected exactly the owned eight-card standby")
    return active[0]["Name"]


def accounting(query=read):
    raw = query(["sacct", "-n", "-P", "-j", PARENT, "-S", "2026-11-24T00:00:00", "-o", COLUMNS])
    rows = [line.split("|") for line in raw if line.strip()]
    need(all(len(row) == 9 for row in rows), "Slurm accounting columns differ")
    return rows


def srun():
    return ["srun", "--jobid=" + PARENT, "--job-name=" + LABEL, "--exact", "--overlap", "--immediate=5",
            "--nodes=1", "--ntasks=1", "--nodelist=" + NODE, "--cpus-per-task=1", "--threads-per-core=1",
            "--cpu-bind=threads", "--gres=none", "--mem=256M", "--time=00:01:00", "--kill-on-bad-exit=1",
            "--output=" + str(ROOT / "probe.stdout"), "--error=" + str(ROOT / "probe.stderr"),
            sys.executable, "-I", "-S", "-B", str(ROOT / "probe.py"), "host"]


def save(name, value):
    with (ROOT / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication", required=True)
    parser.add_argument("--standby", required=True)
    args = parser.parse_args()
    need(re.fullmatch(r"[0-9a-f]{40}", args.publication) and re.fullmatch(PARENT + r"\.[0-9]+", args.standby),
         "invalid publication or original standby scope")
    raw = (ROOT / "probe.py").read_bytes()
    need(len(raw) == 7379 and sha256(raw).hexdigest() == SOURCE_SHA, "frozen observed source differs")
    rows = accounting()
    need(not any(row[1] == LABEL for row in rows), "existing accounting label requires reconcile")
    active = node_owned_step(args.standby)
    prior = [row for row in rows if row[0] == args.standby]
    need(len(prior) == 1 and prior[0][1:4] == [active, "RUNNING", "0:0"], "original standby accounting differs")
    intent = dict(request=REQUEST, publication=args.publication, original_standby=args.standby,
                  source_SHA256=SOURCE_SHA, planned_DCU=0, one_attempt=True, epoch=time.time())
    save("probe-intent.json", intent)
    rc = None
    try:
        result = subprocess.run(srun(), stdin=subprocess.DEVNULL, capture_output=True, timeout=65)
        rc = result.returncode
        metadata = {"srun_stdout_SHA256": sha256(result.stdout).hexdigest(), "srun_stderr_SHA256": sha256(result.stderr).hexdigest()}
    except subprocess.TimeoutExpired:
        metadata = {"srun_timeout": True}
    receipt = dict(intent, **metadata, srun_rc=rc)
    save("launch-receipt.json", receipt)
    print(json.dumps(receipt, sort_keys=True))
    return int(rc != 0)


if __name__ == "__main__":
    raise SystemExit(main())
