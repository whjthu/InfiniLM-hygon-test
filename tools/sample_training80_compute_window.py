"""Read a common DCU busy window using two zero-GRES Slurm steps."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

import sample_training80_busy as sampler


PARENTS = {"1103606", "1194900"}
CONTROL = "f12r2n13"
PYTHON = "/usr/local/python3.12/bin/python3"
LEDGER_SHA256 = "11d4cc4798f6533dc4bcf31d82dc2c206035aa498ab3842878a7cb56d688f71d"
PCI_SHA256 = "df2b483f320d54383a2ccaf6e80eab3a69e6efbb3f0caf6d7292c56eb96f9b39"
sampler.POINTS = 5
require = sampler.require


def read_json(path, cap, pin=None):
    with path.open("rb") as stream:
        raw = stream.read(cap + 1)
    require(0 < len(raw) <= cap and (pin is None or sha256(raw).hexdigest() == pin), "Unexpected input bytes")
    return json.loads(raw)


def write_json(path, record):
    with path.open("x", encoding="utf8") as stream:
        json.dump(record, stream, separators=(",", ":"), sort_keys=True, allow_nan=False)


def scope(root):
    ledger = read_json(root / "ledger.json", 65536, LEDGER_SHA256)
    pins = read_json(root / "pci.json", 65536, PCI_SHA256)
    owners = sampler.node_map(ledger, PARENTS)
    sampler.require_pins(owners, pins)
    return owners, pins


def node(root, scheduled):
    name = socket.gethostname().split(".")[0]
    parent = os.environ["SLURM_JOB_ID"]
    step = os.environ["SLURM_STEP_ID"]
    owners, pins = scope(root)
    require(name in owners and owners[name][0] == parent and re.fullmatch(r"[0-9]+", step),
            "Incorrect hardware or Slurm step")
    points = sampler.collect(Path("/sys/class/drm"), pins[name], scheduled)
    write_json(root / "reports" / (name + ".json"), dict(node=name, step=parent + "." + step, rows=points))


def ownership(parent, members):
    raw = subprocess.check_output(["scontrol", "show", "job", "-o", parent], text=True, timeout=6)
    fields = dict(token.split("=", 1) for token in raw.split() if "=" in token)
    current = subprocess.check_output(["scontrol", "show", "hostnames", fields["NodeList"]], text=True, timeout=6)
    require(fields["JobId"] == parent and fields["JobState"] == "RUNNING"
            and fields["UserId"].endswith(f"({os.getuid()})") and set(current.splitlines()) == set(members)
            and "gres/dcu=320" in fields["AllocTRES"].split(","), "Current owned allocation differs")


def capture(root, request):
    require(re.fullmatch(r"[0-9a-f]{32}", request) and socket.gethostname().split(".")[0] == CONTROL
            and os.environ["SLURM_JOB_ID"] == "1103606", "Incorrect compute controller identity")
    runtime_step = "1103606." + os.environ["SLURM_STEP_ID"]
    require(re.fullmatch(r"1103606\.[0-9]+", runtime_step), "Missing numbered zero-GRES controller")
    owners, pins = scope(root)
    by_parent = {parent: sorted(name for name, (p, _) in owners.items() if p == parent) for parent in sorted(PARENTS)}
    for parent, members in by_parent.items():
        ownership(parent, members)

    target = time.time() + 30
    write_json(root / "window.json", dict(request=request, controller=runtime_step, scheduled_epoch=target,
                                         parents=sorted(PARENTS), points=sampler.POINTS,
                                         GPUs_requested_for_sampling=0))
    (root / "reports").mkdir(mode=0o700)
    clean = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "PMI_", "PMIX_"))}
    children = {}
    for parent, members in by_parent.items():
        label = f"training80-syncbusy-{request[:12]}-{parent}"
        argv = ["srun", "--jobid=" + parent, "--job-name=" + label, "--exact", "--threads=1",
                "--immediate=8", "--nodes=40", "--ntasks=40", "--ntasks-per-node=1",
                "--nodelist=" + ",".join(members), "--cpus-per-task=1", "--mem=256M", "--gres=none",
                "--time=00:01:10", "--kill-on-bad-exit=0", PYTHON, "-B", str(Path(__file__).resolve()),
                "node", str(root), str(target)]
        with (root / f"{parent}.private.log").open("xb") as log:
            children[parent] = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log,
                                               stderr=subprocess.STDOUT, env=clean)

    statuses = {}
    for parent, process in children.items():
        try:
            statuses[parent] = process.wait(timeout=max(0.1, target + 22 - time.time()))
        except subprocess.TimeoutExpired:
            process.terminate()
            statuses[parent] = None
    records = {name: read_json(path, 16384) for name in owners
               if (path := root / "reports" / (name + ".json")).is_file()}
    result = sampler.aggregate(owners, pins, records, target)
    coverage = []
    for i in range(sampler.POINTS):
        rows = [record["rows"][i] for record in records.values()]
        values = [busy for row in rows for busy in row["busy"].values()]
        coverage.append(dict(point=i, valid_samples=len(values), nonzero_cards=sum(b > 0 for b in values),
                             mean_busy_percent=sum(values) / len(values) if values else None,
                             observed_read_span_seconds=max(r["closed_epoch"] for r in rows) -
                             min(r["opened_epoch"] for r in rows) if rows else None))
    result.update(request=request, controller_step=runtime_step, per_point=coverage, foreground_exits=statuses,
                  sampling_DCU_allocated=0, sustained_busy_qualified=False, report_only=True)
    write_json(root / "result.json", result)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0 if result["complete"] and all(rc == 0 for rc in statuses.values()) else 1


if __name__ == "__main__":
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("node", "capture"))
    parser.add_argument("root", type=Path)
    parser.add_argument("identity")
    args = parser.parse_args()
    if args.mode == "node":
        node(args.root, float(args.identity))
    else:
        raise SystemExit(capture(args.root, args.identity))
