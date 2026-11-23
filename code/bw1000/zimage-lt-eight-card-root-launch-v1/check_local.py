"""Check the HOST Slurm adapter against the original read-only Slurm output."""

import argparse
import importlib
import json
import re
from pathlib import Path
import sys


def expand(hosts):
    out = []
    for expression in re.split(r",(?![^\[]*\])", hosts):
        match = re.fullmatch(r"([a-z0-9]+)\[([0-9,-]+)\]([a-z0-9]*)", expression)
        if match is None:
            out.append(expression)
            continue
        prefix, values, suffix = match.groups()
        for value in values.split(","):
            low, separator, high = value.partition("-")
            for number in range(int(low), int(high if separator else low) + 1):
                out.append(f"{prefix}{number:0{len(low)}d}{suffix}")
    return "\n".join(out)


def run(stage, document):
    sys.path.insert(0, str(stage.resolve()))
    scheduler = importlib.import_module("scheduler")
    host = importlib.import_module("host_scope")
    launcher = importlib.import_module("host_run")
    source = document["part"]
    epoch = document["epoch"]
    positions = {"job": "job", "node": "node", "step": "steps"}

    def prior(*argv):
        if argv[:3] == ("scontrol", "show", "hostnames"):
            return expand(argv[-1])
        key = positions[argv[3]] if argv[0] == "scontrol" else "accounting"
        assert source[key]["rc"] == 0
        return source[key]["data"]

    snapshot = scheduler.collect(run=prior, clock=lambda: epoch)
    uid = int(re.search(host.PARENT_UID, snapshot["parent"]["UserId"])[1])
    records, active = host.scheduler_scope(snapshot, epoch, uid)
    old = host.exact_row(records, host.OLD_LABEL)
    assert len(snapshot["parent_nodes"]) == 40
    assert snapshot["target_configured"]["AllocatedMemory_MiB"] == 240000
    assert old[:4] == [host.OLD_STEP, host.OLD_LABEL, "RUNNING", "0:0"]
    assert any(row["StepId"] == host.OLD_STEP for row in active)
    assert launcher.relevant(snapshot)["accounting"] == [old]
    try:
        host.old_standby(records)
    except ValueError as error:
        assert str(error) == "PREDECESSOR_STILL_PROTECTED"
    else:
        raise AssertionError("Live original .2075 was accepted as terminated")

    def broken_node(*argv):
        result = prior(*argv)
        return result.replace("AllocMem=240000", "AllocMem=16384") if argv[3:4] == ("node",) else result

    try:
        scheduler.collect(run=broken_node, clock=lambda: epoch)
    except ValueError as error:
        assert str(error) == "TARGET_ADMISSION_RESOURCES"
    else:
        raise AssertionError("Insufficient step resource declaration passed")
    return {"original_accounting_rows": len(records), "original_numbered_steps": len(snapshot["numbered_steps"]),
            "parent_nodes": 40, "protected_live_predecessor": True, "insufficient_memory_rejected": True,
            "resource_snapshot_only": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    parser.add_argument("original_observation", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.stage, json.loads(args.original_observation.read_bytes())), sort_keys=True))
