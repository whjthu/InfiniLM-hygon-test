"""Join one observed allocation snapshot to the last verified per-node standby records."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import re


def read_pin(spec):
    path, expected = spec
    raw = Path(path).read_bytes()
    if sha256(raw).hexdigest() != expected:
        raise ValueError(f"input SHA-256 differs: {Path(path).name}")
    return json.loads(raw), expected


def predecessor_index(sources):
    index = {}
    for data, digest in sources:
        rows = data.get("per_node", data.get("nodes", data.get("steps")))
        if not isinstance(rows, dict):
            raise ValueError("unsupported predecessor evidence")
        for name, row in rows.items():
            step = row.get("new_step", "")
            if not re.fullmatch(r"(?:1103606|1194900)\.[0-9]+", step) or row.get("growth") != 8:
                raise ValueError(f"unverified predecessor for {name}")
            entry = {"step": step, "source_SHA256": digest}
            if "worker_start" in row or "worker_seconds" in row:
                epoch = row["worker_start"] + row["worker_seconds"]
                if not math.isfinite(epoch) or row["worker_seconds"] <= 0:
                    raise ValueError(f"invalid observed worker budget for {name}")
                entry.update(expected_worker_end_epoch=epoch, original_request=row["request"],
                             original_dispatch_intent=row["original_dispatch_intent"],
                             original_two_point_receipt=row["receipt"])
            index[name] = entry
    return index


def crosswalk(snapshot, previous, excluded):
    mapping = snapshot["nodes"]
    membership = snapshot["membership"]
    parents = membership["parents"]
    if (not snapshot.get("complete") or set(parents) != {"1103606", "1194900"}
            or membership["distinct_nodes"] != 80 or membership["allocated_DCU"] != 640
            or len(mapping) != 80):
        raise ValueError("expected the current owned 640-card membership")
    for parent, group in parents.items():
        if (group["state"] != "RUNNING" or not group["ledger_exact"] or not group["step_locations_exact"]
                or not group["owned_by_observer"]):
            raise ValueError(f"unqualified parent: {parent}")
    missing = []
    plan = []
    for name, node in mapping.items():
        parent = node["parent"]
        usage = parents[parent]["numbered_step_usage"]
        bounds = usage["assigned_DCU_bounds_by_node"][name]
        last = previous.get(name)
        if last and not last["step"].startswith(parent + "."):
            raise ValueError(f"predecessor parent differs: {name}")
        row = {"node": name, "parent": parent, "previous": last, "numbered_DCU_bounds": bounds,
               "excluded": name in excluded}
        if bounds == [0, 0]:
            missing.append(row)
        if last and "expected_worker_end_epoch" in last and name not in excluded:
            plan.append(row)
    plan.sort(key=lambda row: (row["previous"]["expected_worker_end_epoch"], row["node"]))
    missing.sort(key=lambda row: (row["parent"], row["node"]))
    return {"source_request": snapshot["request_id"], "observed_epoch": snapshot["observed_epoch"],
            "minimum_numbered_DCU": membership["minimum_numbered_step_assigned_DCU"],
            "maximum_numbered_DCU": membership["maximum_numbered_step_assigned_DCU"],
            "no_numbered_DCU_at_observation": missing, "expected_expiry_order": plan,
            "dispatch_authorized": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", nargs=2, metavar=("FILE", "SHA256"), required=True)
    parser.add_argument("--previous", nargs=2, action="append", metavar=("FILE", "SHA256"), required=True,
                        help="Verified handoff sources in time order")
    parser.add_argument("--exclude", action="append", default=[], help="Node with a hold, conflict, or consumed renewal")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    snapshot, snapshot_pin = read_pin(args.snapshot)
    report = crosswalk(snapshot, predecessor_index(map(read_pin, args.previous)), set(args.exclude))
    report.update(schema="bw1000.training80.standby-predecessor-crosswalk.v1", snapshot_SHA256=snapshot_pin,
                  prior_sources_SHA256=[row[1] for row in args.previous])
    with args.output.open("x", encoding="utf8") as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
