"""Prepare bounded standby handoff batches from audited worker budgets."""

import argparse
from datetime import datetime
from hashlib import sha256
import json
import math
from pathlib import Path
import re

from report_training80_backlog import read_original_holds, read_queue_holds
from training_dispatch_allocation import REPLACEMENT_LEDGER, REPLACEMENT_KIND, allocation
from tuning_queue import TRAINING80_LEDGER


REPO = Path(__file__).resolve().parents[1]


def read_calendar(path, expected_hash, owners):
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != expected_hash:
        raise ValueError("calendar bytes differ")
    calendar = json.loads(raw)
    if calendar["mode"] != "preparation_only":
        raise ValueError("calendar must contain worker budgets")
    seen_nodes, seen_steps = set(), set()
    counts = dict.fromkeys({parent for parent, _ in owners.values()}, 0)
    for row in calendar["ordinary"]:
        node, parent, step = row["node"], row["parent"], row["previous"]
        expiry, prep = row["planned_expiry_epoch"], row["prepare_epoch"]
        start, seconds = row["worker_start_epoch"], row["worker_seconds"]
        if (node in seen_nodes or step in seen_steps or node not in owners or owners[node][0] != parent
                or not re.fullmatch(rf"{parent}\.[1-9][0-9]*", step)
                or not re.fullmatch("[0-9a-f]{32}", row["request"])
                or any(not re.fullmatch("[0-9a-f]{64}", row[key])
                       for key in ("expected_SHA256", "worker_SHA256"))
                or type(row["worker_PID"]) is not int or row["worker_PID"] <= 0
                or type(seconds) is not int or seconds <= 0
                or not all(math.isfinite(value) for value in (prep, start, expiry))
                or abs(start + seconds - expiry) > 1e-6 or abs(expiry - prep - 900) > 1e-6
                or abs(datetime.fromisoformat(row["planned_expiry_Beijing"]).timestamp() - expiry) > 1e-6
                or abs(datetime.fromisoformat(row["prepare_Beijing"]).timestamp() - prep) > 1e-6):
            raise ValueError(f"invalid worker budget for {node}")
        seen_nodes.add(node)
        seen_steps.add(step)
        counts[parent] += 1
    if counts != calendar["by_parent"]:
        raise ValueError("calendar is missing worker rows")
    return sorted(calendar["ordinary"], key=lambda row: (row["planned_expiry_epoch"], row["node"]))


def group_rows(rows, holds):
    excluded = {row["node"]: sorted(holds[row["node"]]) for row in rows if row["node"] in holds}
    eligible = [row for row in rows if row["node"] not in holds]
    batches = []
    for offset in range(0, len(eligible), 8):
        group = eligible[offset:offset + 8]
        targets = [
            dict(row, planned_end_epoch=row["planned_expiry_epoch"],
                 planned_end_Beijing=row["planned_expiry_Beijing"], previous_worker_pid=row["worker_PID"])
            for row in group
        ]
        batches.append(dict(batch=len(batches) + 1, rows=targets,
                            first_prepare_Beijing=group[0]["prepare_Beijing"],
                            first_expected_end=group[0]["planned_expiry_Beijing"],
                            last_expected_end=group[-1]["planned_expiry_Beijing"]))
    return batches, excluded


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("calendar", type=Path)
    cli.add_argument("calendar_sha256")
    cli.add_argument("output", type=Path)
    cli.add_argument("--exclude", action="append", default=[], metavar="NODE")
    args = cli.parse_args()
    kind, _, owners = allocation(REPLACEMENT_LEDGER.read_bytes())
    if kind != REPLACEMENT_KIND or len(owners) != 80 or set(args.exclude) - owners.keys():
        raise ValueError("expected the owned 80-node inventory")
    rows = read_calendar(args.calendar, args.calendar_sha256, owners)
    hold_rows = read_queue_holds(REPO / "local-only/tuning/queue.sqlite3")
    backlog = json.loads((REPO / "coordination/training/training80-backlog-v1.json").read_bytes())
    known_nodes = set(owners) | set(allocation(TRAINING80_LEDGER.read_bytes())[2])
    direct_holds, marker_errors, _ = read_original_holds(REPO, backlog["items"], known_nodes, hold_rows)
    holds = {node: {"assigned_to_separate_reconciliation"} for node in args.exclude}
    for hold in hold_rows:
        for node in hold["nodes"]:
            holds.setdefault(node, set()).add("durable_queue:" + hold["state"])
    for hold in direct_holds.values():
        for node in hold["nodes"]:
            holds.setdefault(node, set()).add("original_request:" + hold["state"])
    batches, excluded = group_rows(rows, holds)
    result = {
        "schema": "bw1000.training80.standby-renewal-batches.v1",
        "source_calendar_SHA256": args.calendar_sha256,
        "prepared_Beijing": datetime.now().astimezone().isoformat(),
        "preparation_only": True,
        "requires_original_terminal_and_fresh_device_and_queue_gate": True,
        "node_count": len(rows),
        "excluded": excluded,
        "local_marker_error_count": len(marker_errors),
        "candidate_batches_in_chronological_budget_order": batches,
    }
    encoded = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with args.output.open("xb") as stream:
        stream.write(encoded)
    print(json.dumps({"candidate_nodes": len(rows) - len(excluded), "excluded": excluded,
                      "batches": len(batches), "plan_SHA256": sha256(encoded).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
