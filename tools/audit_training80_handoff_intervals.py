"""Measure phases of an archived standby handoff from its original receipts."""

import argparse
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
from statistics import median


BEIJING = timezone(timedelta(hours=8))
GROUP = "old_end_to_latest_group_end_s"
QUERY = "latest_group_end_to_terminal_query_start_s"
WORKER = "terminal_query_start_to_worker_s"


def pinned_json(path, digest):
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != digest:
        raise ValueError(f"archived bytes changed: {path.name}")
    return json.loads(raw)


def slurm_end(value):
    timestamp = datetime.fromisoformat(value)
    if timestamp.tzinfo is not None:
        raise ValueError("unexpected timezone in original Slurm End")
    return timestamp.replace(tzinfo=BEIJING).timestamp()


def measure_group(nodes, receipt):
    if receipt["rc"] != 0 or receipt["observed"]["plans_only"] is not True:
        raise ValueError("receipt is not a successful terminal observation")
    observed = receipt["observed"]
    ready = {t["node"]: t for t in observed["targets"] if t["terminal_ready"] is True}
    latest_end = max(slurm_end(node["prior_end"]) for node in nodes)
    result = []
    for node in nodes:
        target = ready[node["node"]]
        accounting = target["accounting"]
        end = slurm_end(node["prior_end"])
        if (node["prior_step"] != target["step"] or node["prior_step"] != accounting[0]
                or accounting[2:4] != ["COMPLETED", "0:0"] or accounting[5] != node["node"]
                or accounting[8] != node["prior_end"]
                or node["prior_terminal_SHA256"] != target["terminal_SHA256"]
                or node["growth_pass"] is not True or node["advancing_DCU"] != 8):
            raise ValueError("terminal or active worker identity differs")
        intervals = {
            GROUP: latest_end - end,
            QUERY: observed["epoch"] - latest_end,
            WORKER: node["worker_start_epoch"] - observed["epoch"],
        }
        if (min(intervals.values()) < 0
                or abs(sum(intervals.values()) - node["old_end_to_worker_seconds"]) > .001):
            raise ValueError("phase chronology disagrees with the archived interval")
        result.append(intervals)
    return result


def summarize(samples, key):
    values = [row[key] for row in samples]
    return {"minimum_seconds": min(values), "median_seconds": median(values), "maximum_seconds": max(values),
            "eight_card_handoff_hours": 8 * sum(values) / 3600}


def audit(archive, archive_pin, receipts, expected_nodes):
    doc = pinned_json(archive, archive_pin)
    attempts = {row["request"]: row for row in doc["attempts"]}
    groups = {}
    for row in doc["per_node"]:
        groups.setdefault(row["request"], []).append(row)
    if (len(doc["per_node"]) != expected_nodes or len({row["node"] for row in doc["per_node"]}) != expected_nodes
            or set(attempts) != set(groups) or len(attempts) != len(doc["attempts"])):
        raise ValueError("node or original request denominator differs")

    samples, sources = [], []
    for request, nodes in groups.items():
        attempt = attempts[request]
        name = attempt["old_terminal_receipt"]
        if Path(name).name != name or any(ch in name for ch in "/\\"):
            raise ValueError("invalid original receipt basename")
        receipt_pin = attempt["old_terminal_SHA256"]
        samples.extend(measure_group(nodes, pinned_json(receipts / name, receipt_pin)))
        sources.append({"batch": attempt["batch"], "nodes": len(nodes), "query_receipt_SHA256": receipt_pin})
    return {"node_count": len(samples), "worker_DCU_count": len(samples) * 8,
            "matched_original_requests": len(groups), "source_archive_SHA256": archive_pin,
            "source_receipts": sources, "phases": {field: summarize(samples, field) for field in (GROUP, QUERY, WORKER)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("archive_SHA256")
    parser.add_argument("receipt_directory", type=Path)
    parser.add_argument("expected_nodes", type=int)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = audit(args.archive, args.archive_SHA256, args.receipt_directory, args.expected_nodes)
    raw = (json.dumps(data, allow_nan=False, sort_keys=True, indent=2) + "\n").encode()
    with args.output.open("xb") as stream:
        stream.write(raw)
    print(sha256(raw).hexdigest())


if __name__ == "__main__":
    main()
