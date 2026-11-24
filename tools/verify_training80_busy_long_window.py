"""Independently reduce bounded long-window receipts without exporting raw points."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re

from sample_training80_busy import require_pins
from training_dispatch_allocation import allocation


PARENTS = {"1103606": 40, "1194900": 40}
POINTS = 1801
CARDS = 640
INTERVAL_NS = 2_000_000_000
DURATION_NS = 3_600_000_000_000
SPAN_NS = 250_000_000
RECEIPT_MAX = 131_072
DEVICE_SECONDS = 2_304_000
STATUSES = {"COMPLETE", "MISSING", "MISSED_SCHEDULE", "TRANSPORT_ERROR",
            "FRAME_REJECTED", "FRAME_LIMIT", "STOP_REQUESTED", "NONCE_ERROR"}


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def bounded(path):
    require(not path.is_symlink(), "unexpected receipt link")
    with path.open("rb") as stream:
        raw = stream.read(RECEIPT_MAX + 1)
    require(0 < len(raw) <= RECEIPT_MAX, "receipt exceeds its bound")
    return raw


def check_rows(point, owners, pins, intent):
    rows = point["worker_reports"]
    sent = point["controller_send_monotonic_ns"]
    require(len(rows) <= len(owners) and (not rows or type(sent) is int), "invalid respondent set")
    seen, cells, arrivals = set(), {}, []
    for row in rows:
        node = row["node"]
        parent = owners[node][0]
        require(node not in seen and row["parent"] == parent and row["step"] == intent["worker_steps"][parent]
                and row["pci_pins_SHA256"] == intent["pci_pins_SHA256"] and row["request"] == intent["request"]
                and row["event"] == "pulse" and row["sequence"] == point["sequence"] and row["nonce"] == point["nonce"],
                "respondent identity differs")
        seen.add(node)
        arrival = row["controller_receipt_monotonic_ns"]
        require(type(arrival) is int and sent < arrival <= min(sent + SPAN_NS, intent["absolute_deadline_ns"])
                and type(row["local_read_ns"]) is int and 0 <= row["local_read_ns"] <= arrival - sent,
                "respondent is outside the upper clock bound")
        arrivals.append(arrival)
        good, missed = row["busy"], row["failed_pci"]
        require(isinstance(good, dict) and isinstance(missed, list) and len(missed) == len(set(missed))
                and not set(good).intersection(missed) and set(good).union(missed) == set(pins[node])
                and all(type(value) is int and 0 <= value <= 100 for value in good.values())
                and (row["error"] is None or row["error"] == "DRM_INVENTORY_UNVERIFIED" and not good),
                "respondent PCI denominator differs")
        cells.update({(node, pci): (value, arrival) for pci, value in good.items()})
    require(point["observed_samples"] == len(cells) and point["responded_nodes"] == len(seen)
            and point["controller_last_receipt_monotonic_ns"] == (max(arrivals) if arrivals else None)
            and point["sample_span_upper_bound_ns"] == (max(arrivals) - sent if arrivals else None),
            "posted pulse counts or clocks differ")
    return cells


def verify(output, owners, pins, request, pins_sha):
    """Verify a terminal or interrupted collector namespace from its exact point receipts."""
    require(Counter(owner[0] for owner in owners.values()) == PARENTS, "wrong owned nodes")
    require_pins(owners, pins)
    output = Path(output)
    raw = bounded(output / "intent.json")
    intent = json.loads(raw)
    earliest = intent["earliest_first_monotonic_ns"]
    deadline = intent["absolute_deadline_ns"]
    require(re.fullmatch(r"[0-9a-f]{32}", request) and re.fullmatch(r"[0-9a-f]{64}", pins_sha)
            and intent["request"] == request and intent["pci_pins_SHA256"] == pins_sha
            and intent["planned_points"] == POINTS and intent["expected_samples"] == POINTS * CARDS
            and intent["expected_device_seconds"] == DEVICE_SECONDS and intent["parents"] == PARENTS
            and intent["interval_ns"] == INTERVAL_NS and intent["span_limit_ns"] == SPAN_NS
            and intent["GPU_requested"] == 0 and intent["replay_allowed"] is False
            and type(earliest) is int and type(deadline) is int
            and intent["earliest_final_planned_monotonic_ns"] == earliest + DURATION_NS
            and earliest + DURATION_NS + SPAN_NS <= deadline <= earliest + DURATION_NS + INTERVAL_NS,
            "wrong one-hour collector intent")
    steps = intent["worker_steps"]
    require(set(steps) <= set(PARENTS)
            and all(re.fullmatch(parent + r"\.[0-9]+", step) for parent, step in steps.items()),
            "invalid sampler step provenance")
    intent_pin = sha256(raw).hexdigest()
    anchor = earliest
    first = last = None
    prior = {}
    nonces = set()
    chain = bytes(32)
    count = good_points = good_cells = observed = device_seconds = busy_seconds = 0
    absent = False
    for seq in range(POINTS):
        path = output / f"point-{seq:04d}.json"
        if not path.exists():
            absent = True
            continue
        require(not absent, "noncontiguous point receipts")
        raw = bounded(path)
        point = json.loads(raw)
        chain = sha256(chain + sha256(raw).digest()).digest()
        require(point["sequence"] == seq and point["expected_samples"] == CARDS
                and point["status"] in STATUSES and type(point["rejected_frames"]) is int
                and point["rejected_frames"] >= 0, "point sequence or status differs")
        sent = point["controller_send_monotonic_ns"]
        if sent is None:
            require(point["nonce"] is None and not point["worker_reports"], "unsent point has responses")
        else:
            require(set(steps) == set(PARENTS) and intent["ready_nodes"] == 80,
                    "sent point lacks complete observer startup identities")
            planned = anchor + seq * INTERVAL_NS
            require(type(sent) is int and planned <= sent <= planned + SPAN_NS and sent + SPAN_NS <= deadline
                    and re.fullmatch(r"[0-9a-f]{32}", point["nonce"]) and point["nonce"] not in nonces,
                    "sent point exceeded its schedule or repeated its nonce")
            nonces.add(point["nonce"])
            if seq == 0:
                first = anchor = sent
            if seq == POINTS - 1:
                last = sent
        planned = anchor + seq * INTERVAL_NS
        require(point["planned_controller_monotonic_ns"] == planned, "point moved away from the original schedule")
        cells = check_rows(point, owners, pins, intent)
        observed += len(cells)
        if point["rejected_frames"] or point["status"] == "FRAME_LIMIT":
            cells = {}
        valid = len(cells)
        complete = valid == CARDS and point["status"] == "COMPLETE"
        require(point["complete"] == complete and point["valid_samples"] == valid
                and point["missing_samples"] == CARDS - valid, "posted point denominator differs")
        good_cells += valid
        good_points += complete
        if seq:
            pairs = [(prior[key][0], cv) for key, (cv, later) in cells.items()
                     if key in prior and INTERVAL_NS - 2 * SPAN_NS <= later - prior[key][1] <= INTERVAL_NS + 2 * SPAN_NS]
            duration = len(pairs) * 2
            integral = sum(left + right for left, right in pairs)
            stated = point["preceding_interval"]
            require((stated["start_planned_monotonic_ns"], stated["end_planned_monotonic_ns"],
                     stated["expected_device_seconds"], stated["valid_device_seconds"],
                     stated["unknown_device_seconds"], stated["busy_percent_device_seconds"])
                    == (planned - INTERVAL_NS, planned, 1280, duration, 1280 - duration, integral),
                    "posted adjacent interval does not recompute")
            device_seconds += duration
            busy_seconds += integral
        prior = cells
        count += 1

    path = output / "outcome.json"
    original = json.loads(bounded(path)) if path.exists() else None
    elapsed = last - first if first is not None and last is not None else None
    complete = (original is not None and original["stop_reason"] is None and count == good_points == POINTS
                and elapsed is not None and elapsed >= DURATION_NS and device_seconds == DEVICE_SECONDS)
    report = dict(request=request, persisted_points=count, complete_points=good_points,
                  expected_samples=POINTS * CARDS, valid_samples=good_cells, observed_samples=observed,
                  missing_samples=POINTS * CARDS - good_cells, expected_device_seconds=DEVICE_SECONDS,
                  qualified_device_seconds=device_seconds, unknown_device_seconds=DEVICE_SECONDS - device_seconds,
                  measured_send_span_ns=elapsed, hour_interval_coverage_complete=complete,
                  full_hour_interval_busy_percent=busy_seconds / DEVICE_SECONDS if complete else None,
                  intent_SHA256=intent_pin, point_digest_chain_SHA256=chain.hex(), terminal_outcome_present=original is not None)
    if original is not None:
        excluded = {"terminal_outcome_present"}
        require(all(report[k] == original[k] for k in report if k not in excluded),
                "collector summary disagrees with original points")
    report["zero_GRES_terminal_accounting_verified"] = False
    report["sustained_interval_qualified"] = False
    return report


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output", type=Path)
    cli.add_argument("--ledger", type=Path, required=True)
    cli.add_argument("--pci-pins", type=Path, required=True)
    cli.add_argument("--request", required=True)
    args = cli.parse_args()
    _, _, owners = allocation(bounded(args.ledger))
    raw = bounded(args.pci_pins)
    report = verify(args.output, owners, json.loads(raw), args.request, sha256(raw).hexdigest())
    print(json.dumps(report, sort_keys=True))
    return int(not report["hour_interval_coverage_complete"])


if __name__ == "__main__":
    raise SystemExit(main())
