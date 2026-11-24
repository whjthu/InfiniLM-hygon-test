"""Persist a fixed-hour driver-counter schedule for two already-bound zero-GRES steps.

The HOST caller owns Slurm admission, control pipes, and terminal accounting.
This module issues pulses on the supplied transport and never launches a step.
"""

from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import time
from uuid import uuid4

from collect_training80_busy_pulses import INTERVAL_NS, Receiver, require, require_pins, summarize, token


PARENTS = {"1103606": 40, "1194900": 40}
CARDS = 640
DURATION_NS = 3_600_000_000_000
POINTS = DURATION_NS // INTERVAL_NS + 1
SPAN_NS = 250_000_000
MAX_START_LAG_NS = 250_000_000
STOP_CHECK_NS = 200_000_000
MAX_REPLIES = 160
MAX_RECEIPT_BYTES = 131_072
EXPECTED_DEVICE_SECONDS = (POINTS - 1) * CARDS * 2


def write_receipt(path, record):
    """Atomically publish one small receipt inside the unique collector directory."""
    raw = (json.dumps(record, sort_keys=True, allow_nan=False) + "\n").encode()
    require(0 < len(raw) <= MAX_RECEIPT_BYTES and not path.exists(), "invalid or repeated receipt")
    pending = path.with_name(path.name + ".pending")
    with pending.open("xb") as out:
        out.write(raw)
        out.flush()
        os.fsync(out.fileno())
    os.replace(pending, path)
    if os.name == "posix":
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    return sha256(raw).digest()


def wait_for(target, deadline, clock, sleep, stopping):
    """Wait on an absolute clock. Never turn stale schedule slots into a burst."""
    while True:
        if stopping():
            return "STOP_REQUESTED"
        now = clock()
        if now >= deadline:
            return "DEADLINE"
        if now >= target:
            return None
        sleep(min(target - now, deadline - now, STOP_CHECK_NS) / 1e9)


def read_pulse(receiver, transport, packet, sent, cutoff, stopping):
    """Route frames through the short sampler's original authenticated-ready/PCI validator."""
    rows, stale, late, rejected = {}, 0, 0, 0
    issue = None
    try:
        for parent, row, arrived in transport.receive_until(cutoff):
            if stopping():
                return rows, stale, late, rejected, "STOP_REQUESTED"
            if len(rows) + stale + late + rejected >= MAX_REPLIES:
                return rows, stale, late, rejected, "FRAME_LIMIT"
            if (isinstance(row, dict) and row.get("request") == packet["request"]
                    and type(row.get("sequence")) is int and row["sequence"] < packet["sequence"]):
                stale += 1
                continue
            if type(arrived) is not int or arrived <= sent:
                rejected += 1
                continue
            if arrived > cutoff:
                late += 1
                continue
            try:
                receiver.accept(row, parent, "pulse", rows, arrived, packet, sent)
            except (KeyError, TypeError, ValueError):
                rejected += 1
            if len(rows) == len(receiver.owners):
                break
    except (OSError, TypeError, ValueError):
        issue = "TRANSPORT_ERROR"
    if not issue and stopping():
        issue = "STOP_REQUESTED"
    return rows, stale, late, rejected, issue


def point_record(sequence, planned, sent, nonce, rows, receiver, reason, stale=0, late=0, rejected=0):
    original = summarize(receiver.owners, receiver.pins,
                         [{"sent_ns": sent, "nonce": nonce, "rows": rows}], SPAN_NS, True)
    point = original["points"][0]
    valid = point["valid_samples"]
    complete = point["complete"] and reason is None and rejected == 0
    if rejected or reason == "FRAME_LIMIT":
        # Keep the observed records and their provenance, but the compromised point has no qualified values.
        valid = 0
    point.update(sequence=sequence, planned_controller_monotonic_ns=planned,
                 expected_samples=CARDS, valid_samples=valid, missing_samples=CARDS - valid,
                 observed_samples=original["valid_samples"], complete=complete,
                 status=reason or ("FRAME_REJECTED" if rejected else ("COMPLETE" if complete else "MISSING")),
                 stale_frames=stale, late_frames=late, rejected_frames=rejected)
    if not complete:
        point.update(fleet_mean_busy_percent=None, nonzero_cards=None, max_busy_percent=None)
    return point


def point_values(point):
    if not point["valid_samples"]:
        return {}
    return {(row["node"], pci): (value, row["controller_receipt_monotonic_ns"])
            for row in point["worker_reports"] for pci, value in row["busy"].items()}


def preceding_interval(prior, current, planned):
    """Interpolate only adjacent valid endpoints, over one fixed 2 s schedule interval."""
    common = prior.keys() & current.keys()
    values = [(prior[card][0], current[card][0]) for card in common
              if abs(current[card][1] - prior[card][1] - INTERVAL_NS) <= MAX_START_LAG_NS + SPAN_NS]
    seconds = 2 * len(values)
    # Trapezoidal weighting for a 2 s interval: (left + right) / 2 * 2.
    integrated = sum(left + right for left, right in values)
    return dict(start_planned_monotonic_ns=planned - INTERVAL_NS, end_planned_monotonic_ns=planned,
                expected_device_seconds=2 * CARDS, valid_device_seconds=seconds,
                unknown_device_seconds=2 * CARDS - seconds, busy_percent_device_seconds=integrated,
                integration="trapezoid_over_scheduled_2s_from_adjacent_timing_qualified_points")


def collect_window(output: Path, receiver: Receiver, transport, first_ns, absolute_deadline_ns, *,
                   stopping, clock=time.monotonic_ns, sleep=time.sleep, nonce=lambda: uuid4().hex):
    """Run the 1,801 scheduled samples from t0 through t0 + 3600 s.

    The transport provides send(packet, deadline_ns) and receive_until(deadline_ns), yielding
    (parent, parsed row, controller arrival ns). Both calls must honor that absolute cutoff.
    The existing Receiver must contain all 80 real Ready events before pulses are sent.
    """
    parents = Counter(owner[0] for owner in receiver.owners.values())
    require(parents == PARENTS and len(receiver.owners) == 80, "wrong 80-node allocation")
    require_pins(receiver.owners, receiver.pins)
    require(token(receiver.request, 32) and token(receiver.pins_sha, 64), "wrong measurement identity")
    last_ns = first_ns + DURATION_NS
    require(type(first_ns) is int and type(absolute_deadline_ns) is int and first_ns > 0
            and last_ns + SPAN_NS <= absolute_deadline_ns <= last_ns + INTERVAL_NS,
            "absolute cutoff must cover the final pulse and leave a bounded tail")
    output = Path(output)
    output.mkdir(mode=0o700)
    plan = dict(request=receiver.request, parents=dict(parents), ready_nodes=len(receiver.ready),
                worker_steps=dict(receiver.parent_steps), pci_pins_SHA256=receiver.pins_sha,
                planned_points=POINTS, expected_samples=POINTS * CARDS, nodes=80, cards=CARDS,
                earliest_first_monotonic_ns=first_ns, earliest_final_planned_monotonic_ns=last_ns,
                absolute_deadline_ns=absolute_deadline_ns, interval_ns=INTERVAL_NS, span_limit_ns=SPAN_NS,
                expected_device_seconds=EXPECTED_DEVICE_SECONDS,
                replay_allowed=False, GPU_requested=0)
    plan_pin = write_receipt(output / "intent.json", plan)
    chain = b"\0" * 32
    seen_nonces = set()
    persisted = total_valid = total_observed = complete_points = 0
    interval_seconds = interval_busy = 0
    total_busy = 0.0
    prior = {}
    first_sent = last_sent = None
    anchor = first_ns
    stop_reason = None
    if set(receiver.ready) != set(receiver.owners) or set(receiver.parent_steps) != set(parents):
        stop_reason = "STARTUP_INCOMPLETE"
    for seq in range(POINTS):
        if stop_reason:
            break
        planned = anchor + seq * INTERVAL_NS
        stop_reason = wait_for(planned, absolute_deadline_ns, clock, sleep, stopping)
        if stop_reason:
            break
        sent = clock()
        if sent > planned + MAX_START_LAG_NS or sent + SPAN_NS > absolute_deadline_ns:
            point = point_record(seq, planned, None, None, {}, receiver, "MISSED_SCHEDULE")
        else:
            challenge = nonce()
            if not token(challenge, 32) or challenge in seen_nonces:
                stop_reason = "NONCE_ERROR"
                point = point_record(seq, planned, None, None, {}, receiver, stop_reason)
            else:
                seen_nonces.add(challenge)
                packet = dict(request=receiver.request, sequence=seq, nonce=challenge)
                if seq == 0:
                    # Anchor the full hour to the actual first send, not an earlier readiness estimate.
                    anchor = planned = first_sent = sent
                if seq == POINTS - 1:
                    last_sent = sent
                cutoff = min(sent + SPAN_NS, absolute_deadline_ns)
                try:
                    transport.send(packet, cutoff)
                except (OSError, TypeError, ValueError):
                    stop_reason = "TRANSPORT_ERROR"
                    rows, stale, late, rejected = {}, 0, 0, 0
                else:
                    rows, stale, late, rejected, stop_reason = read_pulse(receiver, transport, packet, sent, cutoff, stopping)
                point = point_record(seq, planned, sent, challenge, rows, receiver, stop_reason, stale, late, rejected)
        current = point_values(point)
        if seq:
            interval = preceding_interval(prior, current, planned)
            interval_seconds += interval["valid_device_seconds"]
            interval_busy += interval["busy_percent_device_seconds"]
            point["preceding_interval"] = interval
        prior = current
        point_pin = write_receipt(output / f"point-{seq:04d}.json", point)
        chain = sha256(chain + point_pin).digest()
        persisted += 1
        total_valid += point["valid_samples"]
        total_observed += point["observed_samples"]
        if point["complete"]:
            complete_points += 1
            total_busy += point["fleet_mean_busy_percent"]
    elapsed = last_sent - first_sent if first_sent is not None and last_sent is not None else None
    complete = complete_points == POINTS and stop_reason is None and elapsed is not None and elapsed >= DURATION_NS
    interval_complete = complete and interval_seconds == EXPECTED_DEVICE_SECONDS
    result = dict(request=receiver.request, planned_points=POINTS, persisted_points=persisted,
                  complete_points=complete_points, expected_samples=POINTS * CARDS, valid_samples=total_valid,
                  observed_samples=total_observed, missing_samples=POINTS * CARDS - total_valid,
                  unattempted_range=[persisted, POINTS - 1] if persisted < POINTS else None,
                  intent_SHA256=plan_pin.hex(), point_digest_chain_SHA256=chain.hex(),
                  stop_reason=stop_reason, complete_counter_window=complete, measured_send_span_ns=elapsed,
                  expected_device_seconds=EXPECTED_DEVICE_SECONDS, qualified_device_seconds=interval_seconds,
                  unknown_device_seconds=EXPECTED_DEVICE_SECONDS - interval_seconds, hour_interval_coverage_complete=interval_complete,
                  covered_interval_busy_percent=interval_busy / interval_seconds if interval_seconds else None,
                  full_hour_interval_busy_percent=interval_busy / EXPECTED_DEVICE_SECONDS if interval_complete else None,
                  fleet_sample_mean_busy_percent=total_busy / POINTS if complete else None,
                  zero_GRES_terminal_accounting_verified=False, sustained_interval_qualified=False)
    write_receipt(output / "outcome.json", result)
    return result
