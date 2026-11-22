"""Consume exact, independently validated target-clock bounds for common15."""

from hashlib import sha256
import json
import re

from contract import ACTIVE_SECONDS, COMMON_OFFSET, NANO, PROBES, need
from scope import MAX_NODES, REQUEST


def audit(terminal_raw, clock_raw=None, verified_clock_sha256=None):
    terminal = json.loads(terminal_raw)
    nodes = terminal["nodes"]
    need(terminal["request"] == REQUEST and 0 < len(nodes) == terminal["planned_nodes"] <= MAX_NODES
         and terminal["planned_cards"] == 8*len(nodes), "original common-window scope differs")
    result = dict(request=REQUEST, expected_nodes=len(nodes), expected_cards=8*len(nodes),
                  clock_inter_node_bounded=False, simultaneous_sustained_ns=None,
                  simultaneous_scope_busy_mean_percent=None, simultaneous_scope_samples=0,
                  expected_common_samples=8*len(nodes)*len(PROBES), complete_640_scope=False,
                  method="conservative intersection of completed common15 owner windows")
    result["locally_complete_nodes"] = sum(node.get("common_window_epoch_ns") is not None
                                            and node.get("passed") is True for node in nodes.values())
    if clock_raw is None:
        return result
    need(re.fullmatch(r"[a-f0-9]{64}", verified_clock_sha256 or "")
         and sha256(clock_raw).hexdigest() == verified_clock_sha256,
         "independently validated clock receipt SHA differs")
    clock = json.loads(clock_raw)
    need(clock["source_role"] == "INDEPENDENT_BOUNDED_TARGET_CLOCKS"
         and clock["request"] == REQUEST
         and clock["terminal_SHA256"] == sha256(terminal_raw).hexdigest(),
         "clock receipt belongs to a different observation")
    bounds = clock["nodes"]
    epoch, starts, ends = terminal["campaign_epoch"], [], []
    points_by_node = {}
    common_start = round((epoch+COMMON_OFFSET)*NANO)
    common_end = round((epoch+COMMON_OFFSET+ACTIVE_SECONDS)*NANO)
    need(set(bounds) <= set(nodes), "clock receipt includes nodes outside the original request")
    result["independently_bounded_nodes"] = len(bounds)
    for name, bound in bounds.items():
        node = nodes[name]
        interval = node.get("common_window_epoch_ns")
        if node.get("passed") is not True or interval is None:
            continue
        start, end = interval
        epsilon, begin, finish = (bound[key] for key in ("max_abs_error_ns", "valid_from_epoch_ns",
                                                       "valid_until_epoch_ns"))
        need(all(type(value) is int for value in (start, end, epsilon, begin, finish))
             and bound["step"] == node["original_step"]
             and re.fullmatch(r"[a-f0-9]{64}", bound["raw_clock_evidence_SHA256"])
             and begin <= start < end <= finish and 0 <= epsilon
             and common_start <= start <= common_start+500_000_000
             and common_end-150_000_000 <= end <= common_end+500_000_000,
             "common owner window lacks its exact bounded clock interval")
        starts.append(start+epsilon)
        ends.append(end-epsilon)
        points_by_node[name] = (epsilon,node.get("common_sample_observations", []))
    if len(starts) != len(nodes):
        return result
    begin, finish = max(starts), min(ends)
    result.update(clock_inter_node_bounded=True, simultaneous_sustained_ns=max(0,finish-begin),
                  validated_clock_SHA256=verified_clock_sha256)
    if begin >= finish:
        return result

    grid = {common_start+offset*NANO for offset in PROBES}
    points = {tick:[] for tick in grid}
    for epsilon, observations in points_by_node.values():
        seen = set()
        for observation in observations:
            tick = observation["planned_epoch_ns"]
            opened, closed, total = (observation[key] for key in (
                "opened_epoch_ns","closed_epoch_ns","sum_busy_percent"))
            need(tick in grid and tick not in seen and observation["cards"]==8
                 and all(type(value) is int for value in (opened,closed,total))
                 and tick <= opened <= closed <= tick+150_000_000 and 0<=total<=800,
                 "original common15 driver sample differs")
            seen.add(tick)
            first, last = opened-epsilon, closed+epsilon
            if begin <= first <= last <= finish:
                points[tick].append((first,last,total))
    complete = [rows for rows in points.values() if len(rows)==len(nodes)]
    result["simultaneous_scope_samples"] = len(complete)*8*len(nodes)
    result["common_observation_skew_upper_bound_ns"] = max(
        (max(row[1] for row in rows)-min(row[0] for row in rows) for rows in complete), default=None)
    if len(complete) == len(PROBES):
        result["simultaneous_scope_busy_mean_percent"] = sum(
            row[2] for rows in complete for row in rows)/result["simultaneous_scope_samples"]
    return result
