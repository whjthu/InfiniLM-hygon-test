"""Frozen cohort schedule and equivalent-useful-work contracts for graph depth."""

import math


NANO = 10**9
M, K, N = 186, 4096, 1024
ORDER = ("graph64", "graph32", "graph32", "graph64", "common64")
DEPTH = dict(graph32=32, graph64=64, common64=64)
SETUP_OPEN = (0, 20, 40)
FIRST_ACTIVE = (48, 78, 110)
OFFSETS = (0, 64, 128, 192)
COMMON_OFFSET = 375
ACTIVE_SECONDS = 60
PROBES = tuple(range(2, ACTIVE_SECONDS, 5))
QUARTER_SECONDS = 15
WORKER_SECONDS, SLURM_SECONDS, CONTROLLER_SECONDS = 455, 470, 510
EARLY = frozenset(("f13r3n13", "f13r3n14", "f13r4n05", "f13r4n06", "f13r4n07", "f13r4n08"))
LATE = frozenset(("f12r2n10", "f12r2n11", "f12r2n13", "f13r1n10"))


def need(condition, message):
    if not condition:
        raise ValueError(message)


def assign_cohorts(shards):
    assignment = {}
    load = [0, 0, 0]
    ordinary = []
    for label, unit in sorted(shards.items()):
        nodes = set(unit["nodes"])
        if nodes & LATE:
            pick = 2
        elif nodes and nodes <= EARLY:
            pick = 0
        else:
            ordinary.append((label, nodes))
            continue
        assignment[label] = pick
        load[pick] += len(nodes)
    for label, nodes in ordinary:
        pick = min((1, 2), key=load.__getitem__)
        assignment[label] = pick
        load[pick] += len(nodes)
    return assignment


def cohort_for(plan, node):
    [label] = [label for label, unit in plan["shards"].items() if node in unit["nodes"]]
    return plan["cohort_by_shard"][label]


def grid(plan, cohort):
    epoch = plan["campaign_epoch"]
    need(type(epoch) is float and math.isfinite(epoch) and epoch > 0
         and plan["shared_start_epoch"] == epoch+FIRST_ACTIVE[0]
         and plan["deadline_epoch"] == epoch+WORKER_SECONDS
         and type(cohort) is int and 0 <= cohort < len(FIRST_ACTIVE), "invalid fixed cohort epoch")
    windows = [FIRST_ACTIVE[cohort]+offset for offset in OFFSETS] + [COMMON_OFFSET]
    return tuple((mode, round((epoch+offset)*NANO), round((epoch+offset+ACTIVE_SECONDS)*NANO))
                 for mode, offset in zip(ORDER, windows, strict=True))


def counters(mode, groups):
    need(mode in DEPTH and type(groups) is int and groups >= 0, "invalid graph replay count")
    gemms = groups * DEPTH[mode]
    return dict(groups=groups, gemm_calls=gemms, useful_rows=gemms*M)


def validate_arm(row, index, window):
    mode, start, end = window
    need((row["index"], row["mode"], row["scheduled_start_epoch_ns"], row["scheduled_end_epoch_ns"]) ==
         (index, mode, start, end) and row["captured_gemms_per_replay"] == DEPTH[mode]
         and all(row[key] == value for key, value in counters(mode, row["groups"]).items())
         and row["h2d_payload_bytes"] == row["d2h_payload_bytes"] == 0,
         "fixed arm identity or graph work counters differ")
    state = row["state"]
    need(state in {"COMPLETE", "COOPERATIVE_STOPPED"} and row["passed"] is (state == "COMPLETE"),
         "invalid sustained arm termination")
    quarter_groups = row["complete_groups_by_quarter"]
    need(type(quarter_groups) is list and len(quarter_groups) == 4
         and all(type(value) is int and value >= 0 for value in quarter_groups)
         and sum(quarter_groups) == row["groups"], "quarter replay denominator changed")
    actual, last, finished = (row[key] for key in ("actual_start_epoch_ns", "actual_end_epoch_ns",
                                                 "collection_end_epoch_ns"))
    need(all(type(value) is int for value in (actual, last, finished)) and start <= actual <= start+500_000_000
         and actual <= last <= finished <= end+500_000_000 and row["wall_ns"] == finished-actual
         and row["wall_ns"] > 0, "actual graph-depth window differs")
    if row["groups"]:
        need(row["full_BF16_equality"] is row["CPU_first16"] is True
             and math.isfinite(row["compute_stream_ms"]) and row["compute_stream_ms"] > 0,
             "measured replays lack BF16 proof")
    if state == "COMPLETE":
        need(row["groups"] > 0 and finished >= end and last >= end-150_000_000, "incomplete fixed arm")
    else:
        need(finished < end, "partial arm extends beyond its cooperative stop window")
    return state
