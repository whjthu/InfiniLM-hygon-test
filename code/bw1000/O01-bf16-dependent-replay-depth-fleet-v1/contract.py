"""Shared work, phase, and host/stream boundary contracts for graph depth."""

import math

NANO = 10**9
M, K, N, HEADS, HEAD_DIM = 186, 4096, 1024, 8, 128
LANES = 2
BYTES_PER_STAGE = M*K*2
ORDER = ("depth3", "depth9", "depth9", "depth3", "common9")
DEPTH = dict(depth3=3, depth9=9, common9=9)
CAPTURE_DEPTHS = ("depth3", "depth9")
EARLY = frozenset(("f12r3n17","f12r3n18"))
LATE = frozenset(("f12r2n10","f12r2n11","f12r2n13","f13r1n10"))
HOST_COUNTERS = ("host_submit_ns", "host_sync_wait_ns", "host_event_query_ns")
SETUP_OPEN = (0, 20, 40)
FIRST_ACTIVE = (82, 108, 130)
OFFSETS = (0, 64, 128, 192)
COMMON_OFFSET = 397
ACTIVE_SECONDS = 60
PROBES = tuple(range(2, ACTIVE_SECONDS, 5))
QUARTER_SECONDS = 15
WORKER_SECONDS, SLURM_SECONDS, CONTROLLER_SECONDS = 480, 505, 545
PHASES = ("input_copy", "v_projection", "p_times_v")


def need(condition, message):
    if not condition:
        raise ValueError(message)


def assign_cohorts(shards):
    assignment, ordinary = {}, []
    load = [0,0,0]
    for label, shard in sorted(shards.items()):
        nodes = set(shard["nodes"])
        if nodes & LATE:
            cohort = 2
        elif nodes & EARLY:
            cohort = 0
        else:
            ordinary.append((label,nodes))
            continue
        assignment[label] = cohort
        load[cohort] += len(nodes)
    for label,nodes in ordinary:
        cohort = min((1,2),key=load.__getitem__)
        assignment[label] = cohort
        load[cohort] += len(nodes)
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
    offsets = [FIRST_ACTIVE[cohort]+offset for offset in OFFSETS]+[COMMON_OFFSET]
    return tuple((mode,round((epoch+offset)*NANO),round((epoch+offset+ACTIVE_SECONDS)*NANO))
                 for mode,offset in zip(ORDER,offsets,strict=True))


def counters(mode, groups):
    need(mode in ORDER and type(groups) is int and groups >= 0, "invalid dependent submission count")
    pairs = groups*DEPTH[mode]
    chains = pairs*LANES
    return dict(groups=groups, logical_pairs=pairs, positive_pairs=(pairs+1)//2, negative_pairs=pairs//2,
                projection_calls=chains, consumer_calls=chains, useful_rows=chains*M, input_d2d_bytes=chains*BYTES_PER_STAGE)


def polarity_order(mode, opening_polarity):
    need(mode in DEPTH and type(opening_polarity) is int and opening_polarity in (0,1),
         "invalid replay phase")
    return tuple((opening_polarity+i)%2 for i in range(DEPTH[mode]))


def amortization(graph_extra_ns, eager_ns, graph_ns):
    need(type(graph_extra_ns) is int and graph_extra_ns >= 0 and
         all(type(x) in (int,float) and math.isfinite(x) and x > 0 for x in (eager_ns,graph_ns)),
         "invalid measured cost")
    saved = eager_ns-graph_ns
    if saved <= 0:
        return None
    return math.floor(graph_extra_ns/saved)+1


def validate_arm(row, index, window):
    mode, start, end = window
    need((row["index"], row["mode"], row["scheduled_start_epoch_ns"], row["scheduled_end_epoch_ns"]) ==
         (index, mode, start, end) and row["chains_per_submission"] == DEPTH[mode]*LANES
         and all(type(row[key]) is int and row[key] == value for key, value in counters(mode, row["groups"]).items())
         and row["h2d_payload_bytes"] == row["d2h_payload_bytes"] == 0,
         "fixed arm identity or dependent work counters differ")
    state = row["state"]
    need(state in {"COMPLETE", "COOPERATIVE_STOPPED"} and row["passed"] is (state == "COMPLETE"),
         "invalid sustained arm termination")
    quarters = row["complete_groups_by_quarter"]
    need(type(quarters) is list and len(quarters) == 4
         and all(type(value) is int and value >= 0 for value in quarters)
         and sum(quarters) == row["groups"], "quarter denominator differs")
    began, last, finished = (row[key] for key in ("actual_start_epoch_ns", "actual_end_epoch_ns",
                                                 "collection_end_epoch_ns"))
    need(all(type(value) is int for value in (began, last, finished))
         and start <= began <= start+500_000_000 and began <= last <= finished <= end+500_000_000
         and row["wall_ns"] == finished-began and row["wall_ns"] > 0, "actual dependent window differs")
    host = [row[key] for key in HOST_COUNTERS]
    need(all(type(ns) is int and ns >= 0 for ns in host) and sum(host)<=row["wall_ns"],
         "overlapping or invalid host submit/wait/query accounting")
    if row["groups"]:
        need(row["terminal_polarity"] == (row["logical_pairs"]-1)%2
             and row["polarity_changes_between_replays"] == row["groups"]-1
             and row["projection_full_BF16"] is row["projection_CPU_first16"] is row["consumer_full_CPU"] is True
             and math.isfinite(row["compute_stream_ms"]) and row["compute_stream_ms"] > 0,
             "measured chains lack complete projection and consumer proof")
    if state == "COMPLETE":
        need(row["groups"] > 0 and finished >= end and last >= end-150_000_000, "incomplete fixed arm")
    else:
        need(finished < end, "partial arm extends beyond cooperative stop")
    return state
