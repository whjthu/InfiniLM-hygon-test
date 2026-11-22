"""Equal-work two-replay units with guarded independent device-to-host transfer."""

import math

NANO = 10**9
M, K, N, HEADS, HEAD_DIM = 186, 4096, 1024, 8, 128
LANES = 2
BYTES_PER_STAGE = M*K*2
ORDER = ('one_copy', 'two_chunks', 'two_chunks', 'one_copy', 'common_one_copy')
GRAPH_DEPTH = 9
REPLAYS_PER_UNIT = 2
TRANSFER_BYTES = LANES*M*N*2
FENCES_PER_UNIT = 2
CAPTURE_DEPTHS = ("depth9",)
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


def copy_calls_per_unit(mode):
    need(mode in ORDER, "invalid same-stream copy mode")
    return 2 if mode == "two_chunks" else 1


def counters(mode, groups):
    need(mode in ORDER and type(groups) is int and groups >= 0, "invalid dependent submission count")
    replays = groups*REPLAYS_PER_UNIT
    pairs = replays*GRAPH_DEPTH
    chains = pairs*LANES
    return dict(groups=groups, graph_replays=replays, stream_fences=groups*FENCES_PER_UNIT,
                transfer_calls=groups*copy_calls_per_unit(mode),d2h_payload_bytes=groups*TRANSFER_BYTES,
                logical_pairs=pairs, positive_pairs=pairs//2, negative_pairs=pairs//2,
                projection_calls=chains, consumer_calls=chains, useful_rows=chains*M, input_d2d_bytes=chains*BYTES_PER_STAGE)


def polarity_order(mode, opening_polarity):
    need(mode in CAPTURE_DEPTHS and type(opening_polarity) is int and opening_polarity in (0,1),
         "invalid replay phase")
    return tuple((opening_polarity+i)%2 for i in range(GRAPH_DEPTH))


def validate_arm(row, index, window):
    mode, start, end = window
    need((row["index"], row["mode"], row["scheduled_start_epoch_ns"], row["scheduled_end_epoch_ns"]) ==
         (index, mode, start, end) and row["chains_per_submission"] == REPLAYS_PER_UNIT*GRAPH_DEPTH*LANES
         and all(type(row[key]) is int and row[key] == value for key, value in counters(mode, row["groups"]).items())
         and row["h2d_payload_bytes"] == 0,
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
             and row["polarity_changes_between_replays"] == row["graph_replays"]-1
             and row["projection_full_BF16"] is row["projection_CPU_first16"] is row["consumer_full_CPU"]
                 is row["transfer_payload_guards_verified"] is True
             and all(math.isfinite(row[key]) and row[key]>0 for key in ("compute_stream_ms","copy_stream_ms")),
             "measured chains lack complete projection and consumer proof")
    if state == "COMPLETE":
        need(row["groups"] > 0 and finished >= end and last >= end-150_000_000, "incomplete fixed arm")
    else:
        need(finished < end, "partial arm extends beyond cooperative stop")
    return state
