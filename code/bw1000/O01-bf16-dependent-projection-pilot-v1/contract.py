"""CPU-verifiable boundaries for the two dependent chains in each submission."""

import math

NANO = 10**9
M, K, N, HEADS, HEAD_DIM = 186, 4096, 1024, 8, 128
LANES = 2
BYTES_PER_STAGE = M*K*2
ORDER = ("eager", "graph", "graph", "eager")
SETUP_OPEN = (0,)
FIRST_ACTIVE = (82,)
OFFSETS = (0, 64, 128, 192)
ACTIVE_SECONDS = 60
PROBES = tuple(range(2, ACTIVE_SECONDS, 5))
QUARTER_SECONDS = 15
WORKER_SECONDS, SLURM_SECONDS, CONTROLLER_SECONDS = 350, 365, 400
PHASES = ("input_copy", "v_projection", "p_times_v")


def need(condition, message):
    if not condition:
        raise ValueError(message)


def assign_cohorts(shards):
    need(len(shards) == 1, "pilot requires exactly one shard")
    return dict.fromkeys(shards, 0)


def cohort_for(plan, node):
    [label] = [label for label, unit in plan["shards"].items() if unit["nodes"] == [node]]
    return plan["cohort_by_shard"][label]


def grid(plan, cohort):
    epoch = plan["campaign_epoch"]
    need(type(epoch) is float and math.isfinite(epoch) and epoch > 0
         and plan["shared_start_epoch"] == epoch+FIRST_ACTIVE[0]
         and plan["deadline_epoch"] == epoch+WORKER_SECONDS
         and type(cohort) is int and cohort == 0, "invalid fixed pilot epoch")
    return tuple((mode, round((epoch+FIRST_ACTIVE[0]+offset)*NANO),
                  round((epoch+FIRST_ACTIVE[0]+offset+ACTIVE_SECONDS)*NANO))
                 for mode, offset in zip(ORDER, OFFSETS, strict=True))


def counters(mode, groups):
    need(mode in ORDER and type(groups) is int and groups >= 0, "invalid dependent submission count")
    chains = groups*LANES
    return dict(groups=groups, projection_calls=chains, consumer_calls=chains,
                useful_rows=chains*M, input_d2d_bytes=chains*BYTES_PER_STAGE)


def amortization(graph_extra_ns, eager_ns, graph_ns):
    need(all(type(x) in (int, float) and math.isfinite(x) and x > 0
             for x in (graph_extra_ns, eager_ns, graph_ns)), "invalid measured cost")
    saved = eager_ns-graph_ns
    if saved <= 0:
        return None
    return math.floor(graph_extra_ns/saved)+1


def validate_arm(row, index, window):
    mode, start, end = window
    need((row["index"], row["mode"], row["scheduled_start_epoch_ns"], row["scheduled_end_epoch_ns"]) ==
         (index, mode, start, end) and row["chains_per_submission"] == LANES
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
    if row["groups"]:
        need(row["projection_full_BF16"] is row["projection_CPU_first16"] is row["consumer_full_CPU"] is True
             and math.isfinite(row["compute_stream_ms"]) and row["compute_stream_ms"] > 0,
             "measured chains lack complete projection and consumer proof")
    if state == "COMPLETE":
        need(row["groups"] > 0 and finished >= end and last >= end-150_000_000, "incomplete fixed arm")
    else:
        need(finished < end, "partial arm extends beyond cooperative stop")
    return state
