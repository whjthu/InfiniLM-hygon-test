"""Fixed T01 projection geometries, windows and completion counters."""

import math


NANO = 10**9
USEFUL_M, PADDED_M, K, N, DEPTH = 186, 192, 4096, 1024, 64
ORDER = ("native186", "padded192", "padded192", "native186")
ROWS = {"native186": USEFUL_M, "padded192": PADDED_M}
OFFSETS = (60, 124, 188, 252)
ACTIVE_SECONDS = 60
PROBES = tuple(range(2, ACTIVE_SECONDS, 5))
QUARTER_SECONDS = 15
WORKER_SECONDS, SLURM_SECONDS, CONTROLLER_SECONDS = 330, 345, 385


def need(condition, message):
    if not condition:
        raise ValueError(message)


def grid(plan):
    epoch = plan["campaign_epoch"]
    need(type(epoch) is float and math.isfinite(epoch) and epoch > 0
         and plan["shared_start_epoch"] == epoch+OFFSETS[0]
         and plan["deadline_epoch"] == epoch+WORKER_SECONDS, "invalid fixed sustained Graph64 epoch")
    return tuple((mode, round((epoch+offset)*NANO), round((epoch+offset+ACTIVE_SECONDS)*NANO))
                 for mode, offset in zip(ORDER, OFFSETS, strict=True))


def counters(mode, groups):
    need(mode in ROWS and type(groups) is int and groups >= 0, "invalid Graph64 burst count")
    gemms = groups * DEPTH
    return dict(groups=groups, gemm_calls=gemms, useful_rows=gemms*USEFUL_M,
                physical_rows=gemms*ROWS[mode], padded_rows=gemms*(ROWS[mode]-USEFUL_M))


def validate_arm(row, index, window):
    mode, start, end = window
    need((row["index"], row["mode"], row["scheduled_start_epoch_ns"], row["scheduled_end_epoch_ns"]) ==
         (index, mode, start, end) and row["captured_gemms_per_replay"] == DEPTH
         and all(row[key] == value for key, value in counters(mode, row["groups"]).items())
         and row["h2d_payload_bytes"] == row["d2h_payload_bytes"] == 0,
         "fixed arm identity or Graph64 work counters differ")
    state = row["state"]
    need(state in {"COMPLETE", "COOPERATIVE_STOPPED"} and row["passed"] is (state == "COMPLETE"),
         "invalid sustained arm termination")
    quarter_groups = row["complete_groups_by_quarter"]
    need(isinstance(quarter_groups, list) and len(quarter_groups) == 4
         and all(type(value) is int and value >= 0 for value in quarter_groups)
         and sum(quarter_groups) == row["groups"], "quarter burst denominator changed")
    actual = row["actual_start_epoch_ns"]
    last = row["actual_end_epoch_ns"]
    finished = row["collection_end_epoch_ns"]
    need(all(type(value) is int for value in (actual, last, finished))
         and start <= actual <= start+500_000_000 and actual <= last <= finished <= end+500_000_000
         and row["wall_ns"] == finished-actual and row["wall_ns"] > 0,
         "actual sustained arm window differs")
    if row["groups"]:
        need(row["full_BF16_equality"] is row["CPU_first16"] is row["padding_zero"] is True
             and math.isfinite(row["compute_stream_ms"]) and row["compute_stream_ms"] > 0,
             "measured bursts lack full BF16 and pad-tail proof")
    if state == "COMPLETE":
        need(row["groups"] > 0 and finished >= end and last >= end-150_000_000, "incomplete fixed arm")
    else:
        need(finished < end, "partial arm extends beyond its cooperative stop window")
    return state
