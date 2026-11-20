"""Shared deadlines for a fresh GLM screen pair."""
import math
import time

LOAD_SECONDS = 10800
WORK_SECONDS = 13200
INSIDE_SECONDS = 13500
GPU_SECONDS = 13800
ARM_RESERVE_SECONDS = 600
PAIR_SECONDS = 29400

assert LOAD_SECONDS < WORK_SECONDS < INSIDE_SECONDS
assert INSIDE_SECONDS + 15 < GPU_SECONDS and GPU_SECONDS % 60 == 0
assert 2 * (GPU_SECONDS + ARM_RESERVE_SECONDS) < PAIR_SECONDS


def remaining(epoch, seconds, *, now=None):
    now = time.time() if now is None else now
    if not math.isfinite(epoch) or not math.isfinite(now) or not 0 < epoch <= now:
        raise ValueError("Invalid launch clock")
    return epoch + seconds - now


def require_pair_capacity(epoch, arms_left, *, now=None):
    if arms_left not in (1, 2):
        raise ValueError("Invalid remaining arm count")
    need = arms_left * (GPU_SECONDS + ARM_RESERVE_SECONDS)
    if remaining(epoch, PAIR_SECONDS, now=now) < need:
        raise ValueError("Whole pair budget cannot cover remaining arms and cleanup")
