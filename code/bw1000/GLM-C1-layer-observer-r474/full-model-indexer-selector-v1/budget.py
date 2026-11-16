"""Shared deadlines for a single GLM FFN observation."""
import math
import time

LOAD_SECONDS = 10800
WORK_SECONDS = 57000
INSIDE_SECONDS = 57300
GPU_SECONDS = 57600
ARM_RESERVE_SECONDS = 600
PAIR_SECONDS = 60000

assert LOAD_SECONDS < WORK_SECONDS < INSIDE_SECONDS
assert INSIDE_SECONDS + 15 < GPU_SECONDS and GPU_SECONDS % 60 == 0
assert GPU_SECONDS + ARM_RESERVE_SECONDS < PAIR_SECONDS


def remaining(epoch, seconds, *, now=None):
    now = time.time() if now is None else now
    if not math.isfinite(epoch) or not math.isfinite(now) or not 0 < epoch <= now:
        raise ValueError("Invalid launch clock")
    return epoch + seconds - now


def require_pair_capacity(epoch, arms_left, *, now=None):
    if arms_left != 1:
        raise ValueError("Invalid remaining arm count")
    need = GPU_SECONDS + ARM_RESERVE_SECONDS
    if remaining(epoch, PAIR_SECONDS, now=now) < need:
        raise ValueError("Run budget cannot cover the model step and cleanup")
