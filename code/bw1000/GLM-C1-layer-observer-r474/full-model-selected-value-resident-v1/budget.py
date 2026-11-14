"""One epoch for the resident GLM benchmark and its terminal collector."""
import math
import time

LOAD_SECONDS = 10800
WORK_SECONDS = 25500
INSIDE_SECONDS = 25800
GPU_SECONDS = 26100
WHOLE_SECONDS = 27300

assert LOAD_SECONDS < WORK_SECONDS < INSIDE_SECONDS
assert INSIDE_SECONDS + 15 < GPU_SECONDS and GPU_SECONDS % 60 == 0
assert GPU_SECONDS + 600 < WHOLE_SECONDS


def remaining(epoch, seconds, *, now=None):
    now = time.time() if now is None else now
    if not math.isfinite(epoch) or not math.isfinite(now) or not 0 < epoch <= now:
        raise ValueError("Invalid launch clock")
    return epoch + seconds - now
