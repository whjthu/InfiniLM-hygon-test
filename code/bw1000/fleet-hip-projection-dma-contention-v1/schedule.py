"""Absolute arm grid. Wait never extends or slides a running arm."""

import math
import time


ARMS = ('compute', 'compute_dma', 'compute_dma', 'compute')
OFFSETS = (0, 12, 24, 36)
ACTIVE_SECONDS = 8
SAMPLE_OFFSETS = (1, 3, 5, 7)
NANO = 10**9
START_TOLERANCE_NS = 500_000_000
CLOCK_JUMP_NS = 100_000_000


def need(condition, name):
    if not condition:
        raise ValueError(name)


def bounds(campaign, start, deadline):
    need(all(type(x) in (float, int) and math.isfinite(x) for x in (campaign, start, deadline)),
         'absolute epoch must be finite')
    need(25 <= start-campaign <= 100 and start+OFFSETS[-1]+ACTIVE_SECONDS+8 <= deadline <= campaign+175,
         'shared epoch or fixed arm grid exceeds the one finite campaign')
    return tuple((name, round((start+offset)*NANO), round((start+offset+ACTIVE_SECONDS)*NANO))
                 for name, offset in zip(ARMS, OFFSETS, strict=True))


class Clock:
    """Translate the frozen wall grid once into this node process's monotonic clock."""
    def __init__(self, wall=time.time_ns, monotonic=time.perf_counter_ns, sleep=time.sleep):
        self.wall, self.monotonic, self.sleep = wall, monotonic, sleep
        before = monotonic()
        absolute = wall()
        after = monotonic()
        need(after - before <= 20_000_000, 'clock anchor capture too wide')
        self.anchor = absolute - (before + after)//2

    def verify(self):
        need(abs(self.wall() - self.monotonic() - self.anchor) <= CLOCK_JUMP_NS,
             'local realtime clock changed during the fixed grid')

    def toward(self, target, halted, tolerance_ns=START_TOLERANCE_NS):
        self.verify()
        target_mono = target - self.anchor
        while not halted():
            now = self.monotonic()
            remaining = target_mono-now
            if remaining <= 0:
                need(-remaining <= tolerance_ns, 'absolute window missed')
                self.verify()
                return now
            self.sleep(min(remaining/NANO, .05))
        raise InterruptedError('cooperative stop before next measurement')

    def wall_from_mono(self, value):
        return value + self.anchor


def guaranteed_overlap(node_windows, clock_error_bound_ns):
    """Return a guaranteed interval only when every expected node has an error bound."""
    need(bool(node_windows), 'empty simultaneous-window cohort')
    if set(node_windows) != set(clock_error_bound_ns):
        return None
    safe_starts, safe_ends = [], []
    for node, (start, end) in node_windows.items():
        epsilon = clock_error_bound_ns[node]
        need(type(start) is type(end) is type(epsilon) is int
             and end > start and epsilon >= 0, 'invalid measured per-node clock bound')
        safe_starts.append(start+epsilon)
        safe_ends.append(end-epsilon)
    return max(0, min(safe_ends)-max(safe_starts))
