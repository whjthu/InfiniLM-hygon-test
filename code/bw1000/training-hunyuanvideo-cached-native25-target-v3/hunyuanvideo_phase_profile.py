"""Diagnostic clock boundaries for the original T09 backward and update calls."""

import json
import math
import os
from pathlib import Path
import time


STEPS = (6, 7)
PHASES = ("backward", "gradient_clip", "muon_step")
ENV = "HUNYUAN_T09_PHASE_PROFILE"


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


class StepPhases:
    def __init__(self, cuda, directory, step):
        self.cuda = cuda
        self.output = directory / f"step-{step:02d}.json"
        self.step = step
        self.stream = cuda.current_stream()
        self.events = [cuda.Event(enable_timing=True) for _ in range(4)]
        self.stamps = []
        self.mark()

    def mark(self):
        if len(self.stamps) >= len(self.events) or self.cuda.current_stream() != self.stream:
            raise ValueError("T09 diagnostic must record exactly four marks on one submission stream")
        self.events[len(self.stamps)].record(self.stream)
        self.stamps.append(time.perf_counter_ns())

    def finish(self, mixed_step_seconds):
        # The original backend has already performed its device synchronization.
        if len(self.stamps) != len(self.events) or not self.events[-1].query():
            raise ValueError("Original device synchronization did not cover the diagnostic events")
        elapsed = [before.elapsed_time(after) / 1000 for before, after in zip(self.events, self.events[1:])]
        host = [(after - before) / 1e9 for before, after in zip(self.stamps, self.stamps[1:])]
        if (not all(math.isfinite(value) and value >= 0 for value in elapsed + host)
                or type(mixed_step_seconds) not in (int, float)
                or not math.isfinite(mixed_step_seconds) or mixed_step_seconds <= 0):
            raise ValueError("T09 diagnostic duration is invalid")
        save(self.output, {
            "actual_global_step": self.step,
            "phase_seconds": {phase: {"host_between_marks": h, "device_stream_between_marks": d}
                              for phase, h, d in zip(PHASES, host, elapsed, strict=True)},
            "original_backward_update_seconds_with_observer": mixed_step_seconds,
            "uses_existing_device_synchronization": True,
            "device_time_is_stream_interval_including_possible_idle": True,
            "formal_performance_eligible": False,
        })


class Profile:
    def __init__(self, cuda, output):
        self.cuda = cuda
        self.directory = Path(output) / "t09-phase-profile"
        self.directory.mkdir(mode=0o700)
        save(self.directory / "intent.json", {"planned_global_steps": STEPS, "phases": PHASES,
                                              "formal_performance_eligible": False})
        self.started = []

    def begin(self, step):
        if step not in STEPS:
            return None
        if len(self.started) >= len(STEPS) or step != STEPS[len(self.started)]:
            raise ValueError("T09 diagnostic step repeated or out of order")
        result = StepPhases(self.cuda, self.directory, step)
        self.started.append(step)
        return result


def profile_from_environment(torch, output, *, warmup, measured, restored_step, recovery):
    requested = os.environ.get(ENV)
    if requested is None:
        return None
    if (requested != "6,7" or recovery or (warmup, measured, restored_step) != (5, 20, 0)
            or not torch.version.hip or torch.cuda.device_count() != 1):
        raise ValueError("T09 phase profiling requires the original fresh single-DCU 5+20 recipe")
    return Profile(torch.cuda, output)
