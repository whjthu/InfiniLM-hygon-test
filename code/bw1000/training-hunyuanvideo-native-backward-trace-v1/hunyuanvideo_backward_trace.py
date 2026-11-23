"""Capture one original HunyuanVideo backward and summarize actual device activities."""

from collections import Counter, defaultdict
from contextlib import contextmanager
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import stat


STEP = 6
MARKER = "hunyuanvideo/t09/backward"
ENV = "HUNYUAN_T09_BACKWARD_TRACE"
MAX_TRACE_BYTES = 128 * 1024**2
DEVICE_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}


def duration_us(intervals):
    total = 0.0
    right = -math.inf
    for start, end in sorted(intervals):
        total += max(0, end - max(start, right))
        right = max(right, end)
    return total


def interval(event):
    start, elapsed = event["ts"], event["dur"]
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
           for value in (start, elapsed)) or not math.isfinite(start + elapsed):
        raise ValueError("Invalid timestamp or duration in the profiler trace")
    return start, start + elapsed


def external_id(event):
    value = event.get("args", {}).get("External id")
    return value if type(value) is int and value > 0 else None


def reduce_trace(trace):
    events = trace["traceEvents"]
    if not isinstance(events, list) or not 0 < len(events) <= 500_000:
        raise ValueError("Invalid number of profiler trace records")
    markers = [event for event in events if event.get("ph") == "X" and event.get("cat") == "user_annotation"
               and event.get("name") == MARKER]
    if len(markers) != 1:
        raise ValueError("Expected exactly one original backward marker")
    begin, end = interval(markers[0])
    if end <= begin:
        raise ValueError("Backward marker has no elapsed time")

    cpu_ops = defaultdict(list)
    for event in events:
        if event.get("ph") == "X" and event.get("cat") == "cpu_op":
            key = external_id(event)
            if key is not None:
                cpu_ops[key].append(event["name"])

    by_kernel = defaultdict(list)
    by_cpu_op = defaultdict(list)
    activity = defaultdict(list)
    kernel_streams = set()
    devices = set()
    kernel_intervals = []
    attributed_kernel_us = 0
    correlation_counts = Counter()
    for event in events:
        category = event.get("cat")
        if event.get("ph") != "X" or category not in DEVICE_CATEGORIES:
            continue
        start, end = interval(event)
        elapsed = end - start
        activity[category].append((start, end))
        if category != "kernel":
            continue
        args = event.get("args", {})
        device = args.get("device")
        stream = args.get("stream")
        if type(device) is not int or device < 0 or type(stream) is not int or stream < 0:
            raise ValueError("Kernel is missing its device or stream identity")
        devices.add(device)
        kernel_streams.add(stream)
        kernel_intervals.append((start, end))
        by_kernel[event["name"]].append(elapsed)
        linked = cpu_ops.get(external_id(event), ())
        if len(linked) == 1:
            by_cpu_op[linked[0]].append(elapsed)
            attributed_kernel_us += elapsed
            correlation_counts["unique_cpu_op"] += 1
        else:
            correlation_counts["ambiguous" if linked else "unattributed"] += 1

    if len(devices) != 1 or duration_us(kernel_intervals) <= 0:
        raise ValueError("The trace lacks real kernels from exactly one device")

    def ranking(groups):
        rows = [{
            "name": name[:360], "name_SHA256": sha256(name.encode()).hexdigest(),
            "calls": len(samples), "summed_device_us": math.fsum(samples),
        } for name, samples in groups.items()]
        return sorted(rows, key=lambda row: (-row["summed_device_us"], row["name_SHA256"]))[:20]

    kernel_sum = math.fsum(end - start for start, end in kernel_intervals)
    all_intervals = [item for spans in activity.values() for item in spans]
    return {
        "trace_events": len(events),
        "original_backward_CPU_marker_us": markers[0]["dur"],
        "single_device": next(iter(devices)), "kernel_stream_count": len(kernel_streams),
        "device_activity_calls": {name: len(spans) for name, spans in sorted(activity.items())},
        "kernel_summed_device_us": kernel_sum, "kernel_union_device_us": duration_us(kernel_intervals),
        "all_recorded_activity_union_us": duration_us(all_intervals),
        "recorded_device_span_us": max(last for _, last in all_intervals) - min(first for first, _ in all_intervals),
        "kernel_correlation_counts": dict(correlation_counts),
        "attributed_kernel_summed_device_us": attributed_kernel_us,
        "top_kernels": ranking(by_kernel), "top_cpu_ops_by_linked_kernel_us": ranking(by_cpu_op),
        "formal_performance_eligible": False,
    }


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


class BackwardTrace:
    def __init__(self, profiler, output, *, training_step):
        activities = [profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA]
        if not set(activities) <= profiler.supported_activities():
            raise ValueError("The installed profiler does not expose CPU and device tracing")
        self.profiler = profiler
        self.activities = activities
        self.training_step = training_step
        self.directory = Path(output) / "t09-backward-trace"
        self.directory.mkdir(mode=0o700)
        write_json(self.directory / "intent.json", {
            "planned_global_step": training_step, "marker": MARKER, "formal_performance_eligible": False,
        })
        self.started = False

    @contextmanager
    def capture(self):
        if self.started:
            raise ValueError("The original backward has already entered capture")
        self.started = True
        profiler = self.profiler
        with profiler.profile(activities=self.activities, record_shapes=False, profile_memory=False,
                              with_stack=False, with_flops=False) as capture:
            with profiler.record_function(MARKER):
                yield
        raw = self.directory / "backward.trace.json"
        capture.export_chrome_trace(str(raw))
        with raw.open("rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= MAX_TRACE_BYTES:
                raise ValueError("Trace exceeds the private diagnostics budget")
            data = stream.read(MAX_TRACE_BYTES + 1)
        if len(data) != info.st_size:
            raise ValueError("Profiler trace changed during reading")
        summary = reduce_trace(json.loads(data))
        write_json(self.directory / "backward.summary.json", {
            **summary, "actual_global_step": self.training_step,
            "private_trace_bytes": len(data), "private_trace_SHA256": sha256(data).hexdigest(),
        })


def trace_from_environment(torch, output, *, warmup, measured, restored_step, recovery):
    requested = os.environ.get(ENV)
    if requested is None:
        return None
    if (requested != str(STEP) or recovery or (warmup, measured, restored_step) != (5, 20, 0)
            or not torch.version.hip or torch.cuda.device_count() != 1):
        raise ValueError("Backward profiling requires the original fresh single-DCU 5+20 recipe")
    return BackwardTrace(torch.profiler, output, training_step=STEP)
