"""Identify positively matched gfx936 contraction events in a bound Z-Image trace."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path

TRACE_SHA256 = "81263a4f44c3562e18df359dc93e1e35a32f1cf79b8d92239bb6dc3004dfa469"
TRACE_BYTES = 56_518_532
CPU = "DeviceType.CPU"
DEVICE = "DeviceType.CUDA"
PHASES = ("native.forward", "native.backward", "native.clip_grad", "native.optimizer", "native.synchronize")
MATRIX_OPERATIONS = frozenset(("aten::mm", "aten::matmul", "aten::addmm", "aten::bmm"))


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def contraction(name):
    return name.startswith("Cijk_") and "_ISA936_" in name


def interval(row):
    start, end = row["start_us"], row["end_us"]
    need(all(type(v) in (float, int) and math.isfinite(v) for v in (start, end))
         and 0 <= start <= end, "invalid profiler interval")
    return start, end


def measure_union(windows):
    total, right = 0.0, 0.0
    for start, end in sorted(windows):
        total += max(0, end - max(start, right))
        right = max(end, right)
    return total


def positive_matrix_shapes(row):
    operands = row["input_shapes"][:2]
    if len(operands) != 2 or not all(len(shape) == 2 and
            all(type(n) is int and n > 0 for n in shape) for shape in operands):
        return None
    (m, k), (kb, n) = operands
    return (m, n, k) if k == kb else None


def inside(phase, bounds):
    return phase[0] <= bounds[0] <= bounds[1] <= phase[1]


def read(path):
    need(not path.is_symlink() and path.stat().st_size == TRACE_BYTES, "unexpected profiler file size")
    digest = sha256()
    phases, matrix_events, native_device = {}, [], []
    types, annotation_names = Counter(), Counter()
    count = 0
    with path.open("rb") as stream:
        for raw in stream:
            count += 1
            need(len(raw) <= 128 * 1024 and count <= 200_000, "profiler line or count exceeds bound")
            digest.update(raw)
            event = json.loads(raw)
            name, kind = event["name"], event["device_type"]
            types[kind] += 1
            if kind == CPU and name in PHASES:
                need(name not in phases, "duplicate native phase")
                phases[name] = interval(event)
            if kind == DEVICE:
                if contraction(name):
                    native_device.append((interval(event), event["device_index"], name))
                elif name.startswith(("native.", "dispatch.", "Optimizer.step#")):
                    annotation_names[name] += 1
            if kind == CPU and name in MATRIX_OPERATIONS and event["kernel_links"]:
                links = [link for link in event["kernel_links"] if contraction(link["name"])]
                if links:
                    need(all(math.isfinite(link["duration_us"]) and link["duration_us"] >= 0
                             for link in links), "invalid linked device duration")
                    matrix_events.append((interval(event), name, positive_matrix_shapes(event),
                                          sum(link["duration_us"] for link in links), len(links)))
    need(digest.hexdigest() == TRACE_SHA256 and count == 139746 and types[DEVICE] == 22260,
         "original trace identity or denominator differs")
    need(set(phases) == set(PHASES) and all(phases[left][1] <= phases[right][0]
         for left, right in zip(PHASES, PHASES[1:])), "native phases are incomplete or overlap")
    need(len({index for _, index, _ in native_device}) == 1, "expected one contraction device")

    device_totals, matrix_by_phase = {}, {}
    for phase, phase_bounds in phases.items():
        tagged = [(bounds, name) for bounds, _, name in native_device if inside(phase_bounds, bounds)]
        device_totals[phase] = dict(
            phase_duration_us=phase_bounds[1] - phase_bounds[0],
            identified_gfx936_kernels=len(tagged),
            identified_duration_sum_us=sum(b - a for (a, b), _ in tagged),
            identified_interval_union_us=measure_union([bounds for bounds, _ in tagged]),
            distinct_native_symbols=len({name for _, name in tagged}),
        )
        grouped, unshaped = defaultdict(lambda: [0, 0, 0.0]), 0
        for bounds, name, shape, time_us, linked in matrix_events:
            if not inside(phase_bounds, bounds):
                continue
            if name != "aten::mm" or shape is None:
                unshaped += 1
                continue
            entry = grouped[shape]
            entry[0] += 1
            entry[1] += linked
            entry[2] += time_us
        ranked = sorted(grouped.items(), key=lambda row: (-row[1][2], row[0]))
        matrix_by_phase[phase] = dict(
            matched_mm_cpu_events=sum(row[0] for row in grouped.values()),
            other_matrix_linked_cpu_events=unshaped,
            distinct_mm_shapes=len(grouped),
            top=[dict(M=shape[0], N=shape[1], K=shape[2], mm_calls=counts[0],
                      linked_device_rows=counts[1], linked_sum_us=counts[2])
                 for shape, counts in ranked[:8]],
        )
    return dict(trace_SHA256=TRACE_SHA256, raw_framework_event_count=count, device_type_events=types[DEVICE],
                explicit_device_type_annotation_counts=dict(sorted(annotation_names.items())),
                identified_gfx936_device_kernel_events=len(native_device),
                identified_gfx936_events_outside_complete_phase_windows=len(native_device) -
                    sum(phase["identified_gfx936_kernels"] for phase in device_totals.values()),
                device_temporal_windows=device_totals,
                CPU_matmul_links_by_phase=matrix_by_phase,
                limitations=["Non-contraction events are outside the identified kernel set.",
                             "Temporal overlap does not establish device ownership by host phase.",
                             "Host links can include nested events. Matrix ranks use aten::mm leaf names only.",
                             "Shapes carry no tensor stride or dtype metadata; precision is separately bound by the frozen recipe.",
                             "The trace includes profiler overhead. No wall-time saving or full-card busy is inferred."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    print(json.dumps(read(args.trace), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
