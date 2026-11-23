"""Read metadata from an already finished native Z-Image Kineto session."""

from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re


BACKWARD = "native.backward"
EXPECTED = {(288, 3840, 3840): 240, (288, 3840, 10240): 90, (288, 10240, 3840): 90}
LIMIT = 200_000
UNIQUE_LIMIT = 128
CPU = "DeviceType.CPU"
DEVICE = "DeviceType.CUDA"
CUDA_DEVICE = re.compile(r"cuda:(0|[1-9][0-9]{0,2})")
DTYPE = re.compile(r"torch\.[a-z0-9_]{1,32}")


def need(value, reason):
    if not value:
        raise ValueError(reason)


def matrix_shape(operands):
    if len(operands) != 2:
        return None
    if any(len(shape) != 2 or any(type(dim) is not int or dim <= 0 for dim in shape) for shape in operands):
        return None
    (m, k), (bk, n) = operands
    return (m, n, k) if k == bk else None


def descriptor(value):
    sizes, strides = tuple(value.sizes), tuple(value.strides)
    dtype, device, layout = str(value.dtype), str(value.device), str(value.layout)
    need(len(sizes) == len(strides) == 2 and all(type(dim) is int and 0 < dim <= 2**31 for dim in sizes)
         and all(type(step) is int and 0 <= step <= 2**31 for step in strides)
         and DTYPE.fullmatch(dtype) and CUDA_DEVICE.fullmatch(device) and layout == "torch.strided",
         "unsupported or missing raw CUDA tensor metadata")
    return sizes, strides, dtype, device, layout


def raw_result(profiler):
    result = profiler.profiler.kineto_results
    need(result is not None, "finished Kineto result required")
    return result


def metadata_by_id(result, requested, check):
    found = defaultdict(list)
    pending = [iter(result.experimental_event_tree())]
    count = 0
    while pending:
        event = next(pending[-1], None)
        if event is None:
            pending.pop()
            continue
        count += 1
        need(count <= LIMIT and len(pending) <= 128, "Kineto event tree bound exceeded")
        if count % 1024 == 0:
            check()
        if event.name == "aten::mm" and event.correlation_id in requested:
            rows = found[event.correlation_id]
            if len(rows) < 2:
                try:
                    operands = event.extra_fields.inputs
                    need(len(operands) >= 2, "missing aten::mm operands")
                    rows.append(tuple(descriptor(v) for v in operands[:2]))
                except (AttributeError, TypeError, ValueError):
                    rows.append(None)
        pending.append(iter(event.children))
    return found, count


def verify_smoke(profiler):
    """Require the existing 64x64 CUDA profiler smoke to expose both strides."""
    result = raw_result(profiler)
    events = result.events()
    need(0 < len(events) <= 512, "smoke profiler event denominator invalid")
    hosts = {e.correlation_id() for e in events if str(e.device_type()) == CPU and e.name() == "aten::mm"
             and matrix_shape(e.shapes()[:2]) == (64, 64, 64) and e.correlation_id() > 0}
    need(hosts and any(str(e.device_type()) == DEVICE for e in events), "smoke must be an actual CUDA mm")
    metadata, _ = metadata_by_id(result, hosts, lambda: None)
    qualified = any(len(matches) == 1 and matches[0] is not None
                    and matches[0][0][1] == (64, 1) and matches[0][1][1] == (1, 64)
                    and all(row[0] == (64, 64) and row[3] == "cuda:0" for row in matches[0])
                    for matches in metadata.values())
    need(qualified, "target profiler cannot recover real contiguous and transposed operand strides")


def capture(profiler, *, check):
    """Join the preselected backward hotspot names by unique positive correlation."""
    result = raw_result(profiler)
    check()
    events = result.events()
    need(0 < len(events) <= LIMIT, "Kineto flat event denominator invalid")
    phases = [event for event in events if str(event.device_type()) == CPU and event.name() == BACKWARD]
    need(len(phases) == 1, "expected exactly one native backward range")
    start, end = phases[0].start_ns(), phases[0].end_ns()
    need(0 < start < end, "invalid native backward range")
    cijk = defaultdict(list)
    inside = []
    correlations = Counter()
    for index, event in enumerate(events):
        if index % 1024 == 0:
            check()
        kind, name = str(event.device_type()), event.name()
        if kind == DEVICE and name.startswith("Cijk_") and "_ISA936_" in name:
            owner = event.linked_correlation_id()
            if owner > 0:
                rows = cijk[owner]
                need(len(rows) < 128 and len(name) <= 512, "Cijk provider link bound exceeded")
                rows.append((event.device_index(), name))
        if kind != CPU or name != "aten::mm":
            continue
        correlation = event.correlation_id()
        if correlation > 0:
            correlations[correlation] += 1
        if start <= event.start_ns() <= event.end_ns() <= end:
            shape = matrix_shape(event.shapes()[:2])
            if shape in EXPECTED:
                inside.append((correlation, shape))
    need(len(inside) <= 4096, "selected aten::mm denominator exceeds bound")
    ids = {corr for corr, _ in inside if corr > 0 and corr in cijk and correlations[corr] == 1}
    metadata, tree_events = metadata_by_id(result, ids, check)
    totals = {shape: Counter() for shape in EXPECTED}
    signatures = Counter()
    for correlation, shape in inside:
        total = totals[shape]
        total["host_mm_calls"] += 1
        links = cijk.get(correlation)
        if correlation <= 0 or not links:
            total["missing_cijk_link"] += 1
            continue
        total["linked_host_mm_calls"] += 1
        matches = metadata.get(correlation, ())
        if correlations[correlation] != 1 or len(matches) != 1 or matches[0] is None:
            total["missing_or_ambiguous_metadata"] += 1
            continue
        lhs, rhs = matches[0]
        devices = {device for device, _ in links}
        if matrix_shape((lhs[0], rhs[0])) != shape or lhs[3] != rhs[3] or devices != {int(lhs[3][5:])}:
            total["link_shape_or_device_mismatch"] += 1
            continue
        kernels = tuple(sorted(Counter(name for _, name in links).items()))
        signature = shape, lhs, rhs, kernels
        need(signature in signatures or len(signatures) < UNIQUE_LIMIT, "operand signature bound exceeded")
        signatures[signature] += 1
        total["metadata_joined"] += 1
    check()
    rows = []
    for (shape, lhs, rhs, kernels), calls in sorted(signatures.items()):
        rows.append(dict(MNK=shape, lhs=dict(zip(("shape", "stride", "dtype", "device", "layout"), lhs)),
                         rhs=dict(zip(("shape", "stride", "dtype", "device", "layout"), rhs)),
                         linked_kernel_multiplicity=kernels, host_mm_calls=calls))
    ready = all(total["linked_host_mm_calls"] == total["metadata_joined"] == EXPECTED[shape]
                for shape, total in totals.items())
    return dict(phase=BACKWARD, prior_profile_step="1103606.1768", flat_events=len(events), tree_events=tree_events,
                totals=[dict(MNK=shape, previous_linked_calls=EXPECTED[shape], **totals[shape]) for shape in EXPECTED],
                signatures=rows, all_prior_linked_calls_covered=ready,
                records_tensor_values=False, records_addresses=False,
                measures_formal_performance=False, qualifies_replacement=False)


def write_capture(profiler, output, *, check):
    value = capture(profiler, check=check)
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    need(len(raw) <= 1024 * 1024, "small metadata output bound exceeded")
    with (Path(output) / "backward-mm-operands.json").open("xb") as stream:
        stream.write(raw)
    return {"file": "backward-mm-operands.json", "bytes": len(raw), "SHA256": sha256(raw).hexdigest(),
            "shapes_with_bound_operands": sum(row.get("metadata_joined", 0) > 0 for row in value["totals"]),
            "all_prior_linked_calls_covered": value["all_prior_linked_calls_covered"]}
