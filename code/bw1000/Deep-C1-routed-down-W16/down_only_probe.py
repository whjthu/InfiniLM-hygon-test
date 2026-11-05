"""Isolate W32/W16 down on the sealed routed shadow quantization inputs."""

import ctypes as C
import json
import math
from statistics import mean
from time import perf_counter

import numpy as np

from component_probe import ARMS, FIXTURE_SHA, PASTS, digest

SHAPE = (6, 4096)
P = C.c_void_p
ARGS = (P,) * 6 + (C.c_size_t, P)


def read_inputs(directory):
    raw = (directory / "FIXTURE_MANIFEST.json").read_bytes()
    if digest(raw) != FIXTURE_SHA:
        raise ValueError("unexpected routed down fixture manifest")
    inputs = {}
    for row in json.loads(raw)["rows"]:
        key = row["rank"], row["past"]
        if key in inputs or row["layer"] != 3:
            raise ValueError("duplicate or wrong-layer routed down fixture")
        arrays = []
        for name, dtype, shape in (
            ("reference_I8", "<i1", (6, 256)),
            ("reference_scale", "<f4", (6, 1)),
            ("route_ids", "<i8", (1, 6)),
        ):
            field = row["fields"][name]
            data = (directory / field["file"]).read_bytes()
            if len(data) != field["bytes"] or digest(data) != field["SHA256"]:
                raise ValueError(f"routed down fixture changed: {name}")
            arrays.append(np.frombuffer(data, dtype=dtype).reshape(shape).copy())
        x, xs, ids = arrays
        if np.any(x == -128) or not np.all(np.isfinite(xs) & (xs > 0)):
            raise ValueError("invalid routed quantization values")
        if not np.all((ids >= 0) & (ids < 256)):
            raise ValueError("route outside checkpoint bank")
        inputs[key] = x, xs, ids
    if set(inputs) != {(rank, past) for rank in range(8) for past in PASTS}:
        raise ValueError("incomplete routed down fixtures")
    return inputs


def cpu_reference(inputs, weights, weight_scales):
    """Apply the original INT32 dot, ordered FP32 scaling and BF16 RNE."""
    x, xs, ids = inputs
    expert = ids.reshape(6)
    dots = np.einsum("rk,rnk->rn", x, weights[expert], dtype=np.int32)
    scale = np.multiply(xs, weight_scales[expert, :, 0], dtype=np.float32)
    values = np.multiply(scale, dots.astype(np.float32), dtype=np.float32)
    if not np.isfinite(values).all() or not np.any(values):
        raise ValueError("degenerate routed CPU reference")
    words = values.view(np.uint32)
    return ((words + 0x7FFF + ((words >> 16) & 1)) >> 16).astype(np.uint16)


def verify(output, reference):
    if (output.shape != SHAPE or output.dtype != np.uint16
            or np.any((output & 0x7F80) == 0x7F80)):
        raise ValueError("invalid or partially written down output")
    actual = (output.astype(np.uint32) << 16).view(np.float32)
    expected = (reference.astype(np.uint32) << 16).view(np.float32)
    if not np.allclose(actual, expected, rtol=.008, atol=1e-7):
        raise ValueError("down output differs from independent checkpoint reference")
    return dict(cpu_exact=bool(np.array_equal(output, reference)),
                max_abs_vs_cpu=float(np.max(np.abs(actual - expected))),
                output_SHA256=digest(output.tobytes()))


def compare(invoke, references):
    rows = []
    for trial, mode in enumerate(ARMS):
        for index, past in enumerate(PASTS):
            output, stream_ms, wall_ms = invoke(index, mode)
            if not all(math.isfinite(v) and v > 0 for v in (stream_ms, wall_ms)):
                raise ValueError("invalid isolated down timing")
            rows.append(dict(trial=trial, past=past, mode=mode, stream_ms=stream_ms,
                             launch_and_wait_wall_ms=wall_ms, **verify(output, references[index])))
    return rows


def summarize(owners):
    if set(owners) != set(range(8)):
        raise ValueError("missing isolated down owner")
    result = {}
    for past in PASTS:
        paired = [[x for x in owners[r] if x["past"] == past] for r in range(8)]
        if any([(x["trial"], x["mode"]) for x in rows] != list(enumerate(ARMS)) for rows in paired):
            raise ValueError("unpaired isolated down results")
        result[str(past)] = []
        for start in (2, 6):
            stats = {}
            for field in ("stream_ms", "launch_and_wait_wall_ms"):
                values = [max(rows[i][field] for rows in paired) for i in range(start, start + 4)]
                a, b = (mean(v for i, v in enumerate(values) if ARMS[start+i] == arm) for arm in (0, 1))
                stats[field] = dict(W32=a, W16=b, ratio=b/a)
            result[str(past)].append(stats)
    return result


def bind(lib):
    original = lib.deep_routed_dot_r20
    original.argtypes = (C.c_int, C.c_int, *ARGS)
    original.restype = C.c_int
    candidate = lib.deep_routed_down_w16_c1
    candidate.argtypes, candidate.restype = ARGS, C.c_int

    def launch(mode, args):
        rc = original(0, 32, *args) if mode == 0 else candidate(*args)
        if rc:
            raise RuntimeError(f"isolated down launch failed: {rc}")

    return launch


def run_owner(torch, rank, rank_inputs, references, banks, launch, align, floor_check):
    device = f"cuda:{rank}"
    stream = torch.cuda.current_stream(rank)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    output = torch.empty(SHAPE, dtype=torch.int16, device=device)
    cases = [tuple(torch.from_numpy(a).to(device) for a in row) for row in rank_inputs]
    stream.synchronize()

    def invoke(index, mode):
        output.fill_(0x7FC0)
        stream.synchronize()
        align()
        x, xs, ids = cases[index]
        args = (*(x.data_ptr() for x in (output, x, xs, ids, banks[2], banks[3])),
                6, stream.cuda_stream)
        t = perf_counter()
        start_event.record(stream)
        launch(mode, args)
        end_event.record(stream)
        end_event.synchronize()
        wall = (perf_counter() - t) * 1000
        floor_check()
        return output.cpu().numpy().view("uint16"), start_event.elapsed_time(end_event), wall

    try:
        return compare(invoke, references)
    finally:
        stream.synchronize()
