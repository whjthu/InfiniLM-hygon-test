"""Run the fixed decode AllReduce schedule on eight device-owner threads."""
import ctypes
import hashlib
import json
import math
import threading
from concurrent.futures import ThreadPoolExecutor

DEVICES = 8
SAMPLES = 10
GEOMETRY = {
    8: ((9, 8192), (9, 12288), (10, 2048), (10, 6824), (10, 8192),
        (10, 8496), (10, 8576), (10, 16384), (10, 24576), (10, 28176)),
    16: ((9, 7168), (9, 14336)),
}


def schedule(ranks):
    return tuple((dtype, size, inplace)
                 for dtype, size in GEOMETRY[ranks] for inplace in (False, True))


class Native:
    def __init__(self, path, sha256):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != sha256:
            raise ValueError("probe DSO SHA256 mismatch")
        lib = ctypes.CDLL(str(path))
        self._lib = lib
        integer, size, ptr = ctypes.c_int, ctypes.c_size_t, ctypes.c_void_p
        string = ctypes.c_char_p
        signatures = {
            "error": ([], string),
            "id_size": ([], size),
            "id": ([ptr], integer),
            "geometry": ([integer, size, integer], integer),
            "create": ([integer, integer, integer, ptr], ptr),
            "prepare": ([ptr, size, integer, integer], integer),
            "run": ([ptr, integer], string),
            "close": ([ptr], integer),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(lib, "decode_ccl_ar_" + name)
            function.argtypes = arguments
            function.restype = result
            setattr(self, "_" + name, function)

    def _require(self, success):
        if not success:
            raise RuntimeError((self._error() or b"Native probe failed").decode(
                "utf-8", errors="replace"))

    def bootstrap(self):
        identifier = ctypes.create_string_buffer(self._id_size())
        self._require(self._id(identifier) == 0)
        return identifier.raw

    def identifier(self, raw):
        if len(raw) != self._id_size():
            raise ValueError("Native communicator identifier size mismatch")
        return ctypes.create_string_buffer(raw, len(raw))

    def validate_geometry(self, ranks, points):
        for dtype, size, _ in points:
            if self._geometry(ranks, size, dtype) != 0:
                raise ValueError("Native geometry does not admit the scheduled point")

    def create(self, ranks, rank, device, identifier):
        ctx = self._create(ranks, rank, device, identifier)
        self._require(ctx)
        return ctx

    def prepare(self, ctx, point):
        dtype, size, inplace = point
        self._require(self._prepare(ctx, size, dtype, inplace) == 0)

    def run(self, ctx):
        report = self._run(ctx, SAMPLES)
        self._require(report)
        return json.loads(report)

    def bind_deep_chain(self):
        function = self._lib.decode_ccl_ar_chain
        function.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int]
        function.restype = ctypes.c_char_p
        self._chain = function

    def run_deep_chain(self, ctx, past, fence_each):
        report = self._chain(ctx, past, fence_each, SAMPLES)
        self._require(report)
        return json.loads(report)

    def close(self, ctx):
        self._require(self._close(ctx) == 0)


def validate_report(report, ranks, rank, point):
    dtype, size, inplace = point
    expected = {
        "TP": ranks, "rank": rank, "bytes_per_rank": size,
        "dtype": {9: "BF16", 10: "F32"}[dtype], "inplace": inplace,
        "Model_weights": False, "GPU_F64": False,
        "all_iterations_full_output_checked": True,
        "validation_outside_timed_window": True,
        "out_of_place_input_checked": not inplace,
    }
    for key, value in expected.items():
        if type(report.get(key)) is not type(value) or report[key] != value:
            raise ValueError("Native report mismatch: " + key)
    samples = report["samples"]
    if len(samples) != SAMPLES + 3:
        raise ValueError("Native sample count mismatch")
    for index, sample in enumerate(samples):
        if sample["warmup"] is not (index < 3):
            raise ValueError("Native warmup classification mismatch")
        for key in ("stream_elapsed_ms", "host_enqueue_ms", "wall_ms"):
            value = sample[key]
            if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                raise ValueError("invalid Native timing: " + key)


DEEP_CHAINS = tuple((past, fence) for past in (8496, 8575)
                    for fence in (False, True, True, False))


def validate_chain_report(report, rank, past, fence_each):
    expected = {
        "TP": 8, "rank": rank, "past": past, "inplace": True,
        "allreduce_calls_per_iteration": 107,
        "API_bytes_per_iteration_per_rank": 43 * (8192 + 16384) + 21 * (4 * ((past + 1) // 4)),
        "timed_stream_fences": 107 if fence_each else 1,
        "every_slot_output_checked": True,
        "seed_and_validation_outside_timing": True,
    }
    for key, value in expected.items():
        if type(report.get(key)) is not type(value) or report[key] != value:
            raise ValueError("Deep chain report mismatch: " + key)
    if len(report["samples"]) != SAMPLES + 3:
        raise ValueError("Deep chain sample count differs")
    for index, sample in enumerate(report["samples"]):
        if sample["warmup"] is not (index < 3):
            raise ValueError("Deep chain warmup classification differs")
        for key in ("stream_elapsed_ms", "enqueue_phase_wall_ms", "wall_ms"):
            value = sample[key]
            if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                raise ValueError("invalid Deep chain timing: " + key)


def run_node(native, ranks, node, raw_identifier, bind_cpu, collect, timeout, *, deep_chain=False):
    """Collect is a bounded all-node exchange supplied by the outer controller.

    The caller owns the hard job deadline and process cleanup. A blocked foreign
    call cannot be interrupted by a Python barrier.
    """
    points = DEEP_CHAINS if deep_chain else schedule(ranks)
    nodes = ranks // DEVICES
    if node not in range(nodes) or not 0 < timeout <= 240:
        raise ValueError("invalid node or barrier timeout")
    if deep_chain:
        if ranks != 8:
            raise ValueError("Deep chain requires one TP8 node")
        native.bind_deep_chain()
    else:
        native.validate_geometry(ranks, points)
    identifier = native.identifier(raw_identifier)
    phases = iter(["owners-bound"] + [
        f"point-{index}-{'-'.join(str(int(field)) for field in point)}-{phase}"
        for index, point in enumerate(points)
        for phase in ("prepared", "validated")
    ])

    def align():
        label = next(phases)
        if collect(label, node) != list(range(nodes)):
            raise ValueError("node collective schedule mismatch")

    barrier = threading.Barrier(DEVICES, action=align, timeout=timeout)

    def owner(device):
        rank = node * DEVICES + device
        result = {"rank": rank, "device": device, "points": [],
                  "error": None, "close_error": None, "close_succeeded": None}
        ctx = None
        try:
            bind_cpu(device)
            barrier.wait()
            ctx = native.create(ranks, rank, device, identifier)
            for point in points:
                if not deep_chain:
                    native.prepare(ctx, point)
                barrier.wait()
                if deep_chain:
                    report = native.run_deep_chain(ctx, *point)
                    validate_chain_report(report, rank, *point)
                else:
                    report = native.run(ctx)
                    validate_report(report, ranks, rank, point)
                result["points"].append(report)
                barrier.wait()
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
            barrier.abort()
        finally:
            if ctx is not None:
                try:
                    native.close(ctx)
                    result["close_succeeded"] = True
                except Exception as exc:
                    result["close_error"] = f"{type(exc).__name__}: {exc}"
                    result["close_succeeded"] = False
        return result

    with ThreadPoolExecutor(max_workers=DEVICES) as pool:
        owners = list(pool.map(owner, range(DEVICES)))
    passed = all(owner["error"] is None and owner["close_succeeded"]
                 and len(owner["points"]) == len(points) for owner in owners)
    return {"TP": ranks, "node": node, "passed": passed, "owners": owners}
