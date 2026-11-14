"""Replay the GLM TP16 F32 FFN copy and SUM on an admitted owner stream."""
from array import array
import ctypes as C
import hashlib
import math
import sys
import time

RANKS = 16
WIDTH = 6144
ROWS = (1, 512)
GUARD = 256
WARMUP = 5
MEASURED = 20


def payloads(rank, rows):
    if type(rank) is not int or not 0 <= rank < RANKS or type(rows) is not int or rows not in ROWS:
        raise ValueError("invalid GLM FFN geometry")
    if sys.byteorder != "little" or array("f").itemsize != 4:
        raise ValueError("expected little-endian F32")
    count = rows * WIDTH

    def pattern(multiplier, offset):
        block = array("f", (multiplier * (index - 128) + offset for index in range(257)))
        tiles, remainder = divmod(count, len(block))
        return (block * tiles + block[:remainder]).tobytes()

    return pattern(1, rank - 8), pattern(RANKS, -8)


def replay(api, comm, stream, rank, rows, f32_dtype, sum_op, deadline):
    """Use checked CUDA/InfiniCCL calls supplied by the pinned owner.

    The owner supplies enum values derived from the provider header, an admitted
    communicator and its stream. The timer covers a device-to-device copy,
    in-place SUM of the separate destination, and an isolated stream fence. The model also casts its output before fencing.
    """
    source, expected = payloads(rank, rows)
    payload_bytes = len(source)
    left = bytes([0xA5]) * GUARD
    right = bytes([0x5A]) * GUARD
    source_flat = left + source + right
    reduced_flat = left + bytes([0xFF]) * payload_bytes + right
    expected_flat = left + expected + right
    extent = len(source_flat)
    host_source = C.create_string_buffer(source_flat, extent)
    host_poison = C.create_string_buffer(reduced_flat, extent)
    host_readback = C.create_string_buffer(extent)
    src = C.c_void_p()
    dst = C.c_void_p()
    events = []

    def copy_async(target, origin, size, kind):
        api.copy_async(target, origin, size, kind, stream)

    def sync():
        api.sync_stream(stream)

    def event_time(begin, end):
        value = C.c_float()
        api.event_elapsed(C.byref(value), begin, end)
        if not math.isfinite(value.value) or value.value < 0:
            raise ValueError("invalid collective stream event time")
        return value.value

    samples = []
    try:
        deadline()
        api.alloc(C.byref(src), extent)
        api.alloc(C.byref(dst), extent)
        if not src.value or not dst.value or src.value == dst.value:
            raise ValueError("copy and reduction require distinct storage")
        for _ in range(3):
            event = C.c_void_p()
            api.event_create(C.byref(event))
            events.append(event)
        begin, before_sum, after_sum = events
        src_payload = C.c_void_p(src.value + GUARD)
        dst_payload = C.c_void_p(dst.value + GUARD)
        for iteration in range(WARMUP + MEASURED):
            deadline()
            copy_async(src, host_source, extent, 1)
            copy_async(dst, host_poison, extent, 1)
            sync()
            api.event_record(begin, stream)
            start = time.monotonic_ns()
            copy_async(dst_payload, src_payload, payload_bytes, 3)
            api.event_record(before_sum, stream)
            api.reduce(dst_payload, dst_payload, rows * WIDTH, f32_dtype, sum_op, comm, stream)
            submitted = time.monotonic_ns()
            api.event_record(after_sum, stream)
            sync()
            finished = time.monotonic_ns()

            copy_async(host_readback, src, extent, 2)
            sync()
            if host_readback.raw != source_flat:
                raise ValueError("pre-reduction input or its guard changed")
            copy_async(host_readback, dst, extent, 2)
            sync()
            if host_readback.raw != expected_flat:
                raise ValueError("F32 SUM output or its guard differs")
            if iteration >= WARMUP:
                samples.append(dict(copy_stream_ms=event_time(begin, before_sum),
                                    sum_stream_ms=event_time(before_sum, after_sum),
                                    submit_ns=submitted - start, wall_ns=finished - start))
    finally:
        # Drain the owner stream before releasing asynchronous inputs and events.
        # If draining fails, the enclosing step must fail-stop the owner.
        if src.value or dst.value:
            sync()
        for event in reversed(events):
            api.event_destroy(event)
        if dst.value:
            api.free(dst)
        if src.value:
            api.free(src)
    return dict(shape=[rows, WIDTH], dtype="F32", ranks=RANKS, rank=rank,
                bytes_per_rank=payload_bytes, guard_bytes_each_side=GUARD,
                checked_iterations=WARMUP + MEASURED,
                output_SHA256=hashlib.sha256(expected).hexdigest(), samples=samples)
