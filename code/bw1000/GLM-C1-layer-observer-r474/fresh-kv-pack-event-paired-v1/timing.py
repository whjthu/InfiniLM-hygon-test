"""Fixed same-device HIP event comparison for the qualified fresh K/V exports."""

from contextlib import ExitStack
import ctypes as C
from hashlib import sha256
import math
import struct
from time import monotonic_ns


SHAPES = (1, 512)
SEEDS = {1: 1701, 512: 51201}
TILES = (64, 128)
ORDERS = ((64, 128), (128, 64), (64, 128))
WARM = 32
TIMED = {1: 1024, 512: 128}
FLOOR = 8 * 1024**3
PLAN = {"arms": [[tokens, pair, tile] for pair, order in enumerate(ORDERS) for tokens in SHAPES for tile in order]}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sizes(tokens):
    need(tokens in SHAPES, "unsupported event comparison shape")
    return tuple(tokens * length for length in (2048, 2048, 3584, 128))


def fixture(tokens):
    seed = SEEDS[tokens]

    def encode(count, salt):
        return struct.pack(f"<{count}H", *((0x3D00 + ((index * 193 + salt) % 1531))
                                          | (((index + salt) % 2) << 15) for index in range(count)))

    return encode(tokens * 4 * 448, seed), encode(tokens * 64, seed + 223)


def reference(tokens, projected, rotary):
    need(tuple(map(len, (projected, rotary))) == sizes(tokens)[2:], "wrong BF16 fixture bytes")
    keys, values = [], []
    for token in range(tokens):
        rope = rotary[token * 128:(token + 1) * 128]
        for head in range(4):
            start = (token * 4 + head) * 896
            keys.append(projected[start:start + 384] + rope)
            values.append(projected[start + 384:start + 896])
    return b"".join(keys), b"".join(values)


def hashes(buffers):
    return [sha256(value).hexdigest() for value in buffers]


def fixed_hashes(tokens):
    inputs = fixture(tokens)
    return hashes(inputs), hashes(reference(tokens, *inputs))


def audit(buffers, expected, tick):
    key, value, projected, rotary = buffers
    for item in (projected, rotary):
        item.input_audit(tick)
    actual = (key.readback(), value.readback())
    need(actual == expected, "full BF16 K/V output differs from CPU reference")
    return hashes(actual)


def run_arm(hip, buffers, anchors, invoke, identity, input_hashes, expected, tick, write, clock=monotonic_ns):
    tokens, _, _ = identity
    stream, first, last = anchors
    pointers = tuple(buffer.ptr for buffer in buffers) + (tokens, 4, stream)
    count = {"warm_submitted": 0, "timed_submitted": 0}
    stage = "setup"
    write(dict(state="ARM_START", id=identity, planned_warm=WARM, planned_timed=TIMED[tokens]))
    try:
        for output in buffers[:2]:
            output.sentinel(stream)
        stage = "warm"
        for _ in range(WARM):
            need(invoke(*pointers) == 0, "warm fresh K/V enqueue failed")
            count["warm_submitted"] += 1
        hip.call("StreamSynchronize", stream)
        warm_hashes = audit(buffers, expected, tick)
        for output in buffers[:2]:
            output.sentinel(stream)
        hip.call("StreamSynchronize", stream)
        before = hip.mem()
        tick()
        stage = "timed"
        began = clock()
        hip.call("EventRecord", first, stream)
        for index in range(TIMED[tokens]):
            if index % 32 == 0:
                tick()
            need(invoke(*pointers) == 0, "timed fresh K/V enqueue failed")
            count["timed_submitted"] += 1
        hip.call("EventRecord", last, stream)
        queued = clock()
        hip.call("StreamSynchronize", stream)
        completed = clock()
        elapsed = C.c_float()
        hip.call("EventElapsedTime", C.byref(elapsed), first, last)
        queried = clock()
        need(math.isfinite(elapsed.value) and elapsed.value > 0 and began < queued <= completed <= queried,
             "invalid same-stream or host timing interval")
        stage = "verify"
        timed_hashes = audit(buffers, expected, tick)
        after = hip.mem()
        write(dict(state="ARM_PASS", id=identity, **count, input_SHA256=input_hashes,
                   warm_output_SHA256=warm_hashes, timed_output_SHA256=timed_hashes, guards="bitwise",
                   before=before, after=after, gpu_event_ms=elapsed.value, host_submit_ns=queued-began,
                   host_wait_ns=completed-queued, host_elapsed_ns=completed-began, event_query_ns=queried-completed))
    except Exception as error:
        write(dict(state="ARM_FAIL", id=identity, stage=stage, **count, type=type(error).__name__, error=str(error)[:1024]))
        raise


def run(hip, guarded, exports, tick, write):
    inputs = {tokens: fixture(tokens) for tokens in SHAPES}
    expected = {tokens: reference(tokens, *raw) for tokens, raw in inputs.items()}
    input_hashes = {tokens: hashes(raw) for tokens, raw in inputs.items()}
    with ExitStack() as cleanup:
        buffers = {tokens: [guarded(hip, cleanup, size) for size in sizes(tokens)] for tokens in SHAPES}
        for tokens, owned in buffers.items():
            for item, payload in zip(owned[2:], inputs[tokens], strict=True):
                item.upload(payload)
        with hip.stream() as anchors:
            for identity in PLAN["arms"]:
                tokens, _, tile = identity
                tick()
                run_arm(hip, buffers[tokens], anchors, exports[tile], identity, input_hashes[tokens],
                        expected[tokens], tick, write)
            during = hip.mem()
    return during
