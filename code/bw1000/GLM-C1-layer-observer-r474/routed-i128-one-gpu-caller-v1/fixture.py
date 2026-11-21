"""Deterministic synthetic packed E4M3 inputs, with the full TP16 leaf layout."""
from array import array
from functools import lru_cache
import hashlib
import math
import struct
import sys


HIDDEN, LOCAL, ROUTES, EXPERTS, BLOCK = 6144, 128, 8, 256, 128
CASES = (
    ("offset0_compact", 0, (0, 1, 2, 3, 4, 5, 6, 7)),
    ("offset128_spread", 128, (3, 31, 65, 97, 127, 157, 193, 251)),
    ("offset1024_reordered", 1024, (255, 0, 128, 1, 254, 127, 222, 63)),
    ("offset1920_edges", 1920, (255, 254, 192, 129, 128, 64, 1, 0)),
)
ARMS = {"G1": "gate", "G2": "gate", "D2": "down", "D4": "down"}
PROVIDERS = ("ORIG", *ARMS)
WARM, TIMED = 20, 40
PAIR_REVERSE = (False, True, False)
BF16_COUNTS = {"gate": (ROUTES * LOCAL, ROUTES * LOCAL), "down": (ROUTES * HIDDEN,)}

# All nonzero fixture values are exact multiples of 1/8. Smaller than max
# finite E4M3, they exercise signed cancellation without FP32 precision loss.
CODE = (0x00, 0x20, 0x28, 0x30, 0x38, 0x3c, 0x40, 0x42, 0x44)
EIGHTHS = (0, 1, 2, 4, 8, 12, 16, 20, 24)
DECODE = {code | sign: (-units if sign else units)
          for code, units in zip(CODE, EIGHTHS) for sign in (0, 0x80)}


def sha(data):
    return hashlib.sha256(data).hexdigest()


@lru_cache(maxsize=256)
def block(case, role, index):
    state = (case + 1) * 0x473A57 + (role + 1) * 0x9B5127 + (index + 1) * 0x6E105B
    values = bytearray()
    for _ in range(BLOCK):
        state = (state * 1664525 + 1013904223) & 0xffffffff
        code = CODE[((state >> 16) & 0xffff) % len(CODE)]
        values.append(code | ((state >> 8) & 0x80))
    return bytes(values)


def x_index(row, col):
    return (row * 7 + col * 11 + col // 7) % 17


def w_index(expert, row, col):
    return (expert * 13 + row * 5 + row // 9 + col * 7 + col // 3) % 17


def x_exponent(case, row, col):
    return (case * 2 + row * 3 + col * 5) % 5 - 6


def w_exponent(case, role, expert, row, col):
    return (case * 3 + role * 5 + expert * 7 + row + col * 3) % 7 - 10


def f32_array(values):
    return array("f", values)


def inputs(case):
    _, _, experts = CASES[case]
    if sys.byteorder != "little" or len(f32_array([1]).tobytes()) != 4:
        raise ValueError("fixed little-endian F32/I64 fixture")
    result = {
        "ids": struct.pack("<8q", *experts),
        "gate_x": b"".join(block(case, 0, x_index(0, col)) for col in range(48)),
        "gate_xs": f32_array(math.ldexp(1., x_exponent(case, 0, col)) for col in range(48)),
        "down_x": b"".join(block(case, 1, x_index(row, 0)) for row in range(ROUTES)),
        "down_xs": f32_array(math.ldexp(1., x_exponent(case, row, 0)) for row in range(ROUTES)),
    }
    return {key: value if isinstance(value, bytes) else value.tobytes() for key, value in result.items()}


def weights(case, role):
    """Yield full contiguous banks; the 248 untouched experts are +zero."""
    _, _, active = CASES[case]
    if role not in (2, 3, 4):
        raise ValueError("invalid weight role")
    down = role == 4
    rows, cols = (HIDDEN, 1) if down else (LOCAL, 48)
    expert_bytes = rows * cols * BLOCK
    packed = bytearray(EXPERTS * expert_bytes)
    for expert in active:
        expert_data = b"".join(
            block(case, role, w_index(expert, row, col)) for row in range(rows) for col in range(cols)
        )
        start = expert * expert_bytes
        packed[start:start + expert_bytes] = expert_data
    name = {2: "gate", 3: "up", 4: "down"}[role]
    yield name + "_w", packed
    scales = f32_array(math.ldexp(1., w_exponent(case, role, expert, row, col))
                      for expert in range(EXPERTS)
                      for row in range(48 if down else 16)
                      for col in range(16 if down else 48))
    yield name + "_ws", scales.tobytes()


@lru_cache(maxsize=5 * 17 * 17)
def int_dot(case, role, xindex, windex):
    xv = block(case, int(role == 4), xindex)
    wv = block(case, role, windex)
    return sum(DECODE[x] * DECODE[w] for x, w in zip(xv, wv))


def bf16(value):
    """Round a finite representable F32 value to BF16 with ties to even."""
    bits, = struct.unpack("<I", struct.pack("<f", value))
    return (bits + 0x7fff + ((bits >> 16) & 1)) >> 16


@lru_cache(maxsize=8)
def reference(case, kind):
    """Independent integer dots, accumulated in units of 2**-22.

    The exponent bound here is -22 including the two packed 1/8 factors.
    Check the fixture accumulator's exact representability by F32.
    """
    _, offset, experts = CASES[case]
    shard = offset // LOCAL
    roles = (2, 3) if kind == "gate" else (4,)
    outputs = []
    for role in roles:
        payload = array("H")
        for route, expert in enumerate(experts):
            for row in range(LOCAL if kind == "gate" else HIDDEN):
                total = 0
                for col in range(48 if kind == "gate" else 1):
                    if kind == "gate":
                        xin, scale_row, scale_col = 0, shard, col
                    else:
                        xin, scale_row, scale_col = route, row // LOCAL, shard
                    dot = int_dot(case, role, x_index(xin, col), w_index(expert, row, col))
                    shift = x_exponent(case, xin, col) + w_exponent(case, role, expert,
                                                                   scale_row, scale_col) + 16
                    assert shift >= 0
                    total += dot << shift
                    if abs(total) >= 2**24:
                        raise ValueError("fixture cancellation exceeds exact FP32 accumulator range")
                payload.append(bf16(math.ldexp(total, -22)))
        outputs.append(payload.tobytes())
    return tuple(outputs)


def comparisons(actual, expected):
    if len(actual) != len(expected) or any(len(a) != len(b) for a, b in zip(actual, expected)):
        raise ValueError("wrong BF16 shape")
    def decode(bits):
        return struct.unpack("<f", struct.pack("<I", bits << 16))[0]

    rows = []
    for got, wanted in zip(actual, expected):
        av, ev = array("H"), array("H")
        av.frombytes(got)
        ev.frombytes(wanted)
        bit_diff = numeric_diff = nonfinite = 0
        max_abs_error = 0.
        for a, e in zip(av, ev):
            bit_diff += a != e
            if a & 0x7f80 == 0x7f80:
                nonfinite += 1
                numeric_diff += 1
            elif a != e and (a & 0x7fff or e & 0x7fff):
                numeric_diff += 1
                max_abs_error = max(max_abs_error, abs(decode(a) - decode(e)))
        rows.append(dict(values=len(av), bit_differences=bit_diff, numeric_differences=numeric_diff,
                         nonfinite=nonfinite, max_abs_error=max_abs_error, sha256=sha(got)))
    return rows


def passing(rows):
    return all(row["numeric_differences"] == row["nonfinite"] == 0 for row in rows)
