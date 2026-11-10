"""Independent FP64 reference for sampled pre-pooling C4 projections."""

import math
import struct
from hashlib import sha256
from time import monotonic


HIDDEN = 4096
CHANNELS_PER_WEIGHT = 256
PROJECTED_CHANNELS = 2 * CHANNELS_PER_WEIGHT
ROW_BYTES = 2 * HIDDEN


def decode_bf16_row(buffer, offset):
    """Decode one little-endian BF16 row without rounding its values."""
    values = struct.unpack_from(f"<{HIDDEN}H", buffer, offset)
    expanded = struct.pack(f"<{HIDDEN}I", *(bits << 16 for bits in values))
    row = struct.unpack(f"<{HIDDEN}f", expanded)
    if not all(map(math.isfinite, row)):
        raise ValueError("nonfinite projection operand")
    return row


def compare_projection(hidden, wkv, wgate, observed, samples):
    """Compare (row, channel) pairs against ideal dots before BF16 rounding.

    Both weight buffers must come from the checkpoint loader's verified
    payloads. Observed is the complete private projected_score BF16 tensor.
    The result measures error and leaves the device tolerance to the caller.
    """
    if not hidden or len(hidden) % ROW_BYTES:
        raise ValueError("hidden must contain complete BF16 rows")
    tokens = len(hidden) // ROW_BYTES
    if tokens > 512:
        raise ValueError("projection input exceeds the frozen prepare size")
    if len(wkv) != ROW_BYTES * CHANNELS_PER_WEIGHT or len(wgate) != len(wkv):
        raise ValueError("wrong BF16 projection weight geometry")
    if len(observed) != tokens * PROJECTED_CHANNELS * 2:
        raise ValueError("wrong BF16 projected_score geometry")
    samples = tuple(samples)
    if not samples or len(set(samples)) != len(samples):
        raise ValueError("nonempty unique projection samples required")

    hidden_rows = {}
    weight_rows = {}
    absolute_errors = []
    for row, channel in samples:
        if type(row) is not int or type(channel) is not int:
            raise ValueError("projection coordinates must be integers")
        if not 0 <= row < tokens or not 0 <= channel < PROJECTED_CHANNELS:
            raise ValueError("projection coordinate out of range")
        if row not in hidden_rows:
            hidden_rows[row] = decode_bf16_row(hidden, row * ROW_BYTES)
        if channel not in weight_rows:
            weight = wkv if channel < CHANNELS_PER_WEIGHT else wgate
            offset = channel % CHANNELS_PER_WEIGHT * ROW_BYTES
            weight_rows[channel] = decode_bf16_row(weight, offset)
        expected = math.fsum(
            x * y for x, y in zip(hidden_rows[row], weight_rows[channel])
        )
        bits, = struct.unpack_from("<H", observed, 2 * (row * PROJECTED_CHANNELS + channel))
        actual, = struct.unpack("<f", struct.pack("<I", bits << 16))
        if not math.isfinite(actual):
            raise ValueError("nonfinite projected_score sample")
        absolute_errors.append(abs(actual - expected))

    return {
        "samples": len(absolute_errors),
        "max_abs_error": max(absolute_errors),
        "mean_abs_error": math.fsum(absolute_errors) / len(absolute_errors),
    }

"""Check the documented C4 pooled-output support against logical positions."""

import struct

CHANNELS = 128
C4_WINDOW = 4
MAX_TOKENS = 512
CONTEXT = 32768


def check_pooled_support(past, tokens, observed):
    """Count finite boundary rows and require zero on every other row.

    The caller supplies the full little-endian BF16 pooled tensor copied
    from the pending C4 transaction. Either sign of zero is accepted.
    """
    if type(past) is not int or type(tokens) is not int:
        raise ValueError("C4 geometry must use integers")
    if not 1 <= tokens <= MAX_TOKENS or not 0 <= past <= CONTEXT - tokens:
        raise ValueError("C4 geometry exceeds the frozen input")
    if len(observed) != tokens * CHANNELS * 2:
        raise ValueError("wrong BF16 pooled geometry")

    boundary_rows = 0
    for row in range(tokens):
        bits = struct.unpack_from(f"<{CHANNELS}H", observed, row * CHANNELS * 2)
        position = past + row
        if (position + 1) % C4_WINDOW == 0:
            if any((value & 0x7F80) == 0x7F80 for value in bits):
                raise ValueError(f"nonfinite pooled value at boundary position {position}")
            boundary_rows += 1
        elif any(value & 0x7FFF for value in bits):
            raise ValueError(f"nonzero pooled value at off-boundary position {position}")
    return {
        "boundary_rows": boundary_rows,
        "off_boundary_zero_rows": tokens - boundary_rows,
        "checked_values": tokens * CHANNELS,
    }

PROJECTION_KEYS = ("compressor.wkv.weight", "compressor.wgate.weight")
PROJECTION_BYTES = 256 * 4096 * 2
CHANNEL_SAMPLES = (0, 63, 127, 128, 191, 255, 256, 319, 383, 384, 447, 511)
SAMPLE_ROWS = {
    ("history", 0, 512, 0): (0, 3, 7, 511),
    ("history", 512, 512, 0): (0, 3, 7, 511),
    ("warm_abort", 8496, 1, 0): (0,),
    ("primary", 8496, 1, 0): (0,),
    ("full_boundary", 32767, 1, 0): (0,),
    ("rewrite", 0, 512, 1): (0, 3, 7, 511),
}


def verify_projection_weights(payloads, pins):
    """Retain only immutable projection bytes that match the original pins."""
    verified = []
    for key in PROJECTION_KEYS:
        data = bytes(payloads[key])
        pin = pins["weights"][key]
        if len(data) != PROJECTION_BYTES or pin["bytes"] != PROJECTION_BYTES:
            raise ValueError("wrong projection weight size: " + key)
        if sha256(data).hexdigest() != pin["SHA256"]:
            raise ValueError("wrong projection weight digest: " + key)
        verified.append(data)
    return tuple(verified)


def audit_pending(trial, step, hidden, verified_weights, copy_tensor, deadline):
    """Audit an already prepared transaction before its commit or abort.

    The driver must use an unmeasured diagnostic pass and supply the original
    step binding and checkpoint-verified weights. Only summaries are returned.
    """
    case = step["label"], step["past"], step["tokens"], step["variant"]
    rows = SAMPLE_ROWS.get(case)
    if rows is None:
        return None
    label, past, tokens, variant = case
    hidden_pin = step["inputs"]["hidden_BF16"]
    if (len(hidden) != tokens * 4096 * 2 or len(hidden) != hidden_pin["bytes"]
            or sha256(hidden).hexdigest() != hidden_pin["SHA256"]):
        raise ValueError("diagnostic hidden input differs from the frozen binding")
    start = monotonic()
    outputs = {}
    for name, width in (("projected_score", 512), ("pooled", 128)):
        deadline("numeric.copy." + name)
        metadata, data = copy_tensor(trial, name)
        if (metadata["shape"] != [tokens, width] or metadata["dtype"] != 9
                or metadata["bytes"] != len(data) or len(data) != tokens * width * 2):
            raise ValueError("wrong diagnostic tensor geometry: " + name)
        outputs[name] = data
    copy_seconds = monotonic() - start
    deadline("numeric.compare")
    start = monotonic()
    projection = compare_projection(
        hidden, *verified_weights, outputs["projected_score"],
        [(row, channel) for row in rows for channel in CHANNEL_SAMPLES],
    )
    pooled = check_pooled_support(past, tokens, outputs["pooled"])
    compare_seconds = monotonic() - start
    deadline("numeric.complete")
    return {
        "label": label,
        "past": past,
        "tokens": tokens,
        "variant": variant,
        "projection": projection,
        "pooled": pooled,
        "copy_bytes": sum(map(len, outputs.values())),
        "copy_seconds": copy_seconds,
        "compare_seconds": compare_seconds,
        "timing_scope": "unmeasured diagnostic",
    }
