"""Sample the Deep483 TP1 T1 score arithmetic in host FP64."""

import math
import struct

HEADS = 64
CHANNELS = 128
SLOTS_PER_PAGE = 64
PAGE_BYTES = 8448
CACHE_BYTES = 128 * PAGE_BYTES
FULL_CONTEXT = 32768
HISTORY_SAMPLES = (0, 1, 63, 64, 127, 128, 511, 512, 1023, 1024, 2047, 2048)


def decode_e4m3fn(code):
    magnitude = code & 0x7F
    if magnitude == 0x7F:
        raise ValueError("invalid E4M3FN byte")
    exponent, mantissa = divmod(magnitude, 8)
    value = math.ldexp(1 + mantissa / 8, exponent - 7) if exponent else math.ldexp(mantissa, -9)
    return -value if code & 0x80 else value


def compare_t1_scores(past, raw_cache, query, merged_weights, scores, pending_key=b"", pending_scale=b""):
    """Recompute scores from already quantized query/key data and merged weights.

    The input raw cache is the pre-prepare committed image. The single
    just-compressed key, if present, must be passed separately. The comparison
    reports ideal arithmetic error and does not set a device tolerance.
    """
    if type(past) is not int or not 3 <= past < FULL_CONTEXT:
        raise ValueError("visible T1 geometry required")
    before, visible = past // 4, (past + 1) // 4
    append = visible - before
    if len(raw_cache) != CACHE_BYTES or len(query) != HEADS * CHANNELS:
        raise ValueError("wrong cache or query geometry")
    if len(merged_weights) != HEADS * 4 or len(scores) != visible * 4:
        raise ValueError("wrong head weights or score geometry")
    if len(pending_key) != append * CHANNELS or len(pending_scale) != append * 4:
        raise ValueError("wrong pending key geometry")

    weights = struct.unpack(f"<{HEADS}f", merged_weights)
    observed = struct.unpack(f"<{visible}f", scores)
    if not all(map(math.isfinite, weights)) or not all(map(math.isfinite, observed)):
        raise ValueError("nonfinite head weight or score")
    rows = [
        tuple(map(decode_e4m3fn, query[head * CHANNELS:(head + 1) * CHANNELS]))
        for head in range(HEADS)
    ]

    selected = sorted({slot for slot in (*HISTORY_SAMPLES, before - 1, visible - 1)
                       if 0 <= slot < visible})
    errors = []
    for slot in selected:
        if slot == before:
            encoded_key, encoded_scale = pending_key, pending_scale
        else:
            page, row = divmod(slot, SLOTS_PER_PAGE)
            key_at = page * PAGE_BYTES + row * CHANNELS
            scale_at = page * PAGE_BYTES + SLOTS_PER_PAGE * CHANNELS + row * 4
            encoded_key = raw_cache[key_at:key_at + CHANNELS]
            encoded_scale = raw_cache[scale_at:scale_at + 4]
        scale, = struct.unpack("<f", encoded_scale)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("invalid sampled key scale")
        key = tuple(map(decode_e4m3fn, encoded_key))
        expected = math.fsum(
            weight * max(0.0, scale * math.fsum(x * y for x, y in zip(query_row, key)))
            for weight, query_row in zip(weights, rows)
        )
        errors.append(abs(observed[slot] - expected))

    return {
        "slots": selected,
        "visible_slots": visible,
        "max_abs_error": max(errors),
        "mean_abs_error": math.fsum(errors) / len(errors),
    }
