"""Validate host-only C4 timing against the same original r9193 frame."""
import ctypes
import json
import math
from statistics import median
from time import perf_counter

LABELS = (
    "latent_compressor_prepare", "indexer_validation_and_compressor_prepare",
    "indexer_query_features", "boundary_new_key_features", "quantized_key_history_bank",
    "scores_status_reduce_topk_drain", "indexer_return_and_cleanup",
    "latent_history_bank_and_swa_staging", "selection_readback_and_device_metadata",
)


def bind(library):
    read = library.deep_c4_host_split_read_v1
    read.argtypes = []
    read.restype = ctypes.c_char_p
    return read


def consume(raw, original_packet, past):
    if not raw or type(past) is not int or past < 0:
        raise ValueError("missing C4 packet or original past")
    q = json.loads(raw)
    if q["schema"] != 1 or tuple(q["labels"]) != LABELS:
        raise ValueError("C4 host packet schema differs")
    original = json.loads(original_packet)
    parents = {}
    for row in original["rows"]:
        if row["kind"] != 2:
            continue
        key = row["device"], row["layer"]
        if key in parents or row["tokens"] != 1 or row["requests"] != 1:
            raise ValueError("invalid original C4 frame")
        segment = row["segments"][1]
        bound = segment["host_wall_ms"]
        if segment["segment"] != 1 or not math.isfinite(bound) or bound < 0:
            raise ValueError("invalid original host interval")
        parents[key] = bound
    if len(parents) != 168 or any(sum(d == rank for d, _ in parents) != 21 for rank in range(8)):
        raise ValueError("expected eight owner sets of 21 C4 layers")
    totals = {name: [0] * 8 for name in LABELS}
    coverage = set()
    for row in q["rows"]:
        key = row["device"], row["layer"]
        spans = row["host_ns"]
        if (key in coverage or key not in parents or row["past"] != past
                or row["boundary"] is not (past % 4 == 3)
                or len(spans) != len(LABELS)
                or any(type(n) is not int or n < 0 for n in spans)):
            raise ValueError("invalid C4 subinterval or frame binding")
        # The original and new recorders use monotonic clocks. Allow float rounding.
        if sum(spans) / 1e6 > parents[key] + 0.002:
            raise ValueError("C4 split exceeds the corresponding host enclosure")
        coverage.add(key)
        for name, value in zip(LABELS, spans):
            totals[name][key[0]] += value
    if coverage != set(parents):
        raise ValueError("incomplete C4 timing coverage")
    return {
        "past": past, "boundary": past % 4 == 3, "rows": len(coverage),
        "host_ms_by_phase_across_owners": {
            name: [min(v) / 1e6, median(v) / 1e6, max(v) / 1e6] for name, v in totals.items()
        },
    }


def read_after_original(read, original_packet, past):
    start = perf_counter()
    raw = read()
    read_ms = (perf_counter() - start) * 1000
    result = consume(raw, original_packet, past)
    result["read_wall_ms"] = read_ms
    return result, raw
