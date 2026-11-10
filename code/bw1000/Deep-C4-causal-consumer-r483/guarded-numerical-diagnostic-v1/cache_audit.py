"""Independent byte check for a single-row full-context C4 cache transaction."""

from hashlib import sha256
import math
import struct
from time import monotonic

from score_reference import compare_t1_scores

CONTEXT = 32768
COMPRESS_RATIO = 4
KEY_BYTES = 128
SCALE_BYTES = 4
SLOTS_PER_PAGE = 64
PAGE_BYTES = SLOTS_PER_PAGE * (KEY_BYTES + SCALE_BYTES)
CACHE_BYTES = CONTEXT // COMPRESS_RATIO // SLOTS_PER_PAGE * PAGE_BYTES


def check_packed_keys(first_slot, slot_count, values, scales):
    """Check every new E4M3FN key and its little-endian FP32 scale."""
    if len(values) != slot_count * KEY_BYTES or len(scales) != slot_count * SCALE_BYTES:
        raise ValueError("wrong pending packed key or scale byte count")
    for offset, (scale,) in enumerate(struct.iter_unpack("<f", scales)):
        slot = first_slot + offset
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"invalid packed C4 key scale at slot {slot}")
        row = values[offset * KEY_BYTES:(offset + 1) * KEY_BYTES]
        if 0x7F in row or 0xFF in row:
            raise ValueError(f"nonfinite E4M3FN C4 key at slot {slot}")


def audit_cache_transition(past, tokens, committed, before, after, values, scales):
    """Check the complete raw cache against packed pending bytes and the step."""
    if type(past) is not int or type(tokens) is not int or type(committed) is not bool:
        raise ValueError("invalid C4 transaction geometry")
    if not 1 <= tokens <= 512 or not 0 <= past <= CONTEXT - tokens:
        raise ValueError("C4 transaction exceeds frozen context")
    if len(before) != CACHE_BYTES or len(after) != CACHE_BYTES:
        raise ValueError("wrong full-context cache byte count")

    first_slot = past // COMPRESS_RATIO
    slot_count = (past + tokens) // COMPRESS_RATIO - first_slot
    if len(values) != slot_count * KEY_BYTES or len(scales) != slot_count * SCALE_BYTES:
        raise ValueError("wrong pending packed key or scale byte count")

    expected = bytearray(before)
    if committed:
        for offset in range(slot_count):
            page, row = divmod(first_slot + offset, SLOTS_PER_PAGE)
            base = page * PAGE_BYTES
            key_at = base + row * KEY_BYTES
            scale_at = base + SLOTS_PER_PAGE * KEY_BYTES + row * SCALE_BYTES
            expected[key_at:key_at + KEY_BYTES] = values[offset * KEY_BYTES:(offset + 1) * KEY_BYTES]
            expected[scale_at:scale_at + SCALE_BYTES] = scales[offset * SCALE_BYTES:(offset + 1) * SCALE_BYTES]
    if after != expected:
        raise ValueError("raw cache differs from the complete expected byte image")
    return {
        "checked_cache_bytes": CACHE_BYTES,
        "first_new_slot": first_slot if slot_count else None,
        "pending_slots": slot_count,
        "committed_slots": slot_count if committed else 0,
        "after_SHA256": sha256(after).hexdigest(),
    }

RAW_SHAPE = [128, 8448]


class CacheCapture:
    """A single diagnostic transaction. Copies are outside performance runs."""

    def __init__(self, trial, step, copy_tensor, deadline):
        self.trial = trial
        self.step = step
        self.copy_tensor = copy_tensor
        self.deadline = deadline
        self.copy_bytes = 0
        self.copy_seconds = 0.0
        self.before = self._copy("raw_cache", RAW_SHAPE, 4, CACHE_BYTES, "before")
        self.packed = None
        self.score_report = None

    def _copy(self, field, shape, dtype, nbytes, phase):
        self.deadline(f"cache_diagnostic.{phase}.{field}")
        start = monotonic()
        meta, data = self.copy_tensor(self.trial, field)
        self.copy_seconds += monotonic() - start
        if (meta["shape"] != shape or meta["dtype"] != dtype
                or meta["bytes"] != nbytes or len(data) != nbytes):
            raise ValueError(f"wrong cache diagnostic tensor geometry: {field}")
        self.copy_bytes += nbytes
        return data

    def pending(self):
        if self.packed is not None:
            raise RuntimeError("pending cache already captured")
        middle = self._copy("raw_cache", RAW_SHAPE, 4, CACHE_BYTES, "pending")
        if middle != self.before:
            raise ValueError("prepare mutated the committed raw cache")
        past, tokens = self.step["past"], self.step["tokens"]
        count = (past + tokens) // COMPRESS_RATIO - past // COMPRESS_RATIO
        if count:
            values = self._copy("new_key_values", [count, KEY_BYTES], 4, count * KEY_BYTES, "pending")
            scales = self._copy("new_key_scales", [count, 1], 10, count * SCALE_BYTES, "pending")
            self.deadline("cache_diagnostic.check_packed_keys")
            check_packed_keys(past // COMPRESS_RATIO, count, values, scales)
            self.packed = values, scales
        else:
            self.packed = b"", b""
        if tokens == 1:
            width = (past + 1) // COMPRESS_RATIO
            score_bytes_before = self.copy_bytes
            query = self._copy("query", [1, 64, 128], 4, 8192, "pending")
            weights = self._copy("merged_weights", [1, 64], 10, 256, "pending")
            scores = self._copy("scores", [1, width], 10, width * 4, "pending")
            self.deadline("score_diagnostic.compare")
            start = monotonic()
            self.score_report = compare_t1_scores(past, self.before, query, weights, scores, *self.packed)
            self.score_report.update(
                copy_bytes=self.copy_bytes - score_bytes_before,
                compare_seconds=monotonic() - start,
            )
            self.deadline("score_diagnostic.complete")

    def finish(self):
        if self.packed is None:
            raise RuntimeError("missing pending cache capture")
        after = self._copy("raw_cache", RAW_SHAPE, 4, CACHE_BYTES, "finished")
        self.deadline("cache_diagnostic.compare")
        report = audit_cache_transition(
            self.step["past"], self.step["tokens"], self.step["commit"],
            self.before, after, *self.packed,
        )
        self.deadline("cache_diagnostic.complete")
        report.update(copy_bytes=self.copy_bytes, copy_seconds=self.copy_seconds,
                      finite_packed_key_rows=len(self.packed[1]) // SCALE_BYTES,
                      score=self.score_report)
        return report
