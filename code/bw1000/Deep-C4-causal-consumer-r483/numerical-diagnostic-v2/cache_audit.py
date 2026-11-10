"""Independent byte check for a single-row full-context C4 cache transaction."""

from hashlib import sha256
from time import monotonic

CONTEXT = 32768
COMPRESS_RATIO = 4
KEY_BYTES = 128
SCALE_BYTES = 4
SLOTS_PER_PAGE = 64
PAGE_BYTES = SLOTS_PER_PAGE * (KEY_BYTES + SCALE_BYTES)
CACHE_BYTES = CONTEXT // COMPRESS_RATIO // SLOTS_PER_PAGE * PAGE_BYTES


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
            self.packed = values, scales
        else:
            self.packed = b"", b""

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
        report.update(copy_bytes=self.copy_bytes, copy_seconds=self.copy_seconds)
        return report
