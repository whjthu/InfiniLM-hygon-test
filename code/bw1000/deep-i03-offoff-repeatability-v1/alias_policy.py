"""Same immutable alias-capable Native,all requests OFF; bounded repeatability audit."""

from contextlib import contextmanager
import ctypes as C
from pathlib import Path
import threading

from diagnostic_binding import LIBRARY
from pilot_inputs import compact, require, sha

SET = "deep_c4_history_bank_alias_set_v1"
READ = "deep_c4_history_bank_alias_stats_v1"
SCHEDULE = ((0, 0), (0, 0))
PROMPTS = (
    (1242, "62465754106f1ac717fb73261997c17c67b27c81aceb9ffe3bde723578b9ce46"),
    (538, "6bc6caef0b542ac858832d0ee8aebe8ebf7994e149dab84a8c60db40dfe4a7eb"),
)
AUDIT_FORWARDS = 16
C4_LAYERS = 21
VOCAB = 129280


def expected_counts(trace, mode):
    calls = 0
    eligible = []
    for row in trace:
        if row["kind"] == "prefill":
            calls += 1
            positions = (row["past"],) if row["tokens"] == 1 else ()
        else:
            require(row["kind"] == "decode_summary"
                    and row["last_total"] - row["first_past"] == row["calls"],
                    "unexpected Native continuation")
            calls += row["calls"]
            positions = range(row["first_past"], row["last_total"])
        eligible.extend(past for past in positions if past >= 4 and past % 4 != 3)
    return [C4_LAYERS * calls, C4_LAYERS * len(eligible),
            C4_LAYERS * len(eligible) * mode,
            C4_LAYERS * sum(past // 4 for past in eligible) * 1024 * mode]


def bind(lib):
    setter, reader = getattr(lib, SET), getattr(lib, READ)
    setter.argtypes, setter.restype = [C.c_int], C.c_int
    reader.argtypes, reader.restype = [C.c_int, C.POINTER(C.c_uint64), C.c_size_t], C.c_int

    class Info(C.Structure):
        _fields_ = [("fname", C.c_char_p), ("base", C.c_void_p),
                    ("sname", C.c_char_p), ("address", C.c_void_p)]

    dladdr = C.CDLL(None).dladdr
    dladdr.argtypes = [C.c_void_p, C.POINTER(Info)]
    dladdr.restype = C.c_int
    for symbol in (setter, reader):
        info = Info()
        require(dladdr(C.cast(symbol, C.c_void_p), C.byref(info)) == 1
                and Path(info.fname.decode()).resolve() == LIBRARY.resolve(),
                "foreign history alias provider")
    return setter, reader


def counters(reader):
    rows = []
    for device in range(8):
        values = (C.c_uint64 * 4)()
        require(reader(device, values, len(values)) == 0, "C4 counter read refused")
        rows.append(list(values))
    return rows


def compare(reference, observed):
    import numpy as np

    require(2 <= len(reference) <= AUDIT_FORWARDS and 2 <= len(observed) <= AUDIT_FORWARDS, "missing natural-length prefill audit")
    records = []
    for original, candidate in zip(reference, observed):
        digest, before = original
        current, after = candidate
        if digest != current:
            break
        require(before.shape == after.shape == (VOCAB,), "unexpected BF16 logits shape")
        before_f = (before.astype(np.uint32) << 16).view(np.float32)
        after_f = (after.astype(np.uint32) << 16).view(np.float32)
        require(np.isfinite(before_f).all() and np.isfinite(after_f).all(), "nonfinite audit logits")
        delta = before_f - after_f
        records.append(dict(input_SHA256=digest, first_logits_SHA256=sha(before.tobytes()),
                            repeat_logits_SHA256=sha(after.tobytes()),
                            differing_BF16_values=int(np.count_nonzero(before != after)),
                            max_absolute=float(np.max(np.abs(delta))),
                            rms=float(np.sqrt(np.mean(delta.astype(np.float64) ** 2)))))
    require(len(records) >= 2, "same-input prefill audit missing")
    return records


def install(policy_class):
    original_request, original_forward = policy_class.request, policy_class.forward

    @contextmanager
    def request(self, req, record):
        index = self.index
        control = self.control
        require(0 <= index < len(SCHEDULE) and self.owner == threading.get_ident()
                and self._mode == 0 and not self.poisoned and control.quiescent
                and not control.active, "unavailable serial Native owner")
        case, mode = SCHEDULE[index]
        count, digest = PROMPTS[case]
        require(len(req.prompt_token_ids) == count and req.max_new_tokens == 4096
                and sha(compact(list(req.prompt_token_ids))) == digest,
                "original request geometry or identity differs")
        if index == 0:
            self._alias_set, self._alias_read = bind(control.lib)
        before = counters(self._alias_read)
        if index < 2:
            self._alias_audit = []
        require(self._alias_set(mode) == 0, "C4 policy refused at request boundary")
        try:
            with original_request(self, req, record):
                yield self
        finally:
            require(self._alias_set(0) == 0, "C4 policy could not restore default")
        after = counters(self._alias_read)
        delta = [[b - a for a, b in zip(first, last)] for first, last in zip(before, after)]
        expected = expected_counts(record["Native_continuation"], mode)
        require(mode == 0 and delta == [expected] * 8 and expected[2:] == [0, 0], "OFF-only eight-owner C4 coverage differs")
        record["history_alias"] = dict(mode=mode, counters=delta, provider=str(LIBRARY))
        if index == 0:
            self._alias_reference = self._alias_audit
        if index == 1:
            record["history_alias"]["logits_audit"] = compare(self._alias_reference, self._alias_audit)
            del self._alias_reference
        if index < 2:
            del self._alias_audit

    def forward(self, *args):
        result = original_forward(self, *args)
        audit = getattr(self, "_alias_audit", None)
        if audit is not None and len(audit) < AUDIT_FORWARDS:
            row = result[1][0]
            require(row.shape == (VOCAB,) and row.dtype.name == "uint16", "unexpected BF16 result")
            # Include every causal forward input in the match key, not only the current token.
            # The preceding digest binds the complete prefix since this request's reset.
            previous = audit[-1][0] if audit else None
            audit.append((sha(compact([previous, args])), row.copy()))
        return result

    policy_class.request = request
    policy_class.forward = forward
