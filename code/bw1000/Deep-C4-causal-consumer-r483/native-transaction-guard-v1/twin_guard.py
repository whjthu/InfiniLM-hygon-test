"""Call the isolated twin guard diagnostic on a caller-owned Trial."""

import ctypes as C
from hashlib import sha256
import math
import sys

from causal import inputs
from checkpoint import load_checkpoint
from input_audit import check_before_load

SYMBOL = "deep_c4_probe_transaction_guard_r483"
PLANNED = ("invalid_row", "invalid_positions", "foreign_commit", "stale_commit", "valid_commit")
INPUT_SHA256 = ('5590a7556bf1cc50cc32df660c787f9a607a2c91c24323412d706781b3dc746e', '8aa3d98874e52c2c643383dd67912023e4dff3fbeb3d4f53462d4aaa851b4902')
COUNTERS = ("input_audits", "input_buffers_checked", "input_bytes_checked")


def bind_transaction_guard(trial):
    symbol = getattr(trial.library, SYMBOL)
    symbol.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.c_void_p, C.c_size_t]
    symbol.restype = C.c_int
    trial.api["transaction_guard"] = symbol


def verify_observed(observed, past, completed, counters):
    for field, expected in zip(COUNTERS, counters):
        actual = observed.get(field)
        if type(actual) is not int or actual != expected:
            raise ValueError(f"twin diagnostic {field} differs")
    if observed.get("ready") is not True or observed.get("pending") is not False:
        raise ValueError("twin diagnostic owner state differs")
    if type(observed.get("past")) is not int or observed["past"] != past:
        raise ValueError("twin diagnostic committed position differs")
    if observed.get("transaction_guard_completed") is not completed:
        raise ValueError("twin diagnostic completion differs")
    if type(observed.get("driver_free")) is not int or observed["driver_free"] < 16 * 1024**3:
        raise ValueError("strict16GiB floor")
    seconds = observed.get("input_audit_seconds")
    if type(seconds) not in (int, float) or not 0 <= seconds < math.inf:
        raise ValueError("invalid input audit clock")
    return dict(zip(COUNTERS, counters), input_audit_seconds=seconds)


def run(trial, metadata, pins, checkpoint_root, deadline, progress):
    """The caller supplies the owned and qualified library, keeper and cleanup."""
    progress(dict(event="planned", cases=PLANNED, complete=False))
    phase = "before_load"
    try:
        if sys.flags.optimize:
            raise RuntimeError("checkpoint verification requires assertions enabled")
        deadline(phase)
        bind_transaction_guard(trial)
        fresh = trial.read("observe")
        check_before_load(fresh)
        if fresh.get("transaction_guard_completed") is not False:
            raise ValueError("fresh twin guard entry required")
        headers = {name: entry["header_SHA256"] for name, entry in metadata["headers"].items()}
        if pins["header_SHA256"] != headers:
            raise ValueError("original receipt header digests differ")
        phase = "checkpoint_load"
        checkpoint = load_checkpoint(trial, metadata, pins, checkpoint_root, deadline)
        before = verify_observed(trial.read("observe"), 0, False, (14, 14, 26295808))
        progress(dict(event="loaded", observed=before))
        phase = "transaction_guard"
        hidden, query, _ = inputs(0, 4)
        if tuple(sha256(data).hexdigest() for data in (hidden, query)) != INPUT_SHA256:
            raise ValueError("frozen twin guard inputs differ")
        deadline(phase)
        x = C.create_string_buffer(hidden, len(hidden))
        q = C.create_string_buffer(query, len(query))
        trial.call("transaction_guard", x, len(hidden), q, len(query))
        deadline("after_transaction_guard")
        phase = "result"
        after = verify_observed(trial.read("observe"), 4, True, (15, 18, 26338880))
        if after["input_audit_seconds"] < before["input_audit_seconds"]:
            raise ValueError("twin diagnostic input audit clock decreased")
    except BaseException as error:
        progress(dict(event="failed", phase=phase, exception_type=type(error).__name__,
                      cases=PLANNED, complete=False))
        raise
    progress(dict(event="complete", cases=PLANNED, observed=after, complete=True))
    return dict(cases=PLANNED, checkpoint=checkpoint, observed=after,
                performance_eligible=False, Model_GO=False)
