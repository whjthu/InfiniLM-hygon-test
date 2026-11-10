"""Run the unmeasured C4 diagnostic on a caller-owned fresh Trial."""

import sys

import checkpoint
import exercise
from cache_audit import CacheCapture
from causal import schedule
from input_audit import InputAudit, check_before_load
from numeric import SAMPLE_ROWS, audit_pending, verify_projection_weights


def run_diagnostic(trial, metadata, pins, checkpoint_root, binding, deadline, sink, progress):
    """Load the original seven ranges, then audit six pending transactions.

    The caller owns cleanup even when load or audit raises. Weight and output
    payloads stay in the process or in the caller's private sink.
    """
    if sys.flags.optimize:
        raise RuntimeError("checkpoint verification requires assertions enabled")
    headers = {name: entry["header_SHA256"] for name, entry in metadata["headers"].items()}
    if pins["header_SHA256"] != headers:
        raise ValueError("original receipt header digests differ")

    projection_weights = None

    def retain(payloads):
        nonlocal projection_weights
        projection_weights = verify_projection_weights(payloads, pins)

    try:
        input_audit = InputAudit(metadata)
        deadline("input_audit.before_checkpoint")
        check_before_load(trial.read("observe"))
        loaded = checkpoint.load_checkpoint(
            trial, metadata, pins, checkpoint_root, deadline, on_verified_projection=retain,
        )
        if projection_weights is None:
            raise RuntimeError("verified checkpoint projections were not retained")
    except BaseException as error:
        progress(dict(event="checkpoint_failed", planned_steps=len(schedule()),
                      completed_steps=0, exception_type=type(error).__name__, complete=False))
        raise

    seen = []

    def audit(*args, **kwargs):
        report = audit_pending(*args, verified_weights=projection_weights, **kwargs)
        if report is not None:
            seen.append(tuple(report[key] for key in ("label", "past", "tokens", "variant")))
        return report

    cache_seen = []

    def capture_cache(trial, step, copy_tensor, deadline):
        case = tuple(step[key] for key in ("label", "past", "tokens", "variant"))
        if case in SAMPLE_ROWS:
            cache_seen.append(case)
            return CacheCapture(trial, step, copy_tensor, deadline)
        return None

    result = exercise.run(
        trial, binding, deadline, sink, progress,
        numeric_audit=audit, cache_capture_factory=capture_cache, input_audit=input_audit,
    )
    if tuple(seen) != tuple(SAMPLE_ROWS) or seen != cache_seen or result["performance_eligible"]:
        raise ValueError("incomplete or mislabeled diagnostic coverage")
    return {"checkpoint": loaded, "diagnostic": result}
