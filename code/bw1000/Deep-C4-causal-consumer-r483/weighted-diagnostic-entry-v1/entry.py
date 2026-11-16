"""Bind the first weighted C4 diagnostic to the original range-identity receipt."""

import json
import os
from pathlib import Path
import stat
import time

from range_receipt import EXPECTED_RANGE_BYTES, MAX_RECEIPT_BYTES, validate_receipt


ORIGINAL_RESULT = "Deep483-layer2-C4-indexer-original-seven-weight-range-pins-v1.json"
OUTPUT_LABELS = {"warm_abort", "primary", "full_boundary"}
OUTPUT_FIELDS = {"scores", "indices"}
FLOOR = 16 * 1024**3


class MissingOriginalReceipt(ValueError):
    pass


def require(ok, error):
    if not ok:
        raise ValueError(error)


def identity(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def admit(result_directory, receipt_sha, metadata_raw):
    """Read only the exact original receipt. This gate cannot inspect weights."""
    path = Path(result_directory) / ORIGINAL_RESULT
    try:
        before = path.lstat()
    except FileNotFoundError:
        raise MissingOriginalReceipt("original seven-range identity receipt is absent") from None
    require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= MAX_RECEIPT_BYTES,
            "original receipt must be a small regular file")
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        require(stat.S_ISREG(opened.st_mode) and opened.st_size == before.st_size,
                "original receipt was replaced")
        if os.name == "posix":
            require(identity(opened) == identity(before), "original receipt was replaced")
        raw = stream.read(MAX_RECEIPT_BYTES + 1)
    require(identity(before) == identity(path.lstat()) and len(raw) == before.st_size,
            "original receipt bytes differ")
    return validate_receipt(raw, receipt_sha, metadata_raw)


def deadline_from_intent(epoch, deadline, *, wall=time.time, monotonic=time.monotonic):
    """Constrain every synchronous phase to the original caller's deadline."""
    require(type(epoch) in (int, float) and type(deadline) in (int, float), "original launch times missing")
    now = wall()
    require(0 < epoch <= now < deadline < float("inf"), "original diagnostic deadline is invalid")
    stop = monotonic() + deadline - now

    def check(stage):
        if wall() >= deadline or monotonic() >= stop:
            raise TimeoutError("original weighted diagnostic deadline expired: " + stage)
    return check


def run_weighted(verified_library, binding, metadata_raw, causal_binding, result_directory, claim,
                 checkpoint_root, output, deadline):
    """Run one full guarded, weighted component diagnostic after original receipt admission."""
    pins = admit(result_directory, claim["original_range_receipt_SHA256"], metadata_raw)
    metadata = json.loads(metadata_raw)
    deadline("weighted.before_private_artifacts")
    from checkpoint import config
    from run_diagnostic import run_diagnostic

    output = Path(output)
    output.mkdir(mode=0o700)
    pairs = set()
    with (output / "progress.jsonl").open("x", encoding="utf8") as journal:
        def progress(record):
            journal.write(json.dumps(record, allow_nan=False, sort_keys=True) + "\n")
            journal.flush()
            os.fsync(journal.fileno())

        def sink(label, field, meta, data):
            require(label in OUTPUT_LABELS and field in OUTPUT_FIELDS and (label, field) not in pairs,
                    "unexpected or repeated private output")
            require(0 < len(data) == meta["bytes"] <= 32768, "wrong private output size")
            with (output / f"{label}-{field}.bin").open("xb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            pairs.add((label, field))

        progress({"event": "admitted", "planned_states": 71, "planned_outputs": 6,
                  "original_receipt_SHA256": claim["original_range_receipt_SHA256"], "complete": False})
        trial = None
        failure = None
        try:
            deadline("weighted.before_create")
            trial = binding["Trial"](verified_library, config(metadata), metadata["layer"], 0)
            result = run_diagnostic(trial, metadata, pins, checkpoint_root, causal_binding, deadline, sink, progress)
            require(pairs == {(label, field) for label in OUTPUT_LABELS for field in OUTPUT_FIELDS},
                    "incomplete private output denominator")
            require(result["checkpoint"]["bytes"] == EXPECTED_RANGE_BYTES
                    and not result["diagnostic"]["performance_eligible"],
                    "wrong numeric diagnostic scope")
        except BaseException as error:
            failure = error
            progress({"event": "weighted_failure", "exception_type": type(error).__name__, "complete": False})
            raise
        finally:
            if trial is not None:
                try:
                    closed = trial.close()
                    require(len(closed) == 4 and min(closed[:2]) >= FLOOR and closed[2:] == [1, 1],
                            "native owner was not released at the configured floor")
                    progress({"event": "closed", "driver_free_before_after_close": closed[:2]})
                except BaseException as error:
                    progress({"event": "close_failed", "exception_type": type(error).__name__, "complete": False})
                    if failure is not None:
                        raise failure from error
                    raise
        result["closed"] = closed

        with (output / "weighted-outcome.json").open("x", encoding="utf8") as receipt:
            json.dump(result, receipt, sort_keys=True, indent=2, allow_nan=False)
            receipt.flush()
            os.fsync(receipt.fileno())
        progress({"event": "weighted_complete", "complete": True})
    return result
