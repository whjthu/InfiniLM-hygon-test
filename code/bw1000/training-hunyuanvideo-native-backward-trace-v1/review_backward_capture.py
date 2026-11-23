"""Independently reduce the original native backward trace without replaying training."""

from hashlib import sha256
import json
import os
from pathlib import Path
import stat

from hunyuanvideo_backward_trace import MARKER, MAX_TRACE_BYTES, STEP, reduce_trace


def read_small_file(path, limit):
    def identity(info):
        return (info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns)

    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= limit:
        raise ValueError("Original diagnostic size or type differs")
    with path.open("rb") as stream:
        if identity(os.fstat(stream.fileno())) != identity(before):
            raise ValueError("Original diagnostic open identity differs")
        data = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
    if len(data) != before.st_size or identity(after) != identity(before) or identity(path.lstat()) != identity(before):
        raise ValueError("Original diagnostic changed during review")
    return data


def read_backward_capture(output):
    root = Path(output) / "t09-backward-trace"
    result = {"planned_global_step": STEP, "passed": False, "formal_performance_eligible": False}
    try:
        intent = json.loads(read_small_file(root / "intent.json", 4096))
        if intent != {"planned_global_step": STEP, "marker": MARKER, "formal_performance_eligible": False}:
            raise ValueError("Wrong native backward intent")
        raw = read_small_file(root / "backward.trace.json", MAX_TRACE_BYTES)
        record = read_small_file(root / "backward.summary.json", 32768)
        independent = reduce_trace(json.loads(raw))
        expected = {
            **independent, "actual_global_step": STEP,
            "private_trace_bytes": len(raw), "private_trace_SHA256": sha256(raw).hexdigest(),
        }
        if json.loads(record) != expected:
            raise ValueError("Native backward trace and original summary disagree")
        result.update(expected, passed=True, summary_SHA256=sha256(record).hexdigest())
    except (OSError, KeyError, ValueError, TypeError) as error:
        result["error_type"] = type(error).__name__
    return result
