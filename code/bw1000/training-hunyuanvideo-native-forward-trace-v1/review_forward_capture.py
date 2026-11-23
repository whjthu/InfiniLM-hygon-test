"""Independently reduce the original native forward trace and its SDPA input descriptors."""

from collections import Counter
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import stat

from hunyuanvideo_forward_trace import MARKER, MAX_SDPA_CALLS, MAX_TRACE_BYTES, STEP, reduce_trace


def file_identity(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def read_pinned(path, limit):
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= limit:
        raise ValueError("Original diagnostic size or type differs")
    with path.open("rb") as stream:
        if file_identity(os.fstat(stream.fileno())) != file_identity(before):
            raise ValueError("Original diagnostic open identity differs")
        raw = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
    if len(raw) != before.st_size or file_identity(after) != file_identity(before):
        raise ValueError("Original diagnostic changed during review")
    if file_identity(path.lstat()) != file_identity(before):
        raise ValueError("Original diagnostic path identity differs")
    return raw


def validate_tensor(value):
    keys = {"shape", "stride", "dtype", "device_type", "contiguous"}
    if (set(value) != keys or not 1 <= len(value["shape"]) <= 8 or len(value["shape"]) != len(value["stride"])
            or any(type(x) is not int or x < 1 for x in value["shape"])
            or any(type(x) is not int or x < 0 for x in value["stride"])
            or type(value["dtype"]) is not str or not value["dtype"].startswith("torch.")
            or value["device_type"] not in ("cpu", "cuda") or type(value["contiguous"]) is not bool):
        raise ValueError("Invalid SDPA tensor metadata")


def group_descriptors(descriptors):
    if not isinstance(descriptors, list) or not 0 < len(descriptors) <= MAX_SDPA_CALLS:
        raise ValueError("No bounded native forward SDPA calls")
    expected = {"query", "key", "value", "mask", "is_causal", "dropout_p", "scale", "enable_gqa"}
    groups = Counter()
    for row in descriptors:
        if (set(row) != expected or any(type(row[k]) is not bool for k in ("is_causal", "enable_gqa"))
                or type(row["dropout_p"]) not in (int, float) or not 0 <= row["dropout_p"] <= 1
                or row["scale"] is not None and (type(row["scale"]) not in (int, float) or not math.isfinite(row["scale"]))):
            raise ValueError("Wrong SDPA descriptor schema")
        for key in ("query", "key", "value"):
            validate_tensor(row[key])
            if row[key]["device_type"] != "cuda":
                raise ValueError("Native forward SDPA must use device Q/K/V")
        if row["mask"] is not None:
            validate_tensor(row["mask"])
        groups[json.dumps(row, sort_keys=True, separators=(",", ":"))] += 1
    return [{"calls": count, **json.loads(name)} for name, count in sorted(groups.items(), key=lambda item: (-item[1], item[0]))]


def read_forward_capture(output):
    root = Path(output) / "t09-forward-trace"
    result = {"planned_global_step": STEP, "passed": False, "formal_performance_eligible": False}
    try:
        intent = json.loads(read_pinned(root / "intent.json", 4096))
        if intent != {"planned_global_step": STEP, "marker": MARKER, "formal_performance_eligible": False}:
            raise ValueError("Wrong native forward intent")
        raw = read_pinned(root / "forward.trace.json", MAX_TRACE_BYTES)
        record = read_pinned(root / "forward.summary.json", 32768)
        raw_metadata = read_pinned(root / "sdpa-descriptors.json", 256 * 1024)
        descriptors = json.loads(raw_metadata)
        grouped = group_descriptors(descriptors)
        trace = json.loads(raw)
        independent = reduce_trace(trace)
        expected = {**independent, "actual_global_step": STEP,
                    "private_trace_bytes": len(raw), "private_trace_SHA256": sha256(raw).hexdigest()}
        if json.loads(record) != expected:
            raise ValueError("Native forward trace and original summary disagree")
        selected = Counter(
            event["name"] for event in trace["traceEvents"]
            if event.get("ph") == "X" and event.get("cat") == "cpu_op"
            and event["name"].startswith("aten::")
            and ("scaled_dot_product" in event["name"] or "efficient_attention" in event["name"]
                 or "flash_attention" in event["name"])
        )
        if selected["aten::scaled_dot_product_attention"] < len(descriptors):
            raise ValueError("SDPA descriptor calls lack supporting ATen records")
        result.update(expected, passed=True, summary_SHA256=sha256(record).hexdigest(),
                      sdpa_descriptor_SHA256=sha256(raw_metadata).hexdigest(), sdpa_descriptor_calls=len(descriptors),
                      sdpa_descriptor_groups=grouped[:32], sdpa_distinct_descriptor_count=len(grouped),
                      observed_attention_ATen_ops=dict(selected))
    except (OSError, KeyError, ValueError, TypeError) as error:
        result["error_type"] = type(error).__name__
    return result
