"""Audit a private hipprof JSON trace for the one-call BF16 witness."""

from hashlib import sha256
import json
import re


MAX_TRACE_BYTES = 16 << 20
MAX_EVENTS = 10_000


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def integer(value):
    require(type(value) is int or (isinstance(value, str) and re.fullmatch(r"[0-9]{1,20}", value)),
            "Trace field is not an unsigned integer")
    number = int(value)
    require(number >= 0, "Negative trace integer")
    return number


def contraction(name):
    lowered = name.lower()
    return ((name.startswith("Cijk_") and "_ISA936_" in name)
            or (any(vendor in lowered for vendor in ("tensile", "hipblaslt", "rocblas"))
                and any(op in lowered for op in ("gemm", "matmul", "contraction"))
                and not any(helper in lowered for helper in ("reduce", "reduction", "workspace", "epilogue"))))


def read_trace(path, owner):
    require(path.is_file() and not path.is_symlink(), "Missing original trace JSON")
    with path.open("rb") as stream:
        raw = stream.read(MAX_TRACE_BYTES + 1)
    require(0 < len(raw) <= MAX_TRACE_BYTES, "Trace byte extent exceeded")
    events = json.loads(raw)["traceEvents"]
    require(isinstance(events, list) and len(events) <= MAX_EVENTS, "Trace event extent exceeded")
    phase = [row for row in events if row.get("cat") == "HIPOPS" and row.get("ph") == "X"]
    window_begin, window_end = owner["realtime_ns"]
    require(0 < window_begin < window_end and window_end - window_begin < 20_000_000_000,
            "Invalid one-call host observation window")
    counts = dict(complete_HIPOPS=len(phase), other_owner=0, wrong_device=0, outside_window=0,
                  attached=0, recognized_contractions=0)
    symbols = []
    for item in phase:
        args = item["args"]
        if integer(args["pid"]) != owner["pid"]:
            counts["other_owner"] += 1
            continue
        if integer(args["devId"]) != owner["trace_device_index"]:
            counts["wrong_device"] += 1
            continue
        begin, end = integer(args["BeginNs"]), integer(args["EndNs"])
        if not window_begin <= begin < end <= window_end:
            counts["outside_window"] += 1
            continue
        name = args["name"]
        require(isinstance(name, str) and 0 < len(name) <= 4096 and name.isascii(), "Invalid kernel symbol")
        counts["attached"] += 1
        if contraction(name):
            symbols.append(name)
    counts["recognized_contractions"] = len(symbols)
    proven = len(symbols) == 1 and not any(counts[key] for key in ("other_owner", "wrong_device", "outside_window"))
    return dict(raw_trace_bytes=len(raw), raw_trace_SHA256=sha256(raw).hexdigest(),
                counts=counts, unique_contraction=symbols[0] if proven else None)


def assess(routes, layout, original, baseline, alternative):
    stock = routes.get(baseline, {}).get("unique_contraction")
    tested = routes.get(alternative, {}).get("unique_contraction")
    if stock != original[layout] or not tested:
        return "UNQUALIFIED"
    return "SAME_KERNEL" if stock == tested else "DIFFERENT_KERNEL"
