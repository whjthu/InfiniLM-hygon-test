"""Recognize a failed GLM device worker before its BOUND receipt."""

from hashlib import sha256
import json


def inspect_probe(path, gpu_row, intent_sha, before, probe, host):
    """Drop-in host result reader, reusing the frozen complete-trace parser."""
    if not path.is_file():
        return host.inspect_probe(path, gpu_row, intent_sha, before, probe)
    try:
        raw = host.bounded(path, 262144)
        lines = raw.splitlines(keepends=True)
        if len(lines) < 2 or not lines[1].endswith(b"\n"):
            return host.inspect_probe(path, gpu_row, intent_sha, before, probe)
        candidate = json.loads(lines[1])
    except (OSError, ValueError):
        return host.inspect_probe(path, gpu_row, intent_sha, before, probe)
    if not isinstance(candidate, dict) or candidate.get("state") != "FAIL":
        return host.inspect_probe(path, gpu_row, intent_sha, before, probe)

    result = dict(worker=None, expected=probe.PLANNED, passed={"valid": 0, "rejected": 0},
                  trace_pass=False, trace_SHA256=sha256(raw).hexdigest(), binding_state="UNVERIFIED")
    try:
        host.need(lines[0].endswith(b"\n"), "GPU worker first receipt was interrupted")
        owner = host.own_identity(json.loads(lines[0]), gpu_row, intent_sha, probe)
        host.need(owner["boot_id"] == before["boot_id"], "worker boot differs from fresh admission")
        result["worker"] = owner
        host.need(len(lines) == 2 and candidate.keys() == {"state", "type", "error"},
                  "pre-BOUND failure has unexpected fields or later records")
        host.need(all(type(candidate[key]) is str and 0 < len(candidate[key]) <= 1024 for key in ("type", "error")),
                  "pre-BOUND failure diagnostic is missing or malformed")
        result["worker_failure"] = {key: candidate[key] for key in ("type", "error")}
        result["binding_state"] = "FAILED_BEFORE_BOUND"
    except (ValueError, KeyError, TypeError) as error:
        result["error"] = f"{type(error).__name__}: {str(error)[:220]}"
    return result
