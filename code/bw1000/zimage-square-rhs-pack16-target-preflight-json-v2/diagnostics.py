"""Separate structured preflight results from launch diagnostics."""

import json
from hashlib import sha256
import re
import subprocess


DIAG_PHASES = {"standby_before", "standby_after", "runtime_host", "runtime_inside",
               "container_exec", "container_stdout", "container_result_identity"}
RESULT_CAP = 8192
LOG_HASH_CAP = 1 << 20


def _log_evidence(path, cap):
    with path.open("rb") as handle:
        raw = handle.read(cap + 1)
        nbytes = handle.seek(0, 2)
    if nbytes > cap:
        return None, {"bytes": nbytes, "SHA256": None, "exceeds_hash_cap": True}
    return raw, {"bytes": len(raw), "SHA256": sha256(raw).hexdigest(), "exceeds_hash_cap": False}


class PhaseFailure(ValueError):
    def __init__(self, phase, gate, **details):
        if phase not in DIAG_PHASES:
            raise ValueError("PRECHECK_INVALID_DIAGNOSTIC_PHASE")
        super().__init__(gate)
        self.details = {"phase": phase, "gate": gate, **details}


def decode(phase, raw):
    diagnostic = {"input_bytes": len(raw), "input_SHA256": sha256(raw).hexdigest()}
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as error:
        raise PhaseFailure(phase, "PRECHECK_JSON_DECODE", original_failure_type=type(error).__name__,
                           json_location={"line": error.lineno, "column": error.colno, "offset": error.pos},
                           **diagnostic) from None
    except UnicodeDecodeError as error:
        raise PhaseFailure(phase, "PRECHECK_JSON_ENCODING", original_failure_type=type(error).__name__,
                           input_error_offset=error.start, **diagnostic) from None
    if type(result) is not dict:
        raise PhaseFailure(phase, "PRECHECK_JSON_OBJECT_REQUIRED", **diagnostic)
    return result


def run_container(argv, env, directory, request, step):
    stdout = directory / "container.stdout.private"
    stderr = directory / "container.stderr.private"
    try:
        with stdout.open("xb") as outcome, stderr.open("xb") as errors:
            try:
                rc = subprocess.run(argv, env=env, cwd="/", stdin=subprocess.DEVNULL, stdout=outcome, stderr=errors,
                                    timeout=32, check=False).returncode
            except (OSError, subprocess.TimeoutExpired) as error:
                _, out = _log_evidence(stdout, LOG_HASH_CAP)
                _, err = _log_evidence(stderr, LOG_HASH_CAP)
                raise PhaseFailure("container_exec", "PRECHECK_CONTAINER_EXEC",
                                   original_failure_type=type(error).__name__, stdout=out, stderr=err) from None
    except OSError as error:
        raise PhaseFailure("container_exec", "PRECHECK_PRIVATE_CONTAINER_LOG",
                           original_failure_type=type(error).__name__) from None
    raw, out = _log_evidence(stdout, LOG_HASH_CAP)
    _, err = _log_evidence(stderr, LOG_HASH_CAP)
    context = {"container_rc": rc, "stdout": out, "stderr": err}
    if raw is None or len(raw) > RESULT_CAP:
        raise PhaseFailure("container_stdout", "PRECHECK_CONTAINER_STDOUT_EXTENT", **context)
    try:
        result = decode("container_stdout", raw)
    except PhaseFailure as error:
        error.details.update(context)
        raise
    if result.get("request") != request or result.get("step") != step or result.get("target_GPU_GO") is not False:
        raise PhaseFailure("container_result_identity", "PRECHECK_CONTAINER_RESULT_IDENTITY", **context)
    if rc != 0:
        failure = result.get("failure_type")
        gate = result.get("gate")
        phase = result.get("phase")
        child = {"phase": phase if isinstance(phase, str) and phase in DIAG_PHASES else None,
                 "type": failure if isinstance(failure, str) and failure.isidentifier() else None,
                 "gate": gate if isinstance(gate, str) and gate.isupper() and gate.isidentifier() else None}
        location = result.get("json_location")
        if type(location) is dict and set(location) == {"line", "column", "offset"} and all(
                type(v) is int and 0 <= v <= LOG_HASH_CAP for v in location.values()):
            child["json_location"] = location
        for key in ("input_bytes", "input_error_offset"):
            if type(result.get(key)) is int and 0 <= result[key] <= LOG_HASH_CAP:
                child[key] = result[key]
        value = result.get("input_SHA256")
        if isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value):
            child["input_SHA256"] = value
        cause = result.get("original_failure_type")
        if isinstance(cause, str) and cause.isidentifier():
            child["original_failure_type"] = cause
        raise PhaseFailure("container_exec", "PRECHECK_CONTAINER_EXIT", child_failure=child, **context)
    if result.get("failure_type") is not None:
        raise PhaseFailure("container_result_identity", "PRECHECK_CONTAINER_SUCCESS_CONTRADICTION", **context)
    return {**result, "container_logs": context}
