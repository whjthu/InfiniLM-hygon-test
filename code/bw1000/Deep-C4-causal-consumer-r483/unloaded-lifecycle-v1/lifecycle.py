"""Observe the actual unweighted C4 owner and its failure/cleanup boundaries."""

import ctypes as C
import hashlib
import json
from pathlib import Path

from checkpoint import config, validate_parameters
from input_audit import check_before_load


FLOOR = 16 * 1024**3


def require(ok, message):
    if not ok:
        raise ValueError(message)


def error(api):
    value = api["error"]()
    require(value, "missing C ABI error")
    return C.string_at(value).decode()


def verify_unloaded(handle, api, binding, claim, package):
    path = Path(package) / "CHECKPOINT_METADATA.json"
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == claim["config_metadata_SHA256"], "model metadata identity differs")
    metadata = json.loads(raw)
    require(metadata["layer"] == 2, "unexpected C4 layer")
    cfg = config(metadata)
    trial = binding["Trial"](handle, cfg, metadata["layer"], 0)
    pointer = trial.handle
    negatives = []
    try:
        parameters = trial.read("parameters")
        validate_parameters(parameters, metadata)
        state = trial.read("observe")
        check_before_load(state)

        other = C.c_ubyte()
        require(not api["observe"](C.cast(C.pointer(other), C.c_void_p))
                and error(api) == "unknown owner handle", "foreign handle was not rejected")
        negatives.append("foreign_handle")
        payload = json.dumps(cfg, separators=(",", ":")).encode()
        require(not api["create"](payload, metadata["layer"], 0)
                and error(api) == "one fixed startup phase per process", "second owner was not rejected")
        negatives.append("second_owner")
        check_before_load(trial.read("observe"))

        require(api["postload"](pointer) == 1
                and error(api) == "complete actual checkpoint parameter set required", "incomplete postload was not rejected")
        negatives.append("unloaded_postload")
        require(not api["observe"](pointer) and error(api) == "wrong/poisoned owner", "failed owner stayed usable")
        negatives.append("poisoned_owner")
    finally:
        closed = trial.close()
    require(len(closed) == 4 and min(closed[:2]) >= FLOOR and closed[2:] == [1, 1],
            "guarded owner destruction or memory floor differs")
    require(not api["observe"](pointer) and error(api) == "unknown owner handle", "retired handle remained valid")
    negatives.append("retired_handle")
    return {
        "actual_native_runtime_created": True, "local_heads": state["local_heads"],
        "full_context": state["full_context"], "compressed_capacity": state["compressed_capacity"],
        "matched_unloaded_parameters": len(parameters),
        "parameter_metadata_SHA256": hashlib.sha256(
            json.dumps(parameters, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "verified_rejections": negatives,
        "driver_bytes_after_create": state["driver_free"],
        "driver_free_before_after_close": closed[:2],
        "weak_owner_expired": True, "checkpoint_weight_bytes_read": 0,
        "successful_postloads": 0, "causal_requests_executed": 0,
    }
