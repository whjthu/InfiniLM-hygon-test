"""Observe first-Ops affinity correctly for an admitted-request-free service.

Active inference retains the original strict verifier unchanged. No affinity is
set here. In the original submission source RuntimeWorkspaceHandle calls
apply_once(); apply_once increments changes only after actual OS readback.
An idle service has no model submissions and may remain entirely unbound.
"""
import ctypes
import json


def verify_service_affinity(lib, report, native_forward_calls, strict_verify):
    requests = report.get("agent_requests")
    if not isinstance(requests, list):
        raise AssertionError("Missing explicit service request ledger")
    if requests:
        return strict_verify(lib)
    if native_forward_calls != 0:
        raise AssertionError("Zero request ledger contradicts actual Native forward ledger")
    lib.modern_owner_affinity_report_r2392.restype = ctypes.c_char_p
    actual = json.loads(lib.modern_owner_affinity_report_r2392())
    owners = actual["owners"]
    assert len(owners) == 8 and {x["device"] for x in owners} == set(range(8))
    assert all(x["changes"] == 0 and x["verified_mask"] == "" for x in owners)
    return {"verified": True, "scope": "idle service observation only",
            "classification": "IDLE_ZERO_REQUESTS_NO_MODEL_FORWARD_NOT_PERFORMANCE_QUALIFIED",
            "requests": 0, "native_forward_calls": 0,
            "actual_owner_report": actual,
            "first_Ops_hook_observed": False,
            "active_inference_affinity_qualified": False,
            "source_boundary": "RuntimeWorkspaceHandle constructor -> apply_once; no hand pinning"}
