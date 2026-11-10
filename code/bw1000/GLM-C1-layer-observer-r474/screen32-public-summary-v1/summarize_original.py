"""Build a bounded public summary from the pinned private GLM906 audit."""
import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import re

BINDING_SHA = "81f6b2d68228aafb79993306847b0edb04c1ba101a35ab21bb822ac96ca2ba5e"
READER_SHA = "a9431a8c8071335931552cac815ed2954d8237513c57cc0881fb372d168ada61"
GPU = "910371.906"
KEEPER = "910407.1121"
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY",
            "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}


def digest(data):
    return sha256(data).hexdigest()


def read_pinned(data, expected):
    if not re.fullmatch(r"[0-9a-f]{64}", expected) or digest(data) != expected:
        raise ValueError("GLM906 evidence SHA256 differs")
    return json.loads(data)


def terminal(row, job):
    state = row["normalized_state"]
    exit_code = row["ExitCode"]
    elapsed = row["ElapsedRaw"]
    if (row["JobIDRaw"] != job or state not in TERMINAL or row["State"].split()[0].rstrip("+") != state
            or not re.fullmatch(r"[0-9]+:[0-9]+", exit_code) or type(elapsed) is not int or elapsed < 0):
        raise ValueError("original GLM906 terminal accounting required")
    return dict(job=job, state=state, exit_code=exit_code, elapsed_s=elapsed)


def read_flag(record, name):
    value = record[name]
    if type(value) is not bool:
        raise ValueError(f"GLM906 audit flag differs: {name}")
    return value


def summarize(audit_bytes, expected_sha, binding_bytes):
    audit = read_pinned(audit_bytes, expected_sha)
    binding = read_pinned(binding_bytes, BINDING_SHA)
    plan = binding["private_plan"]
    if (audit["schema"] != "GLM906-original-screen32-root-v1" or audit["reader_SHA256"] != READER_SHA
            or audit["GPU"] != GPU or audit["keeper"] != KEEPER
            or audit["private_plan_SHA256"] != plan["SHA256"]
            or (audit["planned_requests"], audit["unique_screen_planned"], audit["diagnostic_planned"]) != (33, 32, 1)):
        raise ValueError("unexpected GLM906 audit identity")
    gpu = terminal(audit["GPU_accounting"], GPU)
    keeper = terminal(audit["keeper_accounting"], KEEPER)
    gates = {name: read_flag(audit, name) for name in (
        "GPU_accounting_success", "keeper_accounting_success", "keeper_original_producer_passed",
        "engineering_full_entry_passed", "all16_original_owner_validation_passed", "post_clean",
    )}
    for name, row in (("GPU_accounting_success", gpu), ("keeper_accounting_success", keeper)):
        if gates[name] != (row["state"] == "COMPLETED" and row["exit_code"] == "0:0"):
            raise ValueError("GLM906 accounting flags contradict terminal rows")
    rows = {}
    diagnostic_seen = False
    for row in audit["root_requests"]:
        index = row["index"]
        if read_flag(row, "diagnostic"):
            if diagnostic_seen or index != 0 or row["screen_index"] is not None:
                raise ValueError("unexpected GLM906 diagnostic occurrence")
            diagnostic_seen = True
            continue
        if type(index) is not int or not 1 <= index <= 32 or index in rows:
            raise ValueError("invalid or duplicate GLM906 screen slot")
        bound = plan["requests"][index - 1]
        if (row["screen_index"], row["ordinal"], row["suite"]) != (index, bound["ordinal"], bound["suite"]):
            raise ValueError("GLM906 screen slot differs from the frozen input binding")
        fields = {name: read_flag(row, name) for name in ("correct", "parse_failure", "natural_EOS", "cap_hit")}
        rate = row["backend_decode_tok_s"]
        if rate is not None and (type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0):
            raise ValueError("invalid GLM906 backend decode rate")
        rows[index] = dict(**fields, rate=rate)
    if gates["engineering_full_entry_passed"] and (not all(gates.values()) or len(rows) != 32 or not diagnostic_seen):
        raise ValueError("incomplete GLM906 audit marked as an engineering pass")
    suites = {}
    for suite in ("gsm8k", "mmlu"):
        planned = [i for i, row in enumerate(plan["requests"], 1) if row["suite"] == suite]
        validated = [rows[i] for i in planned if i in rows]
        suite_counts = dict(planned=len(planned), independently_validated=len(validated),
                            missing_or_unvalidated=len(planned) - len(validated))
        for field in ("correct", "parse_failure", "natural_EOS", "cap_hit"):
            suite_counts[field] = sum(row[field] for row in validated)
        rates = [row["rate"] for row in validated if row["rate"] is not None]
        suite_counts.update(backend_decode_rate_available=len(rates), backend_strict_gt15=sum(rate > 15 for rate in rates))
        suites[suite] = suite_counts
    counts = {name: sum(s[name] for s in suites.values()) for name in suites["gsm8k"]}
    expected_counts = dict(planned=counts["planned"], validated=counts["independently_validated"],
                           correct=counts["correct"], parse_failure=counts["parse_failure"],
                           natural_EOS=counts["natural_EOS"], cap_hit=counts["cap_hit"],
                           missing_or_unvalidated=counts["missing_or_unvalidated"],
                           strict_gt15=counts["backend_strict_gt15"])
    if audit["screen32"] != expected_counts:
        raise ValueError("GLM906 reader totals differ from the 32 planned slots")
    return dict(schema="GLM906-public-screen32-summary-v1", original_private_audit_SHA256=expected_sha,
                input_binding_SHA256=BINDING_SHA, reader_SHA256=READER_SHA, gpu=gpu, keeper=keeper,
                audit_gates=gates, screen32=counts, suites=suites, diagnostic_validated=diagnostic_seen,
                performance_baseline_paired=False, wide_quality_qualified=False,
                client_SLO_qualified=False, fleet_busy_measured=False, promote=False)


def bounded_read(path, limit):
    with path.open("rb") as file:
        data = file.read(limit + 1)
    if len(data) > limit:
        raise ValueError("GLM906 summary input exceeds its read limit")
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-audit", type=Path, required=True)
    parser.add_argument("--private-audit-sha", required=True)
    parser.add_argument("--binding", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(bounded_read(args.private_audit, 32 * 1024**2), args.private_audit_sha,
                       bounded_read(args.binding, 1024**2))
    print(json.dumps(result, sort_keys=True, allow_nan=False))
