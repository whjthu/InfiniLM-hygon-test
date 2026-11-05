"""Recompute DeepSeek C1 W16 component contrasts and their model-budget context."""

import argparse
import json
import math
from pathlib import Path
import re
from statistics import mean

from component_probe import ARMS, FIXTURE_SHA, PASTS, digest, summarize
from device_probe import FORWARD, NATIVE, OBJECT, POSTLOAD, SOURCE
from down_only_probe import summarize as summarize_isolated

CLOCKS = ("routed_s", "drain_s", "capi_s")
ISOLATED_CLOCKS = ("stream_ms", "launch_and_wait_wall_ms")
REFERENCE_SHA = "40f4a69b6ba3bc50de6594b55f3e7a81ac5c7a773799506a544428edcdd3b6fb"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def sha(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def verify_rows(rows, fields):
    require([(row["trial"], row["mode"], row["past"]) for row in rows]
            == [(trial, mode, past) for trial, mode in enumerate(ARMS) for past in PASTS],
            "component trial order or census differs")
    for row in rows:
        require(all(positive(row[k]) for k in fields) and sha(row["output_SHA256"]),
                "invalid component clock or output digest")
        if fields == CLOCKS:
            require(row["routed_s"] <= row["drain_s"] <= row["capi_s"] and type(row["exact"]) is bool,
                    "routed clocks or numerical record differ")
        else:
            err = row["max_abs_vs_cpu"]
            require(type(row["cpu_exact"]) is bool and (err == 0 or positive(err)),
                    "isolated numeric record differs")


def contrast(owners, past, field, scale):
    maxima = [max(rows[i * len(PASTS) + PASTS.index(past)][field] for rows in owners.values()) * scale
              for i in range(2, len(ARMS))]
    savings = []
    for i in range(0, len(maxima), 2):
        a, b = maxima[i:i + 2]
        savings.append(a - b if ARMS[i + 2] == 0 else b - a)
    return dict(formal_max_rank_ms=maxima, adjacent_W32_minus_W16_ms=savings,
                mean_saved_ms=mean(savings), all_four_pairs_faster=all(value > 0 for value in savings))


def audit(report_raw, claim_raw, build_raw, reference_raw, step):
    require(re.fullmatch(r"\d+\.\d+", step), "expected exact Slurm step")
    report, claim, build = map(json.loads, (report_raw, claim_raw, build_raw))
    require(digest(reference_raw) == REFERENCE_SHA, "model budget reference differs")
    require(claim["GPU_GO"] is True and claim["isolated_down"] is True
            and claim["parent"] == step.split(".")[0] and claim["cards"] == 8
            and claim["fixture_SHA256"] == FIXTURE_SHA and sha(claim["device_probe_SHA256"])
            and claim["build_receipt_SHA256"] == digest(build_raw), "unbound component claim or build")
    require(report["completed"] is True and report["actual_step"] == step and report["node"] == claim["node"]
            and report["fixture_SHA256"] == FIXTURE_SHA and report["claim_SHA256"] == digest(claim_raw)
            and report["build_receipt_SHA256"] == digest(build_raw), "unbound component report")
    require((build["source_SHA256"], build["baseline_object_SHA256"], build["baseline_native_SHA256"])
            == (SOURCE, OBJECT, NATIVE) and sha(build["library_SHA256"]), "wrong native component build")
    require(report["providers"] == dict(deep_routed_w16_component_forward=build["library_SHA256"],
                deep_routed_down_w16_c1=build["library_SHA256"], deep_routed_dot_r20=NATIVE,
                deep_routed_validate_r20=NATIVE), "actual device providers differ")
    isolated = report["isolated_down"]
    require(sha(claim["down_only_probe_SHA256"]) and isolated["probe_SHA256"] == claim["down_only_probe_SHA256"],
            "isolated device probe differs")
    require(set(report["owners"]) == {str(i) for i in range(8)}, "incomplete eight-owner terminal set")
    owners, down = {}, {}
    for rank in range(8):
        owner = report["owners"][str(rank)]
        require(type(owner["rank"]) is int and owner["rank"] == rank and owner["normal_owner_closed"] is True
                and type(owner["destroy_RC"]) is int and owner["destroy_RC"] == 0 and "error_type" not in owner,
                "owner failed or did not close normally")
        postload, forward = owner["postload_free_bytes"], owner["minimum_forward_free_bytes"]
        require(type(postload) is int and type(forward) is int and postload >= POSTLOAD
                and FORWARD <= forward <= postload, "original component memory floor failed")
        owners[rank], down[rank] = owner["trials"], owner["isolated_down_trials"]
        verify_rows(owners[rank], CLOCKS)
        verify_rows(down[rank], ISOLATED_CLOCKS)
    require(summarize(owners) == report["paired_max_owner"]
            and summarize_isolated(down) == isolated["paired_max_owner"], "recorded component summary differs")
    cases = {str(past): dict(
        routed={key: contrast(owners, past, key, 1000) for key in CLOCKS},
        isolated={key: contrast(down, past, key, 1) for key in ISOLATED_CLOCKS},
    ) for past in PASTS}
    budget = json.loads(reference_raw)["measured_model_context"]
    target = budget["strict_15_visible_interval_budget_ms"]
    return dict(
        actual_step=step, report_sha256=digest(report_raw), claim_sha256=digest(claim_raw),
        build_receipt_sha256=digest(build_raw), normal_owners=8, cases=cases,
        consistent_isolated_event_gain=all(row["isolated"]["stream_ms"]["all_four_pairs_faster"] for row in cases.values()),
        consistent_routed_enclosure_gain=all(
            row["routed"][clock]["all_four_pairs_faster"] for row in cases.values() for clock in ("drain_s", "capi_s")),
        observed_other_positions_ms=budget["older_same_native_OFF_interval_gap_ms_range"],
        budget_ms_at_15_tok_s=target,
        required_strict_interval_savings_ms=[gap - target for gap in budget["older_same_native_OFF_interval_gap_ms_range"]],
        independent_slurm_and_post_verified=False,
        scope="Independent rank-slice component clocks at layer 3. Paired directions are descriptive. "
              "The model gaps come from other positions; no rank or layer sum, TP communication, "
              "actual model speedup, or device peak is inferred.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", required=True)
    for name in ("report", "claim", "build", "reference"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    result = audit(args.report.read_bytes(), args.claim.read_bytes(), args.build.read_bytes(),
                   args.reference.read_bytes(), args.step)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
