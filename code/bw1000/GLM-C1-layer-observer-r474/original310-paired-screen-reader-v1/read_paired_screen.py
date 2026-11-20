"""Audit original GLM310 screen pairs after the independent terminal reader."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import median

RUN = "12d5bc02ca0a49a89db5c1628c64160b"
STEP = "1103606.310"
OWNER = f"GLM474-depth-observer-model-{RUN}-socket/owner-00.json"
ARMS = ("OFF", "ON")
SUITES = ("gsm8k", "mmlu")
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
            "NODE_FAIL", "BOOT_FAIL", "DEADLINE", "PREEMPTED", "REVOKED"}
INPUT_SHA = "81f6b2d68228aafb79993306847b0edb04c1ba101a35ab21bb822ac96ca2ba5e"
SCORER_SHA = "c9c39a5fc185ed6bfc5f48c2b547d4ef36a4cfed351626835793fcf0ab5b521c"
IDENTITY = ("ordinal", "case_id", "suite", "prompt_tokens", "prompt_IDs_SHA256", "body_SHA256")


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(value):
    return hashlib.sha256(value).hexdigest()


def load(path, maximum, digest=None):
    with path.open("rb") as stream:
        raw = stream.read(maximum + 1)
    require(0 < len(raw) <= maximum and (digest is None or sha(raw) == digest),
            f"Original file failed byte binding: {path.name}")
    return raw


def percentile(values, fraction):
    return sorted(values)[math.ceil(fraction * len(values)) - 1] if values else None


def occurrence(row, pin, plan, eos, special, grade):
    require(all(row[key] == pin[key] for key in IDENTITY), "wrong frozen input")
    require(all(row[key] is True for key in ("completed", "reset_before", "reset_after")),
            "completed occurrence missing a reset")
    ids, ticks, enter = (row[key] for key in ("generated_token_ids", "sampled_ns", "request_enter_ns"))
    require(row["planned_max_new_tokens"] == 4096 and 0 < len(ids) == len(ticks) <= 4096
            and all(type(t) is int and 0 <= t < 154880 for t in ids)
            and not eos.intersection(ids[:-1])
            and sha(json.dumps(ids, separators=(",", ":")).encode()) == row["generated_token_ids_SHA256"],
            "inconsistent generated token sequence")
    natural = ids[-1] in eos
    cap = not natural and len(ids) == 4096
    require((natural or cap) and row["natural_EOS"] is natural and row["cap_hit"] is cap,
            "invalid EOS or cap")
    require(type(enter) is int and enter > 0 and all(type(tick) is int and tick >= enter for tick in ticks)
            and all(a < b for a, b in zip(ticks, ticks[1:])), "invalid backend timeline")
    visible = [tick for token, tick in zip(ids, ticks) if token not in special]
    ttft = (visible[0] - enter) / 1e9 if visible else None
    rate = (len(visible) - 1) * 1e9 / (visible[-1] - visible[0]) if len(visible) > 1 else None
    require(row["visible_nonEOS_nonspecial"] == len(visible)
            and row["backend_sampled_TTFT_s"] == ttft and row["backend_decode_tok_s"] == rate,
            "independently recomputed clocks differ")
    elapsed = row["whole_backend_s"]
    require(type(elapsed) in (int, float) and math.isfinite(elapsed)
            and elapsed >= (ticks[-1] - enter) / 1e9, "whole backend time too short")
    score = grade(plan["case"], row["output_text"], natural_EOS=natural, cap_hit=cap)
    require(score == row["score"], "independently recomputed score differs")
    gaps = [(b - a) / 1e9 for a, b in zip(visible, visible[1:])]
    return dict(correct=score["correct"], raw_answer_matches_gold=score["answer_matches_gold"],
                parse_failure=score["parse_failure"], natural_EOS=natural, cap_hit=cap,
                visible_tokens=len(visible), backend_TTFT_s=ttft, backend_decode_tok_s=rate,
                backend_ITL_P95_s=percentile(gaps, .95), backend_ITL_P99_s=percentile(gaps, .99))


def match_occurrences(requests, pins, plans, eos, special, grade):
    require(len(requests) <= 66, "wrong planned request denominator")
    sorted_cases = [dict(ordinal=pin["ordinal"], suite=pin["suite"], OFF=None, ON=None) for pin in pins]
    diagnostic_completed = dict.fromkeys(ARMS, False)
    incomplete = False
    for i, row in enumerate(requests):
        pair, arm = i // 2, ARMS[i % 2]
        j = max(0, pair - 1)
        require(type(row["completed"]) is bool and row["arm"] == arm and row["pair"] == pair
                and row["original_diagnostic"] is (pair == 0)
                and row["diagnostic"] is (pair == 0 and arm == "ON")
                and row["screen_index"] == (j if pair else None)
                and all(row[key] == pins[j][key] for key in IDENTITY),
                "occurrence violates frozen alternating order")
        if row["completed"]:
            require(not incomplete, "completed request after an incomplete request")
            result = occurrence(row, pins[j], plans[j], eos, special, grade)
            if pair:
                sorted_cases[j][arm] = result
            else:
                diagnostic_completed[arm] = True
        else:
            incomplete = True
    return sorted_cases, diagnostic_completed


def summarize(cases, wilson):
    n = len(cases)
    sides = {}
    for arm in ARMS:
        seen = [row[arm] for row in cases if row[arm] is not None]
        rates = [row["backend_decode_tok_s"] for row in seen if row["backend_decode_tok_s"] is not None]
        correct = sum(row["correct"] for row in seen)
        sides[arm] = dict(planned=n, complete=len(seen), missing=n-len(seen), correct=correct,
                          planned_accuracy=correct/n, planned_accuracy_Wilson95=wilson(correct, n),
                          missing_outcome_accuracy_bounds=[correct/n, (correct+n-len(seen))/n],
                          raw_answer_matches_gold=sum(row["raw_answer_matches_gold"] for row in seen),
                          parse_failures=sum(row["parse_failure"] for row in seen),
                          capped=sum(row["cap_hit"] for row in seen),
                          measured_backend_decode=len(rates), strict_backend_decode_gt15=sum(r > 15 for r in rates),
                          median_backend_decode_tok_s=median(rates) if rates else None)
    complete = [row for row in cases if all(row[arm] is not None for arm in ARMS)]
    gain = sum(row["ON"]["correct"] and not row["OFF"]["correct"] for row in complete)
    loss = sum(row["OFF"]["correct"] and not row["ON"]["correct"] for row in complete)
    lower, upper = sides["ON"]["missing_outcome_accuracy_bounds"]
    off_lower, off_upper = sides["OFF"]["missing_outcome_accuracy_bounds"]
    count = len(complete)
    delta = (gain-loss)/count if count else None
    radius = math.sqrt(2*math.log(40)/count) if count else None
    ratio = [
        row["ON"]["backend_decode_tok_s"] / row["OFF"]["backend_decode_tok_s"]
        for row in complete
        if all(row[arm]["backend_decode_tok_s"] is not None and row[arm]["backend_decode_tok_s"] > 0 for arm in ARMS)
    ]
    return dict(arms=sides, expected_pairs=n, complete_pairs=count, ON_only_correct=gain, OFF_only_correct=loss,
                observed_ON_minus_OFF=delta,
                observed_pair_Hoeffding95=[max(-1, delta-radius), min(1, delta+radius)] if count else None,
                planned_ON_minus_OFF_missing_outcome_bounds=[lower-off_upper, upper-off_lower],
                timed_pairs=len(ratio), median_ON_over_OFF_backend_rate=median(ratio) if ratio else None)


def audit(terminal_raw, out, source, private_plan):
    terminal = json.loads(terminal_raw)
    gate = terminal["actual_accounting"]["gate"]
    require(terminal["run_id"] == RUN and terminal["original_request_id"] == "5d9c77b0b9004989941fd73eb09b03fe"
            and terminal["claim_commit"] == "9fe9496f286ea293cc55c843c8ba68c92b49c458"
            and terminal["planned_owner_ranks"] == 16 and terminal["planned_requests_per_rank"] == 66
            and type(terminal["launcher_rc"]) is int and len(gate) == 7 and gate[0] == STEP
            and gate[1] == out.name + "-gate" and out.name == Path(OWNER).parent.name
            and gate[2].split()[0].rstrip("+") in TERMINAL, "original terminal evidence required")
    owner_raw = load(out/"owner-00.json", 128*1024**2, terminal["raw_artifact_SHA256"][OWNER])
    owner = json.loads(owner_raw)
    require(owner["global_rank"] == 0 and owner["actual_GPU"] == STEP
            and owner["Native_SHA256"] == "ddfeae81526758fb06d68e4f688e9eeaa48e3b173018d0040a3de0c99ba8dfb5",
            "original rank-zero evidence differs")
    binding = json.loads(load(source/"INPUT_BINDING.json", 65536, INPUT_SHA))
    pin = binding["private_plan"]
    raw_plan = load(private_plan, pin["bytes"], pin["SHA256"])
    pins, plans = pin["requests"], json.loads(raw_plan)["requests"]
    require(len(raw_plan) == pin["bytes"] and len(pins) == len(plans) == len({r["case_id"] for r in pins}) == 32
            and [r["ordinal"] for r in pins] == binding["selected_ordinals"]
            and all(sum(r["suite"] == suite for r in pins) == 16 for suite in SUITES)
            and all(all(plan[k] == v for k, v in expected.items()) and plan["case"]["suite"] == expected["suite"]
                    for expected, plan in zip(pins, plans)), "original private plan differs")
    scoring = {}
    exec(compile(load(source/"scoring.py", 32768, SCORER_SHA), "scoring.py", "exec"), scoring)
    eos = set(binding["eos_token_ids"])
    special = eos | set(binding["tokenizer"]["special_ids"])
    cases, diagnostic = match_occurrences(owner["requests"], pins, plans, eos, special, scoring["grade"])
    return dict(run_id=RUN, gpu_step=STEP, gate_state=gate[2], original_launcher_rc=terminal["launcher_rc"],
                independent_post_clean=terminal["independent_post_clean"],
                original_all_rank_validation=terminal["valid_ffn_evidence"],
                parent_terminal_SHA256=sha(terminal_raw), owner_SHA256=sha(owner_raw), plan_SHA256=pin["SHA256"],
                diagnostic_completed=diagnostic,
                overall=summarize(cases, scoring["wilson"]),
                by_suite={suite: summarize([r for r in cases if r["suite"] == suite], scoring["wilson"])
                          for suite in SUITES},
                cases=cases, timing_scope="Same-owner backend sampled token clocks. One diagnostic occurrence per arm is excluded.",
                broad_quality_qualified=False, client_SLO_qualified=False, noninferiority_proven=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("terminal", "out", "source", "private_plan", "report"):
        parser.add_argument(arg, type=Path)
    args = parser.parse_args()
    result = audit(load(args.terminal, 16*1024**2), args.out, args.source, args.private_plan)
    fd = os.open(args.report, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
