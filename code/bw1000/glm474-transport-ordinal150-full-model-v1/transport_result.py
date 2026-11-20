"""Consume the single causal request and keep private scores/token IDs on the cluster."""
import math
from transport_observer import digest

OUTPUT_CAP = 4096


def verify_request(request, pin, rank, arm):
    """Keep both transport arms and their complete outputs in the denominator."""
    ids = request["generated_token_ids"]
    if (type(ids) is not list or any(type(token) is not int or not 0 <= token < 154880 for token in ids)
            or not 0 < len(ids) <= OUTPUT_CAP
            or digest(ids) != request["generated_token_ids_SHA256"]
            or not request["completed"] or not request["reset_before"] or not request["reset_after"]
            or len(request["sampled_ns"]) != len(ids)
            or len(request["decode_CAPI_s"]) != len(ids) - 1
            or request["planned_max_new_tokens"] != OUTPUT_CAP
            or (request["cap_hit"] and len(ids) != OUTPUT_CAP)
            or bool(request["natural_EOS"]) == bool(request["cap_hit"])):
        raise ValueError("incomplete or changed full decode boundary")
    if not all(request[k] == pin[k] for k in ("ordinal", "prompt_tokens", "prompt_IDs_SHA256", "body_SHA256")):
        raise ValueError("request is not the exact original ordinal")
    if rank != 0:
        if "transport_rank0_sample43" in request:
            raise ValueError("nonleader produced rank-0 capture")
        return None

    if not isinstance(request.get("score"), dict) or type(request["score"].get("correct")) is not bool:
        raise ValueError("rank-0 request score missing")
    capture = request["transport_rank0_sample43"]
    state = capture["state"]
    scalar = dict(state=state, generated_tokens=len(ids), output_SHA256=digest(ids),
                  natural_EOS=request["natural_EOS"], cap_hit=request["cap_hit"],
                  correct=request["score"]["correct"])
    if state == "NO_SAMPLE":
        if len(ids) >= pin["sample"] or capture["observed_tokens"] != len(ids):
            raise ValueError("missing sample overwritten")
        return scalar
    if len(ids) < pin["sample"]:
        raise ValueError("sample recorded after the last observed token")
    prefix = digest(ids[:pin["prefix_len"]])
    choice = digest(ids[pin["prefix_len"]:pin["sample"]])
    if (capture["sample"] != pin["sample"] or capture["past"] != pin["prompt_tokens"] + pin["prefix_len"]
            or capture["prefix_SHA256"] != prefix or capture["chosen_SHA256"] != choice
            or capture["historical_argmax_repeated"] != (choice == pin["historic_argmax_SHA256"][arm])):
        raise ValueError("sample 43 does not match observed sequence")
    scalar.update(prefix_SHA256=prefix, chosen_SHA256=choice,
                  historical_argmax_repeated=capture["historical_argmax_repeated"])
    if state == "PREFIX_MISMATCH" and prefix != pin["common_prefix_SHA256"]:
        return scalar
    if state != "MATCHED_PREFIX" or prefix != pin["common_prefix_SHA256"]:
        raise ValueError("evidence state did not follow the actual prefix")
    top = capture["top_tokens_private"]
    margin = capture["top1_minus_top2"]
    if (len(top) != 8 or len({row[0] for row in top}) != 8 or top[0][0] != ids[pin["prefix_len"]]
            or any(type(t) is not int or not 0 <= t < 154880 or not math.isfinite(float(v)) for t, v in top)
            or [tuple(row) for row in top] != sorted((tuple(row) for row in top), key=lambda row: (-row[1], row[0]))
            or margin != top[0][1] - top[1][1]):
        raise ValueError("rank-0 logit margin is inconsistent")
    raw_sha = capture["logits_SHA256"]
    if not isinstance(raw_sha, str) or len(raw_sha) != 64 or any(c not in "0123456789abcdef" for c in raw_sha):
        raise ValueError("logit digest missing")
    scalar.update(logits_SHA256=raw_sha, top1_minus_top2=margin)
    return scalar


def compare(arms):
    if set(arms) != {"socket", "shca"}:
        raise ValueError("both original transport arms required")
    a, b = arms["socket"], arms["shca"]
    paired = a["state"] == b["state"] == "MATCHED_PREFIX"
    return dict(arms=arms, paired_on_historical_prefix=paired,
                sample43_choice_equal=(a["chosen_SHA256"] == b["chosen_SHA256"]) if paired else None,
                full_output_equal=a["output_SHA256"] == b["output_SHA256"],
                historical_quality_regression_recreated=(a["correct"] and not b["correct"]),
                causality="Transport selection can alter the ordering of reductions or propagate earlier differences. This comparison localizes the decision and its margins; layer isolation requires additional evidence.",
                production_promotion=False, formal_inference_pass=False)
