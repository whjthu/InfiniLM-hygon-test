"""Consume only fixed serial request results and compare matching decode work."""

import math
from statistics import median

from alias_policy import AUDIT_FORWARDS, SCHEDULE, expected_counts
from pilot_inputs import require


def summarize(owners, results):
    require(len(owners) == len(results) == len(SCHEDULE), "incomplete history alias screen")
    for index, (owner, result, (_, mode)) in enumerate(zip(owners, results, SCHEDULE)):
        stats = owner["history_alias"]
        require(owner["r454_index"] == index and stats["mode"] == mode
                and stats["counters"] == [expected_counts(owner["Native_continuation"], mode)] * 8
                and stats["counters"][0][1] > 0 and result["native_forwards"] * 21 == stats["counters"][0][0],
                "independent C4 coverage differs")
    audit = owners[1]["history_alias"]["logits_audit"]
    require(2 <= len(audit) <= AUDIT_FORWARDS
            and all(math.isfinite(x["max_absolute"]) and math.isfinite(x["rms"]) for x in audit),
            "invalid fixed-input audit")
    pairs = []
    for offset in (2, 4, 6):
        off, on = sorted(range(offset, offset + 2), key=lambda i: SCHEDULE[i][1])
        a, b = results[off], results[on]
        require(a["case_id"] == b["case_id"], "unmatched request pair")
        same = a["generated_IDs_SHA256"] == b["generated_IDs_SHA256"]
        eos = a["score"]["natural_EOS"] and b["score"]["natural_EOS"]
        a_rate, b_rate = a["backend_non_eos_decode_tok_s"], b["backend_non_eos_decode_tok_s"]
        ratio = b_rate / a_rate if same and eos and a_rate and b_rate else None
        pairs.append(dict(off=off, on=on, case_id=a["case_id"], same_generated_ids=same,
                          both_natural_EOS=eos, off_tok_s=a_rate, on_tok_s=b_rate,
                          on_over_off=ratio))
    ratios = [x["on_over_off"] for x in pairs if x["on_over_off"] is not None]
    return dict(requests=len(results), correct=sum(x["score"]["correct"] for x in results),
                natural_EOS=sum(x["score"]["natural_EOS"] for x in results),
                generated_ids_match_in_audit_pair=results[0]["generated_IDs_SHA256"] == results[1]["generated_IDs_SHA256"],
                audit_first_forward_limit=AUDIT_FORWARDS, audit_matched_forwards=len(audit),
                audit_bitwise_equal_forwards=sum(x["differing_BF16_values"] == 0 for x in audit),
                audit_max_absolute=max(x["max_absolute"] for x in audit),
                audit_rows=audit, timing_pairs=pairs,
                valid_timing_pairs=len(ratios), median_on_over_off=median(ratios) if len(ratios) == 3 else None,
                stage="two-case serial engineering screen", broad_quality_qualified=False,
                client_per_token_decode_qualified=False, promote=False)
