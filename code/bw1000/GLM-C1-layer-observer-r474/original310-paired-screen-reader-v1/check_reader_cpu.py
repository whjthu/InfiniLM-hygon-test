"""Exercise the paired reader with the original schedule and frozen scorer."""
import json
from pathlib import Path

from read_paired_screen import (
    ARMS, IDENTITY, OWNER, RUN, SCORER_SHA, STEP, audit, load, match_occurrences, sha, summarize,
)


def main():
    source = Path(__file__).resolve().parent.parent/"full-model-depth-stratified-observer-v1"
    scorer = {}
    exec(compile(load(source/"scoring.py", 32768, SCORER_SHA), "scoring.py", "exec"), scorer)
    frozen = json.loads((source/"INPUT_BINDING.json").read_bytes())
    pins = frozen["private_plan"]["requests"]
    key = {"gsm8k": "gold_numeric", "mmlu": "gold_choice"}
    gold = {"gsm8k": "42", "mmlu": "A"}
    correct = {"gsm8k": "#### 42", "mmlu": "Answer: A"}
    plans = [{"case": {"suite": p["suite"], key[p["suite"]]: gold[p["suite"]]}} for p in pins]

    def row(i, *, text=None, ids=None, ticks=None, complete=True, arm_override=None):
        pair, arm = i//2, ARMS[i%2]
        j = max(0, pair-1)
        pin = pins[j]
        r = {k: pin[k] for k in IDENTITY}
        r.update(arm=arm_override or arm, pair=pair, original_diagnostic=(pair==0),
                 diagnostic=(pair==0 and arm=="ON"), screen_index=(j if pair else None),
                 completed=complete)
        if not complete:
            return r
        if ids is None:
            ids = [1, 2, 3, 4, 0]
        if ticks is None:
            ticks = [100, 60_000_100, 120_000_100, 200_000_100, 210_000_100]
        eos = ids[-1] == 0
        visible = ticks[:-1] if eos else ticks
        text = correct[pin["suite"]] if text is None else text
        return dict(r, reset_before=True, reset_after=True, planned_max_new_tokens=4096,
                    request_enter_ns=1, sampled_ns=ticks, generated_token_ids=ids,
                    generated_token_ids_SHA256=sha(json.dumps(ids, separators=(",", ":")).encode()),
                    natural_EOS=eos, cap_hit=not eos, visible_nonEOS_nonspecial=len(visible),
                    backend_sampled_TTFT_s=(visible[0]-1)/1e9 if visible else None,
                    backend_decode_tok_s=(len(visible)-1)*1e9/(visible[-1]-visible[0]) if len(visible)>1 else None,
                    whole_backend_s=1, output_text=text,
                    score=scorer["grade"](plans[j]["case"], text, natural_EOS=eos, cap_hit=not eos))

    def evaluate(rows):
        return match_occurrences(rows, pins, plans, {0}, {0}, scorer["grade"])

    data = [row(i) for i in range(66)]
    cases, diag = evaluate(data)
    full = summarize(cases, scorer["wilson"])
    assert diag == {"OFF": True, "ON": True}
    assert full["complete_pairs"] == 32 and full["median_ON_over_OFF_backend_rate"] == 1
    assert all(v["correct"] == 32 and v["strict_backend_decode_gt15"] == 0 for v in full["arms"].values())
    assert all(c["OFF"]["backend_decode_tok_s"] == 15 for c in cases)
    data = [row(0), row(1), row(2, text=""), row(3), row(4, complete=False)]
    cases, diag = evaluate(data)
    partial = summarize(cases, scorer["wilson"])
    assert (partial["expected_pairs"], partial["complete_pairs"], partial["ON_only_correct"]) == (32, 1, 1)
    assert partial["arms"]["OFF"]["parse_failures"] == 1 and partial["arms"]["OFF"]["missing"] == 31
    assert partial["planned_ON_minus_OFF_missing_outcome_bounds"] == [-30/32, 1.0]
    cases, _ = evaluate([row(0), row(1), row(2, ids=[0], ticks=[100]),
                         row(3, ids=[1]*4096, ticks=list(range(100, 4196)))])
    assert cases[0]["OFF"]["backend_decode_tok_s"] is None
    assert cases[0]["ON"]["cap_hit"] and not cases[0]["ON"]["correct"]
    bad = [
        [dict(row(0), diagnostic=True)],
        [row(0), row(1), row(2, complete=False), row(3)],
        [row(0), row(1), row(2, arm_override="ON")],
        [row(0), row(1), dict(row(2), backend_decode_tok_s=16)],
        [row(0), row(1), dict(row(2), sampled_ns=[100]*5)],
        [row(0), row(1), dict(row(2), score={})],
        [row(0), row(1), dict(row(2), reset_after=False)],
        [row(0), row(1), dict(row(2), prompt_IDs_SHA256="0"*64)],
        [row(0), row(1), dict(row(2), generated_token_ids_SHA256="0"*64)],
    ]
    for attempt in bad:
        try:
            evaluate(attempt)
        except ValueError:
            continue
        raise AssertionError("invalid evidence accepted")
    label = Path(OWNER).parent.name
    running = dict(run_id=RUN, original_request_id="5d9c77b0b9004989941fd73eb09b03fe",
                   claim_commit="9fe9496f286ea293cc55c843c8ba68c92b49c458", planned_owner_ranks=16,
                   planned_requests_per_rank=66, launcher_rc=0,
                   actual_accounting={"gate": [STEP, label+"-gate", "RUNNING", "0:0", "1", "", ""]})
    try:
        audit(json.dumps(running).encode(), Path(label), source, Path("unused"))
    except ValueError as exc:
        assert str(exc) == "original terminal evidence required"
    else:
        raise AssertionError("live accounting accepted")
    print(json.dumps({"complete": 32, "partial": 1, "caps_retained": 1, "rejections": len(bad),
                      "live_gate_rejected": True, "native_calls": 0, "GPU_steps": 0}))


if __name__ == "__main__":
    main()
