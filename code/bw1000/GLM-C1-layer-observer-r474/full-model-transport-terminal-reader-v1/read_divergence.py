"""Locate the original GLM transport output divergence without exporting tokens."""
import hashlib
import json
from pathlib import Path

from read_decode_cost import AUDIT_SHA, BINDING_SHA, GATES, LABEL, read


def first_divergence(left, right):
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right)) if len(left) != len(right) else None


def output(request, pin):
    for key in ("ordinal", "case_id", "suite", "prompt_IDs_SHA256", "body_SHA256"):
        if request[key] != pin[key]:
            raise ValueError("request differs from the frozen input")
    if any(request.get(k) is not True for k in ("completed", "reset_before", "reset_after")):
        raise ValueError("request did not complete and reset")
    ids = request["generated_token_ids"]
    digest = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
    if not 0 < len(ids) <= 4096 or digest != request["generated_token_ids_SHA256"]:
        raise ValueError("incorrect output inventory or hash")
    return ids


def owner_screen(root, arm, summary, pins, rank):
    owner = read(root / (LABEL + "-" + arm) / f"owner-{rank:02d}.json",
                 summary["owners"][str(rank)]["SHA256"], 96 << 20)
    if (owner["global_rank"], owner["actual_GPU"], owner["Native_arm"]) != (rank, GATES[arm], arm):
        raise ValueError("original owner identity differs")
    requests = [r for r in owner["requests"] if r["diagnostic"] is False]
    if len(requests) != len(pins) or len(owner["requests"]) != len(pins) + 1:
        raise ValueError("original screen denominator differs")
    if set(pins) != {r["ordinal"] for r in requests}:
        raise ValueError("missing or foreign ordinal")
    return {r["ordinal"]: (output(r, pins[r["ordinal"]]), r) for r in requests}


def audit(root):
    original = read(root / (LABEL + "-independent-terminal-audit-v1.json"), AUDIT_SHA, 65536)
    if original["complete_engineering_pair"] is not True:
        raise ValueError("original transport audit did not pass")
    binding = read(root / (LABEL + "-dispatch") / "src" / "INPUT_BINDING.json", BINDING_SHA, 1 << 20)
    pins = {r["ordinal"]: r for r in binding["private_plan"]["requests"]}
    ranks = (0, 8)
    first_rank = {}
    for arm, step in GATES.items():
        summary = original["arms"][arm]
        if summary["accounting"]["gate"][:4] != [step, LABEL + "-" + arm + "-gate", "COMPLETED", "0:0"]:
            raise ValueError("unexpected transport gate")
        a, b = [owner_screen(root, arm, summary, pins, rank) for rank in ranks]
        if any(a[key][0] != b[key][0] for key in pins):
            raise ValueError("two-node output mismatch within one arm")
        first_rank[arm] = a
    changes = []
    for ordinal, pin in pins.items():
        (left, base), (right, cand) = (first_rank[arm][ordinal] for arm in GATES)
        index = first_divergence(left, right)
        if index is None:
            continue
        changes.append(dict(ordinal=ordinal, suite=pin["suite"], prompt_tokens=pin["prompt_tokens"],
                            common_generated_prefix_tokens=index,
                            first_difference_phase="prefill_logits" if index == 0 else "autoregressive_logits",
                            decode_forwards_before_first_difference=index,
                            generated_tokens={"socket": len(left), "shca": len(right)},
                            correct={"socket": base["score"]["correct"], "shca": cand["score"]["correct"]},
                            natural_EOS={"socket": base["natural_EOS"], "shca": cand["natural_EOS"]},
                            cap_hit={"socket": base["cap_hit"], "shca": cand["cap_hit"]}))
    return dict(terminal_audit_SHA256=AUDIT_SHA, steps=GATES, verified_ranks_per_arm=list(ranks),
                planned_pairs=len(pins), identical_output_pairs=len(pins)-len(changes),
                divergences=changes, first_divergence_scope="Sampled token sequence; logits and collective operands were not captured for this ordinal.",
                causal_attribution=None, formal_quality_qualified=False)
