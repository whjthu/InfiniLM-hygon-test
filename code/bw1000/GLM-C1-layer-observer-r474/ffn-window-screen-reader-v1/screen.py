"""Regrade the original GLM110 screen and recompute backend sampled clocks in place."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics

RUN = "4d4f0c4b2cd344ae92c81ea40c06aecf"
REQUEST = "29dee83872aa4843afa6d1e46a843ed0"
CLAIM = "fcbe791b159652abb9f04bcb78cf1a80032ce94d"
GPU = "1103606.110"
READER = "375e5e764a62dda57a885a707a83b327e843c00121f1aff1e7676e9e0fda80fe"
INPUT = "81f6b2d68228aafb79993306847b0edb04c1ba101a35ab21bb822ac96ca2ba5e"
SCORER = "c9c39a5fc185ed6bfc5f48c2b547d4ef36a4cfed351626835793fcf0ab5b521c"
LABEL = f"GLM474-ffn-{RUN}-socket"


def need(condition, message):
    if not condition:
        raise ValueError(message)


def read_pinned(path, digest, bound):
    with path.open("rb") as stream:
        raw = stream.read(bound + 1)
    need(0 < len(raw) <= bound and hashlib.sha256(raw).hexdigest() == digest,
         f"wrong original artifact: {path.name}")
    return raw


def summarize_screen(owner, pins, plan, eos_ids, special_ids, grade):
    need((owner["global_rank"], owner["actual_GPU"], owner["passed"]) == (0, GPU, True),
         "wrong original scoring rank")
    requests = owner["requests"]
    need(len(requests) == 33 and requests[0]["diagnostic"] is True and len(pins) == len(plan) == 32,
         "wrong screen inventory")
    suites = {}
    for index, (row, pin, original) in enumerate(zip(requests[1:], pins, plan, strict=True)):
        need(row["diagnostic"] is False and row["screen_index"] == index and row["completed"] is True
             and all(row[k] == pin[k] == original[k] for k in
                     ("ordinal", "case_id", "suite", "prompt_tokens", "prompt_IDs_SHA256", "body_SHA256")),
             "screen request differs from its frozen input")
        ids, ticks = row["generated_token_ids"], row["sampled_ns"]
        start = row["request_enter_ns"]
        need(type(start) is int and start > 0 and 0 < len(ids) == len(ticks) <= 4096
             and all(type(t) is int and t >= start for t in ticks)
             and all(a < b for a, b in zip(ticks, ticks[1:]))
             and all(type(token) is int and 0 <= token < 154880 for token in ids)
             and not any(token in eos_ids for token in ids[:-1]), "invalid sampled token timeline")
        digest = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
        natural = ids[-1] in eos_ids
        cap = not natural and len(ids) == 4096
        need((natural or cap) and row["natural_EOS"] is natural and row["cap_hit"] is cap
             and row["generated_token_ids_SHA256"] == digest, "wrong EOS, cap or output hash")
        score = grade(original["case"], row["output_text"], natural_EOS=natural, cap_hit=cap)
        need(score == row["score"], "frozen scorer differs from original owner")
        visible = [t for token, t in zip(ids, ticks) if token not in eos_ids and token not in special_ids]
        rate = (len(visible) - 1) * 1e9 / (visible[-1] - visible[0]) if len(visible) > 1 else None
        ttft = (visible[0] - start) / 1e9 if visible else None
        need((len(visible), rate, ttft) == (row["visible_nonEOS_nonspecial"], row["backend_decode_tok_s"],
                                          row["backend_sampled_TTFT_s"]), "original backend clocks differ")
        suites.setdefault(pin["suite"], []).append((score, rate, ttft))
    need(set(suites) == {"gsm8k", "mmlu"} and all(len(rows) == 16 for rows in suites.values()),
         "wrong planned suite split")
    result = {}
    for suite, rows in suites.items():
        rates = [rate for _, rate, _ in rows if rate is not None]
        ttft = [first for _, _, first in rows if first is not None]
        result[suite] = dict(planned=16, verified=16, missing_or_unverified=0,
            **{key: sum(score[key] for score, _, _ in rows) for key in
               ("correct", "answer_matches_gold", "parse_failure", "natural_EOS", "cap_hit")},
            backend_decode_rate_available=len(rates), backend_strict_gt15=sum(rate > 15 for rate in rates),
            median_backend_decode_tok_s=statistics.median(rates) if rates else None,
            median_backend_sampled_TTFT_s=statistics.median(ttft) if ttft else None)
    return result


def audit(results, report_path, report_sha):
    report = json.loads(read_pinned(report_path, report_sha, 1 << 20))
    terminal = report["terminal"]
    need((report["terminal_reader_SHA256"], terminal["run_id"], terminal["original_request_id"],
          terminal["claim_commit"]) == (READER, RUN, REQUEST, CLAIM), "wrong original audit identity")
    gate = terminal["actual_accounting"]["gate"]
    need(gate[:2] == [GPU, LABEL + "-gate"] and gate[2].split()[0].rstrip("+") in
         {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
          "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}, "original gate not terminal")
    out = dict(original_GPU=GPU, private_terminal_audit_SHA256=report_sha, planned_requests=33,
               diagnostic_outside_quality_denominator=True, planned_screen=32,
               verified_screen=0, missing_or_unverified_screen=32, per_suite=None,
               same_owner_backend_clock=True, formal_quality_qualified=False, service_qualified=False)
    if not (terminal["valid_ffn_evidence"] is True and terminal["independent_post_clean"] is True
            and terminal["launcher_rc"] == 0 and gate[2:4] == ["COMPLETED", "0:0"]):
        return out
    source = results / f"GLM474-ffn-{RUN}-dispatch" / "src"
    binding = json.loads(read_pinned(source / "INPUT_BINDING.json", INPUT, 1 << 20))
    pin = binding["private_plan"]
    plan = json.loads(read_pinned(Path(pin["path"]), pin["SHA256"], pin["bytes"]))
    scorer_path = source / "scoring.py"
    raw_scorer = read_pinned(scorer_path, SCORER, 65536)
    spec = importlib.util.spec_from_file_location("frozen_glm_scoring", scorer_path)
    scorer = importlib.util.module_from_spec(spec)
    exec(compile(raw_scorer, str(scorer_path), "exec"), scorer.__dict__)
    owner_name = f"{LABEL}/owner-00.json"
    owner = json.loads(read_pinned(results / owner_name, terminal["raw_artifact_SHA256"][owner_name], 128 << 20))
    scores = summarize_screen(owner, pin["requests"], plan["requests"], set(binding["eos_token_ids"]),
                              set(binding["tokenizer"]["special_ids"]), scorer.grade)
    return dict(out, verified_screen=32, missing_or_unverified_screen=0, per_suite=scores)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("private_terminal_report", type=Path)
    parser.add_argument("report_sha256")
    parser.add_argument("public_screen", type=Path)
    args = parser.parse_args()
    report = audit(args.results, args.private_terminal_report, args.report_sha256)
    with args.public_screen.open("x", encoding="utf8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
