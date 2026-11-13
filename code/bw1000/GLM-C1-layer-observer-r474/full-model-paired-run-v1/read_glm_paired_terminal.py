"""Read original paired GLM artifacts after terminal accounting. Emits bounded aggregates."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess

RUN = "c565dff1d4ea4ff989e48392e5f86d16"
PARENT = "1103606"
NODES = ["f13r1n05", "f13r1n07"]
LABEL = "GLM474-pair-" + RUN
SCORER_SHA256 = "c9c39a5fc185ed6bfc5f48c2b547d4ef36a4cfed351626835793fcf0ab5b521c"
ARMS = ("baseline", "single-drain")
COLUMNS = "JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList"
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL",
            "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}


def read(path):
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def is_terminal(row):
    return bool(row and row[2].split()[0] in TERMINAL)


def accounting(rows):
    expected = {f"{LABEL}-{arm}-{phase}" for arm in ARMS for phase in ("pre", "gate", "post")}
    indexed = {}
    handles = set()
    for row in rows:
        if len(row) < 2 or row[1] not in expected:
            continue
        handle = row[0].split(".")
        require(len(row) == 6 and len(handle) == 2 and handle[0] == PARENT
                and handle[1].isdigit() and row[1] not in indexed and row[0] not in handles,
                "invalid or duplicate original Slurm accounting")
        indexed[row[1]] = row
        handles.add(row[0])
    return indexed


def audit_request(row, pin, eos, special, rescore=None):
    fields = ("ordinal", "case_id", "suite", "prompt_IDs_SHA256", "body_SHA256")
    require(all(row[k] == pin[k] for k in fields), "request does not match frozen input")
    require(row["diagnostic"] is False and row["reset_before"] is True
            and row["reset_after"] is True and row["completed"] is True,
            "request lifecycle incomplete")
    ids = row["generated_token_ids"]
    ticks = row["sampled_ns"]
    require(len(ids) == len(ticks) and 0 < len(ids) <= 4096
            and all(type(t) is int and t > 0 for t in ticks)
            and all(a < b for a, b in zip(ticks, ticks[1:]))
            and all(type(i) is int and 0 <= i < 154880 for i in ids),
            "invalid generated tokens or timing")
    digest = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
    require(digest == row["generated_token_ids_SHA256"], "output hash differs")
    natural = ids[-1] in eos
    cap = not natural and len(ids) == 4096
    require((natural or cap) and not (set(ids[:-1]) & eos)
            and row["natural_EOS"] is natural and row["cap_hit"] is cap, "invalid EOS/cap record")
    visible = [tick for token, tick in zip(ids, ticks) if token not in eos and token not in special]
    decode = (len(visible) - 1) * 1e9 / (visible[-1] - visible[0]) if len(visible) > 1 else None
    require(row["visible_nonEOS_nonspecial"] == len(visible) and
            (row["backend_decode_tok_s"] is None if decode is None else
             math.isclose(row["backend_decode_tok_s"], decode, rel_tol=1e-12)), "decode record differs")
    score = row["score"]
    if rescore is not None:
        require(score == rescore(pin["ordinal"], row["output_text"], natural, cap),
                "independent scoring differs")
    flags = ("correct", "answer_matches_gold", "parse_failure", "unexpected_tool_intent")
    require(all(type(score[k]) is bool for k in flags)
            and score["correct"] == (score["answer_matches_gold"] and natural
                                     and not score["parse_failure"] and not score["unexpected_tool_intent"]),
            "saved scorer flags inconsistent")
    return dict(correct=score["correct"], raw_match=score["answer_matches_gold"],
                parse_failure=score["parse_failure"], cap=cap, decode=decode)


def summarize_requests(rows, binding, rescore=None):
    pins = binding["private_plan"]["requests"]
    eos = set(binding["eos_token_ids"])
    special = set(binding["tokenizer"]["special_ids"])
    require(len(pins) == len({row["ordinal"] for row in pins}) == 32, "wrong planned denominator")
    found = {}
    for row in rows:
        if row["diagnostic"] is True:
            continue
        ordinal = row["ordinal"]
        require(ordinal not in found, "duplicate screen request")
        found[ordinal] = row
    require(set(found) <= {pin["ordinal"] for pin in pins}, "unplanned screen request")
    counts = {}
    for suite in ("gsm8k", "mmlu"):
        selected = [pin for pin in pins if pin["suite"] == suite]
        require(len(selected) == 16, "wrong suite denominator")
        result = dict(planned=16, completed_verified=0, missing=0, incomplete=0, invalid=0,
                      correct=0, raw_match=0, parse_failure=0, cap=0, backend_decode_gt_15=0)
        rates = []
        for pin in selected:
            row = found.get(pin["ordinal"])
            if row is None:
                result["missing"] += 1
                continue
            if row.get("completed") is False or row.get("reset_after") is False:
                result["incomplete"] += 1
                continue
            try:
                item = audit_request(row, pin, eos, special, rescore)
            except (ValueError, KeyError, TypeError):
                result["invalid"] += 1
                continue
            result["completed_verified"] += 1
            for key in ("correct", "raw_match", "parse_failure", "cap"):
                result[key] += item[key]
            if item["decode"] is not None:
                rates.append(item["decode"])
                result["backend_decode_gt_15"] += item["decode"] > 15
        result.update(decode_observed=len(rates),
                      backend_decode_tok_s_median=statistics.median(rates) if rates else None)
        counts[suite] = result
    return counts


def summarize(root, rows):
    observed = accounting(rows)
    require(is_terminal(observed.get(LABEL + "-baseline-gate")), "baseline Slurm terminal required")
    src = root / (LABEL + "-dispatch") / "src"
    binding, binding_sha = read(src / "INPUT_BINDING.json")
    require(binding_sha == "81f6b2d68228aafb79993306847b0edb04c1ba101a35ab21bb822ac96ca2ba5e",
            "input binding hash differs")
    plan_pin = binding["private_plan"]
    plan, plan_sha = read(Path(plan_pin["path"]))
    require(plan_sha == plan_pin["SHA256"], "private plan hash differs")
    requests = plan["requests"]
    cases = {row["ordinal"]: row["case"] for row in requests}
    require(list(cases) == [row["ordinal"] for row in plan_pin["requests"]],
            "private plan order differs")
    scorer = (src / "scoring.py").read_bytes()
    require(hashlib.sha256(scorer).hexdigest() == SCORER_SHA256, "frozen scorer hash differs")
    env = {}
    exec(compile(scorer, "frozen scoring.py", "exec"), env)
    grade = env["grade"]

    def rescore(ordinal, text, natural, cap):
        return grade(cases[ordinal], text, natural_EOS=natural, cap_hit=cap, unexpected_tool=False)

    result = {}
    for arm in ARMS:
        label = LABEL + "-" + arm
        phases = {phase: observed.get(label + "-" + phase) for phase in ("pre", "gate", "post")}
        gate = phases["gate"]
        item = {"accounting": phases}
        result[arm] = item
        if not is_terminal(gate):
            item["state"] = "NOT_OBSERVED" if gate is None else "IN_FLIGHT"
            continue
        folder = root / label
        item["state"] = "TERMINAL"
        for phase in ("gate", "post"):
            path = folder / (phase + "-receipt.json")
            if path.is_file():
                value, digest = read(path)
                item[phase + "_receipt"] = dict(SHA256=digest, accounting_matches=value.get("row") == phases[phase],
                                               passed=value.get("passed"), terminal=value.get("terminal"))
        post = phases["post"]
        cleanup = []
        if is_terminal(post):
            for node in NODES:
                path = folder / "post" / (node + ".json")
                before_path = folder / "pre" / (node + ".json")
                if not path.is_file() or not before_path.is_file():
                    continue
                record, digest = read(path)
                before, _ = read(before_path)
                valid = (record.get("step") == post[0] and record.get("node") == node
                         and before.get("node") == node and before.get("boot_id") == record.get("boot_id")
                         and before.get("pci") == record.get("pci") and len(set(record["pci"])) == 8
                         and record["monotonic_ns"] >= before["monotonic_ns"]
                         and record["kfd_holder_count"] == 0
                         and record["ports_bindable"] == {"43843": True, "43844": True})
                cleanup.append(dict(node=node, SHA256=digest, passed=valid))
        item["post_node_checks"] = cleanup
        item["post_verified"] = bool(post and post[2:4] == ["COMPLETED", "0:0"]
                                     and len(cleanup) == 2 and all(v["passed"] for v in cleanup))
        owner_path = folder / "owner-00.json"
        request_rows = []
        if owner_path.is_file():
            owner, digest = read(owner_path)
            require(owner["actual_GPU"] == gate[0] and owner["global_rank"] == 0, "wrong rank-0 source")
            item["rank0_SHA256"] = digest
            item["rank0_normal_close"] = owner.get("normal_close")
            request_rows = owner["requests"]
        item["screen"] = summarize_requests(request_rows, binding, rescore)
        item["diagnostic_completed"] = sum(row.get("diagnostic") is True and row.get("completed") is True
                                          and row.get("reset_after") is True for row in request_rows)
    return dict(run_id=RUN, arms=result, score_scope="frozen scorer rerun on saved rank-0 output, with completion checks",
                timing_scope="backend sampled tokens; no HTTP client or fleet busy evidence")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_root", type=Path)
    args = parser.parse_args()
    raw = subprocess.check_output(["sacct", "-n", "-P", "-j", PARENT, "--format=" + COLUMNS],
                                  text=True, timeout=12)
    print(json.dumps(summarize(args.results_root, [row.split("|") for row in raw.splitlines()]),
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
