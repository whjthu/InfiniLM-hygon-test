"""Audit the frozen v2 pair after its original launcher ends."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

RUN = "90707c8371d4430aa390bd68cf4fefa4"
REQUEST = "06f52eb3cefe4228a9492a2238474903"
COMMIT = "861f006767ade2887364550fd51ec7d9835c5f6d"
CLAIM = "99165a7107501fa4219ac57ff4d2c9f45e416b276451b5e45607c995b6db0a01"
V1_CLAIM = "3f322106a0530b6123efa76072b167a21eca1b8d95b68fa1c6654dfc091e5198"
READER_SHA256 = "0bb4f157c258b03af6d086300f619e914bd94d07809d5df20d5787c93f788e85"
LABEL = "GLM474-pair-" + RUN


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def load_reader(path):
    source = path.read_bytes()
    require(hashlib.sha256(source).hexdigest() == READER_SHA256, "Original reader source changed")
    require(source.count(V1_CLAIM.encode()) == 1, "Original reader claim check changed")
    adapted = source.replace(V1_CLAIM.encode(), CLAIM.encode(), 1)
    namespace = {"__name__": "frozen_paired_reader"}
    exec(compile(adapted, str(path), "exec"), namespace)
    namespace.update(RUN=RUN, LABEL=LABEL)
    return namespace


def audit(root, reader, rows):
    dispatch = root / (LABEL + "-dispatch")
    request, request_sha = read_json(dispatch / "request.json")
    owner, owner_sha = read_json(dispatch / "launcher-owner.json")
    require(all(r["run_id"] == RUN and r["request_id"] == REQUEST
                and r["claim_commit"] == COMMIT for r in (request, owner)), "Dispatch identity changed")
    require(request["claim_SHA256"] == CLAIM and owner["epoch"] == request["epoch"]
            and request["deadline"] - request["epoch"] == 117600, "Pair epoch or claim changed")
    terminal, terminal_sha = read_json(dispatch / "launcher-terminal.json")
    require(terminal["run_id"] == RUN and type(terminal["rc"]) is int
            and terminal["finished"] >= request["epoch"], "Launcher terminal identity changed")

    indexed = reader["accounting"](rows)
    gate = indexed.get(LABEL + "-baseline-gate")
    require(gate and gate[0] == "1103606.78" and reader["is_terminal"](gate),
            "Original baseline gate is not terminal")
    for row in indexed.values():
        require(reader["is_terminal"](row), "A step still lacks terminal accounting")
    result = reader["summarize"](root, rows)
    result["launcher"] = dict(terminal, sha256=terminal_sha, request_sha256=request_sha,
                              owner_sha256=owner_sha)
    for arm, summary in result["arms"].items():
        label = LABEL + "-" + arm
        summary["accounting"] = {phase: indexed.get(label + "-" + phase)
                                 for phase in ("pre", "gate", "post")}
        screen = summary.get("screen", {})
        summary["planned_screen_requests"] = 32
        summary["verified_screen_requests"] = sum(v["completed_verified"] for v in screen.values())
        summary["all_rank_normal_close"] = (
            len(summary.get("owners", {})) == 16
            and all(r["normal_close"] for r in summary["owners"].values())
        )
    ordered = [indexed.get(LABEL + "-" + arm + "-" + phase)
               for arm in ("baseline", "single-drain") for phase in ("pre", "gate", "post")]
    step_numbers = [int(row[0].rsplit(".", 1)[1]) for row in ordered if row]
    result["serial_step_handles_verified"] = len(step_numbers) == 6 and step_numbers == sorted(set(step_numbers))
    result["complete_engineering_pair"] = (
        terminal["rc"] == 0
        and result["serial_step_handles_verified"]
        and all(row[2:4] == ["COMPLETED", "0:0"] for row in ordered)
        and all(
            r["accounting"]["gate"] and r["accounting"]["gate"][2:4] == ["COMPLETED", "0:0"]
            and r["post_verified"] and r["all_rank_normal_close"]
            and all(r.get(phase + "_receipt", {}).get(key) is True
                    for phase in ("gate", "post") for key in ("accounting_matches", "passed", "terminal"))
            and r["diagnostic_completed"] == 1 and r["verified_screen_requests"] == 32
            for r in result["arms"].values()
        )
    )
    result["reference_reader_sha256"] = READER_SHA256
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_root", type=Path)
    parser.add_argument("original_reader", type=Path)
    args = parser.parse_args()
    reader = load_reader(args.original_reader)
    raw = subprocess.check_output(
        ["sacct", "-n", "-P", "-j", reader["PARENT"], "--format=" + reader["COLUMNS"]],
        text=True, timeout=12,
    )
    rows = [row.split("|") for row in raw.splitlines()]
    print(json.dumps(audit(args.results_root, reader, rows), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
