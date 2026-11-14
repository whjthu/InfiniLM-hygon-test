"""Read the original FFN run after Slurm and the launcher have reached terminal state."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import time

from host_launch import ARM, COLUMNS, account, evidence, expand, verify_owners, verify_released
from shared import ARMS, D, HOST, NODES, PARENT, H, load_pinned, namespace, require, verify_manifest
from slurm_accounting import is_terminal
from budget import PAIR_SECONDS


def read_terminal(run, claim_hash, request_id, commit):
    require(re.fullmatch(r"[0-9a-f]{32}", request_id) is not None, "invalid request identity")
    require(re.fullmatch(r"[0-9a-f]{40}", commit) is not None, "invalid published commit")
    label, _ = namespace(run, ARM)
    claim = load_pinned(D / "GPU_RESOURCE_CLAIM.json", claim_hash)
    require(claim["schema"] == "glm474-ffn-model-claim-v1" and claim["run_id"] == run
            and claim["GPU_GO"] is True and claim["arm_order"] == [ARM]
            and claim["parent"] == PARENT and claim["nodes"] == NODES
            and claim["native_arms"] == {ARM: ARMS[ARM][1]}, "wrong published FFN claim")
    root = HOST / "results"
    dispatch = root / f"GLM474-ffn-{run}-dispatch"
    out = root / label
    files = {}

    def read(path):
        raw = path.read_bytes()
        files[str(path.relative_to(root))] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    request = read(dispatch / "request.json")
    owner = read(dispatch / "launcher-owner.json")
    terminal = read(dispatch / "launcher-terminal.json")
    require(all(v["run_id"] == run for v in (request, owner, terminal))
            and request["claim_SHA256"] == claim_hash
            and all(v["request_id"] == request_id and v["claim_commit"] == commit for v in (request, owner))
            and owner["stage"] == "LAUNCHER_STARTED"
            and all(type(owner[k]) is int and owner[k] > 0 for k in ("PID", "startticks"))
            and math.isfinite(request["epoch"]) and request["epoch"] > 0
            and owner["epoch"] == request["epoch"]
            and request["deadline"] == request["epoch"] + PAIR_SECONDS
            and math.isfinite(terminal["finished"]) and terminal["finished"] >= request["epoch"]
            and type(terminal["rc"]) is int, "wrong original launcher accounting")

    rows = {phase: account(f"{label}-{phase}", time.monotonic() + 12)
            for phase in ("pre", "gate", "post")}
    require(is_terminal(rows["gate"]), "original GPU accounting has no terminal row")
    require(all(row is None or is_terminal(row) for row in rows.values()),
            "a related Slurm step has not reached terminal state")
    report = dict(run_id=run, original_request_id=request_id, claim_commit=commit,
                  launcher_rc=terminal["rc"], planned_requests_per_rank=33,
                  planned_owner_ranks=16, accounting_columns=COLUMNS.split(","),
                  actual_accounting=rows, independent_post_clean=False, valid_ffn_evidence=False,
                  full_ffn_coverage=False, raw_artifact_SHA256=files, errors={})

    def verify_step(phase):
        row = rows[phase]
        require(row and row[2:4] == ["COMPLETED", "0:0"] and expand(row[5]) == set(NODES),
                f"original {phase} Slurm step did not complete cleanly")
        receipt = read(out / f"{phase}-receipt.json")
        require(receipt["accounting_columns"] == report["accounting_columns"]
                and receipt["terminal"] is True and receipt["passed"] is True
                and receipt["row"] == row, f"original {phase} receipt disagrees")
        for node in NODES:
            read(out / phase / (node + ".json"))
        return evidence(out, phase, row[0])

    try:
        before = verify_step("pre")
        after = verify_step("post")
        verify_released(before, after)
        report["independent_post_clean"] = True
    except (ValueError, KeyError, OSError, TypeError) as exc:
        report["errors"]["post"] = type(exc).__name__ + ": " + str(exc)

    try:
        require(terminal["rc"] == 0 and report["independent_post_clean"], "launcher or cleanup failed")
        gate = verify_step("gate")
        manifest_path = out / "RUN_MANIFEST.json"
        manifest = read(manifest_path)
        verify_manifest(manifest, run, ARM, manifest["epoch"])
        require(manifest["claim_SHA256"] == claim_hash
                and manifest["source_manifest_SHA256"] == claim["source_manifest_SHA256"]
                and request["epoch"] <= manifest["epoch"] < request["deadline"], "wrong original run manifest")
        for node in range(2):
            read(out / f"worker-{node}.json")
        for rank in range(16):
            read(out / f"owner-{rank:02d}.json")
        observed = verify_owners(out, ARM, H(manifest_path), rows["gate"][0], gate)["ffn_observation"]
        report.update(valid_ffn_evidence=True, full_ffn_coverage=observed["full_coverage"],
                      ffn_observation=observed)
    except (ValueError, KeyError, OSError, TypeError) as exc:
        report["errors"]["ffn"] = type(exc).__name__ + ": " + str(exc)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("claim_sha256")
    parser.add_argument("request_id")
    parser.add_argument("commit")
    parser.add_argument("private_report", type=Path)
    args = parser.parse_args()
    report = read_terminal(args.run, args.claim_sha256, args.request_id, args.commit)
    with args.private_report.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return 0 if report["valid_ffn_evidence"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
