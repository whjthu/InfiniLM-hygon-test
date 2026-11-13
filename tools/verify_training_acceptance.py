"""Verify the training acceptance matrix and the evidence pinned by it."""
import argparse
import hashlib
import json
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(root):
    status = json.loads((root / "status/bw1000/latest.json").read_bytes())
    index = status["unified_training_acceptance_matrix"]

    def pinned(path, expected):
        file = (root / path).resolve(strict=True)
        require(file.is_relative_to(root), f"evidence outside repository: {path}")
        raw = file.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == expected, f"hash mismatch: {path}")
        return raw

    matrix = json.loads(pinned(index["matrix"], index["matrix_SHA256"]))
    cases = matrix["cases"]
    ids = [case["case"] for case in cases]
    require(len(ids) == len(set(ids)) == matrix["case_count"] == index["cases"] == 17,
            "unexpected training case denominator")
    require(matrix["framework_commit"] == index["framework_commit"], "framework pin differs")
    require(all(type(case["stage3_accepted"]) is bool for case in cases), "invalid acceptance flag")
    require(sum(case["stage3_accepted"] for case in cases) == index["stage3_passed"],
            "stage-three count differs")
    ledger = matrix["node_ledger"]
    pinned(ledger["path"], ledger["SHA256"])
    pins = dict(matrix["evidence_SHA256"])
    for case in cases:
        for evidence in case.get("supplemental_evidence", {}).values():
            path, sha = evidence["path"], evidence["SHA256"]
            require(pins.setdefault(path, sha) == sha, f"conflicting evidence hashes: {path}")
        for field in ("reference_evidence", "hygon_evidence"):
            path = case[field]
            require(path is None or path in pins, f"{case['case']}: unpinned {field}")
    for path, sha in pins.items():
        pinned(path, sha)
    human = (root / index["human_readable"]).read_text(encoding="utf-8")
    markdown_ids = [line.split("|")[1].strip() for line in human.splitlines() if line.startswith("| T")
                    and line.split("|")[1].strip() != "用例"]
    require(markdown_ids == ids, "Markdown and JSON training case order differs")
    return {"cases": len(ids), "stage3_passed": index["stage3_passed"], "evidence_files": len(pins)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("repo", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.repo.resolve(strict=True))))


if __name__ == "__main__":
    main()
