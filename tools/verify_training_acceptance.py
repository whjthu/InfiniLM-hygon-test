"""Verify the training acceptance matrix and the evidence pinned by it."""
import argparse
import hashlib
import json
from pathlib import Path

from training_formal_metrics import stage_three


TRAINING_GROUPS = {f"T{i:02d}" for i in range(1, 13)}


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
    require(matrix["stage3"] == {"minimum_A100_throughput_ratio": 0.9,
                                 "maximum_relative_effect_regression": 0.05},
            "unexpected stage-three thresholds")
    ledger = matrix["node_ledger"]
    groups = json.loads(pinned(ledger["path"], ledger["SHA256"]))["groups"]
    require({group for group in groups if group.startswith("T")} == TRAINING_GROUPS,
            "training task groups differ")
    assigned = [case for group in sorted(TRAINING_GROUPS) for case in groups[group]["cases"]]
    require(ids == assigned, "matrix and resource ledger training case identities differ")
    pins = dict(matrix["evidence_SHA256"])
    def verify_formal_attachment(item):
        path, digest = item["path"], item["SHA256"]
        require(pins.setdefault(path, digest) == digest, f"conflicting evidence hashes: {path}")
        # Attachments are small summaries or manifests for private experiment receipts.
        return json.loads(pinned(path, digest))

    for case in cases:
        for evidence in case.get("supplemental_evidence", {}).values():
            path, sha = evidence["path"], evidence["SHA256"]
            require(pins.setdefault(path, sha) == sha, f"conflicting evidence hashes: {path}")
        for field in ("reference_evidence", "hygon_evidence"):
            path = case[field]
            require(path is None or path in pins, f"{case['case']}: unpinned {field}")
        formal = case.get("formal_acceptance")
        if case["stage3_accepted"]:
            require(isinstance(formal, dict), f"{case['case']}: missing formal acceptance evidence")
        if formal is not None:
            gate_paths = set()
            for gate in ("function", "numerics", "recovery"):
                item = formal["gate_evidence"][gate]
                path, digest = item["path"], item["SHA256"]
                require(path not in gate_paths, f"{case['case']}: duplicate gate evidence")
                gate_paths.add(path)
                require(pins.setdefault(path, digest) == digest, f"conflicting evidence hashes: {path}")
                result = json.loads(pinned(path, digest))
                require(result["case"] == case["case"] and result["framework_commit"] == matrix["framework_commit"]
                        and result["gate"] == gate and result["passed"] is True,
                        f"{case['case']}: wrong or failed {gate} gate")
            item = formal["metrics_evidence"]
            path, digest = item["path"], item["SHA256"]
            require(path not in gate_paths, f"{case['case']}: duplicate metrics evidence")
            require(pins.setdefault(path, digest) == digest, f"conflicting evidence hashes: {path}")
            metrics = stage_three(case, matrix["framework_commit"], json.loads(pinned(path, digest)),
                                  verify_formal_attachment)
            require(metrics["passed"] is case["stage3_accepted"],
                    f"{case['case']}: formal measurements and acceptance flag disagree")
    for path, sha in pins.items():
        pinned(path, sha)
    human = pinned(index["human_readable"], index["human_readable_SHA256"]).decode("utf-8")
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
