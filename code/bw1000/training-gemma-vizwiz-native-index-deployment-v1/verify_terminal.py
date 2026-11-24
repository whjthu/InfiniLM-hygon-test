"""Check a completed index against separately captured numeric-step Slurm accounting."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import re
from pathlib import Path

import index_entry as entry
import render


def published_lines():
    source = render.ROOT / "code/bw1000/training-gemma-vizwiz-first-two-updates-v1/TRAINING_SCOPE.json"
    raw = source.read_bytes()
    entry.require(sha256(raw).hexdigest() == "7b494f529c5d202b522fb35cd5cf89af67bb97775534968518aee109668bd9d0",
                  "EXACT_PUBLISHED_TRAINING_SCOPE")
    return {digest for member in json.loads(raw)["existing_frozen_rows"] for digest in member["ordered_line_SHA256"]}


def verify(accounting, parent, node, step, result, index):
    row = render.terminal(accounting, parent, node, step, require_success=result["state"] != "FAILED")
    entry.require(result["CPU_step"] in (step, None) and result["planned_train_rows"] == 12
                  and result["calibration_roles"] == result["validation_roles"] == result["test_roles"] == 0
                  and result["model_loads"] == result["optimizer_updates"] == result["GPU"] == 0,
                  "WORKER_EVIDENCE_SCOPE")
    encodes, collations = result["native_encodes_completed"], result["native_collations_completed"]
    entry.require(type(encodes) is type(collations) is int and 0 <= collations <= encodes <= 12,
                  "WORKER_PLANNED_AND_COMPLETED_DENOMINATOR")
    if result["state"] == "FAILED":
        entry.require(index is None and result["qualified_index_rows"] == 0, "FAILED_INDEX_PRESENT")
        return {"state": "TERMINAL_CPU_INDEX_FAILURE_RETAINED", "accounting": row, "planned_train_rows": 12,
                "qualified_index_rows": 0, "native_encodes_completed": encodes,
                "native_collations_completed": collations}
    entry.require(result["state"] == "EXACT_NATIVE_TRAIN_INDEX_WRITTEN" and result["CPU_step"] == step
                  and encodes == collations == result["native_source_rows"] == result["train_roles"] == 12
                  and index is not None and entry.pin(index) == result["private_index"], "INDEX_IDENTITY")
    document = json.loads(index)
    entry.require(document["view_SHA256"] == entry.VIEW and document["original_media_pins_checked"] == 12
                  and len(document["index"]) == 12 and set(document["index"].values()) == published_lines()
                  and all(re.fullmatch("[0-9a-f]{64}", key) for key in document["index"]),
                  "INDEX_SOURCE_SCOPE")
    return {"state": "ACTUAL_ZERO_GRES_NATIVE_TRAIN_INDEX_VERIFIED", "accounting": row, "planned_train_rows": 12,
            "qualified_index_rows": 12, "native_encodes_completed": encodes, "native_collations_completed": collations,
            "private_index": entry.pin(index)}


def bounded(path):
    entry.require(path.is_file() and not path.is_symlink(), "TERMINAL_EVIDENCE_PATH")
    with path.open("rb") as stream:
        data = stream.read(16385)
    entry.require(0 < len(data) <= 16384, "TERMINAL_EVIDENCE_EXTENT")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("accounting", "worker-result", "index", "receipt"):
        parser.add_argument("--" + name, type=Path, required=name != "index")
    for name in ("parent", "node", "step"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    accounting = bounded(args.accounting)
    worker = bounded(args.worker_result)
    report = verify(accounting.decode(), args.parent, args.node, args.step,
                    json.loads(worker), bounded(args.index) if args.index else None)
    report.update(independent_accounting=entry.pin(accounting), worker_result=entry.pin(worker))
    print(json.dumps(entry.save(args.receipt, report)))
    return int(report["qualified_index_rows"] != 12)


if __name__ == "__main__":
    raise SystemExit(main())
