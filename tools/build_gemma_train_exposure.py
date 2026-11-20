"""Materialize the frozen Gemma training-only view for 25 full updates."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


BINDING_SHA256 = "7182f239841f2b09ebc6645c2ee9a6d3c2dd1416840d1124e6da88b8c6ee921d"
TRAIN_SHA256 = "a897e7d9e7823d01bd2b503104d138f4a50975cabc06a2fefcda1dfb36ff05fd"
TRAIN_FILE = "train-swift.jsonl"
VIEW_FILE = "train800-swift.jsonl"
RECEIPT_FILE = "exposure-ledger-private.json"
SOURCE_ROWS = 72
WORLD_SIZE = 8
DEVICE_BATCH = 1
ACCUMULATION = 4
UPDATES = 25
RESUME_FROM = 20
EXPOSURES = WORLD_SIZE * DEVICE_BATCH * ACCUMULATION * UPDATES


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_pinned_file(path, expected_size, expected_digest):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected regular file: {path.name}")
    with path.open("rb") as stream:
        raw = stream.read(expected_size + 1)
    if len(raw) != expected_size or digest(raw) != expected_digest:
        raise ValueError(f"size or SHA-256 differs: {path.name}")
    return raw


def exposure_plan(lines):
    if len(lines) != SOURCE_ROWS or len(set(lines)) != SOURCE_ROWS:
        raise ValueError("expected 72 distinct frozen source rows")
    if any(not line.strip() or not line.endswith(b"\n") for line in lines):
        raise ValueError("expected complete JSONL source rows")
    views = []
    for line in lines:
        row = json.loads(line)
        messages = row["messages"]
        if ([m["role"] for m in messages] != ["user", "assistant"]
                or not isinstance(row["images"], list) or len(row["images"]) != 1):
            raise ValueError("expected one supervised Gemma text/image source row")
        views.append(row["images"][0])
    if len(set(views)) != 12:
        raise ValueError("expected 12 frozen training images")

    order = [i % SOURCE_ROWS for i in range(EXPOSURES)]
    counts = Counter(order)
    receipt = {
        "schema": "gemma-render72-train800-exposure-v1",
        "source_rows": SOURCE_ROWS,
        "source_images": len(set(views)),
        "physical_rows": EXPOSURES,
        "source_index_by_physical_row": order,
        "source_line_SHA256": [digest(line) for line in lines],
        "exposure_count_by_source_index": [counts[i] for i in range(SOURCE_ROWS)],
        "expected_world_size": WORLD_SIZE,
        "expected_device_batch": DEVICE_BATCH,
        "expected_gradient_accumulation": ACCUMULATION,
        "expected_effective_batch": WORLD_SIZE * DEVICE_BATCH * ACCUMULATION,
        "expected_microbatches_per_rank": EXPOSURES // (WORLD_SIZE * DEVICE_BATCH),
        "expected_continuous_updates": UPDATES,
        "expected_resume_skip_microbatches_per_rank": RESUME_FROM * ACCUMULATION,
        "expected_resume_tail_microbatches_per_rank": (UPDATES - RESUME_FROM) * ACCUMULATION,
        "native_loader_order_observed": False,
        "actual_optimizer_updates": 0,
        "dev_or_blind_rows_read": 0,
        "training_ready": False,
    }
    return b"".join(lines[i] for i in order), receipt


def materialize(prepared, output):
    binding_bytes = read_pinned_file(prepared / "data-binding-private.json", 10123, BINDING_SHA256)
    binding = json.loads(binding_bytes)
    if (binding["case"] != "T02.sft" or binding["max_sequence_length"] != 512
            or binding["files"][TRAIN_FILE] != {"bytes": 54246, "rows": SOURCE_ROWS, "SHA256": TRAIN_SHA256}):
        raise ValueError("Gemma source binding differs")
    train = read_pinned_file(prepared / TRAIN_FILE, 54246, TRAIN_SHA256)
    view, receipt = exposure_plan(train.splitlines(keepends=True))
    receipt.update(source_binding_SHA256=BINDING_SHA256, source_train_SHA256=TRAIN_SHA256,
                   view_SHA256=digest(view), view_bytes=len(view))
    ledger = (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    output.mkdir(mode=0o700)
    (output / VIEW_FILE).write_bytes(view)
    (output / RECEIPT_FILE).write_bytes(ledger)
    return {
        "physical_rows": EXPOSURES, "source_rows": SOURCE_ROWS,
        "source_images": receipt["source_images"], "exposure_min": EXPOSURES // SOURCE_ROWS,
        "exposure_max": (EXPOSURES + SOURCE_ROWS - 1) // SOURCE_ROWS,
        "view_bytes": len(view), "view_SHA256": receipt["view_SHA256"],
        "ledger_SHA256": digest(ledger),
        "actual_optimizer_updates": 0, "training_ready": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(materialize(args.prepared, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
