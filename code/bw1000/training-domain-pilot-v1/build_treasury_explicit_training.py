"""Bind an instruction-only training revision to the original 160 Treasury rows."""

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path, PurePosixPath

from build_treasury_arithmetic_calibration import FORMAT_INSTRUCTION
from prepare_qwen_inputs import TOPIC_QUOTAS, digest, pinned_bytes, rank
from treasury_holdout import TABLE_ID, normalized


ORIGINAL_DATASET_SHA256 = "e0219347307a94259b826b929a88f3fe991b688026677c86c25075c6941bc19b"
ORIGINAL_ID_ORDER_SHA256 = "08a6400d8c8cdf577e77099a4b524f97643b80a23df783f04544488b01de009b"
IMAGE_ROOT = PurePosixPath("/refactor/results/training-qwen-treasury-pilot-data-v1/data/train")


def build(original):
    """Change only 53 user instructions, preserving rows, labels, images and order."""
    rows = [json.loads(line) for line in pinned_bytes(original, ORIGINAL_DATASET_SHA256).splitlines()]
    if len(rows) != 160:
        raise ValueError("original training denominator differs")
    tables = set()
    for row in rows:
        if set(row) != {"messages", "images"} or len(row["images"]) != 1:
            raise ValueError("unexpected training schema")
        image = PurePosixPath(row["images"][0])
        if image.parent != IMAGE_ROOT or image.suffix != ".png" or not TABLE_ID.fullmatch(image.stem):
            raise ValueError("image is outside the original training split")
        tables.add(image.stem)
    if len(tables) != 72:
        raise ValueError("original image denominator differs")
    selected = []
    for topic, count in TOPIC_QUOTAS.items():
        candidates = sorted((rank(f"{table}_{topic}"), f"{table}_{topic}", topic, table) for table in tables)
        selected.extend(candidates[:count])
    selected.sort()
    identities = [item[1] for item in selected]
    if digest("\n".join(identities).encode("ascii")) != ORIGINAL_ID_ORDER_SHA256:
        raise ValueError("original selection order differs")

    output = deepcopy(rows)
    changed = Counter()
    for old, new, (_, _, topic, table) in zip(rows, output, selected, strict=True):
        messages = old["messages"]
        if (len(messages) != 2 or any(set(message) != {"role", "content"} for message in messages)
                or [m["role"] for m in messages] != ["user", "assistant"]
                or not messages[0]["content"].startswith("<image>\n")
                or messages[0]["content"].count("<image>") != 1
                or PurePosixPath(old["images"][0]).stem != table
                or normalized(messages[1]["content"], topic) is None
                or FORMAT_INSTRUCTION in messages[0]["content"]):
            raise ValueError("training example binding differs")
        if topic == "signed_difference":
            new["messages"][0]["content"] += FORMAT_INSTRUCTION
            changed[topic] += 1
        if old["images"] != new["images"] or old["messages"][1] != new["messages"][1]:
            raise ValueError("instruction revision changed image or label")
    if changed != {"signed_difference": 53}:
        raise ValueError("instruction change denominator differs")
    raw = b"".join((json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf8")
                   for row in output)
    return raw, dict(
        state="TRAINING_INSTRUCTION_REVISION_NATIVE_PROCESSING_PENDING", split="train",
        original_dataset_SHA256=ORIGINAL_DATASET_SHA256, dataset_SHA256=digest(raw),
        selected_id_order_SHA256=ORIGINAL_ID_ORDER_SHA256, rows=160, unique_images=72,
        topic_counts=dict(TOPIC_QUOTAS), changed_user_instructions=53,
        labels_changed=0, image_paths_changed=0, row_order_changed=False,
        format_instruction=FORMAT_INSTRUCTION, actual_updates=0, native_processing_verified=False,
        development_or_blind_inputs_read=False, formal_acceptance=False,
        qualification="New training input only; prior native-input checks do not qualify these changed prompts.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw, receipt = build(args.original)
    args.output.mkdir()
    (args.output / "inputs.jsonl").write_bytes(raw)
    (args.output / "receipt.json").write_bytes((json.dumps(receipt, ensure_ascii=False, indent=2) + "\n").encode("utf8"))
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
