"""Freeze an answer-free prompt-format comparison from the training split."""

import argparse
import json
from pathlib import Path

from prepare_qwen_inputs import MANIFEST_SHA256, TOPIC_QUOTAS, digest, pinned_bytes, rank, read_jsonl
from treasury_holdout import TABLE_ID, jsonl, normalized


TOPIC = "signed_difference"
FORMAT_INSTRUCTION = (
    " Output only the amount as a decimal with exactly two digits after the decimal point. "
    "Use a minus sign for negative amounts and no currency symbol, commas, or explanation."
)


def prepare(source, output):
    manifest = json.loads(pinned_bytes(source / "manifest.json", MANIFEST_SHA256))
    proof = manifest["splits"]["train"]
    root = source / "train"
    records = read_jsonl(root / "inputs.jsonl", proof["records_SHA256"])
    labels = read_jsonl(root / "labels.jsonl", proof["labels_SHA256"])
    image_rows = read_jsonl(root / "images.jsonl", proof["image_inventory_SHA256"])
    inventory = dict(image_rows)
    if len(records) != 216 or len(labels) != 216 or len(image_rows) != 72 or len(inventory) != 72:
        raise ValueError("training denominator differs")

    topic_rows = []
    for record, label in zip(records, labels, strict=True):
        if label["topic"] != TOPIC:
            continue
        table = label["table"]
        question, target = record["messages"]
        name = table + ".png"
        if (not TABLE_ID.fullmatch(table) or label["id"] != f"{table}_{TOPIC}"
                or record["images"] != [f"train/{name}"]
                or question["role"] != "user" or target != {"role": "assistant", "content": label["answer"]}
                or normalized(label["answer"], TOPIC) is None
                or not question["content"].startswith("<image>\n") or question["content"].count("<image>") != 1):
            raise ValueError("arithmetic row identity differs")
        topic_rows.append((rank(label["id"]), label["id"], name, question["content"], label["answer"]))

    if len(topic_rows) != 72 or len({item[1] for item in topic_rows}) != 72:
        raise ValueError("arithmetic source denominator differs")
    selected = sorted(topic_rows)[TOPIC_QUOTAS[TOPIC]:]
    if len(selected) != 19:
        raise ValueError("arithmetic calibration denominator differs")
    media, paired, gold = {}, [], []
    for _, identity, name, prompt, answer in selected:
        raw = pinned_bytes(root / name, inventory[name[:-4]])
        if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("invalid arithmetic media input")
        media[name] = raw
        paired.append({"id": identity, "image": f"media/{name}", "original": prompt,
                       "explicit": prompt + FORMAT_INSTRUCTION})
        gold.append({"id": identity, "answer": answer})

    answer_free = jsonl(paired)
    labels_private = jsonl(gold)
    receipt = dict(split="train", calibration_cases=19, source_manifest_SHA256=MANIFEST_SHA256,
                   source_signed_difference_cases=72, prior_signed_difference_training_cases=53,
                   excluded_prior_arithmetic_training_cases=True, cross_topic_training_images_may_overlap=True,
                   prompts_SHA256=digest(answer_free), private_labels_SHA256=digest(labels_private),
                   paired_ids_SHA256=digest(("\n".join(row["id"] for row in paired) + "\n").encode("ascii")),
                   media_SHA256={name: digest(raw) for name, raw in media.items()},
                   media_bytes=sum(map(len, media.values())), format_instruction=FORMAT_INSTRUCTION,
                   formal_holdout=False, formal_effect=False, actual_GPU_steps=0)

    output.mkdir()
    input_root = output / "generation"
    media_root = input_root / "media"
    media_root.mkdir(parents=True)
    (input_root / "prompts.jsonl").write_bytes(answer_free)
    for name, raw in media.items():
        (media_root / name).write_bytes(raw)
    (output / "labels.private.jsonl").write_bytes(labels_private)
    (output / "receipt.json").write_bytes(jsonl([receipt]))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = prepare(args.source, args.output)
    print(json.dumps({key: receipt[key] for key in (
        "split", "calibration_cases", "prompts_SHA256", "paired_ids_SHA256", "media_bytes")}))


if __name__ == "__main__":
    main()
