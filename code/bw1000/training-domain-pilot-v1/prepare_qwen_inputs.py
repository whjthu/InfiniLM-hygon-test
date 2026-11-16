"""Bind a fixed Qwen3-VL domain pilot to the verified Treasury training split."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path, PurePosixPath


MANIFEST_SHA256 = "8a888584e40c7b2bbfea54824a2e80783b05bd166d759133863b435285e1ab4e"
TOPIC_QUOTAS = {"lookup": 54, "signed_difference": 53, "maximum_date": 53}
SEED = b"treasury-qwen3vl-pilot-selection-v1"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def pinned_bytes(path, expected):
    raw = path.read_bytes()
    if digest(raw) != expected:
        raise ValueError(f"SHA-256 mismatch for {path.name}")
    return raw


def read_jsonl(path, expected):
    return [json.loads(line) for line in pinned_bytes(path, expected).splitlines()]


def rank(identity):
    return digest(SEED + b"\0" + identity.encode("ascii"))


def prepare(source, target_root):
    if not target_root.is_absolute() or '..' in target_root.parts or target_root.anchor != '/':
        raise ValueError("target root must be an absolute container path without traversal")
    manifest = json.loads(pinned_bytes(source / "manifest.json", MANIFEST_SHA256))
    proof = manifest["splits"]["train"]
    root = source / "train"
    records = read_jsonl(root / "inputs.jsonl", proof["records_SHA256"])
    labels = read_jsonl(root / "labels.jsonl", proof["labels_SHA256"])
    inventory = dict(read_jsonl(root / "images.jsonl", proof["image_inventory_SHA256"]))
    if len(records) != len(labels) or len(records) != 216 or len(inventory) != 72:
        raise ValueError("unexpected training split denominator")

    candidates = {topic: [] for topic in TOPIC_QUOTAS}
    for index, (record, label) in enumerate(zip(records, labels)):
        relative = PurePosixPath("train", label["table"] + ".png")
        messages = record["messages"]
        if (relative.parts != ("train", relative.name) or record["images"] != [relative.as_posix()]
                or label["table"] not in inventory
                or [message["role"] for message in messages] != ["user", "assistant"]
                or messages[0]["content"].count("<image>") != 1
                or messages[1]["content"] != label["answer"]):
            raise ValueError(f"invalid image or answer binding at training row {index}")
        candidates[label["topic"]].append((rank(label["id"]), index, label, relative))

    selected = []
    for topic, entries in candidates.items():
        if len(entries) != 72 or len({entry[2]["id"] for entry in entries}) != 72:
            raise ValueError(f"invalid identities or denominator for {topic}")
        selected.extend(sorted(entries)[:TOPIC_QUOTAS[topic]])
    selected.sort()

    verified = {}
    output = []
    identities = []
    for _, index, label, relative in selected:
        table = label["table"]
        if table not in verified:
            raw = pinned_bytes(root / relative.name, inventory[table])
            if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError(f"invalid PNG signature for {relative.name}")
            verified[table] = inventory[table]
        output.append({"messages": records[index]["messages"],
                       "images": [(target_root / relative).as_posix()]})
        identities.append(label["id"])

    # The native sampler must log the actual rank/microbatch order separately.
    raw = b"".join((json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
                   for record in output)
    receipt = {
        "state": "BOUND_TRAINING_EXAMPLES_BEFORE_NATIVE_PROCESSING",
        "source_manifest_SHA256": MANIFEST_SHA256,
        "dataset_SHA256": digest(raw),
        "selected_id_order_SHA256": digest("\n".join(identities).encode("ascii")),
        "selected_image_inventory_SHA256": digest(json.dumps(verified, sort_keys=True).encode("ascii")),
        "source_training_rows": len(records),
        "selected_rows": len(output),
        "selected_unique_images": len(verified),
        "topic_counts": dict(Counter(item[2]["topic"] for item in selected)),
        "pilot_cards": 8,
        "pilot_gradient_accumulation": 4,
        "pilot_optimizer_steps": 5,
        "native_processing_verified": False,
    }
    return raw, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target-root", type=PurePosixPath, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw, receipt = prepare(args.source, args.target_root)
    with args.output.open("xb") as handle:
        handle.write(raw)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
