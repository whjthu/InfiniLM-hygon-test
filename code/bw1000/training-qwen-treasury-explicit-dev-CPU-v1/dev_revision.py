"""Derive one answer-free dev108 instruction revision; reuse all original images."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath

OLD_INVENTORY_SHA = "c12efcc46337fe9130ac1ec6defa1d513276640986175d4cf3ed36d9cdc5c903"
TOPICS = ("lookup", "signed_difference", "maximum_date")
FORMAT_INSTRUCTION = (
    " Output only the amount as a decimal with exactly two digits after the decimal point. "
    "Use a minus sign for negative amounts and no currency symbol, commas, or explanation."
)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def require(ok, why):
    if not ok:
        raise ValueError(why)


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf8")


def build(original, inventory):
    require(not original.is_symlink() and original.is_dir() and not inventory.is_symlink(), "aliased original dev")
    raw_inventory = inventory.read_bytes()
    require(digest(raw_inventory) == OLD_INVENTORY_SHA, "original dev inventory differs")
    pins = json.loads(raw_inventory)
    require(len(pins) == 39 and {"manifest.json", "visual.jsonl", "text-only.jsonl"} <= set(pins), "original39files differ")
    observed = {p.relative_to(original).as_posix() for p in original.rglob("*") if p.is_file()}
    require(observed == set(pins) and all(not p.is_symlink() for p in original.rglob("*")), "original dev file set differs")
    raw = {}
    for name, pin in pins.items():
        path = PurePosixPath(name)
        require(not path.is_absolute() and ".." not in path.parts, "unsafe original dev member")
        data = (original / name).read_bytes()
        require(len(data) <= 8 * 1024**2 and digest(data) == pin, "original dev content differs")
        if name in ("manifest.json", "visual.jsonl", "text-only.jsonl"):
            raw[name] = data
    visual = [json.loads(line) for line in raw["visual.jsonl"].splitlines()]
    text = [json.loads(line) for line in raw["text-only.jsonl"].splitlines()]
    manifest = json.loads(raw["manifest.json"])
    require(len(visual) == len(text) == 108 and len({r["id"] for r in visual}) == 108
            and manifest["split"] == "dev" and manifest["requests"] == 108 and manifest["tables"] == 36,
            "original dev denominator differs")
    revised, revised_text, counts, changed = deepcopy(visual), deepcopy(text), Counter(), 0
    for index, (before, before_text) in enumerate(zip(visual, text, strict=True)):
        topic = next((x for x in TOPICS if before["id"].endswith("_" + x)), None)
        require(topic is not None and set(before) == {"id", "images", "messages"}
                and set(before_text) == {"id", "messages"} and before_text["id"] == before["id"]
                and len(before["messages"]) == len(before_text["messages"]) == 1
                and before["messages"][0]["role"] == before_text["messages"][0]["role"] == "user",
                "answer-free row or topic differs")
        prompt = before["messages"][0]["content"]
        require(prompt.startswith("<image>\n") and prompt.count("<image>") == 1
                and before_text["messages"][0]["content"] == prompt[8:]
                and FORMAT_INSTRUCTION not in prompt and len(before["images"]) == 1
                and before["images"][0] in pins, "original visual/text pairing differs")
        counts[topic] += 1
        if topic == "signed_difference":
            revised[index]["messages"][0]["content"] += FORMAT_INSTRUCTION
            revised_text[index]["messages"][0]["content"] += FORMAT_INSTRUCTION
            changed += 1
        require(revised[index]["id"] == before["id"] and revised[index]["images"] == before["images"], "row identity changed")
    require(counts == {x: 36 for x in TOPICS} and changed == 36, "topic or instruction-change denominator differs")
    payload = {"visual.jsonl": b"".join(encoded(x) for x in revised),
               "text-only.jsonl": b"".join(encoded(x) for x in revised_text)}
    manifest.update(visual_SHA256=digest(payload["visual.jsonl"]), text_only_SHA256=digest(payload["text-only.jsonl"]),
                    original_dev_inventory_SHA256=OLD_INVENTORY_SHA, changed_user_instructions=36,
                    revision="same explicit signed-difference instruction for all three development arms")
    payload["manifest.json"] = encoded(manifest)
    new_pins = dict(pins)
    new_pins.update({name: digest(data) for name, data in payload.items()})
    receipt = dict(split="dev", rows=108, tables=36, topic_counts=dict(counts), changed_user_instructions=36,
                   images_changed=0, IDs_changed=0, row_order_changed=False, labels_opened=False,
                   old_inventory_SHA256=OLD_INVENTORY_SHA, format_instruction=FORMAT_INSTRUCTION,
                   inventory_SHA256=digest(encoded(new_pins)), max_new_tokens=64,
                   arms=["base", "old_domain", "domain"], actual_generations=0)
    return payload, encoded(new_pins), receipt


def verify_payload(original, inventory, revised, revised_inventory):
    payload, expected_inventory, receipt = build(original, inventory)
    require(not revised.is_symlink() and revised_inventory.read_bytes() == expected_inventory, "revised inventory differs")
    require({p.name for p in revised.iterdir()} == {"manifest.json", "visual.jsonl", "text-only.jsonl", "media"}
            and (revised / "media").is_dir() and not (revised / "media").is_symlink()
            and not any((revised / "media").iterdir()), "HOST revised payload must contain only metadata and empty media bindpoint")
    for name, raw in payload.items():
        require(not (revised / name).is_symlink() and (revised / name).read_bytes() == raw, "revised dev metadata differs")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--original-inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload, inventory, receipt = build(args.original, args.original_inventory)
    args.output.mkdir()
    (args.output / "dev/media").mkdir(parents=True)
    for name, raw in payload.items():
        (args.output / "dev" / name).write_bytes(raw)
    (args.output / "inventory.json").write_bytes(inventory)
    (args.output / "receipt.json").write_bytes(encoded(receipt))
    print(json.dumps(receipt, ensure_ascii=False))
