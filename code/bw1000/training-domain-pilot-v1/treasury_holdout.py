"""Prepare answer-free requests and score Treasury holdout predictions offline."""

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from datetime import date
from decimal import Decimal
from pathlib import Path

from prepare_qwen_inputs import MANIFEST_SHA256, digest, pinned_bytes, read_jsonl


TOPICS = ("lookup", "signed_difference", "maximum_date")
TABLES = {"dev": 36, "blind": 18}
ARMS = ("base", "domain", "generic", "domain_text_only")
TABLE_ID = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{4}-[0-9]{2}-[0-9]{2}")
DAY = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
DOLLARS = re.compile(r"[+-]?(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)\.[0-9]{2}")


def jsonl(rows):
    return b"".join((json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
                    for row in rows)


def holdout(source, split):
    manifest = json.loads(pinned_bytes(source / "manifest.json", MANIFEST_SHA256))
    proof = manifest["splits"][split]
    root = source / split
    raw_inventory = read_jsonl(root / "images.jsonl", proof["image_inventory_SHA256"])
    images = dict(raw_inventory)
    records = read_jsonl(root / "inputs.jsonl", proof["records_SHA256"])
    count = TABLES[split]
    if len(images) != count or len(raw_inventory) != count or len(records) != len(TOPICS) * count:
        raise ValueError("Unexpected holdout denominator")
    rows = []
    for offset in range(0, len(records), len(TOPICS)):
        table = raw_inventory[offset // len(TOPICS)][0]
        if not TABLE_ID.fullmatch(table):
            raise ValueError("Invalid table identity")
        for index, topic in enumerate(TOPICS):
            record = records[offset + index]
            messages = record["messages"]
            if (set(record) != {"messages", "images"} or record["images"] != [f"{split}/{table}.png"]
                    or len(messages) != 1 or set(messages[0]) != {"role", "content"}
                    or messages[0]["role"] != "user"):
                raise ValueError("Unexpected holdout request schema")
            content = messages[0]["content"]
            if not content.startswith("<image>\n") or content.count("<image>") != 1:
                raise ValueError("Unexpected image marker")
            rows.append((f"{table}_{topic}", table, topic, content))
    return root, proof, rows, images


def prepare(source, split, output):
    # Training and serving never need the answer file. This command does not open it.
    root, proof, rows, images = holdout(source, split)
    media = {name: pinned_bytes(root / f"{name}.png", pin) for name, pin in images.items()}
    visual = jsonl({"id": identity, "messages": [{"role": "user", "content": prompt}],
                    "images": [f"media/{table}.png"]} for identity, table, _, prompt in rows)
    text = jsonl({"id": identity, "messages": [{"role": "user", "content": prompt.removeprefix("<image>\n")}]}
                 for identity, _, _, prompt in rows)
    output.mkdir()
    (output / "media").mkdir()
    for name, raw in media.items():
        (output / "media" / f"{name}.png").write_bytes(raw)
    (output / "visual.jsonl").write_bytes(visual)
    (output / "text-only.jsonl").write_bytes(text)
    receipt = {"source_manifest_SHA256": MANIFEST_SHA256, "split": split, "tables": len(images),
               "requests": len(rows), "source_images_SHA256": proof["image_inventory_SHA256"],
               "visual_SHA256": digest(visual), "text_only_SHA256": digest(text)}
    (output / "manifest.json").write_bytes(jsonl([receipt]))
    return receipt


def normalized(answer, topic):
    if answer is None:
        return None
    answer = answer.strip()
    if topic == "maximum_date" and DAY.fullmatch(answer):
        try:
            return date.fromisoformat(answer)
        except ValueError:
            return None
    if topic != "maximum_date" and DOLLARS.fullmatch(answer):
        return Decimal(answer.replace(",", ""))
    return None


def predictions(path, known):
    raw = path.read_bytes()
    by_id = {}
    for line in raw.splitlines():
        record = json.loads(line)
        if (not isinstance(record, dict) or set(record) != {"id", "answer"}
                or not isinstance(record["id"], str) or record["id"] not in known
                or record["id"] in by_id
                or not (record["answer"] is None or isinstance(record["answer"], str))):
            raise ValueError("Unexpected, unknown, or repeated prediction")
        by_id[record["id"]] = record["answer"]
    return by_id, digest(raw)


def interval(deltas, samples=10000):
    # Resample whole tables, using the same table draw for both arms.
    rng = random.Random(20261117)
    totals = sorted(sum(rng.choices(deltas, k=len(deltas))) / len(deltas) for _ in range(samples))
    return [totals[int((samples - 1) * tail)] for tail in (0.025, 0.975)]


def score(source, split, files):
    root, proof, rows, images = holdout(source, split)
    labels = read_jsonl(root / "labels.jsonl", proof["labels_SHA256"])
    identities = [identity for identity, _, _, _ in rows]
    if len(labels) != len(rows) or [label["id"] for label in labels] != identities:
        raise ValueError("Unexpected label IDs or order")
    truth = {}
    for (identity, table, topic, _), label in zip(rows, labels):
        if label["table"] != table or label["topic"] != topic or not isinstance(label["answer"], str):
            raise ValueError("Unexpected label table, topic, or type")
        gold = normalized(label["answer"], topic)
        if gold is None:
            raise ValueError("Invalid frozen gold answer format")
        truth[identity] = gold
    report = {"source_manifest_SHA256": MANIFEST_SHA256, "split": split, "rows": len(rows),
              "tables": len(images), "labels_SHA256": proof["labels_SHA256"], "arms": {}, "comparisons": {}}
    matches = {}
    for arm in ARMS:
        values, sha = predictions(files[arm], truth)
        buckets = defaultdict(Counter)
        matches[arm] = {}
        for identity, _, topic, _ in rows:
            answer = values.get(identity)
            result = normalized(answer, topic)
            outcome = ("missing" if answer is None else "invalid_format" if result is None
                       else "correct" if result == truth[identity] else "wrong")
            matches[arm][identity] = int(outcome == "correct")
            for key in (topic, "all"):
                buckets[key][outcome] += 1
        measures = {key: {"denominator": sum(counts.values()), "accuracy": counts["correct"] / sum(counts.values()),
                          **{outcome: counts[outcome] for outcome in ("correct", "missing", "invalid_format", "wrong")}}
                    for key, counts in buckets.items()}
        report["arms"][arm] = {"prediction_SHA256": sha, "scores": measures}
    for arm in ("base", "generic", "domain_text_only"):
        by_table = defaultdict(list)
        for identity, table, _, _ in rows:
            by_table[table].append(matches["domain"][identity] - matches[arm][identity])
        deltas = [sum(values) / len(values) for values in by_table.values()]
        report["comparisons"][f"domain_minus_{arm}"] = {
            "accuracy_delta": sum(deltas) / len(deltas), "paired_table_bootstrap_95": interval(deltas)}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    for mode in ("prepare", "score"):
        command = modes.add_parser(mode)
        command.add_argument("--source", type=Path, required=True)
        command.add_argument("--split", choices=TABLES, required=True)
        command.add_argument("--output", type=Path, required=True)
        if mode == "score":
            for arm in ARMS:
                command.add_argument("--" + arm.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        report = prepare(args.source, args.split, args.output)
    else:
        report = score(args.source, args.split, {arm: getattr(args, arm) for arm in ARMS})
        with args.output.open("xb") as handle:
            handle.write(jsonl([report]))
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
