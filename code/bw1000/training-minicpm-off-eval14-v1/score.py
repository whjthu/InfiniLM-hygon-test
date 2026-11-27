"""Score two sealed MiniCPM packaging prediction files against the local holdout."""

import argparse
from collections import Counter, defaultdict
from decimal import Decimal
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import re


NUMBER_UNIT = re.compile(r"([0-9]{1,10}(?:[.,][0-9]{1,8})?)\s*(mg|kg|g|ml|cl|l)", re.IGNORECASE)
RAW_FINISHES = {"stop", "eos", "length", "cap", "error"}
LIMIT = 256 * 1024


def read_pinned(path, expected):
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("invalid SHA-256")
    with Path(path).open("rb") as handle:
        data = handle.read(LIMIT + 1)
    if not 0 < len(data) <= LIMIT or sha256(data).hexdigest() != expected:
        raise ValueError(f"sealed file changed or exceeds size limit: {Path(path).name}")
    return data


def indexed(data, keys):
    result = {}
    for line in data.splitlines():
        row = json.loads(line)
        if set(row) != keys or not re.fullmatch(r"pkg-eval-[0-9]{2}", row["slot"]) or row["slot"] in result:
            raise ValueError("invalid or duplicate slot")
        result[row["slot"]] = row
    return result


def normalize(answer):
    if not isinstance(answer, str) or len(answer) > 256:
        return None
    text = answer.strip().rstrip(".").strip()
    if text.casefold() == "not readable":
        return ("refusal",)
    match = NUMBER_UNIT.fullmatch(text)
    if match:
        value, unit = match.groups()
        return ("quantity", Decimal(value.replace(",", ".")), unit.casefold())
    return None


def grade(gold, predictions):
    counts = Counter()
    correct = {}
    for slot, expected in gold.items():
        row = predictions.get(slot)
        counts[f"{expected[0]}_planned"] += 1
        if row is None:
            counts["missing"] += 1
            correct[slot] = False
            continue
        finish = row["finish_reason"]
        if finish not in RAW_FINISHES:
            raise ValueError("unknown generation finish reason")
        counts[finish] += 1
        value = normalize(row["answer"]) if finish in {"stop", "eos"} else None
        correct[slot] = value == expected
        if finish in {"stop", "eos"} and value is None:
            counts["parse_failure"] += 1
        if value and expected[0] == "refusal" and value[0] != "refusal":
            counts["false_readable"] += 1
        if correct[slot]:
            counts[f"{expected[0]}_correct"] += 1
    return dict(counts, planned=len(gold), correct=sum(correct.values())), correct


def score(packet, gold, base, adapter):
    if not set(base) <= set(packet) or not set(adapter) <= set(packet) or set(packet) != set(gold):
        raise ValueError("prediction or answer slots differ from the frozen prompts")
    groups = Counter(row["group"] for row in packet.values())
    gold = {slot: normalize(row["answer"]) for slot, row in gold.items()}
    if (len(groups) != 7 or set(groups.values()) != {2} or None in gold.values()
            or Counter(x[0] for x in gold.values()) != {"quantity": 7, "refusal": 7}):
        raise ValueError("frozen pilot denominator changed")
    before, a = grade(gold, base)
    after, b = grade(gold, adapter)
    paired = defaultdict(int)
    for slot, row in packet.items():
        paired[row["group"]] += int(b[slot]) - int(a[slot])
    delta = sum(paired.values())
    distribution = [
        sum(direction * value for direction, value in zip(directions, paired.values()))
        for directions in product((-1, 1), repeat=len(paired))
    ]
    return {
        "planned": len(packet),
        "product_groups": len(paired),
        "base": before,
        "adapter": after,
        "correct_difference": delta,
        "accuracy_difference_pp": 100 * delta / len(packet),
        "paired_correct_difference_by_group": dict(paired),
        "two_sided_group_signflip_p": sum(abs(x) >= abs(delta) for x in distribution) / len(distribution),
        "formal_domain_delivery": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--adapter-sha256", required=True)
    args = parser.parse_args()
    if args.base.resolve() == args.adapter.resolve():
        raise ValueError("prediction arms require separate sealed files")
    manifest = json.loads(read_pinned(args.cohort / "frozen-pilot-manifest.json", args.manifest_sha256))
    pins = manifest["input_files_SHA256"]
    pred_keys = {"slot", "answer", "finish_reason"}
    base = indexed(read_pinned(args.base, args.base_sha256), pred_keys)
    adapter = indexed(read_pinned(args.adapter, args.adapter_sha256), pred_keys)
    packet = indexed(
        read_pinned(args.cohort / "prompts.answerless.jsonl", pins["prompts.answerless.jsonl"]),
        {"slot", "group", "image", "image_SHA256", "question"},
    )
    gold = indexed(read_pinned(args.cohort / "gold.private.jsonl", pins["gold.private.jsonl"]), {"slot", "answer"})
    result = score(packet, gold, base, adapter)
    result["source_sha256"] = {
        "manifest": args.manifest_sha256, "base": args.base_sha256, "adapter": args.adapter_sha256,
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
