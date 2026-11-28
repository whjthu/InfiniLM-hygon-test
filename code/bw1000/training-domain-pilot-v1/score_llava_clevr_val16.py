"""Score the sealed OneVision 16-pair CLEVR screen with independent scene labels."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re


ARMS = ("base", "cp25")
SLOTS = range(16)
OUTCOMES = {"eos", "stop", "cap", "error", "missing"}
NUMBER_WORDS = dict(zip("one two three four five six seven eight nine ten".split(), range(1, 11), strict=True))
ITEM = re.compile(rf"(1[0]|[1-9]|{'|'.join(NUMBER_WORDS)})\s+"
                  r"(blue|brown|cyan|gray|grey|green|purple|red|yellow)\s+(cube|cylinder|sphere)s?")


def answer(text):
    if not isinstance(text, str) or not 0 < len(text) <= 8192:
        raise ValueError("Missing or oversized answer")
    parts = re.split(r"[;,\n]+", text.lower().strip().rstrip(". "))
    result = Counter()
    for part in parts:
        item = re.sub(r"^\s*[-*]\s+", "", part).strip()
        match = ITEM.fullmatch(item)
        if match is None:
            raise ValueError("Answer is outside the frozen count and color-shape grammar")
        number, color, shape = match.groups()
        key = ("gray" if color == "grey" else color, shape)
        if key in result:
            raise ValueError("A color-shape pair repeats")
        result[key] = int(number) if number.isdecimal() else NUMBER_WORDS[number]
    return result


def validated(lines, required_sha, limit):
    with lines.open("rb") as stream:
        data = stream.read(limit + 1)
    if not 0 < len(data) <= limit or sha256(data).hexdigest() != required_sha:
        raise ValueError("A sealed input digest or extent differs")
    return [json.loads(line) for line in data.splitlines()]


def score(predictions, references):
    if ([row["slot"] for row in references] != list(SLOTS)
            or any(type(row["slot"]) is not int for row in [*predictions, *references])):
        raise ValueError("Original reference slots differ")
    expected = {r["slot"]: answer(r["reference"]) for r in references}
    keys = [(arm, slot) for arm in ARMS for slot in SLOTS]
    if [(row["arm"], row["slot"]) for row in predictions] != keys:
        raise ValueError("Expected the exact sealed order of 32 distinct generation slots")
    groups, exact_by_slot = {}, {}
    for arm in ARMS:
        statuses = Counter()
        counts = Counter()
        rows = []
        for row in (r for r in predictions if r["arm"] == arm):
            slot, state = row["slot"], row["outcome"]
            if state not in OUTCOMES:
                raise ValueError("Unknown native outcome")
            statuses[state] += 1
            parsed, exact = None, False
            if state == "eos":
                try:
                    parsed = answer(row["text"])
                except ValueError:
                    counts["parse_failures"] += 1
            entry = {"slot": slot, "outcome": state, "parseable": parsed is not None}
            if parsed is not None:
                gold = expected[slot]
                surplus = parsed - gold
                deficit = gold - parsed
                exact = not surplus and not deficit
                measured = {"false_color_shape_pairs": len(parsed.keys() - gold.keys()),
                            "missing_color_shape_pairs": len(gold.keys() - parsed.keys()),
                            "false_object_count": sum(surplus.values()), "missing_object_count": sum(deficit.values())}
                entry.update(measured)
                counts.update(measured)
                counts["parseable_eos"] += 1
                counts["images_with_false_objects"] += bool(surplus)
            counts["exact_multiset"] += exact
            entry["exact_multiset"] = exact
            exact_by_slot[(arm, slot)] = exact
            rows.append(entry)
        report_names = ("parse_failures", "parseable_eos", "exact_multiset", "images_with_false_objects",
                        "false_color_shape_pairs", "missing_color_shape_pairs", "false_object_count", "missing_object_count")
        groups[arm] = {"planned_slots": len(SLOTS), "outcomes": {key: statuses[key] for key in sorted(OUTCOMES)},
                       "generation_failures": len(SLOTS) - statuses["eos"], "slots": rows,
                       **{key: counts[key] for key in report_names}}
    delta = [int(exact_by_slot[("cp25", slot)]) - int(exact_by_slot[("base", slot)]) for slot in SLOTS]
    return {"schema": "LLAVA_CLEVR_VAL16_PAIRED_SCORE_V1", "paired_slots": len(SLOTS), "arms": groups,
            "cp25_wins_ties_losses": [delta.count(1), delta.count(0), delta.count(-1)],
            "exact_gain_percentage_points": 100 * sum(delta) / len(SLOTS), "real_image_quality": None,
            "formal_domain_increment": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("predictions", "references", "out"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    for name in ("predictions-sha", "references-sha"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    result = score(validated(args.predictions, args.predictions_sha, 512 << 10),
                   validated(args.references, args.references_sha, 32 << 10))
    result.update(predictions_SHA256=args.predictions_sha, references_SHA256=args.references_sha)
    with args.out.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
