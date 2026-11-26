"""Aggregate the frozen Gemma 96-question confirmation without raw text."""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import json
from pathlib import Path
from random import Random


ARMS = ("base", "domain25", "question25")
KINDS = (
    "maximum_reported_number",
    "minimum_reported_number",
    "maximum_depth",
    "minimum_depth",
    "depth_change",
    "reported_number_change",
)
SLOTS_PER_ARM = 96
RESAMPLES = 50_000
SEED = 20261126


def require(condition, message):
    if not condition:
        raise ValueError(message)


def interval(differences):
    rng = Random(SEED)
    outcomes = sorted(
        sum(rng.choice(differences) for _ in differences)
        for _ in range(RESAMPLES)
    )
    bounds = []
    for numerator in (1, 39):
        lower, remainder = divmod((RESAMPLES - 1) * numerator, 40)
        bounds.append(
            Fraction((40 - remainder) * outcomes[lower] + remainder * outcomes[lower + 1],
                     40 * SLOTS_PER_ARM)
        )
    return bounds


def aggregate(rows):
    require(len(rows) == SLOTS_PER_ARM, "Expected all 96 questions")
    require(len({row["id"] for row in rows}) == SLOTS_PER_ARM, "Repeated question")
    groups = defaultdict(list)
    for row in rows:
        require(row["kind"] in KINDS, "Unknown question kind")
        require(set(row["arms"]) == set(ARMS), "Expected one result for each arm")
        for arm in ARMS:
            entry = row["arms"][arm]
            require(set(entry) == {"correct", "disposition"}, "Unexpected result fields")
            require(type(entry["correct"]) is bool, "Expected Boolean correctness")
            require(entry["disposition"] in ("completed", "cap", "error", "missing"),
                    "Unknown terminal disposition")
            require(not entry["correct"] or entry["disposition"] == "completed",
                    "Incorrect full denominator handling")
        groups[row["group"]].append(row)
    require(len(groups) == 8, "Expected eight independent groups")
    expected_kinds = {kind: 2 for kind in KINDS}
    require(all(Counter(row["kind"] for row in group) == expected_kinds
                for group in groups.values()), "Group or question kind denominator differs")

    report = {"questions": SLOTS_PER_ARM, "groups": len(groups),
              "formal_delivery_credit": 0, "arms": {}, "comparisons": {}}
    infrastructure_failure = False
    for arm in ARMS:
        counts = Counter(row["arms"][arm]["disposition"] for row in rows)
        infrastructure_failure |= bool(counts["error"] or counts["missing"])
        report["arms"][arm] = {
            "correct": sum(row["arms"][arm]["correct"] for row in rows),
            "denominator": SLOTS_PER_ARM,
            "dispositions": {state: counts[state] for state in
                             ("completed", "cap", "error", "missing")},
            "by_kind": {
                kind: {"correct": sum(row["arms"][arm]["correct"] for row in rows
                                      if row["kind"] == kind), "denominator": 16}
                for kind in KINDS
            },
        }
    for other in ("base", "question25"):
        group_differences = []
        for name in sorted(groups):
            categories = Counter(int(row["arms"]["domain25"]["correct"])
                                 - int(row["arms"][other]["correct"]) for row in groups[name])
            group_differences.append({"wins": categories[1], "losses": categories[-1],
                                      "ties": categories[0], "denominator": 12})
        differences = [group["wins"] - group["losses"] for group in group_differences]
        gain = Fraction(sum(differences), SLOTS_PER_ARM)
        low, high = interval(differences)
        report["comparisons"]["domain25_vs_" + other] = {
            "group_differences": group_differences,
            "gain_percentage_points": float(gain * 100),
            "paired_group_percentile_95_percentage_points": [float(low * 100), float(high * 100)],
            "group_resamples": RESAMPLES,
            "seed": SEED,
        }
        if other == "base":
            report["confirmatory_effect_pass"] = (
                not infrastructure_failure and gain >= Fraction(3, 100) and low > 0
            )
    report["infrastructure_failure"] = infrastructure_failure
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scored_slots", type=Path)
    parser.add_argument("aggregate_output", type=Path)
    args = parser.parse_args()
    with args.scored_slots.open("rb") as stream:
        raw = stream.read(128_001)
    require(0 < len(raw) <= 128_000, "Expected bounded scoring dispositions")
    report = aggregate(json.loads(raw))
    with args.aggregate_output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, allow_nan=False, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
