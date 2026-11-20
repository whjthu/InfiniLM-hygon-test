"""Score paired development generations against the local-only reference."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re


FORM = re.compile(r"(?:FINAL:\s*)?(-?\d+\.\d|\d{4}-\d{2}-\d{2}|[1-7])", re.IGNORECASE)


def rows(path, expected_sha=None):
    raw = path.read_bytes()
    if expected_sha and sha256(raw).hexdigest() != expected_sha:
        raise ValueError("input identity differs")
    items = [json.loads(line) for line in raw.splitlines()]
    mapped = {r["id"]: r for r in items}
    if len(items) != len(mapped):
        raise ValueError("duplicate development identity")
    return mapped, sha256(raw).hexdigest()


def score(predictions, gold):
    if predictions.keys() - gold.keys():
        raise ValueError("foreign development identities")
    results = {}
    for identity, reference in gold.items():
        row = predictions.get(identity)
        if row is None:
            results[identity] = "missing"
            continue
        token = row.get("answer")
        mode = row.get("finish_reason")
        if mode == "length":
            results[identity] = "cap"
            continue
        match = FORM.fullmatch(token.strip()) if mode == "stop" and isinstance(token, str) else None
        if not match:
            results[identity] = "parse_failure"
        elif match[1] == reference["answer"]:
            results[identity] = "correct"
        else:
            results[identity] = "wrong"
    return results


def main(args):
    gold, digest = rows(args.gold, args.gold_sha)
    if len(gold) != 80:
        raise ValueError("expected the complete, frozen 80-item development set")
    arms = {}
    for name in ("base", "domain"):
        prediction, pin = rows(getattr(args, name))
        outcomes = score(prediction, gold)
        counts = {kind: list(outcomes.values()).count(kind)
                  for kind in ("correct", "wrong", "parse_failure", "missing", "cap")}
        arms[name] = dict(prediction_SHA256=pin, counts=counts, exact_accuracy=counts["correct"]/len(gold),
                          outcomes=outcomes)
    print(json.dumps(dict(gold_SHA256=digest, planned=80, arms=arms,
                         difference=arms["domain"]["exact_accuracy"]-arms["base"]["exact_accuracy"],
                         development_only=True), allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("gold", "base", "domain"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--gold-sha", required=True)
    main(parser.parse_args())
