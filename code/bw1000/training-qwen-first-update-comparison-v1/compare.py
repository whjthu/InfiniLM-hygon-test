"""Compare independently read Qwen T01 captures against the frozen A100 reference."""
import argparse
import json
import math
from pathlib import Path
import re

INITIAL = "086ece33e01ff3454ba54f90cb0d30478ed71f172dedc70b1c076e2babd27137"
FIELDS = ("parameter_delta_l2", "preclip_gradient_l2", "postclip_gradient_l2",
          "reported_preclip_norm")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def scalar(value, label):
    require(type(value) in (int, float) and math.isfinite(value), label)
    return value


def difference(target, reference, label):
    a, b = scalar(target, label), scalar(reference, label)
    return dict(target=a, a100=b, absolute=abs(a-b),
                relative=abs(a-b)/abs(b) if b else None)


def summarize(rows, binding):
    require(type(rows) is list and len(rows) == 8, "expected eight ranks")
    for rank, row in enumerate(rows):
        require(type(row["rank"]) is int and row["rank"] == rank, "missing or out-of-order rank")
        require(row["initial_named_parameter_SHA256"] == INITIAL, "different initial parameters")
        require(len(row["micro_forwards"]) == 8 and len(row["updates"]) == 2,
                "incomplete microforward or optimizer capture")
        for i, (forward, pin) in enumerate(zip(row["micro_forwards"], binding[str(rank)]), 1):
            require(forward["ordinal"] == i and forward["global_step"] == (i-1)//4,
                    "wrong accumulation order")
            require(forward["input_content"]["SHA256"] == pin
                    and forward["input_content"]["finite"] is True
                    and forward["output_content"]["finite"] is True, "invalid input or output evidence")
            scalar(forward["loss"], "nonfinite loss")
        for i, update in enumerate(row["updates"], 1):
            require(update["update"] == i and update["learning_rates"] == ([0.0] if i == 1 else [0.0001]),
                    "wrong optimizer order or learning rate")
            require(update["changed_parameter_tensors"] == (0 if i == 1 else 504),
                    "unexpected parameter update boundary")
            for field in FIELDS:
                require(scalar(update[field], field) >= 0, "negative norm")
            require((update["parameter_delta_l2"] == 0) == (i == 1), "wrong update magnitude boundary")
            digest = update["optimizer_after_SHA256"]
            require(isinstance(digest, str) and re.fullmatch("[0-9a-f]{64}", digest),
                    "missing optimizer artifact SHA")


def compare(reference, target, binding):
    require(set(binding) == {str(i) for i in range(8)}
            and all(len(pins) == 8 for pins in binding.values()), "incomplete frozen input binding")
    summarize(reference, binding)
    summarize(target, binding)
    ranks = []
    for a, t in zip(reference, target):
        forwards = []
        for x, y in zip(a["micro_forwards"], t["micro_forwards"]):
            for field in ("label_tokens", "label_sequence_length", "input_sequence_length", "logit_shape"):
                require(x[field] == y[field], f"different forward {field}")
            forwards.append(dict(ordinal=y["ordinal"], loss=difference(y["loss"], x["loss"], "loss"),
                                 output_digest_matches=y["output_content"]["SHA256"] == x["output_content"]["SHA256"],
                                 output_l2=difference(y["output_content"]["floating_l2"],
                                                      x["output_content"]["floating_l2"], "output norm")))
        updates = []
        for x, y in zip(a["updates"], t["updates"]):
            updates.append(dict(update=y["update"], **{k: difference(y[k], x[k], k) for k in FIELDS}))
        ranks.append(dict(rank=t["rank"], forwards=forwards, updates=updates))
    return dict(state="MATCHED_STRUCTURAL_EVIDENCE_NUMERICS_PENDING",
                ranks=ranks, forwards=64, update_records=16, numerical_acceptance=False,
                pending=["original thresholds", "elementwise output, gradient, parameter and optimizer comparison"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("binding", type=Path)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_bytes())["reference"]["ranks"]
    target = json.loads(args.target.read_bytes())["ranks"]
    binding = json.loads(args.binding.read_bytes())["input_fingerprints_by_rank"]
    print(json.dumps(compare(reference, target, binding), allow_nan=False))


if __name__ == "__main__":
    main()
