"""Score the frozen train-split prompt-format pairs in place."""

import json
from collections import Counter

from prepare_qwen_inputs import digest, pinned_bytes, read_jsonl
from treasury_format_native import ARMS, IDS_SHA256, PREPARATION_RECEIPT_SHA256, PROMPTS_SHA256
from treasury_holdout import interval, normalized, predictions


def score(preparation_receipt, labels, output, generation_sha256):
    preparation = json.loads(pinned_bytes(preparation_receipt, PREPARATION_RECEIPT_SHA256))
    generation = json.loads(pinned_bytes(output / "generation-outcome.json", generation_sha256))
    if (generation["split"], generation["prompts_SHA256"], generation["max_new_tokens"],
        generation["actual_updates"], generation["actual_DCU"], generation["shared_model_load"]) != (
            "train", PROMPTS_SHA256, 64, 0, 1, True):
        raise ValueError("calibration generation scope differs")
    gold = read_jsonl(labels, preparation["private_labels_SHA256"])
    ids = [r["id"] for r in gold]
    if (len(ids) != len(set(ids)) or len(ids) != 19
            or digest(("\n".join(ids) + "\n").encode("ascii")) != IDS_SHA256):
        raise ValueError("calibration labels differ")
    truth = {r["id"]: normalized(r["answer"], "signed_difference") for r in gold}
    if any(v is None for v in truth.values()):
        raise ValueError("a calibration label has invalid syntax")

    arm_stats, correct = {}, {}
    for arm in ARMS:
        arm_info = generation[arm]
        values, signature = predictions(output / (arm + ".jsonl"), truth)
        if arm_info["rows"] != len(ids) or list(values) != ids or arm_info["scoring_file_SHA256"] != signature:
            raise ValueError("calibration predictions differ")
        counts = Counter()
        outcomes = []
        for identity in ids:
            answer = values[identity]
            value = normalized(answer, "signed_difference")
            match = value == truth[identity]
            counts["missing" if answer is None else "invalid_format" if value is None else
                   "correct" if match else "wrong"] += 1
            outcomes.append(int(match))
        correct[arm] = outcomes
        arm_stats[arm] = dict(denominator=len(ids), correct=counts["correct"], wrong=counts["wrong"],
                              invalid_format=counts["invalid_format"], missing=counts["missing"],
                              exact_accuracy=counts["correct"] / len(ids),
                              prediction_SHA256=signature)
    deltas = [b - a for a, b in zip(correct["original"], correct["explicit"], strict=True)]
    return dict(split="train", cases=len(ids), arms=arm_stats, explicit_minus_original=sum(deltas) / len(ids),
                paired_case_bootstrap_95=interval(deltas), formal_holdout=False, formal_effect=False)
