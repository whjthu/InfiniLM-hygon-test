"""Score the original Treasury development outputs where the job stored them."""

import argparse
import json
from pathlib import Path

from prepare_qwen_inputs import digest
from treasury_holdout import DEV_ARMS, score_dev_pair


PREDICTIONS = {
    "base": (7503, "23c39f516bc35493033403f9b38df33e861a938101a473a18ffd23f8630f7e3e"),
    "domain": (7509, "4523e0b95f8fe0e7facc69f777670928dcbf9ecb00f2ded6d8b7a2e4ca74dddb"),
}
GENERATION_OUTCOME_SHA256 = "d98bec26f656ba4b47280d709bb61de51d765c37f3b0be454e3cbccbcb0f50bc"


def verified_score(source, output):
    outcome = (output / "generation-outcome.json").read_bytes()
    if digest(outcome) != GENERATION_OUTCOME_SHA256:
        raise ValueError("generation outcome pin differs")
    # The scorer opens the original files once. Its returned digests bind the
    # bytes actually scored, including any change after this size check.
    files = {arm: output / (arm + ".jsonl") for arm in DEV_ARMS}
    for arm, file in files.items():
        if file.stat().st_size != PREDICTIONS[arm][0]:
            raise ValueError("prediction size differs")
    score = score_dev_pair(source, files)
    for arm in DEV_ARMS:
        if score["arms"][arm]["prediction_SHA256"] != PREDICTIONS[arm][1]:
            raise ValueError("scored prediction pin differs")
    return score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--original-output", required=True, type=Path)
    args = parser.parse_args()
    try:
        report = verified_score(args.source, args.original_output)
    except (OSError, ValueError, TypeError, KeyError, ArithmeticError) as exc:
        # Sanitized failure output must not include labels or predictions.
        print(json.dumps({"passed": False, "error_class": type(exc).__name__}))
        return 1
    print(json.dumps({"passed": True, "aggregate": report}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
