"""Score the fixed 36 arithmetic cases without exporting generated text."""

import hashlib
import json

from treasury_holdout import DEV_ARMS, holdout, score_dev_pair


TOPIC = "signed_difference"
CASE_IDS_SHA256 = "9afa5fb0d15ab1f6b797a75348546545e93daeab3993d2e4f34264c65b1055ad"
RESCALE = 3


def score(source, output, pins, terminal_generation_sha):
    receipt = (output / "generation-outcome.json").read_bytes()
    if hashlib.sha256(receipt).hexdigest() != terminal_generation_sha:
        raise ValueError("generation receipt pin differs")
    generation = json.loads(receipt)
    if (generation["actual_DCU"], generation["topic"], generation["planned_rows_per_arm"],
        generation["max_new_tokens"], generation["case_ids_SHA256"]) != (1, TOPIC, 36, 128, CASE_IDS_SHA256):
        raise ValueError("generation receipt scope differs")
    _, _, rows, _ = holdout(source, "dev")
    ids = [identity for identity, _, topic, _ in rows if topic == TOPIC]
    if hashlib.sha256(("\n".join(ids) + "\n").encode("ascii")).hexdigest() != CASE_IDS_SHA256:
        raise ValueError("development arithmetic identity differs")

    paths = {arm: output / (arm + ".jsonl") for arm in DEV_ARMS}
    for arm, path in paths.items():
        content = path.read_bytes()
        if (hashlib.sha256(content).hexdigest() != pins[arm]
                or generation[arm]["scoring_file_SHA256"] != pins[arm]
                or generation[arm]["rows"] != len(ids)):
            raise ValueError("paired output pin differs")
        records = [json.loads(line) for line in content.splitlines()]
        if [record["id"] for record in records] != ids:
            raise ValueError("scored IDs or denominator differ")

    report = score_dev_pair(source, paths)
    totals = {}
    for arm in DEV_ARMS:
        row = report["arms"][arm]
        if row["prediction_SHA256"] != pins[arm]:
            raise ValueError("prediction changed during scoring")
        if any(bucket["missing"] != len(ids) for name, bucket in row["scores"].items()
               if name not in ("all", TOPIC)):
            raise ValueError("other topics leaked into arithmetic score")
        totals[arm] = row["scores"][TOPIC]

    # The frozen scorer resamples complete tables. Two topics have zero paired
    # difference on every table, so scaling recovers the one-topic comparison.
    comparison = report["comparisons"]["domain_minus_base"]
    return dict(topic=TOPIC, rows_per_arm=len(ids), source_manifest_SHA256=report["source_manifest_SHA256"],
                case_ids_SHA256=CASE_IDS_SHA256, predictions_SHA256=pins, max_new_tokens=128,
                arms=totals, domain_minus_base=RESCALE * comparison["accuracy_delta"],
                paired_table_bootstrap_95=[RESCALE * n for n in comparison["paired_table_bootstrap_95"]])
