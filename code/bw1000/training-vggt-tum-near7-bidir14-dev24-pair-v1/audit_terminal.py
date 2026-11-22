"""Consume the new pair's exact accounting and independently recompute Dev24."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re

from dev24_decision import evaluate
from read_vggt_dev24_roundoff_audit import audit


LABEL = "training-vggt-tum-near7-bidir14-dev24-pair-v1"
REQUEST = "d282f24a26e14b35b3e50bf02adbf9ff"
PARENT = "1103606"
NODE = "f13r4n09"
PHASES = ("pre", "gpu", "post")
INPUTS_SHA = "3e6401b9b3adc9d6516f8c75c33b827bcf71dcf0e20ccc258f75e8235b2affa6"
SOURCE_HEAD = "ccf658238846acf43bd32c08ab891d6675b39bef93e85b04db13520cd5b3f296"


def document(path):
    if path.resolve(strict=True) != path or not path.is_file():
        raise ValueError("Expected an original regular metadata file")
    with path.open("rb") as stream:
        raw = stream.read((1 << 20) + 1)
    if not 0 < len(raw) <= 1 << 20:
        raise ValueError("Metadata exceeded independent read boundary")
    return json.loads(raw), sha256(raw).hexdigest()


def accounting_rows(intent, outcome, actual):
    if (intent.get("request") != REQUEST or outcome.get("request") != REQUEST
            or not re.fullmatch(r"[0-9a-f]{40}", intent.get("publication", ""))
            or intent["publication"] != outcome.get("publication")
            or intent.get("parent") != PARENT or intent.get("node") != NODE):
        raise ValueError("Original scoring request or publication differs")
    for phase in PHASES:
        row = actual.get(phase)
        if (type(row) is not list or len(row) != 7 or not re.fullmatch(PARENT + r"\.[0-9]+", row[0])
                or row[1] != LABEL + "-" + phase or row[5] != NODE
                or outcome.get("stages", {}).get(phase, {}).get("row") != row):
            raise ValueError("Independent accounting differs from original " + phase)
    steps = [int(actual[phase][0].rsplit(".", 1)[1]) for phase in PHASES]
    if steps != sorted(set(steps)):
        raise ValueError("Original Step order differs")
    return all(actual[phase][2:4] == ["COMPLETED", "0:0"] for phase in PHASES)


def verify_post(root, actual, probe):
    before, _ = document(root / "before.json")
    after, _ = document(root / "after.json")
    current = probe["nodes"][NODE]
    handle = current.get("device_probe_step", "")
    if (before["step"] != actual["pre"][0] or after["step"] != actual["post"][0]
            or any(before[key] != after[key] for key in ("pci", "pci_to_kfd", "boot_id"))
            or len(set(before["pci"])) != 8 or set(current["PCIs"]) != set(before["pci"])
            or current["parent"] != PARENT or current["active_steps"] != []
            or current["KFD_clean"] is not True or current["PID_clean"] is not True
            or current["device_probe_observed"] is not True or not re.fullmatch(PARENT + r"\.[0-9]+", handle)
            or int(handle.rsplit(".", 1)[1]) <= int(actual["post"][0].rsplit(".", 1)[1])):
        raise ValueError("Independent same-parent PCI/KFD/PID cleanup missing")
    return handle


def consume(root, actual, post):
    if root.name != LABEL:
        raise ValueError("Expected the new paired scoring namespace")
    intent, intent_sha = document(root / "score-intent.json")
    outcome, outcome_sha = document(root / "score-outcome.json")
    complete = accounting_rows(intent, outcome, actual)
    result = dict(request=REQUEST, publication=intent["publication"], planned_windows=24,
                  admitted_windows=22, planned_predictions=44, original_missing_windows=2,
                  optimizer_updates=0, promotion=False, accounting=actual,
                  score_intent_SHA256=intent_sha, score_outcome_SHA256=outcome_sha)
    if not complete or outcome.get("completed") is not True or outcome.get("clean_post") is not True:
        return dict(result, state="ORIGINAL_PAIRED_SCORE_INCOMPLETE", independently_audited_predictions=0,
                    decision="MISSING_OR_INCOMPLETE_DEV24")

    probe_step = verify_post(root, actual, post)
    staged, stage_sha = document(root / "stage-complete.json")
    authority, head_sha = document(root / "head-authority.json")
    if (stage_sha != intent["stage_SHA256"] or head_sha != intent["head_authority_SHA256"]
            or head_sha != staged["head_authority_SHA256"] or set(authority) != {"near7_bidir25"}
            or authority["near7_bidir25"]["file_SHA256"] != SOURCE_HEAD):
        raise ValueError("Scoring input or actual archive authority differs")
    reader = Path(__file__).resolve().parent / "tools/audit_vggt_tum_cross_camera_dev.py"
    independent = audit(reader, staged["pair_reader_SHA256"], root / "inputs/inputs-private.json",
                        INPUTS_SHA, head_sha, root / "score/progress-private.jsonl")
    decision = evaluate(independent)
    return dict(result, state="INDEPENDENT_DEV24_CONSUMED", independent_post_step=probe_step,
                independently_audited_predictions=independent["actual_prediction_events"],
                decision=decision, independent=independent)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    parser.add_argument("--actual-accounting", type=Path, required=True)
    parser.add_argument("--independent-post", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    actual, _ = document(args.actual_accounting)
    post, _ = document(args.independent_post)
    result = consume(args.stage, actual, post)
    with args.output.open("x", encoding="utf8") as output:
        json.dump(result, output, sort_keys=True, allow_nan=False)
        output.write("\n")
