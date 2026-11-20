"""Audit the VGGT RGB development prediction log against the frozen cohort."""

import argparse
import json
from math import isclose, isfinite
from pathlib import Path

from bind_vggt_tum_dev_rgb import read_bounded, sha
from bind_vggt_tum_cross_camera_heads import is_sha
from score_vggt_tum_cross_camera_dev import ARMS, WINDOWS, canonical_scorer, pinned_json, summarize


ANGLE_ABS_TOLERANCE = 2e-5


def key(row):
    return row["sequence"], row["window"]


def events(path):
    raw = read_bounded(path, 2 * 1024**2)
    lines = raw.splitlines()
    if not raw.endswith(b"\n") or len(lines) < 2:
        raise ValueError("prediction progress log is incomplete")
    return [json.loads(line) for line in lines], sha(raw)


def audit(inputs, inputs_sha, heads_sha, progress):
    if not is_sha(heads_sha):
        raise ValueError("expected the frozen SHA for both trained heads")
    manifest = pinned_json(inputs, inputs_sha)
    originals = {key(row): row for row in manifest["windows"]}
    if len(manifest["windows"]) != len(WINDOWS) or set(originals) != WINDOWS:
        raise ValueError("expected the complete 24-window development manifest")
    records, progress_sha = events(progress)
    planned, *prediction_events, terminal = records
    if (planned.get("state") != "PLANNED" or planned["inputs_SHA256"] != inputs_sha
            or planned["heads_SHA256"] != heads_sha or terminal.get("state") != "COMPLETED"
            or terminal.get("optimizer_updates") != 0 or terminal.get("formal_CO3D") is not False):
        raise ValueError("prediction log scope or terminal outcome differs")
    observed = {key(row): row for row in planned["rows"]}
    if len(planned["rows"]) != len(WINDOWS) or set(observed) != WINDOWS:
        raise ValueError("prediction log lost the originally planned windows")
    arms = ("base", *ARMS)
    ready = []
    for identity, row in originals.items():
        entry = observed[identity]
        eligible = row["state"] == "RGB224_AND_RELATIVE_POSE_BOUND"
        pin = row["rgb_file_SHA256"] if eligible else None
        status = "NOT_RUN" if eligible else "MISSING_INPUT"
        if (entry["input_state"] != row["state"] or entry.get("input_SHA256") != pin
                or any(entry[arm] != dict(pair_error_deg=None, state=status) for arm in arms)):
            raise ValueError("planned input state differs from the frozen cohort")
        if eligible:
            ready.append(identity)
    scheduled = [(identity, arm) for arm in arms for identity in sorted(ready)]
    if len(scheduled) != len(prediction_events):
        raise ValueError("prediction event count differs from the three-arm schedule")

    scorer = canonical_scorer()
    invalid = dict.fromkeys(arms, 0)
    raw_predictions = dict.fromkeys(arms, 0)
    for (identity, arm), event in zip(scheduled, prediction_events, strict=True):
        if key(event) != identity or event["arm"] != arm:
            raise ValueError("prediction event is missing, duplicated or out of order")
        result = event["result"]
        elapsed = result["forward_s"]
        if type(elapsed) not in (int, float) or not isfinite(elapsed) or elapsed < 0:
            raise ValueError("invalid original forward duration")
        if result["state"] == "INVALID_PREDICTION":
            if result["pair_error_deg"] is not None or "extrinsics" in result:
                raise ValueError("invalid predictions must retain a failed row")
            invalid[arm] += 1
        else:
            recovered = scorer["angular_errors"](result["extrinsics"], originals[identity]["world_to_camera"])
            expected = "DEGENERATE_TRANSLATION" if recovered["predicted_translation_degenerate"] else "PREDICTED"
            if result["state"] != expected or any(
                    (not isinstance(result[name], (int, float))
                     or not isclose(result[name], value, rel_tol=0, abs_tol=ANGLE_ABS_TOLERANCE))
                    if type(value) is float else result[name] != value for name, value in recovered.items()):
                raise ValueError("recorded angles differ from original extrinsics")
            raw_predictions[arm] += 1
        observed[identity][arm] = result

    rebuilt = summarize([observed[identity] for identity in sorted(observed)], scorer["summarize_development"])
    if (rebuilt != terminal["summary"] or terminal["planned_windows"] != len(WINDOWS)
            or terminal["input_pairs"] != len(ready)):
        raise ValueError("terminal score or denominator differs from the original prediction rows")
    return dict(progress_SHA256=progress_sha, assessed_windows=len(WINDOWS), input_pairs=len(ready),
                actual_prediction_events=len(prediction_events), raw_pose_predictions=raw_predictions,
                invalid_predictions=invalid, angular_replay_absolute_tolerance_degrees=ANGLE_ABS_TOLERANCE,
                AUC=rebuilt, optimizer_updates=0, formal_CO3D=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("inputs", "progress", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    for flag in ("inputs-sha", "heads-sha"):
        parser.add_argument("--" + flag, required=True)
    args = parser.parse_args()
    result = audit(args.inputs, args.inputs_sha, args.heads_sha, args.progress)
    with args.output.open("x", encoding="utf8") as stream:
        json.dump(result, stream, sort_keys=True, allow_nan=False)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
