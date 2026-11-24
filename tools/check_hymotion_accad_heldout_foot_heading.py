"""Check the marker-to-skeleton baseline on the frozen B11 held-out take."""

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

from build_hymotion_accad_neutral_pilot import NEUTRAL, neutral_reference, reconstruct_features
from check_hymotion_accad_foot_heading import compare
from check_hymotion_accad_lower_leg_phase import (
    ARCHIVES, COHORT_CALIBRATION_SHA, COHORTS, SELECTION_SHA, WINDOW,
    read_archive, read_source_frames, source_positions, source_toes, stance_intervals, verified_json,
)
from verify_hymotion_publisher_feature_assets import inspect_wooden, load_geometry


def calibration_passes(witness):
    return (witness["axes"] == [0, 2, 1] and witness["signs"] == [-1, 1, 1]
            and abs(witness["C3D_mm_per_BVH_unit"] - 10) < 0.05 and witness["holdout_RMS_mm"] < 40)


def input_for_take(directory, cohort, subject, witness, row, geometry, target_joints):
    _, size, original_pin, split = ARCHIVES[row["archive"]]
    stand_name, stand_pin = NEUTRAL[subject]
    with read_archive(directory / row["archive"], size, original_pin) as archive:
        original, stand = archive.read(row["member"]), archive.read(stand_name)
    if (len(original) != row["bytes"] or sha256(original).hexdigest() != row["SHA256"]
            or sha256(stand).hexdigest() != stand_pin or row["split"] != split):
        raise ValueError("independent take or stand byte binding changed")
    size, marker_pin, members = COHORTS[subject]
    with read_archive(cohort / f"{subject}Walking_c3d.zip", size, marker_pin) as archive:
        marker = archive.read(members[11])
    if sha256(marker).hexdigest() != witness["C3D_member_SHA256"]:
        raise ValueError("independent physical marker identity differs")
    coords, valid = source_toes(marker, witness["start_offset_C3D_frames"])
    joints, frames = read_source_frames(BytesIO(original), row)
    neutral = neutral_reference(stand, target_joints, geometry[5][:22])
    features, _, geometry_report = reconstruct_features(joints, frames, target_joints, geometry, neutral)
    features = features.astype("f8")
    names, source = source_positions(joints, frames)
    return dict(source_names=names, source=source, source_root=frames[:, :3],
                target_names=[j["name"] for j in target_joints],
                target=features[:, 135:].reshape(WINDOW, 22, 3), target_root=features[:, :3],
                marker_coords=coords, stance=stance_intervals(coords, valid), calibration=witness), geometry_report


def audit(directory):
    cohort = directory / "cohort-calibration-walking-c3d"
    study = verified_json(cohort / "calibration-B10-B11-original-first.json", COHORT_CALIBRATION_SHA)["results"]
    selection = verified_json(directory / "locomotion-selected-85.private.json", SELECTION_SHA)["records"]
    selected = {item["member"]: item for item in selection}
    assets = directory / "wooden-publisher-76957e3"
    geometry = load_geometry(assets)
    joints = inspect_wooden(assets)["target_joints"]
    tested, excluded = {}, {}
    for subject, group in study.items():
        witness = group["pairs"]["11"]
        summary = dict(existing_holdout_RMS_mm=witness["holdout_RMS_mm"],
                       existing_scale_mm_per_BVH_unit=witness["C3D_mm_per_BVH_unit"])
        if not calibration_passes(witness):
            excluded[subject] = summary
            continue
        row = selected[witness["BVH_name"]]
        inputs, geometry_report = input_for_take(directory, cohort, subject, witness, row, geometry, joints)
        tested[subject] = dict(**summary, dataset_split=row["split"], original_BVH_SHA256=row["SHA256"],
                               original_C3D_SHA256=witness["C3D_member_SHA256"],
                               geometry=geometry_report, foot_measurements=compare(inputs))
    return {
        "frozen_calibration_SHA256": COHORT_CALIBRATION_SHA,
        "motion": "B11 135 degree turn",
        "frames_per_processed_take": WINDOW,
        "excluded_by_frozen_geometric_calibration": excluded,
        "tested": tested,
        "accepted_training_rows": 0,
        "training_READY": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    raw = (json.dumps(audit(args.directory), indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with args.output.open("xb") as stream:
        stream.write(raw)
    print(sha256(raw).hexdigest())


if __name__ == "__main__":
    main()
