"""Attribute neutral-pilot foot drift using simultaneous physical stance."""

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import numpy as np

from check_hymotion_accad_physical_foot_stance import (
    COHORT_CALIBRATION_SHA, COHORTS, FPS, NEUTRAL_SUMMARY_SHA, TARGET_TOES,
    WINDOW, read_archive, source_toes, stance_comparison,
)


def bilateral_stance(coords, valid, features):
    baseline = stance_comparison(coords, valid, features)
    source_velocity = np.diff(coords[:, :, :2], axis=0) * FPS
    source_speed = np.linalg.norm(source_velocity, axis=-1)
    ground = np.nanpercentile(np.where(valid, coords[:, :, 2], np.nan), 5, axis=0)
    stance = valid[:-1] & valid[1:] & (source_speed < 0.18)
    stance &= np.maximum(coords[:-1, :, 2], coords[1:, :, 2]) < ground + 0.03
    if stance.sum(axis=0).tolist() != baseline["source_stance_intervals_by_toe"]:
        raise ValueError("physical stance mask differs from the frozen diagnostic")
    bilateral = stance.all(axis=1)
    if not bilateral.any():
        raise ValueError("no simultaneous physical foot contact")

    relative_feet = features[:, 135:].reshape(WINDOW, 22, 3)[:, TARGET_TOES, :].astype("f8")
    relative_velocity = np.diff(relative_feet[:, :, [0, 2]], axis=0) * FPS
    root_velocity = np.diff(features[:, [0, 2]].astype("f8"), axis=0) * FPS
    actual_velocity = relative_velocity + root_velocity[:, None, :]

    # A common translation cancels the mean toe velocity. The remaining
    # velocity, half the difference, is the least achievable bilateral RMS.
    bilateral_relative = relative_velocity[bilateral]
    least_rms = np.linalg.norm(bilateral_relative[:, 0] - bilateral_relative[:, 1], axis=1) * 0.5
    measured_rms = np.sqrt(np.mean(np.sum(actual_velocity[bilateral] ** 2, axis=-1), axis=1))
    measured_source_rms = np.sqrt(np.mean(source_speed[bilateral] ** 2, axis=1))
    if np.any(least_rms > measured_rms + 1e-10):
        raise ValueError("translation-only bound exceeds measured motion")

    # Magnitudes compare without a new heading fit across coordinate systems.
    source_relative = source_velocity[bilateral, 0] - source_velocity[bilateral, 1]
    return {
        "valid_physical_bilateral_intervals": int(bilateral.sum()),
        "left_only_physical_intervals": int((stance[:, 0] & ~stance[:, 1]).sum()),
        "right_only_physical_intervals": int((stance[:, 1] & ~stance[:, 0]).sum()),
        "median_measured_source_toe_RMS_mps": float(np.median(measured_source_rms)),
        "median_unmodified_target_toe_RMS_mps": float(np.median(measured_rms)),
        "median_target_translation_only_lower_bound_toe_RMS_mps": float(np.median(least_rms)),
        "p95_target_translation_only_lower_bound_toe_RMS_mps": float(np.quantile(least_rms, 0.95)),
        "median_target_root_speed_mps": float(np.median(np.linalg.norm(root_velocity[bilateral], axis=1))),
        "median_source_intertoe_relative_velocity_magnitude_mps": float(
            np.median(np.linalg.norm(source_relative, axis=1))
        ),
        "median_target_intertoe_relative_velocity_magnitude_mps": float(2 * np.median(least_rms)),
    }


def audit(directory, candidate):
    cohort = directory / "cohort-calibration-walking-c3d"
    calibration = (cohort / "calibration-B10-B11-original-first.json").read_bytes()
    metadata = (candidate / "geometry-candidate-summary.json").read_bytes()
    if sha256(calibration).hexdigest() != COHORT_CALIBRATION_SHA or sha256(metadata).hexdigest() != NEUTRAL_SUMMARY_SHA:
        raise ValueError("frozen marker calibration or neutral geometry metadata differs")
    geometry = (candidate / "geometry-candidate.private.npz").read_bytes()
    geometry_sha = sha256(geometry).hexdigest()
    if geometry_sha != json.loads(metadata)["candidate_tensor_SHA256"]:
        raise ValueError("frozen geometry bytes differ")
    witnesses = json.loads(calibration)["results"]
    results = {}
    with np.load(BytesIO(geometry), allow_pickle=False) as slots:
        for group, (size, pin, members) in COHORTS.items():
            witness = witnesses[group]["pairs"]["10"]
            with read_archive(cohort / f"{group}Walking_c3d.zip", size, pin) as archive:
                marker = archive.read(members[10])
            if sha256(marker).hexdigest() != witness["C3D_member_SHA256"]:
                raise ValueError("frozen B10 marker bytes differ")
            coords, valid = source_toes(marker, witness["start_offset_C3D_frames"])
            results[group] = bilateral_stance(coords, valid, slots[f"{group}_raw"])
    return {
        "source_marker_provenance_SHA256": COHORT_CALIBRATION_SHA,
        "matched_target_geometry_SHA256": geometry_sha,
        "window_frames_per_take": WINDOW,
        "measurements_by_take": results,
        "target_geometry_modified": False,
        "training_READY": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = (json.dumps(audit(args.directory, args.candidate), sort_keys=True, indent=2) + "\n").encode()
    with args.output.open("xb") as stream:
        stream.write(data)
    print(sha256(data).hexdigest())


if __name__ == "__main__":
    main()
