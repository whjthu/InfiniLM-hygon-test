"""Compare source and WoodenMesh lower-limb changes on original contact intervals."""

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import numpy as np

from audit_hymotion_accad_bvh import ARCHIVES
from audit_hymotion_accad_skeletons import SELECTION_SHA, inspect_mapping, inspect_wooden
from audit_hymotion_accad_source_fk import read_source_frames, source_matrices
from build_hymotion_accad_wooden_pilot import target_world_positions, world_rotations
from check_hymotion_accad_physical_foot_stance import (
    COHORT_CALIBRATION_SHA, COHORTS, FPS, NEUTRAL_SUMMARY_SHA, WINDOW, read_archive, source_toes,
)


SOURCE_CHAINS = (("LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase"),
                 ("RightUpLeg", "RightLeg", "RightFoot", "RightToeBase"))
TARGET_CHAINS = (("L_Hip", "L_Knee", "L_Ankle", "L_Foot"),
                 ("R_Hip", "R_Knee", "R_Ankle", "R_Foot"))
MAX_LAG = 3


def verified_json(path, pin):
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != pin:
        raise ValueError(f"frozen input changed: {path.name}")
    return json.loads(raw)


def source_positions(joints, frames):
    names = [j["name"] for j in joints]
    parents = np.asarray([-1] + [names.index(j["parent"]) for j in joints[1:]])
    if any(parent >= i for i, parent in enumerate(parents[1:], 1)):
        raise ValueError("source joint order is not topological")
    offsets = np.asarray([j["offset"] for j in joints])
    rest = offsets.copy()
    for child, parent in enumerate(parents[1:], 1):
        rest[child] += rest[parent]
    matrices = source_matrices(frames[:, 3:].reshape(WINDOW, len(joints), 3))
    return names, target_world_positions(world_rotations(matrices, parents), rest, parents)


def leg_angles(points, chains, names):
    named = [[names.index(name) for name in chain] for chain in chains]
    hip, knee, ankle, toe = np.moveaxis(points[:, named, :], 2, 0)
    foot, upper, lower = toe - ankle, hip - knee, ankle - knee
    lengths = [np.linalg.norm(segment, axis=-1) for segment in (foot, upper, lower)]
    if not np.isfinite(points).all() or min(float(segment.min()) for segment in lengths) <= 1e-7:
        raise ValueError("nonfinite or degenerate leg geometry")
    elevation = np.degrees(np.arctan2(foot[..., 1], np.hypot(foot[..., 0], foot[..., 2])))
    cosine = np.sum(upper * lower, axis=-1) / (lengths[1] * lengths[2])
    flexion = 180 - np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    return {"ankle_to_toe_elevation": elevation, "knee_flexion": flexion}


def stance_intervals(coords, valid):
    planar_speed = np.linalg.norm(np.diff(coords[:, :, :2], axis=0), axis=-1) * FPS
    ground = np.nanpercentile(np.where(valid, coords[:, :, 2], np.nan), 5, axis=0)
    return valid[:-1] & valid[1:] & (planar_speed < 0.18) & (
        np.maximum(coords[:-1, :, 2], coords[1:, :, 2]) < ground + 0.03
    )


def phase_comparison(source, target, stance):
    source_delta, target_delta = np.diff(source), np.diff(target)
    if source_delta.shape != target_delta.shape or source_delta.shape != stance.shape or not stance.any():
        raise ValueError("angle series and physical interval mask differ")
    zero = target_delta[stance] - source_delta[stance]
    usable = np.flatnonzero(stance)
    usable = usable[(usable >= MAX_LAG) & (usable < len(stance) - MAX_LAG)]
    reference = source_delta[usable]
    correlations = {}
    if len(usable) >= 8 and np.std(reference) >= 0.1:
        for lag in range(-MAX_LAG, MAX_LAG + 1):
            values = target_delta[usable + lag]
            if np.std(values) >= 0.1:
                correlations[lag] = float(np.corrcoef(reference, values)[0, 1])
    best = max(correlations, key=lambda lag: (correlations[lag], -abs(lag), -lag), default=None)
    return {
        "same_time_intervals": int(stance.sum()),
        "common_lag_intervals": len(usable),
        "median_source_abs_change_degrees": float(np.median(np.abs(source_delta[stance]))),
        "median_target_abs_change_degrees": float(np.median(np.abs(target_delta[stance]))),
        "median_abs_same_time_change_error_degrees": float(np.median(np.abs(zero))),
        "p95_abs_same_time_change_error_degrees": float(np.quantile(np.abs(zero), 0.95)),
        "same_time_Pearson": correlations.get(0),
        "best_target_offset_frames_within_three": best,
        "best_offset_Pearson": correlations.get(best),
    }


def audit(directory, candidate):
    cohort = directory / "cohort-calibration-walking-c3d"
    calibration = verified_json(cohort / "calibration-B10-B11-original-first.json", COHORT_CALIBRATION_SHA)["results"]
    selection = verified_json(directory / "locomotion-selected-85.private.json", SELECTION_SHA)["records"]
    selected = {item["member"]: item for item in selection}
    metadata = verified_json(candidate / "geometry-candidate-summary.json", NEUTRAL_SUMMARY_SHA)
    geometry = (candidate / "geometry-candidate.private.npz").read_bytes()
    pin = sha256(geometry).hexdigest()
    if pin != metadata["candidate_tensor_SHA256"]:
        raise ValueError("frozen target geometry differs")
    target_joints = inspect_wooden(directory / "wooden-publisher-76957e3")["target_joints"]
    names = [joint["name"] for joint in target_joints]
    results = {}
    with np.load(BytesIO(geometry), allow_pickle=False) as slots:
        for group, (c3d_size, c3d_pin, markers) in COHORTS.items():
            witness = calibration[group]["pairs"]["10"]
            row = selected[witness["BVH_name"]]
            _, size, source_pin, split = ARCHIVES[row["archive"]]
            with read_archive(directory / row["archive"], size, source_pin) as archive:
                raw = archive.read(row["member"])
            if len(raw) != row["bytes"] or sha256(raw).hexdigest() != row["SHA256"] or row["split"] != split:
                raise ValueError("frozen source clip differs")
            with read_archive(cohort / f"{group}Walking_c3d.zip", c3d_size, c3d_pin) as archive:
                marker = archive.read(markers[10])
            if (sha256(marker).hexdigest() != witness["C3D_member_SHA256"]
                    or witness["axes"] != [0, 2, 1] or witness["signs"] != [-1, 1, 1]):
                raise ValueError("physical capture or vertical-axis binding differs")
            coords, valid = source_toes(marker, witness["start_offset_C3D_frames"])
            joints, frames = read_source_frames(BytesIO(raw), row)
            if inspect_mapping(joints, target_joints)["status"] != "NAME_AND_PARENT_GRAPH_MATCH":
                raise ValueError("anatomical correspondence differs")
            source_names, source = source_positions(joints, frames)
            target = slots[f"{group}_raw"][:, 135:].reshape(WINDOW, 22, 3).astype("f8")
            source_angle = leg_angles(source, SOURCE_CHAINS, source_names)
            target_angle = leg_angles(target, TARGET_CHAINS, names)
            stance = stance_intervals(coords, valid)
            results[group] = {label: {side: phase_comparison(values[:, i], target_angle[label][:, i], stance[:, i])
                                      for i, side in enumerate(("left", "right"))}
                              for label, values in source_angle.items()}
    return {"frame_count_per_take": WINDOW, "source_marker_calibration_SHA256": COHORT_CALIBRATION_SHA,
            "target_geometry_SHA256": pin, "take_results": results, "training_READY": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    raw = (json.dumps(audit(args.directory, args.candidate), indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with args.output.open("xb") as stream:
        stream.write(raw)
    print(sha256(raw).hexdigest())


if __name__ == "__main__":
    main()
