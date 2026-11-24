"""Evaluate independent neutral-capture calibration on three frozen HY-Motion inputs."""

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import numpy as np

from audit_hymotion_accad_bvh import ARCHIVES
from audit_hymotion_accad_cohort_scale import read_archive
from audit_hymotion_accad_skeletons import SELECTION_SHA, inspect_mapping, inspect_wooden, skeleton
from audit_hymotion_accad_source_fk import (
    FRAME_COUNT, POSITION_CHANNELS, ROTATION_CHANNELS, read_source_frames, source_matrices,
)
from build_hymotion_accad_wooden_pilot import (
    COHORT_CALIBRATION_SHA, ground_height, normalize_rows, target_world_positions, world_rotations,
    yaw_to_positive_z,
)
from verify_hymotion_publisher_feature_assets import load_geometry


NEUTRAL = {
    "Female1": ("Female1_A01_Stand.bvh", "9cbbb4749b42922503fb24997d53f60d6c36b62f6c00a12d3c8a1a9cfd5987d4"),
    "Male1": ("Male1_A1_Stand.bvh", "0f820d765d21e9748accbadfdd967e13b0ce71e182d03dc3129fc4d60b9271e2"),
    "Male2": ("Male2_A1_Stand.bvh", "c73f9f931ee72a5241fe8537ba48a6e70a8fa7f7f011431e3012d5247fb1d077"),
}
STANDING_FRAMES = 32


def correspondence(joints, target):
    match = inspect_mapping(joints, target)
    if match["status"] != "NAME_AND_PARENT_GRAPH_MATCH":
        raise ValueError("expected complete source-to-target joint correspondence")
    return match["bvh_joint_indices_in_wooden_order"]


def neutral_reference(raw, target, parents):
    with BytesIO(raw) as stream:
        joints = skeleton(stream)
        count = stream.readline().decode("ascii").split()
        period = stream.readline().decode("ascii").split()
        rows = np.asarray([tuple(map(float, line.split())) for line in stream if line.strip()])
    if (len(count) != 2 or count[0] != "Frames:" or period != ["Frame", "Time:", "0.0333333"]
            or rows.shape != (int(count[1]), 69) or not STANDING_FRAMES <= len(rows) <= 400
            or not np.isfinite(rows).all()):
        raise ValueError("unexpected independent standing capture")
    if (tuple(joints[0]["channels"]) != POSITION_CHANNELS + ROTATION_CHANNELS
            or any(tuple(joint["channels"]) != ROTATION_CHANNELS for joint in joints[1:])):
        raise ValueError("independent standing capture rotation order differs")
    order = correspondence(joints, target)
    translation = rows[:, :3] / 100
    source = source_matrices(rows[:, 3:].reshape(-1, 22, 3)[:, order])
    step_cost = np.sum(np.diff(translation, axis=0) ** 2, axis=1)
    step_cost += np.mean(np.diff(source, axis=0) ** 2, axis=(1, 2, 3))
    start = int(np.argmin(np.convolve(step_cost, np.ones(STANDING_FRAMES - 1), mode="valid")))
    window = slice(start, start + STANDING_FRAMES)
    left, _, right = np.linalg.svd(source[window].mean(axis=0))
    neutral = left @ right
    if np.any(np.linalg.det(neutral) < 1 - 1e-9):
        raise ValueError("the fitted normalized pose is not a proper rotation")
    world = world_rotations(neutral[None], parents)[0]
    offsets = np.asarray([joints[i]["offset"] for i in order])
    outward = world[0] @ (offsets[1] - offsets[2])
    forward = np.cross(outward, [0.0, 1.0, 0.0])
    aligned = yaw_to_positive_z(normalize_rows(forward))
    return neutral, aligned @ world, aligned, {
        "neutral_start_frame": start, "neutral_frames": STANDING_FRAMES, "stand_source_frames": len(rows),
        "window_root_extent_m": np.ptp(translation[window], axis=0).tolist(),
    }


def reconstruct_features(joints, frames, target, geometry, neutral):
    vertices, _, weights, indices, nodes, parents, mean, std = geometry
    reference, world_axes, heading, _ = neutral
    order = correspondence(joints, target)
    source = source_matrices(frames[:, 3:].reshape(-1, 22, 3)[:, order])
    local = world_axes @ (np.swapaxes(reference, -1, -2) @ source) @ np.swapaxes(world_axes, -1, -2)
    facing = yaw_to_positive_z(local[0, 0, :, 2])
    local[:, 0] = facing @ local[:, 0]
    world = np.empty((FRAME_COUNT, len(parents), 3, 3))
    world[:, :22] = world_rotations(local, parents[:22])
    for i, parent in enumerate(parents[22:], 22):
        world[:, i] = world[:, parent]
    joints_3d = target_world_positions(world, nodes, parents)
    translation = (frames[:, :3] - frames[0, :3]) @ (facing @ heading).T / 100
    translation[:, 1] -= ground_height(world, joints_3d, vertices, nodes, indices, weights, translation[:, 1]).min()
    interleaved_cols = local[..., :2].reshape(FRAME_COUNT, 132)
    raw = np.column_stack((translation, interleaved_cols, joints_3d[:, :22].reshape(FRAME_COUNT, 66))).astype("f4")
    normalized = (raw - mean) / np.where(std < 1e-3, 1.0, std)
    encoded = raw[:, 3:135].reshape(FRAME_COUNT, 22, 3, 2)
    col1 = normalize_rows(encoded[..., 0])
    col2 = normalize_rows(encoded[..., 1] - (encoded[..., 1] * col1).sum(axis=-1, keepdims=True) * col1)
    decoded = np.stack((col1, col2, np.cross(col1, col2)), axis=-1)
    error = float(np.max(np.abs(decoded - local)))
    minimum = float(ground_height(world, joints_3d, vertices, nodes, indices, weights, raw[:, 1]).min())
    if raw.shape != (128, 201) or not np.isfinite(normalized).all() or abs(minimum) > 1e-6 or error > 1e-6:
        raise ValueError("pilot ground, rotation format or finite values failed")
    partition = {}
    for name, start, end in (("root", 0, 3), ("rotation", 3, 135), ("joint", 135, 201)):
        absolute = np.abs(normalized[:, start:end])
        partition[name] = {"maximum_abs": float(absolute.max()), "over_8": int(np.count_nonzero(absolute > 8))}
    return raw, normalized, {
        "ground_y_min_m": minimum, "rotation6d_max_roundtrip_error": error, "normalization_by_feature_group": partition,
    }


def audit(directory, asset_directory, output):
    selection = (directory / "locomotion-selected-85.private.json").read_bytes()
    calibration = (directory / "cohort-calibration-walking-c3d/calibration-B10-B11-original-first.json").read_bytes()
    if sha256(selection).hexdigest() != SELECTION_SHA or sha256(calibration).hexdigest() != COHORT_CALIBRATION_SHA:
        raise ValueError("source or calibration selection differs")
    pins = {item["member"]: item for item in json.loads(selection)["records"]}
    pairs = json.loads(calibration)["results"]
    geometry = load_geometry(asset_directory)
    parents = geometry[5]
    target = inspect_wooden(asset_directory)["target_joints"]
    tensor, reports = {}, {}
    for subject, (stand, stand_pin) in NEUTRAL.items():
        name = f"{subject}_B10_WalkTurnLeft45.bvh"
        archives = pins[name]["archive"]
        _, size, pin, split = ARCHIVES[archives]
        with read_archive(directory / archives, size, pin) as container:
            original_stand, motion = container.read(stand), container.read(name)
        calibration_record = pairs[subject]["pairs"]["10"]
        if (sha256(original_stand).hexdigest() != stand_pin or sha256(motion).hexdigest() != pins[name]["SHA256"]
                or pins[name]["split"] != split or calibration_record["BVH_name"] != name
                or calibration_record["axes"] != [0, 2, 1] or calibration_record["signs"] != [-1, 1, 1]
                or abs(calibration_record["C3D_mm_per_BVH_unit"] - 10) >= 0.05
                or calibration_record["holdout_RMS_mm"] >= 40):
            raise ValueError("physical calibration or original source scope differs")
        neutral = neutral_reference(original_stand, target, parents[:22])
        source_joints, frames = read_source_frames(BytesIO(motion), pins[name])
        raw, normalized, geometry_report = reconstruct_features(source_joints, frames, target, geometry, neutral)
        reports[subject] = {**geometry_report, **neutral[-1], "original_BVH_sha256": pins[name]["SHA256"],
                            "neutral_BVH_sha256": stand_pin, "split": split}
        tensor[f"{subject}_raw"], tensor[f"{subject}_normalized"] = raw, normalized
    buffer = BytesIO()
    np.savez_compressed(buffer, **tensor)
    raw = buffer.getvalue()
    with (output / "geometry-candidate.private.npz").open("xb") as stream:
        stream.write(raw)
    return {
        "reports": reports, "candidate_tensor_SHA256": sha256(raw).hexdigest(), "candidate_tensor_bytes": len(raw),
        "CPU_candidate_frames": FRAME_COUNT * len(reports), "feature_dimensions": 201,
        "motion_quality_review_passed": False, "native_target_match_verified": False,
        "training_READY": False, "GPU_steps": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("assets", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir()
    result = audit(args.directory, args.assets, args.output)
    content = (json.dumps(result, sort_keys=True, indent=2) + "\n").encode()
    with (args.output / "geometry-candidate-summary.json").open("xb") as stream:
        stream.write(content)
    print(sha256(content).hexdigest())


if __name__ == "__main__":
    main()
