"""Build a three-clip CPU geometry witness, before native runtime and text checks."""

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import numpy as np

from audit_hymotion_accad_bvh import ARCHIVES
from audit_hymotion_accad_cohort_scale import read_archive
from audit_hymotion_accad_skeletons import SELECTION_SHA, inspect_mapping, inspect_wooden
from audit_hymotion_accad_source_fk import FRAME_COUNT, read_source_frames, source_matrices
from verify_hymotion_publisher_feature_assets import load_geometry


COHORT_CALIBRATION_SHA = "6491c5dc9fff14f6511becbc7f6879d1ebbd11de45a919f2fa56431e02cc0556"


def normalize_rows(values):
    values = np.asarray(values, dtype=np.float64)
    length = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(length < 1e-8):
        raise ValueError("degenerate rest direction")
    return values / length


def rotation_fit(target, source):
    """Fit the rest-pose rotation with a shortest-arc rule for a single direction."""
    a, b = normalize_rows(target), normalize_rows(source)
    if len(a) == 1:
        cosine = a[0] @ b[0]
        if cosine < -1 + 1e-8:
            raise ValueError("180-degree rest-bone twist is ambiguous")
        v = np.cross(a[0], b[0])
        skew = np.array(((0, -v[2], v[1]), (v[2], 0, -v[0]), (-v[1], v[0], 0)))
        return np.eye(3) + skew + (skew @ skew) / (1 + cosine)
    left, values, right = np.linalg.svd(b.T @ a)
    if values[1] < 1e-8:
        raise ValueError("branch rest directions are collinear")
    proper = np.diag((1, 1, 1 if np.linalg.det(left @ right) > 0 else -1))
    return left @ proper @ right


def rest_basis(source_offsets, target_offsets, parents):
    corrections = np.empty((len(parents), 3, 3))
    angles = []
    for joint, parent in enumerate(parents):
        children = np.flatnonzero(parents == joint)
        if len(children) == 0:
            corrections[joint] = corrections[parent]
            continue
        a, b = target_offsets[children], source_offsets[children]
        corrections[joint] = rotation_fit(a, b)
        dots = np.sum(normalize_rows(a @ corrections[joint].T) * normalize_rows(b), axis=1)
        angles.extend(np.rad2deg(np.arccos(np.clip(dots, -1, 1))))
    return corrections, angles


def world_rotations(local, parents):
    result = np.empty_like(local)
    result[:, 0] = local[:, 0]
    for joint, parent in enumerate(parents[1:], 1):
        result[:, joint] = result[:, parent] @ local[:, joint]
    return result


def yaw_to_positive_z(facing):
    x, _, z = facing
    length = np.hypot(x, z)
    if length < 0.25:
        raise ValueError("initial facing is not reliably horizontal")
    c, s = z / length, -x / length
    return np.array(((c, 0, s), (0, 1, 0), (-s, 0, c)))


def target_world_positions(rotation, joints, parents):
    posed = np.empty((*rotation.shape[:2], 3))
    posed[:, 0] = joints[0]
    for child, parent in enumerate(parents[1:], 1):
        posed[:, child] = posed[:, parent] + rotation[:, parent] @ (joints[child] - joints[parent])
    return posed


def ground_height(rotation, positions, vertices, joint_template, indices, weights, root_y):
    """Evaluate native LBS vertex heights in chunks, without keeping a posed mesh."""
    row_y = rotation[..., 1, :]
    offset_y = positions[..., 1] - np.einsum("tji,ji->tj", row_y, joint_template)
    lowest = np.full(len(rotation), np.inf)
    for subset in np.array_split(np.arange(len(vertices)), 16):
        vertex, skin, influence = vertices[subset], indices[subset], weights[subset]
        y = np.einsum("tvki,vi->tvk", row_y[:, skin], vertex) + offset_y[:, skin]
        lowest = np.minimum(lowest, (np.sum(y * influence, axis=2) + root_y[:, None]).min(axis=1))
    return lowest


def build_frames(source_joints, frames, geometry, source_indices):
    vertices, _, weights, skin_indices, target_joints, parents, mean, std = geometry
    if len(frames) != FRAME_COUNT or len(source_indices) != 22:
        raise ValueError("unexpected native frame or joint denominator")
    source_offsets = np.asarray([source_joints[i]["offset"] for i in source_indices])
    target_offsets = target_joints[:22] - target_joints[parents[:22]]
    corrections, errors = rest_basis(source_offsets, target_offsets, parents[:22])
    angles = frames[:, 3:].reshape((FRAME_COUNT, 22, 3))[:, source_indices]
    source_rotation = world_rotations(source_matrices(angles), parents[:22])
    world22 = source_rotation @ corrections
    align = yaw_to_positive_z(world22[0, 0, :, 2])
    world22 = align @ world22
    local = np.empty_like(world22)
    local[:, 0] = world22[:, 0]
    local[:, 1:] = np.swapaxes(world22[:, parents[1:22]], -1, -2) @ world22[:, 1:]
    rotation = np.empty((FRAME_COUNT, len(parents), 3, 3), dtype=np.float64)
    rotation[:, :22] = world22
    for i, parent in enumerate(parents[22:], 22):
        rotation[:, i] = rotation[:, parent]
    relative = (frames[:, :3] - frames[0, :3]) @ align.T / 100
    positioned = target_world_positions(rotation, target_joints, parents)
    floor = ground_height(rotation, positioned, vertices, target_joints, skin_indices, weights, relative[:, 1])
    relative[:, 1] -= floor.min()
    if not np.allclose(np.swapaxes(local, -1, -2) @ local, np.eye(3), atol=1e-9, rtol=0):
        raise ValueError("failed geometric invariants")
    rotations6d = local[..., :2].reshape(FRAME_COUNT, 132)
    raw = np.column_stack((relative, rotations6d, positioned[:, :22].reshape(FRAME_COUNT, 66))).astype("f4")
    encoded = raw[:, 3:135].reshape(FRAME_COUNT, 22, 3, 2).astype("f8")
    col1 = normalize_rows(encoded[..., 0])
    col2 = normalize_rows(encoded[..., 1] - np.sum(encoded[..., 1] * col1, axis=-1, keepdims=True) * col1)
    reconstructed = np.stack((col1, col2, np.cross(col1, col2)), axis=-1)
    roundtrip_error = float(np.max(np.abs(reconstructed - local)))
    final_floor = float(ground_height(rotation, positioned, vertices, target_joints, skin_indices, weights,
                                      raw[:, 1]).min())
    normalized = (raw - mean) / np.where(std < 1e-3, 1.0, std)
    if (raw.shape != (128, 201) or not np.isfinite(normalized).all()
            or roundtrip_error > 1e-6 or abs(final_floor) > 1e-6):
        raise ValueError("unexpected numerical result")
    report = {
        "ground_Y_min_m": final_floor,
        "rotation6d_roundtrip_max_error": roundtrip_error,
        "normalized_abs_max": float(np.max(np.abs(normalized))),
        "normalized_abs_over_8": int(np.count_nonzero(np.abs(normalized) > 8)),
        "normalized_values": normalized.size,
        "max_rest_branch_direction_residual_deg": float(max(errors)),
        "target_joint0_kept_as_native_pose_root": positioned[0, 0].tolist(),
        "first_root_forward_horizontal_XZ": world22[0, 0, [0, 2], 2].tolist(),
        "root_horizontal_extent_m": np.ptp(relative[:, [0, 2]], axis=0).tolist(),
    }
    return raw, normalized, report


def audit(directory, assets, output):
    cal_bytes = (directory / "cohort-calibration-walking-c3d/calibration-B10-B11-original-first.json").read_bytes()
    selection = (directory / "locomotion-selected-85.private.json").read_bytes()
    if sha256(cal_bytes).hexdigest() != COHORT_CALIBRATION_SHA or sha256(selection).hexdigest() != SELECTION_SHA:
        raise ValueError("frozen calibration or source selection differs")
    calibration = json.loads(cal_bytes)["results"]
    sources = {item["member"]: item for item in json.loads(selection)["records"]}
    geometry = load_geometry(assets)
    target = inspect_wooden(assets)["target_joints"]
    tensors, reports = {}, {}
    for subject in ("Female1", "Male1", "Male2"):
        filename = f"{subject}_B10_WalkTurnLeft45.bvh"
        row = sources[filename]
        witness = calibration[subject]["pairs"]["10"]
        if (witness["BVH_name"] != filename or witness["axes"] != [0, 2, 1] or witness["signs"] != [-1, 1, 1]
                or abs(witness["C3D_mm_per_BVH_unit"] - 10) >= 0.05 or witness["holdout_RMS_mm"] >= 40):
            raise ValueError("clip lacks the admitted physical calibration")
        _, size, pin, split = ARCHIVES[row["archive"]]
        with read_archive(directory / row["archive"], size, pin) as archive:
            raw_bvh = archive.read(filename)
        if sha256(raw_bvh).hexdigest() != row["SHA256"] or row["split"] != split:
            raise ValueError("original B10 source member or split differs")
        joints, frames = read_source_frames(BytesIO(raw_bvh), row)
        mapping = inspect_mapping(joints, target)
        raw, normalized, summary = build_frames(joints, frames, geometry, mapping["bvh_joint_indices_in_wooden_order"])
        tensors[f"{subject}_raw"], tensors[f"{subject}_normalized"] = raw, normalized
        reports[subject] = {**summary, "split": split, "original_BVH_sha256": row["SHA256"]}
    buffer = BytesIO()
    np.savez_compressed(buffer, **tensors)
    data = buffer.getvalue()
    with (output / "geometry-candidate.private.npz").open("xb") as stream:
        stream.write(data)
    return {
        "reports": reports,
        "candidate_tensor_SHA256": sha256(data).hexdigest(), "candidate_tensor_bytes": len(data),
        "geometry_candidate_frames": FRAME_COUNT * len(reports), "feature_dimensions": 201,
        "native_target_match_verified": False, "human_motion_or_text_qualification": False,
        "training_READY": False, "GPU_steps": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("publisher_assets", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir()
    doc = audit(args.directory, args.publisher_assets, args.output)
    content = (json.dumps(doc, sort_keys=True, indent=2) + "\n").encode()
    with (args.output / "geometry-candidate-summary.json").open("xb") as stream:
        stream.write(content)
    print(sha256(content).hexdigest())


if __name__ == "__main__":
    main()
