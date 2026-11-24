"""Check native ACCAD BVH poses before applying a target retargeting transform."""

import argparse
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from audit_hymotion_accad_bvh import ARCHIVES
from audit_hymotion_accad_skeletons import SELECTION_SHA, inspect_mapping, inspect_wooden, skeleton


FRAME_COUNT = 128
ROTATION_CHANNELS = ("Zrotation", "Xrotation", "Yrotation")
POSITION_CHANNELS = ("Xposition", "Yposition", "Zposition")


def source_matrices(zxy):
    z, x, y = np.moveaxis(np.deg2rad(zxy), -1, 0)
    cz, cx, cy = (np.cos(angle) for angle in (z, x, y))
    sz, sx, sy = (np.sin(angle) for angle in (z, x, y))
    return np.stack((
        cz * cy - sz * sx * sy, -sz * cx, cz * sy + sz * sx * cy,
        sz * cy + cz * sx * sy, cz * cx, sz * sy - cz * sx * cy,
        -cx * sy, sx, cx * cy,
    ), axis=-1).reshape((*zxy.shape[:-1], 3, 3))


def reference_matrices(zxy):
    angles = np.deg2rad(zxy) / 2
    shape = (*angles.shape[:-1], 3)
    scalar = np.ones(angles.shape[:-1])
    vector = np.zeros(shape)
    for column, axis in enumerate((2, 0, 1)):
        half_angle = angles[..., column]
        axis_scalar = np.cos(half_angle)
        axis_vector = np.zeros(shape)
        axis_vector[..., axis] = np.sin(half_angle)
        product = scalar * axis_scalar - np.sum(vector * axis_vector, axis=-1)
        vector = (scalar[..., None] * axis_vector + axis_scalar[..., None] * vector
                  + np.cross(vector, axis_vector))
        scalar = product
    v = vector[..., :, None]
    twice_cross = 2 * np.cross(v, np.eye(3), axisa=-2, axisb=-2, axisc=-2)
    return np.eye(3) + scalar[..., None, None] * twice_cross + np.cross(v, twice_cross, axis=-2)


def read_source_frames(stream, row):
    root = skeleton(stream)
    count = stream.readline(100).decode("ascii").strip()
    period = stream.readline(100).decode("ascii").strip()
    if (count.split() != ["Frames:", str(row["frames"])]
            or period.split() != ["Frame", "Time:", row["frame_period_seconds"]]
            or row["frames"] < FRAME_COUNT or not row["nominal_30fps"]):
        raise ValueError("original source timing differs")
    root_channels = tuple(root[0]["channels"])
    if root_channels != POSITION_CHANNELS + ROTATION_CHANNELS or any(
            tuple(joint["channels"]) != ROTATION_CHANNELS for joint in root[1:]):
        raise ValueError("source rotation order differs")
    expected = 3 + 3 * len(root)
    frames = np.empty((FRAME_COUNT, expected), dtype=np.float64)
    for i in range(FRAME_COUNT):
        values = stream.readline(4097).split()
        if len(values) != expected:
            raise ValueError("source motion row width differs")
        frames[i] = tuple(float(value) for value in values)
    if not np.isfinite(frames).all():
        raise ValueError("nonfinite source pose")
    return root, frames


def pose_summary(joints, frames):
    relative_translation = frames[:, :3] - frames[0, :3]
    angles = frames[:, 3:].reshape((FRAME_COUNT, len(joints), 3))
    matrices = source_matrices(angles)
    matrix_error = float(np.max(np.abs(matrices - reference_matrices(angles))))
    if matrix_error > 2e-12:
        raise ValueError("Euler matrix and quaternion composition differ")
    indices = {joint["name"]: index for index, joint in enumerate(joints)}
    worlds = np.empty_like(matrices)
    positions = np.empty((FRAME_COUNT, len(joints), 3), dtype=np.float64)
    worlds[:, 0], positions[:, 0] = matrices[:, 0], relative_translation
    bone_error = 0.0
    offsets = np.asarray([joint["offset"] for joint in joints], dtype=np.float64)
    for i, joint in enumerate(joints[1:], 1):
        parent = indices[joint["parent"]]
        worlds[:, i] = worlds[:, parent] @ matrices[:, i]
        positions[:, i] = positions[:, parent] + worlds[:, parent] @ offsets[i]
        lengths = np.linalg.norm(positions[:, i] - positions[:, parent], axis=-1)
        bone_error = max(bone_error, float(np.max(np.abs(lengths - np.linalg.norm(offsets[i])))))
    determinant_error = float(np.max(np.abs(np.linalg.det(worlds) - 1)))
    if bone_error > 1e-8 or determinant_error > 2e-12 or not np.isfinite(positions).all():
        raise ValueError("forward kinematics did not preserve source bones and rotations")
    return {
        "frames": FRAME_COUNT,
        "joint_rotations": FRAME_COUNT * len(joints),
        "max_matrix_quaternion_difference": matrix_error,
        "max_world_rotation_determinant_error": determinant_error,
        "max_bone_length_error_source_units": bone_error,
        "raw_root_motion_ranges_XYZ": np.ptp(relative_translation, axis=0).tolist(),
        "raw_source_left_hip_to_ankle_chain_length": float(sum(
            np.linalg.norm(offsets[indices[name]]) for name in ("LeftLeg", "LeftFoot")
        )),
    }


def audit(directory, wooden_directory):
    raw = (directory / "locomotion-selected-85.private.json").read_bytes()
    if sha256(raw).hexdigest() != SELECTION_SHA:
        raise ValueError("frozen candidate selection differs")
    selection = json.loads(raw)["records"]
    wooden = inspect_wooden(wooden_directory)
    results = []
    for archive, (_, size, pin, split) in ARCHIVES.items():
        with (directory / archive).open("rb") as stream:
            data = stream.read(size + 1)
        if len(data) != size or sha256(data).hexdigest() != pin:
            raise ValueError(f"frozen archive differs: {archive}")
        with ZipFile(BytesIO(data)) as source:
            for row in (r for r in selection if r["archive"] == archive):
                clip = source.read(row["member"])
                if row["split"] != split or len(clip) != row["bytes"] or sha256(clip).hexdigest() != row["SHA256"]:
                    raise ValueError(f"frozen clip differs: {archive}")
                joints, frames = read_source_frames(BytesIO(clip), row)
                mapping = inspect_mapping(joints, wooden["target_joints"])
                summary = {"admitted_to_source_fk": False}
                if mapping["status"] == "NAME_AND_PARENT_GRAPH_MATCH":
                    summary = {"admitted_to_source_fk": True, **pose_summary(joints, frames)}
                results.append({"member": row["member"], "split": split, "source_BVH_SHA256": row["SHA256"],
                                "mapping_status": mapping["status"], **summary})
    if len(results) != len(selection) or Counter((r["split"], r["admitted_to_source_fk"]) for r in results) != {
            ("training", True): 67, ("development", True): 17, ("development", False): 1}:
        raise ValueError("frozen cohort accounting differs")
    return {"clips": results, "selection_SHA256": SELECTION_SHA, "numpy_version": np.__version__,
            "frames_per_clip": FRAME_COUNT, "reference_origin": "same-clip first position channel sample",
            "target_retargeted_rows": 0, "target_runtime_loads": 0, "GPU_steps": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_directory", type=Path)
    parser.add_argument("publisher_skeleton", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    doc = audit(args.candidate_directory, args.publisher_skeleton)
    content = (json.dumps(doc, sort_keys=True, indent=2) + "\n").encode()
    with args.output.open("xb") as stream:
        stream.write(content)
    print(json.dumps({"clips": len(doc["clips"]), "result_SHA256": sha256(content).hexdigest()}))


if __name__ == "__main__":
    main()
