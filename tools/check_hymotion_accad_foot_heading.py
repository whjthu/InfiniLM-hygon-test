"""Separate ACCAD marker and skeletal motion before and after retargeting."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from check_hymotion_accad_lower_leg_phase import (
    FPS, SOURCE_CHAINS, TARGET_CHAINS, read_paired_takes,
)


def foot_points(positions, names, chains):
    indices = [[names.index(chain[-2]), names.index(chain[-1])] for chain in chains]
    return positions[:, indices, :]


def heading_change(points):
    foot = points[:, :, 1] - points[:, :, 0]
    if np.any(np.linalg.norm(foot, axis=-1) < 1e-8):
        raise ValueError("undefined foot segment")
    planar = np.hypot(foot[..., 0], foot[..., 1])
    valid = planar > 0.2 * np.linalg.norm(foot, axis=-1)
    angle = np.arctan2(foot[..., 1], foot[..., 0])
    step = np.diff(angle, axis=0)
    change = np.degrees(np.arctan2(np.sin(step), np.cos(step)))
    return change, valid[:-1] & valid[1:]


def velocity_stats(positions, stance):
    velocity = np.diff(positions[..., :2], axis=0) * FPS
    speed = np.linalg.norm(velocity, axis=-1)
    both = stance.all(axis=1)
    if not both.any() or not np.isfinite(positions).all():
        raise ValueError("missing bilateral support or invalid coordinates")
    pair_speed = speed[both]
    difference = np.linalg.norm(velocity[both, 0] - velocity[both, 1], axis=1)
    return {
        "stance_speed_median_mps": float(np.median(speed[stance])),
        "bilateral_toe_RMS_median_mps": float(np.median(np.sqrt(np.mean(pair_speed ** 2, axis=1)))),
        "bilateral_intertoe_relative_speed_median_mps": float(np.median(difference)),
        "bilateral_intertoe_relative_speed_p95_mps": float(np.quantile(difference, 0.95)),
    }


def compare(take):
    source = foot_points(take["source"], take["source_names"], SOURCE_CHAINS)
    target = foot_points(take["target"], take["target_names"], TARGET_CHAINS)
    # The original physical calibration binds right-handed C3D x/y to the
    # ground plane. For WoodenMesh use (x, -z, y), also right-handed.
    witness = take["calibration"]
    source = (source + take["source_root"][:, None, None, :])[..., witness["axes"]]
    source *= np.asarray(witness["signs"]) * (witness["C3D_mm_per_BVH_unit"] / 1000)
    target = (target + take["target_root"][:, None, None, :])[..., [0, 2, 1]]
    target *= np.array([1, -1, 1])
    stance = take["stance"]
    original_yaw, source_valid = heading_change(source)
    target_yaw, target_valid = heading_change(target)
    heading = {}
    for i, name in enumerate(("left", "right")):
        included = stance[:, i] & source_valid[:, i] & target_valid[:, i]
        if not included.any():
            raise ValueError("no reliably horizontal foot segments during stance")
        s, t = original_yaw[included, i], target_yaw[included, i]
        error = np.abs((t - s + 180) % 360 - 180)
        heading[name] = {
            "observed_stance_intervals": int(stance[:, i].sum()),
            "reliably_projected_intervals": int(included.sum()),
            "source_yaw_abs_change_median_degrees": float(np.median(np.abs(s))),
            "target_yaw_abs_change_median_degrees": float(np.median(np.abs(t))),
            "yaw_change_error_median_degrees": float(np.median(error)),
            "yaw_change_error_p95_degrees": float(np.quantile(error, 0.95)),
        }
    return {
        "physical_bilateral_intervals": int(stance.all(axis=1).sum()),
        "C3D_markers": velocity_stats(take["marker_coords"], stance),
        "original_BVH_toe_base_joints": velocity_stats(source[:, :, 1], stance),
        "WoodenMesh_foot_joints": velocity_stats(target[:, :, 1], stance),
        "BVH_to_WoodenMesh_ground_plane_foot_yaw": heading,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    geometry_pin, takes = read_paired_takes(args.directory, args.candidate)
    doc = {"target_geometry_SHA256": geometry_pin, "takes": {name: compare(take) for name, take in takes.items()},
           "accepted_training_rows": 0, "training_READY": False}
    raw = (json.dumps(doc, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with args.output.open("xb") as stream:
        stream.write(raw)
    print(sha256(raw).hexdigest())


if __name__ == "__main__":
    main()
