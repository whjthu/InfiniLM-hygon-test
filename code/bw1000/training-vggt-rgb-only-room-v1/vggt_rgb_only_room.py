"""Read the frozen Freiburg room RGB-only camera candidate set."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np


SEQUENCE = "freiburg1_room"
WINDOWS = tuple(range(2, 9))
MODE = "tum_room_rgb_only"
PUBLIC_ROOT = Path("results/bw1000/2026-11-25")
_STEM = "specialized48-tum-fr1-room-bag-"
_GROUPS = (
    ("window02", (2,), "-rgb224-local-camera-convention.json",
     "474a68ad0dd36136d8d4e2a3254619165503d02d637ba4582401bb5ca4ebe505",
     "window02-private.json", "fffca1ab8841f137d90bd0ed56bba7be28927a7b1948389d24c9187ff2749c53",
     "frames224-rgb8-private.npy", "poses-first-camera-private.json",
     "rectified_RGB8", "original_RGB8", "absolute_time_difference_ns", "frame_spacing_ns",
     "extrinsics", "intrinsics_per_frame"),
    ("windows03to05", (3, 4, 5), "-rgb224-replay.json",
     "98c9abaae30e27ea0723462f9f49de189d180f25521bef258f75dc0d0703d7da",
     "windows03to05-private-bom.json", "e0fc602003fe54de4cf90b9abc521120b905f97bee061563479fef06cc22bcfd",
     "frames224-private.npy", "camera-targets-private.json",
     "crop_RGB8", "original_RGB8", "absolute_pose_time_error_ns", "interval_ns",
     "camera_from_first", "output_K_per_frame"),
    ("windows06to08", (6, 7, 8), "-rgb224-replay.json",
     "6b9d572ee282c3fd632985fcce406ffa5cc8ef67017ef34515a7b389ff131f62",
     "windows06to08-private-bom.json", "a2975aab55aa433e9754f4d8ad698c113fb7d8267b5b00b671e883a1dbd1398a",
     "frames224-private.npy", "poses-private.json",
     "crop", "rgb8", "time_error_ns", "interval_ns", "camera_from_first", "output_K_per_frame"),
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _pin(payload):
    return {"bytes": len(payload), "SHA256": hashlib.sha256(payload).hexdigest()}


def _verified(path, expected, max_bytes):
    with path.open("rb") as stream:
        payload = stream.read(max_bytes + 1)
    actual = _pin(payload)
    _require(len(payload) <= max_bytes and actual["SHA256"] == expected["SHA256"]
             and actual["bytes"] == expected.get("bytes", actual["bytes"]),
             "RGB-only input pin differs: " + path.name)
    return payload


def _camera_to_world(values):
    pose = np.asarray(values, dtype=np.float64)
    _require(pose.shape == (7,) and np.isfinite(pose).all(), "invalid authoritative pose")
    norm = np.linalg.norm(pose[3:])
    _require(abs(norm - 1) <= 0.002, "invalid authoritative pose quaternion")
    x, y, z, w = pose[3:] / norm
    camera = np.eye(4)
    camera[:3, :3] = (
        (1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)),
        (2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)),
        (2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)),
    )
    camera[:3, 3] = pose[:3]
    return camera


def _checked_pair(images, row, original, target, crop_key, rgb_key, error_key, interval_key,
                  ext_key, intr_key):
    window = row["window"]
    _require(original["window"] == target["window"] == window, "RGB-only window order differs")
    _require(images.shape == (2, 224, 224, 3) and images.dtype == np.uint8 and images.flags.c_contiguous,
             "RGB-only crop storage differs")
    _require(original.get("state") == "BOUND" or original.get("bound_pair") is True,
             "original pair is not bound")
    _require(len(row["frames"]) == len(original["frames"]) == 2, "RGB-only frame count differs")
    pose_valid, camera_to_world, ids, pose_ids = [], [], [], []
    for index, (image, public, private) in enumerate(zip(images, row["frames"], original["frames"])):
        _require(_pin(image.tobytes()) == public[crop_key], "RGB-only crop identity or frame order differs")
        _require(private["state"] == "BOUND" and private["frame_index"] == index
                 and private["BAG_connection"] == 4
                 and {"bytes": private["RGB8_bytes"], "SHA256": private["RGB8_SHA256"]} == public[rgb_key],
                 "original frame identity or frame order differs")
        source_pose = private["original_pose"]
        bag_stamp, pose_stamp = private["BAG_message_time_ns"], source_pose["TUM_pose_timestamp_ns"]
        error = abs(bag_stamp - pose_stamp)
        _require(error == private["absolute_BAG_vs_original_pose_error_ns"] == public[error_key]
                 and error < 20_000_000, "original pose timestamp alignment differs")
        camera_to_world.append(_camera_to_world(source_pose["tx_ty_tz_qx_qy_qz_qw_text"]))
        ids.append(bag_stamp)
        pose_ids.append(pose_stamp)
        pose_valid.append(True)
    _require(ids[0] < ids[1] and pose_ids[0] < pose_ids[1]
             and ids[1] - ids[0] == original["exact_BAG_frame_interval_ns"] == row[interval_key],
             "RGB-only pair timestamp order differs")
    _require("pose_valid" not in target or target["pose_valid"] == pose_valid,
             "pinned pose validity differs")
    extrinsics = np.asarray(target[ext_key], dtype=np.float64)
    first_to_second = np.linalg.solve(camera_to_world[1], camera_to_world[0])
    # The first camera defines the coordinate origin in the pinned target.
    expected = np.stack([np.eye(4)[:3], first_to_second[:3]])
    _require(extrinsics.shape == (2, 3, 4) and np.isfinite(extrinsics).all()
             and np.allclose(extrinsics, expected, rtol=0, atol=1e-10)
             and np.linalg.norm(first_to_second[:3, 3]) > 0, "stored camera extrinsics differ from the matched poses")
    intrinsics_tuple = np.asarray(target[intr_key], dtype=np.float64)
    _require(intrinsics_tuple.shape == (4,) and np.isfinite(intrinsics_tuple).all()
             and (intrinsics_tuple[:2] > 0).all() and (intrinsics_tuple[2:] >= 0).all()
             and (intrinsics_tuple[2:] < 224).all(), "invalid pinned crop intrinsics")
    fx, fy, cx, cy = intrinsics_tuple
    intrinsic_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    return {
        "images": images, "extrinsics": extrinsics.astype(np.float32),
        "intrinsics": np.repeat(intrinsic_matrix[None], len(ids), axis=0),
        "pose_valid": np.asarray(pose_valid, dtype=np.bool_), "seq_name": SEQUENCE, "ids": ids,
    }


def read_rgb_only_room(root: Path):
    """Return all seven bound pairs in frozen order, or fail before training."""
    batches = []
    seen_ids = set()
    for group, windows, suffix, public_sha, bom_name, bom_sha, crop_name, pose_name, *keys in _GROUPS:
        stem = _STEM + group
        public = json.loads(_verified(root / PUBLIC_ROOT / (stem + suffix), {"SHA256": public_sha}, 16384))
        private = root / "local-only"
        bom = json.loads(_verified(private / (stem + "-v1") / bom_name, {"SHA256": bom_sha}, 16384))
        singleton = len(windows) == 1
        block = public["actual"] if singleton else public
        _require(block["sequence"] == bom["sequence"] == SEQUENCE, "RGB-only sequence differs")
        _require(singleton or public["window_denominator"] == bom["original_frozen_window_denominator"] == 12,
                 "RGB-only source denominator differs")
        output = private / (stem + "-rgb224-v1")
        pins = block["private_output_pins"]
        images = np.load(io.BytesIO(_verified(output / crop_name, pins[crop_name], 1_000_000)), allow_pickle=False)
        targets = json.loads(_verified(output / pose_name, pins[pose_name], 16384))
        rows = [block] if singleton else block["windows"]
        originals = [bom] if singleton else bom["new_windows"]
        targets = [targets] if singleton else targets
        images = images[None] if singleton else images
        _require(images.shape == (len(windows), 2, 224, 224, 3) and len(rows) == len(originals) == len(targets) == len(windows)
                 and tuple(row["window"] for row in rows) == windows, "RGB-only group shape or window order differs")
        for crop, row, original, target in zip(images, rows, originals, targets):
            batch = _checked_pair(crop, row, original, target, *keys)
            _require(seen_ids.isdisjoint(batch["ids"]), "duplicate RGB-only frame identity")
            seen_ids.update(batch["ids"])
            batches.append(batch)
    return batches
