"""Produce 224-pixel RGB inputs and RGB-time relative poses for the VGGT cohort."""

import argparse
import json
from pathlib import Path
import re

import numpy as np

from bind_vggt_tum_dev_rgb import (
    CALIBRATION, MAX_IMAGE_BYTES, calibration, read_bounded, rectify_rgb, sha, validate_archive_receipt,
)
from plan_vggt_tum_public_dev_trajectories import MAX_TIMESTAMP_ERROR, MIN_TRANSLATION, SOURCES, nearest, parse_trajectory


def camera_to_world(pose):
    xyz, quaternion = pose[1:4], np.asarray(pose[4:], dtype=np.float64)
    x, y, z, w = quaternion / np.linalg.norm(quaternion)
    transform = np.eye(4)
    transform[:3, :3] = (
        (1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)),
        (2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)),
        (2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)),
    )
    transform[:3, 3] = xyz
    return transform


def synchronized_pose(poses, pose_times, rgb_times):
    selected = [nearest(poses, pose_times, time) for time in rgb_times]
    errors = [abs(pose[0] - time) for pose, time in zip(selected, rgb_times, strict=True)]
    if max(errors) > MAX_TIMESTAMP_ERROR or selected[0][0] >= selected[1][0]:
        raise ValueError("RGB frames lack distinct nearby original mocap poses")
    a, b = [camera_to_world(pose) for pose in selected]
    second_from_first = np.linalg.solve(b, a)
    baseline = float(np.linalg.norm(second_from_first[:3, 3]))
    if baseline < MIN_TRANSLATION:
        raise ValueError("RGB-time relative translation is below the frozen floor")
    return dict(world_to_camera=[np.eye(4)[:3].tolist(), second_from_first[:3].tolist()],
                original_mocap_times=[pose[0] for pose in selected], rgb_pose_time_error_seconds=errors,
                baseline_m=baseline)


def prepare_pair(sequence, row, poses, times, media):
    frames, validity = [], []
    require_media = row["media"]
    if len(require_media) != 2 or [entry["path"] for entry in require_media] != row["frames"]:
        raise ValueError("expected two exact original RGB member pins")
    members = [re.fullmatch(r"rgb/([0-9]+\.[0-9]+)\.png", item["path"]) for item in require_media]
    if any(member is None for member in members):
        raise ValueError("RGB archive member identity differs")
    rgb_times = [float(member[1]) for member in members]
    if rgb_times != row["rgb_times"] or rgb_times[0] >= rgb_times[1]:
        raise ValueError("RGB timestamps differ from ordered archive members")
    geometry = synchronized_pose(poses, times, rgb_times)
    for item in require_media:
        path = media / item["path"]
        if path.resolve(strict=True) != path:
            raise ValueError("original RGB member aliases another file")
        raw = read_bounded(path, MAX_IMAGE_BYTES)
        if (len(raw), sha(raw)) != (item["bytes"], item["SHA256"]):
            raise ValueError("original RGB content changed")
        image, valid = rectify_rgb(raw, sequence)
        if image.shape != (224, 224, 3) or image.dtype != np.uint8 or valid < 0.95:
            raise ValueError("rectified RGB proof is outside the qualified camera area")
        frames.append(image)
        validity.append(valid)
    return np.stack(frames), dict(geometry, rgb_times=rgb_times, intrinsics=calibration(sequence)["output_K"],
                                  valid_pixel_fraction=validity)


def prepare(source, binding_file, binding_sha, archives, output):
    raw = read_bounded(binding_file, 65536)
    if sha(raw) != binding_sha:
        raise ValueError("frozen RGB binding SHA differs")
    binding = json.loads(raw)
    if set(binding) != set(CALIBRATION):
        raise ValueError("unexpected paired development groups")
    frozen = {}
    for sequence, doc in binding.items():
        expected_calibration = json.loads(json.dumps(calibration(sequence)))
        if doc["calibration"] != expected_calibration:
            raise ValueError("official camera calibration differs")
        expected_size, wanted = SOURCES[sequence]
        pose_raw = read_bounded(source / f"{sequence}.txt", expected_size)
        if len(pose_raw) != expected_size or sha(pose_raw) != wanted:
            raise ValueError("official mocap input differs")
        poses, _ = parse_trajectory(pose_raw)
        windows = doc["windows"]
        if [row["window"] for row in windows] != list(range(1, 13)):
            raise ValueError("ordered twelve-window source denominator differs")
        archive = archives / sequence
        index = read_bounded(archive / "rgb.txt", 3 * 1024**2)
        if sha(index) != doc["original_rgb_index_SHA256"]:
            raise ValueError("bound RGB index differs from the completed official archive")
        selected = validate_archive_receipt(sequence, archive, index)
        for row in windows:
            if row["state"] == "RGB_PNG_BYTES_BOUND" and any(
                    selected.get(entry["path"]) != {"bytes": entry["bytes"], "SHA256": entry["SHA256"]}
                    for entry in row["media"]):
                raise ValueError("RGB pins differ from the completed official archive receipt")
        frozen[sequence] = (doc, poses, [r[0] for r in poses])

    output.mkdir(mode=0o700)
    rows, verified = [], 0
    for sequence, (doc, poses, times) in frozen.items():
        for row in doc["windows"]:
            window = row["window"]
            outcome = dict(sequence=sequence, window=window, source_state=row["state"])
            if row["state"] == "RGB_PNG_BYTES_BOUND":
                try:
                    images, geometry = prepare_pair(sequence, row, poses, times, (archives / sequence).resolve())
                except (OSError, ValueError, np.linalg.LinAlgError) as error:
                    outcome.update(state="RGB224_PRECONDITION_FAILED", reason=type(error).__name__)
                else:
                    name = f"{sequence}-window-{window:02d}-rgb224.npy"
                    with (output / name).open("xb") as stream:
                        np.save(stream, images, allow_pickle=False)
                    artifact = (output / name).read_bytes()
                    outcome.update(state="RGB224_AND_RELATIVE_POSE_BOUND", rgb_file=name,
                                   rgb_file_SHA256=sha(artifact), rgb_file_bytes=len(artifact),
                                   original_frames=row["media"], **geometry)
                    verified += 1
            else:
                outcome.update(state="RGB224_PRECONDITION_FAILED", reason=row["state"])
            rows.append(outcome)
    private = (json.dumps({"binding_SHA256": binding_sha, "windows": rows}, separators=(",", ":"),
                          sort_keys=True, allow_nan=False) + "\n").encode()
    with (output / "inputs-private.json").open("xb") as stream:
        stream.write(private)
    return dict(planned_pairs=len(rows), rectified_pose_pairs=verified, failed_pairs=len(rows)-verified,
                manifest_SHA256=sha(private), actual_model_predictions=0, actual_depth_reads=0, formal_CO3D=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "binding", "archives", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--binding-sha", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.binding, args.binding_sha, args.archives, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
