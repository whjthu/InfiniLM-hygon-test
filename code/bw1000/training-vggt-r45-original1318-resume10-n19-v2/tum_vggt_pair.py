"""Prepare one pinned Freiburg 3 RGB-D pair for VGGT on CPU."""

import argparse
from bisect import bisect_left
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image


FRAME_SIZE = (640, 480)
IMAGE_SIZE = 224
CROP = (80, 0, 560, 480)
K = np.array(((535.4, 0, 320.1), (0, 539.2, 247.6), (0, 0, 1)))
SAMPLING_OFFSET = Decimal("0.020")
PAIR_SPACING = (Decimal("0.100"), Decimal("0.500"))
MAX_FILE_BYTES = 4 * 1024 * 1024


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_pinned(path, expected):
    with path.open("rb") as stream:
        raw = stream.read(MAX_FILE_BYTES + 1)
    if not raw or len(raw) > MAX_FILE_BYTES or digest(raw) != expected:
        raise ValueError(f"missing, oversized, or changed input {path.name}")
    return raw


def stamp(text):
    try:
        value = Decimal(text)
    except InvalidOperation as error:
        raise ValueError("invalid TUM timestamp") from error
    if not value.is_finite() or value <= 0:
        raise ValueError("invalid TUM timestamp")
    return value


def parse_index(raw, kind):
    times = []
    values = []
    for line in raw.decode("utf-8").splitlines():
        line = line.partition("#")[0].split()
        if not line:
            continue
        time = stamp(line[0])
        if times and time <= times[-1]:
            raise ValueError(f"unordered or duplicate {kind} timestamp")
        if kind == "pose":
            if len(line) != 8:
                raise ValueError("pose requires XYZ and XYZW")
            pose = np.array(line[1:], dtype=np.float64)
            if not np.isfinite(pose).all() or abs(np.linalg.norm(pose[3:]) - 1) > 0.002:
                raise ValueError("invalid capture pose")
            values.append(pose)
        else:
            if len(line) != 2 or not re.fullmatch(kind + r"/\d+\.\d+\.png", line[1]):
                raise ValueError(f"invalid {kind} frame path")
            if stamp(Path(line[1]).stem) != time:
                raise ValueError(f"{kind} index and frame timestamp differ")
            values.append(line[1])
        times.append(time)
    if not times:
        raise ValueError(f"empty {kind} index")
    return times, values


def nearest(index, query):
    times, values = index
    if not times[0] <= query <= times[-1]:
        raise ValueError("RGB timestamp lies outside synchronized coverage")
    at = bisect_left(times, query)
    selected = min(range(max(0, at - 1), min(len(times), at + 1)), key=lambda k: abs(times[k] - query))
    delta = times[selected] - query
    if abs(delta) > SAMPLING_OFFSET:
        raise ValueError("RGB match exceeds 20 ms")
    return times[selected], values[selected], delta


def camera_to_world(pose):
    x, y, z, w = pose[3:] / np.linalg.norm(pose[3:])
    matrix = np.eye(4)
    matrix[:3, :3] = (
        (1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)),
        (2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)),
        (2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)),
    )
    matrix[:3, 3] = pose[:3]
    return matrix


def read_frame(raw, kind):
    with Image.open(io.BytesIO(raw)) as image:
        modes = {"RGB"} if kind == "rgb" else {"I;16", "I;16B"}
        if image.format != "PNG" or image.size != FRAME_SIZE or image.mode not in modes:
            raise ValueError(f"{kind} PNG geometry or type differs from TUM Freiburg 3")
        resample = Image.Resampling.BILINEAR if kind == "rgb" else Image.Resampling.NEAREST
        array = np.asarray(image.crop(CROP).resize((IMAGE_SIZE, IMAGE_SIZE), resample)).copy()
    return array if kind == "rgb" else array.astype(np.float32) / 5000


def intrinsics():
    scale = IMAGE_SIZE / (CROP[2] - CROP[0])
    offset_x = (-CROP[0] + 0.5) * scale - 0.5
    offset_y = (-CROP[1] + 0.5) * scale - 0.5
    return np.array(((scale, 0, offset_x), (0, scale, offset_y), (0, 0, 1))) @ K


def reprojection(source, target, k, relative):
    grid = np.indices(source.shape)[:, ::4, ::4]
    v, u = grid.reshape(2, -1)
    valid = source[v, u] > 0
    v, u = v[valid], u[valid]
    rays = np.linalg.solve(k, np.stack((u, v, np.ones_like(u))))
    points = relative[:3, :3] @ (rays * source[v, u]) + relative[:3, 3, None]
    points = points[:, points[2] > 1e-3]
    xy = k @ points
    uv = np.rint(xy[:2] / xy[2]).astype(np.int64)
    in_frame = np.all((uv >= 0) & (uv < IMAGE_SIZE), axis=0)
    projected = points[2, in_frame]
    observed = target[uv[1, in_frame], uv[0, in_frame]]
    comparable = observed > 0
    residual = abs(projected[comparable] - observed[comparable])
    consistent = residual <= np.maximum(0.05, projected[comparable] * 0.03)
    if comparable.sum() < 100 or consistent.sum() < len(u) * 0.25:
        raise ValueError("insufficient bidirectional metric depth agreement")
    return dict(source_valid=int(valid.sum()), projected_in_frame=int(in_frame.sum()),
                comparable=int(comparable.sum()), consistent=int(consistent.sum()),
                median_abs_depth_error_m=float(np.median(residual)))


def prepare_pair(root, manifest):
    """Return raw arrays compatible with the existing VGGT batch converter and a small receipt."""
    sequence = manifest["sequence"]
    if not re.fullmatch(r"freiburg3_[a-z0-9_]+", sequence):
        raise ValueError("this calibration is specific to Freiburg 3")
    pins = manifest["files_SHA256"]
    used = {}

    def read(name):
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("frame escapes dataset root")
        raw = read_pinned(path, pins[name])
        used[name] = pins[name]
        return raw

    rgb_times, rgb_files = parse_index(read("rgb.txt"), "rgb")
    depth = parse_index(read("depth.txt"), "depth")
    pose = parse_index(read("groundtruth.txt"), "pose")
    ids = manifest["rgb_indices"]
    if (len(ids) != 2 or any(type(i) is not int for i in ids)
            or not 0 <= ids[0] < ids[1] < len(rgb_times)
            or not PAIR_SPACING[0] <= rgb_times[ids[1]] - rgb_times[ids[0]] <= PAIR_SPACING[1]):
        raise ValueError("expected two fixed, ordered RGB indices 100–500 ms apart")
    images, depths, transforms, matches = [], [], [], []
    for index in ids:
        time = rgb_times[index]
        depth_stamp, depth_name, depth_delta = nearest(depth, time)
        pose_stamp, row, pose_delta = nearest(pose, time)
        images.append(read_frame(read(rgb_files[index]), "rgb"))
        depths.append(read_frame(read(depth_name), "depth"))
        transforms.append(camera_to_world(row))
        matches.append(dict(rgb=str(time), depth=str(depth_stamp), pose=str(pose_stamp),
                            depth_offset_s=str(depth_delta), pose_offset_s=str(pose_delta)))
    if len(used) != 7 or used.keys() != pins.keys():
        raise ValueError("the manifest must identify three indices and four distinct frames")
    k = intrinsics()
    second_from_first = np.linalg.solve(transforms[1], transforms[0])
    baseline = np.linalg.norm(second_from_first[:3, 3])
    if baseline < 0.005:
        raise ValueError("degenerate translation for two-frame qualification")
    check = [reprojection(depths[0], depths[1], k, second_from_first),
             reprojection(depths[1], depths[0], k, np.linalg.inv(second_from_first))]
    batch = dict(images=images, depths=depths, intrinsics=[k, k], seq_name=sequence, ids=ids,
                 extrinsics=[np.eye(4)[:3], second_from_first[:3]], point_masks=[z > 0 for z in depths])
    report = dict(sequence=sequence, rgb_indices=ids, matches=matches, files_SHA256=used,
                  baseline_m=float(baseline), depth_reprojection=check, raw_rgb_shape=list(images[0].shape),
                  valid_depth_pixels=[int(mask.sum()) for mask in batch["point_masks"]])
    return batch, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--manifest-sha", required=True)
    args = parser.parse_args()
    manifest = json.loads(read_pinned(args.manifest, args.manifest_sha))
    _, report = prepare_pair(args.root, manifest)
    print(json.dumps(dict(manifest_SHA256=args.manifest_sha, **report), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
