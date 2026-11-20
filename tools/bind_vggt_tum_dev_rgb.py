"""Bind frozen cross-camera pose anchors to original TUM RGB indexes and bytes."""

import argparse
from bisect import bisect_left
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import struct


ANCHOR_SHA = "94779c64c6832808e4402d06b96c5e276fb230c6833208d6c9966cb5c3ae1aa9"
MAX_TIME_ERROR = 0.02
MAX_IMAGE_BYTES = 4 * 1024**2
PNG_HEADER = b"\x89PNG\x0d\x0a\x1a\x0a\x00\x00\x00\x0dIHDR"
CALIBRATION_URL = "https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats#calibration_of_the_color_camera"
# TUM RGB intrinsics in pixel coordinates and Brown-Conrady d0..d4.
CALIBRATION = {
    "freiburg1_plant": ((517.3, 516.5, 318.6, 255.3), (0.2624, -0.9531, -0.0054, 0.0026, 1.1633)),
    "freiburg2_dishes": ((520.9, 521.0, 325.1, 249.7), (0.2312, -0.7849, -0.0033, -0.0001, 0.9172)),
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_bounded(path, limit):
    if path.is_symlink() or not path.is_file():
        raise ValueError("expected original regular file")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if not 0 < len(raw) <= limit:
        raise ValueError("original file is outside its byte bound")
    return raw


def parse_rgb_index(raw):
    rows = []
    for line in raw.decode("utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        timestamp, path = line.split()
        time = float(timestamp)
        filename = re.fullmatch(r"rgb/([0-9]+\.[0-9]+)\.png", path)
        if filename is None or filename[1] != timestamp or not math.isfinite(time):
            raise ValueError("invalid official TUM RGB row")
        if rows and time <= rows[-1][0]:
            raise ValueError("duplicate or unordered official RGB timestamp")
        rows.append((time, path))
    if not rows:
        raise ValueError("empty original RGB frame index")
    return rows


def calibration(sequence):
    (fx, fy, cx, cy), distortion = CALIBRATION[sequence]
    scale = 224 / 480
    return dict(source=CALIBRATION_URL, original_pixels=[640, 480], original_K=[fx, fy, cx, cy],
                original_distortion=distortion, planned_preprocess="RGB-only rectification with original K, center crop, resize",
                crop_box_xyxy=[80, 0, 560, 480], output_pixels=[224, 224],
                output_K=[fx * scale, fy * scale, (cx - 80 + .5) * scale - .5, (cy + .5) * scale - .5],
                rectification_verified=False, depth_used=False)


def rectify_rgb(raw, sequence):
    """Use the inverse image map, sampling the original RGB only."""
    import io

    import numpy as np
    from PIL import Image

    with Image.open(io.BytesIO(raw)) as source:
        if source.format != "PNG" or source.size != (640, 480) or source.mode != "RGB":
            raise ValueError("expected the original RGB frame format")
        image = np.asarray(source)
    fx, fy, cx, cy = calibration(sequence)["original_K"]
    k1, k2, p1, p2, k3 = CALIBRATION[sequence][1]
    scale = 224 / 480
    y_out, x_out = np.indices((224, 224), dtype=np.float64)
    x = ((x_out + .5) / scale - .5 + 80 - cx) / fx
    y = ((y_out + .5) / scale - .5 - cy) / fy
    r2 = x*x + y*y
    radial = 1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2
    u = fx * (x*radial + 2*p1*x*y + p2*(r2 + 2*x*x)) + cx
    v = fy * (y*radial + p1*(r2 + 2*y*y) + 2*p2*x*y) + cy
    valid = (0 <= u) & (u <= 639) & (0 <= v) & (v <= 479)
    x0 = np.clip(np.floor(u).astype(np.int64), 0, 639)
    y0 = np.clip(np.floor(v).astype(np.int64), 0, 479)
    x1, y1 = np.minimum(x0 + 1, 639), np.minimum(y0 + 1, 479)
    a, b = (u-x0)[..., None], (v-y0)[..., None]
    pixels = (1-b) * ((1-a)*image[y0,x0] + a*image[y0,x1]) + b * ((1-a)*image[y1,x0] + a*image[y1,x1])
    output = np.where(valid[..., None], np.rint(pixels).clip(0, 255), 0).astype(np.uint8)
    return output, float(np.mean(valid))


def nearest_rgb(rows, times, pose_time):
    i = bisect_left(times, pose_time)
    neighbors = rows[max(0, i - 1):i + 1]
    return min(neighbors, key=lambda candidate: (abs(candidate[0] - pose_time), candidate[0]))


def bind_pairs(anchors, rgb):
    if len(anchors) != 12 or [row["window"] for row in anchors] != list(range(1, 13)):
        raise ValueError("expected complete ordered twelve-window pose plan")
    times = [row[0] for row in rgb]
    result = []
    used = set()
    for anchor in anchors:
        if anchor["state"] != "POSE_PAIR_CANDIDATE":
            result.append(dict(window=anchor["window"], state=anchor["state"]))
            continue
        frames = [nearest_rgb(rgb, times, time) for time in anchor["original_pose_times"]]
        errors = [abs(frame[0] - time) for frame, time in zip(frames, anchor["original_pose_times"], strict=True)]
        paths = [frame[1] for frame in frames]
        if max(errors) > MAX_TIME_ERROR:
            state = "RGB_TIMESTAMP_GAP"
        elif frames[0][0] >= frames[1][0] or any(path in used for path in paths):
            state = "RGB_REUSED_OR_REVERSED"
        else:
            state = "RGB_INDEX_MATCHED"
            used.update(paths)
        result.append(dict(window=anchor["window"], state=state, frames=paths, rgb_times=[row[0] for row in frames],
                           rgb_pose_time_error_seconds=errors))
    return result


def pin_frames(rows, media):
    root = media.resolve()
    for row in rows:
        if row["state"] != "RGB_INDEX_MATCHED":
            continue
        pins = []
        try:
            for name in row["frames"]:
                path = root / name
                if path.resolve(strict=True) != path:
                    raise ValueError("RGB frame path aliases another file")
                data = read_bounded(path, MAX_IMAGE_BYTES)
                if data[:16] != PNG_HEADER or len(data) < 33 or struct.unpack(">II", data[16:24]) != (640, 480):
                    raise ValueError("original RGB bytes are not a 640x480 PNG")
                pins.append(dict(path=name, bytes=len(data), SHA256=sha(data)))
        except (OSError, ValueError):
            row["state"] = "RGB_MEDIA_MISSING_OR_INVALID"
        else:
            row["media"] = pins
            row["state"] = "RGB_PNG_BYTES_BOUND"


def validate_archive_receipt(sequence, root, index):
    receipt = json.loads(read_bounded(root / "official-stream-receipt-private.json", 65536))
    family = sequence.split("_", 1)[0]
    archive = f"https://webshare.cvg.cit.tum.de/g/rgbd/dataset/{family}/rgbd_dataset_{sequence}.tgz"
    if (receipt["passed"] is not True or receipt["sequence"] != sequence or receipt["archive"] != archive
            or receipt["frozen_anchor_SHA256"] != ANCHOR_SHA or receipt["rgb_index_sha256"] != sha(index)):
        raise ValueError("original TUM archive stream or RGB index identity differs")
    return receipt["selected_frames"]


def bind(source, output, media_root=None, archive_root=None):
    if archive_root is not None and media_root is not None:
        raise ValueError("archive and independent media roots cannot be combined")
    raw = read_bounded(source / "dev-anchor-plan-private.json", 8192)
    if sha(raw) != ANCHOR_SHA:
        raise ValueError("frozen pose plan changed")
    anchors = json.loads(raw)
    if set(anchors) != set(CALIBRATION):
        raise ValueError("unexpected pose sequences")
    result = {}
    summary = {}
    for sequence in CALIBRATION:
        index_path = archive_root / sequence / "rgb.txt" if archive_root else source / f"{sequence}-rgb.txt"
        index = read_bounded(index_path, 3 * 1024**2)
        rgb = parse_rgb_index(index)
        rows = bind_pairs(anchors[sequence], rgb)
        if archive_root is not None:
            archive_frames = validate_archive_receipt(sequence, archive_root / sequence, index)
            pin_frames(rows, archive_root / sequence)
            for row in rows:
                if row["state"] == "RGB_PNG_BYTES_BOUND" and any(
                        archive_frames.get(pin["path"]) != {"bytes": pin["bytes"], "SHA256": pin["SHA256"]}
                        for pin in row["media"]):
                    raise ValueError("bound original RGB bytes differ from the stream receipt")
        if media_root is not None:
            pin_frames(rows, media_root / sequence)
        result[sequence] = dict(calibration=calibration(sequence), windows=rows, original_rgb_index_SHA256=sha(index))
        summary[sequence] = dict(original_rgb_index_SHA256=sha(index), rgb_index_rows=len(rgb),
                                 states=dict(Counter(row["state"] for row in rows)))
    private = (json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    with output.open("xb") as stream:
        stream.write(private)
    return dict(sequences=summary, planned_pairs=24, private_binding_SHA256=sha(private), private_binding_bytes=len(private),
                actual_predictions=0, rectification_verified=False, formal_CO3D=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--media-root", type=Path)
    parser.add_argument("--archive-root", type=Path)
    args = parser.parse_args()
    print(json.dumps(bind(args.source, args.output, args.media_root, args.archive_root), sort_keys=True))


if __name__ == "__main__":
    main()
