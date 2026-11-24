"""Measure BVH length and axes against the publisher's matching C3D and TXT clip."""

import argparse
from hashlib import sha256
from io import BytesIO
from itertools import permutations, product
import json
from pathlib import Path
import struct
from zipfile import ZipFile

import numpy as np


ASSETS = {
    "quickwalk_BVH.ZIP": (181817, "5d6093360ace16d873f96c43be822bd88ea09e8a7f65b3e9df78b9f4be174e3c", "eric1.bvh", 3091735),
    "quickwalkC3D.ZIP": (92479, "eb7e7ae35f6c8dc11a93cd93ec5c5d5d5016aeeedf52ba31e36a26b115140f82", "QkWalk1.c3d", 106840),
    "quickwalkTXT.ZIP": (108118, "f7926fa55992ae4a8668181328084c45a7f95b992c33e35a148b7fd34499afe6", "Aug210104.txt", 281141),
}
MARKERS = ("LBWT", "RBWT", "LFWT", "RFWT")


def read_asset(directory, filename):
    size, pin, member, member_size = ASSETS[filename]
    with (directory / filename).open("rb") as stream:
        raw = stream.read(size + 1)
    if len(raw) != size or sha256(raw).hexdigest() != pin:
        raise ValueError(f"publisher archive differs: {filename}")
    with ZipFile(BytesIO(raw)) as archive:
        if archive.namelist() != [member] or archive.getinfo(member).file_size != member_size:
            raise ValueError(f"publisher member differs: {filename}")
        return archive.read(member)


def public_c3d_units(raw):
    if len(raw) < 1024 or raw[1] != 80 or raw[0] < 2:
        raise ValueError("unexpected Quick Walk C3D file header")
    index = (raw[0] - 1) * 512
    if index + 4 > len(raw):
        raise ValueError("Quick Walk C3D parameter block is missing")
    end = index + raw[index + 2] * 512
    if raw[index + 3] != 85 or end > len(raw):
        raise ValueError("unexpected Quick Walk C3D parameter header")
    groups, units = {}, {}
    pos = index + 4
    while pos + 4 <= end:
        length, group = struct.unpack_from("bb", raw, pos)
        if not length or not group:
            break
        offset_position = pos + 2 + abs(length)
        if offset_position + 2 > end:
            raise ValueError("C3D parameter name exceeds section")
        name = raw[pos + 2:offset_position].decode("ascii")
        offset = struct.unpack_from("<H", raw, offset_position)[0]
        if offset < 3 or offset_position + offset > end:
            raise ValueError("C3D parameter offset exceeds section")
        data = raw[offset_position + 2:offset_position + offset]
        if group < 0:
            groups[name] = -group
        elif name == "UNITS":
            units[group] = data
        pos = offset_position + offset
    if units.get(groups.get("POINT")) != b"\xff\x01\x02mm\x00":
        raise ValueError("Quick Walk POINT:UNITS is not the pinned mm record")
    return "mm"


def read_c3d_points(raw):
    def vax_float(position):
        word = raw[position:position + 4]
        return struct.unpack("<f", word[2:] + word[:2])[0] / 4.0

    units = public_c3d_units(raw)
    points, analog, first, last = struct.unpack_from("<HHHH", raw, 2)
    block = struct.unpack_from("<H", raw, 16)[0]
    scale, rate = vax_float(12), vax_float(20)
    frames = last - first + 1
    offset = (block - 1) * 512
    if (points, analog, frames, rate) != (41, 0, 307, 120.0) or not 0 < scale < 1 or offset < 512:
        raise ValueError("unexpected Quick Walk C3D integer point geometry")
    body = np.frombuffer(raw, dtype="<i2", count=frames * points * 4, offset=offset)
    return units, rate, scale, body.reshape(frames, points, 4)[..., :3] * scale


def fit_axes(source, target):
    half = len(source) // 2
    wanted = target[:half] - target[:half].mean(axis=0)
    results = []
    for permutation in permutations(range(3)):
        for signs in product((-1, 1), repeat=3):
            basis = np.eye(3)[:, permutation] * signs
            if np.linalg.det(basis) < 0:
                continue
            changed = source @ basis
            centered = changed[:half] - changed[:half].mean(axis=0)
            scale = np.sum(centered * wanted) / np.sum(centered ** 2)
            if scale <= 0:
                continue
            shift = target[:half].mean(axis=0) - scale * changed[:half].mean(axis=0)
            norm = np.linalg.norm(target - (scale * changed + shift), axis=1)
            results.append({
                "target_axis_source_indices": permutation,
                "axis_signs": signs,
                "target_mm_per_bvh_unit": float(scale),
                "fit_RMS_mm": float(np.sqrt(np.mean(norm[:half] ** 2))),
                "validation_RMS_mm": float(np.sqrt(np.mean(norm[half:] ** 2))),
                "validation_max_mm": float(np.max(norm[half:])),
            })
    results.sort(key=lambda record: record["fit_RMS_mm"])
    return half, results


def audit(directory):
    c3d = read_asset(directory, "quickwalkC3D.ZIP")
    units, rate, scale, points = read_c3d_points(c3d)
    txt = read_asset(directory, "quickwalkTXT.ZIP").decode("ascii").splitlines()
    columns = txt[0].split("\t")
    row_data = np.asarray([tuple(map(float, line.split())) for line in txt[1:] if line.strip()])
    restored = row_data[:, 2:].reshape(points.shape)
    valid = (restored != -9999.99).all(axis=2)
    differences = np.abs(points - restored)[valid]
    if np.count_nonzero(valid) != 12406 or np.max(differences) >= 0.2:
        raise ValueError("C3D source points do not reproduce TXT values")
    marker_indices = [columns.index(f"EricAutoLabel:{name}:{axis}") for name in MARKERS for axis in "XYZ"]
    markers = row_data[:, marker_indices].reshape(len(row_data), len(MARKERS), 3)
    complete = (markers != -9999.99).all(axis=(1, 2))
    bvh = read_asset(directory, "quickwalk_BVH.ZIP").decode("ascii").splitlines()
    motion = bvh.index("MOTION")
    if (bvh[motion + 1].split() != ["Frames:", "7200"]
            or bvh[motion + 2].split() != ["Frame", "Time:", "0.008333"]
            or bvh[4].split()[2:5] != ["Xposition", "Yposition", "Zposition"]):
        raise ValueError("Quick Walk BVH format differs")
    translation = np.asarray([tuple(map(float, row.split()[:3])) for row in bvh[motion + 3:] if row.strip()])
    changed = np.flatnonzero(np.linalg.norm(np.diff(translation, axis=0), axis=1) > 0.01)
    if translation.shape != (7200, 3) or not np.array_equal(changed, np.arange(4 * (len(points) - 1))):
        raise ValueError("Quick Walk BVH active lattice differs")
    basis = translation[4 * np.arange(len(points))][complete]
    target = markers[complete].mean(axis=1)
    half, candidates = fit_axes(basis, target)
    if len(basis) != 304 or len(candidates) < 2:
        raise ValueError("incomplete physical calibration sample")
    return {
        "sha256": {name: asset[1] for name, asset in ASSETS.items()},
        "C3D_point_units": units, "C3D_rate_Hz": rate, "C3D_integer_scale": scale,
        "C3D_vs_TXT_max_difference_mm": float(np.max(differences)),
        "C3D_vs_TXT_valid_marker_frames": int(np.count_nonzero(valid)),
        "declared_BVH_frame_period": 0.008333, "BVH_frames": len(translation),
        "BVH_active_frame_intervals": len(changed), "BVH_alignment_stride_in_this_sample": 4,
        "fit_samples": half, "validation_samples": len(basis) - half,
        "compared_positive_scale_hypotheses": len(candidates),
        "best_by_fit_RMS": candidates[0], "runner_up_by_fit_RMS": candidates[1],
        "axes_selected_for_this_sample_only": True,
        "source_training_archive_scale_confirmed": False, "training_rows_generated": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = (json.dumps(audit(args.directory), sort_keys=True, indent=2) + "\n").encode()
    with args.output.open("xb") as stream:
        stream.write(data)
    print(sha256(data).hexdigest())


if __name__ == "__main__":
    main()
