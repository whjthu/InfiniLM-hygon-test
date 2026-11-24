"""Calibrate exact ACCAD cohort BVH turns against corresponding C3D marker captures."""

import argparse
from hashlib import sha256
from io import BytesIO
from itertools import permutations, product
import json
from pathlib import Path
import struct
from zipfile import ZipFile

import numpy as np

from audit_hymotion_accad_bvh import ARCHIVES
from audit_hymotion_accad_quickwalk_scale import MARKERS, public_c3d_units
from audit_hymotion_accad_skeletons import SELECTION_SHA, skeleton


COHORTS = {
    "Female1": (8255188, "43f37f2741b04b40c389c37862c28129db61bcd7cd7c435695d518ac58bcc7b3",
                {10: "B10 - walk turn left (45).c3d", 11: "B11 - walk turn left (135).c3d"}),
    "Male1": (8522050, "cfefdc2b4bff95dc453a6233244a184bee33299ae28d4e86ae04a17d0446f6cc",
              {10: "Walk B10 - Walk turn left 45.c3d", 11: "Walk B11 - Walk turn left 135.c3d"}),
    "Male2": (6678050, "64d8c8b85bbaf801af3fbbadfee8f6441ae952962fb2937f0af41292ce4ff4a1",
              {10: "B10 -  Walk turn left 45.c3d", 11: "B11 -  Walk turn left 135.c3d"}),
}


def read_archive(path, size, pin):
    with path.open("rb") as stream:
        raw = stream.read(size + 1)
    if len(raw) != size or sha256(raw).hexdigest() != pin:
        raise ValueError(f"publisher archive differs: {path.name}")
    return ZipFile(BytesIO(raw))


def c3d_waist(raw):
    units = public_c3d_units(raw)
    points, analog, first, last = struct.unpack_from("<HHHH", raw, 2)
    block = struct.unpack_from("<H", raw, 16)[0]
    scale = struct.unpack("<f", raw[14:16] + raw[12:14])[0] / 4
    rate = struct.unpack("<f", raw[22:24] + raw[20:22])[0] / 4
    frame_count = last - first + 1
    if (points, analog, rate) != (41, 0, 120.0) or not 0 < scale < 1 or not 0 < frame_count < 2500:
        raise ValueError("expected 120Hz DEC integer capture")
    start = (raw[0] - 1) * 512
    end = start + raw[start + 2] * 512
    labels, point_group = {}, None
    pos = start + 4
    while pos + 4 <= end:
        length, group = struct.unpack_from("bb", raw, pos)
        if not length or not group:
            break
        location = pos + 2 + abs(length)
        name = raw[pos + 2:location].decode("ascii")
        offset = struct.unpack_from("<H", raw, location)[0]
        if name == "POINT" and group < 0:
            point_group = -group
        if name == "LABELS" and group > 0:
            labels[group] = raw[location + 2:location + offset]
        pos = location + offset
    column_data = labels[point_group]
    width, count = column_data[2:4]
    if column_data[:2] != b"\xff\x02" or count != points:
        raise ValueError("unexpected C3D point labels")
    names = [column_data[i:i + width].decode("ascii").strip().rsplit(":", 1)[-1]
             for i in range(4, 4 + width * count, width)]
    if len(set(names)) != points:
        raise ValueError("duplicate C3D point labels")
    data = np.frombuffer(raw, dtype="<i2", count=frame_count * points * 4, offset=(block - 1) * 512)
    waist = data.reshape(frame_count, points, 4)[:, [names.index(marker) for marker in MARKERS]]
    return waist[..., :3].mean(axis=1) * scale, (waist[..., 3] >= 0).all(axis=1), units


def bvh_root(raw, evidence):
    if len(raw) != evidence["bytes"] or sha256(raw).hexdigest() != evidence["SHA256"]:
        raise ValueError("selected BVH provenance differs")
    stream = BytesIO(raw)
    joints = skeleton(stream)
    header = stream.readline().decode("ascii").split()
    period = stream.readline().decode("ascii").split()
    if (joints[0]["channels"][:3] != ["Xposition", "Yposition", "Zposition"]
            or header != ["Frames:", str(evidence["frames"])]
            or period != ["Frame", "Time:", "0.0333333"]):
        raise ValueError("unexpected 30Hz BVH root header")
    result = np.asarray([tuple(map(float, row.split()[:3])) for row in stream if row.strip()])
    if result.shape != (evidence["frames"], 3) or not np.isfinite(result).all():
        raise ValueError("unexpected BVH root translation extent")
    return result


def fit(source, target, complete):
    variants = [(axes, signs, basis) for axes in permutations(range(3))
                for signs in product((-1, 1), repeat=3)
                if np.linalg.det(basis := np.eye(3)[:, axes] * signs) > 0]
    rotation = np.stack([source @ basis for _, _, basis in variants])
    samples = 4 * np.arange(len(source))
    first_half = np.arange(len(source)) < len(source) // 2
    attempts, best = 0, None
    for lag in range(len(target) - samples[-1]):
        valid = complete[lag + samples]
        calibration, heldout = valid & first_half, valid & ~first_half
        counts = tuple(int(mask.sum()) for mask in (calibration, heldout))
        if counts[0] < 0.85 * first_half.sum() or counts[1] < 0.85 * (~first_half).sum():
            continue
        matched = target[lag + samples]
        center = rotation[:, calibration].mean(axis=1)
        y_center = matched[calibration].mean(axis=0)
        centered = rotation[:, calibration] - center[:, None]
        scale = (centered * (matched[calibration] - y_center)).sum((1, 2)) / (centered ** 2).sum((1, 2))
        squared = ((matched - rotation * scale[:, None, None]
                    - (y_center - scale[:, None] * center)[:, None]) ** 2).sum(axis=-1)
        eligible = np.flatnonzero(scale > 0)
        attempts += len(eligible)
        for i in eligible:
            training_rms = float(np.sqrt(squared[i, calibration].mean()))
            if best is None or training_rms < best["fit_RMS_mm"]:
                axes, signs, _ = variants[i]
                best = {"fit_RMS_mm": training_rms, "holdout_RMS_mm": float(np.sqrt(squared[i, heldout].mean())),
                        "axes": axes, "signs": signs, "start_offset_C3D_frames": lag,
                        "C3D_mm_per_BVH_unit": float(scale[i]), "valid_frames_by_half": counts}
    if best is None:
        raise ValueError("no comparable C3D/BVH sample support")
    return {**best, "positive_scale_axis_lag_hypotheses": attempts}


def audit(directory):
    selection = (directory / "locomotion-selected-85.private.json").read_bytes()
    if sha256(selection).hexdigest() != SELECTION_SHA:
        raise ValueError("frozen cohort selection differs")
    evidence = {row["member"]: row for row in json.loads(selection)["records"]}
    result = {}
    for name, (size, pin, member) in COHORTS.items():
        c3d_name, bvh_name = f"{name}Walking_c3d.zip", f"{name}_bvh.zip"
        _, bvh_size, bvh_pin, split = ARCHIVES[bvh_name]
        cohort = {}
        with (
            read_archive(directory / "cohort-calibration-walking-c3d" / c3d_name, size, pin) as capture,
            read_archive(directory / bvh_name, bvh_size, bvh_pin) as bvh,
        ):
            for motion_id, angle in ((11, 135), (10, 45)):
                filename = f"{name}_B{motion_id}_WalkTurnLeft{angle}.bvh"
                raw = capture.read(member[motion_id])
                target, valid, units = c3d_waist(raw)
                source = bvh_root(bvh.read(filename), evidence[filename])
                cohort[str(motion_id)] = {**fit(source, target, valid),
                                          "C3D_name": member[motion_id], "BVH_name": filename,
                                          "C3D_member_SHA256": sha256(raw).hexdigest(), "C3D_units": units,
                                          "C3D_frames": len(target), "BVH_frames": len(source)}
        result[name] = {"C3D_archive_sha256": pin, "split": split, "pairs": cohort}
    return {"results": result, "source_archive_scope": "Six corresponding locomotion takes only",
            "target_201_rows": 0, "training_READY": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = (json.dumps(audit(args.candidate_directory), sort_keys=True, indent=2) + "\n").encode()
    with args.output.open("xb") as stream:
        stream.write(data)
    print(sha256(data).hexdigest())


if __name__ == "__main__":
    main()
