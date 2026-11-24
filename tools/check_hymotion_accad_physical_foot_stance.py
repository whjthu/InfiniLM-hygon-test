"""Measure target feet during independent C3D-based contact intervals."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import struct

import numpy as np

from audit_hymotion_accad_cohort_scale import COHORTS, read_archive
from audit_hymotion_accad_quickwalk_scale import public_c3d_units
from build_hymotion_accad_wooden_pilot import COHORT_CALIBRATION_SHA


FPS = 30
WINDOW = 128
TARGET_TOES = [10, 11]
SOURCE_TOES = ["LTOE", "RTOE"]
NEUTRAL_SUMMARY_SHA = "c69bd77305d556536d92d80bf61a87c37a46ed1cd2807009fc80575c8e3d3b0d"


def source_toes(raw, sample_start):
    if public_c3d_units(raw) != "mm":
        raise ValueError("only exact millimetre point data are expected")
    points, analog, first, last = struct.unpack_from("<HHHH", raw, 2)
    block = struct.unpack_from("<H", raw, 16)[0]
    scale = struct.unpack("<f", raw[14:16] + raw[12:14])[0] / 4
    rate = struct.unpack("<f", raw[22:24] + raw[20:22])[0] / 4
    section = (raw[0] - 1) * 512
    end = section + raw[section + 2] * 512
    if (points, analog, rate) != (41, 0, 120) or not 0 < scale < 1:
        raise ValueError("unexpected C3D marker encoding")
    by_group, point_group = {}, None
    section += 4
    while section + 4 <= end:
        length, group = struct.unpack_from("bb", raw, section)
        if length == 0 or group == 0:
            break
        pointer = section + 2 + abs(length)
        tag = raw[section + 2:pointer].decode("ascii")
        jump = struct.unpack_from("<H", raw, pointer)[0]
        if jump < 3 or pointer + jump > end:
            raise ValueError("invalid marker parameter extent")
        if tag == "POINT" and group < 0:
            point_group = -group
        if tag == "LABELS" and group > 0:
            by_group[group] = raw[pointer + 2:pointer + jump]
        section = pointer + jump
    data = by_group[point_group]
    width, count = data[2:4]
    if data[:2] != b"\xff\x02" or count != points:
        raise ValueError("unexpected C3D marker label table")
    names = [data[i:i + width].decode("ascii").strip().rsplit(":", 1)[-1]
             for i in range(4, 4 + width * count, width)]
    if len(set(names)) != points:
        raise ValueError("duplicate C3D marker labels")
    shape = (last - first + 1, points, 4)
    if type(sample_start) is not int:
        raise ValueError("source alignment origin is not an integer index")
    indexes = sample_start + 4 * np.arange(WINDOW)
    if indexes[0] < 0 or indexes[-1] >= shape[0]:
        raise ValueError("original aligned source frames do not cover the candidate")
    coords = np.frombuffer(raw, dtype="<i2", count=shape[0] * points * 4, offset=(block - 1) * 512).reshape(shape)
    selected = coords[indexes][:, [names.index(toe) for toe in SOURCE_TOES]]
    return selected[..., :3].astype("f8") * scale / 1000, selected[..., 3] >= 0


def stance_comparison(coords, valid, features):
    if coords.shape != (WINDOW, 2, 3) or valid.shape != (WINDOW, 2) or features.shape != (WINDOW, 201):
        raise ValueError("unexpected paired frame geometry")
    if not np.isfinite(coords).all() or not np.isfinite(features).all() or np.any(valid.sum(axis=0) == 0):
        raise ValueError("invalid physical or target coordinates")
    feet = features[:, 135:].reshape(WINDOW, 22, 3)[:, TARGET_TOES] + features[:, None, :3]
    source_speed = np.linalg.norm(np.diff(coords[:, :, :2], axis=0), axis=-1) * FPS
    target_speed = np.linalg.norm(np.diff(feet[:, :, [0, 2]], axis=0), axis=-1) * FPS
    low = np.nanpercentile(np.where(valid, coords[:, :, 2], np.nan), 5, axis=0)
    stance = valid[:-1] & valid[1:] & (source_speed < 0.18)
    stance &= np.maximum(coords[:-1, :, 2], coords[1:, :, 2]) < low + 0.03
    if not np.any(stance, axis=0).all():
        raise ValueError("missing source stance intervals")
    return {
        "valid_source_frames_by_toe": valid.sum(axis=0).tolist(),
        "source_stance_intervals_by_toe": stance.sum(axis=0).tolist(),
        "source_stance_horizontal_speed_mps_median": float(np.median(source_speed[stance])),
        "target_stance_horizontal_speed_mps_median": float(np.median(target_speed[stance])),
        "target_stance_horizontal_speed_mps_p95": float(np.quantile(target_speed[stance], 0.95)),
    }


def audit(directory, candidate):
    original = (directory / "cohort-calibration-walking-c3d/calibration-B10-B11-original-first.json").read_bytes()
    metadata = (candidate / "geometry-candidate-summary.json").read_bytes()
    if sha256(original).hexdigest() != COHORT_CALIBRATION_SHA or sha256(metadata).hexdigest() != NEUTRAL_SUMMARY_SHA:
        raise ValueError("frozen calibration or candidate metadata differs")
    clip_reports = json.loads(metadata)
    candidate_path = candidate / "geometry-candidate.private.npz"
    if sha256(candidate_path.read_bytes()).hexdigest() != clip_reports["candidate_tensor_SHA256"]:
        raise ValueError("frozen geometry candidate differs")
    pairs = json.loads(original)["results"]
    result = {}
    with np.load(candidate_path, allow_pickle=False) as slots:
        for group, (size, pin, members) in COHORTS.items():
            witness = pairs[group]["pairs"]["10"]
            with read_archive(directory / "cohort-calibration-walking-c3d" / f"{group}Walking_c3d.zip", size, pin) as z:
                raw = z.read(members[10])
            if sha256(raw).hexdigest() != witness["C3D_member_SHA256"]:
                raise ValueError("frozen physical marker file differs")
            coords, valid = source_toes(raw, witness["start_offset_C3D_frames"])
            result[group] = stance_comparison(coords, valid, slots[f"{group}_raw"])
    return {
        "source_marker_provenance_SHA256": COHORT_CALIBRATION_SHA,
        "matched_target_geometry_SHA256": clip_reports["candidate_tensor_SHA256"],
        "source_toe_stance_heuristic": {"speed_below_mps": 0.18, "height_m_above_per_toe_source_5th_percentile": 0.03},
        "measurements": result,
        "scope": "Diagnostic from the same physical takes and phase alignment. Toe markers and target foot joints have distinct offsets.",
        "training_READY": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = (json.dumps(audit(args.directory, args.candidate), sort_keys=True, indent=2) + "\n").encode()
    with args.output.open("xb") as stream:
        stream.write(data)
    print(sha256(data).hexdigest())


if __name__ == "__main__":
    main()
