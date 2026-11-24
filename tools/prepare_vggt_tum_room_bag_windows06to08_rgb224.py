"""Prepare the next three original Freiburg 1 room RGB-only pose pairs."""

import argparse
from decimal import Decimal
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image

from prepare_vggt_tum_room_bag_windows03to05_rgb224 import (
    CALIBRATION, FR1, GT, GT_SHA, PIXELS, POSE, PROCESSED_SIZE, ROOT, SEQUENCE,
    json_bytes, module, pin, read, replay_rgb,
)


INPUT = ROOT / "local-only/specialized48-tum-fr1-room-bag-windows06to08-v1"
BOM = "a2975aab55aa433e9754f4d8ad698c113fb7d8267b5b00b671e883a1dbd1398a"
ACQUISITION = "a6631f33fcd47d63229c461c1d9ef3b05bb151bda3c7ebf5cb1156d72e677ada"
PAIRS = (6, 7, 8)


def prepare():
    bom = json.loads(read(INPUT / "windows06to08-private-bom.json", 16384, BOM))
    acquisition = json.loads(read(INPUT / "windows06to08-public-evidence.json", 8192, ACQUISITION))
    originals = read(GT, 2 * 1024 * 1024, GT_SHA).decode("ascii").splitlines()
    rows = [line.split() for line in originals if line.strip() and not line.lstrip().startswith("#")]
    by_stamp = {int(Decimal(row[0]) * 1_000_000_000): row for row in rows}
    windows = bom["new_windows"]
    if (bom["sequence"] != SEQUENCE or bom["official_groundtruth_SHA256"] != GT_SHA
            or acquisition["private_bom_SHA256"] != BOM or len(by_stamp) != len(rows)
            or tuple(row["window"] for row in windows) != PAIRS
            or bom["accepted_candidate_windows_including_prior"] != list(range(2, 9))):
        raise ValueError("frozen candidate set or ground truth changed")
    camera = module("tum_room6_8_rgb", CALIBRATION)
    transform = module("tum_room6_8_pose", POSE)
    if camera.CALIBRATION["freiburg1_plant"] != FR1:
        raise ValueError("Freiburg 1 calibration differs")
    camera.CALIBRATION = {SEQUENCE: FR1}
    output, targets, reports = [], [], []
    for candidate, source in zip(windows, acquisition["new_candidates"], strict=True):
        number = candidate["window"]
        if (candidate["state"] != "BOUND" or source["window"] != number
                or source["state"] != "BOUND" or len(candidate["frames"]) != 2):
            raise ValueError("incomplete frozen pair")
        poses, crops, times, frame_pins = [], [], [], []
        for index, frame in enumerate(candidate["frames"]):
            label = f"window-{number:02d}-frame-{index}"
            region = f"{label}-BAG-region-private.bin"
            name = f"{label}-640x480-private.rgb8"
            if frame["private_RGB8"] != name or frame["private_BAG_region"] != region or frame["state"] != "BOUND":
                raise ValueError("original frame identity differs")
            pixels = read(INPUT / name, 921600, frame["RGB8_SHA256"])
            raw = read(INPUT / region, 1024 * 1024, frame["BAG_region_SHA256"])
            if replay_rgb(raw, frame) != pixels or source["RGB8_SHA256"][index] != frame["RGB8_SHA256"]:
                raise ValueError("original BAG or RGB8 pixels differ")
            sample = frame["original_pose"]
            row = by_stamp[sample["TUM_pose_timestamp_ns"]]
            error = abs(sample["TUM_pose_timestamp_ns"] - frame["BAG_message_time_ns"])
            if (len(row) != 8 or row[1:] != sample["tx_ty_tz_qx_qy_qz_qw_text"]
                    or pin((" ".join(row) + "\n").encode("ascii"))["SHA256"] != sample["canonical_row_SHA256"]
                    or error != source["abs_BAG_vs_original_pose_error_ns"][index] or error > 20_000_000):
                raise ValueError("original pose association differs")
            pose = np.asarray(row[1:], dtype=np.float64)
            if not np.isfinite(pose).all() or abs(np.linalg.norm(pose[3:]) - 1) > 0.002:
                raise ValueError("invalid source quaternion")
            buffer = BytesIO()
            Image.frombytes("RGB", PIXELS, pixels).save(buffer, format="PNG")
            crop, coverage = camera.rectify_rgb(buffer.getvalue(), SEQUENCE)
            if crop.shape != PROCESSED_SIZE or crop.dtype != np.uint8 or not 0 < coverage <= 1:
                raise ValueError("invalid color crop")
            crops.append(crop)
            poses.append(transform.camera_to_world(pose))
            times.append(frame["BAG_message_time_ns"])
            frame_pins.append({"rgb8": pin(pixels), "crop": pin(crop.tobytes()),
                               "coverage": coverage, "time_error_ns": error})
        interval = times[1] - times[0]
        relative = np.linalg.solve(poses[1], poses[0])
        reverse = np.linalg.solve(poses[0], poses[1])
        cycle = float(np.max(np.abs(reverse @ relative - np.eye(4))))
        rotation = relative[:3, :3]
        if (interval <= 0 or interval != source["frame_interval_ns"] or np.array_equal(*crops)
                or not np.isfinite(relative).all() or cycle > 1e-10
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10, rtol=0)
                or abs(np.linalg.det(rotation) - 1) > 1e-10):
            raise ValueError("pair interval or relative camera convention differs")
        output.append(np.stack(crops))
        targets.append({"window": number, "camera_from_first": [np.eye(4)[:3].tolist(), relative[:3].tolist()],
                        "output_K_per_frame": camera.calibration(SEQUENCE)["output_K"],
                        "pose_valid": [True, True]})
        reports.append({"window": number, "frames": frame_pins, "interval_ns": interval,
                        "baseline_m": float(np.linalg.norm(relative[:3, 3])), "pose_cycle_error": cycle})
    if len({f["crop"]["SHA256"] for r in reports for f in r["frames"]}) != 6:
        raise ValueError("repeated processed image")
    return np.stack(output), targets, reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    images, poses, reports = prepare()
    args.output.mkdir()
    rgb_file = args.output / "frames224-private.npy"
    target_file = args.output / "poses-private.json"
    with rgb_file.open("xb") as stream:
        np.save(stream, images, allow_pickle=False)
    with target_file.open("xb") as stream:
        stream.write(json_bytes(poses))
    result = {"sequence": SEQUENCE, "portfolio_id": 48, "window_denominator": 12, "actual_new_pairs": 3,
              "shape": list(images.shape), "windows": reports, "source": pin(Path(__file__).read_bytes()),
              "input_pins": {"acquisition_SHA256": ACQUISITION, "original_manifest_SHA256": BOM,
                             "trajectory_SHA256": GT_SHA, "rgb_calibration_SHA256": CALIBRATION[1],
                             "pose_transform_SHA256": POSE[1]},
              "private_output_pins": {f.name: pin(f.read_bytes()) for f in (rgb_file, target_file)},
              "native_target_batches": 0, "optimizer_updates": 0, "network_reads": 0, "training_READY": False}
    summary = args.output / "public-summary.json"
    with summary.open("xb") as stream:
        stream.write(json_bytes(result))
    print(json.dumps({"shape": result["shape"], "sha256": pin(summary.read_bytes())["SHA256"]}))


if __name__ == "__main__":
    main()
