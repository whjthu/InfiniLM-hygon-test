"""Independently read the acquired Freiburg 1 room BAG frames and prepare six RGB crops."""

import argparse
import bz2
from decimal import Decimal
from io import BytesIO
import json
from pathlib import Path
from struct import unpack

import numpy as np
from PIL import Image

from prepare_vggt_tum_room_bag_window02_rgb224 import (
    CALIBRATION, FR1, PIXELS, POSE, PROCESSED_SIZE, ROOT, json_bytes, module, pin, read,
)


SEQUENCE = "freiburg1_room"
INPUT = ROOT / "local-only/specialized48-tum-fr1-room-bag-windows03to05-v1"
GT = ROOT / "local-only/specialized48-tum-fr1-room-pose-candidate-v1/freiburg1_room-groundtruth.txt"
BOM_SHA = "e0fc602003fe54de4cf90b9abc521120b905f97bee061563479fef06cc22bcfd"
EVIDENCE_SHA = "5f068f0c72fce4645d8b9d7830dd0082475405163ddbccdce2a9527adf40b76b"
GT_SHA = "66b6770d887a5a3b69c94c3b186c3f76fe092ca25399772177f41aed984bd1b0"
WINDOWS = (3, 4, 5)
MAX_PLAIN = 4 * 1024 * 1024


def block(source):
    size, = unpack("<I", source.read(4))
    value = source.read(size)
    if not size or len(value) != size:
        raise ValueError("invalid length-prefixed ROS record")
    return value


def fields(source):
    source = BytesIO(source)
    result = {}
    while source.tell() < len(source.getbuffer()):
        key, value = block(source).split(b"=", 1)
        if key in result:
            raise ValueError("duplicate ROS field")
        result[key] = value
    return result


def timestamp(value):
    seconds, nanos = unpack("<II", value)
    if nanos >= 1_000_000_000:
        raise ValueError("invalid ROS timestamp")
    return seconds * 1_000_000_000 + nanos


def replay_rgb(raw, frame):
    container = BytesIO(raw)
    chunk = fields(block(container))
    plain_size, = unpack("<I", chunk[b"size"])
    start = container.tell() + 4
    compressed = block(container)
    region_start, region_end = frame["original_BAG_region"]
    if (chunk[b"op"] != b"\x05" or chunk[b"compression"] != b"bz2"
            or not 0 < plain_size <= MAX_PLAIN or len(raw) != region_end - region_start
            or [region_start + start, region_start + container.tell()] != frame["BAG_payload_range"]
            or pin(compressed)["SHA256"] != frame["BAG_payload_SHA256"]):
        raise ValueError("BAG chunk pin differs")
    index_bytes = container.read()
    if pin(index_bytes)["SHA256"] != frame["chunk_index_SHA256"]:
        raise ValueError("BAG RGB connection index differs")
    indexes = BytesIO(index_bytes)
    connections = {}
    while indexes.tell() < len(index_bytes):
        index_header = fields(block(indexes))
        connection, = unpack("<I", index_header[b"conn"])
        count, = unpack("<I", index_header[b"count"])
        payload = block(indexes)
        if (index_header[b"op"] != b"\x04" or unpack("<I", index_header[b"ver"]) != (1,)
                or connection in connections or len(payload) != count * 12):
            raise ValueError("invalid BAG connection index")
        connections[connection] = payload
    index = connections.get(4, b"")
    if len(index) != 12:
        raise ValueError("expected one RGB index entry")
    indexed_time = timestamp(index[:8])
    if abs(indexed_time - frame["official_png_index_time_ns"]) > 1000:
        raise ValueError("RGB index filename time differs")
    offset, = unpack("<I", index[8:])
    decoder = bz2.BZ2Decompressor()
    plain = decoder.decompress(compressed, plain_size + 1)
    if not decoder.eof or decoder.unused_data or len(plain) != plain_size or not 0 <= offset < plain_size:
        raise ValueError("BAG chunk decode differs")
    source = BytesIO(plain)
    source.seek(offset)
    header = fields(block(source))
    data = BytesIO(block(source))
    data.read(4)
    capture_time = timestamp(data.read(8))
    frame_id = block(data).decode("ascii")
    shape = unpack("<II", data.read(8))
    encoding = block(data)
    endian, stride = unpack("<BI", data.read(5))
    pixels = block(data)
    if (header[b"op"] != b"\x02" or unpack("<I", header[b"conn"]) != (4,)
            or timestamp(header[b"time"]) != indexed_time or indexed_time != capture_time
            or capture_time != frame["BAG_message_time_ns"] or offset != frame["BAG_uncompressed_message_offset"]
            or frame_id != frame["ROS_frame_id"] or shape != (480, 640) or encoding != b"rgb8"
            or endian or stride != 1920 or len(pixels) != 921600 or data.read(1)):
        raise ValueError("decoded ROS RGB8 record differs")
    return pixels


def prepare():
    manifest = json.loads(read(INPUT / "windows03to05-private-bom.json", 16384, BOM_SHA))
    evidence = json.loads(read(INPUT / "windows03to05-public-evidence.json", 8192, EVIDENCE_SHA))
    truth = read(GT, 2 * 1024 * 1024, GT_SHA).decode("ascii").splitlines()
    rows = [line.split() for line in truth if line.strip() and not line.lstrip().startswith("#")]
    truth_by_stamp = {int(Decimal(row[0]) * 1_000_000_000): row for row in rows}
    if (len(truth_by_stamp) != len(rows) or manifest["sequence"] != SEQUENCE
            or manifest["official_groundtruth_SHA256"] != GT_SHA or evidence["private_bom_SHA256"] != BOM_SHA
            or tuple(row["window"] for row in manifest["new_windows"]) != WINDOWS
            or manifest["accepted_candidate_windows_including_window02"] != [2, *WINDOWS]):
        raise ValueError("frozen windows or pose source differ")
    camera = module("room_rgb_rectifier", CALIBRATION)
    transform = module("room_pose_math", POSE)
    if camera.CALIBRATION["freiburg1_plant"] != FR1:
        raise ValueError("Freiburg 1 calibration differs")
    camera.CALIBRATION = {SEQUENCE: FR1}
    crops, targets, summaries = [], [], []
    raw_digests = set()
    for window, public in zip(manifest["new_windows"], evidence["new_candidates"], strict=True):
        number = window["window"]
        if window["state"] != "BOUND" or public["window"] != number or len(window["frames"]) != 2:
            raise ValueError("expected each frozen pair")
        poses, times, summaries_for_pair = [], [], []
        for index, frame in enumerate(window["frames"]):
            label = f"window-{number:02d}-frame-{index}"
            original = f"{label}-640x480-private.rgb8"
            region = f"{label}-BAG-region-private.bin"
            if frame["state"] != "BOUND" or frame["private_RGB8"] != original or frame["private_BAG_region"] != region:
                raise ValueError("unexpected saved frame identity")
            raw = read(INPUT / region, 1024 * 1024, frame["BAG_region_SHA256"])
            rgb = read(INPUT / original, 921600, frame["RGB8_SHA256"])
            if (rgb != replay_rgb(raw, frame) or frame["RGB8_SHA256"] in raw_digests
                    or frame["RGB8_SHA256"] != public["RGB8_SHA256"][index]):
                raise ValueError("independent RGB8 readback or uniqueness failed")
            raw_digests.add(frame["RGB8_SHA256"])
            stamp = frame["original_pose"]["TUM_pose_timestamp_ns"]
            row = truth_by_stamp[stamp]
            expected = frame["original_pose"]
            error = abs(stamp - frame["BAG_message_time_ns"])
            if (len(row) != 8 or row[1:] != expected["tx_ty_tz_qx_qy_qz_qw_text"]
                    or pin((" ".join(row) + "\n").encode("ascii"))["SHA256"] != expected["canonical_row_SHA256"]
                    or error != public["abs_BAG_vs_original_pose_error_ns"][index] or error > 20_000_000):
                raise ValueError("publisher trajectory or capture association differs")
            pose = np.asarray(row[1:], dtype=np.float64)
            if not np.isfinite(pose).all() or abs(np.linalg.norm(pose[3:]) - 1) > 0.002:
                raise ValueError("invalid ground truth pose")
            stream = BytesIO()
            Image.frombytes("RGB", PIXELS, rgb).save(stream, format="PNG")
            image, coverage = camera.rectify_rgb(stream.getvalue(), SEQUENCE)
            if image.dtype != np.uint8 or image.shape != PROCESSED_SIZE or not 0 < coverage <= 1:
                raise ValueError("unexpected RGB crop")
            crops.append(image)
            poses.append(transform.camera_to_world(pose))
            times.append(frame["BAG_message_time_ns"])
            summaries_for_pair.append({"original_RGB8": pin(rgb), "crop_RGB8": pin(image.tobytes()),
                                       "valid_pixel_fraction": coverage, "absolute_pose_time_error_ns": error})
        gap = times[1] - times[0]
        first_to_second = np.linalg.solve(poses[1], poses[0])
        inverse = np.linalg.solve(poses[0], poses[1])
        rot = first_to_second[:3, :3]
        cycle = float(np.max(np.abs(inverse @ first_to_second - np.eye(4))))
        if (gap <= 0 or gap != public["frame_interval_ns"] or not np.isfinite(first_to_second).all()
                or not np.allclose(rot.T @ rot, np.eye(3), atol=1e-10, rtol=0)
                or abs(np.linalg.det(rot) - 1) > 1e-10 or cycle > 1e-10
                or np.array_equal(crops[-2], crops[-1])):
            raise ValueError("pair interval, camera convention or crops differ")
        targets.append({"window": number, "camera_from_first": [np.eye(4)[:3].tolist(), first_to_second[:3].tolist()],
                        "output_K_per_frame": camera.calibration(SEQUENCE)["output_K"]})
        summaries.append({"window": number, "frames": summaries_for_pair, "interval_ns": gap,
                          "relative_baseline_m": float(np.linalg.norm(first_to_second[:3, 3])), "cycle_error": cycle})
    return np.stack(crops).reshape(3, 2, *PROCESSED_SIZE), targets, summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    crops, targets, windows = prepare()
    args.output.mkdir()
    image_path = args.output / "frames224-private.npy"
    pose_path = args.output / "camera-targets-private.json"
    with image_path.open("xb") as stream:
        np.save(stream, crops, allow_pickle=False)
    with pose_path.open("xb") as stream:
        stream.write(json_bytes(targets))
    result = {"sequence": SEQUENCE, "portfolio_id": 48, "window_denominator": 12,
              "BAG_candidate_pairs_including_window02": 4, "independently_replayed_pairs_this_run": 3,
              "actual_local_RGB8_replayed_and_processed": 6, "crop_shape": list(crops.shape), "windows": windows,
              "input_pins": {"acquisition_evidence": EVIDENCE_SHA, "private_manifest": BOM_SHA,
                             "official_groundtruth": GT_SHA, "rectifier": CALIBRATION[1], "camera_pose": POSE[1]},
              "private_output_pins": {path.name: pin(path.read_bytes()) for path in (image_path, pose_path)},
              "depth_used": False, "original_TGZ_PNG_comparisons_this_run": 0, "network_reads": 0,
              "gpu": 0, "optimizer_updates": 0, "target_native_rows": 0, "training_READY": False}
    with (args.output / "public-summary.json").open("xb") as stream:
        stream.write(json_bytes(result))
    print(json.dumps({key: result[key] for key in ("sequence", "actual_local_RGB8_replayed_and_processed", "crop_shape")}))


if __name__ == "__main__":
    main()
