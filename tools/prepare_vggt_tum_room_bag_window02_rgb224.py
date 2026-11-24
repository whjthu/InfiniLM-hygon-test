"""Convert the fixed official TUM room RGB8 pair to VGGT's RGB-only 224-pixel convention."""

import argparse
from hashlib import sha256
import importlib.util
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "local-only/specialized48-tum-fr1-room-bag-window02-v1"
MANIFEST_SHA = "fffca1ab8841f137d90bd0ed56bba7be28927a7b1948389d24c9187ff2749c53"
CALIBRATION = (ROOT / "tools/bind_vggt_tum_dev_rgb.py",
               "8cac4f480f6e2229f08dc9542e2ca3a79d84744fee2f116a3487a0da62e6b1f2")
POSE = (ROOT / "code/bw1000/training-domain-pilot-v1/tum_vggt_pair.py",
        "c453a9c561c6411b8c2f25652f5c004c928db423c03ca0a45eba7d47ed9e3962")
OFFICIAL_CALIBRATION = "https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats#calibration_of_the_color_camera"
FR1 = ((517.3, 516.5, 318.6, 255.3), (0.2624, -0.9531, -0.0054, 0.0026, 1.1633))
PIXELS = (640, 480)
PROCESSED_SIZE = (224, 224, 3)


def pin(raw):
    return {"bytes": len(raw), "SHA256": sha256(raw).hexdigest()}


def read(path, maximum, expected):
    if path.is_symlink() or not path.is_file():
        raise ValueError("unexpected private RGB or source file")
    with path.open("rb") as stream:
        raw = stream.read(maximum + 1)
    if not 0 < len(raw) <= maximum or pin(raw)["SHA256"] != expected:
        raise ValueError("frozen RGB or source digest differs")
    return raw


def module(name, reference):
    path, digest = reference
    raw = read(path, 64 * 1024, digest)
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    exec(compile(raw, str(path), "exec"), vars(loaded))
    return loaded


def prepare_pair():
    m = json.loads(read(SOURCE / "window02-private.json", 8192, MANIFEST_SHA))
    sequence = "freiburg1_room"
    if m["window"] != 2 or m["sequence"] != sequence or m["bound_pair"] is not True or len(m["frames"]) != 2:
        raise ValueError("expected exactly one frozen official BAG-only training candidate")
    camera = module("qualified_fr1_rgb_rectifier", CALIBRATION)
    transforms = module("qualified_vggt_pose_math", POSE)
    if camera.CALIBRATION["freiburg1_plant"] != FR1:
        raise ValueError("qualified Freiburg 1 calibration differs from the primary source")
    camera.CALIBRATION = {sequence: FR1}
    imgs, poses, frames, times = [], [], [], []
    for index, member in enumerate(m["frames"]):
        name = f"frame-{index}-640x480-private.rgb8"
        if (member["frame_index"] != index or member["BAG_connection"] != 4
                or Path(member["private_RGB8"]).name != name or member["RGB8_bytes"] != 921600):
            raise ValueError("frozen room RGB8 member identity differs")
        rgb = read(SOURCE / name, 921600, member["RGB8_SHA256"])
        if len(rgb) != 921600:
            raise ValueError("frozen 640x480 RGB8 buffer differs")
        original_pose = member["original_pose"]
        time_ns = member["BAG_message_time_ns"]
        offset = abs(time_ns - original_pose["TUM_pose_timestamp_ns"])
        if offset != member["absolute_BAG_vs_original_pose_error_ns"] or offset >= 20_000_000:
            raise ValueError("original pair timestamp gate no longer matches")
        image = Image.frombytes("RGB", PIXELS, rgb)
        png = BytesIO()
        image.save(png, format="PNG")
        with Image.open(BytesIO(png.getvalue())) as decoded:
            if decoded.tobytes() != rgb:
                raise ValueError("in-memory serialization changed original RGB pixels")
        rectified, fraction = camera.rectify_rgb(png.getvalue(), sequence)
        if rectified.dtype != np.uint8 or rectified.shape != PROCESSED_SIZE or not 0 < fraction <= 1:
            raise ValueError("RGB-only rectifier returned the wrong image contract")
        pose = np.array(original_pose["tx_ty_tz_qx_qy_qz_qw_text"], dtype=np.float64)
        if pose.shape != (7,) or not np.isfinite(pose).all() or abs(np.linalg.norm(pose[3:]) - 1) > 0.002:
            raise ValueError("frozen pose has an invalid quaternion")
        imgs.append(rectified)
        poses.append(transforms.camera_to_world(pose))
        times.append(time_ns)
        frames.append(dict(original_RGB8=pin(rgb), valid_pixel_fraction=fraction,
                           rectified_RGB8=pin(rectified.tobytes()), absolute_time_difference_ns=offset))

    if times[1] - times[0] != m["exact_BAG_frame_interval_ns"] or times[0] >= times[1]:
        raise ValueError("frozen acquisition order or frame interval changed")
    forward = np.linalg.solve(poses[1], poses[0])
    reverse = np.linalg.solve(poses[0], poses[1])
    rot = forward[:3, :3]
    residuals = dict(orthogonality_error=float(np.max(np.abs(rot.T @ rot - np.eye(3)))),
                     reverse_cycle_error=float(np.max(np.abs(reverse @ forward - np.eye(4)))))
    if not np.isfinite(forward).all() or max(residuals.values()) > 1e-10 or abs(np.linalg.det(rot) - 1) > 1e-10:
        raise ValueError("invalid rigid transform in the first-camera reference frame")
    low = camera.calibration(sequence)
    frames_np = np.stack(imgs)
    if np.array_equal(frames_np[0], frames_np[1]):
        raise ValueError("rectification produced a duplicate frame")
    pose_target = dict(extrinsics=[np.eye(4)[:3].tolist(), forward[:3].tolist()], convention="camera_from_first",
                       intrinsics_per_frame=low["output_K"], depth_used=False, sequence=sequence, window=2)
    public = dict(sequence=sequence, window=2, original_manifest_SHA256=MANIFEST_SHA, actual_CPU_processed_RGB=2,
                  output_shape=list(frames_np.shape), depth_used=False, training_READY=False,
                  official_groundtruth_SHA256=m["official_groundtruth_SHA256"], original_pixels=list(PIXELS),
                  official_calibration_source=OFFICIAL_CALIBRATION, frame_spacing_ns=times[1] - times[0],
                  matched_original_pose_rows=2, source_role="official ROS BAG RGB8",
                  input_PNG_source_claimed=False, frames=frames, transform_residuals=residuals,
                  relative_baseline_m=float(np.linalg.norm(forward[:3, 3])),
                  source_pins={"rgb": CALIBRATION[1], "pose": POSE[1]}, target_native_rows=0, optimizer_updates=0, gpu=0)
    return frames_np, pose_target, public


def json_bytes(obj):
    return (json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    frames, pose, evidence = prepare_pair()
    args.output.mkdir()
    with (args.output / "frames224-rgb8-private.npy").open("xb") as file:
        np.save(file, frames, allow_pickle=False)
    with (args.output / "poses-first-camera-private.json").open("xb") as file:
        file.write(json_bytes(pose))
    evidence["private_output_pins"] = {
        name: pin((args.output / name).read_bytes())
        for name in ("frames224-rgb8-private.npy", "poses-first-camera-private.json")
    }
    with (args.output / "public-summary.json").open("xb") as file:
        file.write(json_bytes(evidence))
    print(json.dumps(evidence, sort_keys=True))


if __name__ == "__main__":
    main()
