"""Verify exact public geometry assets and exercise the native CPU WoodenMesh path."""

import argparse
from hashlib import sha256
from io import BytesIO
import importlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from verify_hymotion_publisher_feature_assets import load_geometry


GEOMETRY_SHA256 = "f3b4859f95bfce119bffb6af0f9f8eaf2c0e4d26c09b61fdfb3d2f3396332e2f"
SOURCE_FILES = {
    "hymotion/pipeline/body_model.py": "01eebf66d87f0ee10ebbd2e7b8ab1a2dd4c581661be99d0597c5286854e95443",
    "hymotion/utils/geometry.py": "15d0c3c093e6fbe7464ce54697c6d909050217d2af8af0813a21cd64a7b3c540",
}
SUBJECTS = ("Female1", "Male1", "Male2")
FRAME_COUNT = 128
KEYPOINT_MAX_ERROR_METRES = 5e-5
NORMALIZED_MAX_ERROR = 1e-3


def pinned_file(path, pin, byte_limit):
    with path.open("rb") as stream:
        raw = stream.read(byte_limit + 1)
    if len(raw) > byte_limit or sha256(raw).hexdigest() != pin:
        raise ValueError(f"frozen input differs: {path.name}")
    return raw


def verify_inputs(asset_dir, upstream, candidate_path):
    for name, pin in SOURCE_FILES.items():
        pinned_file(upstream / name, pin, 131072)
    vertices, _, _, _, nodes, parents, mean, std = load_geometry(asset_dir)
    if (len(vertices), len(nodes), len(parents)) != (24256, 52, 52):
        raise ValueError("WoodenMesh dimensions differ")
    if np.flatnonzero(std < 1e-3).tolist() != [135, 136, 137]:
        raise ValueError("unexpected normalization fallback")
    safe_std = np.where(std < 1e-3, np.float32(1), std)
    raw = pinned_file(candidate_path, GEOMETRY_SHA256, 550000)
    with np.load(BytesIO(raw), allow_pickle=False) as slots:
        if set(slots.files) != {f"{name}_{field}" for name in SUBJECTS for field in ("raw", "normalized")}:
            raise ValueError("unexpected geometry member names")
        takes = {name: (slots[f"{name}_raw"], slots[f"{name}_normalized"]) for name in SUBJECTS}
    for raw, normalized in takes.values():
        if any(a.dtype != np.dtype("float32") or a.shape != (FRAME_COUNT, 201) or not np.isfinite(a).all()
               for a in (raw, normalized)):
            raise ValueError("invalid geometry shape, type or values")
        if not np.array_equal((raw - mean) / safe_std, normalized):
            raise ValueError("frozen normalization differs from producer arithmetic")
    return takes, mean, safe_std


def run_native(upstream, asset_dir, takes, mean, safe_std, deadline):
    for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        if os.environ.get(name) != "-1":
            raise ValueError("CPU-only device visibility is required")
    if time.monotonic() >= deadline:
        raise TimeoutError("the original local CPU budget expired")
    upstream = upstream.resolve(strict=True)
    sys.path.insert(0, str(upstream))
    try:
        import torch

        if torch.cuda.is_initialized():
            raise ValueError("GPU runtime already initialized")
        torch.set_num_threads(4)
        body = importlib.import_module("hymotion.pipeline.body_model")
        geometry = importlib.import_module("hymotion.utils.geometry")
        for module, name in ((body, "hymotion/pipeline/body_model.py"), (geometry, "hymotion/utils/geometry.py")):
            if Path(module.__file__).resolve(strict=True) != upstream / name:
                raise ValueError("native module resolves outside the pinned source")
            pinned_file(upstream / name, SOURCE_FILES[name], 131072)
        model = body.WoodenMesh(str(asset_dir.resolve(strict=True))).to(device="cpu", dtype=torch.float32).eval()
        if any(v.device.type != "cpu" for v in model.buffers()):
            raise ValueError("WoodenMesh buffers are outside CPU")
        results = {}
        with torch.inference_mode():
            for name, (raw, normalized) in takes.items():
                points, min_ground = [], float("inf")
                for start in range(0, FRAME_COUNT, 16):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("the original local CPU budget expired")
                    root = torch.from_numpy(np.ascontiguousarray(raw[start:start + 16, :3]))
                    rotation = torch.from_numpy(np.ascontiguousarray(raw[start:start + 16, 3:135])).reshape(-1, 22, 6)
                    output = model({"rot6d": rotation, "trans": root})
                    expected_frames = root.shape[0]
                    joints = output["keypoints3d"]
                    vertices = output["vertices"]
                    if joints.shape != (expected_frames, 52, 3) or vertices.shape != (expected_frames, 24256, 3):
                        raise ValueError("native joint or vertex denominator differs")
                    if not torch.isfinite(joints).all() or not torch.isfinite(vertices).all():
                        raise ValueError("nonfinite native geometry")
                    points.append(joints[:, :22].reshape(expected_frames, 66).numpy())
                    min_ground = min(min_ground, float(vertices[:, :, 1].min()))
                native = np.concatenate((raw[:, :135], np.concatenate(points, axis=0)), axis=1)
                norm = (native - mean) / safe_std
                joint_error = float(np.max(np.abs(native[:, 135:] - raw[:, 135:])))
                norm_error = float(np.max(np.abs(norm - normalized)))
                results[name] = dict(frames=FRAME_COUNT, max_keypoint_error_metres=joint_error,
                                     max_normalized_error=norm_error, lowest_native_vertex_metres=min_ground,
                                     passed=joint_error <= KEYPOINT_MAX_ERROR_METRES
                                     and norm_error <= NORMALIZED_MAX_ERROR
                                     and abs(min_ground) <= KEYPOINT_MAX_ERROR_METRES)
        if torch.cuda.is_initialized():
            raise ValueError("GPU runtime activated during native check")
        if time.monotonic() >= deadline:
            raise TimeoutError("the original local CPU budget expired")
        return torch.__version__, results
    finally:
        sys.path.remove(str(upstream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", type=Path)
    parser.add_argument("upstream", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--native-cpu", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    takes, mean, safe_std = verify_inputs(args.assets, args.upstream, args.candidate)
    version, outcomes = (None, {})
    if args.native_cpu:
        version, outcomes = run_native(args.upstream, args.assets, takes, mean, safe_std, started + 150)
    result = dict(candidate_SHA256=GEOMETRY_SHA256, source_SHA256=SOURCE_FILES, asset_count=9,
                  local_input_frames_verified=FRAME_COUNT * len(takes), feature_width=201,
                  torch_version=version, actual_native_cpu_checks=outcomes,
                  max_keypoint_error_metres=KEYPOINT_MAX_ERROR_METRES, max_normalized_error=NORMALIZED_MAX_ERROR,
                  text_encoder_calls=0, GPU_steps=0, accepted_training_rows=0, training_READY=False)
    raw = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with args.output.open("xb") as stream:
        stream.write(raw)
    print(sha256(raw).hexdigest())
    return int(bool(outcomes) and not all(outcome["passed"] for outcome in outcomes.values()))


if __name__ == "__main__":
    raise SystemExit(main())
