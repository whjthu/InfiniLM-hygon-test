"""Compare the base and two T12 heads on the fixed 24-window RGB development cohort."""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

from bind_vggt_tum_dev_rgb import CALIBRATION, calibration, read_bounded, sha
from bind_vggt_tum_cross_camera_heads import ARMS, HEAD, is_sha
from plan_vggt_tum_public_dev_trajectories import MIN_TRANSLATION


WINDOWS = {(seq, window) for seq in CALIBRATION for window in range(1, 13)}
TORCH_VERSION = "2.10.0+das.opt1.dtk2604.2606161421.g91a580"
HIP_VERSION = "6.3.26113"
BACKEND_SHA = "49bfe39b7aa6a14df9fdf1dda73e6db00836a81d2806012b312c16b2d3fbed6c"
PROVIDERS = {
    "models/vggt.py": "dbc92d6214882161d9042b1bc60cc932a9fd6bb609fc913d14906328f7833607",
    "models/aggregator.py": "c994d7231eda2380f208355a64bce3cd7c4b97ef06393355e632415bfc1513e5",
    "heads/camera_head.py": "d177f3d64d53210209b7bbb8496faaaec91f3fc51d04a4ee8d9d6b70f053e132",
    "utils/pose_enc.py": "845fc3a46cb851bebf0720b8fc2b9f34334ae8e11f1982376ba59f90ff18f22b",
    "utils/rotation.py": "ad3a904e243c6654f0aa5f1d0206ace5f2d61011bc1bf8f492b40e0b76cb2070",
}
SCORER_SHA = "6454c9a5f10459e5f2eb8b268918d1303d3f04c2038a12cca22c20ac87b58e22"
CANONICAL = Path(__file__).resolve().parents[1] / "code/bw1000/training-domain-pilot-v1/score_vggt_tum_pair.py"


def pinned_json(path, wanted):
    raw = read_bounded(path, 65536)
    if sha(raw) != wanted:
        raise ValueError("frozen manifest or head evidence changed")
    return json.loads(raw)


def canonical_scorer():
    raw = read_bounded(CANONICAL, 16000)
    if sha(raw) != SCORER_SHA:
        raise ValueError("expected the frozen directed-pose scorer")
    scorer = {"__name__": "vggt_tum_canonical_score"}
    exec(compile(raw, str(CANONICAL), "exec"), scorer)
    return scorer


def load_input(root, row, relative_pose):
    sequence, window = row["sequence"], row["window"]
    name = f"{sequence}-window-{window:02d}-rgb224.npy"
    if row["rgb_file"] != name or row["intrinsics"] != calibration(sequence)["output_K"]:
        raise ValueError("RGB224 artifact or camera identity differs")
    with io.BytesIO(read_bounded(root / name, 310000)) as stream:
        raw = stream.getvalue()
        if (len(raw), sha(raw)) != (row["rgb_file_bytes"], row["rgb_file_SHA256"]):
            raise ValueError("RGB224 bytes differ")
        if np.lib.format.read_magic(stream) != (1, 0):
            raise ValueError("unexpected RGB224 NumPy format")
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        if shape != (2, 224, 224, 3) or fortran or dtype != np.uint8 or len(raw) - stream.tell() != np.prod(shape):
            raise ValueError("unexpected RGB224 layout")
        images = np.frombuffer(raw, dtype=dtype, offset=stream.tell()).reshape(shape)
    truth = np.asarray(row["world_to_camera"], dtype=np.float64)
    _, translation = relative_pose(truth)
    if np.linalg.norm(translation) < MIN_TRANSLATION:
        raise ValueError("mocap displacement differs from the frozen floor")
    return images, truth


def read_cases(manifest, wanted, relative_pose):
    rows = pinned_json(manifest, wanted)["windows"]
    keys = [(row["sequence"], row["window"]) for row in rows]
    if len(keys) != len(WINDOWS) or set(keys) != WINDOWS:
        raise ValueError("expected all 24 Freiburg windows exactly once")
    cases, evaluated = [], []
    for row in sorted(rows, key=lambda item: (item["sequence"], item["window"])):
        result = dict(sequence=row["sequence"], window=row["window"], input_state=row["state"])
        if row["state"] == "RGB224_AND_RELATIVE_POSE_BOUND":
            images, truth = load_input(manifest.parent, row, relative_pose)
            cases.append((result, images, truth))
            result["input_SHA256"] = row["rgb_file_SHA256"]
        elif row["state"] != "RGB224_PRECONDITION_FAILED":
            raise ValueError("unrecognized RGB input state")
        result.update({arm: dict(pair_error_deg=None, state="MISSING_INPUT" if "input_SHA256" not in result else "NOT_RUN")
                       for arm in ("base", *ARMS)})
        evaluated.append(result)
    if not cases:
        raise ValueError("no verified RGB inputs")
    return cases, evaluated


def read_heads(path, wanted):
    pins = pinned_json(path, wanted)
    if set(pins) != set(ARMS):
        raise ValueError("expected the two independently verified training heads")
    for arm, proof in pins.items():
        required = (proof["gpu_step"], proof["request"], proof["clean_post"], proof["training_receipt_SHA256"])
        if required != ARMS[arm] or proof.get("global_step") != 25:
            raise ValueError("head evidence differs from the original training job and step")
        if (any(not is_sha(proof[key]) for key in ("file_SHA256", "module_SHA256", "original_outcome_SHA256",
                                                  "original_combined_native_SHA256", "original_continuous_result_SHA256"))
                or type(proof["head_bytes_stat"]) is not int or proof["head_bytes_stat"] <= 0
                or proof["actual_checkpoint_bytes_read"] != 0):
            raise ValueError("head evidence requires original result lineage and a declared artifact identity")
    return pins


def summarize(rows, scorer):
    def group(group_rows):
        scores = {arm: scorer([dict(sequence=r["sequence"], frame_ids=[r["window"]],
                                    baseline=r["base"], updated=r[arm]) for r in group_rows])["updated"]
                  for arm in ("base", *ARMS)}
        scores["AUC_delta_vs_base"] = {
            arm: {metric: scores[arm][metric] - scores["base"][metric]
                  for metric in scores["base"] if metric.startswith("development_AUC_at_")} for arm in ARMS}
        return scores

    return dict(all_windows=group(rows), by_sequence={
        seq: group([row for row in rows if row["sequence"] == seq]) for seq in CALIBRATION})


def measure(cases, arm, model, torch, decode, score, autocast, synchronize, record):
    for result, images, truth in cases:
        gpu = torch.from_numpy(images.astype(np.float32)).permute(0, 3, 1, 2).div(255).unsqueeze(0).cuda()
        synchronize()
        started = perf_counter()
        with torch.inference_mode(), autocast():
            prediction = model(gpu)["pose_enc"]
        synchronize()
        elapsed = perf_counter() - started
        try:
            if tuple(prediction.shape) != (1, 2, 9) or not prediction.is_cuda:
                raise ValueError("expected the native two-camera prediction")
            value, _ = decode(prediction.float(), (224, 224), build_intrinsics=False)
            extrinsics = value[0].cpu().numpy()
            errors = score(extrinsics, truth)
            state = "DEGENERATE_TRANSLATION" if errors["predicted_translation_degenerate"] else "PREDICTED"
            result[arm] = dict(forward_s=elapsed, state=state, extrinsics=extrinsics.tolist(), **errors)
        except ValueError:
            result[arm] = dict(forward_s=elapsed, state="INVALID_PREDICTION", pair_error_deg=None)
        record(dict(sequence=result["sequence"], window=result["window"], arm=arm, result=result[arm]))


def load_pinned_state(torch, path, expected, expected_size=None):
    if path.resolve(strict=True) != path.absolute() or not path.is_file():
        raise ValueError("expected an unaliased model state file")
    with path.open("rb") as stream:
        stat = os.fstat(stream.fileno())
        if expected_size is not None and stat.st_size != expected_size:
            raise ValueError("model state file size differs from original head evidence")
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024**2), b""):
            digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError("model state file SHA differs")
        stream.seek(0)
        state = torch.load(stream, map_location="cpu", weights_only=True)
        final = os.fstat(stream.fileno())
    if (stat.st_size, stat.st_mtime_ns, stat.st_ino) != (final.st_size, final.st_mtime_ns, final.st_ino):
        raise ValueError("model state changed while reading")
    return state


def evaluate(args, cases, rows, heads, original, record):
    import torch
    from gencuda.backends import vggt_training
    from gencuda.runtime import autocast_bf16, synchronize

    if (torch.__version__ != TORCH_VERSION or torch.version.hip != HIP_VERSION
            or sha(read_bounded(Path(vggt_training.__file__), 512 * 1024)) != BACKEND_SHA):
        raise ValueError("expected the pinned VGGT backend and Torch/HIP runtime")
    if torch.cuda.device_count() != 1 or torch.get_default_dtype() != torch.float32:
        raise ValueError("expected one isolated HIP device and FP32 input preparation")
    if any(sha(read_bounded(args.upstream / "vggt" / name, 65536)) != wanted
           for name, wanted in PROVIDERS.items()):
        raise ValueError("the original camera provider differs")
    sys.path.insert(0, str(args.upstream))
    from vggt.models.vggt import VGGT
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    _module_sha = vggt_training._module_sha
    state = load_pinned_state(torch, args.weights, original["WEIGHT_SHA"])
    model = VGGT(enable_camera=True, enable_point=False, enable_depth=False, enable_track=False)
    missing, unused = model.load_state_dict(state.get("model", state), strict=False)
    del state
    if missing or any(not key.startswith(("point_head.", "depth_head.", "track_head.")) for key in unused):
        raise ValueError("the original model state does not fit the camera-only entry")
    if _module_sha(model.camera_head) != original["BASE_HEAD"]:
        raise ValueError("baseline camera head differs")
    if _module_sha(model.aggregator) != original["AGGREGATOR"]:
        raise ValueError("frozen aggregator differs")
    model.requires_grad_(False).eval().cuda()
    roots = {"alternate25": args.alternate, "lowlr25": args.lowlr}
    for arm in ("base", *ARMS):
        if arm != "base":
            proof, path = heads[arm], roots[arm] / HEAD
            state = load_pinned_state(torch, path, proof["file_SHA256"], proof["head_bytes_stat"])
            model.camera_head.load_state_dict(state, strict=True)
            del state
        expected = original["BASE_HEAD"] if arm == "base" else heads[arm]["module_SHA256"]
        if _module_sha(model.camera_head) != expected:
            raise ValueError("camera head content differs")
        measure(cases, arm, model, torch, pose_encoding_to_extri_intri, original["angular_errors"],
                autocast_bf16, synchronize, record)
        if _module_sha(model.camera_head) != expected or _module_sha(model.aggregator) != original["AGGREGATOR"]:
            raise ValueError("evaluation changed the active model state")
    return summarize(rows, original["summarize_development"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("inputs", "head-authority", "alternate", "lowlr", "upstream", "weights", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    for key in ("inputs-sha", "head-authority-sha"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    original = canonical_scorer()
    cases, rows = read_cases(args.inputs, args.inputs_sha, original["relative_pose"])
    heads = read_heads(args.head_authority, args.head_authority_sha)
    args.output.mkdir(mode=0o700)
    with (args.output / "progress-private.jsonl").open("x", encoding="utf8") as out:
        def record(event):
            print(json.dumps(event, allow_nan=False), file=out, flush=True)
            os.fsync(out.fileno())

        record(dict(state="PLANNED", rows=rows, inputs_SHA256=args.inputs_sha, heads_SHA256=args.head_authority_sha))
        try:
            summary = evaluate(args, cases, rows, heads, original, record)
        except Exception as error:
            record(dict(state="EVALUATION_FAILED", reason=type(error).__name__))
            raise
        report = dict(state="COMPLETED", summary=summary, planned_windows=len(rows),
                      input_pairs=len(cases), optimizer_updates=0, formal_CO3D=False)
        record(report)
    print(json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    main()
