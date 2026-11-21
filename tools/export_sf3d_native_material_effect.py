"""Export native SF3D head predictions for a frozen held-out arm."""

import argparse
from hashlib import sha256
import importlib
import inspect
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import sys

from training_material_mae import MATERIAL_FIELDS, POLICY, scorer_sha256


BACKEND_FILES = {
    "backends/sf3d_training.py": "dc57683c87f77c62635358f8a65db30c3070f000d235a60ad28f9395e4b68802",
    "backends/sf3d_reconstruction.py": "aa0d19927e05ffb26931f28c5f29a0f01acaa1d0bcfabe45141b6a562a2decba",
    "runtime.py": "9516e0dfba734e2b58cd0fec18bcaa3c7e70ae32156b492df264fb0f6a87fbdf",
    "hy_resume_checkpoint.py": "aeb21195bb7bfb64f36ea4299e52b359a695d986d4556a70f0b1c5c3915de8d7",
}
BASE = {
    "config.yaml": (2405, "5e880cfec3dec28d4e20afe07bfa1799c8a561f56f0532ebf1b4022ae4ddc8e4"),
    "model.safetensors": (4024289892, "a3416e1cf654e7d4f5e75f116cec2c3f0a14501a77d30c2f6068bbda178de388"),
}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def file_bytes(path, size, expected, *, collect=True):
    require(path.resolve(strict=True) == path and valid_digest(expected),
            "noncanonical artifact path or invalid hash")
    digest = sha256()
    chunks = []
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        require(stat.S_ISREG(before.st_mode) and size == before.st_size, "pinned file size differs")
        amount = 0
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            amount += len(block)
            require(amount <= size, "pinned file grew during verification")
            digest.update(block)
            if collect:
                chunks.append(block)
        after = os.fstat(handle.fileno())
    identity = lambda info: (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
    require(amount == size and identity(before) == identity(after) == identity(path.stat())
            and digest.hexdigest() == expected,
            "pinned bytes changed or digest differs")
    return b"".join(chunks)


def valid_digest(value):
    return isinstance(value, str) and bool(re.fullmatch("[0-9a-f]{64}", value))


def small(artifact, ceiling):
    size = artifact["bytes"]
    require(type(size) is int and 0 < size <= ceiling, "artifact outside declared size limit")
    return file_bytes(Path(artifact["path"]), size, artifact["SHA256"])


def position_data(artifact, count):
    raw = small(artifact, 12 * count)
    require(len(raw) == 12 * count and all(all(map(math.isfinite, point))
            for point in struct.iter_unpack("<fff", raw)), "invalid frozen query positions")
    return raw


def predictions_for_sample(predict, sample, count, checkpoint):
    """Call the native predictor on precisely the bytes that were checked."""
    image, query = sample["RGBA"], sample["queries"]
    rgba = small(image, 32 * 1024**2)
    positions = position_data(query, count)
    fields = predict(rgba, positions)
    require(fields.keys() == {field for field, _ in MATERIAL_FIELDS.values()}, "native material head output inventory differs")
    for name, width in MATERIAL_FIELDS.values():
        rows = fields[name] if name == "albedo" else [fields[name]]
        require(len(rows) == (count if name == "albedo" else 1)
                and all(len(row) == width and all(math.isfinite(x) and 0 <= x <= 1 for x in row) for row in rows),
                "native material head output shape or range differs")
    return dict(sample_id=sample["sample_id"], status="scored", RGBA_SHA256=image["SHA256"],
                query_SHA256=query["SHA256"], checkpoint_SHA256=checkpoint, material=fields)


def checkpoint_head(backend, torch, tensor_names, model, checkpoint, identity):
    if checkpoint["kind"] == "native-base":
        require(checkpoint["SHA256"] == BASE["model.safetensors"][1] and checkpoint["global_step"] == 0,
                "initial head differs from the pinned base")
        return
    require(checkpoint["kind"] == "full-native-checkpoint" and type(checkpoint["bytes"]) is int
            and 0 < checkpoint["bytes"] <= 4 * 1024**3, "unsupported saved head checkpoint")
    path = Path(checkpoint["path"])
    file_bytes(path, checkpoint["bytes"], checkpoint["SHA256"], collect=False)
    payload = backend._sf3d_load_checkpoint(path, torch)
    require(set(payload) == {"metadata", "state", "rng", "integrity_sha256"}, "checkpoint inventory differs")
    metadata = payload["metadata"]
    contract = metadata["contract"]
    require(payload["integrity_sha256"] == backend._sf3d_type_tree_sha256(
        {key: value for key, value in payload.items() if key != "integrity_sha256"})
        and metadata == backend._sf3d_checkpoint_metadata(contract, checkpoint["global_step"])
        and all(contract["identity"][key] == identity[key] for key in ("backend", "weights", "sources")),
        "checkpoint internal identity, native base or stored update differs")
    heads = payload["state"]["material_heads"]
    require(set(heads) == {name for name, _ in tensor_names} and
            all(torch.is_tensor(heads[name]) and heads[name].shape == parameter.shape
                and heads[name].dtype == parameter.dtype and bool(torch.isfinite(heads[name]).all())
                for name, parameter in tensor_names), "native saved material tensors differ")
    backend._load_state(torch, tensor_names, heads)
    model.eval()


def bind_runtime(framework_root, source_root, weights, plan):
    package = framework_root / "generation/src/gencuda"
    for relative, digest in BACKEND_FILES.items():
        path = package / relative
        require(path.stat().st_size <= 256 * 1024, "framework member exceeds size limit")
        file_bytes(path, path.stat().st_size, digest, collect=False)
    for name, (size, pin) in BASE.items():
        file_bytes(weights / name, size, pin, collect=False)
    sys.path[:0] = [str(package.parent), str(source_root)]
    backend = importlib.import_module("gencuda.backends.sf3d_training")
    require(Path(backend.__file__).resolve() == package / "backends/sf3d_training.py",
            "imported unified framework provider differs")
    source_package = source_root / "sf3d"
    sources = sorted(path.resolve(strict=True) for path in source_package.rglob("*.py") if path.is_file())
    require(sources and all(path.is_relative_to(source_package) for path in sources),
            "native SF3D source inventory differs")
    source_sha = backend._sf3d_paths_sha256(sources, source_root)
    require(source_sha == plan["provider_sources_SHA256"], "native SF3D sources differ")
    provider = importlib.import_module("sf3d.system")
    require(Path(inspect.getfile(provider.SF3D)).resolve().is_relative_to(source_package),
            "imported Stable Fast 3D provider differs")
    torch = backend.require_cuda()
    require(torch.cuda.device_count() == 1, "expected one allocated accelerator for material-head evaluation")
    identity = dict(
        backend=backend._sf3d_json_sha256([dict(path="sf3d_training.py", sha256=BACKEND_FILES["backends/sf3d_training.py"])]),
        weights=backend._sf3d_json_sha256([dict(path=name, sha256=pin) for name, (_, pin) in BASE.items()]),
        sources=source_sha,
    )
    model = provider.SF3D.from_pretrained(str(weights), config_name="config.yaml", weight_name="model.safetensors")
    model.to("cuda").eval()
    named = backend._selected_named_parameters(model)
    require(len(named) == 36 and sum(p.numel() for _, p in named) == 2644871,
            "material-head scope differs from the original recipe")
    checkpoint_head(backend, torch, named, model, plan["checkpoint"], identity)
    return backend, torch, model


def native_predict(backend, torch, model, image, positions):
    import numpy as np
    from PIL import Image

    with Image.open(io.BytesIO(image)) as decoded:
        require(decoded.mode == "RGBA" and 0 < decoded.width <= 4096 and 0 < decoded.height <= 4096,
                "expected the frozen RGBA input")
        rgba = decoded.copy()
    points = torch.from_numpy(np.frombuffer(positions, dtype="<f4").copy().reshape(-1, 3)).to("cuda")
    with torch.inference_mode(), backend.autocast_bf16():
        batch = backend._prepare_batch(model, rgba)
        scene_codes, _ = model.get_scene_codes(dict(batch))
        triplane = model.query_triplane(points, scene_codes[0])[0]
        features = backend._image_features(torch, model, batch)
        albedo = torch.sigmoid(model.decoder.heads["features"](triplane))
        roughness = backend._beta_mean(torch, model.image_estimator.heads["roughness"], features)
        metallic = backend._beta_mean(torch, model.image_estimator.heads["metallic"], features)
    return dict(albedo=albedo.float().cpu().tolist(),
                roughness=roughness.float().cpu().reshape(-1).tolist(),
                metallic=metallic.float().cpu().reshape(-1).tolist())


def save_json(path, value):
    raw = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return dict(location=str(path), SHA256=sha256(raw).hexdigest())


def validate_plan(plan):
    samples = plan["samples"]
    ids = [sample["sample_id"] for sample in samples]
    count = plan["query_points_per_sample"]
    require(plan["case"] == "T11.material-head" and plan["arm"] in ("A100", "Hygon", "Hygon_before")
            and (plan["checkpoint"]["kind"] != "native-base" or plan["arm"] == "Hygon_before")
            and all(valid_digest(value) for value in (plan["dataset_SHA256"], plan["provider_sources_SHA256"],
                                                      plan["checkpoint"]["SHA256"]))
            and 0 < len(ids) <= 4096 and all(isinstance(name, str) and name.strip() for name in ids)
            and len(set(ids)) == len(ids)
            and type(count) is int and 0 < count <= 16_384, "invalid frozen sample or query denominator")


def emit(plan, output, predict, plan_sha, execution=None):
    validate_plan(plan)
    samples = plan["samples"]
    count = plan["query_points_per_sample"]
    checkpoint = plan["checkpoint"]["SHA256"]
    rows = []
    for index, sample in enumerate(samples):
        try:
            result = predictions_for_sample(predict, sample, count, checkpoint)
        except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
            result = dict(sample_id=sample["sample_id"], status="failed", failure=type(error).__name__)
        artifact = save_json(output / f"sample-{index:05d}.json", result)
        rows.append(dict(sample_id=result["sample_id"], status=result["status"], artifact=artifact))
    failed = sum(item["status"] == "failed" for item in rows)
    result = dict(case="T11.material-head", arm=plan["arm"], training_checkpoint_SHA256=checkpoint,
                  policy=POLICY, scorer_SHA256=scorer_sha256(), planned=len(samples), scored=len(samples)-failed,
                  plan_SHA256=plan_sha, dataset_SHA256=plan["dataset_SHA256"],
                  provider_sources_SHA256=plan["provider_sources_SHA256"], failed=failed, records=rows,
                  execution=execution)
    save_json(output / "manifest.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "output", "framework-root", "sf3d-source", "weights"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    plan = json.loads(small(dict(path=str(args.plan), bytes=args.plan.stat().st_size, SHA256=args.plan_sha256),
                            8 * 1024**2))
    validate_plan(plan)
    for key in ("SF3D_OFFLINE_INIT", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ[key] = "1"
    require(args.output.resolve() == args.output, "expected a new canonical output path")
    args.output.mkdir(mode=0o700)
    save_json(args.output / "intent.json", dict(plan_SHA256=args.plan_sha256, arm=plan["arm"]))
    try:
        runtime = bind_runtime(args.framework_root, args.sf3d_source, args.weights, plan)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, ImportError, RuntimeError) as error:
        failure = dict(state="NATIVE_PREDICTION_SETUP_FAILED", plan_SHA256=args.plan_sha256,
                       planned=len(plan["samples"]), scored=0, missing=len(plan["samples"]),
                       error_type=type(error).__name__)
        save_json(args.output / "setup-failure.json", failure)
        print(json.dumps(failure), flush=True)
        return 1
    _, torch, _ = runtime
    execution = dict(framework_files=BACKEND_FILES, torch=str(torch.__version__),
                     accelerator=str(torch.cuda.get_device_name(0)), device_count=1,
                     Slurm_job=os.environ.get("SLURM_JOB_ID"), Slurm_step=os.environ.get("SLURM_STEP_ID"),
                     global_step=plan["checkpoint"]["global_step"])
    result = emit(plan, args.output, lambda image, positions: native_predict(*runtime, image, positions),
                  args.plan_sha256, execution)
    print(json.dumps({key: result[key] for key in ("case", "arm", "planned", "scored", "failed")}), flush=True)
    return int(result["failed"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
