#!/usr/bin/env python3
"""Prepare deterministic, verified HumanML3D caches for HY-Motion training."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path, PureWindowsPath
from typing import Any, Iterator

import numpy as np


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def caption_for(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        caption = line.split("#", 1)[0].strip()
        if caption:
            return caption
    raise RuntimeError(f"no usable caption in {path}")


def assemble(rot6d: np.ndarray, joints: np.ndarray) -> np.ndarray:
    if rot6d.ndim != 2 or rot6d.shape[1] != 135:
        raise RuntimeError(f"invalid rot6d shape: {rot6d.shape}")
    if joints.ndim != 3 or joints.shape[1:] != (22, 3):
        raise RuntimeError(f"invalid joint shape: {joints.shape}")
    frames = min(len(rot6d), len(joints))
    return np.concatenate(
        [rot6d[:frames], joints[:frames].reshape(frames, 66)], axis=-1
    ).astype(np.float32, copy=False)


def _validate_train_sample_id(sample_id: str) -> None:
    windows_path = PureWindowsPath(sample_id)
    if (
        not sample_id
        or sample_id in {".", ".."}
        or "/" in sample_id
        or "\\" in sample_id
        or Path(sample_id).is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or ".." in windows_path.parts
    ):
        raise RuntimeError(f"invalid HumanML3D train sample ID: {sample_id!r}")


def iter_selected_samples(
    args: argparse.Namespace, skipped_samples: list[dict[str, Any]]
) -> Iterator[dict[str, Any]]:
    selected_count = 0
    strict = getattr(args, "require_full_frames", False)
    sample_ids: set[str] = set()
    with (args.dataset_root / "train.txt").open(encoding="utf-8") as handle:
        for line in handle:
            name = line.strip()
            if not name:
                continue
            _validate_train_sample_id(name)
            if name in sample_ids:
                raise RuntimeError(f"HumanML3D train split contains duplicate sample ID: {name}")
            sample_ids.add(name)
            rot_path = args.dataset_root / "joints_6d" / f"{name}.npy"
            joints_path = args.dataset_root / "new_joints" / f"{name}.npy"
            text_path = args.dataset_root / "texts" / f"{name}.txt"
            missing = [path for path in (rot_path, joints_path, text_path) if not path.is_file()]
            if missing:
                skipped_samples.append(
                    {
                        "sample_id": name,
                        "reason": "missing_source_files",
                        "missing_paths": [str(path) for path in missing],
                    }
                )
                continue
            rot6d = np.load(rot_path, allow_pickle=False, mmap_mode="r" if strict else None)
            joints = np.load(joints_path, allow_pickle=False, mmap_mode="r" if strict else None)
            if strict:
                if (rot6d.ndim != 2 or rot6d.shape[1] != 135
                    or joints.ndim != 3 or joints.shape[1:] != (22, 3)
                    or len(rot6d) != len(joints)
                    or rot6d.dtype.kind != "f" or joints.dtype.kind != "f"):
                    raise RuntimeError(f"invalid paired native source shapes or dtypes: {name}")
            frames = min(len(rot6d), len(joints))
            if frames < (args.frames if strict else args.min_frames):
                if strict:
                    skipped_samples.append({"sample_id": name, "reason": "source_too_short", "frames": frames})
                continue
            length = min(frames, args.frames)
            motion_135 = rot6d[:length].astype(np.float32, copy=False)
            if strict and not (np.isfinite(motion_135).all() and np.isfinite(joints[:length]).all()):
                raise RuntimeError(f"non-finite native source frames: {name}")
            yield {
                "sample_id": name,
                "prompt": caption_for(text_path),
                "motion_135": motion_135,
                "length": length,
                "sources": {
                    "rot6d": {"path": str(rot_path), "sha256": sha256_path(rot_path)},
                    "joints": {"path": str(joints_path), "sha256": sha256_path(joints_path)},
                    "text": {"path": str(text_path), "sha256": sha256_path(text_path)},
                },
            }
            selected_count += 1
            if args.samples and selected_count == args.samples:
                return
    if args.samples and selected_count != args.samples:
        raise RuntimeError(f"selected {selected_count} samples, expected {args.samples}")
    if selected_count == 0:
        raise RuntimeError("HumanML3D selection produced no samples")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _create_staging_output(output_dir: Path) -> Path:
    if output_dir.exists():
        raise RuntimeError(f"HY-Motion training output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))


def _publish_staged_output(staging_dir: Path, output_dir: Path) -> None:
    if output_dir.exists():
        raise RuntimeError(f"HY-Motion training output already exists: {output_dir}")
    staging_dir.rename(output_dir)


def _write_shard(
    torch: Any,
    path: Path,
    frames: int,
    samples: list[dict[str, Any]],
    dataset_manifest_sha256: str | None,
) -> None:
    temporary = path.with_suffix(".pt.tmp")
    torch.save(
        {
            "schema_version": 1,
            "frames": frames,
            "feature_dim": 201,
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "samples": samples,
        },
        temporary,
    )
    temporary.replace(path)


def _bind_dataset_manifest(
    torch: Any, shards: list[dict[str, Any]], dataset_manifest_sha256: str
) -> None:
    for shard in shards:
        path = Path(shard["file_path"])
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload["dataset_manifest_sha256"] = dataset_manifest_sha256
        temporary = path.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
        shard["sha256"] = sha256_path(path)


def initialize_device(torch: Any, provider: str, seed: int) -> dict[str, str]:
    if provider == "mlu":
        import torch_mlu  # noqa: F401
        torch.manual_seed(seed)
        torch.mlu.manual_seed_all(seed)
        return {"device": provider, "backend": "MLU", "dtype": "bfloat16"}
    if provider != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("an available torch.cuda device is required for CUDA or ROCm")
    backend = "ROCm" if torch.version.hip else "CUDA"
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return {"device": provider, "backend": backend, "dtype": "bfloat16"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/mnt/afs/datasets/generative-mlu/humanml3d"),
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("/mnt/afs/datasets/generative-mlu/motionreward-data/humanml3d.tar.gz"),
    )
    parser.add_argument(
        "--weights", type=Path, default=Path("/mnt/afs/models/hy-motion-1.0")
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/workspace/cambricon_generation/sources/hy-motion"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/afs/datasets/generative-mlu/hy-motion-training-cache"),
    )
    parser.add_argument("--variant", default="HY-Motion-1.0")
    parser.add_argument("--samples", type=int, default=64, help="0 selects every eligible train sample")
    parser.add_argument("--shard-size", type=int, default=64)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--min-frames", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--qwen-path", default="/mnt/afs/models/Qwen3-8B")
    parser.add_argument("--clip-path", default="/mnt/afs/models/clip-vit-large-patch14")
    parser.add_argument("--device", choices=("mlu", "cuda"), default="mlu",
                        help="use torch.cuda for either ROCm or NVIDIA CUDA; default is the original MLU path")
    parser.add_argument("--require-full-frames", action="store_true",
                        help="require paired real source frames for the entire output window")
    parser.add_argument("--check-source-only", action="store_true",
                        help="check the selected paired source windows without models, cache output or devices")
    args = parser.parse_args()
    if args.samples < 0 or args.shard_size <= 0 or args.frames <= 0 or args.min_frames <= 0:
        raise ValueError("samples, shard-size, frames, and min-frames must be positive (samples may be 0)")
    if args.check_source_only:
        if not args.require_full_frames:
            parser.error("--check-source-only requires --require-full-frames")
        skipped = []
        verified = sum(1 for _ in iter_selected_samples(args, skipped))
        print(json.dumps({"source_only": True, "eligible_samples": verified,
                          "verified_frames_per_sample": args.frames, "skipped_samples": len(skipped),
                          "cache_written": False, "model_invocations": 0}, sort_keys=True))
        return 0
    for path in (args.dataset_root, args.weights, args.source_root):
        if not path.is_dir():
            raise RuntimeError(f"required directory is missing: {path}")
    if not args.archive.is_file():
        raise RuntimeError(f"source archive is missing: {args.archive}")
    staging_dir = _create_staging_output(args.output_dir)

    stats_root = args.source_root / "stats"
    mean_path = stats_root / "Mean.npy"
    std_path = stats_root / "Std.npy"
    mean = np.load(mean_path).astype(np.float32)
    std = np.load(std_path).astype(np.float32)
    if mean.shape != (201,) or std.shape != (201,):
        raise RuntimeError(f"invalid HY-Motion statistics: {mean.shape}, {std.shape}")
    safe_std = np.where(std < 1e-3, 1.0, std)

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ["HY_MOTION_QWEN_PATH"] = args.qwen_path
    os.environ["HY_MOTION_CLIP_PATH"] = args.clip_path
    sys.path.insert(0, str(args.source_root))
    import torch
    import yaml

    device_provenance = initialize_device(torch, args.device, args.seed)
    from hymotion.pipeline.body_model import WoodenMesh
    from hymotion.utils.loaders import load_object

    model_config_path = args.weights / args.variant / "config.yml"
    model_config = dict(yaml.safe_load(model_config_path.read_text(encoding="utf-8")))
    pipeline_args = dict(model_config["train_pipeline_args"])
    text_encoder = load_object(
        pipeline_args["text_encoder_module"], pipeline_args["text_encoder_cfg"]
    ).to(device=args.device, dtype=torch.bfloat16)
    text_encoder.eval()
    body_model = WoodenMesh().to(device=args.device, dtype=torch.bfloat16)
    body_model.eval()

    source_samples: list[dict[str, Any]] = []
    skipped_samples: list[dict[str, Any]] = []
    sample_ids: list[str] = []
    shards: list[dict[str, Any]] = []
    shard_samples: list[dict[str, Any]] = []
    raw_abs_sum = 0.0
    normalized_abs_sum = 0.0
    raw_max_abs = 0.0
    normalized_max_abs = 0.0
    raw_value_count = 0
    shards_dir = staging_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for item in iter_selected_samples(args, skipped_samples):
            encoded = text_encoder.encode(text=[item["prompt"]])
            motion_135 = item["motion_135"]
            rot6d = torch.from_numpy(motion_135[:, 3:135].reshape(-1, 22, 6)).to(
                device=args.device, dtype=torch.bfloat16
            )
            translation = torch.from_numpy(motion_135[:, :3]).to(
                device=args.device, dtype=torch.bfloat16
            )
            keypoints = body_model({"rot6d": rot6d, "trans": translation})[
                "keypoints3d"
            ][:, :22].float().cpu().numpy()
            if args.require_full_frames and keypoints.shape != (args.frames, 22, 3):
                raise RuntimeError(f"incomplete WoodenMesh keypoints: {item['sample_id']}")
            raw = assemble(motion_135, keypoints)
            normalized = (raw - mean) / safe_std
            if not np.isfinite(normalized).all():
                raise RuntimeError(f"non-finite normalized sample: {item['sample_id']}")
            padded = np.zeros((args.frames, 201), dtype=np.float32)
            padded[: item["length"]] = normalized[: item["length"]]
            raw_abs_sum += float(np.abs(raw).sum())
            normalized_abs_sum += float(np.abs(normalized).sum())
            raw_max_abs = max(raw_max_abs, float(np.abs(raw).max()))
            normalized_max_abs = max(normalized_max_abs, float(np.abs(normalized).max()))
            raw_value_count += int(raw.size)
            source_samples.append(
                {key: item[key] for key in ("sample_id", "prompt", "length", "sources")}
            )
            sample_ids.append(str(item["sample_id"]))
            shard_samples.append(
                {
                    "sample_id": item["sample_id"],
                    "prompt": item["prompt"],
                    "length": item["length"],
                    "motion": torch.from_numpy(padded),
                    "vtxt_input": encoded[0].detach().cpu().to(torch.bfloat16).contiguous(),
                    "ctxt_input": encoded[1].detach().cpu().to(torch.bfloat16).contiguous(),
                    "ctxt_length": int(encoded[2].reshape(-1)[0].cpu()),
                }
            )
            if len(shard_samples) == args.shard_size:
                shard_path = shards_dir / f"training-cache-{len(shards):05d}.pt"
                _write_shard(torch, shard_path, args.frames, shard_samples, None)
                shards.append(
                    {
                        "file_path": str(shard_path),
                        "path": str(shard_path.relative_to(staging_dir)).replace("\\", "/"),
                        "sample_count": len(shard_samples),
                        "sample_ids": [sample["sample_id"] for sample in shard_samples],
                    }
                )
                shard_samples = []
            del encoded, keypoints, normalized, padded, raw, rot6d, translation
    if shard_samples:
        shard_path = shards_dir / f"training-cache-{len(shards):05d}.pt"
        _write_shard(torch, shard_path, args.frames, shard_samples, None)
        shards.append(
            {
                "file_path": str(shard_path),
                "path": str(shard_path.relative_to(staging_dir)).replace("\\", "/"),
                "sample_count": len(shard_samples),
                "sample_ids": [sample["sample_id"] for sample in shard_samples],
            }
        )
    if not source_samples or raw_value_count == 0:
        raise RuntimeError("HumanML3D selection produced no usable motion values")

    dataset_manifest = {
        "schema_version": 1,
        "dataset": "MotionRewardData HumanML3D derivative",
        "source_repository": "XiaofengTan/MotionRewardData",
        "source_archive": str(args.archive),
        "source_archive_sha256": sha256_path(args.archive),
        "split": "train",
        "selection": "first eligible official train IDs in train.txt order",
        "requested_sample_count": args.samples if args.samples else None,
        "sample_count": len(source_samples),
        "skipped_sample_count": len(skipped_samples),
        "skipped_samples": skipped_samples,
        "frames": args.frames,
        "feature_layout": "translation_3 + rotation_6d_22x6 + WoodenMesh_keypoints_22x3",
        "feature_dim": 201,
        "model_statistics": {
            "mean": {"path": str(mean_path), "sha256": sha256_path(mean_path)},
            "std": {"path": str(std_path), "sha256": sha256_path(std_path)},
        },
        "raw_statistics": {
            "mean_abs": raw_abs_sum / raw_value_count,
            "max_abs": raw_max_abs,
            "normalized_mean_abs": normalized_abs_sum / raw_value_count,
            "normalized_max_abs": normalized_max_abs,
        },
        "samples": source_samples,
    }
    if args.require_full_frames or args.device != "mlu":
        dataset_manifest["producer"] = dict(device_provenance, require_full_frames=args.require_full_frames)
    dataset_manifest_path = staging_dir / "dataset-manifest.json"
    _write_json(dataset_manifest_path, dataset_manifest)
    dataset_manifest_sha = sha256_path(dataset_manifest_path)
    _bind_dataset_manifest(torch, shards, dataset_manifest_sha)
    training_manifest = {
        "schema_version": 2,
        "format": "hy-motion-training-shards-v1",
        "frames": args.frames,
        "feature_dim": 201,
        "sample_count": len(sample_ids),
        "sample_ids": sample_ids,
        "dataset_manifest": dataset_manifest_path.name,
        "dataset_manifest_sha256": dataset_manifest_sha,
        "shards": [
            {
                key: shard[key]
                for key in ("path", "sha256", "sample_count", "sample_ids")
            }
            for shard in shards
        ],
    }
    training_manifest_path = staging_dir / "training-data-manifest.json"
    _write_json(training_manifest_path, training_manifest)
    _publish_staged_output(staging_dir, args.output_dir)
    published_dataset_manifest = args.output_dir / dataset_manifest_path.name
    published_training_manifest = args.output_dir / training_manifest_path.name
    print(
        json.dumps(
            {
                "training_data_manifest": str(published_training_manifest),
                "training_data_manifest_sha256": sha256_path(published_training_manifest),
                "dataset_manifest": str(published_dataset_manifest),
                "dataset_manifest_sha256": sha256_path(published_dataset_manifest),
                "sample_count": len(sample_ids),
                "shard_count": len(shards),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
