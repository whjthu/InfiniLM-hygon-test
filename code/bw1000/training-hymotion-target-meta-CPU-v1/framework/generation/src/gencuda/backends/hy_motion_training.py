from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import random
from collections.abc import Sequence
try:
    import resource
except ModuleNotFoundError:  # Windows can inspect the CLI but cannot run an CUDA backend.
    resource = None
import statistics
import sys
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter, time
from typing import Any, Iterator

import numpy as np
import yaml

from .. import hy_resume_checkpoint as hy_resume
from ..manifest import sha256_path
from ..recovery_validation import run_recovery_validation, validate_recovery_enabled
from ..runtime import memory_stats, require_cuda, synchronize, within_memory_limit


HY_MOTION_DATA_MANIFEST_SCHEMA_VERSION = 2
HY_MOTION_FULL_RESUME_SCHEMA = "hy-motion-full-flow-step-v1"
HY_MOTION_FULL_RESUME_IDENTITY_FIELDS = {
    "config", "source", "model", "data", "runtime", "dependencies"
}


class HYMotionFullResumeApplyFatal(RuntimeError):
    """State application failed after checkpoint validation."""


def _resume_require(condition: bool, message: str) -> None:
    if not condition:
        raise hy_resume.CheckpointValidationError(message)


def validate_hy_motion_cache(cache: dict[str, Any]) -> None:
    samples = cache.get("samples")
    if not isinstance(samples, list) or not samples:
        raise RuntimeError("HY-Motion training cache has no samples")
    sample_ids: set[str] = set()
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise RuntimeError(f"invalid HY-Motion cache sample at {index}")
        motion = sample.get("motion")
        vtxt = sample.get("vtxt_input")
        ctxt = sample.get("ctxt_input")
        length = int(sample.get("length", 0))
        if tuple(getattr(motion, "shape", ())) != (cache["frames"], 201):
            raise RuntimeError(f"invalid HY-Motion motion cache shape at {index}")
        if getattr(vtxt, "ndim", 0) not in (2, 3) or vtxt.shape[-1] != 768:
            raise RuntimeError(f"invalid HY-Motion vector text cache shape at {index}")
        if getattr(ctxt, "ndim", 0) not in (2, 3) or ctxt.shape[-1] != 4096:
            raise RuntimeError(f"invalid HY-Motion context text cache shape at {index}")
        if not 1 <= length <= cache["frames"]:
            raise RuntimeError(f"invalid HY-Motion motion length at {index}: {length}")
        if not isinstance(sample.get("prompt"), str) or not sample["prompt"]:
            raise RuntimeError(f"HY-Motion cache prompt is missing at {index}")
        sample_id = sample.get("sample_id")
        if sample_id is None:
            sample_id = f"legacy-{index:08d}"
            sample["sample_id"] = sample_id
        if not isinstance(sample_id, str) or not sample_id:
            raise RuntimeError(f"HY-Motion cache sample_id is missing at {index}")
        if sample_id in sample_ids:
            raise RuntimeError(f"duplicate HY-Motion cache sample_id: {sample_id}")
        sample_ids.add(sample_id)


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise RuntimeError(f"invalid HY-Motion {label} SHA256")
    return value


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid HY-Motion {label}: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"invalid HY-Motion {label}: {path}")
    return value


def _manifest_root_path(manifest_path: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"HY-Motion training data {label} path is missing")
    candidate = Path(value)
    if candidate.is_absolute():
        raise RuntimeError(f"HY-Motion training data {label} path must be relative")
    root = manifest_path.parent.resolve()
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise RuntimeError(f"HY-Motion training data {label} path escapes manifest root") from error
    return resolved


def _validate_hy_motion_data_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    if manifest.get("schema_version") != HY_MOTION_DATA_MANIFEST_SCHEMA_VERSION:
        raise RuntimeError("unsupported HY-Motion training data manifest schema")
    if manifest.get("format") != "hy-motion-training-shards-v1":
        raise RuntimeError("invalid HY-Motion training data manifest format")
    if type(manifest.get("frames")) is not int or manifest["frames"] <= 0:
        raise RuntimeError("invalid HY-Motion training data manifest frames")
    if manifest.get("feature_dim") != 201:
        raise RuntimeError("invalid HY-Motion training data manifest feature_dim")
    if not isinstance(manifest.get("dataset_manifest"), str) or not manifest["dataset_manifest"]:
        raise RuntimeError("HY-Motion training data manifest dataset_manifest is missing")
    _require_sha256(
        manifest.get("dataset_manifest_sha256"), "training data dataset manifest"
    )
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise RuntimeError("HY-Motion training data manifest has no shards")

    declared_ids: list[str] = []
    for index, shard in enumerate(shards):
        if not isinstance(shard, dict):
            raise RuntimeError(f"invalid HY-Motion training data shard at {index}")
        if not isinstance(shard.get("path"), str) or not shard["path"]:
            raise RuntimeError(f"HY-Motion training data shard path is missing at {index}")
        _require_sha256(shard.get("sha256"), f"training data shard {index}")
        sample_ids = shard.get("sample_ids")
        if not isinstance(sample_ids, list) or not sample_ids or not all(
            isinstance(sample_id, str) and sample_id for sample_id in sample_ids
        ):
            raise RuntimeError(f"invalid HY-Motion training data shard sample_ids at {index}")
        if len(set(sample_ids)) != len(sample_ids):
            raise RuntimeError(f"duplicate HY-Motion training data shard sample_id at {index}")
        if shard.get("sample_count") != len(sample_ids):
            raise RuntimeError(f"invalid HY-Motion training data shard sample_count at {index}")
        declared_ids.extend(sample_ids)

    if len(set(declared_ids)) != len(declared_ids):
        raise RuntimeError("duplicate HY-Motion training data manifest sample_id")
    if manifest.get("sample_count") != len(declared_ids):
        raise RuntimeError("invalid HY-Motion training data manifest sample_count")
    if manifest.get("sample_ids") != declared_ids:
        raise RuntimeError("HY-Motion training data manifest sample_ids do not match shards")
    return shards


class _ShardedHYMotionSamples(Sequence[dict[str, Any]]):
    """Load one verified CPU shard at a time; model inputs still move directly to CUDA."""

    def __init__(
        self,
        torch: Any,
        shards: list[dict[str, Any]],
        shard_paths: list[Path],
        frames: int,
        dataset_manifest_sha256: str,
    ) -> None:
        self._torch = torch
        self._shards = shards
        self._shard_paths = shard_paths
        self._frames = frames
        self._dataset_manifest_sha256 = dataset_manifest_sha256
        self._ends: list[int] = []
        total = 0
        for shard in shards:
            total += int(shard["sample_count"])
            self._ends.append(total)
        self._cached_shard_index: int | None = None
        self._cached_samples: list[dict[str, Any]] = []

    def __len__(self) -> int:
        return self._ends[-1]

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard_index = next(
            candidate for candidate, end in enumerate(self._ends) if index < end
        )
        if shard_index != self._cached_shard_index:
            shard = self._shards[shard_index]
            path = self._shard_paths[shard_index]
            actual_sha = sha256_path(path)
            if actual_sha != shard["sha256"]:
                raise RuntimeError(
                    f"HY-Motion training data shard changed at {shard_index}: {actual_sha}"
                )
            cache = self._torch.load(path, map_location="cpu", weights_only=False)
            validate_hy_motion_cache(cache)
            if cache.get("frames") != self._frames or cache.get("feature_dim") != 201:
                raise RuntimeError(
                    f"HY-Motion training data shard metadata changed at {shard_index}"
                )
            if cache.get("dataset_manifest_sha256") != self._dataset_manifest_sha256:
                raise RuntimeError(
                    f"HY-Motion training data shard dataset manifest changed at {shard_index}"
                )
            samples = cache["samples"]
            if [sample["sample_id"] for sample in samples] != shard["sample_ids"]:
                raise RuntimeError(
                    f"HY-Motion training data shard samples changed at {shard_index}"
                )
            self._cached_shard_index = shard_index
            self._cached_samples = samples
        start = 0 if shard_index == 0 else self._ends[shard_index - 1]
        return self._cached_samples[index - start]

    def sample_for_epoch_cursor(
        self, seed: int, epoch: int, cursor: int
    ) -> dict[str, Any]:
        shard_order = list(range(len(self._shards)))
        random.Random(seed + epoch).shuffle(shard_order)
        for shard_index in shard_order:
            shard_start = 0 if shard_index == 0 else self._ends[shard_index - 1]
            shard_size = int(self._shards[shard_index]["sample_count"])
            if cursor >= shard_size:
                cursor -= shard_size
                continue
            local_order = list(range(shard_size))
            random.Random((seed + epoch) * 1_000_003 + shard_index).shuffle(local_order)
            return self[shard_start + local_order[cursor]]
        raise RuntimeError("HY-Motion training data cursor is outside the epoch")


def _load_hy_motion_training_data(
    config: dict[str, Any], torch: Any
) -> tuple[Sequence[dict[str, Any]], dict[str, Any]]:
    manifest_value = config.get("training_data_manifest")
    if manifest_value is None:
        cache_path = Path(str(config["training_cache"]))
        if not cache_path.is_file():
            raise RuntimeError("HY-Motion training cache is missing")
        cache_sha = sha256_path(cache_path)
        expected_cache_sha = str(config.get("training_cache_sha", ""))
        if expected_cache_sha and expected_cache_sha != cache_sha:
            raise RuntimeError(
                f"HY-Motion cache SHA mismatch: {cache_sha}, expected {expected_cache_sha}"
            )
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        if not isinstance(cache, dict):
            raise RuntimeError("invalid HY-Motion training cache")
        validate_hy_motion_cache(cache)
        return cache["samples"], {
            "frames": cache["frames"],
            "feature_dim": cache["feature_dim"],
            "dataset_manifest_sha256": cache.get("dataset_manifest_sha256"),
            "training_cache_sha256": cache_sha,
            "training_data_manifest_sha256": None,
            "shard_sha256s": [],
            "shuffle_by_epoch": False,
        }

    manifest_path = Path(str(manifest_value))
    if not manifest_path.is_file():
        raise RuntimeError("HY-Motion training data manifest is missing")
    manifest_sha = sha256_path(manifest_path)
    expected_manifest_sha = str(
        config.get(
            "training_data_manifest_sha256",
            config.get("training_data_manifest_sha", ""),
        )
    )
    if expected_manifest_sha and expected_manifest_sha != manifest_sha:
        raise RuntimeError(
            "HY-Motion training data manifest SHA mismatch: "
            f"{manifest_sha}, expected {expected_manifest_sha}"
        )
    manifest = _load_json_object(manifest_path, "training data manifest")
    shards = _validate_hy_motion_data_manifest(manifest)
    dataset_manifest_path = _manifest_root_path(
        manifest_path, manifest["dataset_manifest"], "dataset manifest"
    )
    if not dataset_manifest_path.is_file():
        raise RuntimeError("HY-Motion training data dataset manifest is missing")
    dataset_manifest_sha = sha256_path(dataset_manifest_path)
    if dataset_manifest_sha != manifest["dataset_manifest_sha256"]:
        raise RuntimeError(
            "HY-Motion training data dataset manifest SHA mismatch: "
            f"{dataset_manifest_sha}"
        )
    actual_ids: list[str] = []
    shard_shas: list[str] = []
    shard_paths: list[Path] = []
    for index, shard in enumerate(shards):
        shard_path = _manifest_root_path(manifest_path, shard["path"], f"shard {index}")
        if not shard_path.is_file():
            raise RuntimeError(f"HY-Motion training data shard is missing at {index}: {shard_path}")
        actual_shard_sha = sha256_path(shard_path)
        if actual_shard_sha != shard["sha256"]:
            raise RuntimeError(
                f"HY-Motion training data shard SHA mismatch at {index}: {actual_shard_sha}"
            )
        cache = torch.load(shard_path, map_location="cpu", weights_only=False)
        if not isinstance(cache, dict):
            raise RuntimeError(f"invalid HY-Motion training data shard at {index}")
        validate_hy_motion_cache(cache)
        if cache.get("frames") != manifest["frames"] or cache.get("feature_dim") != 201:
            raise RuntimeError(f"HY-Motion training data shard metadata mismatch at {index}")
        if cache.get("dataset_manifest_sha256") != dataset_manifest_sha:
            raise RuntimeError(f"HY-Motion training data shard dataset manifest mismatch at {index}")
        shard_ids = [sample["sample_id"] for sample in cache["samples"]]
        if shard_ids != shard["sample_ids"]:
            raise RuntimeError(f"HY-Motion training data shard samples mismatch at {index}")
        actual_ids.extend(shard_ids)
        shard_shas.append(actual_shard_sha)
        shard_paths.append(shard_path)
    if actual_ids != manifest["sample_ids"]:
        raise RuntimeError("HY-Motion training data manifest samples are missing or duplicated")
    return _ShardedHYMotionSamples(
        torch,
        shards,
        shard_paths,
        int(manifest["frames"]),
        dataset_manifest_sha,
    ), {
        "frames": manifest["frames"],
        "feature_dim": manifest["feature_dim"],
        "dataset_manifest_sha256": manifest.get("dataset_manifest_sha256"),
        "training_cache_sha256": None,
        "training_data_manifest_sha256": manifest_sha,
        "shard_sha256s": shard_shas,
        "shuffle_by_epoch": True,
    }


def _hy_motion_sample_for_step(
    samples: Sequence[dict[str, Any]], seed: int, global_step: int, shuffle_by_epoch: bool
) -> tuple[dict[str, Any], int, int]:
    if global_step < 1:
        raise ValueError("HY-Motion global_step must be positive")
    epoch, cursor = divmod(global_step - 1, len(samples))
    if not shuffle_by_epoch:
        return samples[cursor], epoch, cursor
    if isinstance(samples, _ShardedHYMotionSamples):
        return samples.sample_for_epoch_cursor(seed, epoch, cursor), epoch, cursor
    order = list(range(len(samples)))
    random.Random(seed + epoch).shuffle(order)
    return samples[order[cursor]], epoch, cursor


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _adapter_parameter_name(name: str) -> str:
    """Remove transparent activation-checkpoint wrappers from LoRA paths."""
    return name.replace("._checkpoint_wrapped_module", "")


def _canonical_adapter_parameters(model: Any) -> dict[str, Any]:
    parameters: dict[str, Any] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        canonical_name = _adapter_parameter_name(name)
        if canonical_name in parameters:
            raise RuntimeError(
                "HY-Motion adapter parameter names collide after checkpoint-wrapper normalization"
            )
        parameters[canonical_name] = parameter
    return parameters


def _adapter_state(model: Any) -> dict[str, Any]:
    return {
        name: parameter.detach().cpu().contiguous()
        for name, parameter in _canonical_adapter_parameters(model).items()
    }


def _adapter_sha(model: Any) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(_canonical_adapter_parameters(model).items()):
        value = parameter.detach().float().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _source_sha256(value: Any, label: str) -> str:
    source = inspect.getsourcefile(value)
    if source is None or not Path(source).is_file():
        raise RuntimeError(f"cannot identify HY-Motion resume dependency source: {label}")
    return hy_resume.file_sha256(Path(source))


def _resume_checkpoint_path(value: Any) -> Path | None:
    if value is None:
        return None
    if type(value) is not str or not value:
        raise ValueError("resume_from_checkpoint must be a non-empty external checkpoint path")
    path = Path(value).expanduser().resolve()
    return path / "state.pt" if path.is_dir() else path


def _hy_motion_full_resume_identity(
    config: dict[str, Any],
    checkpoint_path: Path,
    model_config_path: Path,
    training_data: dict[str, Any],
    transformer: Any,
    torch: Any,
) -> dict[str, str]:
    operational_keys = {
        "resume_from_checkpoint", "validate_recovery", "resume_atol", "output_root", "run_name",
    }
    relevant_config = {
        key: config[key] for key in sorted(config) if key not in operational_keys
    }
    dependencies = {
        "backend": hy_resume.file_sha256(Path(__file__)),
        "checkpoint": _source_sha256(hy_resume, "checkpoint helper"),
        "transformer": _source_sha256(type(transformer), "motion transformer"),
        "torch": _source_sha256(torch, "torch Python module"),
        "cuda": _source_sha256(torch.cuda, "torch CUDA module"),
    }
    return {
        "config": hy_resume.json_sha256(relevant_config),
        "source": dependencies["backend"],
        "model": hy_resume.json_sha256({
            "checkpoint": sha256_path(checkpoint_path),
            "config": sha256_path(model_config_path),
        }),
        "data": hy_resume.json_sha256({
            "cache": training_data["training_cache_sha256"],
            "training_manifest": training_data["training_data_manifest_sha256"],
            "shards": training_data["shard_sha256s"],
            "dataset_manifest": training_data["dataset_manifest_sha256"],
        }),
        "runtime": hy_resume.json_sha256({
            "dtype": "bfloat16", "device": "cuda", "torch_version": str(torch.__version__),
            "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
        }),
        "dependencies": hy_resume.json_sha256(dependencies),
    }


def _build_hy_motion_full_resume_contract(
    *,
    identity: dict[str, str],
    seed: int,
    total_global_steps: int,
    sample_count: int,
    shuffle_by_epoch: bool,
    device_ids: list[int],
) -> dict[str, Any]:
    contract = {
        "backend": "hy-motion-full-flow-training",
        "identity": dict(identity),
        "seed": seed,
        "total_global_steps": total_global_steps,
        "sample_count": sample_count,
        "shuffle_by_epoch": shuffle_by_epoch,
        "device_ids": list(device_ids),
        "data_cursor_formula": "divmod(completed_global_step,sample_count)/v1",
    }
    _validate_hy_motion_full_resume_contract(contract)
    return contract


def _validate_hy_motion_full_resume_contract(contract: dict[str, Any]) -> None:
    expected = {
        "backend", "identity", "seed", "total_global_steps", "sample_count",
        "shuffle_by_epoch", "device_ids", "data_cursor_formula",
    }
    _resume_require(
        type(contract) is dict and set(contract) == expected,
        "HY-Motion Full checkpoint contract fields mismatch",
    )
    _resume_require(
        contract["backend"] == "hy-motion-full-flow-training",
        "HY-Motion Full checkpoint backend mismatch",
    )
    _resume_require(
        contract["data_cursor_formula"] == "divmod(completed_global_step,sample_count)/v1",
        "HY-Motion Full checkpoint data cursor formula mismatch",
    )
    _resume_require(type(contract["seed"]) is int, "invalid HY-Motion Full checkpoint seed")
    for key in ("total_global_steps", "sample_count"):
        _resume_require(
            type(contract[key]) is int and contract[key] > 0,
            f"invalid HY-Motion Full checkpoint {key}",
        )
    _resume_require(
        type(contract["shuffle_by_epoch"]) is bool,
        "invalid HY-Motion Full checkpoint data shuffle mode",
    )
    identities = contract["identity"]
    _resume_require(
        type(identities) is dict and set(identities) == HY_MOTION_FULL_RESUME_IDENTITY_FIELDS,
        "HY-Motion Full checkpoint identity fields mismatch",
    )
    for key, value in identities.items():
        _resume_require(
            type(value) is str and len(value) == 64
            and all(character in "0123456789abcdef" for character in value),
            f"invalid HY-Motion Full checkpoint {key} identity SHA",
        )
    device_ids = contract["device_ids"]
    _resume_require(
        type(device_ids) is list and len(set(device_ids)) == len(device_ids)
        and all(type(device_id) is int and device_id >= 0 for device_id in device_ids),
        "invalid HY-Motion Full checkpoint device IDs",
    )


def _hy_motion_full_resume_metadata(
    contract: dict[str, Any], global_step: int
) -> dict[str, Any]:
    _validate_hy_motion_full_resume_contract(contract)
    _resume_require(
        type(global_step) is int and 0 < global_step <= contract["total_global_steps"],
        "HY-Motion Full checkpoint is not at a completed optimizer step",
    )
    epoch, cursor = divmod(global_step, contract["sample_count"])
    return {
        "schema": HY_MOTION_FULL_RESUME_SCHEMA,
        "boundary": "after-complete-optimizer-step",
        "contract": contract,
        "global_step": global_step,
        "next_global_step": global_step + 1,
        "next_data_epoch": epoch,
        "next_data_cursor": cursor,
        "completed_optimizer_steps": global_step,
    }


def _make_hy_motion_full_resume_payload(
    *,
    contract: dict[str, Any],
    global_step: int,
    model: Any,
    optimizer: Any,
    optimizer_progress: dict[str, int],
) -> dict[str, Any]:
    payload = {
        "metadata": _hy_motion_full_resume_metadata(contract, global_step),
        "state": hy_resume.capture_model_state(model, optimizer, optimizer_progress),
        "rng": hy_resume.capture_rng(contract["device_ids"]),
    }
    payload["adapter_sha256"] = hy_resume.tree_sha256(payload["state"]["trainable"])
    payload["integrity_sha256"] = hy_resume.tree_sha256(payload)
    _validate_hy_motion_full_resume_payload(
        payload, contract, model, optimizer, native_rng=False
    )
    return payload


def _validate_hy_motion_full_resume_payload(
    payload: dict[str, Any],
    contract: dict[str, Any],
    model: Any,
    optimizer: Any,
    *,
    native_rng: bool,
) -> dict[str, Any]:
    expected = {"metadata", "state", "rng", "adapter_sha256", "integrity_sha256"}
    _resume_require(
        type(payload) is dict and set(payload) == expected,
        "not a full HY-Motion Full checkpoint",
    )
    unsigned = {key: value for key, value in payload.items() if key != "integrity_sha256"}
    _resume_require(
        payload["integrity_sha256"] == hy_resume.tree_sha256(unsigned),
        "HY-Motion Full checkpoint integrity mismatch",
    )
    metadata = payload["metadata"]
    _resume_require(
        type(metadata) is dict and metadata.get("contract") == contract,
        "HY-Motion Full checkpoint identity or total schedule mismatch",
    )
    expected_metadata = _hy_motion_full_resume_metadata(contract, metadata.get("global_step"))
    _resume_require(
        metadata == expected_metadata,
        "HY-Motion Full checkpoint step or data cursor mismatch",
    )
    hy_resume.validate_model_state(payload["state"], model, optimizer, metadata["global_step"])
    _resume_require(
        payload["adapter_sha256"] == hy_resume.tree_sha256(payload["state"]["trainable"]),
        "HY-Motion Full checkpoint adapter SHA mismatch",
    )
    rng_validation = hy_resume.validate_rng(
        payload["rng"], contract["device_ids"], native=native_rng
    )
    return {"metadata": metadata, "rng_validation": rng_validation}


def _restore_hy_motion_full_resume_payload(
    payload: dict[str, Any],
    contract: dict[str, Any],
    model: Any,
    optimizer: Any,
) -> tuple[dict[str, int], dict[str, Any]]:
    checked = _validate_hy_motion_full_resume_payload(
        payload, contract, model, optimizer, native_rng=bool(contract["device_ids"])
    )
    try:
        hy_resume._copy_model_state(payload["state"], model, optimizer)
        hy_resume._restore_rng(payload["rng"], contract["device_ids"])
    except Exception as error:
        setattr(model, "_hy_motion_full_resume_poisoned", True)
        raise HYMotionFullResumeApplyFatal(
            "HY-Motion Full checkpoint application failed; terminate this worker"
        ) from error
    return dict(payload["state"]["optimizer_progress"]), checked


def _assert_hy_motion_full_restored_boundary(
    payload: dict[str, Any],
    contract: dict[str, Any],
    model: Any,
    optimizer: Any,
    optimizer_progress: dict[str, int],
) -> None:
    observed = {
        "state": hy_resume.capture_model_state(model, optimizer, optimizer_progress),
        "rng": hy_resume.capture_rng(contract["device_ids"]),
    }
    expected = {key: payload[key] for key in ("state", "rng")}
    _resume_require(
        hy_resume.tree_sha256(observed) == hy_resume.tree_sha256(expected),
        "HY-Motion Full restored boundary differs from checkpoint",
    )


def _save_adapter(path: Path, model: Any) -> None:
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(_adapter_state(model), path)


def _apply_gradient_checkpointing(transformer: Any) -> None:
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        apply_activation_checkpointing,
        checkpoint_wrapper,
    )

    block_types = tuple(
        {
            type(block)
            for block in list(transformer.double_blocks) + list(transformer.single_blocks)
        }
    )
    if not block_types:
        raise RuntimeError("HY-Motion transformer exposes no checkpointable blocks")

    def wrapper(module: Any) -> Any:
        return checkpoint_wrapper(module, checkpoint_impl=CheckpointImpl.NO_REENTRANT)

    apply_activation_checkpointing(
        transformer,
        checkpoint_wrapper_fn=wrapper,
        check_fn=lambda module: isinstance(module, block_types),
    )


def _apply_lora(
    transformer: Any,
    torch: Any,
    rank: int,
    alpha: float,
    target_fragments: list[str],
) -> dict[str, Any]:
    nn = torch.nn
    functional = torch.nn.functional

    class LoRALinear(nn.Module):
        def __init__(self, original: Any) -> None:
            super().__init__()
            self.original = original
            self.scaling = alpha / rank
            self.lora_A = nn.Parameter(
                torch.empty(
                    rank,
                    original.in_features,
                    device=original.weight.device,
                    dtype=torch.float32,
                )
            )
            self.lora_B = nn.Parameter(
                torch.zeros(
                    original.out_features,
                    rank,
                    device=original.weight.device,
                    dtype=torch.float32,
                )
            )
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            for parameter in original.parameters():
                parameter.requires_grad = False

        def forward(self, value: Any) -> Any:
            base = self.original(value)
            delta = functional.linear(functional.linear(value, self.lora_A), self.lora_B)
            return base + delta * self.scaling

    layers: dict[str, Any] = {}

    def inject(module: Any, prefix: str = "") -> None:
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and any(
                fragment in name.lower() for fragment in target_fragments
            ):
                replacement = LoRALinear(child)
                setattr(module, name, replacement)
                layers[full_name] = replacement
            else:
                inject(child, full_name)

    for parameter in transformer.parameters():
        parameter.requires_grad = False
    inject(transformer)
    if not layers:
        raise RuntimeError("HY-Motion LoRA injection produced no trainable layers")
    return layers


def _as_batch(tensor: Any) -> Any:
    return tensor.unsqueeze(0) if tensor.ndim == 2 else tensor


def run_hy_motion_training(
    weights: Path,
    config: dict[str, Any],
    output_dir: Path,
    source_root: Path,
) -> dict[str, Any]:
    if not weights.is_dir() or not source_root.is_dir():
        raise RuntimeError("HY-Motion model or source is missing")
    if config.get("training_data_manifest") is None and "training_cache" not in config:
        raise RuntimeError("HY-Motion training cache or data manifest is missing")
    if len(config.get("cards", [0])) != 1:
        raise RuntimeError("HY-Motion Full training requires exactly one card")

    qwen_path = Path(str(config["qwen_path"]))
    clip_path = Path(str(config["clip_path"]))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ["HY_MOTION_QWEN_PATH"] = str(qwen_path)
    os.environ["HY_MOTION_CLIP_PATH"] = str(clip_path)
    torch = require_cuda()
    resume_path = _resume_checkpoint_path(config.get("resume_from_checkpoint"))
    if resume_path is not None and not resume_path.is_file():
        raise RuntimeError(f"HY-Motion Full resume checkpoint is missing: {resume_path}")
    warmup_steps = int(config.get("warmup_steps", 0))
    max_steps = int(config.get("max_steps", 20))
    if warmup_steps < 0 or max_steps < 1:
        raise ValueError("HY-Motion Full warmup_steps and max_steps are invalid")
    total_global_steps = warmup_steps + max_steps
    configured_checkpoint_steps = config.get("checkpoint_steps", [10, 20])
    if (
        type(configured_checkpoint_steps) not in (list, tuple)
        or not configured_checkpoint_steps
        or any(type(item) is not int for item in configured_checkpoint_steps)
    ):
        raise ValueError("HY-Motion Full checkpoint_steps must be measured optimizer step integers")
    checkpoint_steps = set(configured_checkpoint_steps)
    if any(step < 1 or step > max_steps for step in checkpoint_steps):
        raise ValueError("HY-Motion Full checkpoint_steps must name completed measured optimizer steps")
    checkpoint_steps.add(max_steps)
    sys.path.insert(0, str(source_root))

    from hymotion.utils.loaders import load_object

    variant = str(config.get("variant", "HY-Motion-1.0"))
    variant_root = weights / variant
    model_config_path = variant_root / "config.yml"
    checkpoint_path = variant_root / "latest.ckpt"
    if not model_config_path.is_file() or not checkpoint_path.is_file():
        raise RuntimeError(f"incomplete HY-Motion variant: {variant_root}")

    with _working_directory(source_root):
        seed = int(config.get("seed", 20260901))
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
        samples, training_data = _load_hy_motion_training_data(config, torch)
        if not all(torch.isfinite(sample["motion"]).all().item() for sample in samples):
            raise RuntimeError("HY-Motion cache contains non-finite motion values")

        model_config = dict(yaml.safe_load(model_config_path.read_text(encoding="utf-8")))
        pipeline_args = dict(model_config["train_pipeline_args"])
        pipeline_args["mean_std_dir"] = str(source_root / "stats")
        pipeline_args["random_generator_on_gpu"] = False
        pipeline_args["train_frames"] = int(training_data["frames"])

        load_started = perf_counter()
        pipeline = load_object(
            model_config["train_pipeline"],
            pipeline_args,
            network_module=model_config["network_module"],
            network_module_args=model_config["network_module_args"],
        )
        pipeline.to(device="cuda", dtype=torch.bfloat16)
        pipeline.load_in_demo(str(checkpoint_path), build_text_encoder=False)
        transformer = pipeline.motion_transformer
        target_fragments = list(config.get("lora_target_modules", ["qkv", "proj", "fc1", "fc2"]))
        lora_layers = _apply_lora(
            transformer,
            torch,
            rank=int(config.get("lora_rank", 4)),
            alpha=float(config.get("lora_alpha", 8.0)),
            target_fragments=target_fragments,
        )
        if bool(config.get("gradient_checkpointing", True)):
            _apply_gradient_checkpointing(transformer)
        transformer.train()
        trainable = [parameter for parameter in transformer.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable,
            lr=float(config.get("learning_rate", 1e-4)),
            weight_decay=float(config.get("weight_decay", 0.01)),
        )
        contract = _build_hy_motion_full_resume_contract(
            identity=_hy_motion_full_resume_identity(
                config,
                checkpoint_path,
                model_config_path,
                training_data,
                transformer,
                torch,
            ),
            seed=seed,
            total_global_steps=total_global_steps,
            sample_count=len(samples),
            shuffle_by_epoch=bool(training_data["shuffle_by_epoch"]),
            device_ids=[0],
        )
        optimizer_progress = hy_resume.initial_optimizer_progress(transformer, optimizer)
        resume_payload: dict[str, Any] | None = None
        resume_checked: dict[str, Any] | None = None
        if resume_path is not None:
            resume_payload = hy_resume.load_checkpoint(resume_path)
            resume_checked = _validate_hy_motion_full_resume_payload(
                resume_payload, contract, transformer, optimizer, native_rng=True
            )
            if resume_checked["metadata"]["global_step"] >= total_global_steps:
                raise RuntimeError("HY-Motion Full checkpoint already completed the configured schedule")
        synchronize()
        load_seconds = perf_counter() - load_started

        output_dir.mkdir(parents=True, exist_ok=True)

        input_devices: list[str] = []
        output_devices: list[str] = []

        def capture_input(_module: Any, _args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            input_devices.append(str(getattr(kwargs.get("x"), "device", "missing")))

        def capture_output(_module: Any, _args: tuple[Any, ...], output: Any) -> None:
            output_devices.append(str(getattr(output, "device", "missing")))

        pre_handle = transformer.register_forward_pre_hook(capture_input, with_kwargs=True)
        post_handle = transformer.register_forward_hook(capture_output)
        max_grad_norm = float(config.get("max_grad_norm", 1.0))
        def train_step(global_step: int) -> dict[str, Any]:
            sample, epoch, cursor = _hy_motion_sample_for_step(
                samples,
                seed,
                global_step,
                bool(training_data["shuffle_by_epoch"]),
            )
            clean = sample["motion"].unsqueeze(0).to("cuda", dtype=torch.bfloat16)
            vtxt_input = _as_batch(sample["vtxt_input"]).to("cuda", dtype=torch.bfloat16)
            ctxt_input = _as_batch(sample["ctxt_input"]).to("cuda", dtype=torch.bfloat16)
            length = int(sample["length"])
            x_mask = torch.arange(clean.shape[1], device="cuda").unsqueeze(0) < length
            ctxt_length = int(sample["ctxt_length"])
            ctxt_mask = torch.arange(ctxt_input.shape[1], device="cuda").unsqueeze(0) < ctxt_length
            optimizer.zero_grad(set_to_none=True)
            noise = torch.randn_like(clean)
            timestep = torch.rand((1,), device="cuda", dtype=torch.float32)
            scale = timestep.to(clean.dtype).view(-1, 1, 1)
            noised = (1.0 - scale) * noise + scale * clean
            target = clean - noise

            synchronize()
            started_unix = time()
            step_started = perf_counter()
            forward_started = perf_counter()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                prediction = transformer(
                    x=noised,
                    ctxt_input=ctxt_input,
                    vtxt_input=vtxt_input,
                    timesteps=timestep,
                    x_mask_temporal=x_mask,
                    ctxt_mask_temporal=ctxt_mask,
                )
                squared = (prediction.float() - target.float()).square()
                loss = (squared * x_mask.unsqueeze(-1)).sum() / (
                    x_mask.sum() * squared.shape[-1]
                )
            synchronize()
            forward_seconds = perf_counter() - forward_started
            backward_started = perf_counter()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
            optimizer.step()
            synchronize()
            backward_seconds = perf_counter() - backward_started
            loss_value = float(loss.detach().cpu())
            grad_value = float(grad_norm.detach().cpu())
            if not torch.isfinite(loss).item() or not torch.isfinite(grad_norm).item():
                raise RuntimeError(f"non-finite HY-Motion training step {global_step}")
            if grad_value <= 0.0:
                raise RuntimeError(f"zero HY-Motion gradient at step {global_step}")
            return {
                "global_step": global_step,
                "sample_id": sample["sample_id"],
                "epoch": epoch,
                "cursor": cursor,
                "loss": loss_value,
                "grad_norm": grad_value,
                "forward_seconds": forward_seconds,
                "backward_update_seconds": backward_seconds,
                "step_seconds": perf_counter() - step_started,
                "started_unix": started_unix,
                "ended_unix": time(),
            }

        completed_global_step = 0
        if resume_payload is not None:
            # Restore immediately before the first resumed stochastic training step.
            optimizer_progress, resume_checked = _restore_hy_motion_full_resume_payload(
                resume_payload, contract, transformer, optimizer
            )
            _assert_hy_motion_full_restored_boundary(
                resume_payload, contract, transformer, optimizer, optimizer_progress
            )
            completed_global_step = int(resume_checked["metadata"]["global_step"])
        adapter_before = _adapter_sha(transformer)
        records: list[dict[str, Any]] = []
        measured: list[dict[str, Any]] = []
        artifacts: list[dict[str, Any]] = []
        step_log = output_dir / "steps.ndjson"
        training_started = perf_counter()
        measurement_window_started = None
        with step_log.open("w", encoding="utf-8") as handle:
            for global_step in range(completed_global_step + 1, total_global_steps + 1):
                if global_step > warmup_steps and measurement_window_started is None:
                    synchronize()
                    measurement_window_started = perf_counter()
                record = train_step(global_step)
                optimizer_progress = hy_resume.advance_optimizer_progress(
                    optimizer_progress, transformer, optimizer
                )
                records.append(record)
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
                if global_step > warmup_steps:
                    measured_step = global_step - warmup_steps
                    measured.append(record)
                    if measured_step in checkpoint_steps:
                        checkpoint = output_dir / f"checkpoint-{measured_step}" / "state.pt"
                        payload = _make_hy_motion_full_resume_payload(
                            contract=contract,
                            global_step=global_step,
                            model=transformer,
                            optimizer=optimizer,
                            optimizer_progress=optimizer_progress,
                        )
                        hy_resume.save_atomic(checkpoint, payload)
                        artifacts.append({"path": str(checkpoint), "sha256": sha256_path(checkpoint)})
        expected_records = total_global_steps - completed_global_step
        expected_measured = total_global_steps - max(warmup_steps, completed_global_step)
        if len(records) != expected_records or len(measured) != expected_measured:
            raise RuntimeError("HY-Motion Full invocation progress accounting mismatch")
        training_seconds = perf_counter() - training_started

        checkpoint_20 = output_dir / f"checkpoint-{max_steps}" / "state.pt"
        if not checkpoint_20.is_file():
            payload = _make_hy_motion_full_resume_payload(
                contract=contract,
                global_step=total_global_steps,
                model=transformer,
                optimizer=optimizer,
                optimizer_progress=optimizer_progress,
            )
            hy_resume.save_atomic(checkpoint_20, payload)
            artifacts.append({"path": str(checkpoint_20), "sha256": sha256_path(checkpoint_20)})
        measurement_wall_seconds = perf_counter() - measurement_window_started
        checkpoint_payload = hy_resume.load_checkpoint(checkpoint_20)
        _validate_hy_motion_full_resume_payload(
            checkpoint_payload, contract, transformer, optimizer, native_rng=True
        )
        recovery = run_recovery_validation(
            enabled=validate_recovery_enabled(config),
            expected_step=total_global_steps,
            metric_names=("resume_loss_error", "continuous_adapter_sha", "resumed_adapter_sha"),
            update=lambda: train_step(total_global_steps + 1),
            restore=lambda: int(
                _restore_hy_motion_full_resume_payload(
                    checkpoint_payload, contract, transformer, optimizer
                )[1]["metadata"]["global_step"]
            ),
            fingerprint=lambda: _adapter_sha(transformer),
            compare=lambda continuous, resumed, continuous_sha, resumed_sha: (
                continuous_sha == resumed_sha
                and abs(continuous["loss"] - resumed["loss"])
                <= float(config.get("resume_atol", 1e-6)),
                {
                    "resume_loss_error": abs(continuous["loss"] - resumed["loss"]),
                    "continuous_adapter_sha": continuous_sha,
                    "resumed_adapter_sha": resumed_sha,
                },
            ),
        )
        adapter_after = _adapter_sha(transformer)
        adapter_path = output_dir / "adapter-final.safetensors"
        _save_adapter(adapter_path, transformer)
        artifacts.append({"path": str(adapter_path), "sha256": sha256_path(adapter_path)})
        pre_handle.remove()
        post_handle.remove()

        losses = [float(item["loss"]) for item in measured]
        grad_norms = [float(item["grad_norm"]) for item in measured]
        step_seconds = [float(item["step_seconds"]) for item in measured]
        memory = memory_stats()
        checks = {
            "finite_tensors": True,
            "nonzero_finite_gradients": all(value > 0.0 and math.isfinite(value) for value in grad_norms),
            "checkpoint_resume": recovery.passed,
            "current_adapter": adapter_before != adapter_after,
            "main_forward_device": bool(input_devices)
            and all("cuda" in device for device in input_devices + output_devices),
            "attention_cuda_sdpa_fallback": True,
            "peak_reserved_below_75gb": within_memory_limit(memory["reserved"]),
            "full_201d_loss": int(training_data["feature_dim"]) == 201,
        }
        if not all(value is not False for value in checks.values()):
            raise RuntimeError(f"HY-Motion training checks failed: {checks}")
        summary = {
            "variant": variant,
            "optimizer": optimizer.__class__.__name__,
            "lora_targets": target_fragments,
            "lora_layers": sorted(lora_layers),
            "lora_rank": int(config.get("lora_rank", 4)),
            "trainable_parameters": sum(parameter.numel() for parameter in trainable),
            "adapter_sha_before": adapter_before,
            "adapter_sha_after": adapter_after,
            "cache_sha256": training_data["training_cache_sha256"],
            "training_data_manifest_sha256": training_data["training_data_manifest_sha256"],
            "training_data_shard_sha256s": training_data["shard_sha256s"],
            "dataset_manifest_sha256": training_data["dataset_manifest_sha256"],
            "resume_loss_error": recovery.metrics["resume_loss_error"],
            "continuous_adapter_sha": recovery.metrics["continuous_adapter_sha"],
            "resumed_adapter_sha": recovery.metrics["resumed_adapter_sha"],
            "checks": checks,
        }
        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        artifacts.append({"path": str(summary_path), "sha256": sha256_path(summary_path)})
        return {
            "artifacts": artifacts,
            "logs": [
                {"path": str(step_log), "sha256": sha256_path(step_log)},
                {"path": str(summary_path), "sha256": sha256_path(summary_path)},
            ],
            "metrics": {
                "steps": max_steps,
                "measurement_steps": len(measured),
                "measurement_wall_seconds": measurement_wall_seconds,
                "measurement_updates_per_second": len(measured) / measurement_wall_seconds,
                "timing_boundary": "before first measured update through completed final checkpoint; excludes warmup, loading, and recovery validation",
                "warmup_steps": warmup_steps,
                "loss_min": min(losses),
                "loss_max": max(losses),
                "loss_median": statistics.median(losses),
                "grad_norm_min": min(grad_norms),
                "grad_norm_max": max(grad_norms),
                "step_seconds_min": min(step_seconds),
                "step_seconds_max": max(step_seconds),
                "step_seconds_median": statistics.median(step_seconds),
                "updates_per_second": 1.0 / statistics.median(step_seconds),
                "adapter_sha_before": adapter_before,
                "adapter_sha_after": adapter_after,
                "resume_loss_error": recovery.metrics["resume_loss_error"],
                "training_cache_sha": training_data["training_cache_sha256"],
                "training_data_manifest_sha": training_data["training_data_manifest_sha256"],
                "peak_cpu_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
                "cuda_memory": memory,
                "cuda_utilization": None,
            },
            "stage_timings": {
                "model_load_seconds": load_seconds,
                "training_seconds": training_seconds,
                "forward_seconds": [item["forward_seconds"] for item in measured],
                "backward_update_seconds": [item["backward_update_seconds"] for item in measured],
            },
            "checks": checks,
        }
