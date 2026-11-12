from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import random
import statistics
import sys
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter, time
from types import SimpleNamespace
from typing import Any

import numpy as np

from ..manifest import sha256_path
from ..recovery_validation import run_recovery_validation, validate_recovery_enabled
from ..runtime import autocast_bf16, memory_stats, require_cuda, synchronize
from .. import hy_resume_checkpoint as resume_common
from .. import runtime as runtime_helpers


VGGT_CHECKPOINT_SCHEMA = "vggt-full-optimizer-step-v2"
VGGT_TRAINING_SCOPES = {"camera_head", "camera_backbone"}
_VGGT_IDENTITY_FIELDS = {
    "workload",
    "frozen_model",
    "data",
    "cache",
    "optimizer",
    "sources",
}
_VGGT_CHECKPOINT_CPU_THREADS_LOCK = threading.RLock()


class VGGTCheckpointValidationError(ValueError):
    """The VGGT checkpoint is incomplete or incompatible with this run."""


class VGGTResumeApplyFatal(RuntimeError):
    """Checkpoint application failed after validation; do not keep training."""


def _vggt_require(condition: bool, message: str) -> None:
    if not condition:
        raise VGGTCheckpointValidationError(message)


def _vggt_checkpoint_cpu_threads_enabled(config: dict[str, Any]) -> bool:
    value = config.get("checkpoint_cpu_threads_16", False)
    if type(value) is not bool:
        raise ValueError("VGGT checkpoint_cpu_threads_16 must be a boolean")
    return value


def _vggt_training_preflight(config: dict[str, Any]) -> tuple[str, bool]:
    scope = config.get("training_scope", "camera_head")
    if type(scope) is not str or scope not in VGGT_TRAINING_SCOPES:
        raise ValueError(
            "VGGT training_scope must be 'camera_head' or 'camera_backbone'"
        )
    cache_enabled = config.get("cache_frozen_aggregator_features", False)
    if type(cache_enabled) is not bool:
        raise ValueError("VGGT cache_frozen_aggregator_features must be a boolean")
    if scope == "camera_backbone" and cache_enabled:
        raise ValueError(
            "VGGT camera_backbone training cannot cache trainable aggregator features"
        )
    cards = config.get("cards", [0])
    if (
        type(cards) not in (list, tuple)
        or len(cards) != 1
        or type(cards[0]) is not int
        or cards[0] < 0
    ):
        raise RuntimeError("VGGT training requires exactly one card")
    return scope, cache_enabled


@contextmanager
def _vggt_checkpoint_cpu_thread_limit(torch: Any, enabled: bool):
    if not enabled:
        yield
        return
    with _VGGT_CHECKPOINT_CPU_THREADS_LOCK:
        original = torch.get_num_threads()
        target = min(original, 16)
        if target == original:
            yield
            return
        try:
            torch.set_num_threads(target)
            if torch.get_num_threads() != target:
                raise RuntimeError("VGGT checkpoint CPU thread limit did not take effect")
            yield
        finally:
            # Keep a body error as the restore failure's context if restoration also fails.
            torch.set_num_threads(original)
            if torch.get_num_threads() != original:
                raise RuntimeError("VGGT checkpoint CPU thread limit was not restored")


def _vggt_contract(
    *,
    identity: dict[str, str],
    seed: int,
    warmup_steps: int,
    max_steps: int,
    view_order: list[str],
    device_ids: list[int],
) -> dict[str, Any]:
    contract = {
        "backend": "vggt-camera-training",
        "identity": dict(identity),
        "seed": seed,
        "warmup_steps": warmup_steps,
        "max_steps": max_steps,
        "view_order": list(view_order),
        "device_ids": list(device_ids),
    }
    _validate_vggt_contract(contract)
    return contract


def _validate_vggt_contract(contract: dict[str, Any]) -> None:
    expected = {
        "backend",
        "identity",
        "seed",
        "warmup_steps",
        "max_steps",
        "view_order",
        "device_ids",
    }
    _vggt_require(type(contract) is dict and set(contract) == expected, "checkpoint contract fields mismatch")
    _vggt_require(contract["backend"] == "vggt-camera-training", "checkpoint backend mismatch")
    for key in ("seed", "warmup_steps", "max_steps"):
        minimum = 1 if key == "max_steps" else 0
        _vggt_require(type(contract[key]) is int and contract[key] >= minimum, f"invalid {key}")
    identity = contract["identity"]
    _vggt_require(type(identity) is dict and set(identity) == _VGGT_IDENTITY_FIELDS, "identity fields mismatch")
    for key, value in identity.items():
        _vggt_require(
            type(value) is str
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value),
            f"invalid {key} identity SHA",
        )
    _vggt_require(
        type(contract["view_order"]) is list
        and contract["view_order"]
        and all(type(item) is str and item for item in contract["view_order"]),
        "invalid view order",
    )
    _vggt_require(
        type(contract["device_ids"]) is list
        and len(set(contract["device_ids"])) == len(contract["device_ids"])
        and all(type(item) is int and item >= 0 for item in contract["device_ids"]),
        "invalid device IDs",
    )


def _vggt_boundary_metadata(contract: dict[str, Any], global_optimizer_step: int) -> dict[str, Any]:
    _validate_vggt_contract(contract)
    first_measured_step = contract["warmup_steps"] + 1
    final_global_step = contract["warmup_steps"] + contract["max_steps"]
    _vggt_require(
        type(global_optimizer_step) is int and first_measured_step <= global_optimizer_step <= final_global_step,
        "checkpoint is not at a completed measured optimizer step",
    )
    measured_step = global_optimizer_step - contract["warmup_steps"]
    return {
        "schema": VGGT_CHECKPOINT_SCHEMA,
        "boundary": "after-complete-optimizer-update",
        "contract": contract,
        "global_optimizer_step": global_optimizer_step,
        "completed_measured_steps": measured_step,
        "next_global_optimizer_step": global_optimizer_step + 1,
        "next_batch_cursor": global_optimizer_step % len(contract["view_order"]),
    }


def _vggt_next_coordinate(metadata: dict[str, Any]) -> dict[str, int] | None:
    contract = metadata["contract"]
    if metadata["completed_measured_steps"] == contract["max_steps"]:
        return None
    return {
        "global_optimizer_step": metadata["next_global_optimizer_step"],
        "batch_cursor": metadata["next_batch_cursor"],
    }


def _vggt_capture_rng(torch: Any, device_ids: list[int]) -> dict[str, Any]:
    return resume_common.cpu_tree(resume_common.capture_rng(device_ids))


def _vggt_validate_rng(
    state: dict[str, Any], torch: Any, device_ids: list[int], *, native: bool = True
) -> None:
    try:
        resume_common.validate_rng(state, device_ids, native=native)
    except Exception as error:
        raise VGGTCheckpointValidationError("invalid VGGT RNG state") from error


def _vggt_restore_rng(state: dict[str, Any], torch: Any, device_ids: list[int]) -> None:
    resume_common._restore_rng(state, device_ids)


def _vggt_validate_state(
    state: dict[str, Any],
    module: Any,
    optimizer: Any,
    global_optimizer_step: int,
    torch: Any,
    state_key: str = "camera_head",
) -> None:
    _vggt_require(
        type(state) is dict and set(state) == {state_key, "optimizer", "optimizer_placements"},
        "checkpoint state fields mismatch",
    )
    saved_model = state[state_key]
    live_model = module.state_dict()
    _vggt_require(
        type(saved_model) is dict and set(saved_model) == set(live_model),
        f"{state_key} state keys mismatch",
    )
    for name, value in saved_model.items():
        expected = live_model[name]
        _vggt_require(
            isinstance(value, torch.Tensor)
            and value.device.type == "cpu"
            and value.shape == expected.shape
            and value.dtype == expected.dtype
            and not value.requires_grad
            and bool(torch.isfinite(value).all()),
            f"invalid {state_key} tensor: {name}",
        )
    saved_optimizer = state["optimizer"]
    live_optimizer = optimizer.state_dict()
    _vggt_require(type(saved_optimizer) is dict and set(saved_optimizer) == {"state", "param_groups"}, "optimizer state fields mismatch")
    _vggt_require(type(saved_optimizer["param_groups"]) is list and len(saved_optimizer["param_groups"]) == len(live_optimizer["param_groups"]), "optimizer group count mismatch")
    parameter_by_id: dict[int, Any] = {}
    for saved_group, live_group, live_parameters in zip(saved_optimizer["param_groups"], live_optimizer["param_groups"], optimizer.param_groups):
        _vggt_require(type(saved_group) is dict and set(saved_group) == set(live_group), "optimizer group fields mismatch")
        _vggt_require(saved_group["params"] == live_group["params"], "optimizer parameter order mismatch")
        _vggt_require(
            {key: value for key, value in saved_group.items() if key != "params"}
            == {key: value for key, value in live_group.items() if key != "params"},
            "optimizer identity mismatch",
        )
        for parameter_id, parameter in zip(saved_group["params"], live_parameters["params"]):
            parameter_by_id[parameter_id] = parameter
    saved_slots = saved_optimizer["state"]
    _vggt_require(type(saved_slots) is dict and set(saved_slots) == set(parameter_by_id), "optimizer slots mismatch")
    placements = state["optimizer_placements"]
    _vggt_require(type(placements) is dict and set(placements) == set(saved_slots), "optimizer placement entries mismatch")
    for parameter_id, slots in saved_slots.items():
        _vggt_require(type(slots) is dict and set(slots) == {"step", "exp_avg", "exp_avg_sq"}, "AdamW slot fields mismatch")
        parameter = parameter_by_id[parameter_id]
        slot_placements = placements[parameter_id]
        _vggt_require(type(slot_placements) is dict and set(slot_placements) == set(slots), "optimizer placement fields mismatch")
        for key, placement in slot_placements.items():
            allowed = {"cpu", str(parameter.device)} if key == "step" else {str(parameter.device)}
            _vggt_require(type(placement) is str and placement in allowed, f"invalid AdamW {key} placement")
        step = slots["step"]
        _vggt_require(
            isinstance(step, torch.Tensor)
            and step.device.type == "cpu"
            and step.shape == torch.Size([])
            and step.dtype == torch.float32
            and not step.requires_grad,
            "invalid AdamW step",
        )
        step_value = float(step.item())
        _vggt_require(
            math.isfinite(step_value)
            and step_value.is_integer()
            and int(step_value) == global_optimizer_step,
            "AdamW step differs from global optimizer step",
        )
        for key in ("exp_avg", "exp_avg_sq"):
            value = slots[key]
            _vggt_require(
                isinstance(value, torch.Tensor)
                and value.device.type == "cpu"
                and value.shape == parameter.shape
                and value.dtype == parameter.dtype
                and not value.requires_grad
                and bool(torch.isfinite(value).all()),
                f"invalid AdamW {key}",
            )
        _vggt_require(bool((slots["exp_avg_sq"] >= 0).all()), "negative AdamW second moment")


def _vggt_capture_runtime_state(
    *,
    contract: dict[str, Any],
    module: Any,
    optimizer: Any,
    torch: Any,
    state_key: str = "camera_head",
) -> dict[str, Any]:
    optimizer_state = optimizer.state_dict()
    placements = {}
    for saved_group, live_group in zip(optimizer_state["param_groups"], optimizer.param_groups):
        for parameter_id, parameter in zip(saved_group["params"], live_group["params"]):
            slots = optimizer.state.get(parameter, {})
            if slots:
                placements[parameter_id] = {key: str(value.device) for key, value in slots.items()}
    return {
        "state": {
            state_key: resume_common.cpu_tree(module.state_dict()),
            "optimizer": resume_common.cpu_tree(optimizer_state),
            "optimizer_placements": placements,
        },
        "rng": _vggt_capture_rng(torch, contract["device_ids"]),
    }


def _vggt_capture_boundary(
    *,
    contract: dict[str, Any],
    global_optimizer_step: int,
    module: Any,
    optimizer: Any,
    torch: Any,
    state_key: str = "camera_head",
) -> dict[str, Any]:
    return {
        "metadata": _vggt_boundary_metadata(contract, global_optimizer_step),
        **_vggt_capture_runtime_state(
            contract=contract,
            module=module,
            optimizer=optimizer,
            torch=torch,
            state_key=state_key,
        ),
    }


def _vggt_boundary_sha256(
    *,
    contract: dict[str, Any],
    global_optimizer_step: int,
    module: Any,
    optimizer: Any,
    torch: Any,
    state_key: str = "camera_head",
) -> str:
    return resume_common.tree_sha256(
        _vggt_capture_boundary(
            contract=contract,
            global_optimizer_step=global_optimizer_step,
            module=module,
            optimizer=optimizer,
            torch=torch,
            state_key=state_key,
        )
    )


def _vggt_recovery_boundary_sha256(
    *,
    contract: dict[str, Any],
    global_optimizer_step: int,
    module: Any,
    optimizer: Any,
    torch: Any,
    state_key: str = "camera_head",
) -> str:
    return resume_common.tree_sha256(
        {
            "global_optimizer_step": global_optimizer_step,
            "next_batch_cursor": global_optimizer_step % len(contract["view_order"]),
            **_vggt_capture_runtime_state(
                contract=contract,
                module=module,
                optimizer=optimizer,
                torch=torch,
                state_key=state_key,
            ),
        }
    )


def _vggt_make_checkpoint(
    *,
    contract: dict[str, Any],
    global_optimizer_step: int,
    module: Any,
    optimizer: Any,
    torch: Any,
    checkpoint_cpu_threads_16: bool = False,
    state_key: str = "camera_head",
) -> dict[str, Any]:
    payload = _vggt_capture_boundary(
        contract=contract,
        global_optimizer_step=global_optimizer_step,
        module=module,
        optimizer=optimizer,
        torch=torch,
        state_key=state_key,
    )
    with _vggt_checkpoint_cpu_thread_limit(torch, checkpoint_cpu_threads_16):
        payload["integrity_sha256"] = resume_common.tree_sha256(payload)
    _vggt_validate_checkpoint(
        payload,
        contract,
        module,
        optimizer,
        torch,
        native_rng=False,
        checkpoint_cpu_threads_16=checkpoint_cpu_threads_16,
        state_key=state_key,
    )
    return payload


def _vggt_validate_checkpoint(
    payload: dict[str, Any], contract: dict[str, Any], module: Any, optimizer: Any, torch: Any,
    *,
    native_rng: bool = True,
    checkpoint_cpu_threads_16: bool = False,
    state_key: str = "camera_head",
) -> dict[str, Any]:
    _vggt_require(type(payload) is dict and set(payload) == {"metadata", "state", "rng", "integrity_sha256"}, "not a full VGGT checkpoint")
    unsigned = {key: value for key, value in payload.items() if key != "integrity_sha256"}
    with _vggt_checkpoint_cpu_thread_limit(torch, checkpoint_cpu_threads_16):
        _vggt_require(payload["integrity_sha256"] == resume_common.tree_sha256(unsigned), "checkpoint integrity mismatch")
        metadata = payload["metadata"]
        _vggt_require(type(metadata) is dict and metadata.get("contract") == contract, "checkpoint identity or schedule mismatch")
        _vggt_require(
            metadata == _vggt_boundary_metadata(contract, metadata.get("global_optimizer_step")),
            "checkpoint global step or batch cursor mismatch",
        )
        _vggt_validate_state(
            payload["state"],
            module,
            optimizer,
            metadata["global_optimizer_step"],
            torch,
            state_key,
        )
    _vggt_validate_rng(payload["rng"], torch, contract["device_ids"], native=native_rng)
    return {"metadata": metadata, "next": _vggt_next_coordinate(metadata)}


def _vggt_save_checkpoint(path: Path, payload: dict[str, Any], torch: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _vggt_restore_checkpoint(
    payload: dict[str, Any],
    contract: dict[str, Any],
    module: Any,
    optimizer: Any,
    torch: Any,
    *,
    checkpoint_cpu_threads_16: bool = False,
    state_key: str = "camera_head",
) -> dict[str, Any]:
    checked = _vggt_validate_checkpoint(
        payload,
        contract,
        module,
        optimizer,
        torch,
        checkpoint_cpu_threads_16=checkpoint_cpu_threads_16,
        state_key=state_key,
    )
    try:
        # Optimizer.load_state_dict may retain tensor references from its input.
        restored_state = resume_common.cpu_tree(payload["state"])
        restored_rng = resume_common.cpu_tree(payload["rng"])
        module.load_state_dict(restored_state[state_key], strict=True)
        optimizer.load_state_dict(restored_state["optimizer"])
        _restore_optimizer_placements(
            optimizer, restored_state["optimizer"], restored_state["optimizer_placements"]
        )
        _vggt_restore_rng(restored_rng, torch, contract["device_ids"])
    except Exception as error:
        setattr(module, "_vggt_resume_poisoned", True)
        raise VGGTResumeApplyFatal("VGGT checkpoint application failed; terminate this worker") from error
    return checked


def _vggt_assert_restored_boundary(
    payload: dict[str, Any],
    contract: dict[str, Any],
    module: Any,
    optimizer: Any,
    torch: Any,
    *,
    checkpoint_cpu_threads_16: bool = False,
    state_key: str = "camera_head",
) -> None:
    metadata = payload["metadata"]
    expected = {key: payload[key] for key in ("metadata", "state", "rng")}
    observed = _vggt_capture_boundary(
        contract=contract,
        global_optimizer_step=metadata["global_optimizer_step"],
        module=module,
        optimizer=optimizer,
        torch=torch,
        state_key=state_key,
    )
    with _vggt_checkpoint_cpu_thread_limit(torch, checkpoint_cpu_threads_16):
        _vggt_require(
            resume_common.tree_sha256(observed) == resume_common.tree_sha256(expected),
            "restored VGGT boundary differs from checkpoint",
        )


def _module_sha(module: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().float().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _vggt_bind_trainable_scope(torch: Any, model: Any, scope: str) -> tuple[Any, str]:
    if scope == "camera_head":
        for parameter in model.aggregator.parameters():
            parameter.requires_grad_(False)
        model.aggregator.eval()
        model.camera_head.train()
        return model.camera_head, "camera_head"
    if scope != "camera_backbone":
        raise ValueError(f"unsupported VGGT training scope: {scope}")
    model.train()
    return torch.nn.ModuleDict(
        {"aggregator": model.aggregator, "camera_head": model.camera_head}
    ), "camera_backbone"


def _vggt_assert_optimizer_binding(optimizer: Any, parameters: list[Any]) -> None:
    optimized = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    if not parameters or [id(parameter) for parameter in optimized] != [
        id(parameter) for parameter in parameters
    ]:
        raise RuntimeError("VGGT optimizer is not bound to the complete trainable range")
    if len({id(parameter) for parameter in optimized}) != len(optimized):
        raise RuntimeError("VGGT optimizer contains duplicate parameters")
    if any(not parameter.requires_grad for parameter in optimized):
        raise RuntimeError("VGGT optimizer contains a frozen parameter")


def _vggt_gradient_norm_before_clip(
    parameters: list[Any], torch: Any, label: str
) -> float:
    if not parameters:
        raise RuntimeError(f"VGGT {label} has no trainable parameters")
    squared_norm = None
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            raise RuntimeError(f"VGGT {label} contains an unused optimizer parameter")
        values = gradient.detach().coalesce().values() if gradient.is_sparse else gradient.detach()
        contribution = values.float().square().sum()
        squared_norm = contribution if squared_norm is None else squared_norm + contribution
    norm = float(squared_norm.sqrt().detach().cpu())
    if not math.isfinite(norm) or norm <= 0.0:
        raise RuntimeError(f"VGGT {label} gradients must be finite and nonzero before clipping")
    return norm


def _vggt_assert_optimizer_state_initialized(optimizer: Any, parameters: list[Any]) -> None:
    expected = {id(parameter) for parameter in parameters}
    actual = {id(parameter) for parameter in optimizer.state}
    if actual != expected:
        raise RuntimeError("VGGT AdamW state does not cover the complete trainable range")
    for parameter in parameters:
        if set(optimizer.state[parameter]) != {"step", "exp_avg", "exp_avg_sq"}:
            raise RuntimeError("VGGT AdamW state has unexpected or missing slots")


def _vggt_view_order(batches: list[dict[str, Any]]) -> list[str]:
    return [
        f"{batch['seq_name']}:{','.join(str(item) for item in batch['ids'])}"
        for batch in batches
    ]


def _vggt_batch_content_sha256(batches: list[dict[str, Any]]) -> str:
    return resume_common.tree_sha256(
        [
            {
                "seq_name": batch["seq_name"],
                "ids": list(batch["ids"]),
                "images": batch["images"],
                "depths": batch["depths"],
                "extrinsics": batch["extrinsics"],
                "intrinsics": batch["intrinsics"],
                "point_masks": batch["point_masks"],
            }
            for batch in batches
        ]
    )


def _vggt_optimizer_evidence(optimizer: Any, module: Any) -> dict[str, Any]:
    groups = []
    for group in optimizer.param_groups:
        groups.append({key: value for key, value in group.items() if key != "params"})
    return {
        "type": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
        "parameter_names": [name for name, parameter in module.named_parameters() if parameter.requires_grad],
        "param_groups": groups,
    }


def _vggt_source_sha256(value: Any) -> str:
    source = inspect.getsourcefile(value)
    if source is None or not Path(source).is_file():
        raise VGGTCheckpointValidationError(f"cannot identify VGGT dependency source: {value}")
    return resume_common.file_sha256(source)


def _vggt_source_evidence(model: Any, optimizer: Any) -> dict[str, Any]:
    import torch
    import vggt as upstream
    from vggt.utils.pose_enc import extri_intri_to_pose_encoding

    # Include functional helpers as well as the instantiated model classes.
    upstream_files = {}
    for root in upstream.__path__:
        root = Path(root).resolve()
        for path in sorted(root.rglob("*.py")):
            name = path.relative_to(root).as_posix()
            _vggt_require(name not in upstream_files, "ambiguous VGGT source package roots")
            upstream_files[name] = resume_common.file_sha256(path)
    _vggt_require(bool(upstream_files), "VGGT source package contains no Python sources")
    model_types = {type(module) for _, module in model.named_modules()}
    cuda = getattr(torch, "cuda", None)
    return {
        "backend": resume_common.file_sha256(__file__),
        "runtime": _vggt_source_sha256(runtime_helpers),
        "resume_helper": _vggt_source_sha256(resume_common),
        "recovery_helper": _vggt_source_sha256(run_recovery_validation),
        "upstream_python": upstream_files,
        "model_classes": {
            f"{cls.__module__}.{cls.__qualname__}": _vggt_source_sha256(cls)
            for cls in sorted(model_types, key=lambda cls: (cls.__module__, cls.__qualname__))
        },
        "pose_encoding": _vggt_source_sha256(extri_intri_to_pose_encoding),
        "optimizer": _vggt_source_sha256(type(optimizer)),
        "optimizer_loader": _vggt_source_sha256(optimizer.load_state_dict),
        "torch_version": str(torch.__version__),
        "torch_python": _vggt_source_sha256(torch),
        "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
        "cuda_python": _vggt_source_sha256(cuda) if cuda is not None else None,
    }


def _vggt_resume_identity(
    *,
    weights: Path,
    config: dict[str, Any],
    dataset_root: Path,
    annotation_root: Path,
    model: Any,
    optimizer: Any,
    batches: list[dict[str, Any]],
    cache_enabled: bool,
    training_scope: str | None = None,
    trainable_module: Any | None = None,
    aggregator_sha256: str | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    model_path = weights / "model.pt"
    model_sha = sha256_path(model_path)
    if model_sha is None:
        raise RuntimeError("VGGT model checkpoint is missing for resume identity")
    configured_scope = config.get("training_scope", training_scope or "camera_head")
    scope = training_scope or str(configured_scope)
    if scope not in VGGT_TRAINING_SCOPES:
        raise ValueError(f"unsupported VGGT training scope: {scope}")
    if configured_scope != scope:
        raise ValueError("VGGT training scope and resume binding disagree")
    if scope == "camera_backbone" and trainable_module is None:
        raise ValueError("VGGT camera_backbone resume identity requires the full trainable module")
    bound_module = trainable_module if trainable_module is not None else model.camera_head
    aggregator_sha = aggregator_sha256 or _module_sha(model.aggregator)
    # This only bounds CPU checkpoint validation/hash work, so same-source on/off resumes are compatible.
    evidence = {
        "workload": {
            "backend_source": sha256_path(Path(__file__).resolve()),
            "training_scope": scope,
            "trainable_state_key": scope,
            "image_size": int(config.get("image_size", 224)),
            "frames_per_step": int(config.get("frames_per_step", 2)),
            "minimum_sequence_frames": int(config.get("minimum_sequence_frames", 24)),
            "cached_batches": int(config.get("cached_batches", 4)),
            "max_grad_norm": float(config.get("max_grad_norm", 1.0)),
            "precision": str(config.get("precision", "bfloat16")),
        },
        "frozen_model": {
            "weights_model_pt": model_sha,
            "aggregator_sha256": aggregator_sha,
            "aggregator_role": "frozen" if scope == "camera_head" else "trainable",
        },
        "data": {
            "dataset_root": str(dataset_root.resolve()),
            "annotation_root": str(annotation_root.resolve()),
            "dataset_sha": config.get("dataset_sha"),
            "dataset_manifest": config.get("dataset_manifest"),
            "selected_batch_content_sha256": _vggt_batch_content_sha256(batches),
        },
        "cache": {
            "frozen_aggregator_feature_cache": cache_enabled,
            "cache_entries": len(batches) if cache_enabled else 0,
        },
        "optimizer": _vggt_optimizer_evidence(optimizer, bound_module),
        "sources": _vggt_source_evidence(model, optimizer),
    }
    return {key: resume_common.json_sha256(value) for key, value in evidence.items()}, evidence


def _first_tensor(value: Any) -> Any | None:
    if hasattr(value, "device") and hasattr(value, "shape"):
        return value
    if isinstance(value, dict):
        for item in value.values():
            found = _first_tensor(item)
            if found is not None:
                return found
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _first_tensor(item)
            if found is not None:
                return found
    return None


def _common_config(image_size: int) -> SimpleNamespace:
    return SimpleNamespace(
        debug=True,
        training=True,
        get_nearby=False,
        load_depth=True,
        inside_random=False,
        allow_duplicate_img=False,
        img_size=image_size,
        patch_size=14,
        augs=SimpleNamespace(scales=None),
        rescale=True,
        rescale_aug=False,
        landscape_check=False,
    )


def _tensor_batch(torch: Any, raw: dict[str, Any]) -> dict[str, Any]:
    images = torch.from_numpy(np.stack(raw["images"]).astype(np.float32)).contiguous()
    images = images.permute(0, 3, 1, 2).to(torch.get_default_dtype()).div(255).unsqueeze(0)
    return {
        "images": images,
        "depths": torch.from_numpy(np.stack(raw["depths"]).astype(np.float32)).unsqueeze(0),
        "extrinsics": torch.from_numpy(np.stack(raw["extrinsics"]).astype(np.float32)).unsqueeze(0),
        "intrinsics": torch.from_numpy(np.stack(raw["intrinsics"]).astype(np.float32)).unsqueeze(0),
        "point_masks": torch.from_numpy(np.stack(raw["point_masks"])).unsqueeze(0),
        "seq_name": raw["seq_name"],
        "ids": [int(item) for item in raw["ids"]],
    }


def _device_batch(batch: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.to("cuda", non_blocking=True) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def _cache_frozen_aggregator_features(model: Any, batches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build an CUDA-resident cache only for the fixed, frozen aggregator inputs."""

    cached: list[dict[str, Any]] = []
    with require_cuda().no_grad():
        with autocast_bf16():
            for batch in batches:
                device_batch = _device_batch(batch)
                tokens, patch_start_index = model.aggregator(device_batch["images"])
                cached.append(
                    {
                        "batch": device_batch,
                        "tokens": tuple(token.detach() if token is not None else None for token in tokens),
                        "patch_start_index": int(patch_start_index),
                    }
                )
    synchronize()
    return cached


def _restore_optimizer_placements(
    optimizer: Any, saved_optimizer: dict[str, Any], placements: dict[int, dict[str, str]]
) -> None:
    for saved_group, live_group in zip(saved_optimizer["param_groups"], optimizer.param_groups):
        for parameter_id, parameter in zip(saved_group["params"], live_group["params"]):
            for key, device in placements[parameter_id].items():
                optimizer.state[parameter][key] = optimizer.state[parameter][key].to(device)


def valid_depth_batch(point_masks: Any, minimum_pixels: int = 100) -> bool:
    return bool((point_masks.sum(dim=(-1, -2)) > minimum_pixels).all())


def _camera_loss(torch: Any, pose_encodings: list[Any], batch: dict[str, Any], observer=None) -> Any:
    from vggt.utils.pose_enc import extri_intri_to_pose_encoding

    valid_frames = batch["point_masks"].sum(dim=(-1, -2)) > 100
    ground_truth = extri_intri_to_pose_encoding(
        batch["extrinsics"],
        batch["intrinsics"],
        batch["images"].shape[-2:],
        pose_encoding_type="absT_quaR_FoV",
    )
    if observer is not None:
        observer.targets(ground_truth, valid_frames)
    total = ground_truth.new_zeros(())
    for stage_index, prediction in enumerate(pose_encodings):
        weight = 0.6 ** (len(pose_encodings) - stage_index - 1)
        if valid_frames.any():
            delta = (prediction[valid_frames] - ground_truth[valid_frames]).abs()
            stage_loss = (
                delta[..., :3].clamp(max=100).mean()
                + delta[..., 3:7].mean()
                + 0.5 * delta[..., 7:].mean()
            )
        else:
            stage_loss = prediction.mean() * 0.0
        total = total + torch.nan_to_num(stage_loss, nan=0.0, posinf=100.0, neginf=100.0) * weight
    return total / len(pose_encodings)


def run_vggt_training(
    weights: Path,
    config: dict[str, Any],
    output_dir: Path,
    source_root: Path,
) -> dict[str, Any]:
    checkpoint_cpu_threads_16 = _vggt_checkpoint_cpu_threads_enabled(config)
    training_scope, cache_enabled = _vggt_training_preflight(config)
    torch = require_cuda()
    training_root = source_root / "training"
    if not (weights / "model.pt").is_file() or not training_root.is_dir():
        raise RuntimeError("VGGT weights or official training source are missing")
    sys.path.insert(0, str(training_root))
    sys.path.insert(0, str(source_root))

    from data.datasets.co3d import Co3dDataset
    from vggt.models.vggt import VGGT

    seed = int(config.get("seed", 20260901))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats()

    image_size = int(config.get("image_size", 224))
    frames_per_step = int(config.get("frames_per_step", 2))
    dataset_root = Path(config["dataset_root"])
    annotation_root = Path(config["annotation_root"])
    data_started = perf_counter()
    dataset = Co3dDataset(
        _common_config(image_size),
        split="train",
        CO3D_DIR=str(dataset_root),
        CO3D_ANNOTATION_DIR=str(annotation_root),
        min_num_images=int(config.get("minimum_sequence_frames", 24)),
        len_train=1000,
    )
    if not dataset.sequence_list:
        raise RuntimeError("VGGT CO3D engineering subset is empty")
    batches = []
    batch_count = int(config.get("cached_batches", 4))
    rejected_batches = 0
    candidate_index = 0
    max_candidates = batch_count * len(dataset.sequence_list) * 8
    while len(batches) < batch_count and candidate_index < max_candidates:
        sequence_name = sorted(dataset.sequence_list)[candidate_index % len(dataset.sequence_list)]
        frame_count = len(dataset.data_store[sequence_name])
        ids = np.asarray(
            [(candidate_index * 17 + frame_index * 37) % frame_count for frame_index in range(frames_per_step)],
            dtype=np.int64,
        )
        raw = dataset.get_data(seq_name=sequence_name, ids=ids, aspect_ratio=1.0)
        candidate = _tensor_batch(torch, raw)
        if valid_depth_batch(candidate["point_masks"]):
            batches.append(candidate)
        else:
            rejected_batches += 1
        candidate_index += 1
    if len(batches) < batch_count:
        raise RuntimeError(
            f"only {len(batches)} valid CO3D batches found after rejecting {rejected_batches}"
        )
    data_seconds = perf_counter() - data_started

    raw_state = torch.load(weights / "model.pt", map_location="cpu", weights_only=True)
    state = raw_state.get("model", raw_state) if isinstance(raw_state, dict) else raw_state
    model = VGGT(enable_camera=True, enable_point=False, enable_depth=False, enable_track=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or not all(
        key.startswith(("point_head.", "depth_head.", "track_head.")) for key in unexpected
    ):
        raise RuntimeError(f"unexpected VGGT state mismatch: missing={missing}, unexpected={unexpected[:8]}")
    aggregator_sha_before = _module_sha(model.aggregator)
    camera_head_sha_before = _module_sha(model.camera_head)
    trainable_module, trainable_state_key = _vggt_bind_trainable_scope(
        torch, model, training_scope
    )
    model.to("cuda")
    trainable = [parameter for parameter in trainable_module.parameters() if parameter.requires_grad]
    backbone_trainable = [
        parameter for parameter in model.aggregator.parameters() if parameter.requires_grad
    ]
    camera_head_trainable = [
        parameter for parameter in model.camera_head.parameters() if parameter.requires_grad
    ]
    if training_scope == "camera_backbone" and not backbone_trainable:
        raise RuntimeError("VGGT camera_backbone scope has no trainable aggregator parameters")
    if not camera_head_trainable:
        raise RuntimeError("VGGT camera head has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config.get("learning_rate", 2e-5)),
        weight_decay=float(config.get("weight_decay", 0.01)),
    )
    _vggt_assert_optimizer_binding(optimizer, trainable)
    adapter_sha_before = camera_head_sha_before

    observed_devices: list[str] = []

    def capture_device(_module: Any, _inputs: Any, outputs: Any) -> None:
        tensor = _first_tensor(outputs)
        if tensor is not None:
            observed_devices.append(str(tensor.device))

    hook = model.aggregator.register_forward_hook(capture_device)
    feature_cache: list[dict[str, Any]] | None = None
    cache_build_seconds = 0.0
    if cache_enabled:
        cache_started = perf_counter()
        feature_cache = _cache_frozen_aggregator_features(model, batches)
        cache_build_seconds = perf_counter() - cache_started
    max_steps = int(config.get("max_steps", 20))
    warmup_steps = int(config.get("warmup_steps", 0))
    if max_steps <= 0 or warmup_steps < 0:
        raise ValueError("VGGT max_steps must be positive and warmup_steps must be non-negative")
    final_global_step = warmup_steps + max_steps
    configured_checkpoint_steps = config.get("checkpoint_steps")
    checkpoint_steps = (
        {int(item) for item in configured_checkpoint_steps}
        if configured_checkpoint_steps is not None
        else ({step for step in (10, 20) if warmup_steps < step <= final_global_step} or {final_global_step})
    )
    if not checkpoint_steps or any(step <= warmup_steps or step > final_global_step for step in checkpoint_steps):
        raise ValueError("VGGT checkpoint_steps must be completed global optimizer steps")
    resume_value = config.get("resume_from_checkpoint")
    resume_path = Path(str(resume_value)) if resume_value else None
    resume_identity, resume_identity_evidence = _vggt_resume_identity(
        weights=weights,
        config=config,
        dataset_root=dataset_root,
        annotation_root=annotation_root,
        model=model,
        optimizer=optimizer,
        batches=batches,
        cache_enabled=cache_enabled,
        training_scope=training_scope,
        trainable_module=trainable_module,
        aggregator_sha256=aggregator_sha_before,
    )
    resume_contract = _vggt_contract(
        identity=resume_identity,
        seed=seed,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        view_order=_vggt_view_order(batches),
        device_ids=[0],
    )
    completed_global_step = 0
    completed_measured_steps = 0
    resume_checkpoint_sha = None
    resume_boundary_validated = False
    if resume_path is not None:
        if not resume_path.is_file():
            raise RuntimeError(f"VGGT resume checkpoint is missing: {resume_path}")
        try:
            resume_payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        except Exception as error:
            raise VGGTCheckpointValidationError("unable to load VGGT resume checkpoint") from error
        restored = _vggt_restore_checkpoint(
            resume_payload,
            resume_contract,
            trainable_module,
            optimizer,
            torch,
            checkpoint_cpu_threads_16=checkpoint_cpu_threads_16,
            state_key=trainable_state_key,
        )
        _vggt_assert_restored_boundary(
            resume_payload,
            resume_contract,
            trainable_module,
            optimizer,
            torch,
            checkpoint_cpu_threads_16=checkpoint_cpu_threads_16,
            state_key=trainable_state_key,
        )
        completed_global_step = int(restored["metadata"]["global_optimizer_step"])
        completed_measured_steps = int(restored["metadata"]["completed_measured_steps"])
        if completed_global_step >= final_global_step:
            raise RuntimeError(
                "VGGT resume checkpoint has no remaining optimizer steps in this workload"
            )
        resume_checkpoint_sha = sha256_path(resume_path)
        resume_boundary_validated = True
    output_dir.mkdir(parents=True, exist_ok=True)
    from ..vggt_first_update import create_vggt_recorder
    first_update_recorder = create_vggt_recorder(model, optimizer, config, resume_identity, resume_identity_evidence)
    step_log = output_dir / "steps.ndjson"
    losses: list[float] = []
    grad_norms: list[float] = []
    camera_head_grad_norms: list[float] = []
    backbone_grad_norms: list[float] = []
    measured_steps: list[dict[str, float]] = []
    artifacts: list[dict[str, Any]] = []
    written_checkpoint_paths: set[Path] = set()

    def save_checkpoint(global_optimizer_step: int) -> Path:
        path = output_dir / f"checkpoint-{global_optimizer_step}.pt"
        payload = _vggt_make_checkpoint(
            contract=resume_contract,
            global_optimizer_step=global_optimizer_step,
            module=trainable_module,
            optimizer=optimizer,
            torch=torch,
            checkpoint_cpu_threads_16=checkpoint_cpu_threads_16,
            state_key=trainable_state_key,
        )
        _vggt_save_checkpoint(path, payload, torch)
        written_checkpoint_paths.add(path)
        return path

    def load_checkpoint(path: Path) -> int:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        restored = _vggt_restore_checkpoint(
            payload,
            resume_contract,
            trainable_module,
            optimizer,
            torch,
            checkpoint_cpu_threads_16=checkpoint_cpu_threads_16,
            state_key=trainable_state_key,
        )
        _vggt_assert_restored_boundary(
            payload,
            resume_contract,
            trainable_module,
            optimizer,
            torch,
            checkpoint_cpu_threads_16=checkpoint_cpu_threads_16,
            state_key=trainable_state_key,
        )
        return int(restored["metadata"]["global_optimizer_step"])

    def _native_train_step(global_optimizer_step: int) -> dict[str, Any]:
        if getattr(trainable_module, "_vggt_resume_poisoned", False):
            raise RuntimeError("VGGT checkpoint restore poisoned this worker")
        cache_entry = (
            feature_cache[(global_optimizer_step - 1) % len(feature_cache)] if feature_cache is not None else None
        )
        batch = cache_entry["batch"] if cache_entry is not None else _device_batch(
            batches[(global_optimizer_step - 1) % len(batches)]
        )
        if first_update_recorder is not None:
            first_update_recorder.enter(global_optimizer_step, batch)
        optimizer.zero_grad(set_to_none=True)
        synchronize()
        started_unix = time()
        step_started = perf_counter()
        forward_started = perf_counter()
        with autocast_bf16():
            if cache_entry is None:
                predictions = model(batch["images"])
            else:
                predictions = {"pose_enc_list": model.camera_head(cache_entry["tokens"])}
            loss = _camera_loss(torch, predictions["pose_enc_list"], batch, observer=first_update_recorder)
        if first_update_recorder is not None:
            first_update_recorder.forward(predictions, loss)
        synchronize()
        forward_seconds = perf_counter() - forward_started
        backward_started = perf_counter()
        loss.backward()
        camera_head_grad_norm = _vggt_gradient_norm_before_clip(
            camera_head_trainable, torch, "camera_head"
        )
        backbone_grad_norm = (
            _vggt_gradient_norm_before_clip(backbone_trainable, torch, "aggregator")
            if training_scope == "camera_backbone"
            else None
        )
        if first_update_recorder is not None:
            first_update_recorder.gradients('preclip')
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable, float(config.get("max_grad_norm", 1.0))
        )
        if first_update_recorder is not None:
            first_update_recorder.gradients('postclip', grad_norm)
        optimizer.step()
        _vggt_assert_optimizer_state_initialized(optimizer, trainable)
        if first_update_recorder is not None:
            first_update_recorder.updated()
        synchronize()
        backward_seconds = perf_counter() - backward_started
        value = float(loss.detach().cpu())
        norm = float(grad_norm.detach().cpu()) if hasattr(grad_norm, "detach") else float(grad_norm)
        return {
            "step": float(global_optimizer_step),
            "global_optimizer_step": float(global_optimizer_step),
            "next_global_optimizer_step": float(global_optimizer_step + 1),
            "batch_cursor": float((global_optimizer_step - 1) % len(batches)),
            "started_unix": started_unix,
            "ended_unix": time(),
            "loss": value,
            "grad_norm": norm,
            "camera_head_grad_norm_before_clip": camera_head_grad_norm,
            "backbone_grad_norm_before_clip": backbone_grad_norm,
            "forward_seconds": forward_seconds,
            "backward_update_seconds": backward_seconds,
            "step_seconds": perf_counter() - step_started,
        }

    def train_step(global_optimizer_step: int) -> dict[str, Any]:
        if first_update_recorder is None:
            return _native_train_step(global_optimizer_step)
        return first_update_recorder.guard(_native_train_step, global_optimizer_step)

    training_started = perf_counter()
    measurement_started: float | None = None
    with step_log.open("w", encoding="utf-8") as log_handle:
        for global_optimizer_step in range(completed_global_step + 1, final_global_step + 1):
            if global_optimizer_step > warmup_steps and measurement_started is None:
                synchronize()
                measurement_started = perf_counter()
            record = train_step(global_optimizer_step)
            log_handle.write(json.dumps(record, sort_keys=True) + "\n")
            log_handle.flush()
            if global_optimizer_step > warmup_steps:
                losses.append(record["loss"])
                grad_norms.append(record["grad_norm"])
                camera_head_grad_norms.append(record["camera_head_grad_norm_before_clip"])
                if record["backbone_grad_norm_before_clip"] is not None:
                    backbone_grad_norms.append(record["backbone_grad_norm_before_clip"])
                measured_steps.append(record)
                if global_optimizer_step in checkpoint_steps:
                    path = save_checkpoint(global_optimizer_step)
                    artifacts.append({"path": str(path), "sha256": sha256_path(path)})
    training_seconds = perf_counter() - training_started

    final_checkpoint = output_dir / f"checkpoint-{final_global_step}.pt"
    if final_checkpoint not in written_checkpoint_paths:
        if completed_measured_steps == max_steps and resume_path is not None:
            final_checkpoint = resume_path
        else:
            final_checkpoint = save_checkpoint(final_global_step)
            artifacts.append({"path": str(final_checkpoint), "sha256": sha256_path(final_checkpoint)})

    trained_state_path = output_dir / f"{trainable_state_key}.pt"
    torch.save(
        {
            key: value.detach().cpu()
            for key, value in trainable_module.state_dict().items()
        },
        trained_state_path,
    )
    artifacts.append(
        {"path": str(trained_state_path), "sha256": sha256_path(trained_state_path)}
    )
    camera_head_sha_after = _module_sha(model.camera_head)
    aggregator_sha_after = _module_sha(model.aggregator)
    adapter_sha_after = camera_head_sha_after
    final_state_sha256 = _vggt_boundary_sha256(
        contract=resume_contract,
        global_optimizer_step=final_global_step,
        module=trainable_module,
        optimizer=optimizer,
        torch=torch,
        state_key=trainable_state_key,
    )
    final_checkpoint_sha256 = sha256_path(final_checkpoint)
    step_log_records = [{"path": str(step_log), "sha256": sha256_path(step_log)}]
    full_measurement_wall_seconds = (
        perf_counter() - measurement_started if measurement_started is not None else 0.0
    )
    if len(measured_steps) != max_steps - completed_measured_steps:
        raise RuntimeError("VGGT resume produced an invalid measured-step count")
    if measured_steps and not 0.0 < full_measurement_wall_seconds:
        raise RuntimeError("VGGT measured wall clock did not advance")
    recovery_global_step = final_global_step
    recovery_checkpoint = final_checkpoint
    if resume_path is not None:
        recovery_global_step = completed_global_step
        recovery_checkpoint = resume_path
    elif (output_dir / "checkpoint-20.pt").is_file():
        recovery_global_step = 20
        recovery_checkpoint = output_dir / "checkpoint-20.pt"
    recovery_enabled = validate_recovery_enabled(config)
    if recovery_enabled:
        if not recovery_checkpoint.is_file():
            raise RuntimeError("VGGT recovery validation checkpoint is missing")
        load_checkpoint(recovery_checkpoint)

    def recovery_update() -> dict[str, Any]:
        record = train_step(recovery_global_step + 1)
        return {
            "record": record,
            "parameter_sha256": _module_sha(model.camera_head),
            "boundary_sha256": _vggt_recovery_boundary_sha256(
                contract=resume_contract,
                global_optimizer_step=recovery_global_step + 1,
                module=trainable_module,
                optimizer=optimizer,
                torch=torch,
                state_key=trainable_state_key,
            ),
        }

    recovery = run_recovery_validation(
        enabled=recovery_enabled,
        expected_step=recovery_global_step,
        metric_names=(
            "resume_loss_abs_error",
            "resume_parameter_sha_match",
            "resume_boundary_sha_match",
        ),
        update=recovery_update,
        restore=lambda: load_checkpoint(recovery_checkpoint),
        fingerprint=None,
        compare=lambda continuous, resumed, continuous_sha, resumed_sha: (
            abs(continuous["record"]["loss"] - resumed["record"]["loss"])
            <= float(config.get("resume_atol", 1e-6))
            and continuous["parameter_sha256"] == resumed["parameter_sha256"]
            and continuous["boundary_sha256"] == resumed["boundary_sha256"],
            {
                "resume_loss_abs_error": abs(
                    continuous["record"]["loss"] - resumed["record"]["loss"]
                ),
                "resume_parameter_sha_match": (
                    continuous["parameter_sha256"] == resumed["parameter_sha256"]
                ),
                "resume_boundary_sha_match": (
                    continuous["boundary_sha256"] == resumed["boundary_sha256"]
                ),
            },
        ),
    )
    if recovery_enabled:
        load_checkpoint(final_checkpoint)
    hook.remove()
    memory = memory_stats()

    finite_losses = all(np.isfinite(losses))
    finite_gradients = all(np.isfinite(grad_norms)) and (not grad_norms or min(grad_norms) > 0)
    finite_camera_head_gradients = all(np.isfinite(camera_head_grad_norms)) and (
        not camera_head_grad_norms or min(camera_head_grad_norms) > 0
    )
    finite_backbone_gradients = (
        all(np.isfinite(backbone_grad_norms))
        and len(backbone_grad_norms) == len(measured_steps)
        and (not backbone_grad_norms or min(backbone_grad_norms) > 0)
        if training_scope == "camera_backbone"
        else None
    )
    camera_head_changed = camera_head_sha_before != camera_head_sha_after
    aggregator_changed = aggregator_sha_before != aggregator_sha_after
    main_forward_cuda = bool(observed_devices) and all("cuda" in device for device in observed_devices)
    step_seconds = [record["step_seconds"] for record in measured_steps]
    forward_seconds = [record["forward_seconds"] for record in measured_steps]
    update_seconds = [record["backward_update_seconds"] for record in measured_steps]
    measured_frames = frames_per_step * len(measured_steps)
    full_wall_samples_per_second = (
        measured_frames / full_measurement_wall_seconds if measured_steps else None
    )
    legacy_median_samples_per_second = (
        frames_per_step / statistics.median(step_seconds) if step_seconds else None
    )
    metrics = {
        "training_scope": training_scope,
        "trainable_state_key": trainable_state_key,
        "steps": max_steps,
        "warmup_steps": warmup_steps,
        "measurement_steps": len(measured_steps),
        "configured_measurement_steps": max_steps,
        "measurement_scope": "current_invocation_remaining_post_warmup",
        "completed_global_step_before_resume": completed_global_step,
        "global_optimizer_step": final_global_step,
        "executed_optimizer_steps": len(measured_steps) + max(0, warmup_steps - completed_global_step),
        "frames_per_step": frames_per_step,
        "sequence_count": len(dataset.sequence_list),
        "cached_valid_batches": len(batches),
        "frozen_aggregator_feature_cache": cache_enabled,
        "feature_cache_build_seconds": cache_build_seconds,
        "rejected_invalid_depth_batches": rejected_batches,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "trainable_backbone_parameters": sum(
            parameter.numel() for parameter in backbone_trainable
        ),
        "trainable_camera_head_parameters": sum(
            parameter.numel() for parameter in camera_head_trainable
        ),
        "loss_min": min(losses) if losses else None,
        "loss_max": max(losses) if losses else None,
        "loss_final": losses[-1] if losses else None,
        "grad_norm_min": min(grad_norms) if grad_norms else None,
        "grad_norm_max": max(grad_norms) if grad_norms else None,
        "camera_head_grad_norm_before_clip_min": (
            min(camera_head_grad_norms) if camera_head_grad_norms else None
        ),
        "camera_head_grad_norm_before_clip_max": (
            max(camera_head_grad_norms) if camera_head_grad_norms else None
        ),
        "backbone_grad_norm_before_clip_min": (
            min(backbone_grad_norms) if backbone_grad_norms else None
        ),
        "backbone_grad_norm_before_clip_max": (
            max(backbone_grad_norms) if backbone_grad_norms else None
        ),
        "adapter_sha_before": adapter_sha_before,
        "adapter_sha_after": adapter_sha_after,
        "camera_head_sha_before": camera_head_sha_before,
        "camera_head_sha_after": camera_head_sha_after,
        "aggregator_sha_before": aggregator_sha_before,
        "aggregator_sha_after": aggregator_sha_after,
        "resume_loss_abs_error": recovery.metrics["resume_loss_abs_error"],
        "resume_parameter_sha_match": recovery.metrics["resume_parameter_sha_match"],
        "resume_boundary_sha_match": recovery.metrics["resume_boundary_sha_match"],
        "samples_per_second": full_wall_samples_per_second,
        "full_measurement_wall_seconds": full_measurement_wall_seconds,
        "full_measurement_frames": measured_frames,
        "samples_per_second_median_legacy": legacy_median_samples_per_second,
        "step_seconds_median": statistics.median(step_seconds) if step_seconds else None,
        "step_seconds_min": min(step_seconds) if step_seconds else None,
        "step_seconds_max": max(step_seconds) if step_seconds else None,
        "forward_seconds_median": statistics.median(forward_seconds) if forward_seconds else None,
        "backward_update_seconds_median": statistics.median(update_seconds) if update_seconds else None,
        "checkpoint_schema": VGGT_CHECKPOINT_SCHEMA,
        "checkpoint_identity": resume_contract["identity"],
        "checkpoint_identity_evidence": resume_identity_evidence,
        "checkpoint_final_sha256": final_checkpoint_sha256,
        "final_state_sha256": final_state_sha256,
        "resume_source_sha256": resume_checkpoint_sha,
        "cuda_memory": memory,
        "cuda_utilization": None,
    }
    checks = {
        "finite_tensors": finite_losses,
        "nonzero_finite_gradients": finite_gradients,
        "nonzero_finite_camera_head_gradients": finite_camera_head_gradients,
        "nonzero_finite_backbone_gradients": finite_backbone_gradients,
        "checkpoint_resume": recovery.passed,
        "resume_boundary": resume_path is None or resume_boundary_validated,
        "current_adapter": camera_head_changed,
        "camera_head_weights_changed": camera_head_changed,
        "backbone_weights_changed": (
            aggregator_changed if training_scope == "camera_backbone" else None
        ),
        "frozen_aggregator_unchanged": (
            not aggregator_changed if training_scope == "camera_head" else None
        ),
        "main_forward_device": main_forward_cuda,
        "weights_changed": camera_head_changed
        and (training_scope != "camera_backbone" or aggregator_changed),
    }
    if not all(
        checks[key] is not False
        for key in (
            "finite_tensors",
            "nonzero_finite_gradients",
            "nonzero_finite_camera_head_gradients",
            "nonzero_finite_backbone_gradients",
            "checkpoint_resume",
            "current_adapter",
            "backbone_weights_changed",
            "frozen_aggregator_unchanged",
            "main_forward_device",
            "weights_changed",
        )
    ):
        raise RuntimeError(f"VGGT training acceptance checks failed: {checks}")
    result = {
        "artifacts": artifacts,
        "logs": step_log_records,
        "metrics": metrics,
        "stage_timings": {
            "data_preparation_seconds": data_seconds,
            "feature_cache_build_seconds": cache_build_seconds,
            "training_seconds": training_seconds,
            "full_measurement_wall_seconds": full_measurement_wall_seconds,
            "forward_seconds_median": statistics.median(forward_seconds) if forward_seconds else None,
            "backward_update_seconds_median": statistics.median(update_seconds) if update_seconds else None,
        },
        "checks": checks,
        "observations": {"cuda_utilization_captured": False},
    }
    if not recovery_enabled and measured_steps:
        result["measurement_window"] = {
            "start_monotonic": measurement_started,
            "work_units": len(measured_steps),
            "unit": "optimizer_steps",
        }
    if first_update_recorder is not None:
        first_update_recorder.guard(first_update_recorder.finish, final_global_step, not aggregator_changed)
    return result
