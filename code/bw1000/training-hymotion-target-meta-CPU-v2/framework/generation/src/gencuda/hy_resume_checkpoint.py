"""Completed-policy-step checkpoints for the HY-Motion RL backend.

The module deliberately imports torch lazily so its schema and identity helpers
remain inspectable in CPU-only tooling. Checkpoints are trusted local files;
``torch.load`` must never be used here for third-party input.
"""
from __future__ import annotations

import errno
import hashlib
import inspect
import json
import math
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Callable


SCHEMA = "hy-motion-full-policy-step-v1"
BOUNDARY = "after-complete-policy-update"
IDENTITY_FIELDS = {"config", "source", "data", "model", "reward", "device", "runtime"}


class CheckpointValidationError(ValueError):
    """The checkpoint is incompatible or incomplete and was not applied."""


class ResumeApplyFatal(RuntimeError):
    """Application failed after validation; the worker must terminate."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckpointValidationError(message)


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    def encode_extra(item: Any) -> str:
        if isinstance(item, Path):
            return str(item)
        raise TypeError(f"not JSON serializable: {type(item)}")

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=encode_extra,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def cpu_tree(value: Any) -> Any:
    import numpy as np
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(item) for item in value)
    _require(
        value is None or type(value) in (bool, int, float, str, bytes),
        f"unsupported checkpoint value: {type(value)}",
    )
    return value


def tree_sha256(value: Any) -> str:
    import numpy as np
    import torch

    digest = hashlib.sha256()

    def feed(item: Any) -> None:
        digest.update(type(item).__name__.encode("utf-8") + b"\0")
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(str(item.dtype).encode("ascii"))
            digest.update(str(item.shape).encode("ascii"))
            digest.update(item.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=lambda candidate: (type(candidate).__name__, repr(candidate))):
                feed(key)
                feed(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(str(len(item)).encode("ascii"))
            for child in item:
                feed(child)
        else:
            _require(
                item is None or type(item) in (str, int, float, bool, bytes),
                f"unsupported hash value: {type(item)}",
            )
            digest.update(repr(item).encode("utf-8"))
        digest.update(b"\xff")

    feed(value)
    return digest.hexdigest()


def build_contract(
    *,
    identity: dict[str, str],
    seed: int,
    warmup_steps: int,
    max_steps: int,
    group_size: int,
    sample_order: list[str],
    device_ids: list[int],
) -> dict[str, Any]:
    contract = {
        "backend": "hy-motion-motionrft",
        "identity": dict(identity),
        "seed": seed,
        "warmup_steps": warmup_steps,
        "max_steps": max_steps,
        "group_size": group_size,
        "sample_order": list(sample_order),
        "device_ids": list(device_ids),
        "seed_formula": "base+global*1000+sample/v1",
    }
    validate_contract(contract)
    return contract


def validate_contract(contract: dict[str, Any]) -> None:
    expected = {
        "backend",
        "identity",
        "seed",
        "warmup_steps",
        "max_steps",
        "group_size",
        "sample_order",
        "device_ids",
        "seed_formula",
    }
    _require(type(contract) is dict and set(contract) == expected, "checkpoint contract fields mismatch")
    _require(contract["backend"] == "hy-motion-motionrft", "checkpoint backend mismatch")
    _require(contract["seed_formula"] == "base+global*1000+sample/v1", "seed formula mismatch")
    for key in ("seed", "warmup_steps", "max_steps", "group_size"):
        lower = 1 if key in ("max_steps", "group_size") else 0
        _require(type(contract[key]) is int and contract[key] >= lower, f"invalid {key}")
    identities = contract["identity"]
    _require(type(identities) is dict and set(identities) == IDENTITY_FIELDS, "identity fields mismatch")
    for key, value in identities.items():
        _require(
            type(value) is str
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value),
            f"invalid {key} identity SHA",
        )
    order = contract["sample_order"]
    _require(type(order) is list and order and all(type(item) is str and item for item in order), "invalid sample order")
    _require(len(set(order)) == len(order), "duplicate sample identifiers")
    devices = contract["device_ids"]
    _require(type(devices) is list and len(set(devices)) == len(devices), "invalid device IDs")
    _require(all(type(item) is int and item >= 0 for item in devices), "invalid device ID")


def boundary_metadata(contract: dict[str, Any], global_step: int) -> dict[str, Any]:
    validate_contract(contract)
    warmup_steps = contract["warmup_steps"]
    final_global_step = warmup_steps + contract["max_steps"]
    _require(
        type(global_step) is int and warmup_steps < global_step <= final_global_step,
        "checkpoint is not at a completed measured policy step",
    )
    measured_step = global_step - warmup_steps
    return {
        "schema": SCHEMA,
        "boundary": BOUNDARY,
        "contract": contract,
        "global_step": global_step,
        "measured_step": measured_step,
        "next_global_step": global_step + 1,
        "next_measured_step": measured_step + 1,
        "next_sample_cursor": global_step % len(contract["sample_order"]),
        "completed_policy_updates": global_step,
    }


def next_coordinate(metadata: dict[str, Any]) -> dict[str, Any] | None:
    contract = metadata["contract"]
    if metadata["measured_step"] == contract["max_steps"]:
        return None
    cursor = metadata["next_sample_cursor"]
    global_step = metadata["next_global_step"]
    return {
        "global_step": global_step,
        "measured_step": metadata["next_measured_step"],
        "sample_cursor": cursor,
        "sample_id": contract["sample_order"][cursor],
        "rollout_seeds": [
            contract["seed"] + global_step * 1000 + index
            for index in range(contract["group_size"])
        ],
    }


def capture_rng(device_ids: list[int]) -> dict[str, Any]:
    import numpy as np
    import torch

    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "cuda": {},
    }
    for device_id in device_ids:
        torch.cuda.synchronize(device_id)
        state["cuda"][device_id] = torch.cuda.get_rng_state(device_id).cpu().clone()
    return state


def validate_rng(state: dict[str, Any], device_ids: list[int], *, native: bool) -> str:
    import numpy as np
    import torch

    _require(type(state) is dict and set(state) == {"python", "numpy", "torch_cpu", "cuda"}, "missing RNG stream")
    _require(type(state["cuda"]) is dict and set(state["cuda"]) == set(device_ids), "missing or extra CUDA RNG stream")
    random.Random().setstate(state["python"])
    np.random.RandomState(0).set_state(state["numpy"])
    torch.Generator(device="cpu").set_state(state["torch_cpu"])
    for device_id, value in state["cuda"].items():
        _require(
            isinstance(value, torch.Tensor)
            and value.device.type == "cpu"
            and value.dtype == torch.uint8
            and value.ndim == 1
            and value.numel() > 0,
            f"invalid CUDA RNG state for device {device_id}",
        )
        if native:
            try:
                live = torch.cuda.default_generators[device_id]
                probe = live.clone_state()
                _require(probe is not live, "CUDA RNG clone aliases the global generator")
                probe.set_state(value.clone())
                _require(torch.equal(probe.get_state().cpu(), value), "CUDA RNG clone roundtrip mismatch")
            except CheckpointValidationError:
                raise
            except Exception as error:
                raise CheckpointValidationError(
                    f"cannot validate CUDA RNG independently on device {device_id}"
                ) from error
    return "private-native-clone-validated" if native and device_ids else "cpu-validated"


def _optimizer_implementation(optimizer: Any) -> dict[str, str]:
    import torch

    _require(type(optimizer) is torch.optim.AdamW, "only exact AdamW is supported")
    source = inspect.getsourcefile(type(optimizer))
    _require(source is not None, "cannot identify AdamW implementation")
    return {
        "module": type(optimizer).__module__,
        "qualname": type(optimizer).__qualname__,
        "source_sha256": file_sha256(source),
        "torch_version": str(torch.__version__),
    }


def _optimizer_parameter_names(model: Any, optimizer: Any) -> list[list[str]]:
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    result = []
    for group in optimizer.param_groups:
        _require(all(id(parameter) in names for parameter in group["params"]), "optimizer contains an unnamed parameter")
        result.append([names[id(parameter)] for parameter in group["params"]])
    flattened = [name for group in result for name in group]
    _require(len(flattened) == len(set(flattened)), "optimizer contains duplicate parameters")
    return result


def initial_optimizer_progress(model: Any, optimizer: Any) -> dict[str, int]:
    return {
        name: 0
        for group in _optimizer_parameter_names(model, optimizer)
        for name in group
    }


def advance_optimizer_progress(
    progress: dict[str, int], model: Any, optimizer: Any
) -> dict[str, int]:
    names = _optimizer_parameter_names(model, optimizer)
    flat_names = [name for group in names for name in group]
    _require(set(progress) == set(flat_names), "optimizer progress keys mismatch")
    result = dict(progress)
    for group_names, group in zip(names, optimizer.param_groups):
        for name, parameter in zip(group_names, group["params"]):
            if parameter.grad is not None:
                result[name] += 1
    return result


def capture_model_state(
    model: Any,
    optimizer: Any,
    optimizer_progress: dict[str, int],
) -> dict[str, Any]:
    optimizer_state = optimizer.state_dict()
    names = _optimizer_parameter_names(model, optimizer)
    placements: dict[int, dict[str, str]] = {}
    for saved_group, live_group in zip(optimizer_state["param_groups"], optimizer.param_groups):
        for saved_id, parameter in zip(saved_group["params"], live_group["params"]):
            values = optimizer.state.get(parameter, {})
            if values:
                placements[saved_id] = {
                key: str(value.device)
                for key, value in values.items()
                if hasattr(value, "device")
                }
    return {
        "trainable": cpu_tree(
            {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
        ),
        "buffers": cpu_tree(dict(model.named_buffers())),
        "modes": {name: module.training for name, module in model.named_modules()},
        "optimizer": cpu_tree(optimizer_state),
        "optimizer_parameter_names": names,
        "optimizer_placements": placements,
        "optimizer_implementation": _optimizer_implementation(optimizer),
        "optimizer_progress": cpu_tree(optimizer_progress),
    }


def _validate_tensor_mapping(saved: dict[str, Any], actual: dict[str, Any], label: str) -> None:
    import torch

    _require(type(saved) is dict and set(saved) == set(actual), f"{label} keys mismatch")
    for name, value in saved.items():
        target = actual[name]
        _require(
            isinstance(value, torch.Tensor)
            and value.device.type == "cpu"
            and not value.requires_grad
            and value.layout == torch.strided
            and value.shape == target.shape
            and value.dtype == target.dtype,
            f"{label}/{name} shape, dtype, layout, or placement mismatch",
        )
        if value.is_floating_point() or value.is_complex():
            _require(bool(torch.isfinite(value).all()), f"non-finite {label}/{name}")


def _integer_optimizer_step(value: Any) -> int:
    import torch

    _require(
        isinstance(value, torch.Tensor)
        and value.device.type == "cpu"
        and value.shape == torch.Size([])
        and value.dtype == torch.float32
        and not value.requires_grad,
        "AdamW step must be a CPU float32 scalar",
    )
    number = float(value.item())
    _require(math.isfinite(number) and number.is_integer() and 0 <= number <= 2**24, "invalid AdamW step")
    return int(number)


def validate_model_state(state: dict[str, Any], model: Any, optimizer: Any, global_step: int) -> None:
    import torch

    expected_fields = {
        "trainable",
        "buffers",
        "modes",
        "optimizer",
        "optimizer_parameter_names",
        "optimizer_placements",
        "optimizer_implementation",
        "optimizer_progress",
    }
    _require(type(state) is dict and set(state) == expected_fields, "model state fields mismatch")
    _validate_tensor_mapping(
        state["trainable"],
        {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad},
        "trainable",
    )
    _validate_tensor_mapping(state["buffers"], dict(model.named_buffers()), "buffers")
    modes = state["modes"]
    _require(type(modes) is dict and set(modes) == {name for name, _ in model.named_modules()}, "module topology mismatch")
    _require(all(type(value) is bool for value in modes.values()), "invalid module mode")
    _require(state["optimizer_implementation"] == _optimizer_implementation(optimizer), "AdamW implementation mismatch")
    expected_names = _optimizer_parameter_names(model, optimizer)
    _require(state["optimizer_parameter_names"] == expected_names, "optimizer parameter order mismatch")
    flattened_names = [name for group in expected_names for name in group]
    progress = state["optimizer_progress"]
    _require(type(progress) is dict and set(progress) == set(flattened_names), "optimizer progress keys mismatch")
    _require(
        all(type(value) is int and 0 <= value <= global_step for value in progress.values()),
        "invalid optimizer participation count",
    )

    saved = state["optimizer"]
    live = optimizer.state_dict()
    _require(type(saved) is dict and set(saved) == {"state", "param_groups"}, "optimizer top-level fields mismatch")
    _require(len(saved["param_groups"]) == len(live["param_groups"]) == len(expected_names), "optimizer group count mismatch")
    saved_ids: list[int] = []
    parameters_by_id: dict[int, Any] = {}
    for saved_group, live_group, live_parameters in zip(saved["param_groups"], live["param_groups"], optimizer.param_groups):
        _require(set(saved_group) == set(live_group), "AdamW parameter-group fields mismatch")
        _require(
            tree_sha256({key: value for key, value in saved_group.items() if key != "params"})
            == tree_sha256({key: value for key, value in live_group.items() if key != "params"}),
            "AdamW hyperparameters differ from configured optimizer",
        )
        ids = saved_group["params"]
        _require(type(ids) is list and all(type(item) is int and item >= 0 for item in ids), "invalid optimizer parameter IDs")
        _require(len(ids) == len(live_parameters["params"]), "optimizer parameter count mismatch")
        saved_ids.extend(ids)
        parameters_by_id.update(zip(ids, live_parameters["params"]))
    _require(len(saved_ids) == len(set(saved_ids)), "duplicate optimizer parameter ID")
    progress_by_id = dict(zip(saved_ids, (progress[name] for name in flattened_names)))
    expected_state_ids = {
        saved_id for saved_id, updates in progress_by_id.items() if updates > 0
    }
    _require(set(saved["state"]) == expected_state_ids, "missing or extra AdamW moment data")
    placements = state["optimizer_placements"]
    _require(type(placements) is dict and set(placements) == expected_state_ids, "optimizer placement entries mismatch")
    amsgrad_ids = {
        item
        for group in saved["param_groups"]
        if group.get("amsgrad")
        for item in group["params"]
    }
    for saved_id, values in saved["state"].items():
        expected_keys = {"step", "exp_avg", "exp_avg_sq"} | ({"max_exp_avg_sq"} if saved_id in amsgrad_ids else set())
        _require(type(values) is dict and set(values) == expected_keys, "AdamW state fields mismatch")
        _require(type(placements[saved_id]) is dict and set(placements[saved_id]) == expected_keys, "AdamW placement fields mismatch")
        parameter = parameters_by_id[saved_id]
        for key, value in values.items():
            _require(
                isinstance(value, torch.Tensor)
                and value.device.type == "cpu"
                and not value.requires_grad
                and value.layout == torch.strided
                and bool(torch.isfinite(value).all()),
                "invalid AdamW tensor",
            )
            placement = placements[saved_id][key]
            if key == "step":
                _require(
                    _integer_optimizer_step(value) == progress_by_id[saved_id],
                    "AdamW step does not match observed gradient participation",
                )
                _require(placement in ("cpu", str(parameter.device)), "AdamW step placement mismatch")
            else:
                _require(value.shape == parameter.shape and value.dtype == parameter.dtype, "AdamW moment shape or dtype mismatch")
                _require(placement == str(parameter.device), "AdamW moment placement mismatch")
                if key in ("exp_avg_sq", "max_exp_avg_sq"):
                    _require(bool((value >= 0).all()), "negative AdamW second moment")


def validate_anchor(anchor: dict[str, Any], trainable: dict[str, Any]) -> None:
    # Exported adapters omit activation-checkpoint wrappers; live state does not.
    canonical = {}
    for name, value in trainable.items():
        key = name.replace("._checkpoint_wrapped_module", "")
        _require(key not in canonical, "MotionRFT anchor parameter names collide")
        canonical[key] = value
    _validate_tensor_mapping(anchor, canonical, "MotionRFT anchor")


def make_payload(
    *,
    contract: dict[str, Any],
    global_step: int,
    model: Any,
    optimizer: Any,
    optimizer_progress: dict[str, int],
    anchor: dict[str, Any],
) -> dict[str, Any]:
    metadata = boundary_metadata(contract, global_step)
    state = capture_model_state(model, optimizer, optimizer_progress)
    payload = {
        "metadata": metadata,
        "state": state,
        "anchor": cpu_tree(anchor),
        "rng": capture_rng(contract["device_ids"]),
    }
    payload["adapter_sha256"] = tree_sha256(payload["state"]["trainable"])
    payload["integrity_sha256"] = tree_sha256(payload)
    validate_checkpoint(payload, contract, model, optimizer, native_rng=False)
    return payload


def validate_checkpoint(
    payload: dict[str, Any],
    contract: dict[str, Any],
    model: Any,
    optimizer: Any,
    *,
    native_rng: bool,
    expected_anchor: dict[str, Any] | None = None,
) -> dict[str, Any]:
    expected_fields = {"metadata", "state", "anchor", "rng", "adapter_sha256", "integrity_sha256"}
    _require(type(payload) is dict and set(payload) == expected_fields, "not a full HY-Motion checkpoint")
    unsigned = {key: value for key, value in payload.items() if key != "integrity_sha256"}
    _require(payload["integrity_sha256"] == tree_sha256(unsigned), "checkpoint integrity mismatch")
    metadata = payload["metadata"]
    _require(type(metadata) is dict and metadata.get("contract") == contract, "checkpoint identity or schedule mismatch")
    expected_metadata = boundary_metadata(contract, metadata.get("global_step"))
    _require(metadata == expected_metadata, "checkpoint step or data cursor mismatch")
    validate_model_state(payload["state"], model, optimizer, metadata["global_step"])
    validate_anchor(payload["anchor"], payload["state"]["trainable"])
    if expected_anchor is not None:
        validate_anchor(expected_anchor, payload["state"]["trainable"])
        _require(
            tree_sha256(payload["anchor"]) == tree_sha256(expected_anchor),
            "MotionRFT anchor differs from the configured preset adapter",
        )
    _require(payload["adapter_sha256"] == tree_sha256(payload["state"]["trainable"]), "adapter SHA mismatch")
    rng_validation = validate_rng(payload["rng"], contract["device_ids"], native=native_rng)
    return {
        "metadata": metadata,
        "next": next_coordinate(metadata),
        "rng_validation": rng_validation,
    }


def _copy_model_state(state: dict[str, Any], model: Any, optimizer: Any) -> None:
    import torch

    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    with torch.no_grad():
        for label, targets in (("trainable", parameters), ("buffers", buffers)):
            for name, value in state[label].items():
                targets[name].copy_(value.to(device=targets[name].device, dtype=targets[name].dtype))
    for name, module in model.named_modules():
        module.training = state["modes"][name]
    optimizer.load_state_dict(cpu_tree(state["optimizer"]))
    for saved_group, live_group in zip(state["optimizer"]["param_groups"], optimizer.param_groups):
        for saved_id, parameter in zip(saved_group["params"], live_group["params"]):
            for key, device in state["optimizer_placements"].get(saved_id, {}).items():
                optimizer.state[parameter][key] = optimizer.state[parameter][key].to(device)


def _restore_rng(state: dict[str, Any], device_ids: list[int]) -> None:
    import numpy as np
    import torch

    for device_id in device_ids:
        torch.cuda.set_rng_state(state["cuda"][device_id], device_id)
        torch.cuda.synchronize(device_id)
        _require(
            torch.equal(torch.cuda.get_rng_state(device_id).cpu(), state["cuda"][device_id]),
            f"applied CUDA RNG mismatch on device {device_id}",
        )
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])


def restore_checkpoint(
    payload: dict[str, Any],
    contract: dict[str, Any],
    model: Any,
    optimizer: Any,
    expected_anchor: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, int], dict[str, Any]]:
    checked = validate_checkpoint(
        payload,
        contract,
        model,
        optimizer,
        native_rng=bool(contract["device_ids"]),
        expected_anchor=expected_anchor,
    )
    try:
        _copy_model_state(payload["state"], model, optimizer)
        restored_anchor = cpu_tree(payload["anchor"])
        _restore_rng(payload["rng"], contract["device_ids"])
    except Exception as error:
        setattr(model, "_hy_resume_poisoned", True)
        raise ResumeApplyFatal("checkpoint application failed; terminate this worker") from error
    restored_progress = dict(payload["state"]["optimizer_progress"])
    return restored_anchor, restored_progress, checked


def assert_restored_boundary(
    payload: dict[str, Any],
    contract: dict[str, Any],
    model: Any,
    optimizer: Any,
    anchor: dict[str, Any],
    optimizer_progress: dict[str, int],
) -> None:
    observed = {
        "state": capture_model_state(model, optimizer, optimizer_progress),
        "anchor": cpu_tree(anchor),
        "rng": capture_rng(contract["device_ids"]),
    }
    expected = {key: payload[key] for key in ("state", "anchor", "rng")}
    _require(tree_sha256(observed) == tree_sha256(expected), "restored boundary differs from checkpoint")


_UNSUPPORTED_DIRECTORY_FSYNC_ERRNOS = frozenset(
    {errno.EINVAL, errno.ENOTSUP, getattr(errno, "EOPNOTSUPP", errno.ENOTSUP)}
)


def _sync_replaced_directory(
    directory: Path,
    *,
    platform_name: str | None = None,
    open_directory: Callable[[Path, int], int] | None = None,
    fsync_fd: Callable[[int], None] | None = None,
    close_fd: Callable[[int], None] | None = None,
) -> None:
    """Persist a rename on filesystems that support directory synchronization."""
    if (os.name if platform_name is None else platform_name) == "nt":
        # Windows does not expose a portable directory handle for fsync.
        return

    open_directory = os.open if open_directory is None else open_directory
    fsync_fd = os.fsync if fsync_fd is None else fsync_fd
    close_fd = os.close if close_fd is None else close_fd
    directory_fd = open_directory(directory, os.O_RDONLY)
    fsync_error: BaseException | None = None
    try:
        fsync_fd(directory_fd)
    except OSError as error:
        # Some POSIX filesystems explicitly reject fsync on directory handles.
        if error.errno not in _UNSUPPORTED_DIRECTORY_FSYNC_ERRNOS:
            fsync_error = error
    except BaseException as error:
        fsync_error = error

    try:
        close_fd(directory_fd)
    except BaseException as close_error:
        if fsync_error is None:
            raise
        raise fsync_error from close_error

    if fsync_error is not None:
        raise fsync_error


def save_atomic(
    path: Path | str,
    payload: dict[str, Any],
    *,
    save_fn: Callable[[Any, Any], None] | None = None,
) -> None:
    import torch

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".partial", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            (save_fn or torch.save)(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _sync_replaced_directory(destination.parent)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def load_checkpoint(path: Path | str) -> dict[str, Any]:
    import torch

    return torch.load(Path(path), map_location="cpu", weights_only=False)
