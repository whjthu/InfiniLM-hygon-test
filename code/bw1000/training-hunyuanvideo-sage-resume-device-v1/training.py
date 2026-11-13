from __future__ import annotations

import hashlib
import inspect
import json
import os
import random
try:
    import resource
except ModuleNotFoundError:  # Windows can inspect the CLI but cannot run an CUDA backend.
    resource = None
import statistics
import sys
from pathlib import Path
from time import perf_counter, time
from typing import Any

import numpy as np

from .. import hunyuanvideo_resume_checkpoint as resume_checkpoint
from .. import runtime as runtime_helpers
from ..manifest import sha256_path
from ..recovery_validation import RecoveryValidation, run_recovery_validation, validate_recovery_enabled
from ..runtime import memory_stats, require_cuda, synchronize, within_memory_limit


REQUIRED_CACHE_SHAPES = {
    "latents": (1, 32, 3, 30, 53),
    "text_emb": (1, 300, 3584),
    "text_mask": (1, 300),
    "byt5_text_states": (1, 256, 1472),
    "byt5_text_mask": (1, 256),
}


def validate_hunyuanvideo_cache(cache: dict[str, Any]) -> None:
    for key, expected in REQUIRED_CACHE_SHAPES.items():
        value = cache.get(key)
        actual = tuple(value.shape) if hasattr(value, "shape") else None
        if actual != expected:
            raise RuntimeError(f"invalid HunyuanVideo cache field {key}: {actual}, expected {expected}")
    if not isinstance(cache.get("prompt"), str) or not cache["prompt"]:
        raise RuntimeError("HunyuanVideo cache has no prompt")


def _init_nccl(torch: Any, output_dir: Path) -> tuple[bool, Path]:
    if torch.distributed.is_initialized():
        return False, output_dir / "existing-process-group"
    rendezvous = output_dir / "nccl-rendezvous"
    rendezvous.parent.mkdir(parents=True, exist_ok=True)
    rendezvous.unlink(missing_ok=True)
    torch.distributed.init_process_group(
        backend="nccl",
        init_method=f"file://{rendezvous}",
        rank=0,
        world_size=1,
    )
    return True, rendezvous


def _adapter_state(model: Any) -> dict[str, Any]:
    return {
        name: parameter.detach().cpu().contiguous()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _adapter_sha(model: Any) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        if not parameter.requires_grad:
            continue
        value = parameter.detach().float().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


# Sage-GRPO imports these legacy helpers. Native full-state resume never calls them.
def _cpu_tree(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _optimizer_to_device(optimizer: Any) -> None:
    for parameter, state in optimizer.state.items():
        for key, value in list(state.items()):
            if hasattr(value, "to"):
                state[key] = value.to(parameter.device)


def _seed_initialization(torch: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _source_sha256(value: Any, label: str) -> str:
    source = inspect.getsourcefile(value)
    if source is None or not Path(source).is_file():
        raise RuntimeError(f"cannot identify HunyuanVideo resume dependency source: {label}")
    return resume_checkpoint.file_sha256(Path(source))


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _checkpoint_path(value: Any) -> Path | None:
    if value is None:
        return None
    if type(value) is not str or not value:
        raise ValueError("resume_from_checkpoint must be a non-empty external checkpoint path")
    path = Path(value).expanduser().resolve()
    return path / "state.pt" if path.is_dir() else path


def _dependency_evidence(
    transformer: Any, get_muon_optimizer: Any, lora_config_type: Any, torch: Any
) -> dict[str, Any]:
    attention_sources = {
        f"{name}:{type(module).__module__}.{type(module).__qualname__}": _source_sha256(
            type(module), f"HunyuanVideo attention module {name}"
        )
        for name, module in transformer.named_modules()
        if "attn" in name.lower() and type(module).__module__.startswith("hyvideo.")
    }
    if not attention_sources:
        raise RuntimeError("HunyuanVideo transformer exposed no upstream attention module sources")
    lora_layer_sources = {
        f"{name}:{type(module).__module__}.{type(module).__qualname__}": _source_sha256(
            type(module), f"PEFT LoRA layer {name}"
        )
        for name, module in transformer.named_modules()
        if type(module).__module__.startswith("peft.") and hasattr(module, "lora_A")
    }
    if not lora_layer_sources:
        raise RuntimeError("HunyuanVideo LoRA injection exposed no PEFT LoRA layer sources")
    return {
        "helpers": {
            "backend": resume_checkpoint.file_sha256(Path(__file__)),
            "checkpoint": _source_sha256(resume_checkpoint, "HunyuanVideo checkpoint helper"),
            "hy_resume_checkpoint": _source_sha256(
                resume_checkpoint.common, "shared Hunyuan resume helper"
            ),
            "runtime": _source_sha256(runtime_helpers, "runtime helper"),
            "require_cuda": _source_sha256(require_cuda, "require_cuda helper"),
        },
        "hunyuanvideo": {
            "transformer": _source_sha256(type(transformer), "transformer"),
            "muon": _source_sha256(get_muon_optimizer, "Muon factory"),
            "attention_modules": attention_sources,
        },
        "peft": {
            "lora_config": _source_sha256(lora_config_type, "LoRA config"),
            "lora_layers": lora_layer_sources,
        },
        "torch_cuda": {
            "torch_version": str(torch.__version__),
            "torch_python": _source_sha256(torch, "torch Python module"),
            "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
            "cuda_python": _source_sha256(torch.cuda, "torch CUDA module"),
        },
    }


def _resume_identity(
    *,
    config: dict[str, Any],
    transformer_path: Path,
    cache_sha256: str,
    transformer: Any,
    get_muon_optimizer: Any,
    lora_config_type: Any,
    torch: Any,
) -> dict[str, str]:
    dependencies = _dependency_evidence(transformer, get_muon_optimizer, lora_config_type, torch)
    relevant_config = {
        key: config[key]
        for key in sorted(config)
        if key not in {"resume_from_checkpoint", "validate_recovery", "resume_atol", "output_root", "run_name"}
    }
    return {
        "config": resume_checkpoint.json_sha256(relevant_config),
        "source": dependencies["helpers"]["backend"],
        "model": _tree_sha256(transformer_path),
        "data": resume_checkpoint.json_sha256(
            {"cache_sha256": cache_sha256, "consumption": resume_checkpoint.CACHE_CONSUMPTION}
        ),
        "runtime": resume_checkpoint.json_sha256(
            {"dtype": "bfloat16", "device": "cuda", "torch": dependencies["torch_cuda"]}
        ),
        "dependencies": resume_checkpoint.json_sha256(dependencies),
    }


def _invocation_counts(
    completed_global_step: int, warmup_steps: int, total_global_steps: int
) -> dict[str, int]:
    if not 0 <= completed_global_step < total_global_steps or not 0 <= warmup_steps < total_global_steps:
        raise ValueError("invalid HunyuanVideo invocation schedule")
    start = completed_global_step + 1
    return {
        "executed_global_steps": total_global_steps - completed_global_step,
        "executed_warmup_steps": max(0, warmup_steps - completed_global_step),
        "executed_measured_steps": total_global_steps - max(warmup_steps, completed_global_step),
        "first_global_step": start,
    }


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
            if block is not None
        }
    )
    if not block_types:
        raise RuntimeError("HunyuanVideo transformer exposes no checkpointable blocks")

    def wrapper(module: Any) -> Any:
        return checkpoint_wrapper(module, checkpoint_impl=CheckpointImpl.NO_REENTRANT)

    apply_activation_checkpointing(
        transformer,
        checkpoint_wrapper_fn=wrapper,
        check_fn=lambda module: isinstance(module, block_types),
    )


def _save_adapter(path: Path, model: Any) -> None:
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(_adapter_state(model), path)


def prepare_native_model(torch: Any, weights: Path, config: dict[str, Any], *, device: Any = "cuda") -> tuple[Any, Any, Any, str]:
    from hyvideo.models.transformers.hunyuanvideo_1_5_transformer import HunyuanVideo_1_5_DiffusionTransformer
    from hyvideo.optim.muon import get_muon_optimizer
    from peft import LoraConfig

    version = str(config.get("transformer_version", "480p_t2v"))
    transformer = HunyuanVideo_1_5_DiffusionTransformer.from_pretrained(
        weights / "transformer" / version, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    ).to(device)
    transformer.set_attn_mode("torch")
    transformer.add_adapter(
        LoraConfig(r=int(config.get("lora_rank", 4)), lora_alpha=int(config.get("lora_alpha", 8)),
                   target_modules=list(config.get("lora_target_modules", ["img_attn_q", "img_attn_v"])),
                   lora_dropout=float(config.get("lora_dropout", 0.0)), bias="none", task_type="FEATURE_EXTRACTION"),
        adapter_name="default",
    )
    if bool(config.get("gradient_checkpointing", True)):
        _apply_gradient_checkpointing(transformer)
    transformer.train()
    if not any(p.requires_grad for p in transformer.parameters()):
        raise RuntimeError("HunyuanVideo LoRA injection produced no trainable parameters")
    return transformer, get_muon_optimizer, LoraConfig, version


def prepare_native_cache(torch: Any, cache: dict[str, Any], *, device: Any = "cuda") -> dict[str, Any]:
    values = {key: cache[key].to(device, dtype=torch.bfloat16)
              if key in {"latents", "text_emb", "byt5_text_states"} else cache[key].to(device)
              for key in REQUIRED_CACHE_SHAPES}
    latents = values["latents"]
    values["cond"] = torch.zeros((latents.shape[0], latents.shape[1] + 1, *latents.shape[2:]),
                                device=device, dtype=torch.bfloat16)
    return values


def sample_native_flow_inputs(torch: Any, values: dict[str, Any], config: dict[str, Any], *, device: Any = "cuda") -> tuple[Any, Any, Any]:
    latents = values["latents"]
    num_timesteps = int(config.get("num_train_timesteps", 1000))
    timestep_shift = float(config.get("train_timestep_shift", 3.0))
    noise = torch.randn_like(latents)
    unit_timestep = torch.sigmoid(torch.randn((latents.shape[0],), device=device))
    shifted = timestep_shift * unit_timestep / (1.0 + (timestep_shift - 1.0) * unit_timestep)
    timestep = shifted * num_timesteps
    scale = timestep.view(-1, 1, 1, 1, 1) / num_timesteps
    latents_noised = (1.0 - scale) * latents + scale * noise
    return torch.cat((latents_noised, values["cond"]), dim=1), timestep, noise - latents


def native_flow_mse(torch: Any, transformer: Any, values: dict[str, Any], inputs: tuple[Any, Any, Any]) -> Any:
    hidden_states, timestep, target = inputs
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        prediction = transformer(hidden_states, timestep, text_states=values["text_emb"], text_states_2=None,
                                 encoder_attention_mask=values["text_mask"], vision_states=None, mask_type="t2v",
                                 extra_kwargs={"byt5_text_states": values["byt5_text_states"],
                                               "byt5_text_mask": values["byt5_text_mask"]}, return_dict=False)[0]
        return torch.nn.functional.mse_loss(prediction, target)


def _build_training_result(
    *,
    artifacts: list[dict[str, Any]],
    step_log: Path,
    summary_path: Path,
    metrics: dict[str, Any],
    stage_timings: dict[str, Any],
    checks: dict[str, Any],
    recovery: RecoveryValidation,
) -> dict[str, Any]:
    result_metrics = dict(metrics)
    result_metrics["resume_loss_error"] = recovery.metrics["resume_loss_error"]
    return {
        "artifacts": artifacts,
        "logs": [
            {"path": str(step_log), "sha256": sha256_path(step_log)},
            {"path": str(summary_path), "sha256": sha256_path(summary_path)},
        ],
        "metrics": result_metrics,
        "stage_timings": stage_timings,
        "checks": checks,
    }


def validate_single_rank_launch(config: dict[str, Any], environ: Any = None) -> None:
    """Reject distributed launchers before loading a single-rank model."""
    environ = os.environ if environ is None else environ
    cards = config.get("cards", [0])
    if type(cards) not in (list, tuple) or len(cards) != 1 or type(cards[0]) is not int or cards[0] < 0:
        raise RuntimeError("native HunyuanVideo training requires exactly one card")
    for name, expected in (("world_size", 1), ("rank", 0), ("local_rank", 0)):
        value = config.get(name, expected)
        if type(value) is not int or value != expected:
            raise RuntimeError(f"native HunyuanVideo training rejects {name}={value!r}")
    for name, expected in (("WORLD_SIZE", 1), ("RANK", 0),
                           ("LOCAL_WORLD_SIZE", 1), ("LOCAL_RANK", 0)):
        if name not in environ:
            continue
        value = environ[name]
        if type(value) is not str or value != str(expected):
            raise RuntimeError(f"native HunyuanVideo training rejects launcher {name}={value!r}")


def run_hunyuanvideo_training(
    weights: Path,
    config: dict[str, Any],
    output_dir: Path,
    source_root: Path,
) -> dict[str, Any]:
    validate_single_rank_launch(config)
    if not weights.is_dir() or not source_root.is_dir():
        raise RuntimeError("HunyuanVideo model or source directory is missing")
    cache_path = Path(config["training_cache"])
    if not cache_path.is_file():
        raise RuntimeError(f"HunyuanVideo training cache is missing: {cache_path}")
    expected_cache_sha = str(config.get("training_cache_sha", ""))
    actual_cache_sha = str(sha256_path(cache_path))
    if expected_cache_sha and expected_cache_sha != actual_cache_sha:
        raise RuntimeError(
            f"HunyuanVideo cache SHA mismatch: {actual_cache_sha}, expected {expected_cache_sha}"
        )

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    torch = require_cuda()
    resume_path = _checkpoint_path(config.get("resume_from_checkpoint"))
    if resume_path is not None and not resume_path.is_file():
        raise RuntimeError(f"HunyuanVideo external checkpoint is missing: {resume_path}")
    warmup_steps = int(config.get("warmup_steps", 5))
    max_steps = int(config.get("max_steps", 20))
    if warmup_steps < 0 or max_steps < 1:
        raise ValueError("HunyuanVideo warmup_steps and max_steps are invalid")
    total_global_steps = warmup_steps + max_steps
    if config.get("checkpoint_step_basis", "global-optimizer-step") != "global-optimizer-step":
        raise ValueError("HunyuanVideo only accepts global optimizer checkpoint steps")
    configured_checkpoint_steps = config.get("checkpoint_steps", [10, 20])
    if (
        type(configured_checkpoint_steps) not in (list, tuple)
        or not configured_checkpoint_steps
        or any(type(item) is not int for item in configured_checkpoint_steps)
    ):
        raise ValueError("checkpoint_steps must be global optimizer step integers")
    checkpoint_steps = set(configured_checkpoint_steps)
    if not checkpoint_steps or any(step <= warmup_steps or step > total_global_steps for step in checkpoint_steps):
        raise ValueError("checkpoint_steps must name completed measured global optimizer steps")
    checkpoint_steps.add(total_global_steps)
    if torch.distributed.is_initialized() and (
        torch.distributed.get_world_size() != 1 or torch.distributed.get_rank() != 0
    ):
        raise RuntimeError("native HunyuanVideo resume rejects a multi-rank process group")
    created_group = False
    rendezvous: Path | None = None
    sys.path.insert(0, str(source_root))

    try:
        from hyvideo.commons.parallel_states import initialize_parallel_state
        from hyvideo.models.transformers.hunyuanvideo_1_5_transformer import HunyuanVideo_1_5_DiffusionTransformer
        from hyvideo.optim.muon import get_muon_optimizer
        from peft import LoraConfig

        seed = int(config.get("seed", 20260901))
        _seed_initialization(torch, seed)
        torch.cuda.reset_peak_memory_stats()

        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        validate_hunyuanvideo_cache(cache)
        for key in REQUIRED_CACHE_SHAPES:
            if not torch.isfinite(cache[key].float()).all().item():
                raise RuntimeError(f"non-finite HunyuanVideo cache tensor: {key}")

        load_started = perf_counter()
        transformer, get_muon_optimizer, LoraConfig, transformer_version = prepare_native_model(torch, weights, config)
        lora_targets = list(config.get("lora_target_modules", ["img_attn_q", "img_attn_v"]))
        trainable = [parameter for parameter in transformer.parameters() if parameter.requires_grad]
        optimizer = get_muon_optimizer(
            transformer,
            lr=float(config.get("learning_rate", 1e-5)),
            weight_decay=float(config.get("weight_decay", 0.01)),
        )
        transformer_path = weights / "transformer" / transformer_version
        contract = resume_checkpoint.build_contract(
            identity=_resume_identity(
                config=config,
                transformer_path=transformer_path,
                cache_sha256=actual_cache_sha,
                transformer=transformer,
                get_muon_optimizer=get_muon_optimizer,
                lora_config_type=LoraConfig,
                torch=torch,
            ),
            seed=seed,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            cache_sha256=actual_cache_sha,
            device_ids=[0],
        )
        resume_payload: dict[str, Any] | None = None
        resume_checked: dict[str, Any] | None = None
        if resume_path is not None:
            resume_payload = resume_checkpoint.load_checkpoint(resume_path)
            # Validate corruption and identity before the invocation creates output files.
            resume_checked = resume_checkpoint.validate_checkpoint(
                resume_payload, contract, transformer, optimizer, native_rng=True
            )
            if resume_checked["metadata"]["global_optimizer_step"] >= total_global_steps:
                raise RuntimeError("HunyuanVideo checkpoint already completed the configured schedule")
        load_seconds = perf_counter() - load_started

        output_dir.mkdir(parents=True, exist_ok=True)
        created_group, rendezvous = _init_nccl(torch, output_dir)
        initialize_parallel_state(sp=1)
        flow_values = prepare_native_cache(torch, cache)
        input_devices: list[str] = []
        output_devices: list[str] = []

        def pre_hook(_module: Any, inputs: tuple[Any, ...]) -> None:
            tensor = next((item for item in inputs if hasattr(item, "device")), None)
            input_devices.append(str(tensor.device) if tensor is not None else "missing")

        def post_hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            value = output[0] if isinstance(output, (tuple, list)) else output
            output_devices.append(str(getattr(value, "device", "missing")))

        pre_handle = transformer.register_forward_pre_hook(pre_hook)
        post_handle = transformer.register_forward_hook(post_hook)
        max_grad_norm = float(config.get("max_grad_norm", 1.0))

        completed_global_step = 0
        if resume_payload is not None:
            # This is the final initialization action before noise or timestep sampling.
            resume_checked = resume_checkpoint.restore_checkpoint(
                resume_payload, contract, transformer, optimizer
            )
            resume_checkpoint.assert_restored_boundary(resume_payload, contract, transformer, optimizer)
            completed_global_step = int(resume_checked["metadata"]["global_optimizer_step"])

        invocation = _invocation_counts(
            completed_global_step, warmup_steps, total_global_steps
        )

        def train_step(global_step: int) -> dict[str, float]:
            optimizer.zero_grad(set_to_none=True)
            flow_inputs = sample_native_flow_inputs(torch, flow_values, config)
            synchronize()
            started_unix = time()
            step_started = perf_counter()
            forward_started = perf_counter()
            loss = native_flow_mse(torch, transformer, flow_values, flow_inputs)
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
                raise RuntimeError(f"non-finite HunyuanVideo training step {global_step}")
            if grad_value <= 0.0:
                raise RuntimeError(f"zero HunyuanVideo gradient at step {global_step}")
            return {
                "global_step": global_step,
                "loss": loss_value,
                "grad_norm": grad_value,
                "forward_seconds": forward_seconds,
                "backward_update_seconds": backward_seconds,
                "step_seconds": perf_counter() - step_started,
                "started_unix": started_unix,
                "ended_unix": time(),
            }

        adapter_before = _adapter_sha(transformer)
        records: list[dict[str, Any]] = []
        measured: list[dict[str, Any]] = []
        artifacts: list[dict[str, Any]] = []
        step_log = output_dir / "steps.ndjson"
        training_started = perf_counter()
        with step_log.open("w", encoding="utf-8") as handle:
            for global_step in range(completed_global_step + 1, total_global_steps + 1):
                record = train_step(global_step)
                records.append(record)
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
                if global_step > warmup_steps:
                    measured.append(record)
                    if global_step in checkpoint_steps:
                        checkpoint = output_dir / f"checkpoint-{global_step}" / "state.pt"
                        payload = resume_checkpoint.make_payload(
                            contract=contract,
                            global_step=global_step,
                            model=transformer,
                            optimizer=optimizer,
                        )
                        resume_checkpoint.save_atomic(checkpoint, payload)
                        artifacts.append({"path": str(checkpoint), "sha256": sha256_path(checkpoint)})
        if invocation["executed_global_steps"] != len(records) or invocation["executed_measured_steps"] != len(measured):
            raise RuntimeError("HunyuanVideo invocation progress accounting mismatch")
        training_seconds = perf_counter() - training_started

        final_checkpoint = output_dir / f"checkpoint-{total_global_steps}" / "state.pt"
        if not final_checkpoint.is_file():
            payload = resume_checkpoint.make_payload(
                contract=contract,
                global_step=total_global_steps,
                model=transformer,
                optimizer=optimizer,
            )
            resume_checkpoint.save_atomic(final_checkpoint, payload)
            artifacts.append({"path": str(final_checkpoint), "sha256": sha256_path(final_checkpoint)})
        final_checkpoint_payload = resume_checkpoint.load_checkpoint(final_checkpoint)
        resume_checkpoint.validate_checkpoint(
            final_checkpoint_payload, contract, transformer, optimizer, native_rng=True
        )
        recovery = run_recovery_validation(
            enabled=validate_recovery_enabled(config),
            expected_step=total_global_steps,
            metric_names=("resume_loss_error", "continuous_adapter_sha", "resumed_adapter_sha"),
            update=lambda: train_step(total_global_steps + 1),
            restore=lambda: int(
                resume_checkpoint.restore_checkpoint(final_checkpoint_payload, contract, transformer, optimizer)[
                    "metadata"
                ]["global_optimizer_step"]
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
            "finite_tensors": all(torch.isfinite(cache[key].float()).all().item() for key in REQUIRED_CACHE_SHAPES),
            "nonzero_finite_gradients": all(value > 0.0 for value in grad_norms),
            "checkpoint_resume": recovery.passed,
            "current_adapter": adapter_before != adapter_after,
            "main_forward_device": bool(input_devices)
            and all("cuda" in device for device in input_devices + output_devices),
            "attention_cuda_sdpa_fallback": transformer.attn_mode == "torch",
            "peak_reserved_below_75gb": within_memory_limit(memory["reserved"]),
            "muon_optimizer": optimizer.__class__.__name__ == "Muon",
        }
        if not all(value is not False for value in checks.values()):
            raise RuntimeError(f"HunyuanVideo training checks failed: {checks}")
        summary = {
            "prompt": cache["prompt"],
            "transformer_version": transformer_version,
            "optimizer": optimizer.__class__.__name__,
            "lora_targets": lora_targets,
            "lora_rank": int(config.get("lora_rank", 4)),
            "trainable_parameters": sum(parameter.numel() for parameter in trainable),
            "adapter_sha_before": adapter_before,
            "adapter_sha_after": adapter_after,
            "cache_sha256": actual_cache_sha,
            "checkpoint_step_basis": "global-optimizer-step",
            "completed_global_step_at_start": completed_global_step,
            "external_resume": resume_path is not None,
            "configured_warmup_steps": warmup_steps,
            "configured_measured_steps": max_steps,
            "invocation": invocation,
            "resume_loss_error": recovery.metrics["resume_loss_error"],
            "continuous_adapter_sha": recovery.metrics["continuous_adapter_sha"],
            "resumed_adapter_sha": recovery.metrics["resumed_adapter_sha"],
            "checks": checks,
        }
        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        artifacts.append({"path": str(summary_path), "sha256": sha256_path(summary_path)})
        return _build_training_result(
            artifacts=artifacts,
            step_log=step_log,
            summary_path=summary_path,
            metrics={
                "steps": invocation["executed_measured_steps"],
                "warmup_steps": invocation["executed_warmup_steps"],
                "total_global_steps": total_global_steps,
                "completed_global_step_at_start": completed_global_step,
                "configured_warmup_steps": warmup_steps,
                "configured_measured_steps": max_steps,
                **invocation,
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
                "training_cache_sha": actual_cache_sha,
                "peak_cpu_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
                "cuda_memory": memory,
                "cuda_utilization": None,
            },
            stage_timings={
                "model_load_seconds": load_seconds,
                "training_seconds": training_seconds,
                "forward_seconds": [item["forward_seconds"] for item in measured],
                "backward_update_seconds": [item["backward_update_seconds"] for item in measured],
            },
            checks=checks,
            recovery=recovery,
        )
    finally:
        if created_group and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        if rendezvous is not None:
            rendezvous.unlink(missing_ok=True)
