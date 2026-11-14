from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import random
try:
    import resource
except ModuleNotFoundError:  # Windows can inspect the CLI but cannot run an CUDA backend.
    resource = None
import statistics
import sys
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter, time
from typing import Any

import numpy as np

from ..hy_resume_checkpoint import tree_sha256
from ..manifest import sha256_path
from ..hunyuanvideo_resume_checkpoint import _cpu_tree as snapshot_cpu_tree
from ..recovery_validation import run_recovery_validation, validate_recovery_enabled
from ..runtime import require_cuda, synchronize, within_memory_limit
from .hunyuanvideo_training import (
    REQUIRED_CACHE_SHAPES,
    _adapter_sha,
    _adapter_state,
    _apply_gradient_checkpointing,
    _cpu_tree,
    _optimizer_to_device,
    validate_hunyuanvideo_cache,
)


def video_composite_rewards(videos: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    if videos.ndim != 5 or target.ndim != 4:
        raise ValueError("expected generated N,F,H,W,C and target F,H,W,C videos")
    generated = videos.astype(np.float32)
    reference = target.astype(np.float32)[None]
    alignment = 1.0 - np.abs(generated - reference).mean(axis=(1, 2, 3, 4))
    generated_motion = np.abs(np.diff(generated, axis=1)).mean(axis=(1, 2, 3, 4))
    target_motion = float(np.abs(np.diff(reference, axis=1)).mean())
    motion = 1.0 - np.abs(generated_motion - target_motion)
    spatial_x = np.abs(np.diff(generated, axis=3)).mean(axis=(1, 2, 3, 4))
    spatial_y = np.abs(np.diff(generated, axis=2)).mean(axis=(1, 2, 3, 4))
    exposure = np.abs(generated.mean(axis=(1, 2, 3, 4)) - reference.mean())
    quality = 0.5 * (spatial_x + spatial_y) - 0.25 * exposure
    total = 0.5 * quality + 0.5 * motion + alignment
    return {
        "VQ": quality.astype(np.float32),
        "MQ": motion.astype(np.float32),
        "TA": alignment.astype(np.float32),
        "Overall": total.astype(np.float32),
    }


def _scheduler_class(source_root: Path) -> Any:
    path = source_root / "hyvideo" / "schedulers" / "scheduling_flow_match_discrete.py"
    spec = importlib.util.spec_from_file_location("gencuda_sage_scheduler", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load SAGE scheduler: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FlowMatchDiscreteScheduler


def _scheduler(scheduler_class: Any, sigmas: Any, device: Any, shift: float) -> Any:
    value = scheduler_class(shift=shift, reverse=True, solver="euler")
    value.num_inference_steps = int(sigmas.shape[0] - 1)
    value.sigmas = sigmas.to(device=device, dtype=require_cuda().float32).clone()
    value.timesteps = (value.sigmas[:-1] * value.config.num_train_timesteps).to(device)
    value._step_index = None
    value._begin_index = None
    return value


def _load_state(model: Any, state: dict[str, Any]) -> None:
    torch = require_cuda()
    parameters = dict(model.named_parameters())
    resolved = _resolve_adapter_parameter_names(set(state), set(parameters))
    with torch.no_grad():
        for name, tensor in state.items():
            parameter = parameters[resolved[name]]
            parameter.copy_(tensor.to(parameter.device, parameter.dtype))


def _adapter_parameter_name(name: str) -> str:
    return name.replace("._checkpoint_wrapped_module", "")


def _resolve_adapter_parameter_names(state_names: set[str], parameter_names: set[str]) -> dict[str, str]:
    parameters_by_canonical_name: dict[str, str] = {}
    for name in parameter_names:
        canonical_name = _adapter_parameter_name(name)
        if canonical_name in parameters_by_canonical_name:
            raise RuntimeError("HunyuanVideo adapter parameter names collide after checkpoint-wrapper normalization")
        parameters_by_canonical_name[canonical_name] = name
    resolved: dict[str, str] = {}
    missing: list[str] = []
    for name in state_names:
        parameter_name = parameters_by_canonical_name.get(_adapter_parameter_name(name))
        if parameter_name is None:
            missing.append(name)
        else:
            resolved[name] = parameter_name
    if missing:
        raise RuntimeError(f"adapter parameter is missing from policy: {sorted(missing)}")
    return resolved


def _state_sha(state: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().float().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _save_rl_checkpoint(
    path: Path,
    torch: Any,
    model: Any,
    optimizer: Any,
    anchor: dict[str, Any],
    measured_step: int,
    resume_contract: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "global_step": measured_step + resume_contract["warmup_steps"],
            "measured_step": measured_step,
            "resume_contract": resume_contract,
            "adapter": _adapter_state(model),
            "optimizer": _cpu_tree(optimizer.state_dict()),
            "anchor": _cpu_tree(anchor),
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng_0": torch.cuda.get_rng_state(0).cpu(),
            "cuda_rng_1": torch.cuda.get_rng_state(1).cpu(),
        },
        temporary,
    )
    temporary.replace(path)


def _load_rl_checkpoint(
    path: Path, torch: Any, model: Any, optimizer: Any, resume_contract: dict[str, Any], *, max_measured_step: int
) -> tuple[int, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    measured_step = payload.get("measured_step")
    if (payload.get("resume_contract") != resume_contract
            or type(measured_step) is not int or not 1 <= measured_step <= max_measured_step
            or payload.get("global_step") != measured_step + resume_contract["warmup_steps"]):
        raise ValueError("SAGE-GRPO checkpoint recipe or optimizer-step cursor differs")
    state = tuple(payload[name] for name in (
        "adapter", "optimizer", "anchor", "python_rng", "numpy_rng",
        "torch_rng", "cuda_rng_0", "cuda_rng_1",
    ))
    adapter, optimizer_state, anchor, python_rng, numpy_rng, torch_rng, actor_rng, reward_rng = state
    _load_state(model, adapter)
    optimizer.load_state_dict(optimizer_state)
    _optimizer_to_device(optimizer)
    random.setstate(python_rng)
    np.random.set_state(numpy_rng)
    torch.set_rng_state(torch_rng)
    torch.cuda.set_rng_state(actor_rng, 0)
    torch.cuda.set_rng_state(reward_rng, 1)
    return measured_step, anchor


def _save_adapter(path: Path, model: Any) -> None:
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(_adapter_state(model), path)


def _init_nccl(torch: Any, output_dir: Path) -> tuple[bool, Path]:
    if torch.distributed.is_initialized():
        return False, output_dir / "existing-process-group"
    rendezvous = output_dir / "nccl-rendezvous"
    rendezvous.unlink(missing_ok=True)
    torch.distributed.init_process_group(
        backend="nccl", init_method=f"file://{rendezvous}", rank=0, world_size=1
    )
    return True, rendezvous


def _policy_trace(old_log_probs, new_log_probs, ratios, advantages, policy_losses, moving_kls):
    """Record four rollouts in the order used by the optimizer."""
    fields = {
        "old_log_probs": old_log_probs, "new_log_probs": new_log_probs,
        "ratios": ratios, "policy_losses": policy_losses, "moving_kls": moving_kls,
    }
    if len(advantages) != 4 or any(len(values) != 16 for values in fields.values()):
        raise ValueError("SAGE-GRPO trace requires four rollouts with four transitions each")
    if not all(math.isfinite(value) for values in (*fields.values(), advantages) for value in values):
        raise ValueError("SAGE-GRPO trace contains non-finite policy values")
    if any(value <= 0 for value in ratios):
        raise ValueError("SAGE-GRPO trace contains a non-positive ratio")
    return [
        {"rollout": index, "advantage": advantages[index],
         **{name: values[4 * index:4 * (index + 1)] for name, values in fields.items()}}
        for index in range(4)
    ]


def _load_first_trajectory(torch, path, expected_sha, adapter_sha, input_identity, latent_shape, timesteps, sigmas):
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        if digest.hexdigest() != expected_sha:
            raise ValueError("SAGE-GRPO trajectory SHA256 mismatch")
        stream.seek(0)
        payload = torch.load(stream, map_location="cpu", weights_only=True)
    if (payload["global_step"] != 1 or payload["adapter_sha"] != adapter_sha
            or payload["input_identity"] != input_identity):
        raise ValueError("SAGE-GRPO trajectory input identity differs")
    for key, shape in (("latent", (4, 5, *latent_shape[1:])),
                       ("noise", (4, 4, *latent_shape[1:])), ("sampled_log_prob", (4, 4))):
        value = payload[key]
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                or value.dtype != torch.float32 or not bool(torch.isfinite(value).all())):
            raise ValueError(f"SAGE-GRPO trajectory has invalid {key}")
    for key, expected in (("timestep", timesteps), ("sigmas", sigmas)):
        value = payload[key]
        if (not isinstance(value, torch.Tensor) or value.dtype != expected.dtype
                or not torch.equal(value, expected.detach().cpu())):
            raise ValueError(f"SAGE-GRPO trajectory has mismatched {key}")
    rewards = payload["reward"]
    if (not isinstance(rewards, dict) or set(rewards) != {"VQ", "MQ", "TA", "Overall"}
            or any(not isinstance(values, list) or len(values) != 4
                   or any(type(value) not in (float, int) or not math.isfinite(value) for value in values)
                   for values in rewards.values())):
        raise ValueError("SAGE-GRPO trajectory has invalid reference rewards")
    return payload


@contextmanager
def _trace_actor_block_boundaries(model: Any, enabled: bool, detail_block: int | None = None):
    """Observe the checkpoint wrapper outputs from one actor forward."""
    if not enabled:
        yield None
        return

    records = {}
    handles = []

    def record(name, output):
        if name in records:
            raise RuntimeError(f"SAGE-GRPO actor block {name} ran twice in one forward")
        outputs = output if isinstance(output, tuple) else (output,)
        rows = []
        records[name] = rows
        for tensor in outputs:
            if not hasattr(tensor, "requires_grad"):
                raise RuntimeError(f"SAGE-GRPO actor block {name} returned a non-tensor")
            row = {"output_sha256": tree_sha256(tensor), "requires_grad": tensor.requires_grad}
            rows.append(row)
            if tensor.requires_grad:
                def record_gradient(gradient, row=row):
                    if "output_gradient_sha256" in row:
                        raise RuntimeError("SAGE-GRPO actor block output received two gradients")
                    row["output_gradient_sha256"] = tree_sha256(gradient)

                tensor.register_hook(record_gradient)

    try:
        for family in ("double_blocks", "single_blocks"):
            for index, module in enumerate(getattr(model, family)):
                name = f"{family}.{index}"
                handles.append(module.register_forward_hook(
                    lambda _module, _inputs, output, name=name: record(name, output)
                ))
        if detail_block is not None:
            if detail_block >= len(model.double_blocks):
                raise ValueError("SAGE-GRPO detailed block is outside the double-stream stack")
            wrapped = model.double_blocks[detail_block]
            prefix = f"double_blocks.{detail_block}."
            for stream in ("img", "txt"):
                for name in ("attn_q", "attn_k", "attn_v", "attn_proj", "mlp"):
                    name = f"{stream}_{name}"
                    module = getattr(wrapped._checkpoint_wrapped_module, name)
                    handles.append(module.register_forward_hook(
                        lambda _module, _inputs, output, key=prefix + name: record(key, output)
                    ))
                    if name.endswith(("attn_proj", "mlp")):
                        handles.append(module.register_forward_pre_hook(
                            lambda _module, inputs, key=prefix + name + ".input": record(key, inputs)
                        ))
        if not handles:
            raise RuntimeError("SAGE-GRPO actor has no block boundaries to trace")
        yield records
        if len(records) != len(handles):
            raise RuntimeError("SAGE-GRPO actor block forward coverage is incomplete")
    finally:
        for handle in handles:
            handle.remove()


def run_hunyuanvideo_sage_grpo(
    weights: Path,
    config: dict[str, Any],
    output_dir: Path,
    source_root: Path,
    sage_root: Path,
) -> dict[str, Any]:
    capture_first_policy = config.get("capture_first_policy", False)
    capture_only = config.get("capture_first_policy_only", False)
    resume_from = config.get("resume_from_checkpoint")
    replay_path = config.get("replay_first_trajectory")
    replay_sha = config.get("replay_first_trajectory_sha256")
    trace_blocks = config.get("trace_first_actor_blocks", False)
    detail_block = config.get("trace_first_actor_detail_block")
    if detail_block is not None and (
        type(detail_block) is not int or detail_block < 0 or not trace_blocks or replay_path is None
    ):
        raise ValueError("SAGE-GRPO detailed block tracing requires block tracing and fixed replay")
    if type(trace_blocks) is not bool or (trace_blocks and not (capture_only or replay_path)):
        raise ValueError("SAGE-GRPO actor block tracing requires capture-only or fixed replay")
    if (replay_path is None) != (replay_sha is None):
        raise ValueError("SAGE-GRPO replay requires both a trajectory and its SHA256")
    if replay_path is not None:
        if (type(replay_sha) is not str or len(replay_sha) != 64
                or set(replay_sha) - set("0123456789abcdef")):
            raise ValueError("SAGE-GRPO replay requires a lowercase SHA256")
        replay_path = Path(replay_path)
    if type(capture_first_policy) is not bool:
        raise ValueError("capture_first_policy must be boolean")
    if type(capture_only) is not bool or (capture_only and not capture_first_policy):
        raise ValueError("capture_first_policy_only requires capture_first_policy")
    if capture_first_policy and replay_path is not None:
        raise ValueError("SAGE-GRPO capture and replay are separate diagnostics")
    if (capture_first_policy or replay_path is not None) and any(type(config.get(key, 4)) is not int or config.get(key, 4) != 4
                                    for key in ("num_generations", "sampling_steps")):
        raise ValueError("SAGE-GRPO policy capture requires the original four-by-four rollout")
    if resume_from is not None and (capture_first_policy or replay_path is not None):
        raise ValueError("SAGE-GRPO checkpoint recovery and first-policy diagnostics are separate runs")
    vae_root = weights / "vae"
    if "sage_vae_dir" in config:
        vae_root = Path(config["sage_vae_dir"])
        if not vae_root.is_absolute() or not (vae_root / "config.json").is_file():
            raise ValueError("separate SAGE VAE requires a configured absolute directory")
    torch = require_cuda()
    if torch.cuda.device_count() < 2:
        raise RuntimeError("HunyuanVideo SAGE-GRPO requires actor and reward CUDA cards")
    cache_path = Path(config["training_cache"])
    target_video_path = Path(config["target_video"])
    initial_adapter_path = Path(config["initial_adapter"])
    for path in (cache_path, target_video_path, initial_adapter_path):
        if not path.is_file():
            raise RuntimeError(f"required HunyuanVideo RL input is missing: {path}")
    expected_adapter_file_sha = str(config.get("initial_adapter_file_sha", ""))
    if expected_adapter_file_sha and sha256_path(initial_adapter_path) != expected_adapter_file_sha:
        raise RuntimeError("HunyuanVideo initial RL adapter file SHA mismatch")
    input_identity = None
    if capture_first_policy or replay_path is not None:
        input_identity = {"cache_sha256": sha256_path(cache_path),
                          "target_video_sha256": sha256_path(target_video_path)}

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_path = output_dir / "first-policy-update.json"
    update_state_path = output_dir / "first-policy-update-state.pt"
    if (capture_first_policy or replay_path is not None) and (policy_path.exists() or update_state_path.exists() or
                                 (output_dir / "trajectories" / "step-0001.pt").exists()):
        raise FileExistsError("SAGE-GRPO policy capture requires fresh output paths")
    created_group, rendezvous = _init_nccl(torch, output_dir)
    sys.path.insert(0, str(source_root))
    actor_device = torch.device("cuda:0")
    reward_device = torch.device("cuda:1")

    try:
        import imageio.v2 as imageio
        from hyvideo.commons.parallel_states import initialize_parallel_state
        from hyvideo.models.autoencoders.hunyuanvideo_15_vae import AutoencoderKLConv3D
        from hyvideo.models.transformers.hunyuanvideo_1_5_transformer import (
            HunyuanVideo_1_5_DiffusionTransformer,
        )
        from hyvideo.optim.muon import get_muon_optimizer
        from peft import LoraConfig
        from safetensors.torch import load_file

        seed = int(config.get("seed", 20260901))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats(0)
        torch.cuda.reset_peak_memory_stats(1)
        initialize_parallel_state(sp=1)
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        validate_hunyuanvideo_cache(cache)
        scheduler_class = _scheduler_class(sage_root)

        load_started = perf_counter()
        transformer_version = str(config.get("transformer_version", "480p_t2v"))
        model = HunyuanVideo_1_5_DiffusionTransformer.from_pretrained(
            weights / "transformer" / transformer_version,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to(actor_device)
        model.set_attn_mode("torch")
        model.add_adapter(
            LoraConfig(
                r=int(config.get("lora_rank", 4)),
                lora_alpha=int(config.get("lora_alpha", 8)),
                target_modules=list(config.get("lora_target_modules", ["img_attn_q", "img_attn_v"])),
                lora_dropout=0.0,
                bias="none",
                task_type="FEATURE_EXTRACTION",
            ),
            adapter_name="default",
        )
        _apply_gradient_checkpointing(model)
        _load_state(model, load_file(initial_adapter_path, device="cpu"))
        expected_semantic_sha = str(config.get("initial_adapter_sha", ""))
        if expected_semantic_sha and _adapter_sha(model) != expected_semantic_sha:
            raise RuntimeError("HunyuanVideo initial RL adapter semantic SHA mismatch")
        optimizer = get_muon_optimizer(
            model,
            lr=float(config.get("learning_rate", 1e-5)),
            weight_decay=float(config.get("weight_decay", 0.01)),
        )
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        vae = AutoencoderKLConv3D.from_pretrained(
            vae_root, torch_dtype=torch.bfloat16
        ).to(reward_device)
        vae.set_tile_sample_min_size(256, 0.125)
        vae.eval()
        load_seconds = perf_counter() - load_started

        latents_shape = REQUIRED_CACHE_SHAPES["latents"]
        text_emb = cache["text_emb"].to(actor_device, dtype=torch.bfloat16)
        text_mask = cache["text_mask"].to(actor_device)
        byt5_states = cache["byt5_text_states"].to(actor_device, dtype=torch.bfloat16)
        byt5_mask = cache["byt5_text_mask"].to(actor_device)
        target_frames = np.stack(imageio.mimread(target_video_path)).astype(np.float32) / 255.0
        if tuple(target_frames.shape) != (9, 480, 848, 3):
            raise RuntimeError(f"invalid target reward video shape: {target_frames.shape}")

        observed_devices: list[str] = []

        def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
            tensor = next((item for item in inputs if hasattr(item, "device")), None)
            observed_devices.append(str(tensor.device) if tensor is not None else "missing")

        hook_handle = model.register_forward_pre_hook(hook)
        rollout_count = int(config.get("num_generations", 4))
        sampling_steps = int(config.get("sampling_steps", 4))
        eta = float(config.get("eta", 0.7))
        shift = float(config.get("flow_shift", 5.0))
        clip_range = float(config.get("clip_range", 1e-4))
        kl_weight = float(config.get("kl_weight", 1e-5))
        dual_step_weight = float(config.get("dual_step_weight", 0.1))
        max_grad_norm = float(config.get("max_grad_norm", 1.0))
        anchor_interval = int(config.get("anchor_interval", 10))

        def model_output(latents: Any, timestep: Any, gradients: bool) -> Any:
            batch = latents.shape[0]
            condition = torch.zeros(
                (batch, latents.shape[1] + 1, *latents.shape[2:]),
                device=actor_device,
                dtype=torch.bfloat16,
            )
            hidden = torch.cat((latents.to(torch.bfloat16), condition), dim=1)
            context = torch.enable_grad() if gradients else torch.no_grad()
            with context, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                return model(
                    hidden,
                    timestep.expand(batch).to(actor_device),
                    text_states=text_emb.expand(batch, -1, -1),
                    text_states_2=None,
                    encoder_attention_mask=text_mask.expand(batch, -1),
                    vision_states=None,
                    mask_type="t2v",
                    extra_kwargs={
                        "byt5_text_states": byt5_states.expand(batch, -1, -1),
                        "byt5_text_mask": byt5_mask.expand(batch, -1),
                    },
                    return_dict=False,
                )[0]

        def transition(
            latents: Any,
            next_latents: Any | None,
            timestep: Any,
            sigmas: Any,
            index: int,
            gradients: bool,
        ) -> tuple[Any, Any, Any, Any, Any]:
            prediction = model_output(latents, timestep, gradients)
            schedule = _scheduler(scheduler_class, sigmas, actor_device, shift)
            schedule._step_index = index
            next_value, _, log_prob, mean, std = schedule.sde_step_with_logprob(
                prediction,
                latents,
                timestep,
                eta,
                prev_sample=next_latents,
                grpo=True,
                sde_solver=True,
                sde_type="sage_grpo",
            )
            return next_value, log_prob, mean, std, prediction

        def decode(final_latents: Any) -> np.ndarray:
            videos = []
            scaling = float(vae.config.scaling_factor)
            shift_factor = float(getattr(vae.config, "shift_factor", 0.0) or 0.0)
            for index in range(final_latents.shape[0]):
                value = final_latents[index : index + 1].to(reward_device)
                value = value / scaling + shift_factor
                with torch.cuda.device(1), torch.no_grad(), torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16
                ), vae.memory_efficient_context():
                    vae.enable_tiling()
                    decoded = vae.decode(value, return_dict=False)[0]
                    vae.disable_tiling()
                array = (decoded / 2 + 0.5).clamp(0, 1).cpu().float().numpy()[0]
                videos.append(array.transpose(1, 2, 3, 0))
            return np.stack(videos)

        trajectory_paths: list[Path] = []
        video_artifacts: list[dict[str, Any]] = []
        anchor_state = _adapter_state(model)
        anchor_sha = _state_sha(anchor_state)

        def rl_step(global_step: int, save_outputs: bool = True) -> dict[str, Any]:
            nonlocal anchor_state, anchor_sha
            rollout_adapter_sha = _adapter_sha(model)
            rollout_started = perf_counter()
            initial = []
            seeds = []
            if replay_path is None:
                for sample_index in range(rollout_count):
                    sample_seed = seed + global_step * 100 + sample_index
                    seeds.append(sample_seed)
                    torch.cuda.manual_seed(sample_seed)
                    initial.append(torch.randn(latents_shape, device=actor_device, dtype=torch.float32))
                current = torch.cat(initial, dim=0)
            base_schedule = scheduler_class(shift=shift, reverse=True, solver="euler")
            base_schedule.set_timesteps(sampling_steps, device=actor_device)
            sigmas = base_schedule.sigmas.to(actor_device)
            timesteps = base_schedule.timesteps
            replay = None
            if replay_path is not None:
                replay = _load_first_trajectory(torch, replay_path, replay_sha, rollout_adapter_sha,
                                                input_identity, latents_shape, timesteps, sigmas)
                seeds = replay["seed"]
                current = replay["latent"][:, 0].to(actor_device)
            trajectory = [current.detach().cpu()]
            old_log_probs = []
            old_means = []
            noises = []
            model.eval()
            for index, timestep in enumerate(timesteps):
                fixed_next = replay["latent"][:, index + 1].to(actor_device) if replay is not None else None
                next_value, log_prob, mean, std, _ = transition(
                    current, fixed_next, timestep, sigmas, index, gradients=False
                )
                old_log_probs.append(log_prob.detach().cpu())
                old_means.append(mean.detach().cpu())
                if replay is None:
                    noise = (next_value - mean) / std
                    noises.append(noise.detach().cpu())
                current = next_value.detach() if fixed_next is None else fixed_next
                trajectory.append(current.cpu())
            synchronize()
            rollout_seconds = perf_counter() - rollout_started
            reward_started = perf_counter()
            videos = decode(current)
            rewards = video_composite_rewards(videos, target_frames)
            reward_values = rewards["Overall"]
            reward_variance = float(np.var(reward_values))
            if not math.isfinite(reward_variance) or reward_variance <= 0.0:
                raise RuntimeError(f"zero or invalid HunyuanVideo reward variance: {reward_variance}")
            advantages = (reward_values - reward_values.mean()) / (reward_values.std() + 1e-6)
            reward_seconds = perf_counter() - reward_started

            trajectory_tensor = torch.stack(trajectory, dim=1)
            evaluated_old_log = torch.stack(old_log_probs, dim=1)
            old_log_tensor = evaluated_old_log if replay is None else replay["sampled_log_prob"]
            capture_policy = (capture_first_policy and global_step == 1) or replay is not None
            trajectory_path = replay_path
            if replay is None and (save_outputs or capture_policy):
                trajectory_path = output_dir / "trajectories" / f"step-{global_step:04d}.pt"
                trajectory_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "latent": trajectory_tensor,
                        "timestep": timesteps.detach().cpu(),
                        "noise": torch.stack(noises, dim=1),
                        "sampled_log_prob": old_log_tensor,
                        "reward": {key: value.tolist() for key, value in rewards.items()},
                        "prompt": cache["prompt"],
                        "seed": seeds,
                        "global_step": global_step,
                        "adapter_sha": rollout_adapter_sha,
                        **({"sigmas": sigmas.detach().cpu(), "input_identity": input_identity}
                           if capture_policy else {}),
                    },
                    trajectory_path,
                )
                trajectory_paths.append(trajectory_path)
                if save_outputs and global_step in {1, int(config.get("max_steps", 20))}:
                    for sample_index, video in enumerate(videos):
                        path = output_dir / f"step-{global_step:04d}-sample-{sample_index}.mp4"
                        imageio.mimwrite(
                            path,
                            np.rint(video * 255).clip(0, 255).astype(np.uint8),
                            fps=8,
                            codec="libx264",
                            quality=8,
                            macro_block_size=None,
                        )
                        video_artifacts.append({"path": str(path), "sha256": sha256_path(path)})

            actor_start_sha = _adapter_sha(model)
            current_state = _adapter_state(model)
            if anchor_sha == actor_start_sha:
                anchor_means = old_means
            else:
                _load_state(model, anchor_state)
                anchor_means = []
                model.eval()
                for sample_index in range(rollout_count):
                    per_sample = []
                    for index, timestep in enumerate(timesteps):
                        latent = trajectory_tensor[sample_index : sample_index + 1, index].to(actor_device)
                        next_latent = trajectory_tensor[sample_index : sample_index + 1, index + 1].to(actor_device)
                        _, _, mean, _, _ = transition(
                            latent, next_latent, timestep, sigmas, index, gradients=False
                        )
                        per_sample.append(mean.detach().cpu())
                    anchor_means.append(torch.cat(per_sample, dim=0))
                _load_state(model, current_state)

            update_state = None
            if capture_policy:
                named_trainable = {name: value for name, value in model.named_parameters()
                                   if value.requires_grad}
                names_by_id = {id(value): name for name, value in named_trainable.items()}
                update_state = {
                    "schema": "hunyuanvideo-sage-policy-update-v1",
                    "global_step": global_step,
                    "optimizer_parameter_names": [
                        [names_by_id[id(value)] for value in group["params"]]
                        for group in optimizer.param_groups
                    ],
                    "parameters_before": snapshot_cpu_tree(named_trainable),
                    "optimizer_before": snapshot_cpu_tree(optimizer.state_dict()),
                    "transition_losses": [],
                    "transition_boundaries": [],
                }
            optimizer.zero_grad(set_to_none=True)
            update_started = perf_counter()
            new_log_values = []
            ratio_values = []
            moving_kls = []
            policy_losses = []
            balancing = _scheduler(scheduler_class, sigmas, actor_device, shift).compute_grad_balancing_factors(
                eta=eta, device=actor_device, sde_type="sage_grpo", sde_solver=True, sigmas=sigmas
            )
            model.train()
            transition_count = rollout_count * sampling_steps
            for sample_index in range(rollout_count):
                for index, timestep in enumerate(timesteps):
                    latent = trajectory_tensor[sample_index : sample_index + 1, index].to(actor_device)
                    next_latent = trajectory_tensor[sample_index : sample_index + 1, index + 1].to(actor_device)
                    old_log = old_log_tensor[sample_index : sample_index + 1, index].to(actor_device)
                    with _trace_actor_block_boundaries(
                        model, trace_blocks and sample_index == 0 and index == 0, detail_block
                    ) as block_boundaries:
                        _, new_log, mean, std, prediction = transition(
                            latent, next_latent, timestep, sigmas, index, gradients=True
                        )
                    ratio = torch.exp(new_log - old_log)
                    advantage = torch.tensor(advantages[sample_index], device=actor_device)
                    unclipped = -advantage * ratio
                    clipped = -advantage * torch.clamp(
                        ratio, 1.0 - clip_range, 1.0 + clip_range
                    )
                    policy_loss = torch.maximum(unclipped, clipped).mean() * balancing[index]
                    anchor_mean = (
                        anchor_means[index][sample_index : sample_index + 1]
                        if anchor_sha == actor_start_sha
                        else anchor_means[sample_index][index : index + 1]
                    ).to(actor_device)
                    moving_kl = ((mean - anchor_mean) ** 2).mean() / (2 * std.float().square() + 1e-6)
                    stepwise_kl = (old_log - new_log).mean()
                    loss = policy_loss + kl_weight * (moving_kl + dual_step_weight * stepwise_kl)
                    if update_state is not None:
                        prediction.retain_grad()
                    (loss / transition_count).backward()
                    if update_state is not None:
                        update_state["transition_losses"].append(float(loss.detach().cpu()))
                        if prediction.grad is None:
                            raise RuntimeError("SAGE-GRPO actor output gradient was not produced")
                        update_state["transition_boundaries"].append({
                            "rollout": sample_index,
                            "transition": index,
                            "actor_output_sha256": tree_sha256(prediction),
                            "actor_output_gradient_sha256": tree_sha256(prediction.grad),
                            "anchor_mean_sha256": tree_sha256(anchor_mean),
                            "accumulated_lora_gradients_sha256": tree_sha256(
                                {name: value.grad for name, value in named_trainable.items()}
                            ),
                        })
                    if block_boundaries is not None:
                        gradient_count = sum(
                            "output_gradient_sha256" in row
                            for rows in block_boundaries.values() for row in rows
                        )
                        if not gradient_count:
                            raise RuntimeError("SAGE-GRPO actor block tracing recorded no output gradients")
                        update_state["first_actor_block_boundaries"] = {
                            name: rows for name, rows in block_boundaries.items() if name.count(".") == 1
                        }
                        if detail_block is not None:
                            update_state["first_actor_internal_boundaries"] = {
                                name: rows for name, rows in block_boundaries.items() if name.count(".") > 1
                            }
                    del prediction
                    new_log_values.append(float(new_log.detach().cpu()))
                    ratio_values.append(float(ratio.detach().cpu()))
                    moving_kls.append(float(moving_kl.detach().cpu()))
                    policy_losses.append(float(policy_loss.detach().cpu()))
            if update_state is not None:
                update_state["gradients_before_clip"] = snapshot_cpu_tree(
                    {name: value.grad for name, value in named_trainable.items()})
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
            if update_state is not None:
                update_state["gradients_after_clip"] = snapshot_cpu_tree(
                    {name: value.grad for name, value in named_trainable.items()})
            optimizer.step()
            synchronize()
            update_seconds = perf_counter() - update_started
            grad_value = float(grad_norm.detach().cpu())
            if not math.isfinite(grad_value) or grad_value <= 0.0:
                raise RuntimeError(f"invalid HunyuanVideo RL gradient: {grad_value}")
            adapter_after = _adapter_sha(model)
            if adapter_after == actor_start_sha:
                raise RuntimeError("HunyuanVideo RL adapter did not change")
            if global_step % anchor_interval == 0:
                anchor_state = _adapter_state(model)
                anchor_sha = _state_sha(anchor_state)
            old_values = old_log_tensor.numpy().reshape(-1)
            new_values = np.asarray(new_log_values, dtype=np.float32)
            errors = np.abs(new_values - old_values)
            if capture_policy:
                update_state.update(parameters_after=snapshot_cpu_tree(named_trainable),
                                    optimizer_after=snapshot_cpu_tree(optimizer.state_dict()))
                with update_state_path.open("xb") as stream:
                    torch.save(update_state, stream)
                policy = {
                    "global_step": global_step,
                    "trajectory": {"path": str(trajectory_path), "sha256": sha256_path(trajectory_path)},
                    "reward": {key: value.tolist() for key, value in rewards.items()},
                    "rollouts": _policy_trace(old_values.tolist(), new_log_values, ratio_values,
                                              advantages.tolist(), policy_losses, moving_kls),
                    "adapter_before": actor_start_sha, "adapter_after": adapter_after,
                    "grad_norm_before_clip": grad_value,
                    "update_state": {"path": str(update_state_path), "sha256": sha256_path(update_state_path)},
                }
                if replay is not None:
                    policy["recomputed_old_log_probs"] = evaluated_old_log.tolist()
                    policy["reference_reward"] = replay["reward"]
                encoded = json.dumps(policy, allow_nan=False, indent=2) + "\n"
                with policy_path.open("x", encoding="utf-8") as stream:
                    stream.write(encoded)
            return {
                "global_step": global_step,
                "started_unix": time() - rollout_seconds - reward_seconds - update_seconds,
                "ended_unix": time(),
                "rollout_seconds": rollout_seconds,
                "reward_seconds": reward_seconds,
                "update_seconds": update_seconds,
                "step_seconds": rollout_seconds + reward_seconds + update_seconds,
                "reward": {key: value.tolist() for key, value in rewards.items()},
                "reward_variance": reward_variance,
                "grad_norm": grad_value,
                "policy_loss_mean": statistics.mean(policy_losses),
                "moving_kl_mean": statistics.mean(moving_kls),
                "ratio_mean": statistics.mean(ratio_values),
                "ratio_min": min(ratio_values),
                "ratio_max": max(ratio_values),
                "log_prob_mae": float(errors.mean()),
                "log_prob_p99": float(np.percentile(errors, 99)),
                "rollout_adapter_sha": rollout_adapter_sha,
                "actor_start_adapter_sha": actor_start_sha,
                "adapter_sha_after": adapter_after,
                "anchor_sha": anchor_sha,
            }

        if replay_path is not None or capture_only:
            rl_step(1, save_outputs=False)
            hook_handle.remove()
            policy_artifact = {"path": str(policy_path), "sha256": sha256_path(policy_path)}
            logs = [{"path": str(update_state_path), "sha256": sha256_path(update_state_path)}]
            if capture_only:
                logs.extend({"path": str(path), "sha256": sha256_path(path)} for path in trajectory_paths)
            return {
                "artifacts": [policy_artifact],
                "logs": logs,
                "metrics": {"diagnostic_fixed_trajectory_replay": replay_path is not None,
                            "diagnostic_capture_first_policy_only": capture_only, "policy_updates": 1,
                            "performance_accepted": False},
                "checks": {"policy_update_recorded": True, "current_adapter": True,
                           **({"fixed_trajectory_consumed": True} if replay_path is not None else {})},
                "stage_timings": {"model_load_seconds": load_seconds},
            }

        warmup_steps = int(config.get("warmup_steps", 0))
        max_steps = int(config.get("max_steps", 20))
        if warmup_steps < 0 or max_steps < 1:
            raise ValueError("SAGE-GRPO requires nonnegative warmup and at least one measured update")
        checkpoint_steps = {int(item) for item in config.get("checkpoint_steps", [10, 20])}
        resume_contract = {
            "schema": "sage-policy-resume-v1",
            "native_source": str(source_root.resolve()),
            "warmup_steps": warmup_steps,
            "weights": str(weights.resolve()),
            "vae": str(vae_root.resolve()),
            "scheduler_sha256": sha256_path(sage_root / "hyvideo/schedulers/scheduling_flow_match_discrete.py"),
            "input_sha256": {name: sha256_path(path) for name, path in (
                ("training_cache", cache_path), ("target_video", target_video_path),
                ("initial_adapter", initial_adapter_path)
            )},
            "policy_config": {key: value for key, value in config.items() if key not in {
                "resume_from_checkpoint", "validate_recovery", "max_steps", "checkpoint_steps",
            }},
        }
        resumed_steps = 0
        if resume_from is not None:
            previous = Path(resume_from).expanduser()
            if previous.is_dir():
                previous /= "state.pt"
            resumed_steps, anchor_state = _load_rl_checkpoint(
                previous, torch, model, optimizer, resume_contract, max_measured_step=max_steps - 1
            )
            anchor_sha = _state_sha(anchor_state)
        first_global_step = warmup_steps + resumed_steps + 1 if resumed_steps else 1
        records = []
        measured = []
        artifacts: list[dict[str, Any]] = []
        step_log = output_dir / "steps.ndjson"
        with step_log.open("x" if resumed_steps else "w", encoding="utf-8") as handle:
            for global_step in range(first_global_step, warmup_steps + max_steps + 1):
                if global_step == warmup_steps + resumed_steps + 1:
                    torch.cuda.synchronize(actor_device)
                    torch.cuda.synchronize(reward_device)
                    measurement_started = perf_counter()
                record = rl_step(global_step, save_outputs=global_step > warmup_steps)
                records.append(record)
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
                if global_step > warmup_steps:
                    measured_step = global_step - warmup_steps
                    measured.append(record)
                    if measured_step in checkpoint_steps:
                        path = output_dir / f"checkpoint-{measured_step}" / "state.pt"
                        _save_rl_checkpoint(path, torch, model, optimizer, anchor_state, measured_step, resume_contract)
                        artifacts.append({"path": str(path), "sha256": sha256_path(path)})

        checkpoint = output_dir / f"checkpoint-{max_steps}" / "state.pt"
        if not checkpoint.is_file():
            _save_rl_checkpoint(checkpoint, torch, model, optimizer, anchor_state, max_steps, resume_contract)
            artifacts.append({"path": str(checkpoint), "sha256": sha256_path(checkpoint)})
        torch.cuda.synchronize(actor_device)
        torch.cuda.synchronize(reward_device)
        measurement_seconds = perf_counter() - measurement_started
        if not math.isfinite(measurement_seconds) or measurement_seconds <= 0:
            raise RuntimeError("SAGE-GRPO measured window duration is invalid")

        def restore_recovery_checkpoint() -> int:
            nonlocal anchor_state, anchor_sha
            restored_step, anchor_state = _load_rl_checkpoint(
                checkpoint, torch, model, optimizer, resume_contract, max_measured_step=max_steps
            )
            anchor_sha = _state_sha(anchor_state)
            return restored_step

        recovery = run_recovery_validation(
            enabled=validate_recovery_enabled(config),
            expected_step=max_steps,
            metric_names=("resume_reward_error", "resume_parameter_sha_match"),
            update=lambda: rl_step(warmup_steps + max_steps + 1, save_outputs=False),
            restore=restore_recovery_checkpoint,
            fingerprint=lambda: _adapter_sha(model),
            compare=lambda continuous, resumed, continuous_sha, resumed_sha: (
                continuous_sha == resumed_sha
                and max(
                    abs(a - b)
                    for a, b in zip(
                        continuous["reward"]["Overall"], resumed["reward"]["Overall"]
                    )
                ) <= float(config.get("resume_atol", 1e-6)),
                {
                    "resume_reward_error": max(
                        abs(a - b)
                        for a, b in zip(
                            continuous["reward"]["Overall"], resumed["reward"]["Overall"]
                        )
                    ),
                    "resume_parameter_sha_match": continuous_sha == resumed_sha,
                },
            ),
        )
        adapter_after = _adapter_sha(model)
        adapter_path = output_dir / "adapter-final.safetensors"
        _save_adapter(adapter_path, model)
        artifacts.extend(video_artifacts)
        artifacts.append({"path": str(adapter_path), "sha256": sha256_path(adapter_path)})
        hook_handle.remove()

        step_seconds = [item["step_seconds"] for item in measured]
        rollout_seconds = [item["rollout_seconds"] for item in measured]
        reward_seconds = [item["reward_seconds"] for item in measured]
        update_seconds = [item["update_seconds"] for item in measured]
        reward_variances = [item["reward_variance"] for item in measured]
        grad_norms = [item["grad_norm"] for item in measured]
        log_prob_mae = [item["log_prob_mae"] for item in measured]
        log_prob_p99 = [item["log_prob_p99"] for item in measured]
        ratios = [item["ratio_mean"] for item in measured]
        actor_memory = {
            "allocated": int(torch.cuda.max_memory_allocated(0)),
            "reserved": int(torch.cuda.max_memory_reserved(0)),
        }
        reward_memory = {
            "allocated": int(torch.cuda.max_memory_allocated(1)),
            "reserved": int(torch.cuda.max_memory_reserved(1)),
        }
        checks = {
            "finite_tensors": all(math.isfinite(value) for value in step_seconds + ratios),
            "nonzero_finite_gradients": all(value > 0 and math.isfinite(value) for value in grad_norms),
            "checkpoint_resume": recovery.passed,
            "current_adapter": all(
                item["rollout_adapter_sha"] == item["actor_start_adapter_sha"] for item in measured
            ),
            "main_forward_device": bool(observed_devices) and all("cuda:0" in value for value in observed_devices),
            "reward_variance_nonzero": all(value > 0 for value in reward_variances),
            "log_prob_alignment": max(log_prob_mae) <= 0.02 and max(log_prob_p99) <= 0.10,
            "ratio_finite": all(math.isfinite(value) and value > 0 for value in ratios),
            "sage_sde": True,
            "peak_reserved_below_75gb": within_memory_limit(max(actor_memory["reserved"], reward_memory["reserved"])),
        }
        if not all(value is not False for value in checks.values()):
            raise RuntimeError(f"HunyuanVideo SAGE-GRPO checks failed: {checks}")
        summary = {
            "reward_backend": "local_vq_mq_condition_alignment",
            "videoalign_status": "blocked: external VideoReward weights unavailable from server network",
            "sage_source_sha": str(config.get("sage_source_sha")),
            "initial_adapter_file_sha": sha256_path(initial_adapter_path),
            "adapter_sha_after": adapter_after,
            "resume_reward_error": recovery.metrics["resume_reward_error"],
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
                *({"path": str(path), "sha256": sha256_path(path)} for path in trajectory_paths),
                *([{"path": str(policy_path), "sha256": sha256_path(policy_path)}] if capture_first_policy else []),
                *([{"path": str(update_state_path), "sha256": sha256_path(update_state_path)}]
                  if capture_first_policy else []),
            ],
            "metrics": {
                "steps": len(measured),
                "warmup_steps": 0 if resumed_steps else warmup_steps,
                "completed_global_step": warmup_steps + max_steps,
                "resumed_from_global_step": warmup_steps + resumed_steps if resumed_steps else None,
                "num_generations": rollout_count,
                "sampling_steps": sampling_steps,
                "reward_variance_min": min(reward_variances),
                "reward_variance_median": statistics.median(reward_variances),
                "grad_norm_min": min(grad_norms),
                "grad_norm_max": max(grad_norms),
                "ratio_mean": statistics.mean(ratios),
                "log_prob_mae_max": max(log_prob_mae),
                "log_prob_p99_max": max(log_prob_p99),
                "step_seconds_min": min(step_seconds),
                "step_seconds_median": statistics.median(step_seconds),
                "step_seconds_max": max(step_seconds),
                "updates_per_second": 1.0 / statistics.median(step_seconds),
                "measurement_steps": len(measured),
                "measurement_wall_seconds": measurement_seconds,
                "measurement_updates_per_second": len(measured) / measurement_seconds,
                "measurement_samples_per_second": len(measured) * rollout_count / measurement_seconds,
                "policy_window_boundary": (
                    "before first measured update through final checkpoint write and hash and both device drains; "
                    "includes trajectory, video and step-log writes; excludes warmup, recovery and final adapter export"
                ),
                "adapter_sha_after": adapter_after,
                "resume_reward_error": recovery.metrics["resume_reward_error"],
                "actor_cuda_memory": actor_memory,
                "reward_cuda_memory": reward_memory,
                "peak_cpu_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
                "cuda_utilization": None,
                **({"diagnostic_capture_first_policy": True, "performance_accepted": False}
                   if capture_first_policy else {}),
            },
            "stage_timings": {
                "model_load_seconds": load_seconds,
                "rollout_seconds": rollout_seconds,
                "reward_seconds": reward_seconds,
                "update_seconds": update_seconds,
            },
            "checks": checks,
        }
    finally:
        if created_group and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        rendezvous.unlink(missing_ok=True)
