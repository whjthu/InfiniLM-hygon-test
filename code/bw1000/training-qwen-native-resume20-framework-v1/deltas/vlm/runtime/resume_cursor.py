"""Plan an epoch-boundary Swift/Trainer resume without loading model state.

The caller must separately verify the actual dataloader length, topology,
checkpoint files and runtime restoration. A computed cursor is not evidence
that subsequent batches or optimizer/RNG state were restored.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


FRESH_ONLY_CALLBACKS = frozenset({
    "nvidia_baseline", "native_lora_dropout_alignment", "original_first_updates", "native_initial_trainables", "native_first_updates",
})


def read_pinned_json(path: Path, sha256: str, *, maximum_bytes: int) -> dict:
    if (len(sha256) != 64 or any(c not in "0123456789abcdef" for c in sha256)
            or maximum_bytes <= 0):
        raise ValueError("invalid JSON identity or size bound")
    with path.open("rb") as stream:
        raw = stream.read(maximum_bytes + 1)
    if len(raw) > maximum_bytes or hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError("saved JSON size or SHA256 differs")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("saved JSON must be an object")
    return value


def plan_epoch_boundary_resume(
    state: dict, saved_args: dict, *, checkpoint_step: int, max_steps: int,
    world_size: int, per_device_batch: int, gradient_accumulation_steps: int,
    dataloader_batches_per_rank: int,
) -> dict:
    """Compute the pinned Trainer's full-epoch cursor, never a restore verdict.

    Only complete accumulation groups and an exact epoch boundary are supported.
    Mid-epoch checkpoints need their own data-skip and RNG witness before use.
    ``dataloader_batches_per_rank`` must come from the real loader; a dataset
    row count alone does not establish that length or subsequent input equality.
    """
    values = (checkpoint_step, max_steps, world_size, per_device_batch,
              gradient_accumulation_steps, dataloader_batches_per_rank)
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("resume dimensions and steps must be positive integers")
    if checkpoint_step >= max_steps:
        raise ValueError("checkpoint must precede the unchanged training endpoint")
    if dataloader_batches_per_rank % gradient_accumulation_steps:
        raise ValueError("partial accumulation groups require a separate witness")
    updates_per_epoch = dataloader_batches_per_rank // gradient_accumulation_steps
    if checkpoint_step % updates_per_epoch:
        raise ValueError("mid-epoch resume is outside this cursor contract")
    expected_epoch = checkpoint_step // updates_per_epoch
    for key, expected in (("global_step", checkpoint_step), ("max_steps", max_steps),
                          ("train_batch_size", per_device_batch)):
        if type(state.get(key)) is not int or state[key] != expected:
            raise ValueError(f"saved Trainer {key} differs")
    epoch = state.get("epoch")
    if type(epoch) not in (int, float) or epoch != expected_epoch:
        raise ValueError("saved epoch does not match the proposed loader cursor")
    for key, expected in (("max_steps", max_steps),
                          ("per_device_train_batch_size", per_device_batch),
                          ("gradient_accumulation_steps", gradient_accumulation_steps)):
        if type(saved_args.get(key)) is not int or saved_args[key] != expected:
            raise ValueError(f"saved recipe {key} differs")
    for key in ("ignore_data_skip", "resume_only_model"):
        if saved_args.get(key) is not False:
            raise ValueError(f"full resume requires explicit {key}=false")
    callbacks = saved_args.get("callbacks")
    if (not isinstance(callbacks, list)
            or any(not isinstance(name, str) or not name for name in callbacks)
            or len(set(callbacks)) != len(callbacks)):
        raise ValueError("saved callback identities are missing or ambiguous")
    return {
        "checkpoint_global_step": checkpoint_step,
        "unchanged_max_steps": max_steps,
        "world_size_to_verify": world_size,
        "per_device_batch": per_device_batch,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "dataloader_batches_per_rank_to_verify": dataloader_batches_per_rank,
        "updates_per_epoch": updates_per_epoch,
        "expected_epochs_trained": expected_epoch,
        "expected_batches_to_skip_in_resumed_epoch": 0,
        "remaining_optimizer_updates": max_steps - checkpoint_step,
        "planned_micro_forwards_per_rank": (max_steps - checkpoint_step)
            * gradient_accumulation_steps,
        "effective_batch": world_size * per_device_batch * gradient_accumulation_steps,
        "fresh_only_callbacks_requiring_replacement": sorted(
            set(callbacks).intersection(FRESH_ONLY_CALLBACKS)),
        "required_restore_witnesses": [
            "adapter parameter identities and values",
            "optimizer moments, steps and parameter mapping",
            "scheduler state and unchanged full-run schedule",
            "per-rank Python, NumPy, CPU and device RNG after native restore",
            "actual loader length, sampler epoch and accumulation boundary",
            "same subsequent inputs, loss and final state versus continuous run",
        ],
        "qualification": "Epoch-boundary cursor plan only; no model or checkpoint state loaded.",
        "subsequent_input_equality_verified": False,
        "recovery_qualified": False,
        "launch_ready": False,
        "formal_performance": False,
    }
