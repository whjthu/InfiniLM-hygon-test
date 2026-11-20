"""Count actual native Swift training forwards using the exact overlay policy."""
import atexit
import json
import os
from functools import partial
from pathlib import Path

from qwen_paired_layers_core import try_paired_layers as run_policy

_attempts = 0
_applied = 0


def try_paired_layers(model, *args, **kwargs):
    global _attempts, _applied
    root = os.environ.get("INFINILM_QWEN_PAIRED_AUDIT")
    if not root:
        raise RuntimeError("QWEN_POLICY_AUDIT_DIRECTORY_MISSING")

    import torch
    value = run_policy(model, *args, **kwargs)
    if not model.training or not torch.is_grad_enabled():
        return value

    rank = int(os.environ["RANK"])
    assert 0 <= rank < 8 and int(os.environ["WORLD_SIZE"]) == 8
    enabled = os.environ["INFINILM_QWEN_PAIRED_CHECKPOINTS"] == "1"
    path = Path(root)
    if _attempts == 0:
        checkpoint = getattr(model.layers[0], "_gradient_checkpointing_func", None)
        with (path / f"rank-{rank:02d}-first.json").open("x") as stream:
            json.dump(dict(rank=rank, enabled=enabled, applied=value[0],
                           checkpointing=getattr(model.layers[0],"gradient_checkpointing",None),
                           reentrant=checkpoint.keywords.get("use_reentrant") if type(checkpoint) is partial else None,
                           model_class=type(model).__name__), stream)

        def finish():
            with (path / f"rank-{rank:02d}-final.json").open("x") as stream:
                json.dump(dict(rank=rank, attempts=_attempts, applied=_applied, enabled=enabled), stream)

        atexit.register(finish)

    _attempts += 1
    _applied += int(value[0])
    if value[0] is not enabled:
        raise RuntimeError("ACTUAL_QWEN_SWIFT_CHECKPOINT_ROUTE_MISMATCH")
    return value
