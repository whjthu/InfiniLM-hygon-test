"""Verify the exact read-only overlay with the canonical Transformers import."""

import hashlib
import json
import os
from pathlib import Path
import importlib.metadata

import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig

from build_overlay import LAYER_SOURCE_SHA256
import qwen_paired_layers


def run(model_type, route_count, enabled, prefix):
    torch.manual_seed(51329)
    cfg = Qwen3VLTextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=36,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, use_cache=False,
        max_position_embeddings=32, attention_dropout=0.0,
        rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2], "mrope_interleaved": True},
    )
    cfg._attn_implementation = "sdpa"
    model = model_type(cfg).train()
    model.gradient_checkpointing_enable({"use_reentrant": False, "preserve_rng_state": True})
    os.environ["INFINILM_QWEN_PAIRED_CHECKPOINTS"] = "1" if enabled else "0"
    state = torch.randn(1, 6, 32, requires_grad=True)
    mask = torch.tensor([[False, True, False, True, False, False]])
    visual = [torch.randn(2, 32, requires_grad=True) for _ in range(prefix)]
    before = len(route_count)
    output = model(inputs_embeds=state, attention_mask=torch.ones(1, 6, dtype=torch.long),
                   visual_pos_masks=mask, deepstack_visual_embeds=visual, use_cache=False)
    if len(route_count) != before + 1 or route_count[-1] is not enabled:
        raise AssertionError("actual text decoder did not select the requested route")
    if output.past_key_values is not None:
        raise AssertionError("unexpected cache")
    hidden = output.last_hidden_state
    loss = hidden.square().mean()
    loss.backward()
    grads = {name: param.grad for name, param in model.named_parameters() if param.grad is not None}
    if len(grads) < 200 or any(x.grad is None for x in visual):
        raise AssertionError("incomplete connected gradient set")
    return dict(hidden=hidden.detach(), loss=loss.detach(), state=state.grad,
                visual=dict(enumerate(v.grad for v in visual)), grad=grads, rng=torch.random.get_rng_state())


def error(a, b):
    if a.keys() != b.keys():
        raise AssertionError("different connected gradient set")
    return max((float((a[k] - b[k]).abs().max()) for k in a), default=0.0)


def main():
    from transformers.models.qwen3_vl import modeling_qwen3_vl as original

    provider = Path("/minicpm-transformers/transformers")
    src = provider / "models/qwen3_vl/modeling_qwen3_vl.py"
    layers = provider / "modeling_layers.py"
    if Path(original.__file__).resolve() != src.resolve():
        raise AssertionError("unexpected installed model module")
    overlay_sha = hashlib.sha256(src.read_bytes()).hexdigest()
    if overlay_sha != "40e50bc545325c01bc2a50c324ba161dc1d29aef04d09c58b73f859208f76013":
        raise AssertionError("canonical import did not resolve to the pinned private bind")
    if hashlib.sha256(layers.read_bytes()).hexdigest() != LAYER_SOURCE_SHA256:
        raise AssertionError("checkpoint wrapper provider identity changed")
    if importlib.metadata.version("ms-swift") != "4.4.2":
        raise AssertionError("unexpected Swift distribution")
    if original.Qwen3VLTextModel.__module__ != original.__name__:
        raise AssertionError("unexpected Qwen class module")
    candidate = original.Qwen3VLTextModel
    calls = []
    wrapped = qwen_paired_layers.try_paired_layers

    def record(*args, **kwargs):
        value = wrapped(*args, **kwargs)
        calls.append(value[0])
        return value

    qwen_paired_layers.try_paired_layers = record
    torch.set_num_threads(1)
    metrics = {}
    for visual_prefix in (0, 3):
        baseline = run(candidate, calls, False, visual_prefix)
        result = run(candidate, calls, True, visual_prefix)
        maximum = max(error(baseline["grad"], result["grad"]), error(baseline["visual"], result["visual"]),
                      error({k: baseline[k] for k in ("hidden", "loss", "state")},
                            {k: result[k] for k in ("hidden", "loss", "state")}))
        if maximum > 1e-6 or not torch.equal(baseline["rng"], result["rng"]):
            raise AssertionError(f"actual text decoder numerical comparison failed: {maximum}")
        metrics[visual_prefix] = dict(max_absolute=maximum, compared_parameter_gradients=len(result["grad"]))
    if torch.cuda.is_initialized():
        raise AssertionError("unexpected device initialization")
    print(json.dumps(dict(passed=True, varied_prefill_visual_injections=metrics, actual_native_text_class=True,
                          applied_routes=calls, accelerator_initialized=False, original_weights_opened=False,
                          overlay_SHA256=overlay_sha, canonical_module=original.__name__, ms_swift='4.4.2')))


if __name__ == "__main__":
    main()
