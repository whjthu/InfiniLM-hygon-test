"""Verify the actual patched Qwen text class in a private Python process."""

import hashlib
import json
import os
from pathlib import Path
import sys
import types

import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig

import build_overlay
import qwen_paired_layers


def load_candidate(patched, filename):
    package = "transformers.models.qwen3_vl"
    module = types.ModuleType(package + ".private_checkpoint_comparison")
    module.__package__ = package
    module.__file__ = str(filename)
    sys.modules[module.__name__] = module
    exec(compile(patched, str(filename), "exec"), vars(module))
    return module.Qwen3VLTextModel


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
    patched = build_overlay.render(src.read_bytes(), layers.read_bytes())
    candidate = load_candidate(patched, src)
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
                          overlay_SHA256=hashlib.sha256(patched).hexdigest())))


if __name__ == "__main__":
    main()
