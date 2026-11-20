"""Qualification on the project's pinned CPU-capable HIP torch and Transformers providers."""

import hashlib
import json
import os
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers.modeling_layers import GradientCheckpointingLayer

import build_overlay
from qwen_paired_layers import try_paired_layers


class Decoder(GradientCheckpointingLayer):
    def __init__(self, policy):
        super().__init__()
        self.gradient_checkpointing = True
        self._gradient_checkpointing_func = policy
        self.projection = nn.Linear(8, 8)
        self.dropout = nn.Dropout(0.05)

    def forward(self, state, attention_mask, position_ids, past_key_values, position_embeddings):
        if past_key_values is not None:
            raise AssertionError("unexpected cache on checkpointed path")
        return state + self.projection(self.dropout(state + position_embeddings)) / 8


class Model(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.layers = nn.ModuleList(Decoder(policy) for _ in range(36))
        self.config = SimpleNamespace()

    def _deepstack_process(self, hidden, mask, addition):
        hidden = hidden.clone()
        hidden[mask, :] = hidden[mask, :] + addition
        return hidden


def call(model, state, positions, visual_mask, embeds, **overrides):
    options = dict(cache=None, use_cache=False, kwargs={})
    options.update(overrides)
    return try_paired_layers(model, state, None, None, positions, options["cache"],
                             options["use_cache"], visual_mask, embeds, options["kwargs"],
                             decoder_class=Decoder, checkpoint_base=GradientCheckpointingLayer)


def run(grouped, prefix):
    torch.manual_seed(19383)
    policy = partial(checkpoint, use_reentrant=False, preserve_rng_state=True)
    model = Model(policy)
    state = torch.randn(4, 8, requires_grad=True)
    positions = torch.randn(4, 8, requires_grad=True)
    mask = torch.tensor([False, True, False, True])
    embeds = [torch.randn(2, 8, requires_grad=True) for _ in range(prefix)]
    if grouped:
        used, output = call(model, state, positions, mask, embeds)
        if not used:
            raise AssertionError("candidate route did not execute")
    else:
        output = state
        for index, layer in enumerate(model.layers):
            output = layer(output, attention_mask=None, position_ids=None,
                           past_key_values=None, position_embeddings=positions)
            if index < prefix:
                output = model._deepstack_process(output, mask, embeds[index])
    loss = output.square().mean()
    loss.backward()
    named = [p.grad for p in model.parameters()]
    return [output, loss, state.grad, positions.grad, *(e.grad for e in embeds), *named,
            torch.random.get_rng_state()]


def maximum_test_error(a, b):
    if len(a) != len(b) or any(left is None or right is None for left, right in zip(a,b,strict=True)):
        raise AssertionError("missing comparison item")
    return max(float((left.float() - right.float()).abs().max())
               for left, right in zip(a, b, strict=True))


def verify_fallbacks():
    state = torch.ones(4, 8, requires_grad=True)
    pos = state.clone()
    model = Model(partial(checkpoint, use_reentrant=False))
    no_use = []
    def rejected(value):
        used, result = value
        no_use.append(used is False and result is state)
    rejected(call(model, state, pos, None, None, cache=object()))
    rejected(call(model, state, pos, None, None, use_cache=True))
    rejected(call(model, state, pos, None, None, kwargs={"output_hidden_states": True}))
    rejected(call(model, state, pos, None, [state, state]))
    hook = model.layers[1].projection.register_forward_hook(lambda *_: None)
    rejected(call(model, state, pos, None, None))
    hook.remove()
    with torch.no_grad():
        rejected(call(model, state, pos, None, None))
    model.layers[1]._gradient_checkpointing_func = partial(checkpoint, use_reentrant=True)
    rejected(call(model, state, pos, None, None))
    model.eval()
    rejected(call(model, state, pos, None, None))
    model.train()
    os.environ.pop("INFINILM_QWEN_PAIRED_CHECKPOINTS")
    rejected(call(model, state, pos, None, None))
    if not all(no_use):
        raise AssertionError("one unsafe route did not fall back")
    return len(no_use)


def main():
    provider = Path("/minicpm-transformers/transformers")
    model_source = provider / "models/qwen3_vl/modeling_qwen3_vl.py"
    layer_source = provider / "modeling_layers.py"
    overlay = build_overlay.render(model_source.read_bytes(), layer_source.read_bytes())
    torch.set_num_threads(1)
    os.environ["INFINILM_QWEN_PAIRED_CHECKPOINTS"] = "1"
    bounds = {n: maximum_test_error(run(False,n), run(True,n)) for n in (0, 1, 3)}
    if max(bounds.values()) > 1e-7:
        raise AssertionError(f"paired errors {bounds}")
    rejection_count = verify_fallbacks()
    if torch.cuda.is_initialized():
        raise AssertionError("unexpected device initialization")
    print(json.dumps(dict(passed=True, overlay_SHA256=hashlib.sha256(overlay).hexdigest(),
                          varied_prefill_visual_injections=bounds, rejected=rejection_count,
                          torch_version=str(torch.__version__), hip=str(torch.version.hip),
                          accelerator_initialized=False, original_weights_opened=False)))


if __name__ == "__main__":
    main()
