"""Opt-in checkpoint groups for the pinned Transformers Qwen3-VL text decoder."""

from functools import partial
import os


def _has_execution_hooks(module):
    return bool(module._forward_hooks or module._forward_pre_hooks or module._backward_hooks
                or module._backward_pre_hooks)


def try_paired_layers(model, hidden, attention_mask, position_ids, position_embeddings, cache,
                      use_cache, visual_mask, visual_embeddings, kwargs, *, decoder_class, checkpoint_base):
    """Return (applied, hidden). A rejected candidate leaves the stock loop in control."""
    if os.environ.get("INFINILM_QWEN_PAIRED_CHECKPOINTS") != "1" or not model.training:
        return False, hidden

    import torch
    from torch.nn.modules import module as hooks
    from torch.utils.checkpoint import checkpoint

    if (not torch.is_grad_enabled() or torch.jit.is_tracing() or cache is not None or use_cache
            or any(kwargs.get(k) is not None for k in ("past_key_values", "past_key_value", "layer_past"))
            or kwargs.get("use_cache")
            or any(kwargs.get(k) or getattr(model.config, k, False)
                   for k in ("output_attentions", "output_hidden_states", "output_router_logits"))
            or any(getattr(hooks, name, None) for name in ("_global_forward_hooks", "_global_forward_pre_hooks",
                                                         "_global_backward_hooks", "_global_backward_pre_hooks"))):
        return False, hidden

    layers = model.layers
    if len(layers) != 36:
        return False, hidden
    prefix = 0 if visual_embeddings is None else len(visual_embeddings)
    if prefix > len(layers) - 2 or (prefix and visual_mask is None):
        return False, hidden
    pair_end = prefix + (len(layers) - prefix) // 2 * 2
    first = layers[prefix]
    checkpoint_fn = getattr(first, "_gradient_checkpointing_func", None)
    if (type(checkpoint_fn) is not partial or checkpoint_fn.func is not checkpoint or checkpoint_fn.args
            or checkpoint_fn.keywords.get("use_reentrant") is not False
            or checkpoint_fn.keywords.get("preserve_rng_state", True) is not True):
        return False, hidden
    if not all(type(layers[i]) is decoder_class and layers[i].__class__.__mro__[1] is checkpoint_base
               and layers[i].training and layers[i].gradient_checkpointing
               and getattr(layers[i], "_gradient_checkpointing_func", None) is checkpoint_fn
               and not any(_has_execution_hooks(m) for m in layers[i].modules())
               for i in range(prefix, pair_end)):
        return False, hidden

    common = dict(attention_mask=attention_mask, position_ids=position_ids, past_key_values=None,
                  position_embeddings=position_embeddings, **kwargs)
    for index in range(prefix):
        hidden = layers[index](hidden, **common)
        hidden = model._deepstack_process(hidden, visual_mask, visual_embeddings[index])

    def apply_pair(left, right, state):
        # Retain nn.Module.__call__ on both layers while the outer checkpoint owns recomputation.
        state = super(checkpoint_base, left).__call__(state, **common)
        return super(checkpoint_base, right).__call__(state, **common)

    for index in range(prefix, pair_end, 2):
        pair = partial(apply_pair, layers[index], layers[index + 1])
        hidden = checkpoint_fn(pair, hidden)
    for index in range(pair_end, len(layers)):
        hidden = layers[index](hidden, **common)
    return True, hidden
