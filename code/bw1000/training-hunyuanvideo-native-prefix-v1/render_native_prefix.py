"""Insert the opt-in comparison at the exact original T09 training boundaries."""

from hashlib import sha256
import ast


ORIGINAL_SHA = "2331d0fa942eb3908dc5392df6008355524f013a0fbdb5067202f9a74400ccae"


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError("Frozen original training anchor differs")
    return source.replace(old, new, 1)


def render(backend):
    if sha256(backend).hexdigest() != ORIGINAL_SHA:
        raise ValueError("Frozen original HunyuanVideo backend differs")
    original = backend.decode()
    pairs = (
        (
            "        def train_step(global_step: int) -> dict[str, float]:\n",
            "        from .hunyuanvideo_native_prefix import prefix_session_from_environment\n"
            "        prefix_session = prefix_session_from_environment(\n"
            "            torch, output_dir, transformer, optimizer, config=config, cache=cache,\n"
            "            warmup=warmup_steps, measured=max_steps, restored=completed_global_step,\n"
            "        )\n\n"
            "        def train_one(global_step: int) -> dict[str, float]:\n",
        ),
        (
            "            hidden_states = torch.cat((latents_noised, cond), dim=1)\n",
            "            hidden_states = torch.cat((latents_noised, cond), dim=1)\n"
            "            if prefix_session is not None:\n"
            "                prefix_session.inputs(noise=noise, timestep=timestep, hidden_states=hidden_states, target=target)\n",
        ),
        (
            "                loss = torch.nn.functional.mse_loss(prediction, target)\n",
            "                loss = torch.nn.functional.mse_loss(prediction, target)\n"
            "            if prefix_session is not None:\n"
            "                prefix_session.forward(prediction, loss)\n",
        ),
        (
            "            loss.backward()\n"
            "            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)\n"
            "            optimizer.step()\n",
            "            loss.backward()\n"
            "            if prefix_session is not None:\n"
            "                prefix_session.backward()\n"
            "            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)\n"
            "            if prefix_session is not None:\n"
            '                prefix_session.gradients("clipped_gradients")\n'
            "            optimizer.step()\n"
            "            if prefix_session is not None:\n"
            "                prefix_session.updated()\n",
        ),
        (
            "        adapter_before = _adapter_sha(transformer)\n",
            "        def train_step(global_step: int) -> dict[str, float]:\n"
            "            if prefix_session is None:\n"
            "                return train_one(global_step)\n"
            "            return prefix_session.run(global_step, train_one)\n\n"
            "        adapter_before = _adapter_sha(transformer)\n",
        ),
    )
    modified = original
    for old, new in pairs:
        modified = replace_once(modified, old, new)
    restored = modified
    for old, new in reversed(pairs):
        restored = replace_once(restored, new, old)
    if restored != original:
        raise ValueError("Native prefix modified source outside the frozen hooks")
    ast.parse(modified)
    return modified.encode()
