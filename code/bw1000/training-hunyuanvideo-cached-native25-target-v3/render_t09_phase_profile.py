"""Insert opt-in phase marks in the exact native HunyuanVideo training step."""

import ast
from hashlib import sha256


BACKEND_SHA = "2331d0fa942eb3908dc5392df6008355524f013a0fbdb5067202f9a74400ccae"


def render(backend):
    if sha256(backend).hexdigest() != BACKEND_SHA:
        raise ValueError("Unexpected HunyuanVideo backend")
    original = backend.decode()
    head = "        def train_step(global_step: int) -> dict[str, float]:\n"
    expanded_head = (
        "        from .hunyuanvideo_phase_profile import profile_from_environment\n"
        "        phase_profile = profile_from_environment(\n"
        "            torch, output_dir, warmup=warmup_steps, measured=max_steps,\n"
        '            restored_step=completed_global_step, recovery=config.get("validate_recovery", False),\n'
        "        )\n\n" + head
    )
    phases = (
        "            loss.backward()\n"
        "            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)\n"
        "            optimizer.step()\n"
        "            synchronize()\n"
        "            backward_seconds = perf_counter() - backward_started\n"
    )
    expanded_phases = (
        "            phase_record = phase_profile.begin(global_step) if phase_profile is not None else None\n"
        "            loss.backward()\n"
        "            if phase_record is not None:\n"
        "                phase_record.mark()\n"
        "            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)\n"
        "            if phase_record is not None:\n"
        "                phase_record.mark()\n"
        "            optimizer.step()\n"
        "            if phase_record is not None:\n"
        "                phase_record.mark()\n"
        "            synchronize()\n"
        "            backward_seconds = perf_counter() - backward_started\n"
        "            if phase_record is not None:\n"
        "                phase_record.finish(backward_seconds)\n"
    )
    if original.count(head) != 1 or original.count(phases) != 1:
        raise ValueError("Original training phase anchors changed")
    modified = original.replace(head, expanded_head, 1).replace(phases, expanded_phases, 1)
    if modified.replace(expanded_head, head, 1).replace(expanded_phases, phases, 1) != original:
        raise ValueError("Profiling changed the original backend outside the inserted marks")
    ast.parse(modified)
    return modified.encode()
