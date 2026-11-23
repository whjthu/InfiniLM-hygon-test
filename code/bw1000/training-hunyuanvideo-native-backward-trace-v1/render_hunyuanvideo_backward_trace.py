"""Insert a single complete backward capture in the frozen T09 training backend."""

import ast
from hashlib import sha256


BACKEND_SHA = "2331d0fa942eb3908dc5392df6008355524f013a0fbdb5067202f9a74400ccae"


def render(backend):
    if sha256(backend).hexdigest() != BACKEND_SHA:
        raise ValueError("Unexpected original HunyuanVideo training backend")
    head = "        def train_step(global_step: int) -> dict[str, float]:\n"
    initialization = (
        "        from .hunyuanvideo_backward_trace import STEP as TRACE_STEP, trace_from_environment\n"
        "        backward_trace = trace_from_environment(\n"
        "            torch, output_dir, warmup=warmup_steps, measured=max_steps,\n"
        '            restored_step=completed_global_step, recovery=config.get("validate_recovery", False),\n'
        "        )\n\n"
    )
    original_call = "            loss.backward()\n"
    profiled_call = (
        "            if backward_trace is not None and global_step == TRACE_STEP:\n"
        "                with backward_trace.capture():\n"
        "                    loss.backward()\n"
        "            else:\n"
        "                loss.backward()\n"
    )
    original = backend.decode()
    if original.count(head) != 1 or original.count(original_call) != 1:
        raise ValueError("Original backward or training step source anchor changed")
    modified = original.replace(head, initialization + head, 1).replace(original_call, profiled_call, 1)
    if modified.replace(initialization + head, head, 1).replace(profiled_call, original_call, 1) != original:
        raise ValueError("Source outside the marked backward changed")
    ast.parse(modified)
    return modified.encode()
