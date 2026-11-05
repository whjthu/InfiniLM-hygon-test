"""Compare GLM continuation after a block rewind with serial fresh-prefix continuation."""

import ctypes
from hashlib import sha256
import json
from time import perf_counter

from .accepted_prefix_commit import StagedPrefix, agree_ready
from .accepted_prefix_owner import stage_and_commit
from .owner_compare import CONTEXT, OWNERS, VOCAB

EOS = {154820, 154827, 154829}


def compare_continuation(*, lib, ctx, torch, stage, rank, control, ptr, stream,
                         sequence, prompt, chunk_widths, proposal, accepted, continuation):
    """One existing Context, two original resets before arms and one final reset.

    Every owner supplies the same immutable plan. A failed call must go to
    the original owner failure path; this function never retries or reloads.
    """
    device = f"cuda:{rank}"

    def reduce(values):
        frame = torch.tensor(values, device=device, dtype=torch.float32)
        control(frame)
        return frame.cpu().tolist()

    def unanimous(ok):
        return reduce([int(ok)]) == [OWNERS]

    identity = bytes(32)
    plan_error = None
    try:
        prompt, chunk_widths, proposal, continuation = map(tuple, (prompt, chunk_widths, proposal, continuation))
        if (type(stage) is not int or type(rank) is not int or stage not in (0, 1) or not 0 <= rank < 8
                or not prompt or not chunk_widths
                or any(type(w) is not int or not 1 <= w <= 512 for w in chunk_widths)
                or sum(chunk_widths) != len(prompt)
                or any(type(t) is not int or not 0 <= t < VOCAB for t in prompt + proposal + continuation)
                or set(proposal + continuation) & EOS
                or not 1 <= len(continuation) <= 2
                or len(prompt) + accepted + len(continuation) > CONTEXT
                or any(a == b for a, b in zip(proposal[accepted:], continuation))):
            raise ValueError("invalid continuation diagnostic geometry")
        StagedPrefix(sequence, len(prompt), len(proposal), accepted, "0" * 64).vote_identity()
        identity = sha256(json.dumps(
            (sequence, prompt, chunk_widths, proposal, accepted, continuation),
            separators=(",", ":"),
        ).encode("ascii")).digest()
    except Exception as error:
        plan_error = error
    if not agree_ready(reduce, identity, plan_error is None):
        raise RuntimeError("GLM rollback continuation plan disagreement") from plan_error

    def reset():
        error = None
        ok = False
        try:
            ok = lib.modern_glm_model_reset(ctx) == 0
        except Exception as exc:
            error = exc
        if not unanimous(ok):
            raise RuntimeError("GLM original reset failed") from error

    def forward(past, values):
        setup_error = None
        try:
            width = len(values)
            desc = (ctypes.c_size_t * 3)(0, past, width)
            ids = torch.tensor(values if stage == 0 else [0] * width, device=device, dtype=torch.int64)
            pos = torch.arange(past, past + width, device=device, dtype=torch.int64)
            output = torch.full((1, VOCAB), float("nan"), device=device, dtype=torch.bfloat16)
            data = (ptr(ids), ptr(pos), ptr(output), stream())
            torch.cuda.synchronize(rank)
        except Exception as error:
            setup_error = error
        if not unanimous(setup_error is None):
            raise RuntimeError("GLM continuation forward setup failed") from setup_error
        call_error = None
        ok = False
        try:
            started = perf_counter()
            rc = lib.modern_glm_model_forward(ctx, width, 1, desc, *data[:3], 0, data[3])
            elapsed = perf_counter() - started
            ok = rc == 0 and (stage == 0 or bool(torch.isfinite(output).all().item()))
        except Exception as error:
            call_error = error
        if not unanimous(ok):
            raise RuntimeError("GLM continuation Native or finite check failed") from call_error
        return output, elapsed

    def prefill():
        reset()
        past = 0
        for width in chunk_widths:
            forward(past, prompt[past:past + width])
            past += width

    past = len(prompt)
    accepted_end = past + accepted
    prefill()
    for offset, value in enumerate(proposal[:accepted]):
        forward(past + offset, (value,))
    reference = [forward(accepted_end + i, (value,)) for i, value in enumerate(continuation)]

    prefill()
    _, block = stage_and_commit(lib=lib, ctx=ctx, torch=torch, stage=stage, rank=rank,
                                control=control, ptr=ptr, stream=stream, sequence=sequence,
                                past=past, proposal=proposal, accepted=accepted)
    candidate = [forward(accepted_end + i, (value,)) for i, value in enumerate(continuation)]
    statistics = []
    stats_error = None
    try:
        if stage == 1:
            for (ref, _), (actual, _) in zip(reference, candidate, strict=True):
                ref, actual = ref[0].float(), actual[0].float()
                delta = actual - ref
                statistics.append({
                    "same_top1": actual.argmax().item() == ref.argmax().item(),
                    "max_abs": delta.abs().max().item(),
                    "rmse": delta.square().mean().sqrt().item(),
                    "reference_rms": ref.square().mean().sqrt().item(),
                })
    except Exception as error:
        stats_error = error
    votes = reduce([int(stats_error is None), int(stats_error is None and
                 all(row["same_top1"] for row in statistics))])
    if votes[0] != OWNERS:
        raise RuntimeError("GLM continuation logit comparison failed") from stats_error
    equal = votes[1] == OWNERS
    reset()
    return {
        "plan_sha256": identity.hex(), "prefix_tokens": past, "width": len(proposal),
        "accepted": accepted, "continuation_tokens": len(continuation),
        "first_continuation_past": accepted_end, "final_reset": True,
        "block": block, "all_tail_continuation_top1_equal": equal,
        "tail_continuation_logits": statistics,
        "serial_continuation_CAPI_s": [elapsed for _, elapsed in reference],
        "rollback_continuation_CAPI_s": [elapsed for _, elapsed in candidate],
        "scope": "Fixed token continuation and original owner CAPI clocks, no draft acceptance or service rate.",
    }
