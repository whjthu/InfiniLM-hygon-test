"""Private GLM block stage and commit on the loaded original owner thread."""

import ctypes
from hashlib import sha256
import json
from time import perf_counter

from .accepted_prefix_commit import StagedPrefix, agree_ready, commit_after_forward_drain
from .owner_compare import VOCAB


def stage_and_commit(*, lib, ctx, torch, stage, rank, control, ptr, stream,
                     sequence, past, proposal, accepted):
    """Return original tail logits and clocks after all 16 owners confirm commit.

    The caller owns the original fatal cleanup. Do not call another forward
    when this routine raises. The enclosing context must have loaded its
    prefix and carry the matching private staged-prefix Native.
    """
    device = f"cuda:{rank}"

    def reduce(values):
        frame = torch.tensor(values, device=device, dtype=torch.float32)
        control(frame)
        return frame.cpu().tolist()

    start = perf_counter()
    identity = bytes(32)
    setup_error = None
    try:
        proposal = tuple(proposal)
        if (type(stage) is not int or type(rank) is not int or stage not in (0, 1)
                or not 0 <= rank < 8
                or any(type(token) is not int or not 0 <= token < VOCAB for token in proposal)):
            raise ValueError("unexpected original owner or proposal")
        plan = StagedPrefix(sequence, past, len(proposal), accepted, sha256(
            json.dumps(proposal, separators=(",", ":")).encode("ascii")).hexdigest())
        identity = plan.vote_identity()
        stage_block = lib.modern_glm_block_stage_v1
        prepare = lib.modern_glm_block_prepare_v1
        publish = lib.modern_glm_block_publish_v1
        mutation_types = [ctypes.c_void_p, ctypes.c_uint64, *([ctypes.c_size_t] * 3)]
        stage_block.argtypes = mutation_types[:-1] + [ctypes.c_void_p] * 4
        for operation in (stage_block, prepare, publish):
            operation.restype = ctypes.c_int
        prepare.argtypes = publish.argtypes = mutation_types
        ids = torch.tensor(proposal if stage == 0 else [0] * plan.width,
                           device=device, dtype=torch.int64)
        positions = torch.arange(plan.past, plan.past + plan.width, device=device, dtype=torch.int64)
        logits = torch.full((plan.width, VOCAB), float("nan"), device=device, dtype=torch.bfloat16)
        data = (ptr(ids), ptr(positions), ptr(logits), stream())
        torch.cuda.synchronize(rank)
    except Exception as error:
        setup_error = error
    setup_done = perf_counter()
    if not agree_ready(reduce, identity, setup_error is None):
        raise RuntimeError("GLM block stage setup or 16-owner agreement failed") from setup_error

    stage_error = None
    stage_ok = False
    tick = perf_counter()
    try:
        rc = stage_block(ctx, plan.sequence, plan.past, plan.width, *data)
        capi_s = perf_counter() - tick
        stage_ok = rc == 0 and (stage == 0 or bool(torch.isfinite(logits).all().item()))
    except Exception as error:
        stage_error = error

    def apply(operation, p):
        return operation(ctx, p.sequence, p.past, p.width, p.accepted) == 0

    def preflight(p):
        if stage_error is not None:
            raise stage_error
        return stage_ok and apply(prepare, p)

    commit_start = perf_counter()
    end = commit_after_forward_drain(plan, reduce=reduce, can_publish=preflight,
                                     publish=lambda p: apply(publish, p))
    finished = perf_counter()
    return (logits if stage == 1 else None), {
        "sequence": plan.sequence, "staged_width": plan.width, "accepted": plan.accepted,
        "staged_input_sha256": plan.staged_input_sha256, "committed_end": end,
        "owner_setup_s": setup_done - start, "prestage_agreement_s": tick - setup_done,
        "block_CAPI_s": capi_s, "postforward_checks_s": commit_start - tick - capi_s,
        "prepare_publish_control_s": finished - commit_start, "owner_total_s": finished - start,
    }
