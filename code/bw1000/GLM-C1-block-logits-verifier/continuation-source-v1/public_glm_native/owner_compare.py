"""Private C1 comparison on a loaded GLM952 owner; no model or job launcher."""

import ctypes
import hashlib
import json
import math
from time import perf_counter

CONTEXT = 32768
VOCAB = 154880
OWNERS = 16
PIPELINE_FIELDS = (
    "host_incoming", "host_layers", "host_outgoing_final_sync",
    "stream_incoming", "stream_layers", "stream_outgoing",
)


def run_loaded(*, lib, ctx, torch, stage, rank, control, ptr, stream,
               prompt, chunk_widths, proposal, capture_pipeline=False):
    """All 16 owners must call on the owning thread with an identical plan.

    The caller owns watchdog, context and cleanup. If Native or transport raises,
    unwind the diagnostic; do not enter another collective from cleanup.
    """
    if stage not in (0, 1) or not 0 <= rank < 8:
        raise ValueError("expected TP8 PP2 owner")
    prompt, chunk_widths, proposal = map(tuple, (prompt, chunk_widths, proposal))
    valid = (
        bool(prompt)
        and len(proposal) == 8
        and len(prompt) + 8 <= CONTEXT
        and all(type(w) is int and 1 <= w <= 512 for w in chunk_widths)
        and sum(chunk_widths) == len(prompt)
        and all(type(token) is int and 0 <= token < VOCAB for token in prompt + proposal)
    )
    device = f"cuda:{rank}"

    def reduce(values):
        tensor = torch.tensor(values, device=device, dtype=torch.float32)
        control(tensor)
        return tensor.cpu().tolist()

    def unanimous(ok):
        return reduce([int(ok)]) == [OWNERS]

    if not unanimous(valid):
        raise ValueError("invalid prompt, chunks, or eight-token proposal")
    digest = hashlib.sha256(json.dumps(
        (prompt, chunk_widths, proposal), separators=(",", ":")
    ).encode("ascii")).digest()
    agreement = hashlib.sha256(b'pipeline-profile\0' + digest).digest() if capture_pipeline else digest
    if not unanimous(reduce(list(agreement)) == [OWNERS * b for b in agreement]):
        raise ValueError("owners disagree on comparison plan or pipeline profile mode")

    getter = None
    if capture_pipeline:
        try:
            getter = lib.modern_glm_pipeline_last_profile
            getter.argtypes = []
            getter.restype = ctypes.POINTER(ctypes.c_double)
        except AttributeError:
            getter = None
        if not unanimous(getter is not None):
            raise RuntimeError("pipeline profile symbol missing on an owner")
    pipeline = []

    def reset():
        if not unanimous(lib.modern_glm_model_reset(ctx) == 0):
            raise RuntimeError("owner reset failed")

    def forward(past, values, all_positions, sample=None):
        width = len(values)
        desc = (ctypes.c_size_t * 3)(0, past, width)
        ids = torch.tensor(values if stage == 0 else (0,) * width,
                           device=device, dtype=torch.int64)
        positions = torch.arange(past, past + width, device=device, dtype=torch.int64)
        rows = width if all_positions else 1
        output = torch.full((rows, VOCAB), float("nan"),
                            device=device, dtype=torch.bfloat16)
        torch.cuda.synchronize(rank)
        started = perf_counter()
        rc = lib.modern_glm_model_forward(
            ctx, width, 1, desc, ptr(ids), ptr(positions), ptr(output),
            int(all_positions), stream()
        )
        elapsed = perf_counter() - started
        profile_ok = True
        if getter is not None and sample is not None:
            profile_ok = False
            if rc == 0:
                before = perf_counter()
                values_ptr = getter()
                if values_ptr:
                    values_s = [float(values_ptr[i]) for i in range(len(PIPELINE_FIELDS))]
                    profile_ok = (all(math.isfinite(v) and 0 <= v <= elapsed for v in values_s)
                                  and values_s[1] > 0 and values_s[4] > 0)
                    if profile_ok:
                        pipeline.append(dict(sample=sample, past=past, width=width,
                                             values_s=values_s, getter_wall_s=perf_counter() - before))
        finite = stage == 0 or bool(torch.isfinite(output).all().item())
        if not unanimous(rc == 0 and finite and profile_ok):
            raise RuntimeError("owner forward failed, tail did not write finite logits, or pipeline profile invalid")
        return output, elapsed

    def begin_arm():
        reset()
        past = 0
        for width in chunk_widths:
            forward(past, prompt[past:past + width], False)
            past += width

    begin_arm()
    serial, serial_times = [], []
    for offset in range(8):
        output, elapsed = forward(len(prompt) + offset, proposal[offset:offset + 1], False, offset)
        if stage == 1:
            serial.append(output[0].float())
        serial_times.append(elapsed)

    blocks = []
    for width in (4, 8):
        begin_arm()
        output, elapsed = forward(len(prompt), proposal[:width], True, f"block{width}")
        stats = []
        if stage == 1:
            for reference, row in zip(serial, output):
                row = row.float()
                delta = row - reference
                stats.append({
                    "same_top1": row.argmax().item() == reference.argmax().item(),
                    "max_abs": delta.abs().max().item(),
                    "rmse": delta.square().mean().sqrt().item(),
                    "reference_rms": reference.square().mean().sqrt().item(),
                })
        parity = unanimous(all(row["same_top1"] for row in stats))
        blocks.append({
            "width": width, "native_capi_s": elapsed,
            "all_tail_top1_equal": parity, "tail_row_stats": stats,
        })
    reset()
    return {
        "stage": stage, "rank": rank, "plan_sha256": digest.hex(),
        "prefix_tokens": len(prompt), "prefix_chunks": len(chunk_widths),
        "serial_native_capi_s": serial_times, "blocks": blocks, "final_reset": True,
        "scope": "fixed proposals, fresh equal prefixes, no rollback or MTP",
        **({"pipeline_profile": {"fields": PIPELINE_FIELDS, "samples": pipeline}} if capture_pipeline else {}),
    }
