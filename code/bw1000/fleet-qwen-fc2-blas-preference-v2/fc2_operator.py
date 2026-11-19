"""Compare BLAS preferences with the exact Qwen vision FC2 geometry."""
import math
import statistics
import time

import torch
from torch.nn import functional as F

CALLS_PER_BLOCK = 3072
BATCH = 32
ORDER = (0, 1, 1, 0, 1, 0, 0, 1)
INPUTS = 4
BACKENDS = ("cublas", "cublaslt")


def exercise(device, rank, measured, deadline, calls_per_block=CALLS_PER_BLOCK):
    if calls_per_block <= 0 or calls_per_block % (BATCH * INPUTS):
        raise ValueError("GRAPH_COMPLETE_INPUT_CYCLES")
    cfg = torch.backends.cuda.preferred_blas_library
    inherited = cfg()
    expected = (
        torch.backends.cuda._BlasBackends["cublas"],
        torch.backends.cuda._BlasBackends["cublaslt"],
    )
    generator = torch.Generator(device=device).manual_seed(90831 + rank)
    weights = torch.randn((1152, 4304), device=device, generator=generator).mul_(.015).to(torch.bfloat16)
    bias = torch.randn((1152,), device=device, generator=generator).mul_(.01).to(torch.bfloat16)
    inputs = [
        torch.randn((640, 4304), device=device, generator=generator).mul_(.25).to(torch.bfloat16).requires_grad_()
        for _ in range(INPUTS)
    ]
    rows = []

    def set_preference(arm):
        if cfg(BACKENDS[arm]) != expected[arm] or cfg() != expected[arm]:
            raise RuntimeError("GRAPH_BLAS_PREFERENCE_READBACK")

    try:
        with torch.enable_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            set_preference(0)
            anchors = [F.linear(x, weights, bias).detach() for x in inputs]
            if any(tuple(a.shape) != (640, 1152) or a.dtype != torch.bfloat16
                   or not bool(torch.isfinite(a).all().item()) for a in anchors):
                raise RuntimeError("GRAPH_FC2_ANCHOR")
            anchor_bits = [a.view(torch.int16) for a in anchors]
            for idx, arm in enumerate(ORDER if rank % 2 == 0 else tuple(1 - a for a in ORDER)):
                if time.time() >= deadline:
                    raise TimeoutError("GRAPH_COMPONENT_BUDGET")
                set_preference(arm)
                for x in inputs:
                    F.linear(x, weights, bias)
                torch.cuda.synchronize(device)
                if idx == 0:
                    measured()
                changed = torch.zeros((), dtype=torch.int64, device=device)
                changed_calls = torch.zeros_like(changed)
                max_abs = torch.zeros((), dtype=torch.float32, device=device)
                device_ms = []
                events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                start = time.monotonic()
                for off in range(0, calls_per_block, BATCH):
                    if time.time() >= deadline:
                        raise TimeoutError("GRAPH_COMPONENT_BUDGET")
                    slot = (off // BATCH) % INPUTS
                    events[0].record()
                    outs = [F.linear(inputs[slot], weights, bias) for _ in range(BATCH)]
                    events[1].record()
                    events[1].synchronize()
                    device_ms.append(events[0].elapsed_time(events[1]))
                    if not all(y.requires_grad and y.dtype == torch.bfloat16 for y in outs):
                        raise RuntimeError("GRAPH_FC2_OUTPUT_CONTEXT")
                    with torch.no_grad():
                        values = torch.stack(outs)
                        counts = (values.view(torch.int16) != anchor_bits[slot]).sum(dim=(1, 2))
                        changed += counts.sum()
                        changed_calls += (counts != 0).sum()
                        max_abs = torch.maximum(max_abs, (values.float() - anchors[slot].float()).abs().amax())
                    del values, counts, outs
                torch.cuda.synchronize(device)
                magnitude = max_abs.item()
                if not math.isfinite(magnitude) or not all(math.isfinite(x) and x > 0 for x in device_ms):
                    raise RuntimeError("GRAPH_NONFINITE_OUTPUT_OR_EVENT")
                rows.append(dict(arm=BACKENDS[arm], observed_preference=str(cfg()), calls=calls_per_block,
                                 changed_calls=changed_calls.item(), changed_elements=changed.item(),
                                 max_abs_against_first_cublas=magnitude, linear_event_ms=sum(device_ms),
                                 median_batch_event_ms=statistics.median(device_ms),
                                 wall_with_comparisons_seconds=time.monotonic() - start))
    finally:
        cfg(inherited)
        if cfg() != inherited:
            raise RuntimeError("GRAPH_BLAS_PREFERENCE_RESTORE")
    return dict(rank=rank, runtime=dict(torch=torch.__version__, hip=torch.version.hip),
                preference_is_dispatch_proof=False, input_kind="deterministic_synthetic_BF16",
                exact_original_FC2_shapes=[[640, 4304], [1152, 4304], [1152], [640, 1152]],
                blocks=rows, restored_preference=str(cfg()), compiled_kernel_identity_proven=False)
