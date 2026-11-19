"""Compare allocated and reused zero padding at the measured Qwen v_proj shape."""
import math
import statistics
import time

import torch
from torch.nn.functional import linear, pad

TOKENS, WIDTH, OUT = 186, 4096, 1024
ALIGNED = 192
ARMS = ("alloc_pad", "reuse_zero_tail")


def exercise(device, rank, measured, deadline, calls_per_block=32, cases=384):
    if calls_per_block not in (8, 32) or cases not in (2, 384):
        raise ValueError("GRAPH_CASE_COUNTS")
    api = torch.backends.cuda.preferred_blas_library
    inherited = api()
    expected = torch.backends.cuda._BlasBackends["cublas"]
    generator = torch.Generator(device=device).manual_seed(90209461 + rank)
    torch.set_float32_matmul_precision("highest")

    def budget():
        if time.time() >= deadline:
            raise TimeoutError("GRAPH_COMPONENT_BUDGET")

    def operand(shape, scale):
        return torch.randn(shape, device=device, generator=generator).mul_(scale).bfloat16()

    def invoke(x, weight, work, reuse):
        if reuse:
            work[:, :TOKENS, :].copy_(x)
            result = linear(work, weight)
        else:
            result = linear(pad(x, (0, 0, 0, ALIGNED - TOKENS)), weight)
        return result

    summary = []
    try:
        if api("cublas") != expected or api() != expected:
            raise ValueError("GRAPH_BLAS_PREFERENCE_READBACK")
        with torch.no_grad():
            work = torch.zeros((1, ALIGNED, WIDTH), dtype=torch.bfloat16, device=device)
            before, after = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            for case in range(cases):
                budget()
                x = operand((1,TOKENS,WIDTH), .1)
                # Modest input-scale variation exercises the same BF16 shape across distinct operands.
                weight = operand((OUT,WIDTH), (.012, .015, .018)[case % 3])
                reference = linear(x.float(), weight.float()).bfloat16()
                order = (0, 1) if (rank + case) % 2 == 0 else (1, 0)
                outputs = {}
                row = dict(case=case, order=[ARMS[i] for i in order], arms={})
                for reuse in order:
                    budget()
                    full = invoke(x, weight, work, reuse)
                    if full.shape != (1, ALIGNED, OUT) or full.dtype != torch.bfloat16:
                        raise ValueError("GRAPH_OUTPUT_GEOMETRY")
                    visible = full[:, :TOKENS, :]
                    outputs[reuse] = visible.clone()
                    maximum = (visible.float() - reference.float()).abs().amax()
                    invalid_output = torch.count_nonzero(full[:, TOKENS:, :])
                    torch.cuda.synchronize(device)
                    if case == 0:
                        measured()
                    times = []
                    changed = torch.zeros((), dtype=torch.int64, device=device)
                    for _ in range(calls_per_block // 8):
                        before.record()
                        for _ in range(8):
                            timed = invoke(x, weight, work, reuse)[:, :TOKENS, :]
                        after.record()
                        after.synchronize()
                        times.append(before.elapsed_time(after) / 8)
                        changed += (timed.view(torch.int16) != outputs[reuse].view(torch.int16)).sum()
                    error = maximum.item()
                    valid = [ms for ms in times if math.isfinite(ms) and ms > 0]
                    row["arms"][ARMS[reuse]] = dict(max_abs=error if math.isfinite(error) else None,
                        changed=changed.item(), padding_nonzero=invalid_output.item(),
                        invalid_events=len(times)-len(valid), event_ms=statistics.median(valid) if valid else None)
                row["paired_differences"] = (outputs[0].view(torch.int16) != outputs[1].view(torch.int16)).sum().item()
                row["scratch_tail_nonzero"] = torch.count_nonzero(work[:, TOKENS:, :]).item()
                summary.append(row)
    finally:
        api(inherited)
        if api() != inherited:
            raise RuntimeError("GRAPH_BLAS_PREFERENCE_RESTORE")
    return dict(rank=rank, cases=summary, calls_per_arm_case=calls_per_block,
                input_kind="deterministic_synthetic_BF16",
                runtime=dict(torch=str(torch.__version__), hip=torch.version.hip),
                restored_preference=str(inherited), compiled_kernel_identity_proven=False)
