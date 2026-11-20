"""Test row alignment for the frozen Qwen value-projection input gradient."""
import math
import statistics
import time

import torch
from torch.nn.functional import pad

ROWS, INPUT_WIDTH, OUTPUT_WIDTH = 186, 4096, 1024
ALIGNED_ROWS = 192
ARMS = ("original_186", "zero_pad_192")


def exercise(device, rank, measured, deadline, calls_per_block=32, cases=384):
    if calls_per_block not in (8, 32) or cases not in (2, 384):
        raise ValueError("VPROJ_INPUT_GRADIENT_CASE_COUNTS")
    api = torch.backends.cuda.preferred_blas_library
    inherited = api()
    expected = torch.backends.cuda._BlasBackends["cublas"]
    precision = torch.get_float32_matmul_precision()
    rng = torch.Generator(device=device).manual_seed(28173317 + rank)

    def budget():
        if time.time() >= deadline:
            raise TimeoutError("VPROJ_INPUT_GRADIENT_DEADLINE")

    def operand(shape, scale):
        return torch.randn(shape, device=device, generator=rng).mul_(scale).bfloat16()

    def invoke(upstream, weight, aligned):
        if aligned:
            upstream = pad(upstream, (0, 0, 0, ALIGNED_ROWS - ROWS))
        return torch.mm(upstream, weight)

    cases_out = []
    try:
        if api("cublas") != expected or api() != expected:
            raise ValueError("VPROJ_INPUT_GRADIENT_BLAS_READBACK")
        torch.set_float32_matmul_precision("highest")
        with torch.no_grad():
            begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            for case in range(cases):
                budget()
                gradient_scale = (.0625, .125, .25)[case % 3]
                weight_scale = (.012, .015, .018)[case // 3 % 3]
                dy = operand((ROWS, OUTPUT_WIDTH), gradient_scale)
                weight = operand((OUTPUT_WIDTH, INPUT_WIDTH), weight_scale)
                reference = torch.mm(dy.float(), weight.float()).bfloat16()
                order = (0, 1) if (rank + case) % 2 == 0 else (1, 0)
                outputs = {}
                row = dict(case=case, order=[ARMS[i] for i in order], arms={})
                for arm in order:
                    budget()
                    full = invoke(dy, weight, arm)
                    if full.shape != (ALIGNED_ROWS if arm else ROWS, INPUT_WIDTH) or full.dtype != torch.bfloat16:
                        raise ValueError("VPROJ_INPUT_GRADIENT_OUTPUT")
                    output = full[:ROWS]
                    outputs[arm] = output.clone()
                    maximum = (output.float() - reference.float()).abs().amax()
                    padding_nonzero = torch.count_nonzero(full[ROWS:])
                    torch.cuda.synchronize(device)
                    if case == 0:
                        measured()
                    times = []
                    changed = 0
                    for _ in range(calls_per_block // 8):
                        begin.record()
                        for _ in range(8):
                            timed = invoke(dy, weight, arm)
                        end.record()
                        end.synchronize()
                        times.append(begin.elapsed_time(end) / 8)
                        changed += torch.count_nonzero(timed.view(torch.int16) != full.view(torch.int16)).item()
                    value = maximum.item()
                    valid = [ms for ms in times if math.isfinite(ms) and ms > 0]
                    row["arms"][ARMS[arm]] = dict(max_abs=value if math.isfinite(value) else None,
                        changed=changed, padding_nonzero=padding_nonzero.item(),
                        invalid_events=len(times)-len(valid), event_ms=statistics.median(valid) if valid else None)
                row["paired_differences"] = (outputs[0].view(torch.int16) != outputs[1].view(torch.int16)).sum().item()
                cases_out.append(row)
    finally:
        torch.set_float32_matmul_precision(precision)
        api(inherited)
        if api() != inherited:
            raise RuntimeError("VPROJ_INPUT_GRADIENT_BLAS_RESTORE")
    return dict(rank=rank,cases=cases_out,calls_per_arm_case=calls_per_block,
                geometry=dict(rows=ROWS,padded_rows=ALIGNED_ROWS,K=OUTPUT_WIDTH,N=INPUT_WIDTH),
                input_kind="generated_BF16_not_native_gradient",
                runtime=dict(torch=str(torch.__version__),hip=torch.version.hip))
