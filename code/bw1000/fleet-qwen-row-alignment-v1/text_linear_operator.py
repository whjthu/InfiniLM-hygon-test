"""Check whether zero row padding stabilizes native Qwen text linear geometry."""
import math
import statistics
import time

import torch
from torch.nn.functional import linear, pad

SHAPES = (
    ("decoder_06.v_proj", 186, 4096, 1024),
    ("decoder_06.mlp_down_proj", 216, 12288, 4096),
    ("decoder_10.q_proj", 216, 4096, 4096),
)
ARMS = ("native_rows", "padded_32")


def exercise(device, rank, measured, deadline, calls_per_block=256):
    if type(calls_per_block) is not int or not 16 <= calls_per_block <= 256 or calls_per_block % 16:
        raise ValueError("GRAPH_CALL_COUNT")
    api = torch.backends.cuda.preferred_blas_library
    inherited = api()
    expected = torch.backends.cuda._BlasBackends["cublas"]
    order = (0, 1, 1, 0) if rank % 2 == 0 else (1, 0, 0, 1)
    generator = torch.Generator(device=device).manual_seed(80197)
    torch.set_float32_matmul_precision("highest")
    rows = []

    def budget():
        if time.time() >= deadline:
            raise TimeoutError("GRAPH_COMPONENT_BUDGET")

    def operand(shape, scale):
        return torch.randn(shape, device=device, generator=generator).mul_(scale).to(torch.bfloat16)

    try:
        if api("cublas") != expected or api() != expected:
            raise ValueError("GRAPH_BLAS_PREFERENCE_READBACK")
        with torch.no_grad():
            for site, tokens, width, projected in SHAPES:
                budget()
                x = operand((1, tokens, width), .1)
                weight = operand((projected, width), .015)
                reference = linear(x.float(), weight.float()).to(torch.bfloat16)
                extra_rows = (-tokens) % 32
                anchors = {}
                events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                for arm in order:
                    start = time.monotonic()
                    error = torch.zeros((), device=device)
                    finite = torch.ones((), dtype=torch.bool, device=device)
                    changed = torch.zeros((), dtype=torch.int64, device=device)
                    nonzero_padding = torch.zeros_like(changed)
                    timings = []

                    def forward():
                        input_tensor = pad(x, (0, 0, 0, extra_rows)) if arm else x
                        return linear(input_tensor, weight)

                    for _ in range(2):
                        forward()
                    for _ in range(16):
                        budget()
                        full = forward()
                        expected_rows = tokens + (extra_rows if arm else 0)
                        if full.shape != (1, expected_rows, projected) or full.dtype != torch.bfloat16:
                            raise ValueError("GRAPH_OUTPUT_GEOMETRY")
                        visible = full[:, :tokens, :]
                        bits = visible.view(torch.int16)
                        if arm not in anchors:
                            anchors[arm] = bits.clone()
                        changed += (bits != anchors[arm]).sum()
                        finite &= torch.isfinite(full).all()
                        error = torch.maximum(error, (visible.float() - reference.float()).abs().amax())
                        if arm:
                            nonzero_padding += torch.count_nonzero(full[:, tokens:, :])
                    torch.cuda.synchronize(device)
                    measured()
                    for _ in range(calls_per_block // 16):
                        budget()
                        events[0].record()
                        for _ in range(16):
                            timed = forward()[:, :tokens, :]
                        events[1].record()
                        events[1].synchronize()
                        timings.append(events[0].elapsed_time(events[1]) / 16)
                        finite &= torch.isfinite(timed).all()
                        changed += (timed.view(torch.int16) != anchors[arm]).sum()
                    maximum = error.item()
                    changed_elements = changed.item()
                    padding_nonzero = nonzero_padding.item()
                    is_finite = bool(finite.item()) and math.isfinite(maximum)
                    valid = [value for value in timings if math.isfinite(value) and value > 0]
                    rows.append(dict(site=site, tokens=tokens, executed_rows=expected_rows,
                                     arm=ARMS[arm], observed_preference=str(expected),
                                     calls=calls_per_block, numeric_calls=16, finite_numeric=is_finite,
                                     numeric_gate_passed=is_finite and maximum <= .0625,
                                     max_abs=maximum if math.isfinite(maximum) else None,
                                     changed_elements=changed_elements, nonzero_padding_elements=padding_nonzero,
                                     event_p50_ms=statistics.median(valid) if valid else None,
                                     invalid_events=len(timings)-len(valid),
                                     wall_seconds=time.monotonic()-start))
    finally:
        api(inherited)
        if api() != inherited:
            raise RuntimeError("GRAPH_BLAS_PREFERENCE_RESTORE")
    return dict(rank=rank, blocks=rows, input_kind="deterministic_synthetic_BF16",
                runtime=dict(torch=str(torch.__version__), hip=torch.version.hip),
                restored_preference=str(inherited), compiled_kernel_identity_proven=False)
