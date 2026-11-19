"""Compare BLAS preferences for the rank-16 Qwen LoRA weight gradient."""
import math
import statistics
import time

import torch
from torch.nn.functional import linear

SHAPES = (
    ("decoder_06.v_proj.lora_A", 186, 4096, 16),
    ("decoder_10.q_proj.lora_A", 216, 4096, 16),
    ("decoder_10.q_proj.lora_B", 216, 16, 4096),
)
PREFERENCES = ("cublas", "cublaslt")


def exercise(device, rank, measured, deadline, calls_per_block=256):
    if type(calls_per_block) is not int or not 0 < calls_per_block <= 256:
        raise ValueError("GRAPH_CALL_COUNT")
    api = torch.backends.cuda.preferred_blas_library
    inherited = api()
    forward_preference = torch.backends.cuda._BlasBackends["cublaslt"]
    order = (0, 1, 1, 0) if rank % 2 == 0 else (1, 0, 0, 1)
    generator = torch.Generator(device=device).manual_seed(80197)
    torch.set_float32_matmul_precision("highest")
    rows = []
    marked = False

    def budget():
        if time.time() >= deadline:
            raise TimeoutError("GRAPH_COMPONENT_BUDGET")

    def new(shape, scale):
        return torch.randn(shape, device=device, generator=generator).mul_(scale).to(torch.bfloat16)

    try:
        for site, tokens, width, projected in SHAPES:
            budget()
            x = new((1, tokens, width), .1)
            weight = new((projected, width), .015).requires_grad_()
            upstream = new((1, tokens, projected), .1)
            with torch.no_grad():
                forward_ref = linear(x.float(), weight.float()).to(torch.bfloat16)
                vjp_ref = (upstream[0].float().T @ x[0].float()).to(torch.bfloat16)
            anchors = {}
            first = None
            events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
            for arm in order:
                preference = PREFERENCES[arm]
                expected = torch.backends.cuda._BlasBackends[preference]
                max_y = torch.zeros((), device=device)
                max_dw = torch.zeros_like(max_y)
                changed_y = torch.zeros((), dtype=torch.int64, device=device)
                changed_dw = torch.zeros_like(changed_y)
                cross_y = torch.zeros_like(changed_y)
                cross_dw = torch.zeros_like(changed_y)
                numeric = torch.ones((), dtype=torch.bool, device=device)
                forward_ms, vjp_ms = [], []
                start = time.monotonic()
                for index in range(calls_per_block + 2):
                    budget()
                    torch.cuda.synchronize(device)
                    try:
                        if api("cublaslt") != forward_preference or api() != forward_preference:
                            raise ValueError("GRAPH_FORWARD_PREFERENCE_READBACK")
                        events[0].record()
                        y = linear(x, weight)
                        events[1].record()
                        if api(preference) != expected or api() != expected:
                            raise ValueError("GRAPH_BACKWARD_PREFERENCE_READBACK")
                        dw, = torch.autograd.grad(y, weight, upstream)
                        events[2].record()
                        events[2].synchronize()
                    finally:
                        api(inherited)
                    if api() != inherited or not y.requires_grad or y.dtype != torch.bfloat16:
                        raise RuntimeError("GRAPH_FORWARD_OR_RESTORE")
                    if dw.dtype != torch.bfloat16 or dw.shape != weight.shape or y.shape != forward_ref.shape:
                        raise RuntimeError("GRAPH_OUTPUT_GEOMETRY")
                    if index < 2:
                        continue
                    if not marked:
                        measured()
                        marked = True
                    forward_ms.append(events[0].elapsed_time(events[1]))
                    vjp_ms.append(events[1].elapsed_time(events[2]))
                    actual_y = y.detach()
                    bits = actual_y.view(torch.int16), dw.view(torch.int16)
                    if arm not in anchors:
                        anchors[arm] = tuple(t.clone() for t in bits)
                        if first is None:
                            first = anchors[arm]
                    numeric &= torch.isfinite(actual_y).all() & torch.isfinite(dw).all()
                    max_y = torch.maximum(max_y, (actual_y.float() - forward_ref.float()).abs().amax())
                    max_dw = torch.maximum(max_dw, (dw.float() - vjp_ref.float()).abs().amax())
                    changed_y += (bits[0] != anchors[arm][0]).sum()
                    changed_dw += (bits[1] != anchors[arm][1]).sum()
                    cross_y += (bits[0] != first[0]).sum()
                    cross_dw += (bits[1] != first[1]).sum()
                y_error, dw_error = max_y.item(), max_dw.item()
                is_finite = bool(numeric.item()) and math.isfinite(y_error) and math.isfinite(dw_error)
                valid_forward = [value for value in forward_ms if math.isfinite(value) and value > 0]
                valid_vjp = [value for value in vjp_ms if math.isfinite(value) and value > 0]
                rows.append(dict(site=site, tokens=tokens, input_width=width, output_width=projected, arm=preference, observed_preference=str(expected), calls=calls_per_block,
                                 finite_numeric=is_finite,
                                 numeric_gate_passed=is_finite and max(y_error, dw_error) <= .0625,
                                 forward_max_abs_fp32_rounded=y_error if math.isfinite(y_error) else None,
                                 vjp_max_abs_fp32_rounded=dw_error if math.isfinite(dw_error) else None, forward_changed_elements=changed_y.item(),
                                 vjp_changed_elements=changed_dw.item(), forward_cross_first_elements=cross_y.item(),
                                 vjp_cross_first_elements=cross_dw.item(), forward_p50_event_ms=statistics.median(valid_forward) if valid_forward else None,
                                 dispatch_gap_plus_vjp_p50_event_ms=statistics.median(valid_vjp) if valid_vjp else None,
                                 invalid_forward_event_count=len(forward_ms)-len(valid_forward),
                                 invalid_vjp_event_count=len(vjp_ms)-len(valid_vjp),
                                 wall_seconds=time.monotonic()-start))
    finally:
        api(inherited)
        if api() != inherited:
            raise RuntimeError("GRAPH_BLAS_PREFERENCE_RESTORE")
    return dict(rank=rank, blocks=rows, input_kind="deterministic_synthetic_BF16", gradient="weight",
                restored_preference=str(inherited), forward_preference=str(forward_preference),
                runtime=dict(torch=str(torch.__version__), hip=torch.version.hip), compiled_kernel_identity_proven=False)
