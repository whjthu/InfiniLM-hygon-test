"""Frozen BF16 text linears and real input VJPs at three observed Qwen sites."""
import math
import statistics
import time

import torch
from torch.nn.functional import linear

SHAPES = (
    ("decoder_06.v_proj", 186, 4096, 1024),
    ("decoder_10.q_proj", 216, 4096, 4096),
    ("decoder_06.mlp_down_proj", 216, 12288, 4096),
)
PREFERENCES = ("cublas", "cublaslt")


def exercise(device, rank, measured, deadline, calls_per_block=256):
    if type(calls_per_block) is not int or not 0 < calls_per_block <= 256:
        raise ValueError("GRAPH_CALL_COUNT")
    api = torch.backends.cuda.preferred_blas_library
    inherited = api()
    order = (0, 1, 1, 0) if rank % 2 == 0 else (1, 0, 0, 1)
    generator = torch.Generator(device=device).manual_seed(80197 + rank)
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
            x = new((1, tokens, width), .1).requires_grad_()
            weight = new((projected, width), .015)
            upstream = new((1, tokens, projected), .1)
            with torch.no_grad():
                forward_ref = linear(x.float(), weight.float()).to(torch.bfloat16)
                vjp_ref = (upstream.float() @ weight.float()).to(torch.bfloat16)
            anchors = {}
            first = None
            events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
            for arm in order:
                preference = PREFERENCES[arm]
                expected = torch.backends.cuda._BlasBackends[preference]
                max_y = torch.zeros((), device=device)
                max_dx = torch.zeros_like(max_y)
                changed_y = torch.zeros((), dtype=torch.int64, device=device)
                changed_dx = torch.zeros_like(changed_y)
                cross_y = torch.zeros_like(changed_y)
                cross_dx = torch.zeros_like(changed_y)
                numeric = torch.ones((), dtype=torch.bool, device=device)
                forward_ms, vjp_ms = [], []
                start = time.monotonic()
                for index in range(calls_per_block + 2):
                    budget()
                    torch.cuda.synchronize(device)
                    try:
                        if api(preference) != expected or api() != expected:
                            raise ValueError("GRAPH_BLAS_PREFERENCE_READBACK")
                        events[0].record()
                        y = linear(x, weight)
                        events[1].record()
                    finally:
                        api(inherited)
                    if api() != inherited or not y.requires_grad or y.dtype != torch.bfloat16:
                        raise RuntimeError("GRAPH_FORWARD_OR_RESTORE")
                    dx, = torch.autograd.grad(y, x, upstream)
                    events[2].record()
                    events[2].synchronize()
                    if dx.dtype != torch.bfloat16 or dx.shape != x.shape or y.shape != forward_ref.shape:
                        raise RuntimeError("GRAPH_OUTPUT_GEOMETRY")
                    if index < 2:
                        continue
                    if not marked:
                        measured()
                        marked = True
                    forward_ms.append(events[0].elapsed_time(events[1]))
                    vjp_ms.append(events[1].elapsed_time(events[2]))
                    actual_y = y.detach()
                    bits = actual_y.view(torch.int16), dx.view(torch.int16)
                    if arm not in anchors:
                        anchors[arm] = tuple(t.clone() for t in bits)
                        if first is None:
                            first = anchors[arm]
                    numeric &= torch.isfinite(actual_y).all() & torch.isfinite(dx).all()
                    max_y = torch.maximum(max_y, (actual_y.float() - forward_ref.float()).abs().amax())
                    max_dx = torch.maximum(max_dx, (dx.float() - vjp_ref.float()).abs().amax())
                    changed_y += (bits[0] != anchors[arm][0]).sum()
                    changed_dx += (bits[1] != anchors[arm][1]).sum()
                    cross_y += (bits[0] != first[0]).sum()
                    cross_dx += (bits[1] != first[1]).sum()
                y_error, dx_error = max_y.item(), max_dx.item()
                if (not numeric.item() or not all(math.isfinite(t) and t > 0 for t in forward_ms + vjp_ms)
                        or max(y_error, dx_error) > .0625):
                    raise RuntimeError("GRAPH_FROZEN_BF16_FORWARD_VJP_NUMERICS")
                rows.append(dict(site=site, arm=preference, observed_preference=str(expected), calls=calls_per_block,
                                 finite_numeric=True, forward_max_abs_fp32_rounded=y_error,
                                 vjp_max_abs_fp32_rounded=dx_error, forward_changed_elements=changed_y.item(),
                                 vjp_changed_elements=changed_dx.item(), forward_cross_first_elements=cross_y.item(),
                                 vjp_cross_first_elements=cross_dx.item(), forward_p50_event_ms=statistics.median(forward_ms),
                                 dispatch_gap_plus_vjp_p50_event_ms=statistics.median(vjp_ms),
                                 wall_seconds=time.monotonic()-start))
    finally:
        api(inherited)
        if api() != inherited:
            raise RuntimeError("GRAPH_BLAS_PREFERENCE_RESTORE")
    return dict(rank=rank, blocks=rows, input_kind="deterministic_synthetic_BF16",
                restored_preference=str(inherited), backward_preference=str(inherited),
                runtime=dict(torch=str(torch.__version__), hip=torch.version.hip), compiled_kernel_identity_proven=False)
