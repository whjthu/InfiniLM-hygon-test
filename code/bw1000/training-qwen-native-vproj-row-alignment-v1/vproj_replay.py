"""Profile on the first frozen operand captured by the real Swift model."""
import statistics

import torch
from torch.nn.functional import linear, pad

from replay_state import describe

ROWS = 186
WIDTH = 4096
OUT = 1024
EXTRA = 6
SITE = "decoder_06.v_proj"
ARMS = ("native_rows", "aligned_rows", "aligned_rows", "native_rows")
CAPTURE_ARMS = ("native_rows", "allocated_padding", "reused_padding")


def check_input(x, module, *, native_stride=False):
    if (tuple(x.shape) != (1, ROWS, WIDTH) or x.dtype != torch.bfloat16
            or not x.is_cuda or not x.is_contiguous()
            or (native_stride and tuple(x.stride()) != (ROWS*WIDTH, WIDTH, 1))
            or module.bias is not None or tuple(module.weight.shape) != (OUT, WIDTH)
            or module.weight.dtype != torch.bfloat16 or module.weight.requires_grad
            or module.weight.device != x.device or not module.weight.is_contiguous()):
        raise ValueError("EXACT_FROZEN_VPROJ_OPERAND")


def replay(x, weight, observed, check):
    api = torch.backends.cuda.preferred_blas_library
    enums = torch.backends.cuda._BlasBackends
    previous = api()
    expect = enums["cublas"]
    precision = torch.get_float32_matmul_precision()
    records = []
    anchors = {}
    seed = describe(x)
    measured_calls = 32
    try:
        if api("cublas") != expect or api() != expect:
            raise ValueError("ORIGINAL_VPROJ_BLAS_PREFERENCE")
        torch.set_float32_matmul_precision("highest")
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            reference = linear(x.float(), weight.float()).to(torch.bfloat16)
            workspace = torch.zeros((1, ROWS + EXTRA, WIDTH), device=x.device, dtype=x.dtype)
            events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]

            def execute(arm):
                if arm == "native_rows":
                    return linear(x, weight)
                if arm == "allocated_padding":
                    full = linear(pad(x, (0, 0, 0, EXTRA)), weight)
                else:
                    workspace[:, :ROWS, :].copy_(x)
                    full = linear(workspace, weight)
                return full

            for arm in (*CAPTURE_ARMS, *reversed(CAPTURE_ARMS)):
                times = []
                numerics = []
                for index in range(2 + measured_calls):
                    check()
                    events[0].record()
                    full = execute(arm)
                    visible = full[:, :ROWS, :]
                    events[1].record()
                    events[1].synchronize()
                    expected_rows = ROWS if arm == "native_rows" else ROWS + EXTRA
                    if (tuple(full.shape) != (1, expected_rows, OUT)
                            or full.dtype != torch.bfloat16 or not bool(torch.isfinite(full).all())):
                        raise ValueError("REPLAY_VPROJ_OUTPUT")
                    if index < 2:
                        continue
                    ms = events[0].elapsed_time(events[1])
                    if not 0 < ms < float("inf"):
                        raise ValueError("REPLAY_VPROJ_EVENT")
                    times.append(ms)
                    bits = visible.view(torch.int16)
                    if arm not in anchors:
                        anchors[arm] = bits.clone()
                    numerics.append((
                        (visible.float() - reference.float()).abs().amax().item(),
                        torch.count_nonzero(bits != anchors[arm]).item(),
                        torch.count_nonzero(bits != observed.view(torch.int16)).item(),
                        torch.count_nonzero(full[:, ROWS:, :]).item(),
                    ))
                records.append(dict(arm=arm, calls=len(times), median_event_ms=statistics.median(times),
                                    max_abs_rounded_fp32=max(v[0] for v in numerics),
                                    changed_from_arm_anchor=sum(v[1] for v in numerics),
                                    changed_from_original_leaf_output=sum(v[2] for v in numerics),
                                    nonzero_discarded_output=sum(v[3] for v in numerics)))
            if describe(x) != seed or torch.count_nonzero(workspace[:, ROWS:, :]).item():
                raise ValueError("VPROJ_REPLAY_MUTATION")
            exact = torch.count_nonzero(anchors["allocated_padding"] != anchors["reused_padding"]).item()
            return dict(input=seed, output_observed=describe(observed), rows=records,
                        allocation_vs_reuse_anchor_bit_differences=exact,
                        workspace_bytes=workspace.numel()*workspace.element_size(),
                        formal_training=False)
    finally:
        torch.set_float32_matmul_precision(precision)
        api(previous)
        if api() != previous:
            raise RuntimeError("REPLAY_VPROJ_BLAS_RESTORE")
