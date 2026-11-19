"""Capture the native gradient on the single padded v_proj edge and replay its VJP."""
import statistics
import time

import torch
from torch.nn.functional import linear, pad

from replay_state import describe

SITE = "decoder_06.v_proj"
ROWS, WIDTH, OUT, EXTRA = 186, 4096, 1024, 6
SHAPE = (1, ROWS, WIDTH)
ORDER = ("native_rows", "aligned_rows", "aligned_rows", "native_rows")


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


class ProjectionInputGradient:
    def __init__(self):
        self.microbatch = 0
        self.operand = None
        self.weight = None
        self.upstream = None
        self.padded_input_grad = None

    def forward(self, original, module, value):
        weight = module.weight
        need(value.dtype == weight.dtype == torch.bfloat16 and value.is_cuda
             and weight.device == value.device and tuple(value.shape) == SHAPE
             and value.is_contiguous() and weight.is_contiguous()
             and not weight.requires_grad and module.bias is None and tuple(weight.shape) == (OUT, WIDTH),
             "ACTUAL_VPROJ_FROZEN_BF16")
        version = value._version, weight._version
        padded = pad(value, (0, 0, 0, EXTRA))
        full = original(padded)
        out = full[0, :ROWS, :].unsqueeze(0)
        need(tuple(full.shape) == (1, ROWS + EXTRA, OUT)
             and out.requires_grad == (torch.is_grad_enabled() and value.requires_grad)
             and version == (value._version, weight._version), "ACTUAL_VPROJ_PADDED_PATH")

        if self.microbatch == 1 and out.requires_grad:
            need(self.operand is None, "DUPLICATE_GRAD_ENABLED_FIRST_VPROJ")
            self.operand = value.detach().clone()
            self.weight = weight.detach()
            padded.register_hook(self.keep_padded_grad)
            out.register_hook(self.keep_upstream)
        return out

    def keep_padded_grad(self, grad):
        need(self.microbatch == 1 and self.padded_input_grad is None
             and tuple(grad.shape) == (1, ROWS + EXTRA, WIDTH)
             and grad.dtype == torch.bfloat16 and grad.device == self.operand.device,
             "WRONG_ACTUAL_PADDED_INPUT_GRADIENT")
        self.padded_input_grad = grad.detach().clone()
        return None

    def keep_upstream(self, grad):
        need(self.microbatch == 1 and self.upstream is None
             and tuple(grad.shape) == (1, ROWS, OUT)
             and grad.dtype == torch.bfloat16 and grad.device == self.operand.device,
             "WRONG_ACTUAL_VPROJ_UPSTREAM")
        self.upstream = grad.detach().clone()
        return None

    def evaluate(self, deadline):
        need(all(item is not None for item in (self.operand, self.weight, self.upstream, self.padded_input_grad)),
             "MISSING_NATIVE_VPROJ_INPUT_GRADIENT")
        try:
            return replay(self.operand, self.weight, self.upstream, self.padded_input_grad, deadline)
        finally:
            self.operand = self.weight = self.upstream = self.padded_input_grad = None


def replay(operand, weight, upstream, padded_dx, deadline):
    preferences = torch.backends.cuda.preferred_blas_library
    expected = torch.backends.cuda._BlasBackends["cublas"]
    need(preferences() == expected, "INHERITED_FROZEN_VPROJ_BLAS")
    precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        with torch.autocast("cuda", enabled=False), torch.no_grad():
            reference = (upstream.float() @ weight.float()).to(torch.bfloat16).float()
    finally:
        torch.set_float32_matmul_precision(precision)
    native_dx = padded_dx[0, :ROWS, :].unsqueeze(0)
    need(bool(torch.isfinite(padded_dx).all() & torch.isfinite(reference).all()),
         "NONFINITE_NATIVE_VPROJ_DX_OR_REFERENCE")
    native_bits = native_dx.view(torch.int16)
    anchors = {}
    rows = []
    events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    ref_norm = torch.linalg.vector_norm(reference).item()
    measured = 16

    for arm in ORDER:
        ms = []
        max_abs = rel = 0.0
        vs_native = changed = 0
        for index in range(measured + 2):
            if time.time() >= deadline:
                raise TimeoutError("VPROJ_INPUT_VJP_DEADLINE")
            x = operand.detach().requires_grad_()
            with torch.enable_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                if arm == "aligned_rows":
                    full = linear(pad(x, (0, 0, 0, EXTRA)), weight)
                    output = full[0, :ROWS, :].unsqueeze(0)
                else:
                    output = linear(x, weight)
                need(tuple(output.shape) == (1, ROWS, OUT) and output.dtype == torch.bfloat16,
                     "VPROJ_REPLAY_OUTPUT_GEOMETRY")
                events[0].record()
                dx, = torch.autograd.grad(output, x, upstream)
                events[1].record()
                events[1].synchronize()
            need(tuple(dx.shape) == SHAPE and dx.dtype == torch.bfloat16 and x.grad is None
                 and preferences() == expected and bool(torch.isfinite(dx).all()),
                 "VPROJ_REPLAY_DX")
            if index < 2:
                continue
            dt = events[0].elapsed_time(events[1])
            need(0 < dt < float("inf"), "INVALID_VPROJ_DX_EVENT")
            ms.append(dt)
            bits = dx.view(torch.int16)
            if arm not in anchors:
                anchors[arm] = bits.clone()
            step_error = dx.float() - reference
            max_abs = max(max_abs, step_error.abs().amax().item())
            if ref_norm:
                rel = max(rel, torch.linalg.vector_norm(step_error).item() / ref_norm)
            else:
                need(not torch.count_nonzero(step_error).item(), "ZERO_VPROJ_DX_REFERENCE")
            vs_native += torch.count_nonzero(bits != native_bits).item()
            changed += torch.count_nonzero(bits != anchors[arm]).item()
        rows.append(dict(arm=arm, calls=len(ms), median_vjp_event_ms=statistics.median(ms),
                         max_abs_vs_rounded_fp32=max_abs, relative_l2_vs_rounded_fp32=rel,
                         changed_from_native_padded_dx=vs_native, changed_from_arm_anchor=changed))
    return dict(site=SITE, operand=describe(operand), upstream=describe(upstream), native_dx=describe(native_dx),
                native_padding_tail_nonzero=torch.count_nonzero(padded_dx[:, ROWS:, :]).item(),
                padded_rows=ROWS+EXTRA, weight_shape=list(weight.shape), rows=rows)
