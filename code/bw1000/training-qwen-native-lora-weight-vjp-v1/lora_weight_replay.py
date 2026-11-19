"""Measure isolated weight gradients using detached operands from the real Swift backward."""
import statistics
import time

import torch
from torch.nn.functional import linear


ARMS = ("cublas", "cublaslt")
MEASURED_CALLS = 16


class LoRAWeightReplay:
    def __init__(self, modules):
        if set(modules) != {"A", "B"}:
            raise ValueError("EXPECTED_QUERY_LORA_A_AND_B")
        self.microbatch = 0
        self.operands = {}
        self.native_weight_gradients = {}
        self.handles = [
            module.register_forward_hook(self.observe(name))
            for name, module in modules.items()
        ]
        self.handles.extend(
            module.weight.register_hook(self.observe_native_weight_gradient(name))
            for name, module in modules.items()
        )

    def observe_native_weight_gradient(self, name):
        def hook(gradient):
            if self.microbatch == 1:
                if name not in self.operands or name in self.native_weight_gradients:
                    raise ValueError("ACTUAL_QUERY_LORA_WEIGHT_GRADIENT_ORDER")
                if (gradient.dtype != torch.float32 or
                        list(gradient.shape) != self.operands[name][3]["weight_shape"]):
                    raise ValueError("ACTUAL_QUERY_LORA_MASTER_GRADIENT")
                self.native_weight_gradients[name] = gradient.detach().clone()
            return None
        return hook

    def observe(self, name):
        def hook(module, args, output):
            if self.microbatch != 1 or not output.requires_grad:
                return
            operand, = args
            if (module.bias is not None or module.weight.dtype != torch.float32
                    or not torch.is_autocast_enabled("cuda")
                    or torch.get_autocast_dtype("cuda") != torch.bfloat16
                    or operand.dtype not in (torch.float32, torch.bfloat16)
                    or output.dtype != torch.bfloat16 or operand.device.type != "cuda"
                    or not 2 <= operand.ndim <= 3 or tuple(operand.shape[:-1]) != tuple(output.shape[:-1])
                    or not 0 < operand.numel() // module.in_features <= 4096
                    or operand.shape[-1] != module.in_features or output.shape[-1] != module.out_features):
                raise ValueError("ACTUAL_QUERY_LORA_AUTOCAST_INPUT")
            native = dict(input_shape=list(operand.shape), input_stride=list(operand.stride()),
                          input_dtype=str(operand.dtype), output_dtype=str(output.dtype),
                          master_dtype=str(module.weight.dtype),
                          output_shape=list(output.shape), weight_shape=list(module.weight.shape))

            def take_grad(gradient):
                if (name in self.operands or self.microbatch != 1 or gradient.dtype != torch.bfloat16
                        or list(gradient.shape) != native["output_shape"] or gradient.device != operand.device):
                    raise ValueError("ACTUAL_FIRST_QUERY_LORA_OUTPUT_GRADIENT")
                self.operands[name] = (
                    operand.detach().clone(), module.weight.detach().clone(), gradient.detach().clone(),
                    dict(native, upstream_stride=list(gradient.stride())),
                )
                return None

            output.register_hook(take_grad)
        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def evaluate(self, rank, deadline):
        self.close()
        if set(self.operands) != {"A", "B"} or set(self.native_weight_gradients) != {"A", "B"}:
            raise ValueError("MISSING_NATIVE_QUERY_LORA_GRADIENT")
        try:
            return {
                name: replay_weight_gradient(*self.operands[name], self.native_weight_gradients[name],
                                             rank=rank, deadline=deadline)
                for name in ("A", "B")
            }
        finally:
            self.operands.clear()
            self.native_weight_gradients.clear()


def replay_weight_gradient(operand, master, upstream, native, native_dw, *, rank, deadline):
    api = torch.backends.cuda.preferred_blas_library
    enums = torch.backends.cuda._BlasBackends
    inherited = api()
    if inherited != enums["cublas"]:
        raise ValueError("INHERITED_NATIVE_QUERY_LORA_BACKEND")
    input2d = operand.reshape(-1, operand.shape[-1])
    upstream2d = upstream.reshape(-1, upstream.shape[-1])
    prev_precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        with torch.autocast("cuda", enabled=False):
            reference = (upstream2d.float().T @ input2d.to(torch.bfloat16).float()).to(torch.bfloat16).float()
    finally:
        torch.set_float32_matmul_precision(prev_precision)
    if not bool(torch.isfinite(reference).all() & torch.isfinite(native_dw).all()):
        raise ValueError("NONFINITE_NATIVE_QUERY_LORA_REFERENCE")
    scale = torch.linalg.vector_norm(reference).item()
    native_bits = native_dw.contiguous().view(torch.int32)

    order = ARMS + ARMS[::-1] if rank % 2 == 0 else ARMS[::-1] + ARMS
    rows = []
    first_by_arm = {}
    first = None
    weight = master.requires_grad_()
    events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    for arm in order:
        elapsed = []
        max_abs = 0.
        max_relative_l2 = 0.
        changed = cross = native_changed = 0
        for index in range(MEASURED_CALLS + 2):
            if time.time() >= deadline:
                raise TimeoutError("ISOLATED_NATIVE_LORA_WEIGHT_REPLAY_DEADLINE")
            with torch.enable_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                output = linear(operand, weight)
            if output.dtype != torch.bfloat16 or list(output.shape) != native["output_shape"]:
                raise ValueError("REPLAY_QUERY_LORA_AUTOCAST_OUTPUT")
            try:
                if api(arm) != enums[arm] or api() != enums[arm]:
                    raise ValueError("REPLAY_QUERY_LORA_BACKEND_READBACK")
                events[0].record()
                gradient, = torch.autograd.grad(output, weight, upstream)
                events[1].record()
                events[1].synchronize()
            finally:
                api(inherited)
            if (api() != inherited or gradient.dtype != torch.float32
                    or list(gradient.shape) != native["weight_shape"]):
                raise ValueError("REPLAY_QUERY_LORA_MASTER_GRADIENT")
            if not bool(torch.isfinite(gradient).all()):
                raise ValueError("NONFINITE_QUERY_LORA_MASTER_GRADIENT")
            if index < 2:
                continue
            value = events[0].elapsed_time(events[1])
            if not 0 < value < float("inf"):
                raise ValueError("INVALID_QUERY_LORA_WEIGHT_EVENT")
            elapsed.append(value)
            bits = gradient.contiguous().view(torch.int32)
            if arm not in first_by_arm:
                first_by_arm[arm] = bits.clone()
            if first is None:
                first = first_by_arm[arm]
            max_abs = max(max_abs, (gradient - reference).abs().max().item())
            delta = torch.linalg.vector_norm(gradient - reference).item()
            if scale:
                max_relative_l2 = max(max_relative_l2, delta / scale)
            elif delta:
                max_relative_l2 = None
            changed += int(torch.count_nonzero(bits != first_by_arm[arm]).item())
            cross += int(torch.count_nonzero(bits != first).item())
            native_changed += int(torch.count_nonzero(bits != native_bits).item())
        rows.append(dict(arm=arm, calls=len(elapsed), weight_vjp_event_median_ms=statistics.median(elapsed),
                         max_abs_vs_rounded_fp32=max_abs, relative_l2_vs_rounded_fp32=max_relative_l2,
                         changed_from_arm_anchor=changed, changed_from_first_anchor=cross,
                         changed_from_native_weight_gradient=native_changed))
    if weight.grad is not None:
        raise ValueError("ISOLATED_QUERY_LORA_REPLAY_ACCUMULATED_GRADIENT")
    return dict(native=native, replay_input_stride=list(operand.stride()),
                replay_upstream_stride=list(upstream.stride()), scope="first_native_microbatch_replay",
                timing="event envelope around autograd.grad for weight only; separately from training",
                rows=rows)
