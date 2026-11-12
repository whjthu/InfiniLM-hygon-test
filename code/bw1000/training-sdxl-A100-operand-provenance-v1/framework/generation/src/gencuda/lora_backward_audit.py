"""Observe one real LoRA-B weight gradient without changing the backward result."""

from __future__ import annotations

import hashlib


class LinearBackwardAudit:
    """Capture one module's actual operands, then calculate a CPU-only reference.

    Hooks return None and never replace activations or gradients. Model matching
    and module selection are exact; repeated invocations fail rather than mixing
    microbatches. Captures and their costs are diagnostic, never timed training.
    """

    def __init__(self, model_type, module_name, *, max_rows=4096,
                 max_features=2048, max_capture_bytes=16 * 1024**2):
        self.model_type = model_type
        self.module_name = module_name
        self.max_rows = max_rows
        self.max_features = max_features
        self.max_capture_bytes = max_capture_bytes
        self.handles = []
        self.values = {}
        self.counts = {"model": 0, "forward": 0, "output_gradient": 0, "weight_gradient": 0}
        self.capture_bytes = 0

    def __enter__(self):
        import torch

        self.handles.append(torch.nn.modules.module.register_module_forward_pre_hook(self._find))
        return self

    def _capture(self, name, value):
        import torch

        if name in self.values or value.dtype != torch.bfloat16:
            raise ValueError("repeated capture or non-BF16 operand")
        size = value.numel() * value.element_size()
        if self.capture_bytes + size > self.max_capture_bytes:
            raise ValueError("backward capture byte budget exceeded")
        self.capture_bytes += size
        self.values[name] = value.detach().clone()

    def _find(self, model, args):
        import torch

        if not isinstance(model, self.model_type):
            return
        self.counts["model"] += 1
        if self.counts["model"] != 1:
            raise ValueError("repeated model forward")
        module = model.get_submodule(self.module_name)
        if (not isinstance(module, torch.nn.Linear) or module.bias is not None
                or module.in_features != 4 or module.out_features > self.max_features
                or not module.weight.requires_grad or module.weight.dtype != torch.bfloat16):
            raise ValueError("expected one trainable rank-four BF16 bias-free LoRA-B Linear")

        def forward(_module, inputs, output):
            self.counts["forward"] += 1
            if self.counts["forward"] != 1 or len(inputs) != 1:
                raise ValueError("repeated or unsupported LoRA-B forward")
            x = inputs[0]
            if (x.ndim < 2 or x.shape[-1] != 4 or not 0 < x.numel() // 4 <= self.max_rows
                    or output.shape != (*x.shape[:-1], module.out_features)
                    or output.device != x.device or not output.requires_grad):
                raise ValueError("invalid LoRA-B operand topology")
            self._capture("input", x)

            def gradient(dy):
                self.counts["output_gradient"] += 1
                self._capture("output_gradient", dy)
                return None

            self.handles.append(output.register_hook(gradient))
            return None

        def weight_gradient(gradient):
            self.counts["weight_gradient"] += 1
            self._capture("weight_gradient", gradient)
            return None

        self.handles += [module.register_forward_hook(forward), module.weight.register_hook(weight_gradient)]
        return None

    def report(self, reference_gradient):
        import torch

        if any(value != 1 for value in self.counts.values()) or set(self.values) != {
                "input", "output_gradient", "weight_gradient"}:
            raise ValueError("incomplete single-invocation backward capture")
        cpu = {key: value.detach().cpu() for key, value in self.values.items()}
        if (reference_gradient.device.type != "cpu" or reference_gradient.dtype != torch.bfloat16
                or reference_gradient.shape != cpu["weight_gradient"].shape):
            raise ValueError("reference gradient schema mismatch")
        if any(not bool(torch.isfinite(value).all()) for value in [*cpu.values(), reference_gradient]):
            raise ValueError("nonfinite backward capture")
        # Convert only after D2H: no FP64 tensor or operation is placed on a GPU.
        x = cpu["input"].reshape(-1, 4).double()
        dy = cpu["output_gradient"].reshape(-1, cpu["weight_gradient"].shape[0]).double()
        exact = dy.T @ x
        absolute_sum = dy.abs().T @ x.abs()
        rounded = exact.to(torch.bfloat16)
        native = cpu["weight_gradient"]

        def comparison(observed, expected):
            a, b = observed.double(), expected.double()
            difference = float((a - b).norm())
            denominator = float(b.norm())
            return {"difference_l2": difference, "reference_l2": denominator,
                    "relative_l2": difference / denominator if denominator else (0.0 if not difference else None),
                    "sign_differences": int((a.sign() != b.sign()).sum()),
                    "elements": a.numel(), "exact_equal": bool(torch.equal(observed, expected))}

        nonzero = exact.abs() > 0
        ratios = absolute_sum[nonzero] / exact.abs()[nonzero]
        zero_cancellation = int(((~nonzero) & (absolute_sum > 0)).sum())
        hashes = {key: hashlib.sha256(value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
                  for key, value in cpu.items()}
        return {
            "module": self.module_name, "counts": dict(self.counts),
            "input_shape": list(cpu["input"].shape), "output_gradient_shape": list(cpu["output_gradient"].shape),
            "weight_shape": list(native.shape), "operand_dtype": str(native.dtype),
            "operand_SHA256": hashes, "captured_tensor_bytes": self.capture_bytes,
            "local_reference_device": "cpu", "local_reference_dtype": "torch.float64",
            "native_vs_local_F64_rounded": comparison(native, rounded),
            "native_vs_original_gradient": comparison(native, reference_gradient),
            "local_F64_rounded_vs_original_gradient": comparison(rounded, reference_gradient),
            "cancellation_ratio_at_least": {str(bound): int((ratios >= bound).sum()) + zero_cancellation
                                             for bound in (10, 100, 1000, 10000)},
            "exact_zero_with_nonzero_absolute_sum": zero_cancellation,
            "FP64_product_adds": x.shape[0] * 4 * dy.shape[1],
            "qualification": "One linear weight-gradient diagnostic using actual local operands. Not a repaired gradient, model numerical acceptance, performance result or cause proven across platforms.",
        }

    def close(self):
        while self.handles:
            self.handles.pop().remove()

    def __exit__(self, *_error):
        self.close()
