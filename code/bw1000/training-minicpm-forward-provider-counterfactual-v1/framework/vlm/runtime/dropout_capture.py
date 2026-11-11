"""Capture native dropout draws for controlled-input diagnostics, including zero inputs.

The dispatcher delegates to the original operator exactly once. Capture returns its
unchanged result; replay preserves that draw/RNG advancement and supplies the frozen
mask to the native dropout arithmetic. Neither operation is a performance benchmark.
"""
from __future__ import annotations

import hashlib
import math

import torch
import torch.nn.functional as functional
from torch.utils._python_dispatch import TorchDispatchMode


def _mask_sha(mask: torch.Tensor) -> str:
    return hashlib.sha256(mask.view(torch.uint8).numpy().tobytes()).hexdigest()


def _input_metadata(value: torch.Tensor, probability: float) -> dict:
    if not math.isfinite(probability) or not 0 < probability < 1:
        raise ValueError("diagnostic capture requires 0 < dropout probability < 1")
    if value.layout != torch.strided or not value.is_floating_point() or value.numel() == 0:
        raise ValueError("expected a nonempty floating strided dropout input")
    return {
        "probability": probability, "shape": list(value.shape),
        "stride": list(value.stride()), "dtype": str(value.dtype),
    }


def validate_record(value: torch.Tensor, record: dict) -> torch.Tensor:
    required = {"probability", "shape", "stride", "dtype", "source_operator",
                "source_device", "mask", "mask_SHA256"}
    if not isinstance(record, dict) or set(record) != required:
        raise ValueError("incomplete native dropout record")
    metadata = _input_metadata(value, record["probability"])
    if any(record[key] != item for key, item in metadata.items()):
        raise ValueError("dropout input shape, stride or dtype differs")
    mask = record["mask"]
    if (not isinstance(mask, torch.Tensor) or mask.device.type != "cpu"
            or mask.dtype != torch.bool or list(mask.shape) != metadata["shape"]
            or not mask.is_contiguous() or mask.requires_grad):
        raise ValueError("expected the complete contiguous CPU boolean mask")
    if _mask_sha(mask) != record["mask_SHA256"]:
        raise ValueError("native dropout mask identity differs")
    if record["source_operator"] not in ("aten::native_dropout", "aten::bernoulli_"):
        raise ValueError("unsupported native dropout mask provider")
    return mask


class _NativeMaskMode(TorchDispatchMode):
    def __init__(self, value: torch.Tensor, probability: float, replay: dict | None):
        super().__init__()
        self.value = value
        self.probability = probability
        self.replay = replay
        self.frozen = None if replay is None else validate_record(value, replay)
        self.mask = None
        self.provider = None
        self.draws = 0

    def __torch_dispatch__(self, operation, types, args=(), kwargs=None):
        result = operation(*args, **(kwargs or {}))
        name = operation._schema.name
        if name not in ("aten::native_dropout", "aten::bernoulli_"):
            return result
        self.draws += 1
        if self.draws != 1:
            raise RuntimeError("dropout scope produced multiple native masks")
        self.provider = name
        if name == "aten::native_dropout":
            if len(result) != 2 or args[0] is not self.value:
                raise RuntimeError("unexpected native dropout outputs or input")
            output, native_mask = result
            if tuple(native_mask.shape) != tuple(self.value.shape):
                raise RuntimeError("native dropout mask shape differs")
            self.mask = native_mask.detach().to(device="cpu", dtype=torch.bool).contiguous().clone()
            if self.frozen is None:
                return result
            fixed = self.frozen.to(device=native_mask.device, dtype=native_mask.dtype)
            output = torch.ops.aten.native_dropout_backward.default(
                self.value, fixed, 1.0 / (1.0 - self.probability))
            return output, fixed
        if tuple(result.shape) != tuple(self.value.shape):
            raise RuntimeError("native Bernoulli dropout mask shape differs")
        if len(args) < 2 or float(args[1]) != 1.0 - self.probability:
            raise RuntimeError("unexpected native Bernoulli keep probability")
        observed = result.detach().to(device="cpu").contiguous()
        mask = observed.to(dtype=torch.bool)
        if not torch.equal(observed, mask.to(dtype=observed.dtype)):
            raise RuntimeError("native Bernoulli output is not a binary mask")
        self.mask = mask.clone()
        if self.frozen is not None:
            # Preserve the actual sampler call, then use the shared draw before
            # the native dropout's division/multiplication consume this tensor.
            result.copy_(self.frozen.to(device=result.device, dtype=result.dtype))
        return result


def capture_dropout(value: torch.Tensor, probability: float) -> tuple[torch.Tensor, dict]:
    metadata = _input_metadata(value, probability)
    mode = _NativeMaskMode(value, probability, None)
    with mode:
        output = functional.dropout(value, p=probability, training=True, inplace=False)
    if mode.draws != 1 or mode.mask is None:
        raise RuntimeError("backend did not expose a supported native dropout mask")
    record = {
        **metadata, "source_operator": mode.provider, "source_device": str(value.device),
        "mask": mode.mask, "mask_SHA256": _mask_sha(mode.mask),
    }
    return output, record


def replay_dropout(value: torch.Tensor, record: dict) -> torch.Tensor:
    mode = _NativeMaskMode(value, record["probability"], record)
    with mode:
        output = functional.dropout(value, p=record["probability"], training=True, inplace=False)
    if mode.draws != 1:
        raise RuntimeError("backend did not consume exactly one native dropout draw")
    return output
