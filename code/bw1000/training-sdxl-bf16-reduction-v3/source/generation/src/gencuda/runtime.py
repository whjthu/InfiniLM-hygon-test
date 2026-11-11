from __future__ import annotations

from dataclasses import dataclass
from typing import Any


MEMORY_LIMIT_BYTES = 75_000_000_000


def within_memory_limit(reserved_bytes: int, limit_bytes: int = MEMORY_LIMIT_BYTES) -> bool:
    return 0 <= reserved_bytes <= limit_bytes


@dataclass(frozen=True)
class RuntimeInfo:
    torch_version: str | None
    cuda_available: bool
    cuda_count: int
    device_name: str | None
    accelerator: str | None = None
    accelerator_version: str | None = None


def get_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required; use a CUDA or HIP PyTorch environment") from exc
    return torch


def inspect_runtime() -> RuntimeInfo:
    try:
        torch = get_torch()
    except RuntimeError:
        return RuntimeInfo(None, False, 0, None)
    cuda = getattr(torch, "cuda", None)
    available = bool(cuda and cuda.is_available())
    count = int(cuda.device_count()) if available else 0
    name = str(cuda.get_device_name(0)) if available and count else None
    versions = getattr(torch, "version", None)
    hip = getattr(versions, "hip", None)
    cuda_version = getattr(versions, "cuda", None)
    accelerator = "hip" if hip else "cuda" if cuda_version else None
    version = hip or cuda_version
    return RuntimeInfo(torch.__version__, available, count, name, accelerator, version)


def validate_accelerator(runtime: RuntimeInfo, expected: str, card_count: int) -> None:
    if runtime.accelerator != expected:
        found = runtime.accelerator or "unreported"
        raise RuntimeError(f"expected {expected} PyTorch runtime, found {found}")
    if not runtime.cuda_available or runtime.cuda_count != card_count:
        raise RuntimeError(
            f"expected {card_count} visible {expected} device(s), found {runtime.cuda_count}"
        )


def require_cuda() -> Any:
    torch = get_torch()
    if not getattr(torch, "cuda", None) or not torch.cuda.is_available():
        raise RuntimeError("PyTorch accelerator is unavailable; check CUDA or HIP device assignment")
    return torch


def autocast_bf16() -> Any:
    torch = require_cuda()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def synchronize() -> None:
    torch = require_cuda()
    torch.cuda.synchronize()


def memory_stats() -> dict[str, int]:
    torch = require_cuda()
    allocated = int(torch.cuda.max_memory_allocated())
    reserved = int(torch.cuda.max_memory_reserved())
    return {
        "allocated": allocated,
        "reserved": reserved,
        "peak_allocated_bytes": allocated,
        "peak_reserved_bytes": reserved,
    }
