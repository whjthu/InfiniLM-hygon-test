import importlib
import torch
from typing import Any


def is_torch_npu_available():
    return importlib.util.find_spec("torch_npu") is not None


def is_torch_mlu_available():
    if importlib.util.find_spec("torch_mlu") is None:
        return False
    import torch_mlu  # noqa: F401

    return torch.mlu.is_available()


IS_CUDA_AVAILABLE = torch.cuda.is_available()
IS_NPU_AVAILABLE = is_torch_npu_available() and torch.npu.is_available()
IS_MLU_AVAILABLE = is_torch_mlu_available()

if IS_NPU_AVAILABLE:
    import torch_npu

    torch.npu.config.allow_internal_format = False


def get_device_type() -> str:
    """Get the active CPU, CUDA, NPU, or MLU device type."""
    if IS_CUDA_AVAILABLE:
        device = "cuda"
    elif IS_NPU_AVAILABLE:
        device = "npu"
    elif IS_MLU_AVAILABLE:
        device = "mlu"
    else:
        device = "cpu"

    return device


def get_torch_device() -> Any:
    """Get torch attribute based on device type, e.g. torch.cuda or torch.npu"""
    device_name = get_device_type()

    try:
        return getattr(torch, device_name)
    except AttributeError:
        print(f"Device namespace '{device_name}' not found in torch, try to load 'torch.cuda'.")
        return torch.cuda


def get_device_id() -> int:
    """Get current device id based on device type."""
    return get_torch_device().current_device()


def get_device_name() -> str:
    """Get current device name based on device type."""
    return f"{get_device_type()}:{get_device_id()}"


def synchronize() -> None:
    """Execute torch synchronize operation."""
    get_torch_device().synchronize()


def empty_cache() -> None:
    """Execute torch empty cache operation."""
    get_torch_device().empty_cache()


def get_nccl_backend() -> str:
    """Return distributed communication backend type based on device type."""
    if IS_CUDA_AVAILABLE:
        return "nccl"
    elif IS_NPU_AVAILABLE:
        return "hccl"
    elif IS_MLU_AVAILABLE:
        return "cncl"
    else:
        raise RuntimeError(f"No available distributed communication backend found on device type {get_device_type()}.")


def enable_high_precision_for_bf16():
    """
    Set high accumulation dtype for matmul and reduction.
    """
    if IS_CUDA_AVAILABLE:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

    if IS_NPU_AVAILABLE:
        torch.npu.matmul.allow_tf32 = False
        torch.npu.matmul.allow_bf16_reduced_precision_reduction = False


def parse_device_type(device):
    if isinstance(device, str):
        if device.startswith("cuda"):
            return "cuda"
        elif device.startswith("npu"):
            return "npu"
        elif device.startswith("mlu"):
            return "mlu"
        else:
            return "cpu"
    elif isinstance(device, torch.device):
        return device.type


def parse_nccl_backend(device_type):
    if device_type == "cuda":
        return "nccl"
    elif device_type == "npu":
        return "hccl"
    elif device_type == "mlu":
        return "cncl"
    else:
        raise RuntimeError(f"No available distributed communication backend found on device type {device_type}.")


def get_available_device_type():
    return get_device_type()
