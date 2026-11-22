"""Opt into deterministic FlashAttention 2 backward around the native HIP training loop."""

from collections import Counter
from contextlib import contextmanager
from functools import wraps
from hashlib import sha256
import importlib
import inspect
from pathlib import Path


POLICY = "hip-flash2-deterministic-backward-v1"


def verify_provider(module, pin):
    path = Path(module.__file__).resolve(strict=True)
    if path.stat().st_size != pin["bytes"]:
        raise ValueError(f"Unexpected provider size: {path.name}")
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != pin["SHA256"]:
        raise ValueError(f"Unexpected provider bytes: {path.name}")


def resolve(torch, binding, deadline):
    if binding["policy"] != POLICY or torch.version.hip != binding["HIP"] or torch.version.cuda is not None:
        raise ValueError("Unexpected policy or Torch HIP runtime")
    native = importlib.import_module("diffsynth.core.attention.attention")
    flash = importlib.import_module("flash_attn")
    interface = importlib.import_module("flash_attn.flash_attn_interface")
    extension = importlib.import_module("flash_attn_2_cuda")
    if native.ATTENTION_IMPLEMENTATION != "flash_attention_2" or native.flash_attn is not flash:
        raise ValueError("The frozen DiffSynth backend did not select native FlashAttention 2")
    function = flash.flash_attn_func
    if (flash.__version__ != binding["flash_version"] or function is not interface.flash_attn_func
            or inspect.getmodule(function) is not interface
            or not any(value is extension for value in vars(interface).values())):
        raise ValueError("Unexpected FlashAttention 2 dispatch or extension")
    for key, module in (("native", native), ("initializer", flash), ("interface", interface),
                        ("extension", extension)):
        deadline()
        verify_provider(module, binding["providers"][key])
    return flash, function


@contextmanager
def deterministic_backward(flash, function, torch, deadline):
    signature = inspect.signature(function)
    if signature.parameters["deterministic"].default is not False:
        raise ValueError("Unexpected FlashAttention 2 default")
    if flash.flash_attn_func is not function:
        raise ValueError("FlashAttention 2 dispatch is already modified")
    counts = Counter()
    ledger = {"forced_calls": 0, "unchanged_calls": 0}

    @wraps(function)
    def call(*args, **kwargs):
        deadline()
        bound = signature.bind(*args, **kwargs)
        if "deterministic" in bound.arguments:
            raise ValueError("Native call already specifies a backward policy")
        tensors = tuple(bound.arguments[name] for name in ("q", "k", "v"))
        if not torch.is_grad_enabled() or not any(tensor.requires_grad for tensor in tensors):
            output = function(*args, **kwargs)
            ledger["unchanged_calls"] += 1
            return output
        if any(tensor.device.type != "cuda" or tensor.dtype != torch.bfloat16 for tensor in tensors):
            raise ValueError("Deterministic backward requires the frozen native HIP BF16 inputs")
        key = tuple((tuple(tensor.shape), tuple(tensor.stride())) for tensor in tensors)
        if key not in counts and len(counts) >= 512:
            raise ValueError("Unexpected number of native QKV layouts")
        output = function(*args, **kwargs, deterministic=True)
        counts[key] += 1
        ledger["forced_calls"] += 1
        return output

    flash.flash_attn_func = call
    try:
        yield ledger
    finally:
        if flash.flash_attn_func is not call:
            raise RuntimeError("FlashAttention 2 dispatch changed during training")
        flash.flash_attn_func = function
        ledger["grad_layouts"] = [
            {"qkv": [{"shape": shape, "stride": stride} for shape, stride in key], "calls": value}
            for key, value in sorted(counts.items())
        ]
