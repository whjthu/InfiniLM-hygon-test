"""Describe the original SDPA calls without inspecting tensor values."""

from contextlib import contextmanager
from functools import wraps


POSITIONAL = ("query", "key", "value", "attn_mask", "dropout_p", "is_causal")
OPTIONAL = ("scale", "enable_gqa")
MAX_SDPA_CALLS = 256


def tensor_descriptor(tensor):
    if tensor is None:
        return None
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "device_type": tensor.device.type,
        "contiguous": bool(tensor.is_contiguous()),
    }


def call_descriptor(args, kwargs):
    given = dict(zip(POSITIONAL, args))
    if len(args) > len(POSITIONAL) or given.keys() & kwargs.keys() or kwargs.keys() - set(POSITIONAL + OPTIONAL):
        raise ValueError("Unrecognized real SDPA argument layout")
    given.update(kwargs)
    if not {"query", "key", "value"} <= given.keys():
        raise ValueError("Real SDPA operands are incomplete")
    return {
        "query": tensor_descriptor(given["query"]), "key": tensor_descriptor(given["key"]),
        "value": tensor_descriptor(given["value"]), "mask": tensor_descriptor(given.get("attn_mask")),
        "dropout_p": given.get("dropout_p", 0.0), "is_causal": given.get("is_causal", False),
        "scale": given.get("scale"), "enable_gqa": given.get("enable_gqa", False),
    }


@contextmanager
def audit_sdpa_descriptors(functional):
    original = functional.scaled_dot_product_attention
    entries = []

    @wraps(original)
    def audited(*args, **kwargs):
        if len(entries) >= MAX_SDPA_CALLS:
            raise ValueError("Native forward SDPA call budget exceeded")
        # The call inherits the original arguments and Torch's original dispatch policy.
        descriptor = call_descriptor(args, kwargs)
        output = original(*args, **kwargs)
        entries.append(descriptor)
        return output

    functional.scaled_dot_product_attention = audited
    try:
        yield entries
    finally:
        functional.scaled_dot_product_attention = original
