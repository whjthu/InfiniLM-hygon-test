"""Shared original-forward value copying and CPU-only error summaries."""
import hashlib
import torch


def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for raw in iter(lambda: stream.read(1024**2), b''):
            h.update(raw)
    return h.hexdigest()


def to_cpu(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().clone()
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError('nonfinite forward output')
        return value
    if isinstance(value, dict):
        return {k: to_cpu(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(to_cpu(v) for v in value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError('unsupported diagnostic value')


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, dict):
        return {k: to_device(v, device) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(to_device(v, device) for v in value)
    return value


def metrics(actual, reference, check):
    if actual.shape != reference.shape or actual.dtype != reference.dtype:
        raise ValueError('output shape or dtype changed')
    x = actual.reshape(-1); y = reference.reshape(-1)
    square = norm = 0.; maximum = 0.; unequal = 0
    for start in range(0, x.numel(), 1024**2):
        check(); a = x[start:start+1024**2].double(); b = y[start:start+1024**2].double()
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError('nonfinite metric operands')
        delta = a-b; square += float(delta.square().sum()); norm += float(b.square().sum())
        maximum = max(maximum, float(delta.abs().max())); unequal += int((a != b).sum())
    return dict(shape=list(actual.shape), dtype=str(actual.dtype), max_abs=maximum,
                relative_l2=(square/norm)**.5 if norm else None, unequal_elements=unequal,
                exact=unequal == 0)
