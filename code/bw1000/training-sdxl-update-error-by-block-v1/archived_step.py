"""Attribute the SDXL first-step mismatch using the archived optimizer operands."""
import hashlib
import math

import torch


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with path.open('rb') as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
        return digest.hexdigest()


def load(path, pin):
    require(path.stat().st_size == pin['bytes'] and sha(path) == pin['sha256'],
            'ARCHIVE_SIZE_AND_HASH')
    return torch.load(path, map_location='cpu', weights_only=True)


def group(payload, count):
    saved = payload['initial_optimizer']
    result, = saved['param_groups']
    require(not saved['state'] and result['params'] == list(range(count)), 'FRESH_ADAMW_STATE')
    require(result['decoupled_weight_decay'] is True and not any(result.get(k, False) for k in
            ('amsgrad', 'maximize', 'capturable', 'differentiable')), 'ADAMW_FIRST_STEP_VARIANT')
    lr, wd, eps = (result[k] for k in ('lr', 'weight_decay', 'eps'))
    beta1, beta2 = result['betas']
    require(all(math.isfinite(x) for x in (lr, wd, eps, beta1, beta2))
            and lr > 0 and eps > 0 and wd >= 0 and 0 <= beta1 < 1 and 0 <= beta2 < 1,
            'ADAMW_OPTIONS')
    return result


def checked_tensor(table, name, shape):
    value = table[name]
    require(value.device.type == 'cpu' and value.dtype == torch.bfloat16
            and value.shape == shape and bool(torch.isfinite(value).all()), 'TENSOR_SCHEMA')
    return value


def squared(tensor):
    return float(tensor.square().sum().item())
