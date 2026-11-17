"""CPU summaries of complete native first-update artifacts; no raw tensor export."""
from __future__ import annotations

import hashlib
import json
import math

import torch


def fingerprint(value):
    """Hash types/shapes/values and audit finiteness with bounded temporary buffers."""
    digest = hashlib.sha256()
    count = floating = elements = 0
    square = 0.0

    def tag(value):
        encoded = json.dumps(value, separators=(',', ':'), ensure_ascii=True).encode()
        digest.update(str(len(encoded)).encode() + b':' + encoded)

    def visit(item):
        nonlocal count, floating, elements, square
        if isinstance(item, torch.Tensor):
            if item.device.type != 'cpu' or item.is_complex() or item.layout != torch.strided:
                raise ValueError('artifact audit requires dense real CPU tensors')
            flat = item.detach().contiguous().reshape(-1)
            tag(['tensor', str(item.dtype), list(item.shape)])
            count += 1
            elements += item.numel()
            for chunk in flat.split(1024 * 1024):
                if item.is_floating_point():
                    if not bool(torch.isfinite(chunk).all()):
                        raise ValueError('nonfinite saved tensor')
                    floating += chunk.numel()
                    square += float(chunk.double().square().sum())
                digest.update(chunk.view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            if not all(isinstance(key, str) for key in item):
                raise ValueError('expected named artifact dictionary keys')
            tag(['dict', len(item)])
            for key in sorted(item):
                tag(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            tag(['tuple' if isinstance(item, tuple) else 'list', len(item)])
            for child in item:
                visit(child)
        elif item is None or isinstance(item, (str, bool, int)):
            tag([type(item).__name__, item])
        elif isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError('nonfinite saved scalar')
            tag(['float', item.hex()])
        else:
            raise ValueError('unsupported saved artifact value type')

    visit(value)
    return dict(SHA256=digest.hexdigest(), tensors=count, elements=elements,
                floating_elements=floating, floating_l2=math.sqrt(square), finite=True)


def parameter_delta(before, after):
    if not before or set(before) != set(after):
        raise ValueError('saved parameter names differ across an optimizer update')
    squared = 0.0
    changed = tensors = 0
    maximum = 0.0
    for name in sorted(before):
        left, right = before[name], after[name]
        if (not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor)
                or left.device.type != 'cpu' or right.device.type != 'cpu'
                or left.shape != right.shape or left.dtype != right.dtype):
            raise ValueError('saved parameter shape/dtype/device differs')
        a, b = left.reshape(-1), right.reshape(-1)
        count = 0
        for start in range(0, a.numel(), 1024 * 1024):
            delta = b[start:start + 1024 * 1024].double() - a[start:start + 1024 * 1024].double()
            if not bool(torch.isfinite(delta).all()):
                raise ValueError('nonfinite parameter update')
            squared += float(delta.square().sum())
            count += int(torch.count_nonzero(delta))
            if delta.numel():
                maximum = max(maximum, float(delta.abs().max()))
        changed += count
        tensors += bool(count)
    return dict(l2=math.sqrt(squared), max_abs=maximum, changed_elements=changed,
                changed_tensors=tensors, parameter_tensors=len(before))


def optimizer_steps(optimizer):
    state = optimizer['state_by_name']
    steps = []
    empty = 0
    for item in state.values():
        if not item:
            empty += 1
            continue
        step = item.get('step')
        if isinstance(step, torch.Tensor):
            if step.device.type != 'cpu' or step.numel() != 1:
                raise ValueError('invalid saved optimizer step tensor')
            step = float(step)
        if not isinstance(step, (int, float)) or not math.isfinite(step) or step < 0 or int(step) != step:
            raise ValueError('invalid saved optimizer step')
        steps.append(int(step))
    return dict(entries=len(state), populated=len(steps), empty=empty,
                minimum=min(steps) if steps else None, maximum=max(steps) if steps else None)


def summarize_update(before, after, preclip, postclip, *, ordinal):
    expected = ordinal - 1
    if (before['global_step'] != expected or preclip['global_step'] != expected
            or postclip['global_step'] != expected or after['global_step_before_increment'] != expected):
        raise ValueError('saved optimizer/clip boundaries differ')
    parameters = before['parameters']
    names = set(parameters)
    for record in (preclip, postclip):
        if set(record['gradients']) != names:
            raise ValueError('gradient names do not match the trainable inventory')
        for name, value in record['gradients'].items():
            if value is not None and (value.shape != parameters[name].shape or value.dtype != parameters[name].dtype):
                raise ValueError('saved gradient shape/dtype differs')
    if set(before['optimizer']['state_by_name']) != names or set(after['optimizer']['state_by_name']) != names:
        raise ValueError('optimizer state does not cover the trainable inventory')
    groups = before['optimizer']['param_groups']
    group_names = [name for group in groups for name in group['parameter_names']]
    if len(group_names) != len(names) or set(group_names) != names:
        raise ValueError('optimizer parameter groups are not a partition')
    return dict(ordinal=ordinal, parameters_before=fingerprint(parameters),
                parameters_after=fingerprint(after['parameters']),
                parameter_update=parameter_delta(parameters, after['parameters']),
                preclip_gradients=fingerprint(preclip['gradients']),
                postclip_gradients=fingerprint(postclip['gradients']),
                reported_preclip_norm=fingerprint(postclip['reported_preclip_norm']),
                optimizer_state_before=fingerprint(before['optimizer']['state_by_name']),
                optimizer_state_after=fingerprint(after['optimizer']['state_by_name']),
                optimizer_before_steps=optimizer_steps(before['optimizer']),
                optimizer_after_steps=optimizer_steps(after['optimizer']),
                before_group_learning_rates=[float(group['lr']) for group in groups],
                qualification='Same-platform saved-value audit, not cross-platform error tolerances or fresh recovery')
