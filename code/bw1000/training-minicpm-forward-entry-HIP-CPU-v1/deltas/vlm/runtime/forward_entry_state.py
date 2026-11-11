"""Bounded, opt-in entry-state hashes; never replace model state or RNG."""
from __future__ import annotations

import hashlib
import json
import time

import torch

from .forward_context_state import context_snapshot, input_layouts


def _identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def tensor_digest(value, *, check, chunk_bytes=4 * 1024**2):
    """Hash dense logical values using bounded CPU payload chunks.

    Noncontiguous tensors are split along the first nonsingleton dimension,
    preserving logical order without materializing a full device-side clone.
    Only CPU finiteness checks are used; no device F64 arithmetic is introduced.
    """
    if type(chunk_bytes) is not int or not 1024 <= chunk_bytes <= 16 * 1024**2:
        raise ValueError('invalid entry-state copy bound')
    if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
            or value.is_complex() or value.is_quantized or value.device.type not in ('cpu', 'cuda')):
        raise ValueError('entry-state hash requires a dense real CPU or device tensor')
    digest = hashlib.sha256()
    copied = peak = 0

    def visit(part):
        nonlocal copied, peak
        check()
        size = part.numel() * part.element_size()
        if size > chunk_bytes:
            dim = next(i for i, length in enumerate(part.shape) if length > 1)
            per_slice = size // part.shape[dim]
            count = max(1, chunk_bytes // per_slice)
            for start in range(0, part.shape[dim], count):
                visit(part.narrow(dim, start, min(count, part.shape[dim] - start)))
            return
        block = part.detach().to(device='cpu').contiguous().reshape(-1)
        if block.is_floating_point() and not bool(torch.isfinite(block).all()):
            raise ValueError('nonfinite entry-state tensor')
        digest.update(block.view(torch.uint8).numpy().tobytes())
        copied += size
        peak = max(peak, size)

    visit(value)
    return dict(SHA256=digest.hexdigest(), logical_bytes=copied, peak_chunk_bytes=peak)


def _input_alignments(args, kwargs):
    result = {}

    def walk(value, path):
        if isinstance(value, torch.Tensor):
            if len(result) >= 512 or value.layout != torch.strided or value.device.type not in ('cpu', 'cuda'):
                raise ValueError('unsupported input alignment inventory')
            result[path] = dict(data_mod_256=value.data_ptr() % 256,
                                storage_mod_256=value.untyped_storage().data_ptr() % 256)
        elif isinstance(value, dict):
            for key in sorted(value):
                walk(value[key], path + '/' + key)
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                walk(item, path + '/' + str(index))

    walk(args, 'args')
    walk(kwargs, 'kwargs')
    return result


def _selected_runtime(model):
    result = {}
    for name, module in model.named_modules():
        attributes = vars(module)
        row = {}
        for key in ('gradient_checkpointing', 'max_seq_len_cached', 'original_max_seq_len',
                    'attention_scaling', 'rope_type'):
            if key in attributes:
                value = attributes[key]
                if value is not None and type(value) not in (str, bool, int, float):
                    raise ValueError('unsupported selected model runtime attribute')
                row[key] = value
        for key in ('chunk_gated_delta_rule', 'recurrent_gated_delta_rule'):
            if key in attributes:
                value = attributes[key]
                row[key] = dict(module=getattr(value, '__module__', None),
                                qualname=getattr(value, '__qualname__', None))
        config = attributes.get('config')
        if config is not None:
            row['config'] = {key: getattr(config, key, None) for key in
                             ('model_type', 'use_cache', '_attn_implementation', 'rope_parameters')}
        if row:
            _identity(row)
            result[name] = row
    return result


def capture_entry_state(model, args, kwargs, *, check, rng_snapshot,
                        device_type='cuda', chunk_bytes=4 * 1024**2,
                        max_logical_bytes=16 * 1024**3):
    """Observe one actual forward entry, including nonpersistent buffers.

    The caller owns the absolute deadline and supplies a hash-only RNG snapshot.
    Copies can synchronize execution and affect allocator/timing behavior. This
    function is diagnostic only and does not qualify recovery or performance.
    """
    if type(max_logical_bytes) is not int or not 0 < max_logical_bytes <= 64 * 1024**3:
        raise ValueError('invalid entry-state total byte bound')
    started = time.monotonic()
    rng_before = rng_snapshot()
    _identity(rng_before)  # reject raw tensors and nonfinite/unserializable data
    check()
    context = context_snapshot(model, device_type=device_type)
    selected_runtime = _selected_runtime(model)
    layouts = input_layouts(args, kwargs)
    alignments = _input_alignments(args, kwargs)
    rows, seen = [], {}
    logical_bytes = copied_bytes = peak = 0
    with torch.no_grad():
        for module_name, module in model.named_modules(remove_duplicate=False):
            for kind, registry in (('parameter', module._parameters), ('buffer', module._buffers)):
                for local_name, value in sorted(registry.items()):
                    check()
                    if len(rows) >= 100000:
                        raise ValueError('entry-state registration bound exceeded')
                    name = (module_name + '.' if module_name else '') + local_name
                    key = kind + ':' + name
                    row = dict(name=name, kind=kind, present=value is not None)
                    if kind == 'buffer':
                        row['persistent'] = local_name not in module._non_persistent_buffers_set
                    if value is not None:
                        size = value.numel() * value.element_size()
                        logical_bytes += size
                        if logical_bytes > max_logical_bytes:
                            raise ValueError('entry-state total byte bound exceeded')
                        row.update(shape=list(value.shape), dtype=str(value.dtype),
                                   stride=list(value.stride()), storage_offset=value.storage_offset(),
                                   requires_grad=bool(value.requires_grad), logical_bytes=size)
                        if id(value) in seen:
                            row['alias_of'] = seen[id(value)]
                        else:
                            seen[id(value)] = key
                            hashed = tensor_digest(value, check=check, chunk_bytes=chunk_bytes)
                            row['SHA256'] = hashed['SHA256']
                            copied_bytes += hashed['logical_bytes']
                            peak = max(peak, hashed['peak_chunk_bytes'])
                    rows.append(row)
    check()
    rng_after = rng_snapshot()
    if rng_before != rng_after:
        raise ValueError('RNG changed while capturing entry state')
    result = dict(schema=1, RNG=rng_before, observer_RNG_unchanged=True,
                  context=context, input_layouts=layouts, input_alignments=alignments,
                  selected_module_runtime=selected_runtime,
                  registered_state=rows, registered_state_SHA256=_identity(rows),
                  logical_bytes_including_aliases=logical_bytes, unique_payload_bytes_read=copied_bytes,
                  peak_payload_chunk_bytes=peak, wall_seconds=time.monotonic() - started,
                  all_registered_parameters_and_buffers_observed=True,
                  unregistered_python_state_observed=False, inner_kernel_context_observed=False,
                  recovery_qualified=False, formal_performance=False)
    return result
