"""Header-only checkpoint closure using the installed HF loading key rules.

This does not qualify tensor values, actual from_pretrained loading or training.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
from pathlib import Path
import struct

WIDTH = {'BOOL': 1, 'U8': 1, 'I8': 1, 'I16': 2, 'U16': 2, 'F16': 2,
         'BF16': 2, 'I32': 4, 'U32': 4, 'F32': 4, 'I64': 8, 'U64': 8, 'F64': 8}


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate checkpoint metadata key')
        result[key] = value
    return result


def read_header(path, expected_size, *, check):
    """Read exactly the eight-byte length and bounded JSON header, never values."""
    path = Path(path)
    check()
    if path.is_symlink() or type(expected_size) is not int or expected_size <= 8:
        raise ValueError('invalid checkpoint file identity')
    before = path.stat()
    if before.st_size != expected_size:
        raise ValueError('checkpoint size differs')
    with path.open('rb') as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError('missing checkpoint header length')
        size = struct.unpack('<Q', prefix)[0]
        if not 0 < size <= 2 * 1024**2 or 8 + size > before.st_size:
            raise ValueError('invalid checkpoint header length')
        raw = stream.read(size)
        if len(raw) != size or stream.tell() != 8 + size:
            raise ValueError('incomplete checkpoint header')
    check()
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError('checkpoint changed during header read')
    header = json.loads(raw, object_pairs_hook=_unique)
    if not isinstance(header, dict):
        raise ValueError('invalid checkpoint header object')
    records = {}
    spans = []
    for name, value in header.items():
        if name == '__metadata__':
            if not isinstance(value, dict) or any(not isinstance(v, str) for v in value.values()):
                raise ValueError('invalid checkpoint string metadata')
            continue
        if not name or not isinstance(value, dict) or set(value) != {'dtype', 'shape', 'data_offsets'}:
            raise ValueError('invalid checkpoint tensor metadata')
        shape, offsets, dtype = value['shape'], value['data_offsets'], value['dtype']
        if (not isinstance(shape, list) or len(shape) > 16
                or any(type(d) is not int or not 0 <= d <= 2**40 for d in shape)
                or dtype not in WIDTH or not isinstance(offsets, list) or len(offsets) != 2
                or any(type(x) is not int for x in offsets)):
            raise ValueError('invalid checkpoint shape, dtype or offsets')
        elements = math.prod(shape)
        first, last = offsets
        if not 0 <= first <= last <= expected_size - 8 - size or last - first != elements * WIDTH[dtype]:
            raise ValueError('checkpoint extent differs from tensor size')
        records[name] = dict(shape=shape, dtype=dtype, bytes=last-first, offsets=offsets)
        spans.append((first, last))
    cursor = 0
    for first, last in sorted(spans):
        if first != cursor:
            raise ValueError('checkpoint payload overlaps or has holes')
        cursor = last
    if not records or cursor != expected_size - 8 - size:
        raise ValueError('checkpoint payload is incomplete')
    return dict(records=records, header_bytes=size, bytes_read=8+size,
                header_SHA256=hashlib.sha256(raw).hexdigest(), file_bytes=expected_size,
                stat=dict(device=after.st_dev, inode=after.st_ino, mtime_ns=after.st_mtime_ns),
                payload_bytes_read=0, full_file_SHA256=None)


def audit_native_schema(model, index, headers, provider_pins, *, check):
    """Match every original key against the real full-model meta state dict."""
    from transformers.conversion_mapping import get_model_conversion_mapping
    from transformers.core_model_loading import WeightConverter, WeightRenaming, rename_source_key

    providers = {}
    for method in (get_model_conversion_mapping, rename_source_key):
        path = Path(inspect.getsourcefile(method)).resolve()
        pin = provider_pins.get(path.name)
        raw = path.read_bytes()
        if not pin or len(raw) != pin['bytes'] or hashlib.sha256(raw).hexdigest() != pin['SHA256']:
            raise ValueError('installed checkpoint loading provider differs')
        providers[path.name] = dict(path=str(path), bytes=len(raw), SHA256=pin['SHA256'])
    state = model.state_dict()
    if not state or any(p.device.type != 'meta' for p in state.values()):
        raise ValueError('checkpoint schema audit requires full meta model')
    mapping = get_model_conversion_mapping(model)
    renamings = [m for m in mapping if isinstance(m, WeightRenaming)]
    converters = [m for m in mapping if isinstance(m, WeightConverter)]
    if converters or len(renamings) != len(mapping):
        raise ValueError('header-only audit does not qualify tensor conversion operations')
    weight_map = index.get('weight_map')
    if not isinstance(weight_map, dict) or not weight_map or set(weight_map.values()) != set(headers):
        raise ValueError('checkpoint shard index differs')
    observed = {}
    for shard, header in headers.items():
        for name, record in header['records'].items():
            if name in observed or weight_map.get(name) != shard:
                raise ValueError('checkpoint header and index key ownership differ')
            observed[name] = record
    if set(observed) != set(weight_map):
        raise ValueError('checkpoint header/index keys are incomplete')
    dtype_names = {'torch.bfloat16': 'BF16', 'torch.float16': 'F16', 'torch.float32': 'F32',
                   'torch.float64': 'F64', 'torch.int64': 'I64', 'torch.int32': 'I32',
                   'torch.int8': 'I8', 'torch.uint8': 'U8', 'torch.bool': 'BOOL'}
    rows, targets = [], set()
    for original in sorted(weight_map):
        check()
        target, converted = rename_source_key(original, renamings, converters, model.base_model_prefix, state)
        # Preserve the actual loading function's fallback for already-modern keys.
        if target not in state and original in state:
            target, converted = rename_source_key(original, [], [], model.base_model_prefix, state)
        if converted is not None or target not in state or target in targets:
            raise ValueError('native checkpoint key missing, colliding or requires value conversion')
        record = observed[original]
        if list(state[target].shape) != record['shape'] or dtype_names.get(str(state[target].dtype)) != record['dtype']:
            raise ValueError('checkpoint shape or dtype differs from native meta model')
        targets.add(target)
        rows.append(dict(original=original, target=target, shard=weight_map[original], **record))
    if targets != set(state):
        raise ValueError('checkpoint does not cover the complete native state dict')
    total = sum(r['bytes'] for r in rows)
    if index.get('metadata', {}).get('total_size') != total:
        raise ValueError('checkpoint total_size differs')
    return dict(passed=True, model_class=type(model).__name__, keys=len(rows), meta_state_keys=len(state),
                changed_names=sum(r['original'] != r['target'] for r in rows), tensor_bytes=total,
                providers=providers, transforms=[dict(type=type(m).__name__, source=m.source_patterns,
                    target=m.target_patterns, scope=m.scope_prefix, base=m.base_model_prefix) for m in mapping],
                records=rows, payload_bytes_read=0, actual_weight_loading_qualified=False,
                qualification='Native HF key/shape/dtype header correspondence only; no checkpoint values or model loading.')
