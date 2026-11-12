"""CPU metadata checks for the original Z-Image Turbo differential adapter.

This module never imports a tensor library, reads tensor bodies or fuses weights.
The original loader can silently ignore unmatched adapter targets; preflight
therefore requires complete, unambiguous A/B coverage before a device run.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import struct


TARGETS = ('to_q', 'to_k', 'to_v', 'to_out.0', 'w1', 'w2', 'w3')
DTYPE_BYTES = {'F32': 4, 'BF16': 2}


def validate_header(tensors: dict, *, file_bytes: int, header_bytes: int) -> None:
    if type(tensors) is not dict or not tensors:
        raise ValueError('empty or malformed safetensors header')
    if type(file_bytes) is not int or type(header_bytes) is not int or header_bytes <= 0:
        raise ValueError('invalid safetensors extent')
    ranges = []
    for name, value in tensors.items():
        if not isinstance(name, str) or not name or type(value) is not dict:
            raise ValueError('invalid tensor entry')
        if set(value) != {'dtype', 'shape', 'data_offsets'}:
            raise ValueError('unexpected tensor fields')
        shape, offsets = value['shape'], value['data_offsets']
        if (type(shape) is not list or not shape
                or any(type(d) is not int or d <= 0 for d in shape)
                or value['dtype'] not in DTYPE_BYTES):
            raise ValueError('invalid tensor shape or dtype')
        if (type(offsets) is not list or len(offsets) != 2
                or any(type(x) is not int for x in offsets)):
            raise ValueError('invalid tensor offsets')
        begin, end = offsets
        size = math.prod(shape) * DTYPE_BYTES[value['dtype']]
        if begin < 0 or end - begin != size or size > 2**63 - 1:
            raise ValueError('tensor byte extent differs from its shape')
        ranges.append((begin, end))
    ranges.sort()
    cursor = 0
    for begin, end in ranges:
        if begin != cursor:
            raise ValueError('overlapping or missing tensor payload range')
        cursor = end
    if cursor != file_bytes - 8 - header_bytes:
        raise ValueError('tensor payload does not cover the complete file')


def read_header(path: Path, deadline, *, expected: dict) -> dict:
    deadline()
    if path.is_symlink() or not path.is_file():
        raise ValueError('missing or symlinked original safetensors file')
    before = path.stat()
    if before.st_size != expected['bytes']:
        raise ValueError('safetensors file size differs from original')
    with path.open('rb') as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError('missing safetensors length')
        length = struct.unpack('<Q', prefix)[0]
        if not 0 < length <= 4 * 1024**2 or length != expected['header_bytes']:
            raise ValueError('safetensors header exceeds original bounded extent')
        raw = stream.read(length)
    if len(raw) != length or hashlib.sha256(raw).hexdigest() != expected['header_SHA256']:
        raise ValueError('safetensors header differs from original')
    after = path.stat()
    if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)):
        raise ValueError('safetensors file changed during header read')
    value = json.loads(raw)
    if type(value) is not dict:
        raise ValueError('malformed safetensors header')
    value.pop('__metadata__', None)
    validate_header(value, file_bytes=before.st_size, header_bytes=length)
    deadline()
    return value


def adapter_pairs(adapter: dict) -> dict[str, tuple[str, str]]:
    """The observed canonical PEFT A/B format; other formats require new evidence."""
    pairs = {}
    for name in adapter:
        if not name.startswith('diffusion_model.'):
            raise ValueError('unexpected preset adapter prefix')
        if not name.endswith(('.lora_A.weight', '.lora_B.weight')):
            raise ValueError('unsupported preset adapter tensor')
        suffix = '.lora_A.weight' if name.endswith('.lora_A.weight') else '.lora_B.weight'
        target = name[len('diffusion_model.'):-len(suffix)]
        if not target or '..' in target:
            raise ValueError('invalid preset target')
        a = 'diffusion_model.' + target + '.lora_A.weight'
        b = 'diffusion_model.' + target + '.lora_B.weight'
        if a not in adapter or b not in adapter:
            raise ValueError('orphan preset adapter factor')
        pairs[target] = (a, b)
    if not pairs or {key for pair in pairs.values() for key in pair} != set(adapter):
        raise ValueError('preset adapter coverage is incomplete')
    return pairs


def check_metadata(base_shards: list[dict], adapter: dict) -> dict:
    base = {}
    for shard in base_shards:
        if set(base).intersection(shard):
            raise ValueError('duplicate base tensor across shards')
        base.update(shard)
    pairs = adapter_pairs(adapter)
    dimensions = {}
    for target, (a, b) in pairs.items():
        if target + '.weight' not in base:
            raise ValueError('preset adapter target is absent from base')
        aa, bb, ww = adapter[a], adapter[b], base[target + '.weight']
        if (len(aa['shape']) != 2 or len(bb['shape']) != 2 or len(ww['shape']) != 2
                or aa['shape'][0] != bb['shape'][1]
                or [bb['shape'][0], aa['shape'][1]] != ww['shape']):
            raise ValueError('preset adapter matrix shapes do not match base')
        if aa['dtype'] != 'BF16' or bb['dtype'] != 'BF16' or ww['dtype'] != 'F32':
            raise ValueError('original preset/base storage dtype differs')
        dimensions[target] = dict(A=aa['shape'], B=bb['shape'], base=ww['shape'])
    new_targets = {}
    for name, value in base.items():
        if not name.endswith('.weight'):
            continue
        module = name[:-len('.weight')]
        if any(module == suffix or module.endswith('.' + suffix) for suffix in TARGETS):
            if len(value['shape']) != 2 or value['dtype'] != 'F32':
                raise ValueError('new LoRA target differs from original linear storage')
            new_targets[module] = value['shape']
    if not new_targets:
        raise ValueError('no original differential LoRA targets')
    return dict(base_tensors=len(base), base_elements=sum(math.prod(v['shape']) for v in base.values()),
                preset_tensors=len(adapter), preset_pairs=len(pairs),
                preset_elements=sum(math.prod(v['shape']) for v in adapter.values()),
                preset_ranks=sorted({adapter[a]['shape'][0] for a, _ in pairs.values()}),
                new_rank4_modules=len(new_targets), new_rank4_tensors=2*len(new_targets),
                new_rank4_elements=sum(4*sum(s) for s in new_targets.values()),
                preset_dimensions=dimensions, new_target_shapes=new_targets,
                qualification='Complete header name/shape coverage only; values, actual modules, fusion, freezing and numerical equivalence remain unverified.',
                tensor_body_bytes_read=0, preset_adapter_fused=False)
