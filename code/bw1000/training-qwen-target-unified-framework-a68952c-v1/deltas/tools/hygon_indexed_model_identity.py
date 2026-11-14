"""Stream-hash a declared indexed safetensors model without decoding weights."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath


DTYPE_BYTES = {'BOOL': 1, 'U8': 1, 'I8': 1, 'I16': 2, 'I32': 4, 'I64': 8,
               'F16': 2, 'BF16': 2, 'F32': 4, 'F64': 8}


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('duplicate model metadata key')
        value[key] = item
    return value


def flat_name(value):
    if not isinstance(value, str) or not value or Path(value).name != value or '/' in value or '\\' in value or value in ('.', '..'):
        raise ValueError('model identity requires exact flat file names')
    return value


def stat_identity(path):
    value = path.stat()
    return dict(bytes=value.st_size, mtime_ns=value.st_mtime_ns, inode=value.st_ino, device=value.st_dev)


def header_layout(raw, payload_bytes):
    header = json.loads(raw, object_pairs_hook=unique_object)
    if not isinstance(header, dict):
        raise ValueError('invalid safetensors header')
    tensors = {}
    for name, value in header.items():
        if name == '__metadata__':
            if not isinstance(value, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
                raise ValueError('invalid safetensors metadata')
            continue
        if not isinstance(name, str) or not name or not isinstance(value, dict) or set(value) != {'dtype', 'shape', 'data_offsets'}:
            raise ValueError('invalid safetensors tensor schema')
        shape, offsets, dtype = value['shape'], value['data_offsets'], value['dtype']
        if (dtype not in DTYPE_BYTES or not isinstance(shape, list)
                or any(type(n) is not int or n < 0 for n in shape)
                or not isinstance(offsets, list) or len(offsets) != 2
                or any(type(n) is not int for n in offsets)):
            raise ValueError('invalid safetensors dimensions,dtype or offsets')
        count = math.prod(shape)
        begin, end = offsets
        if not 0 <= begin <= end <= payload_bytes or end - begin != count * DTYPE_BYTES[dtype]:
            raise ValueError('safetensors shape/byte range mismatch')
        tensors[name] = dict(dtype=dtype, shape=shape, data_offsets=offsets, elements=count, bytes=end-begin)
    cursor = 0
    for value in sorted(tensors.values(), key=lambda v: tuple(v['data_offsets'])):
        if value['data_offsets'][0] != cursor:
            raise ValueError('safetensors ranges overlap or contain holes')
        cursor = value['data_offsets'][1]
    if not tensors or cursor != payload_bytes:
        raise ValueError('safetensors payload is not completely described')
    return tensors


def inspect_indexed_model(root, index_name, index_SHA256, expected_stats, *, check_deadline, progress):
    """One complete read per declared shard; caller supplies the original deadline."""
    if not callable(check_deadline) or not callable(progress) or not expected_stats:
        raise ValueError('model identity requires a deadline,progress sink and declared shards')
    root = Path(root)
    index = root / flat_name(index_name)
    check_deadline()
    if index.is_symlink() or not 0 < index.stat().st_size <= 8*1024**2:
        raise ValueError('invalid model index file')
    before = stat_identity(index)
    raw = index.read_bytes()
    if hashlib.sha256(raw).hexdigest() != index_SHA256 or before != stat_identity(index):
        raise ValueError('model index identity differs')
    document = json.loads(raw, object_pairs_hook=unique_object)
    weight_map = document['weight_map']
    if not isinstance(weight_map, dict) or not weight_map or any(not isinstance(n, str) or not n for n in weight_map):
        raise ValueError('invalid model index weight map')
    shards = sorted({flat_name(name) for name in weight_map.values()})
    if set(shards) != set(expected_stats) or any(not n.endswith('.safetensors') for n in shards):
        raise ValueError('model shard set differs from declaration')
    complete, all_tensors, read_bytes = {}, {}, 0
    progress(dict(stage='index_verified', planned_shards=shards, completed_shards=[], bytes_read=0))
    for name in shards:
        check_deadline()
        path = root / name
        before = stat_identity(path)
        if path.is_symlink() or any(before[k] != v for k, v in expected_stats[name].items()):
            raise ValueError('model shard stat differs from declaration')
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            first = stream.read(8)
            if len(first) != 8:
                raise ValueError('missing safetensors header length')
            size = int.from_bytes(first, 'little')
            if not 0 < size <= 8*1024**2 or size+8 >= before['bytes']:
                raise ValueError('safetensors header exceeds bound')
            raw_header = stream.read(size)
            if len(raw_header) != size:
                raise ValueError('incomplete safetensors header')
            tensors = header_layout(raw_header, before['bytes']-8-size)
            if set(tensors) != {key for key, shard in weight_map.items() if shard == name}:
                raise ValueError('safetensors header/index tensor set differs')
            digest.update(first);digest.update(raw_header)
            consumed = 8+size
            next_progress = 256*1024**2
            while True:
                check_deadline()
                chunk = stream.read(8*1024**2)
                if not chunk:
                    break
                digest.update(chunk);consumed += len(chunk)
                if consumed >= next_progress:
                    progress(dict(stage='reading_shard', planned_shards=shards, completed_shards=list(complete),
                                  current_shard=name, current_shard_bytes=consumed, bytes_read=read_bytes+consumed))
                    next_progress = consumed+256*1024**2
        if consumed != before['bytes'] or before != stat_identity(path):
            raise ValueError('model shard changed during complete hash')
        read_bytes += consumed
        complete[name] = dict(**before, SHA256=digest.hexdigest(), header_bytes=size,
                              header_SHA256=hashlib.sha256(raw_header).hexdigest(), tensors=tensors)
        for key, value in tensors.items():
            if key in all_tensors:
                raise ValueError('duplicate tensor across model shards')
            all_tensors[key] = value
        progress(dict(stage='shard_complete', planned_shards=shards, completed_shards=list(complete), bytes_read=read_bytes))
    check_deadline()
    tensor_bytes = sum(value['bytes'] for value in all_tensors.values())
    if set(all_tensors) != set(weight_map) or document.get('metadata', {}).get('total_size', tensor_bytes) != tensor_bytes:
        raise ValueError('complete model index payload denominator differs')
    return dict(index=dict(name=index_name, bytes=len(raw), SHA256=index_SHA256), shards=complete,
                full_file_bytes_read=read_bytes, tensors=len(all_tensors), tensor_payload_bytes=tensor_bytes,
                serialized_tensor_elements=sum(v['elements'] for v in all_tensors.values()),
                weight_bytes_read_for_hash=True, tensor_values_decoded=False, model_loaded=False, GPU_calls=0)


def inspect_listed_files(root, expected_stats, *, check_deadline, progress):
    """One complete read per exact declared file, including multi-component models.

    No directory discovery is performed. Stat anchors must come from the owning
    host. Each safetensors header is validated during that same full-file read.
    This establishes current file identities, not historical or release provenance.
    """
    if (not callable(check_deadline) or not callable(progress)
            or type(expected_stats) is not dict or not 0 < len(expected_stats) <= 128):
        raise ValueError('listed identity requires bounded files,deadline and progress')
    root=Path(root)
    if not root.is_dir() or root.is_symlink():
        raise ValueError('invalid listed model root')
    names=[];total=0
    for name,pin in expected_stats.items():
        if not isinstance(name,str) or not name or '\\' in name or ':' in name:
            raise ValueError('invalid listed relative path')
        part=PurePosixPath(name)
        if (part.is_absolute() or not part.parts or part.as_posix()!=name
                or any(p in ('.','..') for p in part.parts)):
            raise ValueError('listed path escapes or aliases the root')
        if (type(pin) is not dict or set(pin)!={'bytes','mtime_ns','inode','device'}
                or any(type(v) is not int or v<0 for v in pin.values())):
            raise ValueError('invalid listed stat anchor')
        total+=pin['bytes'];names.append(name)
    if total>64*1024**3:
        raise ValueError('listed files exceed64GiB read budget')
    names.sort(key=lambda n:PurePosixPath(n).parts)
    # Reject a late invalid path/stat before paying for any tensor payload.
    for name in names:
        check_deadline();p=root
        for part in PurePosixPath(name).parts:
            p=p/part
            if p.is_symlink():raise ValueError('listed identity rejects symlinked paths')
        if not p.is_file() or stat_identity(p)!=expected_stats[name]:
            raise ValueError('listed file stat differs before reading')
    complete={};read_bytes=0
    progress(dict(stage='listed_plan',planned_files=names,planned_bytes=total,completed_files=[],bytes_read=0))
    for name in names:
        check_deadline();path=root/name;before=stat_identity(path)
        if before!=expected_stats[name] or path.is_symlink():
            raise ValueError('listed file changed before its read')
        digest=hashlib.sha256();consumed=0;layout=None;header_info={}
        with path.open('rb') as stream:
            if name.endswith('.safetensors'):
                first=stream.read(8)
                if len(first)!=8:raise ValueError('missing listed safetensors length')
                length=int.from_bytes(first,'little')
                if not 0<length<=8*1024**2 or length+8>=before['bytes']:
                    raise ValueError('listed safetensors header exceeds bound')
                raw=stream.read(length)
                if len(raw)!=length:raise ValueError('truncated listed safetensors header')
                layout=header_layout(raw,before['bytes']-8-length)
                digest.update(first);digest.update(raw);consumed=8+length
                header_info=dict(header_bytes=length,header_SHA256=hashlib.sha256(raw).hexdigest(),
                                 tensors=layout,tensor_payload_bytes=before['bytes']-8-length)
            next_progress=256*1024**2
            while True:
                check_deadline();chunk=stream.read(4*1024**2)
                if not chunk:break
                digest.update(chunk);consumed+=len(chunk)
                if consumed>=next_progress:
                    progress(dict(stage='listed_reading',planned_files=names,completed_files=list(complete),
                                  current_file=name,current_file_bytes=consumed,bytes_read=read_bytes+consumed))
                    next_progress=consumed+256*1024**2
        if consumed!=before['bytes'] or stat_identity(path)!=before or path.is_symlink():
            raise ValueError('listed file changed during complete read')
        read_bytes+=consumed
        complete[name]=dict(**before,SHA256=digest.hexdigest(),**header_info)
        progress(dict(stage='listed_complete',planned_files=names,completed_files=list(complete),
                      current_file=name,file=complete[name],bytes_read=read_bytes))
    check_deadline()
    if read_bytes!=total or set(complete)!=set(expected_stats):
        raise ValueError('listed file read denominator differs')
    canonical={n:dict(bytes=p['bytes'],SHA256=p['SHA256']) for n,p in complete.items()}
    identity=hashlib.sha256(json.dumps(canonical,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return dict(files=complete,planned_files=len(names),completed_files=len(complete),
                full_file_bytes_read=read_bytes,canonical_file_manifest_SHA256=identity,
                canonical_manifest_algorithm='SHA256 of UTF8 JSON,sort_keys=True,separators comma/colon;relative path -> bytes and full-file SHA256.',
                tensor_values_decoded=False,model_loaded=False,GPU_calls=0,
                historical_or_release_provenance_qualified=False)
