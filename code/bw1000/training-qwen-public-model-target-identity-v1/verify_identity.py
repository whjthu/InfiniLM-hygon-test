"""Verify the original Qwen relay result and the target model's complete identity."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import stat
import time

INDEX = 'model.safetensors.index.json'


def _stable_stat(path):
    value = path.lstat()
    if not stat.S_ISREG(value.st_mode):
        raise ValueError('expected regular target file')
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _read_receipt(path):
    before = _stable_stat(path)
    if not 0 < before[2] <= 65536:
        raise ValueError('invalid relay receipt size')
    raw = path.read_bytes()
    if before != _stable_stat(path):
        raise ValueError('relay receipt changed')
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def verify(root, binding, relay, check_deadline):
    """Use the existing indexed-model reader once, then check remaining metadata."""
    root = Path(root)
    model = root / 'model'
    if not model.is_dir() or model.is_symlink() or os.path.lexists(root / 'model.incomplete'):
        raise ValueError('target model publication is incomplete')
    intent, _ = _read_receipt(root / 'intent.json')
    terminal, terminal_hash = _read_receipt(root / 'outcome.json')
    files = binding['files']
    for doc in (intent, terminal):
        if any(doc.get(key) != value for key, value in relay.items()) or doc.get('files') != files:
            raise ValueError('original relay identity differs')
    if (terminal.get('passed') is not True or terminal.get('owner') != intent.get('owner')
            or terminal.get('model_directory') != str(model)
            or terminal.get('total_file_bytes') != sum(pin['bytes'] for pin in files.values())):
        raise ValueError('target relay terminal did not prove publication')
    if set(files) != {p.name for p in model.iterdir()}:
        raise ValueError('published model file set differs')
    initial = {name: _stable_stat(model / name) for name in files}
    for name, pin in files.items():
        if initial[name][2] != pin['bytes']:
            raise ValueError('target file size differs')
    shards = binding['headers']
    if set(shards) != {n for n in files if n.endswith('.safetensors')}:
        raise ValueError('unexpected weight shard set')
    indexed = inspect_indexed_model(
        model, INDEX, files[INDEX]['SHA256'],
        {n: dict(zip(('device', 'inode', 'bytes', 'mtime_ns'), initial[n][:4])) for n in shards},
        check_deadline=check_deadline, progress=lambda _: None)
    if (indexed['tensors'] != binding['index_tensors'] or any(
            indexed[k] != binding[k] for k in ('tensor_payload_bytes', 'serialized_tensor_elements'))):
        raise ValueError('target index denominator differs')
    actual_dtype = Counter()
    for name, header in shards.items():
        row = indexed['shards'][name]
        if row['SHA256'] != files[name]['SHA256'] or any(row[k] != v for k, v in header.items()):
            raise ValueError('target shard or header digest differs')
        actual_dtype.update(v['dtype'] for v in row['tensors'].values())
    if dict(actual_dtype) != binding['dtype_tensor_counts']:
        raise ValueError('target tensor dtype distribution differs')
    for name in sorted(set(files) - set(shards) - {INDEX}):
        check_deadline()
        raw = (model / name).read_bytes()
        if len(raw) != files[name]['bytes'] or hashlib.sha256(raw).hexdigest() != files[name]['SHA256']:
            raise ValueError('target metadata digest differs')
    check_deadline()
    if any(_stable_stat(model / name) != before for name, before in initial.items()):
        raise ValueError('published model changed during verification')
    return dict(passed=True, model_directory=str(model), relay_outcome_SHA256=terminal_hash,
                files={n: dict(**pin, device=initial[n][0], inode=initial[n][1],
                               mtime_ns=initial[n][3], ctime_ns=initial[n][4]) for n,pin in files.items()},
                index_tensors=indexed['tensors'], weight_bytes_read=indexed['full_file_bytes_read'],
                total_file_bytes=sum(p['bytes'] for p in files.values()),
                dtype_tensor_counts=dict(actual_dtype), model_loaded=False, GPU_calls=0)


def run_remote(root, binding, relay, seconds):
    start = time.monotonic()
    def check():
        if time.monotonic() - start >= seconds:
            raise TimeoutError('target identity deadline expired')
    result = verify(root, binding, relay, check)
    result['elapsed_seconds'] = time.monotonic() - start
    return result
