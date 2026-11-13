"""Verify the successful Qwen suffix transfer and the target model's complete identity."""
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
    document = json.loads(raw)
    if not isinstance(document, dict) or len(raw) != before[2]:
        raise ValueError('invalid receipt document')
    return document, hashlib.sha256(raw).hexdigest()



def check_local_receipts(directory, transfer, files):
    directory = Path(directory)
    identity = {key: transfer[key] for key in ('claim', 'request_id', 'epoch')}
    intent, intent_sha = _read_receipt(directory / 'intent.json')
    expected_intent = dict(identity, deadline=transfer['deadline'], wire_bytes=transfer['wire_bytes'],
                           suffix_bytes=transfer['suffix_bytes'], replay_allowed=False)
    if intent != expected_intent:
        raise ValueError('local suffix intent differs from the original request')
    outcome, outcome_sha = _read_receipt(directory / 'outcome.json')
    if (any(outcome.get(key) != value for key, value in identity.items())
            or outcome.get('passed') is not True or outcome.get('transport_rcs') != [0, 0]
            or outcome.get('wire_bytes') != transfer['wire_bytes']
            or 'cleanup_pending' in outcome or 'error_type' in outcome):
        raise ValueError('local suffix relay has no successful terminal outcome')
    proofs = dict(local_intent_SHA256=intent_sha, local_outcome_SHA256=outcome_sha)
    for role, filename in (('sender', 'sender.stderr.private'), ('receiver', 'receiver.stdout.private')):
        path = directory / filename
        before = _stable_stat(path)
        if not 0 < before[2] <= 32768:
            raise ValueError('invalid local endpoint receipt size')
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if (len(raw) != before[2] or _stable_stat(path) != before
                or digest != outcome.get(role + '_receipt_sha256')):
            raise ValueError('local endpoint receipt bytes differ')
        records = []
        for line in raw.splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and row.get('request_id') == identity['request_id']:
                records.append(row)
        expected = dict(identity, role=role, passed=True, suffix_bytes=transfer['suffix_bytes'], files=files)
        if records != [expected] or records[0]['passed'] is not True:
            raise ValueError('local endpoint did not attest the exact completed transfer')
        proofs['local_' + role + '_SHA256'] = digest
    return proofs


def verify(root, binding, transfer, check_deadline):
    root = Path(root)
    scope = root / 'verified-suffix-v1'
    if scope.is_symlink() or not scope.is_dir():
        raise ValueError('target suffix receipt directory is missing or indirect')
    identity = {key: transfer[key] for key in ('claim', 'request_id', 'epoch')}
    intent, intent_hash = _read_receipt(scope / 'intent.json')
    terminal, terminal_hash = _read_receipt(scope / 'outcome.json')
    expected_intent = dict(identity, role='receiver', passed=False, deadline=transfer['deadline'])
    pid = intent.get('pid')
    if (type(pid) is not int or pid <= 0 or intent.get('passed') is not False
            or intent != dict(expected_intent, pid=pid)):
        raise ValueError('target suffix intent differs from the original request')
    files = binding['files']
    if (terminal.get('passed') is not True or terminal !=
            dict(identity, role='receiver', passed=True, suffix_bytes=transfer['suffix_bytes'], files=files)):
        raise ValueError('target suffix terminal did not confirm the exact published model')
    model = root / 'model'
    if not model.is_dir() or model.is_symlink() or os.path.lexists(root / 'model.incomplete'):
        raise ValueError('target model publication is incomplete')
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
    return dict(passed=True, model_directory=str(model), suffix_target_intent_SHA256=intent_hash,
                suffix_target_outcome_SHA256=terminal_hash,
                files={n: dict(**pin, device=initial[n][0], inode=initial[n][1],
                               mtime_ns=initial[n][3], ctime_ns=initial[n][4]) for n,pin in files.items()},
                index_tensors=indexed['tensors'], weight_bytes_read=indexed['full_file_bytes_read'],
                total_file_bytes=sum(p['bytes'] for p in files.values()),
                dtype_tensor_counts=dict(actual_dtype), model_loaded=False, GPU_calls=0)


def run_remote(root, binding, transfer, seconds):
    start = time.monotonic()
    def check():
        if time.monotonic() - start >= seconds:
            raise TimeoutError('target identity deadline expired')
    result = verify(root, binding, transfer, check)
    result['elapsed_seconds'] = time.monotonic() - start
    return result
