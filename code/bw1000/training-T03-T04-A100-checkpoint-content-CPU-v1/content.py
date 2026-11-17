"""Bounded byte identity and non-executing format inspection of four checkpoints."""
import collections
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import pickletools
import pwd
import re
import resource
import signal
import socket
import stat
import struct
import time
import zipfile


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def stable(st):
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns


def owner():
    return dict(host=socket.gethostname(), uid=os.getuid(), PID=os.getpid(),
                startticks=int(Path('/proc/self/stat').read_text().split(') ', 1)[1].split()[19]),
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())


def safe_path(path):
    require(path.is_absolute() and not any(p.is_symlink() for p in [path, *path.parents]), 'PATH_LINK_OR_RELATIVE')
    require(path.stat().st_uid == os.getuid(), 'PATH_NOT_OWNED_UID')


class Budget:
    def __init__(self, deadline, byte_cap):
        self.deadline, self.byte_cap, self.read_bytes = deadline, byte_cap, 0

    def check(self, size=0):
        self.read_bytes += size
        require(time.time() < self.deadline and self.read_bytes <= self.byte_cap, 'DEADLINE_OR_BYTE_BUDGET')


class CountingFile:
    """Bounds archive index/header/metadata reads as well as the whole-file hash."""
    def __init__(self, stream, budget):
        self.stream, self.budget = stream, budget

    def read(self, size=-1):
        self.budget.check()
        if size == -1:
            size = os.fstat(self.stream.fileno()).st_size - self.stream.tell()
        require(type(size) is int and 0 <= size <= 8 * 1024**2, 'UNBOUNDED_ARCHIVE_READ')
        value = self.stream.read(size)
        self.budget.check(len(value))
        return value

    def seek(self, *args):
        self.budget.check()
        return self.stream.seek(*args)

    def tell(self):
        return self.stream.tell()

    def seekable(self):
        return True


def json_read(path, limit, budget):
    with checked_file(path, None, budget) as (stream, before):
        require(0 < before.st_size <= limit, 'JSON_BYTE_LIMIT')
        raw = stream.read(before.st_size)
        require(len(raw) == before.st_size, 'JSON_SHORT_READ')
        return json.loads(raw), dict(bytes=len(raw), SHA256=hashlib.sha256(raw).hexdigest())


class checked_file:
    def __init__(self, path, expected_bytes, budget):
        self.path, self.expected_bytes, self.budget = path, expected_bytes, budget

    def __enter__(self):
        self.budget.check(); safe_path(self.path)
        self.fd = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW)
        self.before = os.fstat(self.fd)
        require(stat.S_ISREG(self.before.st_mode) and self.before.st_size > 0, 'NOT_POSITIVE_REGULAR_FILE')
        if self.expected_bytes is not None:
            require(self.before.st_size == self.expected_bytes, 'ORIGINAL_FILE_SIZE_CHANGED')
        self.raw = os.fdopen(self.fd, 'rb', buffering=0, closefd=False)
        return CountingFile(self.raw, self.budget), self.before

    def __exit__(self, kind, value, tb):
        try:
            require(stable(self.before) == stable(os.fstat(self.fd)) == stable(self.path.stat()), 'ORIGINAL_STABLE_FD_CHANGED')
        finally:
            self.raw.close(); os.close(self.fd)


DTYPES = {'F64': 8, 'F32': 4, 'F16': 2, 'BF16': 2, 'I64': 8, 'I32': 4,
          'I16': 2, 'I8': 1, 'U8': 1, 'BOOL': 1}


def safetensors_header(stream, total_bytes):
    stream.seek(0)
    first = stream.read(8); require(len(first) == 8, 'SAFETENSORS_FIRST8')
    length = struct.unpack('<Q', first)[0]
    require(0 < length <= 4 * 1024**2 and 8 + length <= total_bytes, 'SAFETENSORS_HEADER_LIMIT')
    raw = stream.read(length); require(len(raw) == length, 'SAFETENSORS_HEADER_SHORT')
    header = json.loads(raw); require(isinstance(header, dict) and len(header) <= 4097, 'SAFETENSORS_SCHEMA')
    tensors, ranges = [], []
    for name, row in header.items():
        if name == '__metadata__':
            continue
        require(isinstance(name, str) and len(name) <= 512 and isinstance(row, dict), 'SAFETENSORS_NAME')
        dtype, shape, offsets = row.get('dtype'), row.get('shape'), row.get('data_offsets')
        require(dtype in DTYPES and isinstance(shape, list) and len(shape) <= 8, 'SAFETENSORS_DTYPE_SHAPE')
        require(all(type(x) is int and x > 0 for x in shape), 'SAFETENSORS_DIMENSION')
        require(isinstance(offsets, list) and len(offsets) == 2 and all(type(x) is int for x in offsets), 'SAFETENSORS_OFFSETS')
        count = math.prod(shape); lo, hi = offsets
        require(0 <= lo < hi <= total_bytes - 8 - length and hi - lo == count * DTYPES[dtype], 'SAFETENSORS_RANGE')
        tensors.append(dict(name=name, dtype=dtype, shape=shape, elements=count))
        ranges.append((lo, hi))
    ranges.sort()
    require(ranges and ranges[0][0] == 0 and ranges[-1][1] == total_bytes - 8 - length,
            'SAFETENSORS_PAYLOAD_COVERAGE')
    require(all(a[1] == b[0] for a, b in zip(ranges, ranges[1:])), 'SAFETENSORS_OVERLAP_OR_GAP')
    return dict(header_bytes=length, header_SHA256=hashlib.sha256(raw).hexdigest(),
                tensor_count=len(tensors), parameter_elements=sum(t['elements'] for t in tensors),
                dtype_counts=dict(collections.Counter(t['dtype'] for t in tensors)), tensors=tensors,
                floating_values_or_finiteness_checked=False)


def static_pickle(raw):
    """Scan opcodes; never unpickle, resolve a GLOBAL, REDUCE, BUILD or call a constructor."""
    require(0 < len(raw) <= 4 * 1024**2, 'PICKLE_METADATA_LIMIT')
    histogram = collections.Counter(); globals_seen = set(); markers = set(); stopped = False
    known = {'state', 'param_groups', 'step', 'exp_avg', 'exp_avg_sq', 'max_exp_avg_sq',
             'python', 'numpy', 'cpu', 'cuda', 'last_epoch', '_step_count', 'base_lrs', '_last_lr'}
    for count, (op, arg, position) in enumerate(pickletools.genops(raw), 1):
        require(count <= 500000, 'PICKLE_OPCODE_LIMIT')
        histogram[op.name] += 1
        if op.name == 'GLOBAL':
            require(isinstance(arg, str) and len(arg) <= 256 and re.fullmatch(r'[A-Za-z0-9_\.]+ [A-Za-z0-9_]+', arg),
                    'PICKLE_GLOBAL_ENCODING')
            globals_seen.add(arg)
            require(len(globals_seen) <= 64, 'PICKLE_GLOBAL_COUNT_LIMIT')
        elif op.name in {'UNICODE', 'BINUNICODE', 'SHORT_BINUNICODE', 'BINUNICODE8'} and arg in known:
            markers.add(arg)
        if op.name == 'STOP':
            require(position + 1 == len(raw), 'PICKLE_TRAILING_BYTES')
            stopped = True
    require(stopped, 'PICKLE_STOP_MISSING')
    return dict(bytes=len(raw), SHA256=hashlib.sha256(raw).hexdigest(), opcode_counts=dict(histogram),
                GLOBAL_names=sorted(globals_seen), structural_keyword_markers=sorted(markers),
                GLOBAL_or_REDUCE_executed=False, complete_optimizer_schema_interpreted=False,
                tensor_shapes_or_values_decoded=False, RNG_restoration_qualified=False)


def archive_metadata(stream):
    stream.seek(0)
    with zipfile.ZipFile(stream, 'r') as archive:
        entries = archive.infolist(); require(0 < len(entries) <= 20000, 'ARCHIVE_MEMBER_LIMIT')
        names = [entry.filename for entry in entries]
        require(len(set(names)) == len(names), 'DUPLICATE_ARCHIVE_MEMBER')
        for entry in entries:
            parts = Path(entry.filename).parts
            require(not entry.filename.startswith('/') and '..' not in parts and '\\' not in entry.filename,
                    'ARCHIVE_MEMBER_PATH')
            require(not entry.flag_bits & 1, 'ENCRYPTED_ARCHIVE_REJECTED')
        matches = [e for e in entries if e.filename.endswith('/data.pkl') or e.filename == 'data.pkl']
        require(len(matches) == 1 and matches[0].file_size <= 4 * 1024**2, 'ARCHIVE_PICKLE_METADATA')
        raw = archive.read(matches[0])
        scan = static_pickle(raw)
        storage_entries = [e for e in entries if '/data/' in e.filename]
        return dict(member_count=len(entries), storage_member_count=len(storage_entries),
                    declared_storage_bytes=sum(e.file_size for e in storage_entries), pickle_metadata=scan,
                    archive_storage_members_extracted=False, arbitrary_pickle_execution=False)


def tail_keys(native):
    rows = []
    def walk(v, depth=0):
        require(depth <= 12, 'NATIVE_TAIL_DEPTH')
        if isinstance(v, dict):
            step = v.get('global_step', v.get('step'))
            if type(step) is int and 21 <= step <= 25:
                names, losses, references = [], {}, []
                for key, value in v.items():
                    if not isinstance(key, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]{0,63}', key):
                        continue
                    if any(word in key.lower() for word in ('password', 'secret', 'token', 'credential', 'environment')):
                        continue
                    names.append(key)
                    if 'loss' in key.lower() and type(value) in (int, float) and math.isfinite(value):
                        losses[key] = value
                    if any(word in key.lower() for word in ('input', 'rng', 'microbatch', 'accum', 'cursor', 'sampler', 'optimizer')):
                        references.append(dict(field=key, value_type=type(value).__name__, nonempty=bool(value)))
                require(len(names) <= 64 and len(references) <= 32 and len(rows) < 256, 'NATIVE_TAIL_FIELD_LIMIT')
                rows.append(dict(global_step=step, field_names=sorted(names), finite_loss_scalars=losses,
                                 reference_fields=references))
            for value in v.values():
                if isinstance(value, (dict, list)):
                    walk(value, depth + 1)
        elif isinstance(v, list):
            require(len(v) <= 4096, 'NATIVE_TAIL_LIST_LIMIT')
            for value in v:
                if isinstance(value, (dict, list)):
                    walk(value, depth + 1)
    walk(native)
    return rows


def main(context, plan, claim):
    wall = time.monotonic(); epoch, deadline = context['epoch'], context['deadline']
    require(deadline == epoch + 600 and epoch <= time.time() < deadline, 'ORIGIN_EPOCH')
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('ORIGINAL_EPOCH_EXPIRED')))
    signal.setitimer(signal.ITIMER_REAL, deadline - time.time())
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2, 512 * 1024**2))
    require(claim['root_GO'] is True and claim['CPU_GO'] is True and claim['GPU_GO'] is False, 'CLAIM_GO')
    require(pwd.getpwuid(os.getuid()).pw_name == 'whj' and {2, 3}.issubset(os.sched_getaffinity(0)), 'ORIGINAL_ACCOUNT_OR_CPUS')
    os.sched_setaffinity(0, {2, 3}); require(os.sched_getaffinity(0) == {2, 3}, 'CPU_AFFINITY')
    who = owner(); run = Path(plan['new_namespace']); safe_path(run.parent); run.mkdir(mode=0o700, exist_ok=False)
    def save(name, value):
        with (run / name).open('xb') as stream:
            stream.write(encoded(value)); stream.flush(); os.fsync(stream.fileno())
    save('intent.json', dict(**context, owner=who, actual_affinity=[2, 3]))
    print(json.dumps(dict(kind='READY', **context, owner=who)), flush=True)
    budget = Budget(deadline, plan['maximum_explicit_read_bytes'])
    result = dict(**context, owner=who, cases=[dict(case=c['case'], checkpoints=[], tail_metadata=[], state='NOT_STARTED')
                                             for c in plan['cases']], planned_files=56, planned_checkpoints=4,
                  planned_whole_file_bytes=plan['whole_file_bytes'], passed=False, GPU_GO=False, recovery_qualified=False)
    failure = None; hashed_bytes = 0
    try:
        prior, prior_pin = json_read(Path(plan['prior_remote_outcome']), 2 * 1024**2, budget)
        require(prior_pin['SHA256'] == plan['prior_remote_outcome_SHA256'] and prior['request_id'] == plan['prior_request_id']
                and prior['passed'] is True, 'ORIGINAL_METADATA_AUTHORITY')
        for case in plan['cases']:
            original_case = next(c for c in prior['cases'] if c['case'] == case['case'])
            require({r['rank'] for r in original_case['native_rank_metadata']} == set(range(8)), 'PRIOR_NATIVE_RANK_SCHEMA')
            for cp in original_case['checkpoints']:
                require(set(cp['small_JSON']) == {'trainer_state.json', 'args.json', 'adapter_config.json'},
                        'PRIOR_SMALL_JSON_SCHEMA')
        result['prior_metadata_outcome'] = prior_pin
        for case, row in zip(plan['cases'], result['cases']):
            row['state'] = 'READING_CONTENT_IDENTITY'
            original_case = next(c for c in prior['cases'] if c['case'] == case['case'])
            binding, binding_pin = json_read(Path(case['original_root']) / 'src' / case['native_binding_file'],
                                            2 * 1024**2, budget)
            require(binding_pin['SHA256'] == case['native_binding_SHA256'], 'ORIGINAL_CASE_NATIVE_BINDING_CHANGED')
            require(binding['spec']['case'] == case['case'], 'NATIVE_BINDING_CASE_DIFFERS')
            trainables = binding['trainable_manifest']
            require(isinstance(trainables, list) and 0 < len(trainables) <= 4096, 'NATIVE_BINDING_MANIFEST')
            require(len({t['name'] for t in trainables}) == len(trainables), 'NATIVE_BINDING_DUPLICATE_NAMES')
            for tensor in trainables:
                require(isinstance(tensor, dict) and isinstance(tensor['shape'], list)
                        and all(type(x) is int and x > 0 for x in tensor['shape'])
                        and math.prod(tensor['shape']) == tensor['numel'], 'NATIVE_BINDING_SHAPE_ELEMENTS')
            expected_tensors, expected_elements = len(trainables), sum(t['numel'] for t in trainables)
            require(expected_tensors == case['expected_adapter_tensors']
                    and expected_elements == case['expected_adapter_elements'], 'CASE_NATIVE_SCHEMA_AUTHORITY')
            row['native_binding_authority'] = dict(**binding_pin, trainable_tensors=expected_tensors,
                                                  trainable_elements=expected_elements,
                                                  initial_tensor_values_reverified=False)
            for cp in case['checkpoints']:
                root = Path(cp['path']); safe_path(root)
                original = next(c for c in original_case['checkpoints'] if c['expected_global_step'] == cp['global_step'])
                require(original['path'] == str(root) and original['step_matches'], 'ORIGINAL_CHECKPOINT_PATH')
                out = dict(path=str(root), global_step=cp['global_step'], members={})
                row['checkpoints'].append(out)
                for name, pin in cp['members'].items():
                    require(original['members'][name]['present'] and original['members'][name]['bytes'] == pin['bytes'],
                            'ORIGINAL_MEMBER_STAT_AUTHORITY')
                    with checked_file(root / name, pin['bytes'], budget) as (stream, before):
                        digest = hashlib.sha256(); remaining = before.st_size
                        while remaining:
                            block = stream.read(min(1024**2, remaining)); require(block, 'HASH_SHORT_READ')
                            digest.update(block); remaining -= len(block); hashed_bytes += len(block)
                        entry = dict(bytes=before.st_size, SHA256=digest.hexdigest())
                        out['members'][name] = entry
                        if name.endswith('.json'):
                            require(entry['SHA256'] == original['small_JSON'][name]['SHA256'], 'ORIGINAL_SMALL_JSON_CHANGED')
                        elif name.endswith('.safetensors'):
                            entry['format'] = safetensors_header(stream, before.st_size)
                            require(entry['format']['tensor_count'] == expected_tensors
                                    and entry['format']['parameter_elements'] == expected_elements,
                                    'ORIGINAL_ADAPTER_PARAMETER_SCHEMA')
                            entry['format']['counts_vs_same_case_binding_match'] = True
                            entry['format']['exact_names_and_order_vs_model_qualified'] = False
                        elif name.endswith(('.pt', '.pth')):
                            entry['format'] = archive_metadata(stream)
                out['full_14_byte_identities_verified'] = len(out['members']) == 14
            root = Path(case['original_root'])
            for rank in range(8):
                native, pin = json_read(root / ('train/nvidia-baseline/rank-%02d.json' % rank), 524288, budget)
                original_rank = next(r for r in original_case['native_rank_metadata'] if r['rank'] == rank)
                require(pin['SHA256'] == original_rank['SHA256'], 'ORIGINAL_NATIVE_METADATA_CHANGED')
                row['tail_metadata'].append(dict(rank=rank, **pin, tail_rows=tail_keys(native)))
            row['state'] = 'CONTENT_IDENTITY_COMPLETED_SCHEMA_LIMITATIONS_RETAINED'
        result['passed'] = True
    except BaseException as exc:
        failure = exc; result.update(error_type=type(exc).__name__, error_reason=str(exc)[:180])
    finally:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        result.update(finished_epoch=time.time(), wall_seconds=time.monotonic() - wall, hashed_file_bytes=hashed_bytes,
                      explicit_read_bytes=budget.read_bytes, actual_affinity=sorted(os.sched_getaffinity(0)), owner_end=owner(),
                      self_cpu_user_seconds=usage.ru_utime, self_cpu_system_seconds=usage.ru_stime,
                      self_maxrss_bytes=usage.ru_maxrss * 1024,
                      arbitrary_pickle_executed=False, Torch_or_SDK_imported=False,
                      optimizer_tensor_numeric_or_full_structure_qualified=False)
        save('outcome.json', result)
        summary = [dict(case=r['case'], state=r['state'], checkpoints=len(r['checkpoints']),
                        files=sum(len(c['members']) for c in r['checkpoints']), tail_ranks=len(r['tail_metadata']))
                   for r in result['cases']]
        print(json.dumps(dict(kind='OUTCOME', passed=result['passed'], summary_cases=summary,
                              hashed_file_bytes=hashed_bytes, explicit_read_bytes=budget.read_bytes,
                              outcome_SHA256=hashlib.sha256(encoded(result)).hexdigest(), GPU_GO=False)), flush=True)
    if failure is not None:
        raise SystemExit(1)


if __name__ == '__main__':
    main(CONTEXT, PLAN, CLAIM)
