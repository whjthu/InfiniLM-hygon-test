"""Pure stdlib metadata relay contract; no framework or device imports."""
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import resource
import signal
import stat
import struct
import time
import zlib


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode('utf-8') + b'\n'


def require(test, message):
    if not test:
        raise ValueError(message)


def relative(name):
    p = PurePosixPath(name)
    require(isinstance(name, str) and name == p.as_posix() and not p.is_absolute()
            and '..' not in p.parts and '\\' not in name and name not in ('', '.'),
            'unsafe member path')
    return p


def stable(st):
    return [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]


def no_links(path):
    path = Path(path)
    require(path.is_absolute(), 'absolute path required')
    for part in [*reversed(path.parents), path]:
        require(not part.is_symlink(), 'symlink path rejected')
    return path


def tick():
    require(time.time() < DEADLINE and time.monotonic() < MONO_END, 'global epoch exhausted')


def arm(epoch, deadline):
    global DEADLINE, MONO_END
    require(math.isfinite(epoch) and math.isfinite(deadline) and deadline == epoch + 300,
            'single300s epoch required')
    remaining = deadline - time.time()
    require(0 < remaining <= 300, 'expired or invalid epoch')
    DEADLINE, MONO_END = deadline, time.monotonic() + remaining
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('metadata deadline')))
    signal.setitimer(signal.ITIMER_REAL, remaining)
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2, 512 * 1024**2))
    allowed = sorted(os.sched_getaffinity(0))
    require(len(allowed) >= 2, 'two available CPU cores required')
    os.sched_setaffinity(0, allowed[:2])
    require(os.nice(10) == 10, 'endpoint nice must be10')
    return allowed[:2]


def save(path, value):
    raw = canonical(value)
    with Path(path).open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def sync_directory(root):
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def durable_bundle(root, compressed, derived, context):
    """Publish recoverable source artifacts before any binary relay output."""
    for name, raw in [('bundle.jsonframe.gz', compressed), ('initial-bindings.json', derived)]:
        with (root/name).open('xb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    receipt = dict(context, validated=True, metadata_only=True,
                   bundle=dict(name='bundle.jsonframe.gz', bytes=len(compressed), SHA256=sha(compressed)),
                   initial_bindings=dict(name='initial-bindings.json', bytes=len(derived), SHA256=sha(derived)),
                   source_validation_not_receiver_success=True)
    save(root/'validation.json', receipt)
    sync_directory(root)
    return receipt


def system_cost(started_mono):
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return dict(endpoint_wall_seconds=time.monotonic()-started_mono,
                nice=os.getpriority(os.PRIO_PROCESS, 0),
                self_user_CPU_seconds=usage.ru_utime, self_system_CPU_seconds=usage.ru_stime,
                self_maxrss_bytes=int(usage.ru_maxrss)*1024, self_maxrss_native_unit='Linux_KiB',
                maxrss_is_process_high_watermark_not_cgroup_limit=True,
                RLIMIT_AS_bytes=512*1024**2, GPU_busy='NOT_MEASURED',
                kernel_and_filesystem_physical_readahead_IO='UNKNOWN_NOT_MEASURED')


def own(root, context):
    root = no_links(root)
    require(root.parent.is_dir(), 'registered parent missing')
    root.mkdir(mode=0o700)  # Existing namespace is never retried or overwritten.
    fields = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
    identity = dict(context, pid=os.getpid(), startticks=int(fields[19]),
                    uid=os.getuid(), boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    save(root / 'intent.json', identity)
    return identity


def read_pinned(path, pin, maximum):
    tick()
    path = no_links(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and before.st_size == pin['bytes']
                and 0 < before.st_size <= maximum, 'pinned file size/type differs')
        raw = stream.read(before.st_size + 1)
        require(len(raw) == before.st_size and sha(raw) == pin['SHA256'], 'pinned file SHA differs')
        require(stable(before) == stable(os.fstat(stream.fileno())) == stable(path.stat()),
                'file identity changed')
    tick()
    return raw, stable(before)


def initial_row(case, rank, native, receipt, native_pin, receipt_pin):
    require(native['rank'] == receipt['rank'] == rank and native['world_size'] == receipt['world_size'] == 8,
            'rank identity differs')
    require(native['global_step'] == 25 and [x['step'] for x in native['step_records']] == list(range(1, 26)),
            'native25 boundary differs')
    require(receipt['completed_global_step'] == 2 and
            (receipt['optimizer_updates'], receipt['micro_forwards'], receipt['clip_calls']) == (2, 8, 2),
            'two-update capture boundary differs')
    rows = receipt['files']
    require(len(rows) == 25 and len({x['name'] for x in rows}) == 25, 'capture member denominator differs')
    initial = [x for x in rows if x['name'] == 'initial.pt']
    require(len(initial) == 1, 'missing or duplicate initial artifact')
    initial = initial[0]
    require(type(initial['bytes']) is int and 0 < initial['bytes'] < 2 * 1024**3 and
            re.fullmatch('[0-9a-f]{64}', initial['SHA256']) is not None and
            re.fullmatch('[0-9a-f]{64}', native['initial_trainable_sha256']) is not None,
            'initial pin malformed or omitted')
    base = case['run_root']
    return dict(rank=rank, source_native=dict(path=base + '/train/nvidia-baseline/rank-%02d.json' % rank, **native_pin),
                source_capture_receipt=dict(path=base + '/capture/rank-%02d/receipt.json' % rank, **receipt_pin),
                initial_path=base + '/capture/rank-%02d/initial.pt' % rank,
                initial_file_bytes=initial['bytes'], initial_file_SHA256=initial['SHA256'],
                named_initial_digest=native['initial_trainable_sha256'], native_global_step=25,
                captured_optimizer_updates=2, initial_tensor_read=False)


def pack(files, plan, context):
    rows = []
    for name in sorted(files):
        relative(name)
        raw = files[name]
        require(0 < len(raw) <= plan['max_file_bytes'], 'member budget exceeded')
        rows.append(dict(name=name, bytes=len(raw), SHA256=sha(raw)))
    index = canonical(dict(context=context, files=rows))
    require(len(index) <= plan['max_index_bytes'], 'index budget exceeded')
    raw = struct.pack('<Q', len(index)) + index + b''.join(files[x['name']] for x in rows)
    require(len(raw) <= plan['max_decompressed_bytes'], 'raw bundle budget exceeded')
    compressed = gzip.compress(raw, compresslevel=6, mtime=0)
    require(len(compressed) <= plan['max_compressed_bytes'], 'compressed budget exceeded')
    return compressed


def unpack(compressed, plan, context):
    require(0 < len(compressed) <= plan['max_compressed_bytes'], 'compressed budget exceeded')
    dec = zlib.decompressobj(16 + zlib.MAX_WBITS)
    raw = dec.decompress(compressed, plan['max_decompressed_bytes'] + 1)
    require(len(raw) <= plan['max_decompressed_bytes'] and dec.eof and not dec.unconsumed_tail
            and not dec.unused_data, 'truncated, trailing or oversized compressed bundle')
    require(len(raw) >= 8, 'truncated index prefix')
    n = struct.unpack('<Q', raw[:8])[0]
    require(0 < n <= plan['max_index_bytes'] and len(raw) >= 8 + n, 'index bounds differ')
    index = json.loads(raw[8:8+n])
    require(canonical(index) == raw[8:8+n] and index['context'] == context, 'index/context differs')
    rows = index['files']
    require([x['name'] for x in rows] == sorted({x['name'] for x in rows}), 'duplicate/unsorted members')
    files, offset = {}, 8 + n
    for row in rows:
        relative(row['name'])
        size = row['bytes']
        require(type(size) is int and 0 < size <= plan['max_file_bytes'], 'member size differs')
        data = raw[offset:offset+size]
        require(len(data) == size and sha(data) == row['SHA256'], 'member SHA/truncation differs')
        files[row['name']] = data
        offset += size
    require(offset == len(raw), 'trailing bundle body')
    return files
