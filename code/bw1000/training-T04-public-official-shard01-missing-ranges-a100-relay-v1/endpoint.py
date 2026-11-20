"""Whole-file relay endpoint for the official, pinned LLaVA checkpoint shard."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import sys
import time

CHUNK = 1024**2
CHECKPOINT_SIZE = 64 * CHUNK
COMMIT = b'LLAVA_SOURCE_WHOLE_SHA_VERIFIED\n'


def validate(ok, reason):
    if not ok:
        raise ValueError(reason)


def identity(info):
    return dict(device=info.st_dev, inode=info.st_ino, size=info.st_size,
                mtime_ns=str(info.st_mtime_ns), ctime_ns=str(info.st_ctime_ns))


def open_regular(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    stream = os.fdopen(fd, 'rb')
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        stream.close()
        raise ValueError('EXPECTED_REGULAR_FILE')
    return stream


def blocks(stream, remaining):
    while remaining:
        data = stream.read(min(CHUNK, remaining))
        if not data:
            raise EOFError('INCOMPLETE_CHECKPOINT_STREAM')
        remaining -= len(data)
        yield data


def frame(binding):
    return (json.dumps(binding, sort_keys=True, separators=(',', ':')) + '\n').encode()


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def receipt(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    sync_directory(path.parent)


def segments(plan):
    cursor = 0
    retained = plan['retained']
    validate(len(retained) == 8 and sum(r['length'] for r in retained) == plan['bytes']-plan['wire_bytes'],
             'EXACT_RETAINED_SCOPE')
    for item in retained:
        offset, length = item['offset'], item['length']
        validate(cursor <= offset and length > 0 and offset+length <= plan['bytes'], 'RETAINED_RANGE_ORDER')
        if offset > cursor:
            yield offset-cursor, None
        yield length, item
        cursor = offset+length
    if cursor < plan['bytes']:
        yield plan['bytes']-cursor, None


def retained_blocks(item):
    path = Path(item['path'])
    with open_regular(Path(item['receipt'])) as source:
        raw = source.read(16385)
    validate(0 < len(raw) <= 16384 and hashlib.sha256(raw).hexdigest() == item['receipt_SHA256'],
             'RETAINED_RECEIPT_SHA')
    parsed = json.loads(raw)
    validate(parsed['request'] == item['request']
             and parsed.get('source_sha',parsed.get('range_SHA256')) == item['SHA256'], 'RETAINED_RECEIPT_BINDING')
    with open_regular(path) as source:
        orig = identity(os.fstat(source.fileno()))
        validate(orig == identity(path.lstat()) and orig['size'] == item['length'], 'RETAINED_FILE_IDENTITY')
        digest = hashlib.sha256()
        for data in blocks(source,item['length']):
            digest.update(data)
            yield data
        validate(not source.read(1) and digest.hexdigest() == item['SHA256'], 'RETAINED_TARGET_RANGE_SHA')
        validate(orig == identity(os.fstat(source.fileno())) == identity(path.lstat()), 'RETAINED_TARGET_CHANGED')


def target_blocks(plan, stream):
    for length, item in segments(plan):
        yield from retained_blocks(item) if item else blocks(stream,length)


def send(plan, binding):
    path = Path(plan['source_file'])
    output = sys.stdout.buffer
    with open_regular(path) as stream:
        original = identity(os.fstat(stream.fileno()))
        validate(original == plan['source_stat'] == identity(path.lstat()), 'SOURCE_FILE_IDENTITY')
        digest = hashlib.sha256()
        output.write(frame(binding))
        for length, retained in segments(plan):
            part = hashlib.sha256() if retained else None
            for data in blocks(stream,length):
                digest.update(data)
                if part is None:
                    validate(output.write(data) == len(data), 'SHORT_SOURCE_PIPE_WRITE')
                else:
                    part.update(data)
            if retained:
                validate(part.hexdigest() == retained['SHA256'], 'RETAINED_SOURCE_RANGE_SHA')
        validate(not stream.read(1) and digest.hexdigest() == plan['SHA256'], 'OFFICIAL_SOURCE_WHOLE_SHA')
        validate(original == identity(os.fstat(stream.fileno())) == identity(path.lstat()), 'SOURCE_CHANGED')
    output.flush()


def receive(plan, binding):
    root = Path(plan['target'])
    validate(root.parent.is_dir() and not root.parent.is_symlink(), 'TARGET_PARENT')
    validate(shutil.disk_usage(root.parent).free > plan['bytes'] + CHECKPOINT_SIZE, 'TARGET_DISK_CAPACITY')
    root.mkdir(mode=0o700)
    receipt(root / 'intent.json', binding)
    source = sys.stdin.buffer
    written = 0
    durable = 0
    digest = hashlib.sha256()
    published = False
    error = None
    try:
        validate(source.readline(4096) == frame(binding), 'EXACT_STREAM_FRAME')
        part = root / 'shard.incomplete'
        with part.open('xb') as output:
            try:
                for data in target_blocks(plan, source):
                    validate(output.write(data) == len(data), 'SHORT_TARGET_FILE_WRITE')
                    written += len(data)
                    digest.update(data)
                    if written - durable >= CHECKPOINT_SIZE:
                        output.flush()
                        os.fsync(output.fileno())
                        durable = written
                        receipt(root / f'prefix-{durable:010d}.json',
                                dict(binding, durable_bytes=durable, prefix_SHA256=digest.hexdigest()))
            except (EOFError, OSError, TimeoutError, ValueError):
                output.flush()
                os.fsync(output.fileno())
                if os.fstat(output.fileno()).st_size == written:
                    durable = written
                    receipt(root / 'retained-prefix.json',
                            dict(binding, durable_bytes=durable, prefix_SHA256=digest.hexdigest()))
                raise
            output.flush()
            os.fsync(output.fileno())
            durable = written
            validate(os.fstat(output.fileno()).st_size == plan['bytes']
                     and digest.hexdigest() == plan['SHA256'], 'OFFICIAL_TARGET_WHOLE_SHA')
            descriptor = identity(os.fstat(output.fileno()))
        validate(identity(part.lstat()) == descriptor, 'TARGET_FILE_REPLACED')
        validate(source.readline(len(COMMIT)) == COMMIT and not source.read(1), 'MISSING_SOURCE_COMMIT_OR_TRAILER')
        validate(identity(part.lstat()) == descriptor, 'TARGET_CHANGED_BEFORE_PUBLISH')
        final = root / plan['file']
        os.link(part, final, follow_symlinks=False)
        sync_directory(root)
        part.unlink()
        sync_directory(root)
        published = True
    except (EOFError, OSError, ValueError) as exc:
        error = type(exc).__name__
    result = dict(binding, role='receiver', passed=published, written_bytes=written, durable_bytes=durable,
                  whole_SHA256=digest.hexdigest() if published else None, error=error)
    receipt(root / 'terminal.json', result)
    print(json.dumps(result, allow_nan=False), flush=True)
    return int(not published)


def audit(plan, binding):
    root = Path(plan['target'])
    with open_regular(root / 'terminal.json') as f:
        raw = f.read(16385)
    validate(len(raw) <= 16384, 'TERMINAL_RECEIPT_BOUND')
    terminal = json.loads(raw)
    validate(all(terminal[k] == v for k, v in binding.items()) and terminal['passed']
             and terminal['written_bytes'] == terminal['durable_bytes'] == plan['bytes']
             and terminal['whole_SHA256'] == plan['SHA256'], 'TERMINAL_BINDING')
    path = root / plan['file']
    with open_regular(path) as source:
        descriptor = os.fstat(source.fileno())
        path_info = path.lstat()
        initial = identity(descriptor)
        validate(stat.S_ISREG(path_info.st_mode) and os.path.samestat(descriptor, path_info)
                 and initial['size'] == plan['bytes'], 'PUBLISHED_SIZE_AND_IDENTITY')
        digest = hashlib.file_digest(source, 'sha256').hexdigest()
        validate(initial == identity(os.fstat(source.fileno()))
                 and identity(path_info) == identity(path.lstat())
                 and digest == plan['SHA256'], 'PUBLISHED_WHOLE_SHA')
    print(json.dumps(dict(binding, verified=True, bytes=initial['size'], SHA256=digest,
                          original_terminal_SHA256=hashlib.sha256(raw).hexdigest()), allow_nan=False), flush=True)
    return 0


def endpoint(plan, binding, role, active_seconds):
    validate(plan['request'] == binding['request'] and plan['file'] == binding['file']
             and plan['SHA256'] == binding['SHA256'] and plan['bytes'] == binding['bytes']
             and Path(plan['file']).name == plan['file'], 'EXACT_BOUND_PLAN')

    def timeout(*_):
        raise TimeoutError('RELAY_DEADLINE')

    signal.signal(signal.SIGALRM, timeout)
    limit = min(active_seconds, binding['deadline'] - time.time()) if role != 'audit' else active_seconds
    validate(limit > 0, 'RELAY_EXPIRED')
    signal.setitimer(signal.ITIMER_REAL, limit)
    if role == 'sender':
        send(plan, binding)
        print(json.dumps(dict(binding, role='sender', passed=True), allow_nan=False), file=sys.stderr, flush=True)
        return 0
    return {'receiver': receive, 'audit': audit}[role](plan, binding)
