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


def send(plan, binding):
    path = Path(plan['source_file'])
    output = sys.stdout.buffer
    with open_regular(path) as stream:
        original = identity(os.fstat(stream.fileno()))
        validate(original == plan['source_stat'] == identity(path.lstat()), 'SOURCE_FILE_IDENTITY')
        digest = hashlib.sha256()
        output.write(frame(binding))
        for data in blocks(stream, plan['bytes']):
            digest.update(data)
            validate(output.write(data) == len(data), 'SHORT_SOURCE_PIPE_WRITE')
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
                for data in blocks(source, plan['bytes']):
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
