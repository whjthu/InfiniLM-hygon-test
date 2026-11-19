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


def bounded_receipt(path):
    with open_regular(path) as stream:
        raw = stream.read(16385)
    validate(0 < len(raw) <= 16384, 'RETAINED_RECEIPT_SIZE')
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def inspect_prior(plan):
    prior = plan['prior']
    root = Path(prior['target'])
    validate(root.is_dir() and not root.is_symlink(), 'PRIOR_DIRECTORY')
    intent, intent_sha = bounded_receipt(root / 'intent.json')
    terminal, terminal_sha = bounded_receipt(root / 'terminal.json')
    validate(intent_sha == prior['intent_SHA256'] and intent['request'] == prior['request']
             and intent['publication'] == prior['publication']
             and all(intent[k] == plan[k] for k in ('file', 'bytes', 'SHA256'))
             and all(terminal.get(k) == v for k, v in intent.items()), 'PRIOR_REQUEST_BINDING')
    validate(not os.path.lexists(root / plan['file']) and terminal['role'] == 'receiver'
             and terminal['passed'] is False and terminal['error'] in ('EOFError', 'TimeoutError', 'ConnectionResetError'),
             'PRIOR_RECEIVER_NOT_PARTIAL_TERMINAL')
    durable, written = terminal['durable_bytes'], terminal['written_bytes']
    validate(type(durable) is int and type(written) is int
             and 0 < durable <= written < plan['bytes'], 'PRIOR_PARTIAL_EXTENT')

    candidates = [(durable, root / 'retained-prefix.json')]
    candidates.extend((n, root / f'prefix-{n:010d}.json')
                      for n in range((durable // CHECKPOINT_SIZE) * CHECKPOINT_SIZE, 0, -CHECKPOINT_SIZE))
    for offset, path in candidates:
        if not os.path.lexists(path):
            continue
        prefix, prefix_receipt_sha = bounded_receipt(path)
        validate(type(offset) is int and 0 < offset <= durable
                 and all(prefix.get(k) == v for k, v in intent.items())
                 and prefix['durable_bytes'] == offset
                 and isinstance(prefix['prefix_SHA256'], str)
                 and len(prefix['prefix_SHA256']) == 64
                 and all(c in '0123456789abcdef' for c in prefix['prefix_SHA256']), 'EXACT_RETAINED_PREFIX_BINDING')
        break
    else:
        raise ValueError('NO_FSYNCED_PRIOR_PREFIX')
    partial = root / 'shard.incomplete'
    with open_regular(partial) as stream:
        before = os.fstat(stream.fileno())
        current = partial.lstat()
        original = identity(current)
        validate(stat.S_ISREG(current.st_mode) and os.path.samestat(before, current)
                 and original['size'] == written, 'PRIOR_PARTIAL_FILE_IDENTITY')
    proof = dict(request=prior['request'], intent_SHA256=intent_sha, terminal_SHA256=terminal_sha,
                 prefix_receipt_SHA256=prefix_receipt_sha, offset=offset, prefix_SHA256=prefix['prefix_SHA256'],
                 partial_identity=original, terminal_error=terminal['error'])
    return partial, proof


def use_target_prefix(plan, proof, digest, destination=None):
    path, current = inspect_prior(plan)
    validate(current == proof, 'PUBLISHED_TARGET_PREFIX_PROOF')
    with open_regular(path) as stream:
        first_fd = os.fstat(stream.fileno())
        first_path = path.lstat()
        validate(stat.S_ISREG(first_path.st_mode) and os.path.samestat(first_fd, first_path)
                 and identity(first_path) == proof['partial_identity'], 'TARGET_PREFIX_REPLACED')
        for data in blocks(stream, proof['offset']):
            digest.update(data)
            if destination is not None:
                validate(destination.write(data) == len(data), 'SHORT_PREFIX_COPY')
        validate(identity(first_fd) == identity(os.fstat(stream.fileno()))
                 and identity(first_path) == identity(path.lstat())
                 and digest.hexdigest() == proof['prefix_SHA256'], 'TARGET_PREFIX_VALUES')
    _, current = inspect_prior(plan)
    validate(current == proof, 'TARGET_PRIOR_CHANGED_DURING_READ')


def prepare(plan):
    _, proof = inspect_prior(plan)
    use_target_prefix(plan, proof, hashlib.sha256())
    print(json.dumps(proof, allow_nan=False), flush=True)
    return 0


def send(plan, binding):
    path = Path(plan['source_file'])
    output = sys.stdout.buffer
    with open_regular(path) as stream:
        first_fd = os.fstat(stream.fileno())
        first_path = path.lstat()
        validate(stat.S_ISREG(first_path.st_mode) and os.path.samestat(first_fd, first_path)
                 and identity(first_path) == plan['source_stat'], 'SOURCE_FILE_IDENTITY')
        digest = hashlib.sha256()
        resume = binding['resume']
        for data in blocks(stream, resume['offset']):
            digest.update(data)
        validate(digest.hexdigest() == resume['prefix_SHA256'], 'SOURCE_PREFIX_VALUES')
        banner = frame(binding)
        validate(output.write(banner) == len(banner), 'SHORT_SOURCE_FRAME')
        for data in blocks(stream, plan['bytes'] - resume['offset']):
            digest.update(data)
            validate(output.write(data) == len(data), 'SHORT_SOURCE_PIPE_WRITE')
        validate(not stream.read(1) and digest.hexdigest() == plan['SHA256'], 'OFFICIAL_SOURCE_WHOLE_SHA')
        validate(identity(first_fd) == identity(os.fstat(stream.fileno()))
                 and identity(first_path) == identity(path.lstat()), 'SOURCE_CHANGED')
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
                resume = binding['resume']
                use_target_prefix(plan, resume, digest, output)
                written = resume['offset']
                output.flush()
                os.fsync(output.fileno())
                durable = written
                receipt(root / 'reused-prefix.json',
                        dict(binding, durable_bytes=durable, prefix_SHA256=digest.hexdigest()))
                for data in blocks(source, plan['bytes'] - written):
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
            descriptor = os.fstat(output.fileno())
            file_at_path = part.lstat()
            file_identity = identity(file_at_path)
            validate(stat.S_ISREG(file_at_path.st_mode) and os.path.samestat(descriptor, file_at_path)
                     and descriptor.st_size == file_identity['size'], 'TARGET_FILE_REPLACED')
        validate(identity(part.lstat()) == file_identity, 'TARGET_CHANGED_AT_CLOSE')
        validate(source.readline(len(COMMIT)) == COMMIT and not source.read(1), 'MISSING_SOURCE_COMMIT_OR_TRAILER')
        validate(identity(part.lstat()) == file_identity, 'TARGET_CHANGED_BEFORE_PUBLISH')
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
    limit = min(active_seconds, binding['deadline'] - time.time()) if role not in ('audit', 'prepare') else active_seconds
    validate(limit > 0, 'RELAY_EXPIRED')
    signal.setitimer(signal.ITIMER_REAL, limit)
    if role == 'prepare':
        return prepare(plan)
    resume = binding['resume']
    validate(type(resume['offset']) is int and 0 < resume['offset'] < plan['bytes']
             and resume['request'] == plan['prior']['request'], 'STRICT_SUFFIX_EXTENT')
    if role == 'sender':
        send(plan, binding)
        print(json.dumps(dict(binding, role='sender', passed=True), allow_nan=False), file=sys.stderr, flush=True)
        return 0
    return {'receiver': receive, 'audit': audit}[role](plan, binding)
