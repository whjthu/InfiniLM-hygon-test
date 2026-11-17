"""Pure-stdlib bounded byte provision endpoints; never decode tensors or load models."""
import base64
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import resource
import signal
import socket
import stat
import struct
import sys
import time

H = lambda value: hashlib.sha256(value).hexdigest()


def require(test, message):
    if not test:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode('utf-8')+b'\n'


def stable(value):
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]


def no_links(path):
    path = Path(path)
    require(path.is_absolute(), 'absolute registered path required')
    for item in [*reversed(path.parents), path]:
        require(not item.is_symlink(), 'registered path contains symlink')
    return path


def member_name(value):
    path = PurePosixPath(value)
    require(not path.is_absolute() and '..' not in path.parts and '\\' not in value
            and value == path.as_posix() and value not in ('', '.'), 'unsafe output member')
    return path


def save(path, value, *, replace=False):
    target = path.with_name(path.name+'.tmp-'+str(os.getpid())) if replace else path
    with target.open('xb') as stream:
        stream.write(canonical(value)); stream.flush(); os.fsync(stream.fileno())
    if replace:
        os.replace(target, path)


def tick():
    require(time.time() < CONTEXT['deadline'] and time.monotonic() < MONO_END, 'original provision epoch exhausted')


def arm():
    global MONO_END
    require(CONTEXT['deadline'] == CONTEXT['epoch']+PLAN['whole_seconds']
            and PLAN['whole_seconds'] == 21600, 'one original six-hour epoch required')
    remaining = CONTEXT['deadline']-time.time()
    require(math.isfinite(remaining) and 0 < remaining <= 21600, 'original deadline invalid/expired')
    MONO_END = time.monotonic()+remaining
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('provision deadline')))
    signal.setitimer(signal.ITIMER_REAL, remaining)
    resource.setrlimit(resource.RLIMIT_AS, (PLAN['endpoint_AS_bytes'], PLAN['endpoint_AS_bytes']))
    cores = sorted(os.sched_getaffinity(0))
    require(len(cores) >= 2, 'two available CPU affinity cores required')
    os.sched_setaffinity(0, cores[:2])
    require(os.nice(10) == 10, 'endpoint actual nice must be10')
    return cores[:2]


def owned_root(mode, cores):
    root = no_links(PLAN['source_namespace'] if mode == 'source' else PLAN['target_namespace'])
    require(root.parent.is_dir(), 'registered namespace parent missing')
    root.mkdir(mode=0o700)  # Existing/partial namespace is never overwritten.
    fields = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
    identity = dict(CONTEXT, mode=mode, host=socket.gethostname(), pid=os.getpid(), startticks=int(fields[19]), uid=os.getuid(),
                    boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                    CPU_affinity=cores, nice=os.getpriority(os.PRIO_PROCESS, 0),
                    AS_limit_bytes=PLAN['endpoint_AS_bytes'], actual_GPU_cards=0,
                    expected_files=26, expected_payload_bytes=PLAN['stream_payload_bytes'])
    save(root/'intent.json', identity)
    return root, identity


def control(value):
    sys.stdout.buffer.write(canonical(value)); sys.stdout.buffer.flush()


def exact(stream, count):
    require(0 <= count <= PLAN['chunk_bytes'], 'single read exceeds bounded chunk')
    blocks, used = [], 0
    while used < count:
        tick()
        block = stream.read(count-used)
        if not block:
            raise EOFError('source stream truncated')
        blocks.append(block); used += len(block)
    return b''.join(blocks)


def header_parts(stream, row):
    prefix = exact(stream, 8)
    require(struct.unpack('<Q', prefix)[0] == row['header_bytes'], 'original shard header length differs')
    header = exact(stream, row['header_bytes'])
    require(H(header) == row['header_SHA256'] and
            len(set(json.loads(header))-{'__metadata__'}) == row['tensors'], 'original shard header differs')
    return prefix+header


def costs(started):
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return dict(endpoint_wall_seconds=time.monotonic()-started, actual_nice=os.getpriority(os.PRIO_PROCESS, 0),
                user_CPU_seconds=usage.ru_utime, system_CPU_seconds=usage.ru_stime,
                self_maxrss_bytes=int(usage.ru_maxrss)*1024, Linux_maxrss_original_unit='KiB',
                not_cgroup_RSS_limit=True, physical_IO_and_wire_bytes='UNKNOWN_NOT_MEASURED')


def validate_reference(raw):
    pin = PLAN['local_reference']
    require(len(raw) == pin['bytes'] and H(raw) == pin['SHA256'], 'bound reference bytes differ')
    value = json.loads(raw)
    require(value['schema'] == 'native_first_two_inputs_v1' and value['case'] == 'T03.sft'
            and value['world_size'] == 8 and value['accumulation'] == 4 and len(value['rows']) == 64,
            'reference case/denominator differs')


def endpoint(mode):
    started = time.monotonic()
    cores = arm()
    root, identity = owned_root(mode, cores)
    total, completed, active, offset = 0, [], None, 0
    last_bytes, last_time = 0, time.monotonic()

    def progress(digest, *, force=False):
        nonlocal last_bytes, last_time
        if force or total-last_bytes >= PLAN['progress_bytes'] or time.monotonic()-last_time >= PLAN['progress_seconds']:
            save(root/'progress.json', dict(identity, logical_payload_bytes=total, completed_files=len(completed),
                 active_member=active, member_offset=offset, member_prefix_SHA256=digest.hexdigest(),
                 observed_epoch=time.time(), replay_allowed=False), replace=True)
            last_bytes, last_time = total, time.monotonic()

    try:
        require(len(PLAN['files']) == len({row['name'] for row in PLAN['files']}) == 26
                and sum(row['bytes'] for row in PLAN['files']) == PLAN['stream_payload_bytes'], 'static file/byte budget differs')
        reference = base64.b64decode(REFERENCE_B64, validate=True) if mode == 'source' else None
        if mode == 'source':
            validate_reference(reference)
            for row in PLAN['files']:
                member_name(row['name'])
                if row['source_kind'] == 'A100_original_file':
                    path = no_links(row['source']); value = path.stat()
                    require(stat.S_ISREG(value.st_mode) and value.st_size == row['bytes'], 'source preflight stat differs')
        else:
            free = os.statvfs(root)
            require(free.f_bavail*free.f_frsize > PLAN['stream_payload_bytes']+64*1024**2,
                    'target filesystem available bytes insufficient; user quota remains unknown')
        control(dict(identity, stage='READY', user_quota_qualified=False))
        for ordinal, row in enumerate(PLAN['files']):
            tick(); active, offset = row['name'], 0
            member_name(active); digest = hashlib.sha256()
            if mode == 'source':
                if row['source_kind'] == 'local_bound_reference':
                    chunks = [reference]
                    stream, before = None, None
                else:
                    path = no_links(row['source'])
                    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                    stream = os.fdopen(fd, 'rb', buffering=0)
                    before = os.fstat(stream.fileno())
                    require(stat.S_ISREG(before.st_mode) and before.st_size == row['bytes'], 'source open stat differs')
                    chunks = [header_parts(stream, row)] if 'header_bytes' in row else []
                try:
                    while offset < row['bytes']:
                        block = chunks.pop(0) if chunks else exact(stream, min(PLAN['chunk_bytes'], row['bytes']-offset))
                        require(len(block) <= PLAN['chunk_bytes'], 'source chunk too large')
                        digest.update(block)
                        require(sys.stdout.buffer.write(block) == len(block), 'short source pipe write')
                        offset += len(block); total += len(block); progress(digest)
                    require(digest.hexdigest() == row['SHA256'], 'source whole file SHA differs')
                    if stream:
                        require(not stream.read(1) and stable(before) == stable(os.fstat(stream.fileno())) == stable(path.stat()),
                                'source stable FD identity changed')
                finally:
                    if stream:
                        stream.close()
            else:
                destination = root/active
                destination.parent.mkdir(parents=True, exist_ok=True)
                no_links(destination.parent)
                partial = destination.with_name(destination.name+'.incomplete')
                with partial.open('xb') as stream:
                    opened = os.fstat(stream.fileno())
                    first = header_parts(sys.stdin.buffer, row) if 'header_bytes' in row else None
                    while offset < row['bytes']:
                        block = first if first is not None else exact(sys.stdin.buffer, min(PLAN['chunk_bytes'], row['bytes']-offset))
                        first = None
                        digest.update(block)
                        require(stream.write(block) == len(block), 'short receiver file write')
                        offset += len(block); total += len(block); progress(digest)
                    require(digest.hexdigest() == row['SHA256'], 'receiver whole file SHA differs')
                    stream.flush(); os.fsync(stream.fileno())
                    closed = os.fstat(stream.fileno())
                    require(closed.st_size == row['bytes'] and (opened.st_dev, opened.st_ino) == (closed.st_dev, closed.st_ino)
                            and stable(closed) == stable(partial.stat()), 'written stable FD/size differs')
                require(not destination.exists() and not destination.is_symlink(), 'final member already exists')
                partial.rename(destination)
            item = dict(name=active, bytes=offset, SHA256=digest.hexdigest())
            if mode == 'receiver':
                item['target_stat'] = stable(destination.stat())
            elif before is not None:
                item['source_stat'] = stable(before)
            completed.append(item)
            progress(digest, force=True)
        require(total == PLAN['stream_payload_bytes'] and len(completed) == 26, 'incomplete whole provision')
        if mode == 'source':
            sys.stdout.buffer.flush()
        else:
            require(not sys.stdin.buffer.read(1), 'unexpected trailing payload')
            raw = (root/'reference/input-fingerprints.json').read_bytes()
            validate_reference(raw)
            folder = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            try: os.fsync(folder)
            finally: os.close(folder)
        outcome = dict(identity, passed=True, completed_files=26, logical_payload_bytes=total,
                       verified_members=completed, raw_byte_identity_only=True, full_model_files_verified=17,
                       whole_shards_verified=4, initial_files_verified=8, reference_verified=True,
                       tensor_decodes=0, Native_Torch_SDK_GPU_calls=0, model_numeric_qualified=False,
                       receiver_SHA_semantics='incoming stream plus successful full write/fsync/stableFD; no second body read',
                       post_transfer_disk_body_rehashed=False, GPU_GO=False, Model_GO=False,
                       replay_allowed=False, system_cost=costs(started))
    except BaseException as error:
        outcome = dict(identity, passed=False, error_type=type(error).__name__, completed_files=len(completed),
                       logical_payload_bytes=total, active_member=active, member_offset=offset,
                       verified_members=completed, tensor_decodes=0, Native_Torch_SDK_GPU_calls=0,
                       model_numeric_qualified=False, GPU_GO=False, Model_GO=False,
                       replay_allowed=False, system_cost=costs(started))
    outcome['finished_epoch'] = time.time()
    save(root/'outcome.json', outcome)
    if mode == 'receiver':
        control({key:value for key,value in outcome.items() if key != 'verified_members'})
    return 0 if outcome['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(endpoint(MODE))
