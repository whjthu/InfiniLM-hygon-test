"""Finite pure-stdlib endpoints. Inject PLAN, CONTEXT, MODE; stdin remains byte stream."""
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import socket
import stat
import sys
import time


def require(value, reason):
    if not value:
        raise ValueError(reason)


def encode(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def safe(path):
    path = Path(path)
    require(path.is_absolute(), 'absolute path required')
    for item in [*reversed(path.parents), path]:
        require(not item.is_symlink(), 'path contains symlink')
    return path


def stable(st):
    return [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]


def save(path, value, replace=False):
    target = path.with_name(path.name+'.tmp') if replace else path
    with target.open('xb') as stream:
        stream.write(encode(value)); stream.flush(); os.fsync(stream.fileno())
    if replace:
        os.replace(target, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def read_json(path, limit=65536):
    path = safe(path)
    before = path.stat()
    require(stat.S_ISREG(before.st_mode) and before.st_uid == os.getuid()
            and 0 < before.st_size <= limit, 'small receipt bound/owner differs')
    with path.open('rb', buffering=0) as stream:
        raw = stream.read(limit+1)
        require(stable(before) == stable(os.fstat(stream.fileno())) == stable(path.stat()), 'receipt changed')
    require(len(raw) == before.st_size, 'receipt truncated')
    return json.loads(raw)


def validate_phase(plan, index):
    require(type(index) is int and 0 <= index < len(plan['phases']), 'unknown phase')
    phase = plan['phases'][index]
    require(phase['id'] == index and 0 < phase['payload_bytes'] <= 1073741824, 'phase budget differs')
    require(sum(row['bytes'] for row in phase['segments']) == phase['payload_bytes'], 'segment budget differs')
    seen = set()
    for row in phase['segments']:
        name = row['name']
        require(name in plan['files'] and '/' not in name and '\\' not in name
                and name not in ('.', '..') and name not in seen, 'foreign/duplicate member')
        require(type(row['offset']) is int and type(row['bytes']) is int and row['offset'] >= 0
                and row['bytes'] > 0 and row['offset']+row['bytes'] <= plan['files'][name]['bytes'], 'invalid range')
        seen.add(name)
    return phase


def cross_receipt(source, identity, segments):
    require(source['passed'] is True and source['mode'] == 'source', 'source did not validate')
    for key in ('request_id', 'publication', 'source_manifest_SHA256', 'asset_plan_SHA256', 'endpoint_SHA256', 'phase', 'epoch', 'deadline'):
        require(source[key] == identity[key], 'source/receiver context differs')
    require(source['segments'] == segments and source['host'] and source['pid'] > 0
            and source['startticks'] > 0 and source['uid'] == 1001, 'source range identity differs')


def tick():
    require(time.time() < CONTEXT['deadline']-20 and time.monotonic() < MONO_END, 'original phase epoch exhausted')


def exact(stream, count):
    require(0 < count <= PLAN['bridge_block_bytes'], 'single buffer exceeds bound')
    result = bytearray()
    while len(result) < count:
        tick()
        block = stream.read(count-len(result))
        if not block:
            raise EOFError('payload truncated')
        result.extend(block)
    return result


def hard_exit(*_):
    os._exit(124)  # Only this endpoint; no children, process-group or remote guesses.


def work_alarm(*_):
    # The first alarm stops work. A second alarm is the original hard deadline,
    # including blocked receipt writes/stdout in the FAIL tail; never reset epoch.
    signal.signal(signal.SIGALRM, hard_exit)
    remaining = min(CONTEXT['deadline']-time.time(), HARD_MONO_END-time.monotonic())
    if remaining <= 0:
        hard_exit()
    signal.setitimer(signal.ITIMER_REAL, remaining)
    raise TimeoutError('original phase work deadline; bounded outcome tail')


def candidate_inventory():
    rows = []
    for base in PLAN['candidate_model_roots']:
        root = safe(base)
        for name in PLAN['files']:
            path = safe(root/name)
            row = dict(path=path.as_posix(), exists=path.exists(), identity='UNKNOWN_STAT_ONLY')
            if row['exists']:
                st = path.stat()
                require(stat.S_ISREG(st.st_mode), 'candidate not regular')
                row.update(bytes=st.st_size, stat=stable(st))
            rows.append(row)
    return rows


def main():
    global MONO_END, HARD_MONO_END
    started = time.monotonic()
    phase = validate_phase(PLAN, CONTEXT['phase'])
    seconds = PLAN['phase0_epoch_seconds'] if CONTEXT['phase'] == 0 else PLAN['epoch_seconds']
    require(CONTEXT['deadline'] == CONTEXT['epoch']+seconds, 'epoch reset/extension')
    remaining = CONTEXT['deadline']-time.time()-20
    require(math.isfinite(remaining) and 0 < remaining <= seconds, 'expired epoch')
    MONO_END = time.monotonic()+remaining
    HARD_MONO_END = MONO_END+20
    signal.signal(signal.SIGALRM, work_alarm)
    signal.setitimer(signal.ITIMER_REAL, remaining)
    resource.setrlimit(resource.RLIMIT_AS, (PLAN['AS_bytes'], PLAN['AS_bytes']))
    cores = sorted(set(os.sched_getaffinity(0))-{0, 1})
    require(len(cores) >= 2, 'two cores excluding0/1 required')
    os.sched_setaffinity(0, cores[:2]); os.nice(10)
    require(os.getpriority(os.PRIO_PROCESS, 0) == 10, 'nice differs')
    require(os.getuid() == {'source': 1001, 'receiver': 154459}[MODE], 'endpoint UID differs')
    base = safe(PLAN['source_namespace'] if MODE == 'source' else PLAN['target_namespace'])
    if CONTEXT['phase'] == 0:
        candidates = candidate_inventory() if MODE == 'receiver' else []
        # Existing candidate weight bodies are unqualified: stop for exact-path reconciliation.
        if any(row['exists'] and row['path'].endswith('.safetensors') for row in candidates):
            sys.stdout.buffer.write(encode(dict(CONTEXT, mode=MODE, stage='PRE_NAMESPACE_HOLD',
                candidate_inventory=candidates, execution_state='UNKNOWN_UNQUALIFIED_CANDIDATE', replay_allowed=False)))
            sys.stdout.buffer.flush()
            raise ValueError('EXACT_CANDIDATE_WEIGHT_PRESENT_UNQUALIFIED_HOLD')
        base.mkdir(mode=0o700)
        (base/'phases').mkdir(mode=0o700)
        save(base/'asset-intent.json', dict(case=PLAN['case'], asset_plan_SHA256=CONTEXT['asset_plan_SHA256'],
                                          endpoint_SHA256=CONTEXT['endpoint_SHA256'],
                                          candidate_inventory=candidates, complete_weights_qualified=False))
    else:
        require(base.is_dir() and base.stat().st_uid == os.getuid(), 'owned asset namespace missing')
        anchor = read_json(base/'asset-intent.json')
        require(anchor['case'] == 'T04.sft' and anchor['asset_plan_SHA256'] == CONTEXT['asset_plan_SHA256']
                and anchor['endpoint_SHA256'] == CONTEXT['endpoint_SHA256'],
                'asset source changed')
        prior = read_json(base/'phases'/('%02d' % (CONTEXT['phase']-1))/'outcome.json')
        require(prior['passed'] is True and prior['mode'] == MODE, 'prior phase not terminal PASS; reconcile exact request')
    root = safe(base/'phases'/('%02d' % CONTEXT['phase']))
    root.mkdir(mode=0o700)  # fixed phase, any nonce/partial already present blocks re-execution
    fields = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
    identity = dict(CONTEXT, mode=MODE, host=socket.gethostname(), uid=os.getuid(), pid=os.getpid(),
                    startticks=int(fields[19]), boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                    CPU_affinity=cores[:2], AS_bytes=PLAN['AS_bytes'], nice=10, actual_GPU_cards=0)
    save(root/'intent.json', identity)
    completed, total, partials = [], 0, []
    try:
        if MODE == 'receiver':
            fs = os.statvfs(root)
            require(fs.f_bavail*fs.f_frsize > phase['payload_bytes']+(64<<20), 'filesystem free bytes insufficient; quota unknown')
        sys.stdout.buffer.write(encode(dict(identity, stage='READY', payload_bytes=phase['payload_bytes'])))
        sys.stdout.buffer.flush()
        for ordinal, row in enumerate(phase['segments']):
            tick(); pin = PLAN['files'][row['name']]; digest = hashlib.sha256(); count = 0
            path = safe(Path(PLAN['source_model_root'])/row['name']) if MODE == 'source' else root/('%02d.partial' % ordinal)
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK) if MODE == 'source' else os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, 'rb' if MODE == 'source' else 'wb', buffering=0) as stream:
                before = os.fstat(stream.fileno())
                require(stat.S_ISREG(before.st_mode), 'file not regular')
                if MODE == 'source':
                    require(before.st_size == pin['bytes'], 'source size differs')
                    if row['name'].endswith('.safetensors'):
                        require([before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns]
                                == [pin['device'], pin['inode'], pin['bytes'], pin['mtime_ns']], 'original wholeSHA stat reuse invalid')
                    stream.seek(row['offset'])
                while count < row['bytes']:
                    block = exact(stream if MODE == 'source' else sys.stdin.buffer, min(PLAN['bridge_block_bytes'], row['bytes']-count))
                    digest.update(block)
                    destination = sys.stdout.buffer if MODE == 'source' else stream
                    require(destination.write(block) == len(block), 'short payload write')
                    count += len(block); total += len(block)
                    if count == row['bytes'] or count % (16<<20) == 0:
                        save(root/'progress.json', dict(identity, logical_bytes=total, active_member=row['name'],
                             active_range_bytes=count, completed_segments=len(completed), observed_epoch=time.time()), replace=True)
                if MODE == 'source':
                    require(stable(before) == stable(os.fstat(stream.fileno())) == stable(path.stat()), 'source stable FD changed')
                else:
                    stream.flush(); os.fsync(stream.fileno())
                    require(os.fstat(stream.fileno()).st_size == row['bytes'], 'target size differs')
                if row['offset'] == 0 and row['bytes'] == pin['bytes']:
                    require(digest.hexdigest() == pin['SHA256'], 'complete metadata SHA differs')
            item = dict(row, SHA256=digest.hexdigest())
            completed.append(item)
            if MODE == 'receiver':
                partials.append(path)
        if MODE == 'source':
            sys.stdout.buffer.flush()
        else:
            raw = sys.stdin.buffer.readline(65537)
            require(0 < len(raw) <= 65536 and raw.endswith(b'\n'), 'source receipt truncated/oversized')
            cross_receipt(json.loads(raw), identity, completed)
            require(not sys.stdin.buffer.read(1), 'trailing payload')
            for ordinal, path in enumerate(partials):
                os.link(path, root/('%02d.chunk' % ordinal), follow_symlinks=False)
                path.unlink()
            save(root/'source-validation.json', json.loads(raw))
        usage = resource.getrusage(resource.RUSAGE_SELF)
        result = dict(identity, passed=True, segments=completed, logical_payload_bytes=total,
                      full_asset_qualified=False, tensor_decodes=0, GPU_GO=False, finished_epoch=time.time(),
                      system_cost=dict(wall_seconds=time.monotonic()-started, CPU_user=usage.ru_utime,
                      CPU_system=usage.ru_stime, self_maxrss_bytes=usage.ru_maxrss*1024,
                      physical_IO_wire_bytes='UNKNOWN', quota_qualified=False))
    except BaseException as error:
        result = dict(identity, passed=False, segments=completed, logical_payload_bytes=total,
                      error_type=type(error).__name__, error_reason=str(error)[:180], finished_epoch=time.time(),
                      full_asset_qualified=False, GPU_GO=False, replay_allowed=False)
    save(root/'outcome.json', result)
    sys.stdout.buffer.write(encode(result)); sys.stdout.buffer.flush()
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
