"""Separately gated CPU-only final body verification; one original basename per epoch.

Inject PLAN, CLAIM, CONTEXT and MEMBER; concatenate durable chunks into a new file.
This is NOT a <=1GiB-transfer phase: network0, declared disk read/write=file bytes.
"""
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import socket
import stat
import time


def require(value, reason):
    if not value:
        raise ValueError(reason)


def safe(path):
    path = Path(path)
    require(path.is_absolute(), 'absolute path required')
    for item in [*reversed(path.parents), path]:
        require(not item.is_symlink(), 'symlink refused')
    return path


def stable(st):
    return [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]


def save(path, value):
    with path.open('x', encoding='utf8') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def small(path):
    path = safe(path)
    require(path.is_file() and path.stat().st_uid == os.getuid() and path.stat().st_size <= 65536, 'receipt bound/owner')
    with path.open('rb', buffering=0) as stream:
        before = os.fstat(stream.fileno()); raw = stream.read(65537)
        require(stable(before) == stable(os.fstat(stream.fileno())) == stable(path.stat()), 'receipt changed')
    return json.loads(raw)


def main():
    require(CLAIM['root_GO'] is True and CLAIM['finalize_GO'] is True, 'whole-body CPU scope not approved')
    require(MEMBER in PLAN['files'] and CONTEXT['deadline'] == CONTEXT['epoch']+7200, 'member/epoch differs')
    remaining = CONTEXT['deadline']-time.time()-20
    require(0 < remaining <= 7200, 'expired final verification epoch')
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('final original epoch')))
    signal.setitimer(signal.ITIMER_REAL, remaining)
    resource.setrlimit(resource.RLIMIT_AS, (256<<20, 256<<20))
    cores = sorted(set(os.sched_getaffinity(0))-{0, 1}); require(len(cores) >= 2, 'two CPU cores required')
    os.sched_setaffinity(0, cores[:2]); os.nice(10)
    require(os.getuid() == 154459 and os.getpriority(os.PRIO_PROCESS, 0) == 10, 'endpoint owner/nice differs')
    base = safe(PLAN['target_namespace'])
    anchor = small(base/'asset-intent.json')
    require(anchor['asset_plan_SHA256'] == CONTEXT['asset_plan_SHA256']
            and anchor['endpoint_SHA256'] == CONTEXT['endpoint_SHA256'], 'asset plan/endpoint differs')
    verifies = base/'whole-verifications'; verifies.mkdir(mode=0o700, exist_ok=True)
    root = safe(verifies/MEMBER); root.mkdir(mode=0o700)
    fields = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
    identity = dict(CONTEXT, member=MEMBER, host=socket.gethostname(), pid=os.getpid(), startticks=int(fields[19]),
        uid=os.getuid(), boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), CPU_affinity=cores[:2], actual_GPU_cards=0)
    save(root/'intent.json', identity)
    model = safe(base/'model'); model.mkdir(mode=0o700, exist_ok=True)
    require(model.stat().st_uid == os.getuid(), 'model directory owner differs')
    final = safe(model/MEMBER); partial = root/'file.incomplete'
    pin, count, digest, used = PLAN['files'][MEMBER], 0, hashlib.sha256(), []
    try:
        with partial.open('xb', buffering=0) as out:
            for phase in PLAN['phases']:
                phase_root = safe(base/'phases'/('%02d' % phase['id']))
                receipt = small(phase_root/'outcome.json')
                require(receipt['passed'] is True and receipt['mode'] == 'receiver' and receipt['phase'] == phase['id']
                        and receipt['asset_plan_SHA256'] == CONTEXT['asset_plan_SHA256']
                        and receipt['endpoint_SHA256'] == CONTEXT['endpoint_SHA256'], 'phase receipt not qualified')
                source = small(phase_root/'source-validation.json')
                require(source['passed'] is True and source['segments'] == receipt['segments']
                        and all(source[k] == receipt[k] for k in ('request_id', 'phase', 'epoch', 'deadline', 'publication', 'source_manifest_SHA256', 'asset_plan_SHA256', 'endpoint_SHA256')),
                        'source/receiver phase cross-binding differs')
                for ordinal, row in enumerate(phase['segments']):
                    if row['name'] != MEMBER:
                        continue
                    require(row['offset'] == count and receipt['segments'][ordinal]['name'] == MEMBER
                            and receipt['segments'][ordinal]['offset'] == count and receipt['segments'][ordinal]['bytes'] == row['bytes'], 'range gap/receipt differs')
                    path = safe(phase_root/('%02d.chunk' % ordinal)); chunk_digest = hashlib.sha256(); got = 0
                    with path.open('rb', buffering=0) as stream:
                        before = os.fstat(stream.fileno())
                        require(stat.S_ISREG(before.st_mode) and before.st_uid == os.getuid() and before.st_size == row['bytes'], 'chunk owner/size differs')
                        while got < row['bytes']:
                            require(time.time() < CONTEXT['deadline']-20, 'final epoch exhausted')
                            block = stream.read(min(1<<20, row['bytes']-got)); require(block, 'chunk truncated')
                            digest.update(block); chunk_digest.update(block)
                            require(out.write(block) == len(block), 'assembly short write')
                            count += len(block); got += len(block)
                        require(stable(before) == stable(os.fstat(stream.fileno())) == stable(path.stat()), 'chunk changed')
                    require(chunk_digest.hexdigest() == receipt['segments'][ordinal]['SHA256'], 'durable range SHA differs')
                    used.append(dict(phase=phase['id'], range_bytes=got, SHA256=chunk_digest.hexdigest()))
            require(count == pin['bytes'] and digest.hexdigest() == pin['SHA256'], 'FINAL_WHOLE_FILE_SHA_DIFFERENT')
            os.fsync(out.fileno())
            require(os.fstat(out.fileno()).st_size == count, 'assembled size differs')
        os.link(partial, final, follow_symlinks=False)  # EEXIST, never replace a final or a partial namespace
        partial.unlink()
        fd = os.open(model, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        result = dict(identity, passed=True, bytes=count, SHA256=digest.hexdigest(), ranges=used,
                      assembled_stat=stable(final.stat()), whole_file_qualified=True, full_asset_qualified=False,
                      disk_payload_bytes_read=count, disk_payload_bytes_written=count, network_payload_bytes=0,
                      tensor_decodes=0, GPU_GO=False, finished_epoch=time.time())
    except BaseException as error:
        result = dict(identity, passed=False, error_type=type(error).__name__, error_reason=str(error)[:180],
                      bytes=count, whole_file_qualified=False, GPU_GO=False, replay_allowed=False, finished_epoch=time.time())
    save(root/'outcome.json', result)
    print(json.dumps(result, allow_nan=False))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
