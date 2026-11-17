"""Read only the two declared immutable Swift source files; no provider imports."""
import ast
import base64
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import socket
import stat
import time


def need(ok, why):
    if not ok: raise ValueError(why)


def main(context):
    started = time.monotonic(); initial = context['deadline'] - time.time()
    need(0 < initial <= 60, 'original source RO epoch exhausted')
    def check():
        if min(context['deadline'] - time.time(), initial - (time.monotonic() - started)) <= 0:
            raise TimeoutError('original source RO deadline')
    def expired(*unused): raise TimeoutError('original source RO deadline')
    signal.signal(signal.SIGALRM, expired); signal.setitimer(signal.ITIMER_REAL, min(50, initial))
    allowed = sorted(os.sched_getaffinity(0) - {0, 1}); need(len(allowed) >= 2, 'two CPU cores excluding0/1 unavailable')
    os.sched_setaffinity(0, set(allowed[:2])); need(os.sched_getaffinity(0) == set(allowed[:2]), 'actual affinity differs')
    result = dict(request_id=context['request_id'], epoch=context['epoch'], deadline=context['deadline'],
                  publication=context['publication'], passed=False, files=[], logical_source_bytes=0,
                  planned_source_files=2, planned_source_bytes=43908, CPU_affinity=allowed[:2], GPU=0,
                  provider_import=False, native_template_run=False, source_read_only=True,
                  host=socket.gethostname(), UID=os.getuid(), PID=os.getpid(),
                  boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                  startticks=int(Path('/proc/self/stat').read_text().split(') ', 1)[1].split()[19]))
    try:
        need(len(context['files']) == 2 and sum(p['bytes'] for p in context['files']) == 43908, 'exact two source pins required')
        for pin in context['files']:
            check(); p = Path(pin['canonical_path']); resolved = p.resolve(strict=True)
            need(p.is_absolute() and '..' not in p.parts and resolved == p and not p.is_symlink(), 'canonical source path differs')
            fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW)
            try:
                before = os.fstat(fd)
                need(stat.S_ISREG(before.st_mode) and before.st_uid in (0, os.getuid()) and before.st_size == pin['bytes'], 'source type/owner/size differs')
                with os.fdopen(fd, 'rb', buffering=0, closefd=False) as stream: raw = stream.read(pin['bytes'] + 1)
                after = os.fstat(fd); current = p.stat()
                key = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
                need(key(before) == key(after) == key(current) and p.resolve(strict=True) == resolved,
                     'source identity changed during read')
                need(len(raw) == pin['bytes'] and hashlib.sha256(raw).hexdigest() == pin['SHA256'], 'source whole SHA differs')
                ast.parse(raw)
                result['files'].append(dict(name=pin['name'], canonical_path=str(p), bytes=len(raw),
                    SHA256=pin['SHA256'], stat_identity=list(key(before)), source_base64=base64.b64encode(raw).decode()))
                result['logical_source_bytes'] += len(raw)
            finally: os.close(fd)
        check(); result['passed'] = True
    except BaseException as error:
        result.update(error_type=type(error).__name__, error_reason=str(error)[:160])
    result.update(finished=time.time(), endpoint_wall_seconds=time.monotonic() - started,
                  self_user_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime,
                  self_system_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_stime,
                  peak_self_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    raw = (json.dumps(result, sort_keys=True) + '\n').encode()
    need(len(raw) <= 96 * 1024, 'bounded source response cap exceeded')
    print(raw.decode(), end='', flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main(READ_CONTEXT))
