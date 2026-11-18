"""One original-login PID identity observation, without receipts or weight access."""
import hashlib
import json
import math
import os
import signal
import socket
import time

ANCHOR = dict(request_id='fcafa5f620f6415b86e5a525f3fdcf0c', host='zz-login02',
              boot_id='8975b5df-9ebb-4734-8ef4-38e8f118e711', UID=154459,
              PID=1369647, startticks=840030493)


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def read_fd(fd, cap):
    raw = os.read(fd, cap + 1)
    need(len(raw) <= cap, 'BOUNDED_KERNEL_RECORD')
    return raw


def boot():
    fd = os.open('/proc/sys/kernel/random/boot_id', os.O_RDONLY | os.O_NOFOLLOW)
    try:
        return read_fd(fd, 128).decode('ascii').strip()
    finally:
        os.close(fd)


def proc_stat(fd):
    member = os.open('stat', os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
    try:
        raw = read_fd(member, 8192)
    finally:
        os.close(member)
    parts = raw.decode('ascii').rsplit(') ', 1)[1].split()
    need(len(parts) >= 20, 'STAT_SCHEMA')
    return dict(PID=int(raw.split(b' ', 1)[0]), startticks=int(parts[19]),
                state=parts[0], record_bytes=len(raw), SHA256=hashlib.sha256(raw).hexdigest())


def observe(context, claim):
    need(claim['READ_GO'] is True and claim['max_invocations'] == 1, 'ONE_ADOPTED_READ')
    epoch = context['epoch']
    need(type(epoch) in (int, float) and math.isfinite(epoch)
         and 0 <= time.time() - epoch < 20, 'ORIGINAL30_SECOND_EPOCH')
    signal.signal(signal.SIGALRM, lambda *_: os._exit(124))
    signal.setitimer(signal.ITIMER_REAL, epoch + 25 - time.time())
    cpus = sorted(os.sched_getaffinity(0) - {0, 1})
    need(len(cpus) >= 2, 'TWO_READ_CPUS')
    os.sched_setaffinity(0, cpus[:2])
    host, boot_id = socket.gethostname(), boot()
    result = dict(context=context, expected=ANCHOR, observed_host=host,
                  observed_boot_id=boot_id, observed_UID=os.getuid(),
                  state='UNKNOWN_HOST_BOOT_OR_UID', original_process_terminal=None,
                  PID_queried=False, receipts_read=0, weights_read=0,
                  other_processes_queried=0, new_SSH_hops=0, GPU_calls=0,
                  transfer_or_model_replay=False, retries=0)
    fd = None
    try:
        if (host, boot_id, os.getuid()) != (ANCHOR['host'], ANCHOR['boot_id'], ANCHOR['UID']):
            return result
        result['PID_queried'] = True
        try:
            fd = os.open('/proc/' + str(ANCHOR['PID']),
                         os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        except (FileNotFoundError, ProcessLookupError):
            result.update(state='ORIGINAL_PID_ABSENT', original_process_terminal=True)
            return result
        before = os.fstat(fd)
        result['proc_directory_UID'] = before.st_uid
        if before.st_uid != ANCHOR['UID']:
            result['state'] = 'UNKNOWN_PID_DIRECTORY_UID'
            return result
        first, last = proc_stat(fd), proc_stat(fd)
        after = os.fstat(fd)
        need(first['PID'] == last['PID'] == ANCHOR['PID'], 'EXACT_PID')
        need(first['startticks'] == last['startticks'], 'PID_REUSE_DURING_READ')
        need((before.st_dev, before.st_ino, before.st_uid) ==
             (after.st_dev, after.st_ino, after.st_uid), 'PROC_DIRECTORY_CHANGED')
        result['stat_observation'] = last
        if last['startticks'] != ANCHOR['startticks']:
            result.update(state='ORIGINAL_PID_REUSED', original_process_terminal=True)
        elif last['state'] in ('Z', 'X'):
            result.update(state='ORIGINAL_PID_TERMINAL', original_process_terminal=True)
        else:
            result.update(state='ORIGINAL_PID_LIVE', original_process_terminal=False)
        return result
    except (OSError, ValueError, IndexError, UnicodeError) as error:
        result.update(state='UNKNOWN_IDENTITY_OBSERVATION', original_process_terminal=None,
                      error_type=type(error).__name__, errno=getattr(error, 'errno', None),
                      reason_SHA256=hashlib.sha256(str(error).encode()).hexdigest())
        return result
    finally:
        if fd is not None:
            os.close(fd)
        # Even an absence observation is usable only on the same original host/boot.
        if socket.gethostname() != host or boot() != boot_id:
            result.update(state='UNKNOWN_HOST_BOOT_CHANGED', original_process_terminal=None)
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == '__main__':
    print(json.dumps(observe(CONTEXT, READ_CLAIM), sort_keys=True))
