"""Inject PLAN, CONTEXT and MODE; read original small records and known PID only."""
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import stat
import time


def process_state(identity, observer_host, observer_boot, current):
    if identity['host'] != observer_host:
        return None
    if identity['boot_id'] != observer_boot or current is None:
        return False
    return current['startticks'] == identity['startticks'] and current['state'] != 'Z'


def main():
    remaining = READ_DEADLINE-time.time()
    if not 0 < remaining <= 90:
        raise ValueError('readonly epoch expired')
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('readonly deadline')))
    signal.setitimer(signal.ITIMER_REAL, remaining)
    base = Path(PLAN['source_namespace'] if MODE == 'source' else PLAN['target_namespace'])
    root = base/'phases'/('%02d' % CONTEXT['phase'])
    result = dict(context=CONTEXT, mode=MODE, namespace_exists=root.exists(), records={},
                  execution_state='UNKNOWN', payload_reads=0, remote_writes=0, replay_allowed=False)
    for item in [*reversed(root.parents), root]:
        if item.is_symlink():
            raise ValueError('original namespace symlink')
    if not root.exists():
        return result  # Absence alone never establishes not executed.
    if root.stat().st_uid != os.getuid():
        raise ValueError('namespace owner differs')
    for name in ('intent', 'progress', 'outcome', 'source-validation'):
        path = root/(name+'.json')
        if not path.exists():
            continue
        if path.is_symlink() or not stat.S_ISREG(path.stat().st_mode) or path.stat().st_size > 65536:
            raise ValueError('small receipt bound')
        with path.open('rb', buffering=0) as stream:
            before = os.fstat(stream.fileno()); raw = stream.read(65537); after = os.fstat(stream.fileno())
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) or len(raw) != before.st_size:
            raise ValueError('receipt unstable')
        result['records'][name] = dict(bytes=len(raw), SHA256=hashlib.sha256(raw).hexdigest(), value=json.loads(raw))
    intent = result['records'].get('intent', {}).get('value')
    if not intent:
        return result
    if any(intent.get(k) != v for k, v in CONTEXT.items()) or intent['mode'] != MODE:
        raise ValueError('exact original nonce/epoch differs')
    if not (type(intent['pid']) is int and intent['pid'] > 0 and type(intent['startticks']) is int
            and intent['startticks'] > 0 and intent['host'] and intent['uid'] == os.getuid()):
        raise ValueError('original endpoint owner/PID identity invalid')
    host, boot = socket.gethostname(), Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    current = None
    if host == intent['host'] and boot == intent['boot_id']:
        path = Path('/proc')/str(intent['pid'])/'stat'
        try:
            fields = path.read_text().rsplit(')', 1)[1].split()
            current = dict(state=fields[0], startticks=int(fields[19]))
        except FileNotFoundError:
            pass
    live = process_state(intent, host, boot, current)
    result.update(observer_host=host, original_host=intent['host'], known_process_live=live)
    value = result['records'].get('outcome', {}).get('value')
    if value:
        if any(value.get(k) != v for k, v in CONTEXT.items()) or any(value.get(k) != intent[k] for k in ('host', 'boot_id', 'pid', 'startticks', 'uid', 'mode')):
            raise ValueError('outcome identity differs')
        if live is False:
            result['execution_state'] = 'TERMINAL_PASS' if value['passed'] else 'TERMINAL_FAIL'
    return result


if __name__ == '__main__':
    print(json.dumps(main(), sort_keys=True, allow_nan=False))
