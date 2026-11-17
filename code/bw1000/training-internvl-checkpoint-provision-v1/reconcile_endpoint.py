"""Read only the original namespace receipts, known PID and target file stats."""
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import stat
import time


def require(value, message):
    if not value:
        raise ValueError(message)


def safe(path):
    require(path.is_absolute(), 'absolute registered path required')
    for item in [*reversed(path.parents), path]:
        require(not item.is_symlink(), 'registered path contains link')
    return path


def read_small(path, limit=65536):
    safe(path)
    if not path.exists():
        return None
    before = path.stat()
    require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= limit, 'receipt bound differs')
    with path.open('rb', buffering=0) as stream:
        raw = stream.read(limit+1)
        after = os.fstat(stream.fileno())
    require(len(raw) == before.st_size and (before.st_ino, before.st_dev, before.st_size, before.st_mtime_ns)
            == (after.st_ino, after.st_dev, after.st_size, after.st_mtime_ns), 'receipt changed during read')
    return dict(bytes=len(raw), SHA256=hashlib.sha256(raw).hexdigest(), value=json.loads(raw))


def process_observation(identity, observer_host, boot, status):
    same_host = observer_host == identity['host']
    alive = None if not same_host else False
    if same_host and boot == identity['boot_id'] and status.exists():
        fields = status.read_text().rsplit(')', 1)[1].split()
        alive = int(fields[19]) == identity['startticks'] and fields[0] != 'Z'
    return dict(pid=identity['pid'], startticks=identity['startticks'], same_boot=boot == identity['boot_id'],
                observer_host=observer_host, original_host=identity['host'], same_host=same_host,
                original_process_live=alive, no_process_scan=True)


def main():
    remaining = READ_DEADLINE-time.time()
    require(0 < remaining <= 90, 'readonly reconciliation deadline expired')
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('readonly deadline')))
    signal.setitimer(signal.ITIMER_REAL, remaining)
    root = safe(Path(PLAN['source_namespace'] if MODE == 'source' else PLAN['target_namespace']))
    result = dict(context=CONTEXT, mode=MODE, namespace_exists=root.exists(), observed_epoch=time.time(),
                  source_payload_reads=0, target_payload_rehashes=0, remote_writes=0, replay_allowed=False,
                  execution_state='UNKNOWN', busy=None)
    if not root.exists():
        return result
    require(root.is_dir() and root.stat().st_uid == os.getuid(), 'original namespace owner/type differs')
    records = {name:read_small(root/(name+'.json')) for name in ('intent', 'progress', 'outcome')}
    result['records'] = records
    intent = records['intent']
    if not intent:
        return result
    identity = intent['value']
    require(all(identity.get(key) == value for key,value in CONTEXT.items()) and identity['mode'] == MODE,
            'original request/epoch/namespace context differs')
    require(identity['expected_files'] == 26 and identity['expected_payload_bytes'] == PLAN['stream_payload_bytes'],
            'original denominator differs')
    require(len(identity['CPU_affinity']) == 2 and identity['nice'] == 10
            and identity['AS_limit_bytes'] == 536870912 and identity['actual_GPU_cards'] == 0,
            'original CPU-only resource receipt differs')
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    observer_host = socket.gethostname()
    require(isinstance(identity['host'], str) and identity['host'], 'original host required')
    require(identity['uid'] == os.getuid(), 'original endpoint UID differs')
    pid = identity['pid']
    require(type(pid) is int and pid > 0 and type(identity['startticks']) is int, 'known original PID required')
    status = Path('/proc')/str(pid)/'stat'
    observation = process_observation(identity, observer_host, boot, status)
    result['known_process'] = observation
    outcome = records['outcome']
    if outcome:
        value = outcome['value']
        require(all(value.get(key) == expected for key,expected in CONTEXT.items()), 'outcome context differs')
        require(all(value.get(key) == identity[key] for key in ('mode','host','pid','startticks','uid','boot_id')),
                'outcome original endpoint identity differs')
        expected = {row['name']:row for row in PLAN['files']}
        members = value['verified_members']
        require(len(members) == value['completed_files'] and len({row['name'] for row in members}) == len(members),
                'completed member denominator differs')
        for row in members:
            pin = expected[row['name']]
            require(row['bytes'] == pin['bytes'] and row['SHA256'] == pin['SHA256'], 'receipt member pin differs')
        if value['passed']:
            require(len(members) == 26 and value['logical_payload_bytes'] == PLAN['stream_payload_bytes']
                    and value['tensor_decodes'] == value['Native_Torch_SDK_GPU_calls'] == 0,
                    'passed receipt completeness differs')
            require(value['finished_epoch'] <= CONTEXT['deadline'] and value['system_cost']['actual_nice'] == 10,
                    'actual endpoint deadline/cost receipt differs')
        if observation['same_host'] and observation['original_process_live'] is False:
            result['execution_state'] = 'TERMINAL_PASS' if value['passed'] else 'TERMINAL_FAIL'
    if MODE == 'receiver':
        rows = []
        receipts = {row['name']:row for row in outcome['value']['verified_members']} if outcome else {}
        for pin in PLAN['files']:
            path = safe(root/pin['name'])
            item = dict(name=pin['name'], exists=path.exists())
            if path.exists():
                value = path.stat()
                actual = [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]
                require(stat.S_ISREG(value.st_mode) and value.st_uid == os.getuid(), 'target file type/owner differs')
                item.update(bytes=value.st_size, stat=actual,
                            receipt_stat_matches=pin['name'] in receipts and receipts[pin['name']]['target_stat'] == actual)
            item['incomplete_exists'] = path.with_name(path.name+'.incomplete').exists()
            rows.append(item)
        result['target_stat_rows'] = rows
        if result['execution_state'] == 'TERMINAL_PASS':
            require(all(row['exists'] and row['receipt_stat_matches'] and not row['incomplete_exists'] for row in rows),
                    'current target stats differ from original full-write receipt')
    return result


if __name__ == '__main__':
    print(json.dumps(main(), sort_keys=True, allow_nan=False))
