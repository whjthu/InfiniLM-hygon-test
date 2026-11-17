"""User-authorized exact service tree stop, with PID handles and no process-group kill.

BOUND supplies the eight original GPU UUID/PCI rows. No argv/environment is read.
This script never starts a reference or restarts a service. Existing intent is a hard stop.
"""
from pathlib import Path
import datetime
import hashlib
import json
import os
import signal
import subprocess
import time

OUT = Path('/home/whj/nvidia-baseline/alignment-assets/hygon-a100-service-pause-20261117-v1')
EXPECTED = {943105: (1, 467829073), 943244: (943105, 467830353),
            943277: (943105, 467830390)}


def require(value, message):
    if not value:
        raise ValueError(message)


def query(kind):
    options = {'cards': '--query-gpu=index,uuid,pci.bus_id',
               'apps': '--query-compute-apps=gpu_uuid,pid'}
    raw = subprocess.check_output(['nvidia-smi', options[kind], '--format=csv,noheader,nounits'],
                                  timeout=10, text=True)
    return [list(map(str.strip, line.split(','))) for line in raw.splitlines() if line.strip()]


def identity(pid):
    p = Path('/proc') / str(pid)
    try:
        a = (p / 'stat').read_text().rsplit(') ', 1)[1].split()
        u = [int(x) for x in (p / 'status').read_text().split('Uid:', 1)[1].splitlines()[0].split()]
        b = (p / 'stat').read_text().rsplit(') ', 1)[1].split()
    except FileNotFoundError:
        return None
    require(a[19] == b[19] and a[1] == b[1], 'process changed during identity read')
    return dict(pid=pid, parent=int(b[1]), startticks=int(b[19]), UID=u, state=b[0],
                PGID=int(b[2]), comm=(p / 'comm').read_text().strip())


def tree():
    todo = [(943105, 0)]
    result = {}
    while todo:
        pid, depth = todo.pop()
        require(depth < 9 and len(result) < 32 and pid not in result, 'unexpected service tree')
        value = identity(pid)
        require(value is not None and value['UID'] == [1001] * 4, 'service process identity unavailable')
        require(pid == 943105 or value['parent'] in result, 'descendant parent changed')
        result[pid] = value
        children = (Path('/proc') / str(pid) / 'task' / str(pid) / 'children').read_text().split()
        todo.extend((int(child), depth + 1) for child in children)
    return result


def main():
    require(hasattr(os, 'pidfd_open') and hasattr(signal, 'pidfd_send_signal'), 'race-safe PID handles unavailable')
    require(query('cards') == BOUND['cards'], 'original eight GPU identities changed')
    before_apps = query('apps')
    require(sorted({int(x[1]) for x in before_apps}) == sorted(EXPECTED), 'different compute process set')
    owned = tree()
    for pid, (parent, ticks) in EXPECTED.items():
        require(pid in owned and (owned[pid]['parent'], owned[pid]['startticks']) == (parent, ticks),
                'original approved service identity changed')
    require(owned[943105]['comm'] == 'python3'
            and owned[943244]['comm'].startswith('sglang')
            and owned[943277]['comm'].startswith('sglang'), 'different approved service')
    handles = {}
    for pid, old in owned.items():
        handles[pid] = os.pidfd_open(pid)
        now = identity(pid)
        require(now and now['startticks'] == old['startticks'] and now['UID'] == old['UID'],
                'process changed before PID handle acquisition')
    require(set(tree()) == set(owned), 'service spawned children during admission')
    OUT.mkdir(mode=0o700, exist_ok=False)
    intent = dict(authorization='USER_EXPLICIT_REPLY_ALLOW_PAUSE_FOR_A100_REFERENCES',
                  epoch=time.time(), original_tree=owned, before_apps=before_apps,
                  cards=BOUND['cards'], process_group_signals=False, argv_read=False,
                  environment_read=False, maximum_seconds=55, replay_allowed=False)
    with (OUT / 'intent.json').open('x') as stream:
        json.dump(intent, stream, indent=2)
    started = time.monotonic()
    events = []

    def alive(pid):
        now = identity(pid)
        return bool(now and now['startticks'] == owned[pid]['startticks'] and now['state'] != 'Z')

    def send(pid, sig):
        if not alive(pid):
            return
        try:
            signal.pidfd_send_signal(handles[pid], sig)
            events.append(dict(pid=pid, startticks=owned[pid]['startticks'], signal=sig.name,
                               elapsed_seconds=time.monotonic() - started))
        except ProcessLookupError:
            events.append(dict(pid=pid, signal=sig.name, exited_before_signal=True))

    def wait(seconds):
        end = min(started + 43, time.monotonic() + seconds)
        while any(alive(pid) for pid in owned) and time.monotonic() < end:
            time.sleep(.25)

    # Give the server the first opportunity to perform its own graceful teardown.
    send(943105, signal.SIGTERM)
    wait(20)
    for pid in reversed(list(owned)):
        send(pid, signal.SIGTERM)
    wait(10)
    for pid in reversed(list(owned)):
        send(pid, signal.SIGKILL)
    wait(5)
    after_apps = query('apps')
    remaining = [pid for pid in owned if alive(pid)]
    result = dict(state='APPROVED_SERVICE_STOPPED' if not remaining and not after_apps else 'STOP_ATTEMPTED_REMAINDER_OR_NEW_OCCUPANCY',
                  observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  owned_tree=owned, signals=events, original_processes_remaining=remaining,
                  compute_apps_after=after_apps, eight_GPU_identities_match=query('cards') == BOUND['cards'],
                  new_processes_signalled=False, process_group_signals=False,
                  reference_launched=False, replay_allowed=False,
                  elapsed_seconds=time.monotonic() - started)
    with (OUT / 'result.json').open('x') as stream:
        json.dump(result, stream, indent=2)
    for handle in handles.values():
        os.close(handle)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
