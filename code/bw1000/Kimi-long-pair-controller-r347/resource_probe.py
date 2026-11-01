"""Host-only KFD/port check, or one bounded passive sole-listener observation."""
import argparse
import datetime
import json
import os
from pathlib import Path
import socket
import subprocess
import time


def listener_owner(port, output):
    """Never connect: an active TCP probe would consume r334's single accept."""
    inodes = set()
    for table in ['/proc/net/tcp', '/proc/net/tcp6']:
        for line in Path(table).read_text().splitlines()[1:]:
            fields = line.split()
            if int(fields[1].split(':')[-1], 16) == port and fields[3] == '0A':
                inodes.add(fields[9])
    matches = []
    if not inodes: return matches
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit(): continue
        try:
            args = (proc/'cmdline').read_bytes().split(b'\0')
            expected = b'/refactor/incoming/Kimi-long-draft-worker-r334b/worker.py'
            if expected not in args or os.fsencode(output) not in args: continue
            sockets = set()
            for fd in (proc/'fd').iterdir():
                try: link = os.readlink(fd)
                except OSError: continue
                if link.startswith('socket:['): sockets.add(link[8:-1])
            if sockets & inodes: matches.append(int(proc.name))
        except (OSError, PermissionError): continue
    return sorted(matches)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ports', default='32947,39947,45947,46947')
    p.add_argument('--listener-output')
    p.add_argument('--port', type=int, default=46947)
    p.add_argument('--deadline-unix', type=float)
    args = p.parse_args()
    q = dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
             node=socket.gethostname().split('.')[0], CPU_only=True, GPU_initialized=False,
             actual_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'])
    if args.listener_output:
        if args.deadline_unix is None: raise ValueError('Original deadline required')
        until = min(time.time()+120, args.deadline_unix-180)
        while time.time() < until:
            owners = listener_owner(args.port, args.listener_output)
            if len(owners) > 1: raise RuntimeError('Ambiguous worker listener owner')
            if owners:
                q.update(listener_ready=True, port=args.port, worker_PID=owners[0],
                         passive_proc_observation=True, TCP_connections_made=0)
                print(json.dumps(q)); return
            time.sleep(.1)
        raise TimeoutError('Owned worker listener was not observed; no connection retry')
    kfd = Path('/sys/class/kfd/kfd/proc')
    if not kfd.is_dir(): raise RuntimeError('Actual KFD proc directory required')
    pids = sorted(int(x.name) for x in kfd.iterdir() if x.name.isdigit())
    ports = [int(v) for v in args.ports.split(',')]
    bindable = {}
    for port in ports:
        with socket.socket() as s:
            try: s.bind(('0.0.0.0', port)); bindable[str(port)] = True
            except OSError: bindable[str(port)] = False
    raw = subprocess.check_output(['ip', '-j', '-4', 'address'], text=True, timeout=5)
    ips = sorted({a['local'] for row in json.loads(raw) if row['ifname'].startswith('ib')
                  for a in row.get('addr_info', []) if a.get('scope') == 'global'})
    q.update(kfd_pids=pids, KFD_empty=not pids, ports_bindable=bindable, ib_ipv4=ips,
             passed=not pids and all(bindable.values()) and bool(ips))
    print(json.dumps(q))


if __name__ == '__main__': main()
