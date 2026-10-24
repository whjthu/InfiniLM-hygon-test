import json
import os
import pathlib
import socket

holders = []
for proc in pathlib.Path('/proc').iterdir():
    if not proc.name.isdigit():
        continue
    try:
        if proc.stat().st_uid != os.getuid():
            continue
        devices = []
        for fd in (proc / 'fd').iterdir():
            try:
                link = os.readlink(fd)
                if link == '/dev/kfd' or link.startswith('/dev/dri/'):
                    devices.append(link)
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                pass
        if devices:
            holders.append({'pid': int(proc.name), 'devices': sorted(set(devices))})
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        pass
v = os.statvfs('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results')
report = {'node': socket.gethostname(), 'uid': os.getuid(), 'GPU_processes_owned_by_user': holders,
          'scratch_free_bytes': v.f_bavail * v.f_frsize}
report['passed'] = not holders and report['scratch_free_bytes'] >= 40 * 1024 ** 3
print(json.dumps(report), flush=True)
assert report['passed']
