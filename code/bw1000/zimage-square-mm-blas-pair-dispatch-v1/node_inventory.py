"""Read original collector device identity on one known Slurm node."""
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import sys

PROC = Path("/proc")
DRM = Path("/sys/class/drm")
KFD_PROCESSES = Path("/sys/class/kfd/kfd/proc")


def require(value, message):
    if not value:
        raise ValueError(message)


def device_status(ports):
    pci = sorted((entry / 'device').resolve(strict=True).name.lower()
                 for entry in DRM.iterdir() if re.fullmatch(r'renderD[0-9]+', entry.name))
    processes = {entry.name for entry in KFD_PROCESSES.iterdir()}
    require(all(pid.isdecimal() for pid in processes), 'unknown KFD process inventory format')
    own_device_fds = set()
    for entry in PROC.iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            for fd in (entry / 'fd').iterdir():
                try:
                    target = os.readlink(fd).removesuffix(' (deleted)')
                except FileNotFoundError:
                    continue
                if target == '/dev/kfd' or target.startswith('/dev/dri/'):
                    own_device_fds.add(entry.name)
        except FileNotFoundError:
            continue
    occupied_ports = set()
    if ports:
        for table in ('tcp', 'tcp6', 'udp', 'udp6'):
            lines = (PROC / 'net' / table).read_text().splitlines()
            require(lines and 'local_address' in lines[0], 'unknown kernel socket table format')
            occupied_ports.update(int(line.split()[1].rsplit(':', 1)[1], 16) for line in lines[1:])
    return dict(PCIs=pci, kfd_process_count=len(processes), own_device_fd_process_count=len(own_device_fds),
                occupied_requested_ports=sorted(set(ports) & occupied_ports))


if __name__ == "__main__":
    parent, node = sys.argv[1:]
    step = os.environ["SLURM_STEP_ID"]
    if socket.gethostname().split(".")[0] != node or os.environ["SLURM_JOB_ID"] != parent or not step.isdecimal():
        raise ValueError("Outside the exact zero-GRES inventory job")
    identity = hashlib.sha256((PROC / "sys/kernel/random/boot_id").read_bytes()).hexdigest()
    print(json.dumps(dict(node=node, step=parent+"."+step, boot_id_SHA256=identity, **device_status([]))))
