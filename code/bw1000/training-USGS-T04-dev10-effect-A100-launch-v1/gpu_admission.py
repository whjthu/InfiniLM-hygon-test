"""Passive selected physical A100 ownership snapshot, no Torch or device open."""
import csv
import io
import os
from pathlib import Path
import socket
import subprocess
import time


def need(ok, code):
    if not ok:
        raise ValueError(code)


def pci(value):
    parts = value.strip().lower().split(':')
    need(len(parts) == 3, 'PCI_COLUMNS')
    return '%04x:%02x:%s' % (int(parts[0], 16), int(parts[1], 16), parts[2])


def query(fields, kind, deadline):
    remaining = deadline - time.time() - 2
    need(remaining > 0, 'ADMISSION_EPOCH_EXPIRED')
    result = subprocess.run(['nvidia-smi', '--query-' + kind + '=' + fields, '--format=csv,noheader,nounits'],
        capture_output=True, text=True, timeout=min(5, remaining), check=False)
    need(result.returncode == 0 and len(result.stdout) <= 65536 and len(result.stderr) <= 65536,
         'BOUNDED_NVIDIA_QUERY_FAILED')
    return list(csv.reader(io.StringIO(result.stdout.strip()))) if result.stdout.strip() else []


def selected_fd_holders(minor, deadline):
    # Same-user processes only, never signal another owner. Unreadable live FD inventory is UNKNOWN.
    rows = []; checked = 0; incomplete = []
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        need(time.time() < deadline and checked < 4096, 'FD_INVENTORY_BOUND')
        try:
            if directory.stat().st_uid != os.getuid():
                continue
            checked += 1
            before = (directory / 'stat').read_text().rsplit(') ', 1)[1].split()
            if before[0] in ('Z', 'X'):
                continue
            targets = []
            for ordinal, fd in enumerate((directory / 'fd').iterdir()):
                need(time.time() < deadline and ordinal < 8192, 'PER_PROCESS_FD_SNAPSHOT_BOUND')
                try:
                    target = os.readlink(fd)
                except FileNotFoundError:
                    continue
                if target in ('/dev/nvidia' + str(minor), '/dev/kfd'):
                    targets.append(target)
            after = (directory / 'stat').read_text().rsplit(') ', 1)[1].split()
            need(before[19] == after[19], 'PROC_IDENTITY_CHANGED_DURING_FD_SNAPSHOT')
            if targets:
                rows.append(dict(PID=int(directory.name), startticks=int(before[19]), targets=sorted(set(targets))))
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            incomplete.append(int(directory.name))
    need(not incomplete, 'LIVE_OWN_UID_FD_INVENTORY_UNKNOWN')
    return dict(checked_own_UID_processes=checked, selected_device_or_KFD_holders=rows, incomplete_processes=0)


def snapshot(expected, master_port, deadline):
    devices = query('index,uuid,pci.bus_id,minor_number,memory.free,memory.total', 'gpu', deadline)
    match = [row for row in devices if len(row) == 6 and row[1].strip() == expected['UUID']]
    need(len(match) == 1, 'ONE_EXACT_PHYSICAL_UUID_REQUIRED')
    index, uuid, bus, minor, free, total = [value.strip() for value in match[0]]
    need(int(index) == expected['index'] and pci(bus) == pci(expected['PCI']), 'PHYSICAL_INDEX_UUID_PCI_MISMATCH')
    need(0 < int(free) <= int(total) and int(total) == expected['total_MiB'] == 40960
         and int(free) >= expected['minimum_free_MiB'] == 38000, 'A100_40GB_PRE_ADMISSION_MEMORY')
    applications = query('gpu_uuid,pid', 'compute-apps', deadline)
    need(all(len(row) == 2 for row in applications), 'COMPUTE_APP_COLUMNS')
    active = [row for row in applications if row[0].strip() == uuid]
    need(not active, 'SELECTED_GPU_ALREADY_HAS_COMPUTE_PROCESS_NO_KILL_OR_FALLBACK')
    fds = selected_fd_holders(int(minor), deadline)
    need(not fds['selected_device_or_KFD_holders'], 'SELECTED_GPU_OR_KFD_HAS_EXISTING_OWNER')
    need(type(master_port) is int and 1024 <= master_port <= 65535, 'FIXED_PORT_REQUIRED')
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', master_port))
    return dict(observed_epoch=time.time(), observed_monotonic=time.monotonic(),
        physical_GPU_count=1, devices=[dict(index=int(index), UUID=uuid, PCI=pci(bus), minor=int(minor),
        free_MiB=int(free), total_MiB=int(total))], active_compute_processes=0,
        master_port_bindable=True, own_UID_FD_snapshot=fds, GPU_busy_measurement=None)
