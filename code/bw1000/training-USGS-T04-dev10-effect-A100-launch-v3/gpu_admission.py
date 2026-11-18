"""Passive selected physical A100 ownership snapshot, no Torch or device open."""
import csv
import io
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import time
import hashlib
import json
import transport


def need(ok, code):
    if not ok:
        raise ValueError(code)


class AdmissionFailure(ValueError):
    """Constant public code, bounded private query/driver diagnostics."""
    def __init__(self, code, diagnostic):
        super().__init__(code)
        self.code = code
        self.diagnostic = diagnostic
        need(len(json.dumps(diagnostic, ensure_ascii=True).encode()) <= 16384, 'PRIVATE_DIAGNOSTIC_BOUND')


def log_record(fields, kind, rc, stdout, stderr, **extra):
    def body(x):
        return x.encode('utf8', errors='replace') if isinstance(x, str) else x or b''
    out, err = body(stdout), body(stderr)
    return dict(query_fields=fields, query_kind=kind, returncode=rc,
        stdout_bytes=len(out), stdout_SHA256=hashlib.sha256(out).hexdigest(),
        stderr_bytes=len(err), stderr_SHA256=hashlib.sha256(err).hexdigest(),
        stdout_preview_private=out[:1024].decode('utf8', errors='replace'),
        stderr_preview_private=err[:1024].decode('utf8', errors='replace'), **extra)


def pci(value):
    parts = value.strip().lower().split(':')
    need(len(parts) == 3, 'PCI_COLUMNS')
    return '%04x:%02x:%s' % (int(parts[0], 16), int(parts[1], 16), parts[2])


def query(fields, kind, deadline):
    remaining = deadline - time.time() - 2
    need(remaining > 0, 'ADMISSION_EPOCH_EXPIRED')
    try:
        result = subprocess.run(['nvidia-smi', '--query-' + kind + '=' + fields, '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=min(5, remaining), check=False)
    except subprocess.TimeoutExpired as error:
        raise AdmissionFailure('NVIDIA_QUERY_TIMEOUT', log_record(fields, kind, None, error.stdout, error.stderr,
            error_type='TimeoutExpired')) from None
    except OSError as error:
        raise AdmissionFailure('NVIDIA_QUERY_OSERROR', log_record(fields, kind, None, b'', str(error),
            error_type=type(error).__name__, errno=error.errno)) from None
    diagnostic = log_record(fields, kind, result.returncode, result.stdout, result.stderr)
    if result.returncode != 0 or len(result.stdout.encode()) > 65536 or len(result.stderr.encode()) > 65536:
        raise AdmissionFailure('BOUNDED_NVIDIA_QUERY_FAILED', diagnostic)
    return list(csv.reader(io.StringIO(result.stdout.strip()))) if result.stdout.strip() else []


def parse_information(raw, expected):
    need(isinstance(raw, bytes) and 0 < len(raw) <= 4096 and b'\0' not in raw, 'DRIVER_INFORMATION_BODY_BOUND')
    text = raw.decode('ascii', errors='strict'); required = {'GPU UUID': [], 'Bus Location': [], 'Device Minor': []}
    for line in text.splitlines():
        key, separator, value = line.partition(':')
        if separator and key.strip() in required:
            required[key.strip()].append(value.strip())
    need(all(len(values) == 1 for values in required.values()), 'DRIVER_INFORMATION_REQUIRED_FIELDS_ONCE')
    uuid = required['GPU UUID'][0]; bus = pci(required['Bus Location'][0]); minor = required['Device Minor'][0]
    need(uuid == expected['UUID'] and bus == pci(expected['PCI']), 'DRIVER_INFORMATION_UUID_PCI_MISMATCH')
    need(re.fullmatch(r'[0-9]{1,3}', minor) is not None and 0 <= int(minor) <= 255, 'DRIVER_INFORMATION_MINOR_INTEGER')
    return dict(UUID=uuid, PCI=bus, Device_Minor=int(minor))


def driver_information(expected, deadline):
    need(time.time() < deadline, 'DRIVER_INFORMATION_EPOCH')
    path = Path('/proc/driver/nvidia/gpus') / pci(expected['PCI']) / 'information'
    raw = b''
    try:
        before = path.lstat()
        need(not path.is_symlink() and stat.S_ISREG(before.st_mode), 'EXACT_DRIVER_INFORMATION_REGULAR')
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, 'rb') as stream:
            opened = os.fstat(stream.fileno()); raw = stream.read(4097); after = os.fstat(stream.fileno())
        identity = lambda s: (s.st_dev, s.st_ino, s.st_mtime_ns)
        need(identity(before) == identity(opened) == identity(after) == identity(path.lstat()), 'STABLE_DRIVER_INFORMATION')
        fields = parse_information(raw, expected)
        need(time.time() < deadline, 'DRIVER_INFORMATION_EPOCH')
        return dict(path=str(path), bytes=len(raw), SHA256=hashlib.sha256(raw).hexdigest(), fields=fields,
                    minor_source='Actual UUID/PCI-bound Device Minor field, never physical index')
    except (OSError, ValueError, UnicodeError) as error:
        raise AdmissionFailure('DRIVER_INFORMATION_FAILED', dict(path=str(path), bytes=len(raw),
            SHA256=hashlib.sha256(raw).hexdigest(), error_type=type(error).__name__, error_code=str(error)[:256])) from None


def selected_fd_holders(minor, deadline, owner):
    # Only this runtime owner's exact direct SSH parent may remain uninspected.
    transport_before = transport.capture(owner, deadline)
    rows = []; checked = 0; incomplete = []; exceptions = []
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        need(time.time() < deadline and checked < 4096, 'FD_INVENTORY_BOUND')
        candidate = None; stage = 'proc_identity'
        try:
            proc_uid = directory.stat().st_uid
            if proc_uid != os.getuid():
                continue
            checked += 1
            before = (directory / 'stat').read_text().rsplit(') ', 1)[1].split()
            candidate = dict(PID=int(directory.name), PPID=int(before[1]), startticks=int(before[19]), proc_uid=proc_uid)
            if before[0] in ('Z', 'X'):
                continue
            targets = []
            stage = 'fd_directory'
            for ordinal, fd in enumerate((directory / 'fd').iterdir()):
                stage = 'fd_target'
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
            if transport.eligible(transport_before, candidate, stage):
                need(not exceptions, 'ONE_CURRENT_OWN_TRANSPORT_EXCEPTION_ONLY')
                exceptions.append(dict(**candidate, reason='Exact current own SSH directparent FD directory PermissionError',
                    fd_uid=transport_before['chain'][1]['fd_uid'], fd_mode=transport_before['chain'][1]['fd_mode'],
                    fd_error_stage=stage, exe_identity='UNKNOWN', transport_FDs_inspected=False))
            else:
                incomplete.append(int(directory.name))
    need(not incomplete, 'LIVE_OWN_UID_FD_INVENTORY_UNKNOWN')
    evidence = transport.confirm(transport_before, transport.capture(owner, deadline))
    used = bool(exceptions)
    evidence['transport_FDs_inspected'] = not used
    return dict(checked_own_UID_processes=checked, selected_device_or_KFD_holders=rows, incomplete_processes=0,
        uninspected_transport_ancestor=exceptions[0] if used else None, transport_exception_count=len(exceptions),
        transport_evidence=evidence, all_UID_FD_complete=not used, nontransport_UID_FD_complete=True,
        transport_FDs_inspected=not used, exe_identity='UNKNOWN',
        scope_policy='scope_admission_satisfied_with_transport_exception' if used else 'scope_admission_satisfied_with_complete_own_UID_FD_inspection')


def snapshot(expected, master_port, deadline, owner=None):
    devices = query('index,uuid,pci.bus_id,memory.free,memory.total', 'gpu', deadline)
    match = [row for row in devices if len(row) == 5 and row[1].strip() == expected['UUID']]
    need(len(match) == 1, 'ONE_EXACT_PHYSICAL_UUID_REQUIRED')
    index, uuid, bus, free, total = [value.strip() for value in match[0]]
    need(int(index) == expected['index'] and pci(bus) == pci(expected['PCI']), 'PHYSICAL_INDEX_UUID_PCI_MISMATCH')
    need(0 < int(free) <= int(total) and int(total) == expected['total_MiB'] == 40960
         and int(free) >= expected['minimum_free_MiB'] == 38000, 'A100_40GB_PRE_ADMISSION_MEMORY')
    applications = query('gpu_uuid,pid', 'compute-apps', deadline)
    need(all(len(row) == 2 for row in applications), 'COMPUTE_APP_COLUMNS')
    active = [row for row in applications if row[0].strip() == uuid]
    need(not active, 'SELECTED_GPU_ALREADY_HAS_COMPUTE_PROCESS_NO_KILL_OR_FALLBACK')
    information = driver_information(expected, deadline)
    minor = information['fields']['Device_Minor']
    fds = selected_fd_holders(minor, deadline, owner)
    need(not fds['selected_device_or_KFD_holders'], 'SELECTED_GPU_OR_KFD_HAS_EXISTING_OWNER')
    need(type(master_port) is int and 1024 <= master_port <= 65535, 'FIXED_PORT_REQUIRED')
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', master_port))
    return dict(observed_epoch=time.time(), observed_monotonic=time.monotonic(),
        physical_GPU_count=1, devices=[dict(index=int(index), UUID=uuid, PCI=pci(bus), minor=int(minor),
        free_MiB=int(free), total_MiB=int(total))], active_compute_processes=0,
        master_port_bindable=True, own_UID_FD_snapshot=fds, driver_information=information, GPU_busy_measurement=None,
        scope_policy=fds['scope_policy'], scope_admission_satisfied=True, all_UID_FD_complete=fds['all_UID_FD_complete'],
        nontransport_UID_FD_complete=True, transport_FDs_inspected=fds['transport_FDs_inspected'],
        exe_identity='UNKNOWN', exclusive_GPU_proven=False)
