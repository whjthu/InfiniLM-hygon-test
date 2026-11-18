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


def add_incomplete(state, record):
    """Keep total counts even when the max64 /16KiB private identity array is full."""
    state['total'] += 1
    state.setdefault('PIDs', set()).add(record['PID'])
    counter = 'permission_total' if record['error_type'] == 'PermissionError' else 'gone_or_closed_race_total'
    state[counter] = state.get(counter, 0) + 1
    candidate = state['records'] + [record]
    size = len(json.dumps(candidate, ensure_ascii=True, sort_keys=True, allow_nan=False).encode())
    if len(state['records']) < 64 and size <= 16384:
        state['records'].append(record); state['saved_bytes'] = size
    state['truncated_count'] = state['total'] - len(state['records'])


def process_identity(directory):
    raw = (directory / 'stat').read_text(); values = raw.rsplit(') ', 1)[1].split()
    need(int(raw.split(' ', 1)[0]) == int(directory.name) and len(values) >= 20, 'FD_PROCESS_STAT_IDENTITY')
    return dict(PID=int(directory.name), startticks=int(values[19]), PPID=int(values[1]), uid=directory.stat().st_uid)


def fd_directory_metadata(directory):
    value = (directory / 'fd').stat()
    return dict(fd_uid=value.st_uid, fd_mode=stat.S_IMODE(value.st_mode))


def observe_process(directory, minor, deadline, context, holders, incomplete):
    context['stage'] = 'proc_owner'; context['uid'] = directory.stat().st_uid
    if context['uid'] != os.getuid():
        return False
    context['stage'] = 'proc_stat'; context['identity_before'] = process_identity(directory)
    context['stage'] = 'fd_metadata'; context.update(fd_directory_metadata(directory))
    context['stage'] = 'fd_directory'
    # Keep scandir open while reading targets. In particular, our own scan FD cannot
    # become an artificial missing /proc/self/fd entry when the iterator is closed.
    with os.scandir(directory / 'fd') as entries:
        for ordinal, entry in enumerate(entries):
            need(time.time() < deadline and ordinal < 8192, 'PER_PROCESS_FD_SNAPSHOT_BOUND')
            context['stage'] = 'fd_target'
            context['fd_number'] = int(entry.name) if entry.name.isdigit() else None
            try:
                target = os.readlink(entry.path)
            except (FileNotFoundError, ProcessLookupError) as error:
                context['had_lifecycle_race'] = True
                add_incomplete(incomplete, incomplete_record(directory, context, error))
                continue  # This FD closed; continue observing other actual readable targets.
            if target in ('/dev/nvidia' + str(minor), '/dev/kfd'):
                holders.append(dict(**context['identity_before'], target=target))
    context['stage'] = 'after_proc_stat'; context['identity_after'] = process_identity(directory)
    need(context['identity_before'] == context['identity_after'], 'FD_PROCESS_IDENTITY_CHANGED')
    return True


def incomplete_record(directory, context, error):
    stage = context['stage']
    # Only permission and explicit process/FD lifecycle observations may be UNKNOWN.
    # Other OSError/ValueError remains an error, never a successful partial scan.
    metadata_errors = []
    try:
        context['identity_after'] = process_identity(directory)
    except (PermissionError, FileNotFoundError, ProcessLookupError) as secondary:
        context['identity_after'] = None
        metadata_errors.append(dict(stage='recover_after_proc_stat', error_type=type(secondary).__name__, errno=secondary.errno))
    if context['fd_uid'] is None:
        try:
            context.update(fd_directory_metadata(directory))
        except (PermissionError, FileNotFoundError, ProcessLookupError) as secondary:
            metadata_errors.append(dict(stage='recover_fd_metadata', error_type=type(secondary).__name__, errno=secondary.errno))
    before, after = context['identity_before'], context['identity_after']
    if before is not None and after is not None:
        need(before == after, 'FD_PROCESS_IDENTITY_CHANGED')
    available = before or after or {}
    error_type = 'PermissionError' if isinstance(error, PermissionError) else 'FileNotFoundError' if isinstance(error, FileNotFoundError) else 'ProcessLookupError'
    return dict(PID=int(directory.name), startticks=available.get('startticks'), PPID=available.get('PPID'),
        uid=context['uid'] if context['uid'] is not None else available.get('uid'), stage=stage,
        errno=error.errno, error_type=error_type, fd_uid=context['fd_uid'], fd_mode=context['fd_mode'],
        fd_number=context.get('fd_number'), identity_before=before, identity_after=after,
        identity_stable=(before == after) if before is not None and after is not None else None,
        metadata_errors=metadata_errors, original_PID_termination_proven=False)


def selected_fd_holders(minor, deadline):
    holders = []; attempted = 0; completed = 0; visited = 0
    incomplete = dict(total=0, records=[], saved_bytes=2, truncated_count=0, PIDs=set(), permission_total=0, gone_or_closed_race_total=0)
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        visited += 1
        need(time.time() < deadline and attempted < 4096 and visited <= 8192, 'FD_INVENTORY_BOUND')
        context = dict(PID=int(directory.name), uid=None, stage='proc_owner', identity_before=None,
                       identity_after=None, fd_uid=None, fd_mode=None)
        try:
            if observe_process(directory, minor, deadline, context, holders, incomplete):
                attempted += 1
                completed += int(not context.get('had_lifecycle_race', False))
        except (PermissionError, FileNotFoundError, ProcessLookupError) as error:
            attempted += 1
            add_incomplete(incomplete, incomplete_record(directory, context, error))
    partial = incomplete['total'] > 0
    need(attempted-completed == len(incomplete['PIDs']), 'FD_COVERAGE_PROCESS_ACCOUNTING')
    return dict(checked_own_UID_processes=attempted, completed_own_UID_processes=completed,
        visited_processes=visited, attempted_own_or_unknown_UID_processes=attempted,
        process_attempt_limit=4096, global_visit_limit=8192, selected_device_or_KFD_holders=holders,
        incomplete_processes=len(incomplete['PIDs']), incomplete_records=incomplete['records'],
        partial_process_attempts=attempted-completed,
        incomplete_total=incomplete['total'], incomplete_saved_bytes=incomplete['saved_bytes'],
        permission_incomplete_total=incomplete['permission_total'], gone_or_closed_race_total=incomplete['gone_or_closed_race_total'],
        incomplete_truncated=incomplete['truncated_count'] > 0, incomplete_truncated_count=incomplete['truncated_count'],
        FD_coverage_state='PARTIAL_UNKNOWN' if partial else 'COMPLETE_OWN_UID_READABLE',
        FD_unknown_reasons=(['PERMISSION'] if incomplete['permission_total'] else []) +
            (['GONE_OR_CLOSED_RACE'] if incomplete['gone_or_closed_race_total'] else []),
        FD_observation_is_auxiliary=True, all_UID_FD_complete=not partial, nontransport_UID_FD_complete=None,
        transport_FDs_inspected=None, exe_identity='UNKNOWN', all_FD_clean_qualified=False, exclusive_GPU_proven=False,
        UID_scope='own UID plus unknown-UID discovery races only; identity may be incomplete; no ancestor/executable checks; vanished PID or closed FD is not terminal proof')


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
    fds = selected_fd_holders(minor, deadline)
    need(not fds['selected_device_or_KFD_holders'], 'SELECTED_GPU_OR_KFD_HAS_EXISTING_OWNER')
    need(type(master_port) is int and 1024 <= master_port <= 65535, 'FIXED_PORT_REQUIRED')
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', master_port))
    return dict(observed_epoch=time.time(), observed_monotonic=time.monotonic(),
        physical_GPU_count=1, devices=[dict(index=int(index), UUID=uuid, PCI=pci(bus), minor=int(minor),
        free_MiB=int(free), total_MiB=int(total))], active_compute_processes=0,
        master_port_bindable=True, own_UID_FD_snapshot=fds, driver_information=information, GPU_busy_measurement=None,
        scope_policy='primary_GPU_admission_satisfied_with_auxiliary_FD_coverage', scope_admission_satisfied=True, primary_GPU_admission_satisfied=True,
        FD_coverage_state=fds['FD_coverage_state'], FD_incomplete_total=fds['incomplete_total'],
        FD_incomplete_processes=fds['incomplete_processes'], FD_observation_is_auxiliary=True,
        all_UID_FD_complete=fds['all_UID_FD_complete'], nontransport_UID_FD_complete=None, transport_FDs_inspected=None,
        exe_identity='UNKNOWN', all_FD_clean_qualified=False, exclusive_GPU_proven=False)
