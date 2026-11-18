"""Only the current supervisor's exact own SSH direct parent may remain uninspected.

No executable reads, FD-content reads, signals, privilege changes or stored PID reuse.
Status identity fields must remain equal; transient nonterminal scheduler State may vary.
"""
import os
from pathlib import Path
import re
import socket
import stat
import time
import hashlib


def need(ok, code):
    if not ok:
        raise ValueError(code)


def stat_identity(raw, pid):
    values = raw.rsplit(') ', 1)[1].split()
    need(int(raw.split(' ', 1)[0]) == pid and len(values) >= 20, 'TRANSPORT_PROC_STAT_IDENTITY')
    need(values[0] not in ('Z', 'X') and len(values[0]) == 1, 'TRANSPORT_PROCESS_NOT_LIVE')
    return dict(PID=pid, PPID=int(values[1]), PGID=int(values[2]), startticks=int(values[19]),
                comm=raw.split('(', 1)[1].rsplit(') ', 1)[0], state=values[0])


def status_identity(raw, row, proc_uid):
    keys = {'PPid','Uid','Gid','TracerPid','NSpid','CapEff','NoNewPrivs','State'}
    fields = {}
    for line in raw.decode('ascii', errors='strict').splitlines():
        key, separator, value = line.partition(':')
        if separator and key in keys:
            need(key not in fields, 'TRANSPORT_DUPLICATE_STATUS_IDENTITY')
            fields[key] = value.strip()
    need(set(fields) == keys, 'TRANSPORT_STATUS_IDENTITY_FIELDS')
    for key in ['Uid', 'Gid']:
        values = fields[key].split(); need(len(values) == 4 and all(x.isdigit() for x in values), 'TRANSPORT_STATUS_IDS')
        fields[key] = [int(x) for x in values]
    for key in ['PPid','TracerPid','NoNewPrivs']:
        need(fields[key].isdigit(), 'TRANSPORT_STATUS_INTEGER'); fields[key] = int(fields[key])
    need(all(x.isdigit() for x in fields['NSpid'].split()), 'TRANSPORT_STATUS_NAMESPACE_PID')
    fields['NSpid'] = [int(x) for x in fields['NSpid'].split()]
    need(fields['PPid'] == row['PPID'] and fields['Uid'] == [proc_uid] * 4
         and fields['NSpid'] == [row['PID']] and fields['TracerPid'] == 0
         and fields['NoNewPrivs'] in (0, 1), 'TRANSPORT_STATUS_PROC_IDENTITY_MISMATCH')
    need(re.fullmatch('[0-9a-fA-F]{16}', fields['CapEff']) is not None, 'TRANSPORT_CAPABILITY_ENCODING')
    fields['CapEff'] = fields['CapEff'].lower()
    state = fields.pop('State').split()[0]
    need(len(state) == 1 and state not in ('Z','X'), 'TRANSPORT_STATUS_TERMINAL')
    return fields, state


def read_process(pid, deadline):
    need(time.time() < deadline and type(pid) is int and pid > 0, 'TRANSPORT_METADATA_BUDGET')
    path = Path('/proc') / str(pid)
    directory = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        def read(name, cap):
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
            with os.fdopen(fd, 'rb') as stream:
                raw = stream.read(cap + 1)
            need(len(raw) <= cap, 'TRANSPORT_METADATA_BOUND'); return raw
        proc = os.fstat(directory)
        before = stat_identity(read('stat', 8192).decode('ascii'), pid)
        fd_before = os.stat('fd', dir_fd=directory, follow_symlinks=False)
        status_raw = read('status', 65536)
        status, state = status_identity(status_raw, before, proc.st_uid)
        fd_after = os.stat('fd', dir_fd=directory, follow_symlinks=False)
        after = stat_identity(read('stat', 8192).decode('ascii'), pid)
        stable = ['PID','PPID','PGID','startticks','comm']
        need(all(before[key] == after[key] for key in stable), 'TRANSPORT_PROCESS_CHANGED_DURING_METADATA')
        fd_key = lambda value: (value.st_dev, value.st_ino, value.st_uid, stat.S_IMODE(value.st_mode))
        need(stat.S_ISDIR(fd_before.st_mode) and fd_key(fd_before) == fd_key(fd_after)
             and proc.st_uid == os.fstat(directory).st_uid, 'TRANSPORT_FD_DIRECTORY_METADATA_CHANGED')
        need(time.time() < deadline, 'TRANSPORT_METADATA_BUDGET')
        return dict(**{key: before[key] for key in stable}, proc_uid=proc.st_uid,
            proc_device=proc.st_dev, proc_inode=proc.st_ino, fd_uid=fd_before.st_uid,
            fd_mode=stat.S_IMODE(fd_before.st_mode), fd_device=fd_before.st_dev, fd_inode=fd_before.st_ino,
            status_identity=status, observed_states=[before['state'], state, after['state']],
            status_bytes=len(status_raw), status_SHA256=hashlib.sha256(status_raw).hexdigest(),
            exe_identity='UNKNOWN', exe_read=False, FD_targets_read=False)
    finally:
        os.close(directory)


def validate_chain(owner, rows, host, boot, self_pid, self_uid):
    need(len(rows) == 4 and len({r['PID'] for r in rows}) == 4, 'EXACT_FOUR_PROCESS_TRANSPORT_CHAIN')
    need(self_pid == owner['PID'] == rows[0]['PID'] and owner['startticks'] == rows[0]['startticks']
         and self_uid == owner['uid'] == rows[0]['proc_uid'] == 1001
         and host == owner['host'] and boot == owner['boot_id'], 'CURRENT_SAVED_SUPERVISOR_OWNER')
    need(all(rows[i]['PPID'] == rows[i+1]['PID'] for i in range(3)) and rows[3]['PPID'] == 0,
         'CURRENT_DIRECT_PARENT_AND_TWO_ROOT_LEVELS')
    need(rows[1]['proc_uid'] == self_uid and rows[2]['proc_uid'] == rows[3]['proc_uid'] == 0,
         'TRANSPORT_CHAIN_UIDS')
    for row in rows:
        status = row['status_identity']
        need(status['PPid'] == row['PPID'] and status['Uid'] == [row['proc_uid']] * 4
             and status['NSpid'] == [row['PID']] and status['TracerPid'] == 0,
             'TRANSPORT_CHAIN_STATUS_IDENTITY')
        need(row['exe_identity'] == 'UNKNOWN' and row['exe_read'] is row['FD_targets_read'] is False,
             'TRANSPORT_NO_EXECUTABLE_OR_FD_INSPECTION_CLAIM')
    for row in rows[1:]:
        need(row['comm'] == 'sshd' and row['fd_uid'] == 0 and row['fd_mode'] == 0o500,
             'TRANSPORT_PRIVILEGED_FD_DIRECTORY_METADATA')
    need(rows[1]['status_identity']['CapEff'] == '0000000000000000'
         and rows[1]['status_identity']['NoNewPrivs'] == 0, 'OBSERVED_UNPRIVILEGED_DIRECT_TRANSPORT')
    need(rows[0]['fd_uid'] == self_uid and rows[0]['fd_mode'] == 0o500, 'SUPERVISOR_FD_DIRECTORY_METADATA')
    return True


def capture(owner, deadline):
    need(isinstance(owner, dict), 'CURRENT_TRANSPORT_OWNER_REQUIRED')
    host = socket.gethostname(); boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    pid = os.getpid(); rows = []
    for _ in range(4):
        row = read_process(pid, deadline); rows.append(row); pid = row['PPID']
    validate_chain(owner, rows, host, boot, os.getpid(), os.getuid())
    need(os.getppid() == rows[1]['PID'], 'TRANSPORT_DIRECT_PARENT_CHANGED')
    return dict(host=host, boot_id=boot, supervisor_PID=owner['PID'], scope_request_id=owner['scope_request_id'],
                chain=rows, exe_identity='UNKNOWN', executable_identity_verified=False)


def eligible(before, candidate, stage):
    parent = before['chain'][1]
    return bool(stage == 'fd_directory' and candidate is not None
        and all(candidate[key] == parent[key] for key in ['PID','PPID','startticks','proc_uid']))


def confirm(before, after):
    need(len(before['chain']) == len(after['chain']) == 4, 'TRANSPORT_COMPARISON_CHAIN_DENOMINATOR')
    need(all(before[key] == after[key] for key in ['host','boot_id','supervisor_PID','scope_request_id']),
         'TRANSPORT_HOST_BOOT_SCOPE_CHANGED')
    keys = ['PID','PPID','PGID','startticks','comm','proc_uid','proc_device','proc_inode',
            'fd_uid','fd_mode','fd_device','fd_inode','status_identity']
    need(all(all(first[key] == second[key] for key in keys) for first, second in zip(before['chain'], after['chain'])),
         'TRANSPORT_ANCESTRY_OR_IDENTITY_CHANGED')
    return dict(before=before, after=after, stable_identity=True,
        transient_nonterminal_states_may_vary=True, status_identity_fields_stable=True,
        exe_identity='UNKNOWN', transport_FDs_inspected=False,
        observation_scope='Current exact own SSH directparent only; comm is auxiliary and executable identity remains UNKNOWN')
