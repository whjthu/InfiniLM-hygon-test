"""Read the original VGGT eight-pair Slurm phases and its separate HOST cleanup."""

from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time


LABEL = 'training-vggt-tum-eight-pairs25-recovery-v1'
REQUEST = 'c55668b8ef104a9cb18be34d94251351'
PUBLICATION = '1c70a9618539ae085fbd0c792abd71d1f7f069ed'
LOCAL_REQUEST_SHA = '32ed148dc6782010f6aa87e8faf29b1fec101c0b838ba863b21083f9bbabfe1c'
PARENT, NODE = '1087656', 'f11r4n19'
ROOT = Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results') / LABEL
PHASES = ('pre', 'gpu', 'post')
COLUMNS = 'JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES'
ACCOUNTING = ('sacct', '-n', '-P', '-j', PARENT, '-S', '2026-11-16T00:00:00',
              '-E', '2026-11-18T23:59:59', '-o', COLUMNS)
TERMINAL = frozenset(('BOOT_FAIL', 'CANCELLED', 'COMPLETED', 'DEADLINE', 'FAILED', 'NODE_FAIL',
                      'OUT_OF_MEMORY', 'PREEMPTED', 'REVOKED', 'TIMEOUT'))
PCI = re.compile(r'[a-f0-9]{4,8}:[a-f0-9]{2}:[a-f0-9]{2}\.[0-7]')
MAX_ACCOUNTING = 2 * 1024 * 1024


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def present(path):
    return path.exists() or path.is_symlink()


def accounting_rows(raw):
    need(len(raw) <= MAX_ACCOUNTING, 'accounting observation exceeds the fixed bound')
    labels = {LABEL + '-' + phase: phase for phase in PHASES}
    phases = dict.fromkeys(PHASES)
    for line in raw.decode('utf8').splitlines():
        if not line.strip():
            continue
        row = line.split('|')
        need(len(row) == 7, 'expected seven explicit Slurm columns')
        phase = labels.get(row[1])
        if phase is None:
            continue
        need(phases[phase] is None and re.fullmatch(PARENT + r'\.[0-9]+', row[0]) and row[5] == NODE and
             row[2].split() and re.fullmatch(r'[0-9]+:[0-9]+', row[3]) and row[4].isdecimal(),
             'foreign, duplicated or malformed original phase')
        phases[phase] = row
    ordinals = [int(row[0].split('.')[1]) for row in phases.values() if row]
    need(ordinals == sorted(set(ordinals)), 'original pre/gpu/post step order differs')
    return phases


def read_json(root, name, hashes):
    path = root / name
    need(path.resolve(strict=True) == path and path.is_file(), 'missing or aliased original small receipt')
    before = path.stat()
    with path.open('rb') as stream:
        raw = stream.read(64 * 1024 + 1)
    after = path.stat()
    need(0 < len(raw) == before.st_size <= 64 * 1024 and
         (before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns),
         'original small receipt changed or exceeds bound')
    value = json.loads(raw)
    need(isinstance(value, dict), 'original receipt must contain one JSON object')
    hashes[name] = sha256(raw).hexdigest()
    return value


def verify_tres(phase, row):
    fields = [token.split('=', 1) for token in row[6].split(',')]
    tres = dict(fields)
    gpu = phase == 'gpu'
    need(len(tres) == len(fields) and tres.get('node') == '1' and tres.get('cpu') == ('4' if gpu else '2') and
         tres.get('mem') in ({'64G', '65536M', '67108864K'} if gpu else {'2G', '2048M', '2097152K'}) and
         {key: value for key, value in tres.items() if key.startswith('gres/')} == ({'gres/dcu': '1'} if gpu else {}),
         'original phase allocation does not match the frozen claim')


def verify_cleanup(root, phases, intent, hashes):
    phase_records = {}
    for phase, row in phases.items():
        verify_tres(phase, row)
        record = read_json(root, phase + '-accounting.json', hashes)
        need(record['row'] == row and record['terminal_visible'] is True and
             record['original_deadline'] == intent['deadline'] and type(record['passed']) is bool,
             'producer phase receipt differs from original accounting')
        if phase != 'gpu':
            need(record['passed'] is True and type(record['rc']) is int and record['rc'] == 0,
                 'original independent HOST phase did not pass')
        else:
            rc = record['rc']
            need((rc is None or type(rc) is int) and record['passed'] == (rc == 0 and row[2:4] == ['COMPLETED', '0:0']),
                 'original GPU launch result disagrees with its own accounting')
        phase_records[phase] = record
    snapshots = {}
    for phase, name in (('pre', 'before.json'), ('post', 'after.json')):
        snapshot = read_json(root, name, hashes)
        logged = read_json(root, phase + '.log', hashes)
        pci = snapshot['pci']
        need(snapshot == logged and snapshot['step'] == phases[phase][0] and snapshot['node'] == NODE and
             snapshot['kind'] == 'training_qwen_device_snapshot' and snapshot['mode'] == 'host_complete' and
             snapshot['holder_inventory_basis'] == 'kernel_KFD_plus_sameUID_FD_complete' and
             snapshot['full_fd_inventory_performed'] is True and snapshot['pci_to_kfd'] is None and
             type(snapshot['kfd_holder_count']) is int and snapshot['kfd_holder_count'] == 0 and
             isinstance(pci, list) and len(set(pci)) == len(pci) == 8 and all(PCI.fullmatch(value) for value in pci) and
             isinstance(snapshot['boot_id'], str) and snapshot['boot_id'] and
             type(snapshot['monotonic_ns']) is int and snapshot['monotonic_ns'] > 0,
             'separate HOST PCI/KFD/FD cleanup scope differs')
        snapshots[phase] = snapshot
    before, after = snapshots['pre'], snapshots['post']
    need(before['pci'] == after['pci'] and before['boot_id'] == after['boot_id'] and
         before['monotonic_ns'] < after['monotonic_ns'], 'original before/post device or boot identity differs')
    assigned = None
    if present(root / 'assignment.json'):
        assigned = read_json(root, 'assignment.json', hashes)
        need(assigned['step'] == phases['gpu'][0] and assigned['node'] == NODE and
             assigned['pci'] in before['pci'] and assigned['kfd']['pci'] == assigned['pci'] and
             type(assigned['pid']) is int and assigned['pid'] > 1 and
             type(assigned['startticks']) is int and assigned['startticks'] > 0,
             'original GPU assignment differs from the phase or PCI inventory')
    need(assigned is not None or phase_records['gpu']['passed'] is False,
         'successful original GPU phase requires its assignment receipt')
    original_outcome = None
    if present(root / 'outcome.json'):
        original_outcome = read_json(root, 'outcome.json', hashes)
        need(original_outcome['request'] == REQUEST and original_outcome['publication'] == PUBLICATION and
             original_outcome['stages'] == phase_records and type(original_outcome['passed']) is bool and
             (not original_outcome['passed'] or all(record['passed'] for record in phase_records.values())),
             'original collector outcome differs from this request or phase records')
    return dict(assignment_observed=assigned is not None,
                original_collector_passed=None if original_outcome is None else original_outcome['passed'])


def reconcile(root, raw):
    result = dict(request=REQUEST, publication=PUBLICATION, local_request_SHA256=LOCAL_REQUEST_SHA,
                  parent=PARENT, node=NODE, expected_DCU=1, accounting_columns=COLUMNS.split(','),
                  accounting_SHA256=sha256(raw).hexdigest(), original_receipt_SHA256={}, state='UNKNOWN',
                  gpu_terminal=False, independent_post_clean=False, hold_retirement_evidence=False,
                  replay_allowed=False, training_result_audited=False, new_slurm_submissions=0)
    hashes = result['original_receipt_SHA256']
    try:
        phases = accounting_rows(raw)
        result['phase_rows'] = phases
        if not present(root):
            return dict(result, reason='original_namespace_missing')
        need(root.name == LABEL and root.resolve(strict=True) == root and root.is_dir(), 'original namespace differs')
        intent = read_json(root, 'intent.json', hashes)
        epoch = intent['epoch']
        need((intent['publication'], intent['request'], intent['parent'], intent['node']) ==
             (PUBLICATION, REQUEST, PARENT, NODE) and type(epoch) is float and math.isfinite(epoch) and epoch > 0 and
             intent['deadline'] == epoch + 1800, 'original request, placement or epoch differs')
        result['original_intent_bound'] = True
        gpu = phases['gpu']
        result['gpu_terminal'] = bool(gpu and gpu[2].split()[0] in TERMINAL)
        if not result['gpu_terminal']:
            return dict(result, reason='original_gpu_terminal_unobserved')
        result['state'] = 'ORIGINAL_GPU_TERMINAL_POST_UNVERIFIED'
        for phase in ('pre', 'post'):
            row = phases[phase]
            need(row and row[2:4] == ['COMPLETED', '0:0'], 'actual separate HOST pre/post terminal is missing or failed')
        result.update(verify_cleanup(root, phases, intent, hashes))
    except (OSError, ValueError, KeyError, TypeError) as error:
        return dict(result, reason=type(error).__name__, failed_check=str(error)[:160])
    return dict(result, state='ORIGINAL_TERMINAL_POST_CLEAN', independent_post_clean=True,
                hold_retirement_evidence=True)


def main():
    try:
        raw = subprocess.check_output(ACCOUNTING, timeout=10, env={**os.environ, 'TZ': 'UTC'})
    except (OSError, subprocess.SubprocessError) as error:
        result = reconcile(ROOT, b'')
        result.update(accounting_query_error=type(error).__name__)
    else:
        result = reconcile(ROOT, raw)
    result['observed_epoch'] = time.time()
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
