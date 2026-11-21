"""Admit and reconcile exactly one gfx936 build of the two inverse RoPE variants."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time

from read_training_slurm_membership import allocated_dcu, snapshot


REPO = Path(__file__).resolve().parents[1]
ROOT = Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
LABEL = 'Deep-inverse-output-rope-gfx936-CPU-v1'
PARENT, NODE, ZONE = '1103606', 'f12r2n12', 'O01'
PACKAGE = Path('code/bw1000/Deep-C4-causal-consumer-r483/inverse-output-rope-CPU-v1')
MANIFEST = 'fb0edd19695838a6df632e88ae10351569362de28ed11067848bb909b3aa7465'
CLAIM = Path('coordination/tasks/deep-inverse-output-rope-gfx936-host-compile-v1.json')
LEDGER = Path('coordination/training/training80-active-node-task-map-v1.json')
ACCOUNTING = Path('code/bw1000/training-qwen-usgs-exposure25-resume20-hip-v1/slurm_accounting.py')
COLUMNS = ('JobIDRaw', 'JobName%100', 'State', 'ExitCode', 'ElapsedRaw', 'NodeList', 'AllocTRES')
REQUEST = sha256(f'{LABEL}:{PARENT}:{NODE}:{MANIFEST}'.encode()).hexdigest()[:32]
EXPORTS = ['deep_inverse_rope_h1_v1', 'deep_inverse_rope_h2_v1']


def need(condition, message):
    if not condition:
        raise ValueError(message)


def pin(path, wanted, limit=2 * 1024 * 1024):
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    need(0 < len(raw) <= limit and sha256(raw).hexdigest() == wanted, 'frozen bytes differ: ' + path.name)
    return raw


def verify_claim(wanted):
    claim = json.loads(pin(REPO / CLAIM, wanted))
    need(claim['request'] == REQUEST and claim['label'] == LABEL and claim['CPU_GO'] is True
         and claim['GPU_GO'] is False and claim['placement'] == {'parent': PARENT, 'node': NODE, 'zone': ZONE,
         'cpus': 4, 'memory_MiB': 8192, 'DCU': 0} and claim['wall_seconds'] == 300, 'wrong frozen compile scope')
    files = claim['host_files']
    need(set(files) == {'tools/control_deep_inverse_rope_compile.py', 'tools/read_training_slurm_membership.py',
                        LEDGER.as_posix(), ACCOUNTING.as_posix()}, 'unexpected HOST source set')
    for name, digest in files.items():
        pin(REPO / name, digest)
    need(claim['package_manifest_SHA256'] == MANIFEST, 'wrong frozen package')
    members = json.loads(pin(REPO / PACKAGE / 'SOURCE_FILES.json', MANIFEST))['files_SHA256']
    need(len(members) == 8 and all(Path(name).name == name for name in members), 'wrong package member set')
    for name, digest in members.items():
        pin(REPO / PACKAGE / name, digest)
    return claim


def check_slot():
    ledger = json.loads((REPO / LEDGER).read_bytes())
    parents = {job['JobId'] for job in ledger['jobs']}
    groups = {parent: {zone: group['allocations'][parent] for zone, group in ledger['groups'].items()}
              for parent in parents}
    report = snapshot(groups, owner_uid=os.getuid())
    node = report['parents'][PARENT]['coverage_by_group_and_node'][ZONE][NODE]
    need(report['membership_qualified'] and node['state'] == 'NO_NUMBERED_STEP' and
         node['numbered_step_assigned_DCU_bounds'] == [0, 0], 'original O01 node unavailable for compile')


def accounting_row(raw):
    need(len(raw) <= 2*1024**2, 'accounting data exceeds bound')
    fields = [line.split('|') for line in raw.decode().splitlines() if line]
    need(all(len(row) == len(COLUMNS) for row in fields), 'unexpected accounting columns')
    rows = [row for row in fields if row[1] == LABEL]
    need(len(rows) <= 1, 'original build label has multiple accounting rows')
    row = rows[0] if rows else None
    need(row is None or (re.fullmatch(re.escape(PARENT) + r'\.[0-9]+', row[0]) and row[2].strip() and row[5] == NODE),
         'original build label resolved to a different step or node')
    return row


def accounting(timeout):
    return accounting_row(subprocess.check_output(['sacct', '-n', '-P', '-j', PARENT,
        '-S', '2026-11-20T00:00:00', '-o', ','.join(COLUMNS)], timeout=timeout))


def command(epoch):
    # The worker creates its output directory exclusively under the mounted candidates root.
    package = f'/refactor/incoming/{LABEL}'
    env = {'TUNING_CPU_PARENT': PARENT, 'TUNING_CPU_NODE': NODE, 'TUNING_CPU_EPOCH': str(epoch),
           'TUNING_CPU_DEADLINE': str(epoch + 300), 'TUNING_CPU_MANIFEST_SHA': MANIFEST,
           'CUDA_VISIBLE_DEVICES': '-1', 'HIP_VISIBLE_DEVICES': '-1', 'ROCR_VISIBLE_DEVICES': '-1', 'LC_ALL': 'C'}
    inner = ['env', *(f'{key}={value}' for key, value in env.items()), '/usr/bin/python3', '-I', '-S', '-B', '-u',
             f'{package}/compile_once.py']
    shell = ('set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh; '
             'unset LD_PRELOAD LD_AUDIT GCC_EXEC_PREFIX COMPILER_PATH LIBRARY_PATH CPATH CPLUS_INCLUDE_PATH C_INCLUDE_PATH; '
             'exec ' + shlex.join(inner))
    producer = 'Deep-C4-constant-slots-r482-Native-CPU-v1'
    binds = [(ROOT, '/refactor', 'ro'), (ROOT / 'candidates', '/refactor/candidates', 'rw'),
             (ROOT / 'candidates' / producer, '/refactor/candidates/' + producer, 'ro'),
             (REPO / PACKAGE, package, 'ro'), (Path('/opt/hyhal'), '/opt/hyhal', 'ro')]
    container = ['/public/software/apps/apptainer/1.3.4/bin/apptainer', 'exec', '--no-home', '--pwd', '/']
    for src, dest, mode in binds:
        container.extend(['--bind', f'{src}:{dest}:{mode}'])
    return ['srun', '--jobid=' + PARENT, '--exact', '--immediate=5', '--nodes=1', '--ntasks=1', '--cpus-per-task=4',
            '--mem=8192M', '--gres=none', '--nodelist=' + NODE, '--time=00:05:00', '--job-name=' + LABEL,
            *container, str(ROOT / 'containers/dtk26-sandbox-r1'), '/bin/bash', '--noprofile', '--norc', '-c', shell]


def save_once(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def check_result(path, row, epoch):
    if not row or row[2:4] != ['COMPLETED', '0:0'] or allocated_dcu(row[6]) != 0:
        return None
    resources = dict(part.split('=', 1) for part in row[6].split(','))
    need(resources.get('cpu') == '4' and resources.get('mem') in {'8G', '8192M'} and
         row[4].isdecimal() and int(row[4]) <= 300, 'original CPU accounting differs')
    with (path / 'CPU-result.json').open('rb') as stream:
        raw = stream.read(1024**2 + 1)
    need(0 < len(raw) <= 1024**2, 'original compiler receipt exceeds bound')
    result = json.loads(raw)
    expected = dict(label=LABEL, original_step=row[0], node=NODE, manifest_SHA256=MANIFEST,
                    epoch=epoch, deadline=epoch+300, GRES='none', planned_device_TUs=1, actual_device_TUs=1,
                    actual_GPU_steps=0, explicit_Runtime_calls=0, exports=EXPORTS)
    need(all(type(result.get(key)) is type(value) and result[key] == value for key, value in expected.items())
         and result.get('passed') is True and result.get('completed') is True
         and result.get('device_qualified') is False, 'original compiler receipt differs')
    pin(path / 'inverse_output_rope.o', result['object_SHA256'], limit=16*1024**2)
    pin(path / 'inverse_output_rope.before.d', result['fresh_M_SHA256'])
    pin(path / 'inverse_output_rope.d', result['actual_MD_SHA256'])
    return dict(receipt_SHA256=sha256(raw).hexdigest(), object_SHA256=result['object_SHA256'])


def controller(run_path, output, claim_sha, wait_terminal, *, query=accounting, observe_slot=check_slot,
               run=subprocess.run, wall=time.time, monotonic=time.monotonic):
    need(not run_path.exists() and not output.exists(), 'original compile namespace already exists')
    observe_slot()
    need(query(8) is None, 'original compile already submitted')
    run_path.mkdir(mode=0o700)
    epoch, started = wall(), monotonic()
    save_once(run_path / 'intent.json', dict(request=REQUEST, source_manifest_SHA256=MANIFEST, claim_SHA256=claim_sha,
              parent=PARENT, node=NODE, epoch=epoch, deadline=epoch+300, replay_allowed=False))
    result = dict(request=REQUEST, state='UNKNOWN', actual_GPU_steps=0, accounting_columns=list(COLUMNS))

    def left():
        return min(300 - (monotonic() - started), epoch + 300 - wall())

    def read_accounting():
        remaining = left()
        if remaining <= 2:
            raise TimeoutError('original accounting deadline elapsed')
        return query(min(8, remaining - 1))

    try:
        remaining = left() - 25
        need(remaining > 5, 'original compile budget elapsed before submission')
        with (run_path / 'srun.log').open('xb') as stream:
            try:
                result['client_rc'] = run(command(epoch), stdin=subprocess.DEVNULL, stdout=stream,
                                          stderr=subprocess.STDOUT, timeout=remaining, check=False).returncode
            except subprocess.TimeoutExpired:
                result['client_timeout'] = True
        terminal = wait_terminal(read_accounting, monotonic() + min(20, left()-2), max_observations=6)
        result.update(terminal)
        if terminal['terminal_visible']:
            result['state'] = 'TERMINAL_UNVERIFIED'
            verified = check_result(output, terminal['row'], epoch)
            if verified is not None:
                result.update(verified, state='COMPILE_VERIFIED')
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        result['error'] = f'{type(error).__name__}: {str(error)[:200]}'
    finally:
        save_once(run_path / 'accounting-result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--claim-sha256', required=True)
    args = parser.parse_args()
    claim = verify_claim(args.claim_sha256)
    # Verify these exact bytes before importing the existing bounded helper.
    namespace = {'__name__': 'frozen_terminal_accounting'}
    helper = REPO / ACCOUNTING
    raw = pin(helper, claim['host_files'][ACCOUNTING.as_posix()])
    exec(compile(raw, str(helper), 'exec'), namespace)
    result = controller(ROOT / 'results' / (LABEL + '-controller'), ROOT / 'candidates' / LABEL,
                        args.claim_sha256, namespace['wait_terminal'])
    print(json.dumps({key: result.get(key) for key in ('request', 'state', 'row', 'object_SHA256', 'error')},
                     allow_nan=False), flush=True)
    return int(result['state'] != 'COMPILE_VERIFIED')


if __name__ == '__main__':
    raise SystemExit(main())
