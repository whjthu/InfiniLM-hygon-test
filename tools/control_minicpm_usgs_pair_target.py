"""Run the published MiniCPM domain pair once in its owned Slurm allocation."""

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import time

from prepare_minicpm_usgs_pair_target import (CORE_ASSETS_SHA, CORE_SHA, COLUMNS, LABEL, NODE, PARENT,
                                             PORT, REPO, REQUEST, RESULTS, VIEW_SHA, need, small, write)
from read_training_slurm_membership import allocated_dcu, snapshot as membership


RUN = RESULTS / LABEL
HOST_PYTHON = '/usr/local/python3.12/bin/python3'
WORKER_SRC = REPO / 'code/bw1000' / LABEL
ACCOUNTING_HELPER = REPO / 'code/bw1000/training-qwen-native25-hip-v3'
PHASES = {'pre': (0, 2, 2048, 60), 'gpu': (8, 64, 224000, 2100), 'post': (0, 2, 2048, 90)}
WHOLE, NATIVE, WORKER_CLEANUP = 2400, 1980, 2080
SOURCE_MANIFEST = 'coordination/training/minicpm-usgs-target-control-files-v1.json'
REQUEST_PATTERN = re.compile(PARENT + r'\.\d+')
WORKER_NAMES = ('worker.py', 'snapshot.py', 'owned_processes.py', 'CONFIG_SCHEMA.json', 'CLAIM.json',
                'README.zh-CN.md', 'check_worker_CPU.py', 'SOURCE_PROOF.json', 'SOURCE_FILES.json')
PUBLISHED_FILES = (
    'tools/control_minicpm_usgs_pair_target.py', 'tools/prepare_minicpm_usgs_pair_target.py',
    'tools/read_training_slurm_membership.py', 'coordination/training/training80-active-node-task-map-v1.json',
    'code/bw1000/training-qwen-native25-hip-v3/slurm_accounting.py',
    *((WORKER_SRC / name).relative_to(REPO).as_posix() for name in WORKER_NAMES),
)


def pin(path, raw):
    return dict(path=str(path), bytes=len(raw), SHA256=sha256(raw).hexdigest())


def accepted_stage(root, expected):
    marker = root / 'asset-stage-complete.json'
    raw = small(marker, expected)
    info = json.loads(raw)
    need(info['state'] == 'MINICPM_PAIR_TARGET_ASSETS_STAGED' and info['request'] == REQUEST
         and (info['parent'], info['node'], info['port']) == (PARENT, NODE, PORT)
         and info['actual_GPU_steps'] == info['actual_optimizer_updates'] == 0
         and info['source_rows'] == 88 and info['train_media'] == 22
         and info['source_core_manifest_SHA256'] == CORE_SHA and info['asset_binding_SHA256'] == CORE_ASSETS_SHA,
         'original staged scope differs')
    def member(ref, relative):
        path = root / relative
        raw = small(path, ref['SHA256'])
        actual = pin(path, raw)
        need(actual == ref if 'path' in ref else {k: actual[k] for k in ('bytes', 'SHA256')} == ref,
             'original staged member differs')
        return actual
    plans = {arm: member(info['plans'][arm], arm + '-plan.json') for arm in ('continuous', 'resumed')}
    view = member(info['view']['view'], 'view/training800.jsonl')
    need(view['SHA256'] == VIEW_SHA, 'original exposure view differs')
    return dict(asset_stage=pin(marker, raw), native88_independent=member(info['native88_receipt'], 'native88-independent.json'),
                asset_binding=pin(root / 'src/ASSETS.json', small(root / 'src/ASSETS.json', CORE_ASSETS_SHA)),
                plans=plans, view=view, ledger=member(info['view']['ledger'], 'view/exposure-ledger-private.json'))


def worker_config(epoch, artifacts, before):
    need(type(epoch) in (int, float) and math.isfinite(epoch) and epoch > 0, 'invalid original epoch')
    return dict(schema='MINICPM_USGS_TARGET_WORKER_CONFIG_V1', request=REQUEST, run_root=str(RUN),
                core_directory=str(RUN / 'src'), framework_directory=str(RUN / 'framework'),
                parent=PARENT, node=NODE, port=PORT, core_manifest_SHA256=CORE_SHA,
                epoch=epoch, deadline=epoch+WHOLE, native_deadline=epoch+NATIVE,
                root_GO=True, training_GO=True, GPU_GO=True, before=before, **artifacts)


def accounting(label):
    result = subprocess.check_output(['sacct', '-n', '-P', '-j', PARENT, '-S', '2026-11-19T00:00:00', '-o', COLUMNS],
                                     text=True, timeout=8)
    rows = [line.split('|') for line in result.splitlines() if line.strip()]
    need(all(len(row) == 7 for row in rows), 'malformed original Slurm accounting')
    rows = [row for row in rows if row[1] == label]
    need(len(rows) <= 1, 'original Slurm label is duplicated')
    return rows[0] if rows else None


def step_command(phase, command):
    devices, cpus, memory, seconds = PHASES[phase]
    return ['srun', '--jobid=' + PARENT, '--exact', '--immediate=5', '--cpu-bind=none', '-N1', '-n1',
            '-c' + str(cpus), f'--mem={memory}M', '--gres=' + (f'dcu:{devices}' if devices else 'none'),
            '-w', NODE, '-t', str(math.ceil(seconds / 60)), '--job-name=' + LABEL + '-' + phase, *command]


def execute_step(phase, command, elapsed, epoch):
    sys.path.insert(0, str(ACCOUNTING_HELPER))
    from slurm_accounting import wait_terminal
    label = LABEL + '-' + phase
    need(accounting(label) is None, 'original Slurm label is already present')
    used = elapsed()
    cap = PHASES[phase][3] + 5
    if phase == 'gpu':
        need(used < NATIVE - 60, 'insufficient original native budget')
        cap = min(cap, PHASES[phase][3] - used)
    else:
        need(used + cap + 26 < WHOLE, 'insufficient CPU step and accounting budget')
    result = dict(phase=phase, rc=None, terminal_visible=False, row=None, passed=False, observations=[])
    try:
        with (RUN / (phase + '.log')).open('xb') as log:
            try:
                result['rc'] = subprocess.run(step_command(phase, command), stdout=log, stderr=subprocess.STDOUT,
                                              timeout=cap).returncode
            except subprocess.TimeoutExpired:
                result['client_timeout'] = True
        deadline = time.monotonic() + min(24, max(0, WHOLE - elapsed() - 9))
        result.update(wait_terminal(lambda: accounting(label), deadline, max_observations=8))
        row = result['row']
        result['passed'] = bool(result['rc'] == 0 and row and REQUEST_PATTERN.fullmatch(row[0])
             and row[2:4] == ['COMPLETED', '0:0'] and row[5] == NODE and allocated_dcu(row[6]) == PHASES[phase][0])
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        result['observation_error'] = type(error).__name__
        raise
    finally:
        write(RUN / (phase + '-accounting.json'), dict(epoch=epoch, **result))
    return result


def proc_exists(pid, startticks):
    need(type(pid) is int and pid > 1 and type(startticks) is int and startticks > 0, 'invalid original process identity')
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()
    except FileNotFoundError:
        return False
    return fields[0] != 'Z' and int(fields[19]) == startticks


def cpu_snapshot(phase):
    sys.path.insert(0, str(RUN / 'target-worker'))
    from snapshot import snapshot
    record = snapshot(False)
    need(record['kfd_holder_count'] == 0 and record['full_fd_inventory_performed'], 'active accelerator holder')
    with socket.socket() as port:
        port.bind(('127.0.0.1', PORT))
    record['master_port'] = PORT
    record['owned_processes_remaining'] = []
    record['owned_registry_complete'] = None
    if phase == 'post':
        def original(name):
            path = RUN / name
            return json.loads(small(path)) if path.exists() else None
        owner = original('worker-owner.json')
        outcome = original('worker-outcome.json')
        if owner:
            need(owner['request'] == REQUEST and owner['boot_id'] == record['boot_id'], 'original GPU process identity differs')
            if proc_exists(owner['PID'], owner['startticks']):
                record['owned_processes_remaining'].append(owner['PID'])
        observed_arms = set()
        if outcome:
            need(outcome['request'] == REQUEST, 'original worker outcome differs')
            for arm in outcome['arms']:
                observed_arms.add(arm['arm'])
                for entry in arm['owners'].values():
                    if proc_exists(entry['PID'], entry['startticks']):
                        record['owned_processes_remaining'].append(entry['PID'])
        attempted = {arm for arm in ('continuous', 'resumed') if (RUN / arm / 'native-intent.json').is_file()}
        record['owned_registry_complete'] = bool(owner and outcome and attempted <= observed_arms
                                                  and all(arm['cleanup_known'] for arm in outcome['arms']))
    print(json.dumps(record), flush=True)


def gpu_command(config, epoch):
    r = RESULTS.parent
    runtime = ['/public/software/apps/apptainer/1.3.4/bin/apptainer', 'exec', '--rocm', '--no-home', '--pwd', '/']
    binds = ((RUN, RUN, 'rw'), (RESULTS / 'training-minicpm-original-model-v1', None, 'ro'),
             (RESULTS / 'training-minicpm-native-mask-assets-v1', None, 'ro'),
             (RESULTS / 'training-vlm-original-assets-v1', None, 'ro'),
             (RESULTS / 'training-domain-usgs-table-reading-v1/data/train', None, 'ro'),
             (RESULTS / 'training-vlm-swift-offline-runtime-v1', '/swift-runtime', 'ro'),
             (RESULTS / 'training-qwen-vision014-CPU-v2/vision', '/vision', 'ro'),
             (RESULTS / 'training-minicpm-hygon-processor-r2/cpu/runtime-site', '/minicpm-transformers', 'ro'),
             (Path('/opt/hyhal'), None, 'ro'))
    for source, target, mode in binds:
        runtime.extend(['--bind', f'{source}:{source if target is None else target}:{mode}'])
    pythonpath = '/vision:/minicpm-transformers:/swift-runtime/site:' + str(RUN / 'framework') + ':' + str(RUN / 'src')
    worker = shlex.join(['python3', '-B', '-u', str(RUN / 'target-worker/worker.py'), '--config', config['path'],
                         '--config-sha256', config['SHA256']])
    remaining = shlex.quote(f'import time; print(max(0, int({epoch + WORKER_CLEANUP!r} - time.time())))')
    shell = ('set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh; '
             'export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 '
             'OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1; '
             'export PYTHONPATH=' + shlex.quote(pythonpath) + '; '
             'LEFT=$(python3 -B -c ' + remaining + '); test "$LEFT" -gt 30; '
             'exec /usr/bin/timeout --kill-after=10s "$LEFT" ' + worker)
    return [*runtime, str(r / 'containers/dtk26-sandbox-r1'), '/bin/bash', '--noprofile', '--norc', '-c', shell]


def original_membership():
    ledger = json.loads(small(REPO / 'coordination/training/training80-active-node-task-map-v1.json'))
    groups = {parent: {name: group['allocations'][parent] for name, group in ledger['groups'].items()}
              for parent in ('1087656', '1103606')}
    live = membership(groups, owner_uid=os.getuid())
    position = live['parents'][PARENT]['coverage_by_group_and_node']['T05'][NODE]
    need(live['membership_qualified'] and position['state'] == 'NO_NUMBERED_STEP', 'expected a free original owner node')


def stage_worker(source_sha):
    manifest = json.loads(small(REPO / SOURCE_MANIFEST, source_sha))
    required = {(WORKER_SRC / name).relative_to(REPO).as_posix() for name in WORKER_NAMES}
    need(set(manifest) == set(PUBLISHED_FILES), 'source manifest does not close the exact controller and runtime worker')
    worker_files = {}
    for name, wanted in manifest.items():
        path = REPO / name
        need(path.is_relative_to(REPO) and '..' not in path.parts, 'published source escaped the checkout')
        content = small(path, wanted)
        if name in required:
            worker_files[path.name] = content
    return worker_files


def controller(stage_sha, source_sha, publication):
    need(re.fullmatch('[0-9a-f]{40}', publication or '') and re.fullmatch('[0-9a-f]{64}', stage_sha or '')
         and re.fullmatch('[0-9a-f]{64}', source_sha or ''), 'exact publication and source/stage pins required')
    artifacts = accepted_stage(RUN, stage_sha)
    worker = stage_worker(source_sha)
    original_membership()
    need(all(accounting(LABEL + '-' + phase) is None for phase in PHASES), 'original Slurm request already present')
    epoch, mono = time.time(), time.monotonic()
    elapsed = lambda: max(time.time() - epoch, time.monotonic() - mono)
    write(RUN / 'training-intent.json', dict(request=REQUEST, publication=publication, epoch=epoch,
          deadline=epoch+WHOLE, native_deadline=epoch+NATIVE, GPU_client_deadline=epoch+PHASES['gpu'][3],
          asset_stage_SHA256=stage_sha, source_manifest_SHA256=source_sha, node=NODE, parent=PARENT, replay_allowed=False))
    phases, clean, completed, error = {}, False, False, None
    try:
        folder = RUN / 'target-worker'
        folder.mkdir(mode=0o700)
        for name, raw in worker.items():
            write(folder / name, raw)
        snapshot = [HOST_PYTHON, '-B', str(Path(__file__).resolve()), '--snapshot']
        phases['pre'] = execute_step('pre', [*snapshot, 'pre'], elapsed, epoch)
        if phases['pre']['passed']:
            before = json.loads(small(RUN / 'pre.log'))
            need(before['step'] == phases['pre']['row'][0] and before['node'] == NODE, 'preflight accounting identity differs')
            bound = write(RUN / 'before.json', before)
            cfg = write(RUN / 'worker-config.json', worker_config(epoch, artifacts, bound))
            phases['gpu'] = execute_step('gpu', gpu_command(cfg, epoch), elapsed, epoch)
            if phases['gpu']['terminal_visible']:
                phases['post'] = execute_step('post', [*snapshot, 'post'], elapsed, epoch)
                if phases['post']['passed']:
                    after = json.loads(small(RUN / 'post.log'))
                    write(RUN / 'after.json', after)
                    clean = (after['step'] == phases['post']['row'][0] and after['node'] == NODE
                        and all(after[k] == before[k] for k in ('boot_id', 'pci', 'pci_to_kfd', 'master_port'))
                        and not after['owned_processes_remaining'] and after['owned_registry_complete'] is True)
        if all(phases.get(p, {}).get('passed') for p in PHASES) and clean:
            assigned = json.loads(small(RUN / 'worker-assigned.json'))
            outcome = json.loads(small(RUN / 'worker-outcome.json'))
            recovery = json.loads(small(RUN / 'paired-recovery.json'))
            completed = (assigned['actual_step'] == phases['gpu']['row'][0] and outcome['request'] == REQUEST
                and outcome['epoch'] == epoch and outcome['passed'] is True and recovery['exact_paired_recovery'] is True)
    except (OSError, TimeoutError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        error = type(exc).__name__
        raise
    finally:
        result = dict(request=REQUEST, publication=publication, epoch=epoch, phases=phases, clean_post=clean,
                      completed=completed, error=error, independently_audited_training=False,
                      exact_native_training_claimed=False, formal17_qualified=False)
        write(RUN / 'training-outcome.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', choices=('pre', 'post'))
    for arg in ('stage-sha', 'source-sha', 'publication'):
        parser.add_argument('--' + arg)
    args = parser.parse_args()
    if args.snapshot:
        cpu_snapshot(args.snapshot)
        return 0
    result = controller(args.stage_sha, args.source_sha, args.publication)
    print(json.dumps(result), flush=True)
    return int(not result['completed'])


if __name__ == '__main__':
    raise SystemExit(main())
