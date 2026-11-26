"""Refill a released node alongside one identified CPU-only full-hour observer."""

from hashlib import sha256
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
PYTHON = '/usr/local/python3.12/bin/python3'
HEX32 = re.compile(r'[0-9a-f]{32}')
HEX64 = re.compile(r'[0-9a-f]{64}')
PCI = re.compile(r'[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]')
NODE = re.compile(r'f[0-9]{2}r[0-9]+n[0-9]{2}')


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def read(path, expected, cap=65536):
    require(path.is_file() and path.resolve(strict=True) == path.absolute(), 'exact evidence path required')
    with path.open('rb') as stream:
        raw = stream.read(cap + 1)
    require(0 < len(raw) <= cap and sha256(raw).hexdigest() == expected, 'evidence bytes differ')
    return raw


def save(path, obj):
    raw = (json.dumps(obj, sort_keys=True, allow_nan=False) + '\n').encode()
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return sha256(raw).hexdigest()


def query(*argv):
    env = dict(os.environ, TZ='UTC')
    result = subprocess.run(argv, stdin=subprocess.DEVNULL, capture_output=True, check=True, timeout=7, env=env)
    require(len(result.stdout) <= 262144, 'Slurm response exceeds bound')
    return result.stdout.decode().splitlines()


def fields(line):
    return dict(part.split('=', 1) for part in line.split() if '=' in part)


def terminal_predecessor(plan, run=query):
    rows = run('sacct', '-nP', '-j', plan['predecessor'], '-o',
               'JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES,Start,End')
    exact = [row for line in rows if (row := line.split('|'))[0] == plan['predecessor']]
    require(len(exact) == 1 and len(exact[0]) == 9, 'missing exact predecessor accounting')
    [row] = exact
    require(row[1].startswith('training80-dcu-standby-') and row[1].endswith('-' + plan['node'])
            and row[2:4] == ['COMPLETED', '0:0'] and row[5:7] == [plan['node'], 'cpu=8,gres/dcu=8,mem=16G,node=1']
            and row[8] not in ('', 'Unknown'), 'predecessor has not naturally completed on this node')
    return row


def colocated(plan, *, run=query, uid=None):
    parent, node = plan['parent'], plan['node']
    [row] = run('scontrol', 'show', 'job', parent, '--oneliner')
    job = fields(row)
    require(job.get('JobId') == parent and job.get('JobState') == 'RUNNING'
            and job.get('UserId', '').endswith(f'({os.getuid() if uid is None else uid})')
            and 'gres/dcu=320' in job.get('AllocTRES', '').split(','), 'parent ownership or resources differ')
    pool = set(run('scontrol', 'show', 'hostnames', job['NodeList']))
    require(len(pool) == 40 and node in pool, 'target parent roster differs')
    local = {}
    lines = run('scontrol', 'show', 'step', parent, '--oneliner')
    require(lines, 'step inventory unavailable')
    for line in lines:
        row = fields(line)
        step = row['StepId']
        if step in (parent + '.batch', parent + '.extern'):
            continue
        require(re.fullmatch(re.escape(parent) + r'\.[0-9]+', step), 'unknown step identity')
        expression = row['NodeList']
        members = [expression] if NODE.fullmatch(expression) else run('scontrol', 'show', 'hostnames', expression)
        require(len(members) == len(set(members)) == int(row['Nodes']) and set(members) <= pool,
                'step placement differs')
        if node not in members:
            continue
        # Every other colocated controller-visible record blocks this specialized route.
        require(step == plan['observer_step'] and step not in local and row['State'] == 'RUNNING'
                and row.get('Name') == f"training80-long-busy-{plan['observer_request']}-{parent}"
                and set(members) == pool and row.get('CPUs') == row.get('Tasks') == '40'
                and row.get('TRES') == 'cpu=40,mem=10G,node=40', 'non-observer or overlapping step on target')
        local[step] = {'name': row['Name'], 'TRES': row['TRES'], 'nodes': len(members)}
    require(set(local) == {plan['observer_step']}, 'bound zero-GRES observer is no longer live')
    return local


def validate(plan):
    require(plan['parent'] in {'1103606', '1194900'} and NODE.fullmatch(plan['node'])
            and HEX32.fullmatch(plan['request']) and HEX32.fullmatch(plan['observer_request'])
            and plan['request'] != plan['observer_request'] and HEX64.fullmatch(plan['ready_SHA256']), 'bad request scope')
    pattern = re.escape(plan['parent']) + r'\.[0-9]+'
    require(re.fullmatch(pattern, plan['predecessor']) and re.fullmatch(pattern, plan['observer_step'])
            and plan['predecessor'] != plan['observer_step'] and plan['seconds'] == 10800,
            'invalid bounded resource scope')
    pcis = plan['pci']
    require(len(pcis) == len(set(pcis)) == 8 and all(PCI.fullmatch(pci) for pci in pcis), 'invalid PCI pins')
    expiry = datetime.fromisoformat(plan['submit_by_UTC'])
    require(expiry.utcoffset() is not None, 'submission deadline needs a timezone')
    return expiry.timestamp()


def probe(plan, staged_collector):
    request = plan['request']
    argv = ['srun', '--threads=1', '--jobid=' + plan['parent'], '--job-name=training80-coexist-probe-' + request[:12],
            '--exact', '--immediate=5', '--nodes=1', '--ntasks=1', '--nodelist=' + plan['node'], '--cpus-per-task=1',
            '--mem=256M', '--gres=none', '--time=00:01:00', '--kill-on-bad-exit=1', PYTHON, '-B',
            str(staged_collector), 'node', request, plan['parent'], '[]']
    env = {key: value for key, value in os.environ.items() if key not in {'SLURM_OVERLAP', 'SRUN_OVERLAP'}}
    result = subprocess.run(argv, stdin=subprocess.DEVNULL, capture_output=True, check=True, timeout=12, env=env)
    require(0 < len(result.stdout) <= 4096, 'missing bounded CPU device probe')
    row = json.loads(result.stdout)
    require(row['parent'] == plan['parent'] and row['request_id'] == request and row['node'] == plan['node']
            and row['observer_uid'] == os.getuid() and re.fullmatch(re.escape(plan['parent']) + r'\.[0-9]+', row['step'])
            and 'probe_error' not in row and sorted(row['PCIs']) == sorted(plan['pci'])
            and type(row['kfd_process_count']) is int and type(row['own_device_fd_process_count']) is int
            and row['kfd_process_count'] == row['own_device_fd_process_count'] == 0,
            'physical device admission failed')
    return row, sha256(result.stdout).hexdigest()


def execute(plan, here, publication):
    import fcntl

    expiry = validate(plan)
    require(re.fullmatch(r'[0-9a-f]{40}', publication) and here.name == 'training80-observer-refill-' + plan['request'],
            'wrong published scope')
    output = ROOT / 'results' / ('training80-long-busy-' + plan['observer_request'])
    ready = json.loads(read(output / 'ready.json', plan['ready_SHA256']))
    require(ready['request'] == plan['observer_request'] and ready['ready_nodes'] == 80
            and ready['actual_READY_steps'][plan['parent']] == plan['observer_step'], 'observer receipt differs')
    collector_root = ROOT / 'results' / ('training80-replacement-admission-' + plan['collector_stage'])
    require(HEX32.fullmatch(plan['collector_stage']), 'invalid collector stage identity')
    for name, pin in plan['collector_SHA256'].items():
        require(re.fullmatch(r'[a-z0-9_]+\.py', name), 'invalid collector member')
        read(collector_root / name, pin)
    controller = read(here / 'controller.py', plan['controller_SHA256'])
    worker = read(here / 'worker.py', plan['worker_SHA256'])
    implementation = {'__name__': 'pinned_standby_controller'}
    exec(compile(controller, str(here / 'controller.py'), 'exec'), implementation)
    target = ROOT / 'results' / ('training80-dcu-standby-' + plan['request'])
    with (ROOT / 'results/.training80-dcu-standby.lock').open('ab') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(time.time() < expiry, 'original submission deadline passed')
        require(not target.exists() and not target.is_symlink(), 'original standby result exists')
        save(here / 'INTENT.json', {'request': plan['request'], 'publication': publication, 'replay_allowed': False})
        predecessor = terminal_predecessor(plan)
        before = colocated(plan)
        checked, raw_sha = probe(plan, collector_root / 'collect_training80_dispatch_snapshot.py')
        require(colocated(plan) == before, 'placement changed after physical probe')
        require(time.time() < expiry, 'original submission deadline passed after probe')
        admission = save(here / 'admission.json', {'terminal_predecessor': predecessor, 'permitted_observer': before,
                         'device_probe': checked, 'raw_probe_SHA256': raw_sha, 'epoch': time.time()})
        target.mkdir(mode=0o700)
        implementation['write'](target, 'intent.json', {'request': plan['request'], 'publication': publication,
                                'synthetic': True, 'nodes': [plan['node']], 'seconds': plan['seconds'],
                                'epoch': time.time(), 'admission_SHA256': admission, 'replay_allowed': False})
        with (target / 'worker.py').open('xb') as stream:
            stream.write(worker)
        result = implementation['spawn'](target, plan['node'], {'parent': plan['parent'], 'pci': plan['pci']},
                                         plan['seconds'], plan['request'])
        response = {'request': plan['request'], 'publication': publication, 'synthetic': True,
                    'submit_records': [result], 'launch_failures': [], 'card_steps_observed': 0,
                    'admission_SHA256': admission, 'observer_step': plan['observer_step']}
        implementation['write'](target, 'submission-receipt.json', response)
        print(json.dumps(response, sort_keys=True))


if __name__ == '__main__':
    stage = Path(__file__).resolve().parent
    require(sys.flags.isolated and sys.flags.dont_write_bytecode and len(sys.argv) == 3, 'isolated once launch required')
    os.umask(0o077)
    execute(json.loads(read(stage / 'claim.json', sys.argv[1])), stage, sys.argv[2])
