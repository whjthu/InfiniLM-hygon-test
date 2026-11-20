"""Validate one source packet and launch its assigned CPU-only Slurm step."""

import base64
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import socket
import subprocess
import time


ROOT = Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
LABEL = 'DeepC4-T1-history-guarded-probe-link-v1'
PARENT = '1103606'
NODE = 'f13r2n01'
REQUEST = 'ae85c97293ae410483e55d94383a6c48'
UPSTREAM_STEPS = {'1103606.156', '1103606.487'}


def need(condition, error):
    if not condition:
        raise ValueError(error)


def relative(name):
    path = PurePosixPath(name)
    need(bool(path.parts) and not path.is_absolute() and path.as_posix() == name
         and '..' not in path.parts and '\\' not in name and ':' not in name,
         'noncanonical staged source path')
    return path


def decode(payload):
    need(payload['request'] == REQUEST and re.fullmatch('[0-9a-f]{40}', payload['publication']),
         'wrong frozen original request or publication')
    files = {name: base64.b64decode(raw, validate=True) for name, raw in payload['files'].items()}
    mf = files['SOURCE_FILES.json']
    need(sha256(mf).hexdigest() == payload['manifest_SHA256'], 'wrong source manifest bytes')
    scope = json.loads(mf)
    need((scope['label'], scope['parent'], scope['node'], scope['request']) == (LABEL, PARENT, NODE, REQUEST),
         'wrong CPU relink placement')
    pins = scope['files']
    need(set(files) == set(pins) | {'SOURCE_FILES.json'} and sum(map(len, files.values())) < 90000,
         'wrong bounded source member set')
    for name, pin in pins.items():
        relative(name)
        need(sha256(files[name]).hexdigest() == pin['SHA256'], 'staged source hash mismatch')
    return files


def check_slurm(query, uid):
    def fields(row):
        return dict(re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S+)', row))

    line, = query('scontrol', 'show', 'job', PARENT, '-o')
    parent = fields(line)
    need(parent['JobState'] == 'RUNNING' and parent['UserId'].endswith(f'({uid})')
         and parent['NumNodes'] == '40' and 'gres/dcu=320' in parent['AllocTRES'].split(',')
         and NODE in query('scontrol', 'show', 'hostnames', parent['NodeList']), 'parent ownership or placement differs')
    steps = query('scontrol', 'show', 'step', PARENT, '-o')
    need(steps, 'current step visibility missing')
    for row in map(fields, steps):
        step_id = row.get('StepId')
        if step_id in (PARENT + '.extern', PARENT + '.batch'):
            continue
        node_list = row.get('NodeList', '')
        need(step_id and re.fullmatch(PARENT + r'\.[0-9]+', step_id) and node_list not in ('', '(null)')
             and NODE not in query('scontrol', 'show', 'hostnames', node_list), 'active or unknown target step')
    for previous in sorted(UPSTREAM_STEPS):
        rows = [row.split('|') for row in query('sacct', '-nP', '-j', previous, '-o', 'JobIDRaw,State,ExitCode')]
        exact = [row for row in rows if row[0] == previous]
        need(exact == [[previous, 'COMPLETED', '0:0']], 'upstream target build terminal did not match')
    for parent_id in ('1087656', PARENT):
        rows = query('sacct', '-nP', '-j', parent_id, '-o', 'JobIDRaw,JobName%100')
        need(all(row.split('|')[1:2] != [LABEL] for row in rows), 'relink label already exists')


def run(payload):
    epoch = time.time()
    deadline = epoch + 300
    files = decode(payload)

    def query(*argv):
        raw = subprocess.check_output(argv, text=True, timeout=8)
        need(len(raw) <= 400000, 'oversized Slurm query')
        return raw.splitlines()

    package = ROOT / 'incoming' / LABEL
    destination = ROOT / 'candidates' / LABEL
    intent = ROOT / 'results' / (LABEL + '-intent.json')
    launch = ROOT / 'results' / (LABEL + '-launch.json')
    srun_log = ROOT / 'logs' / (LABEL + '-srun.log')
    need(not any(path.exists() for path in (package, destination, intent, launch, srun_log)),
         'relink namespace exists; reconcile original')
    check_slurm(query, os.getuid())
    package.mkdir(mode=0o700)
    for name, raw in files.items():
        path = package.joinpath(*relative(name).parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as handle:
            handle.write(raw)
    need(time.time() < deadline - 90, 'original launch budget too short')
    argv = ['srun', '--jobid=' + PARENT, '--exact', '--immediate=5', '--cpu-bind=none', '-N1', '-n1', '-c4',
            '--mem=8192M', '--gres=none', '-w', NODE, '-t', '00:05:00', '--job-name=' + LABEL,
            '--output=' + str(ROOT / 'logs' / (LABEL + '-%J.log')), 'env',
            'TUNING_CPU_MANIFEST_SHA=' + payload['manifest_SHA256'], 'TUNING_CPU_REQUEST=' + REQUEST,
            'TUNING_CPU_PUBLICATION=' + payload['publication'], 'TUNING_CPU_EPOCH=' + str(epoch),
            'TUNING_CPU_DEADLINE=' + str(deadline), 'WAVE_INSIDE_R2323=' + LABEL + '/CPU_inside.sh',
            '/bin/bash', str(package / 'CPU_outer.sh'), 'probe']
    identity = dict(request=REQUEST, publication=payload['publication'], manifest_SHA256=payload['manifest_SHA256'],
                    epoch=epoch, deadline=deadline, parent=PARENT, node=NODE, GPU=False)
    with intent.open('x', encoding='utf-8') as handle:
        json.dump(dict(identity, argv=argv, replay_allowed=False), handle)
        handle.flush()
        os.fsync(handle.fileno())
    with srun_log.open('xb') as log:
        process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    fields = (Path('/proc') / str(process.pid) / 'stat').read_text().rsplit(') ', 1)[1].split()
    row = dict(identity, host=socket.gethostname(), pid=process.pid, pgid=int(fields[2]),
               startticks=int(fields[19]), intent=str(intent))
    with launch.open('x', encoding='utf-8') as handle:
        json.dump(row, handle)
    print(json.dumps(row), flush=True)


if __name__ == '__main__':
    run(PAYLOAD)
