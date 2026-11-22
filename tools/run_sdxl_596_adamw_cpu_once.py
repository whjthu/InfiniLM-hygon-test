"""Run the frozen SDXL .596 update attribution on one owned CPU-only step."""

import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


ROOT = Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
PARENT = '1087656'
NODE = 'f12r3n20'
REQUEST = 'bb82d832018c4d909a82b2a09939fb16'
LABEL = 'training-sdxl-original596-update-attribution-cpu-v1'
COLUMNS = 'JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES'
TERMINAL = {'BOOT_FAIL', 'CANCELLED', 'COMPLETED', 'DEADLINE', 'FAILED', 'NODE_FAIL',
            'OUT_OF_MEMORY', 'PREEMPTED', 'REVOKED', 'TIMEOUT'}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def save(path, value):
    with path.open('x', encoding='utf8') as out:
        json.dump(value, out, sort_keys=True, allow_nan=False)
        out.write('\n')


def query(*args):
    raw = subprocess.check_output(args, text=True, timeout=7)
    need(len(raw) <= 2 * 1024**2, 'Slurm query exceeded the bounded output size')
    return raw.splitlines()


def fields(line):
    return dict(word.split('=', 1) for word in line.split() if '=' in word)


def hosts(expression):
    need(re.fullmatch(r'[A-Za-z0-9_,\[\]-]+', expression), 'Invalid Slurm node expression')
    return set(query('scontrol', 'show', 'hostnames', expression))


def active_placement():
    line, = query('scontrol', 'show', 'job', PARENT, '-o')
    job = fields(line)
    need(job['JobId'] == PARENT and job['JobState'] == 'RUNNING'
         and job['UserId'].endswith(f'({os.getuid()})') and job['NumNodes'] == '40'
         and NODE in hosts(job['NodeList']), 'Owned long allocation is unavailable')
    for line in query('scontrol', 'show', 'step', PARENT, '-o'):
        if line == 'No steps in the system':
            continue
        step = fields(line)
        ident = step['StepId']
        if ident in {PARENT + '.batch', PARENT + '.extern'}:
            continue
        need(re.fullmatch(PARENT + r'\.\d+', ident) and NODE not in hosts(step['NodeList']),
             'An active Slurm step conflicts with this CPU diagnostic')


def accounting():
    rows = [line.split('|') for line in query('sacct', '-n', '-P', '-j', PARENT, '--format=' + COLUMNS)]
    chosen = [row for row in rows if len(row) > 1 and row[1] == LABEL]
    need(len(chosen) <= 1 and all(len(row) == 7 for row in chosen), 'Duplicate or malformed diagnostic label')
    return chosen[0] if chosen else None


def device_free_step(row):
    if row is None:
        return False
    tres = dict(item.split('=', 1) for item in row[6].split(','))
    return bool(re.fullmatch(PARENT + r'\.\d+', row[0]) and row[5] == NODE
                and tres.get('cpu') == '4' and tres.get('mem') in {'8G', '8192M'}
                and tres.get('node') == '1' and not any(k.startswith('gres/') for k in tres))


def process_start(pid):
    return int(Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()[19])


def signal_owned(child, startticks, sig):
    if child.poll() is not None:
        return
    try:
        need(process_start(child.pid) == startticks and os.getpgid(child.pid) == child.pid,
             'Owned CPU subprocess identity changed')
        os.killpg(child.pid, sig)
    except (FileNotFoundError, ProcessLookupError):
        return


def worker_args(stage):
    source = ROOT / 'results'
    mounts = {'audit': stage, 'original': source / 'training-sdxl-sdpa-adjoint-fp32-v1-a93d41a0d31140f1ad99bb105a024e99',
              'reference': source / 'training-t06-sdxl-first-reference-r2',
              'training-runtime': source / 'training-t06-sdxl-runtime-v1'}
    image = ROOT / 'containers/dtk26-sandbox-r1'
    need(all(p.is_dir() for p in (*mounts.values(), image)), 'A pinned CPU runtime mount is unavailable')
    argv = ['/public/software/apps/apptainer/1.3.4/bin/apptainer', 'exec', '--no-home', '--pwd', '/']
    for dest, path in mounts.items():
        argv.extend(['--bind', f'{path}:/{dest}:ro'])
    argv.extend(['--bind', '/opt/hyhal:/opt/hyhal:ro', str(image), '/bin/bash', '--noprofile', '--norc', '-c',
                 'set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh; '
                 'export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/training-runtime/site '
                 'OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1; '
                 'exec /usr/bin/timeout --kill-after=5s 100s python3 -B -u /audit/read_original.py '
                 '--run /original --public /audit/original596-public.json --reference-dir /reference '
                 '--optimizer-helper /audit/adamw398-worker.py'])
    return argv


def run(stage, publication, manifest_sha):
    start = time.monotonic()
    need(stage == ROOT / 'incoming' / LABEL and re.fullmatch(r'[0-9a-f]{40}', publication), 'Unexpected source identity')
    raw = (stage / 'SOURCE_FILES.json').read_bytes()
    need(digest(raw) == manifest_sha, 'Incorrect reader manifest')
    pin = json.loads(raw)['files']['read_original.py']
    reader = (stage / 'read_original.py').read_bytes()
    need((len(reader), digest(reader)) == (pin['bytes'], pin['SHA256']), 'Incorrect reader source')
    active_placement()
    need(accounting() is None and time.monotonic() - start < 45, 'Original accounting or launch interval changed')
    output = ROOT / 'results' / LABEL
    output.mkdir(mode=0o700)
    save(output / 'intent.json', dict(request=REQUEST, publication=publication, manifest_SHA256=manifest_sha,
                                     parent=PARENT, node=NODE, epoch=time.time(), GRES=0))
    command = ['srun', '--jobid=' + PARENT, '--job-name=' + LABEL, '--exact', '--nodes=1', '--ntasks=1',
               '--nodelist=' + NODE, '--cpus-per-task=4', '--mem=8192M', '--gres=none', '--immediate=5',
               '--time=00:02:00', *worker_args(stage)]
    timed_out = False
    with (output / 'result.private.json').open('xb') as stdout, (output / 'stderr.private').open('xb') as stderr:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, start_new_session=True)
        try:
            startticks = process_start(child.pid)
        except FileNotFoundError:
            need(child.poll() is not None, 'CPU subprocess start identity unavailable')
            startticks = None
        save(output / 'srun.json', {'PID': child.pid, 'PGID': child.pid, 'startticks': startticks})
        try:
            rc = child.wait(timeout=128)
        except subprocess.TimeoutExpired:
            timed_out = True
            signal_owned(child, startticks, signal.SIGTERM)
            try:
                rc = child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                signal_owned(child, startticks, signal.SIGKILL)
                rc = child.wait(timeout=3)
    row = None
    for _ in range(10):
        row = accounting()
        if row and row[2].split()[0].rstrip('+') in TERMINAL:
            break
        if time.monotonic() - start >= 195:
            break
        time.sleep(1)
    path = output / 'result.private.json'
    if 0 < path.stat().st_size <= 131072:
        raw = path.read_bytes()
    else:
        raw = b''
    try:
        metrics = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        metrics = {}
    valid_metrics = (isinstance(metrics, dict) and metrics.get('actual_gpu_step') == '1087656.596'
                     and metrics.get('actual_comparator_step') == '1087656.598'
                     and metrics.get('GPU_cards') == 0 and metrics.get('tensors') == 1120)
    success = bool(rc == 0 and valid_metrics and device_free_step(row) and row[2:4] == ['COMPLETED', '0:0'])
    result = dict(request=REQUEST, accounting_columns=COLUMNS.split(','), actual=row, rc=rc, timed_out=timed_out,
                  result_bytes=len(raw), result_SHA256=digest(raw), passed=success, GPU_steps=0)
    save(output / 'terminal.json', result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return int(not success)


if __name__ == '__main__':
    raise SystemExit(run(Path(sys.argv[1]), sys.argv[2], sys.argv[3]))
