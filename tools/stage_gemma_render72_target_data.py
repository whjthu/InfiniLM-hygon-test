"""Stage the published Gemma training packet after reconciling its exact CPU predecessor."""

import argparse
from hashlib import sha256
import io
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from read_training_slurm_membership import allocated_dcu


REPO = Path(__file__).resolve().parents[1]
PACKAGE = REPO / 'code/bw1000/training-gemma-render72-target-data-stage-v1'
PACKAGE_COMMIT = '21c0207c1c83106ff18e4573448bb699a1b899b2'
MANIFEST_SHA = 'd9b739bd375e60b4cd09ddec37c4851366f9c6df0384b204c328c9a376784d2b'
PACKET_SHA = '05621957b708b85ea2ab9978a41cf863b65a65ce07f73e96ce01a0a339550a56'
PACKET_BYTES = 1120578
RESULTS = Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results')
CPU_LABEL = 'training-gemma-usgs-render72-native-input-audit-v1'
CPU_REQUEST = '8d6d573cc64b4ab3b082b17682043a11'
STAGE_LABEL = 'training-gemma-usgs-render72-native-CPU-v1'
STAGE_REQUEST = '0f9b465a7c60419c8350d44e8f2f7dfd'
PARENT = '1103606'
COLUMNS = 'JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES'


def need(condition, why):
    if not condition:
        raise ValueError(why)


def present(path):
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def bounded(path, maximum):
    need(path.resolve(strict=True) == path, 'published file is aliased')
    with path.open('rb') as source:
        raw = source.read(maximum + 1)
    need(0 < len(raw) <= maximum, 'published file is empty or oversized')
    return raw


def published_package():
    raw = bounded(PACKAGE / 'SOURCE_FILES.json', 4096)
    need(sha256(raw).hexdigest() == MANIFEST_SHA, 'published Gemma source manifest differs')
    members = json.loads(raw)['files']
    checked = {}
    for name, pin in members.items():
        need(Path(name).name == name, 'unexpected published file path')
        body = bounded(PACKAGE / name, pin['bytes'])
        need(len(body) == pin['bytes'] and sha256(body).hexdigest() == pin['SHA256'], 'Gemma staging member differs')
        checked[name] = body
    claim, inputs = (json.loads(checked[name]) for name in ('CLAIM.json', 'INPUTS.json'))
    need(claim['request_id'] == STAGE_REQUEST and claim['bound_CPU_request'] == CPU_REQUEST
         and claim['target'] == (RESULTS / STAGE_LABEL).as_posix() and claim['max_target_invocations'] == 1
         and claim['target_seconds'] == 120 and claim['overwrite_or_unknown_replay_allowed'] is False,
         'published packet is outside the original scope')
    namespace = {'__name__': 'pinned_gemma_receiver', '__file__': str(PACKAGE / 'receive_once.py')}
    exec(compile(checked['receive_once.py'], namespace['__file__'], 'exec'), namespace)
    return inputs, namespace['install']


def observe(results, query, uid):
    need(results.resolve(strict=True) == results and results.is_dir(), 'the project results root is aliased')
    observation = dict(parent=PARENT, cpu_request=CPU_REQUEST, stage_request=STAGE_REQUEST,
                       cpu_namespace_present=present(results / CPU_LABEL),
                       stage_namespace_present=present(results / STAGE_LABEL))
    job = query('scontrol', '-o', 'show', 'job', PARENT)
    need(len(job) <= 32768 and len(job.splitlines()) == 1
         and re.search(r'\bUserId=[^\s(]+\(' + str(uid) + r'\)(?=\s|$)', job)
         and re.search(r'\bJobId=' + PARENT + r'(?=\s|$)', job)
         and re.search(r'\bJobState=RUNNING(?=\s|$)', job), 'owned parent not confirmed running')
    accounting = query('sacct', '-nP', '-j', PARENT, '-S', '2026-11-20T00:00:00', '-o', COLUMNS)
    need(len(accounting) <= 2 * 1024**2, 'oversized Slurm accounting output')
    rows = [line.split('|') for line in accounting.splitlines() if line.strip()]
    need(all(len(row) == 7 for row in rows), 'malformed accounting output')
    parents = [row for row in rows if row[0] == PARENT]
    need(len(parents) == 1 and parents[0][2] == 'RUNNING' and allocated_dcu(parents[0][6]) == 320,
         'owned parent allocation not confirmed by accounting')
    matches = [row for row in rows if row[1] == CPU_LABEL]
    need(len(matches) <= 1 and all(re.fullmatch(PARENT + r'\.\d+', row[0]) for row in matches),
         'ambiguous original CPU accounting')
    observation.update(cpu_accounting_row=matches[0] if matches else None,
                       accounting_SHA256=sha256(accounting.encode()).hexdigest())
    observation['cpu_namespace_present'] |= present(results / CPU_LABEL)
    observation['stage_namespace_present'] |= present(results / STAGE_LABEL)
    observation['state'] = ('RECONCILE_ORIGINAL' if observation['cpu_namespace_present']
                            or observation['stage_namespace_present'] or matches else 'ORIGINAL_NAMESPACES_CLEAR')
    return observation


def receive(stream, results, epoch, check, inputs, install):
    first = check()
    need(first['state'] == 'ORIGINAL_NAMESPACES_CLEAR', 'original request visible; reconcile before staging')
    raw = stream.read(PACKET_BYTES + 1)
    need(len(raw) == PACKET_BYTES and sha256(raw).hexdigest() == PACKET_SHA, 'original private packet differs')
    last = check()
    need(last['state'] == 'ORIGINAL_NAMESPACES_CLEAR', 'original request changed during transfer')
    outcome = install(io.BytesIO(raw), results / STAGE_LABEL, inputs, epoch)
    result = {key: outcome[key] for key in ('state', 'passed', 'request_id', 'planned_files', 'completed_files',
                                           'actual_bytes_written', 'READY')}
    result.update(packet_SHA256=PACKET_SHA, package_commit=PACKAGE_COMMIT,
                  preceding_accounting_SHA256=[first['accounting_SHA256'], last['accounting_SHA256']])
    with (results / STAGE_LABEL / 'stage-admission.json').open('x', encoding='utf-8') as report:
        json.dump(dict(result, original_cpu_request=CPU_REQUEST, epoch=epoch), report, sort_keys=True, allow_nan=False)
        report.write('\n')
        report.flush()
        os.fsync(report.fileno())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('inspect', 'receive'))
    parser.add_argument('--publication', required=True)
    parser.add_argument('--epoch', type=float)
    args = parser.parse_args()
    need(os.name == 'posix' and args.publication == PACKAGE_COMMIT, 'expected target HOST and reviewed data publication')
    wall = time.time()
    need((args.action == 'inspect' and args.epoch is None)
         or (args.action == 'receive' and args.epoch is not None and math.isfinite(args.epoch)
             and 0 < args.epoch <= wall < args.epoch + 120),
         'expected one original 120 second receive epoch')
    deadline = time.monotonic() + (25 if args.action == 'inspect' else args.epoch + 120 - wall)

    def expired(*_):
        raise TimeoutError('original Gemma staging deadline')

    def query(*argv):
        remaining = deadline - time.monotonic()
        need(remaining > 1, 'Gemma staging observation budget exhausted')
        return subprocess.check_output(argv, text=True, timeout=min(8, remaining-1)).strip()

    def check():
        return observe(RESULTS, query, os.getuid())

    signal.signal(signal.SIGALRM, expired)
    seconds = deadline - time.monotonic()
    need(seconds > 0, 'Gemma staging observation budget exhausted')
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        inputs, install = published_package()
        if args.action == 'inspect':
            result = check()
        else:
            cpus = sorted(os.sched_getaffinity(0))
            need(len(cpus) >= 2, 'two HOST CPUs unavailable')
            os.sched_setaffinity(0, cpus[:2])
            os.umask(0o077)
            result = receive(sys.stdin.buffer, RESULTS, args.epoch, check, inputs, install)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return int(args.action == 'receive' and not result['passed'])


if __name__ == '__main__':
    raise SystemExit(main())
