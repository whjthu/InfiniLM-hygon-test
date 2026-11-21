"""The owned O01 single-node HIP copy pilot, with original-only reconciliation."""

import argparse
import base64
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
HERE = Path(globals().get('__file__', '<stdin>')).resolve().parent
PARENT, NODE, ZONE = '1103606', 'f12r2n13', 'O01'
LABEL = 'O01-hip-pinned-duplex-pilot-v1'
REQUEST = '773e6c71d59d885f3c8fa91602085be3'
CLAIM_SHA = '76d2e7003ca5610eecb252bf8f22cdf5c6d6d937d124db39ba2db7c435d7fe85'
WORKER_MANIFEST = '1aad87e16d12febe77c539180929696c97bf6d7fd2d93cfaa85ff7bbd7cb8aaa'
RUN_NAME = LABEL + '-' + REQUEST
SLURM_SECONDS, WORKER_SECONDS, CONTROLLER_SECONDS = 150, 180, 210
APPTAINER = '/public/software/apps/apptainer/1.3.4/bin/apptainer'
HOST_PYTHON = '/usr/local/python3.12/bin/python3'
SOURCE_NAMES = {'PILOT_CLAIM.json', 'SOURCE_FILES.json', 'pilot.py', 'worker.py', 'hip_runtime.py',
                'node_sample.py', 'qualified_sampler.py', 'sync_dispatch.py', 'admission.py', 'ledger.json'}
FROZEN = {'PILOT_CLAIM.json': CLAIM_SHA, 'SOURCE_FILES.json': WORKER_MANIFEST,
          'sync_dispatch.py': '3c13176628661ddc2474fa96b15f37da1c33cba49ad5ecf1493a8711587b7aaa',
          'admission.py': 'fc89becc1b18bbab71e9a0c1dcdb05482b00643f7dff8cdfcbe1f2dd265291e5',
          'ledger.json': '36361b92773f8eda412132ea176c29e814e4620ea8392ff287e68d854d67f5c9'}
HEX = re.compile(r'[a-f0-9]{64}')
PCI = re.compile(r'[a-f0-9]{4}:[a-f0-9]{2}:[01][a-f0-9]\.[0-7]')
HIP_SHA256 = 'b125b6b4153f5d25dd8e445b43b2f57b0eab9b6305eb765430a82a4359e1870b'
SIZES, ARMS = (1024**2, 32 * 1024**2), ('serial', 'overlap', 'overlap', 'serial')


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha(raw):
    return sha256(raw).hexdigest()


def source_identity(files):
    pins = {name: sha(raw) for name, raw in sorted(files.items())}
    return sha(json.dumps(pins, separators=(',', ':')).encode())


def save(path, value):
    with path.open('x', encoding='utf8') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def read(path, limit=128 * 1024):
    need(path.resolve(strict=True) == path and path.is_file(), 'original receipt path is not canonical')
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    need(0 < len(raw) <= limit, 'original receipt is empty or too large')
    return raw


def source_verify(files):
    need(set(files) == SOURCE_NAMES and all(0 < len(raw) <= 32 * 1024 for raw in files.values()), 'source pack bounds')
    for name, expected in FROZEN.items():
        need(sha(files[name]) == expected, f'{name} differs from fixed pilot dependency')
    manifest = json.loads(files['SOURCE_FILES.json'])['files']
    for name in ('worker.py', 'hip_runtime.py', 'node_sample.py', 'qualified_sampler.py'):
        pin = manifest[name]
        need(len(files[name]) == pin['bytes'] and sha(files[name]) == pin['SHA256'], f'{name} differs from worker manifest')


def scope(plan):
    allowed = {'request', 'epoch', 'PCI', 'before_CPU_step', 'source_contract_SHA256', 'before_snapshot_SHA256',
               'publication', 'handoff_SHA256'}
    need(set(plan) == allowed and plan['request'] == REQUEST and type(plan['epoch']) is float
         and math.isfinite(plan['epoch']) and re.fullmatch(PARENT + r'\.[0-9]+', plan['before_CPU_step'])
         and re.fullmatch(r'[a-f0-9]{40}', plan['publication'])
         and all(HEX.fullmatch(plan[key]) for key in ('before_snapshot_SHA256', 'source_contract_SHA256', 'handoff_SHA256')),
         'original pilot request or CPU admission receipt differs')
    need(set(plan['PCI']) == {NODE} and len(set(plan['PCI'][NODE])) == len(plan['PCI'][NODE]) == 8
         and all(PCI.fullmatch(value) for value in plan['PCI'][NODE]), 'exact O01 card set differs')


def load_scheduler(target):
    source_verify({name: read(target / name, 32 * 1024) for name in SOURCE_NAMES})
    sys.path.insert(0, str(target))
    import sync_dispatch

    return sync_dispatch


def command(target, run, pci_file, epoch):
    runtime = [APPTAINER, 'exec', '--rocm', '--no-home', '--pwd', '/',
               '--bind', str(run) + ':' + str(run), '--bind', str(target) + ':' + str(target) + ':ro',
               '--bind', '/opt/hyhal:/opt/hyhal:ro', str(ROOT / 'containers/dtk26-sandbox-r1'),
               '/bin/bash', '--noprofile', '--norc', '-c',
               'set -eo pipefail; source /opt/dtk/env.sh; exec python3 -I -S -B -u "$@"',
               'hip-duplex-pilot', str(target / 'worker.py'), str(run / 'nodes'),
               str(epoch + WORKER_SECONDS), str(pci_file)]
    return ['srun', '--jobid=' + PARENT, '--job-name=' + LABEL, '--exact', '--nodes=1', '--ntasks=1',
            '--ntasks-per-node=1', '--nodelist=' + NODE, '--immediate=5', '--kill-on-bad-exit=0',
            '--cpus-per-task=12', '--mem=12288M', '--gres=dcu:8', '--time=00:02:30',
            '--output=' + str(run / '%N.stdout'), '--error=' + str(run / '%N.stderr'), *runtime]


def observation(row):
    if not row:
        return dict(state='UNKNOWN', terminal=False)
    need(len(row) == 7 and re.fullmatch(PARENT + r'\.[0-9]+', row[0]) and row[1] == LABEL
         and row[5] == NODE, 'accounting row differs from original node or label')
    state = row[2].split()[0].rstrip('+')
    terminal = state in ('COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'NODE_FAIL', 'OUT_OF_MEMORY',
                         'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED')
    return dict(state=state, terminal=terminal, accounting=row,
                accounting_columns=['JobIDRaw','JobName%100','State','ExitCode','ElapsedRaw','NodeList','AllocTRES'])


def reconcile(target, scheduler):
    run = ROOT / 'results' / RUN_NAME
    plan_raw = read(target / 'plan.json')
    plan = json.loads(plan_raw)
    scope(plan)
    need(source_identity({name: read(target / name, 32 * 1024) for name in SOURCE_NAMES}) ==
         plan['source_contract_SHA256'], 'original staged source identity differs')
    result = dict(request=REQUEST, label=LABEL, node=NODE, planned_cards=8, valid_cards=0, replay_allowed=False,
                  plan_SHA256=sha(plan_raw), device_qualified=False, raw_record_SHA256={}, failures=[],
                  independent_post_required=True)
    result.update(observation(scheduler.account({LABEL}).get(LABEL)))
    if not result['terminal']:
        return result
    row = result['accounting']
    tres = dict(field.split('=', 1) for field in row[6].split(',') if '=' in field)
    if (row[2:4] != ['COMPLETED', '0:0'] or tres.get('node') != '1' or tres.get('gres/dcu') != '8'
            or tres.get('cpu') != '12' or tres.get('mem') not in ('12G','12288M')
            or not row[4].isdecimal() or not 0 < int(row[4]) <= SLURM_SECONDS):
        result['failures'].append('SLURM_TERMINAL_SCOPE_OR_EXIT')
    cards = run / 'nodes' / NODE
    successful_pci = set()
    for card in range(8):
        journal = f'card-{card}.progress.jsonl'
        journal_bytes = None
        if (cards / journal).is_file():
            try:
                journal_bytes = read(cards / journal)
                result['raw_record_SHA256'][journal] = sha(journal_bytes)
            except (ValueError, OSError):
                result['failures'].append('TRUNCATED_OR_INVALID_' + journal)
        name = f'card-{card}.json'
        try:
            raw = read(cards / name)
            result['raw_record_SHA256'][name] = sha(raw)
            value = json.loads(raw)
            pci = value.get('PCI')
            need(value.get('step') == row[0] and value.get('node') == NODE
                 and value.get('card') == card and pci in plan['PCI'][NODE] and pci not in successful_pci
                 and value.get('normal_release') is True and value.get('error') is None
                 and isinstance(value.get('cases'), list) and len(value['cases']) == 2
                 and journal in result['raw_record_SHA256'], 'owner identity or raw progress missing')
            completed = []
            for case, size in zip(value['cases'], SIZES, strict=True):
                need(case['bytes'] == size and case['passed'] is True and len(case['arms']) == 4,
                     'owner case scope is incomplete')
                need(len(case['source_SHA256']) == 2 and len(set(case['source_SHA256'])) == 2
                     and all(HEX.fullmatch(item) for item in case['source_SHA256']), 'distinct input byte identities absent')
                for index, (arm, expected) in enumerate(zip(case['arms'], ARMS, strict=True)):
                    need(arm['mode'] == expected and arm['warm_completed'] == 3 and len(arm['samples']) == 12
                         and arm['payload_and_guards_passed'] is True, 'owner arm scope is incomplete')
                    completed.append(dict(bytes=size, arm_index=index, result=arm))
                    for sample in arm['samples']:
                        need(sample['combined_bytes'] == 16*size and type(sample['begin_ns']) is int
                             and type(sample['end_ns']) is int and sample['end_ns'] > sample['begin_ns']
                             and all(math.isfinite(sample[key]) and sample[key] > 0
                                     for key in ('h2d_stream_ms','d2h_stream_ms')),
                             'recorded sample is missing measured bytes or time')
            need([json.loads(line) for line in journal_bytes.splitlines()] == completed, 'durable progress and owner final differ')
            successful_pci.add(pci)
        except (OSError, ValueError, AttributeError, KeyError, TypeError):
            result['failures'].append('CARD_INCOMPLETE_' + str(card))
    result['valid_cards'] = len(successful_pci)
    for name in ('intent.json','terminal.json'):
        try:
            raw = read(cards / name)
            result['raw_record_SHA256'][name] = sha(raw)
            item = json.loads(raw)
            need(item.get('node') == NODE and item.get('step') == row[0] and item.get('planned_cards') == 8,
                 'worker original step differs')
            if name == 'terminal.json':
                need(item.get('qualified_cards') == item.get('normal_release_count') == 8
                     and item.get('cooperative_stop_requested') is False, 'node terminal is partial')
            else:
                need(sorted(item.get('PCI', [])) == sorted(plan['PCI'][NODE])
                     and item.get('HIP_SHA256') == HIP_SHA256 and item.get('deadline') == plan['epoch'] + WORKER_SECONDS
                     and item.get('sizes') == list(SIZES) and item.get('arm_order') == list(ARMS)
                     and item.get('warm') == 3 and item.get('samples') == 12 and item.get('copies_per_direction') == 8,
                     'worker original workload or device binding differs')
        except (OSError, ValueError, AttributeError, KeyError, TypeError):
            result['failures'].append('NODE_INCOMPLETE_' + name)
    result['device_qualified'] = result['valid_cards'] == 8 and not result['failures']
    return result


def dispatch(target, scheduler):
    plan = json.loads(read(target / 'plan.json'))
    scope(plan)
    epoch = plan['epoch']
    need(0 <= time.time() - epoch < 25, 'admission not within the original epoch')
    ledger = json.loads(read(target / 'ledger.json'))
    owners = scheduler.node_map(ledger)
    need(owners[NODE] == (PARENT, ZONE), 'exact parent O01 membership differs')
    all_nodes = sorted(node for node, (job, _) in owners.items() if job == PARENT)
    need(NODE not in scheduler.allocation(PARENT, all_nodes), 'target now has a numbered Slurm step')
    need(not scheduler.account({LABEL}), 'original label already exists; reconcile instead')
    run = ROOT / 'results' / RUN_NAME
    need((ROOT / 'containers/dtk26-sandbox-r1').is_dir() and os.access(APPTAINER, os.X_OK),
         'qualified DTK sandbox or Apptainer is absent')
    need(time.time() < epoch + 30, 'insufficient original dispatch time')
    run.mkdir(mode=0o700, exist_ok=False)
    (run / 'nodes').mkdir(mode=0o700)
    pci_file = run / 'pci.json'
    save(pci_file, plan['PCI'])
    argv = command(target, run, pci_file, epoch)
    save(run / 'intent.json', dict(plan=plan, argv=argv, replay_allowed=False,
                                  absolute_worker_deadline=epoch+WORKER_SECONDS,
                                  absolute_controller_deadline=epoch+CONTROLLER_SECONDS))
    child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, start_new_session=True)
    save(run / 'submitted.json', dict(request=REQUEST, srun_PID=child.pid, attempted=True))
    try:
        rc = child.wait(timeout=max(0.01, epoch + CONTROLLER_SECONDS - 12 - time.time()))
    except subprocess.TimeoutExpired:
        rc = None
    result = observation(scheduler.account({LABEL}).get(LABEL))
    result.update(request=REQUEST, srun_client_rc=rc, owned_srun_pid=child.pid, replay_allowed=False)
    save(run / 'first-accounting.json', result)
    return result


def receive(payload):
    os.umask(0o077)
    plan = payload['plan']
    scope(plan)
    need(0 <= time.time() - plan['epoch'] < 25, 'shared host plan expired')
    packed = payload['files']
    need(set(packed) == SOURCE_NAMES, 'incorrect staged member names')
    files = {}
    for name, (pin, encoded) in packed.items():
        need(len(encoded) <= 46000, 'encoded source outside fixed bounds')
        raw = base64.b64decode(encoded, validate=True)
        need(sha(raw) == pin, 'envelope source differs')
        files[name] = raw
    source_verify(files)
    need(source_identity(files) == plan['source_contract_SHA256'], 'finite source envelope is not the original')
    target = ROOT / 'incoming' / RUN_NAME
    target.mkdir(mode=0o700, exist_ok=False)
    for name, raw in files.items():
        with (target / name).open('xb') as stream:
            stream.write(raw)
    save(target / 'plan.json', plan)
    worker = subprocess.run(
        [HOST_PYTHON, '-I','-S','-B','-u',str(target / 'pilot.py'),'--dispatch'],
        stdin=subprocess.DEVNULL, timeout=max(0.01, plan['epoch'] + CONTROLLER_SECONDS - time.time()))
    return worker.returncode


def main():
    if 'PAYLOAD' in globals():
        return receive(globals()['PAYLOAD'])
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--dispatch', action='store_true')
    modes.add_argument('--reconcile', action='store_true')
    args = parser.parse_args()
    need(HERE == ROOT / 'incoming' / RUN_NAME, 'expected unique staged pilot path')
    scheduler = load_scheduler(HERE)
    result = dispatch(HERE, scheduler) if args.dispatch else reconcile(HERE, scheduler)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
