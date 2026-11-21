"""One finite fleet DMA campaign, with unique independently auditable Slurm shards."""

import argparse
import base64
from collections import defaultdict
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time


ROOT = Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
HERE = Path(globals().get('__file__', '<stdin>')).resolve().parent
REQUEST = '754ce2ebd4637a1f201c4df282d839f7'
LABEL = 'O01-hip-pinned-duplex-fleet-v1'
RUN = LABEL + '-' + REQUEST
PARENTS = ('1087656', '1103606')
ZONES = ('O01', *(f'T{i:02d}' for i in range(1, 13)), *(f'I{i:02d}' for i in range(1, 4)))
EXCLUDED = 'f12r2n13'
PILOT = '088fa0a7ad19aba63158cf65d942a853'
HIP_PIN = 'b125b6b4153f5d25dd8e445b43b2f57b0eab9b6305eb765430a82a4359e1870b'
PYTHON = '/usr/local/python3.12/bin/python3'
APPTAINER = '/public/software/apps/apptainer/1.3.4/bin/apptainer'
SIZES = (1024**2, 32 * 1024**2)
ARMS = ('serial', 'overlap', 'overlap', 'serial')
HEX = re.compile(r'[a-f0-9]{64}')
PCI = re.compile(r'[a-f0-9]{4}:[a-f0-9]{2}:[01][a-f0-9]\.[0-7]')
NODE = re.compile(r'[af]\d+r\d+n\d+')
FROZEN = {
    'CLAIM.json': 'a80b71cf09dd89a049dca04716c8760a6f6544b8c9e6f08340a38a7d35ecb1a5',
    'WORKER_MANIFEST.json': '3f655c886a6f90f76c6f1e501c480df8edacb60ef1844f8741cdaa8f52f8469c',
    'sync_dispatch.py': '3c13176628661ddc2474fa96b15f37da1c33cba49ad5ecf1493a8711587b7aaa',
    'ledger.json': '36361b92773f8eda412132ea176c29e814e4620ea8392ff287e68d854d67f5c9',
    'admission.py': 'fc89becc1b18bbab71e9a0c1dcdb05482b00643f7dff8cdfcbe1f2dd265291e5',
}
WORKERS = {'worker.py', 'hip_runtime.py', 'node_sample.py', 'qualified_sampler.py'}
MEMBERS = set(FROZEN) | WORKERS | {'controller.py'}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return sha256(raw).hexdigest()


def read(path, limit=128 * 1024):
    need(path.resolve(strict=True) == path and path.is_file(), 'original source or receipt path differs')
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    need(0 < len(raw) <= limit, 'original source or receipt exceeds bound')
    return raw


def save(path, value):
    with path.open('x', encoding='utf8') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def source_identity(files):
    return sha(json.dumps({name: sha(raw) for name, raw in sorted(files.items())}, separators=(',', ':')).encode())


def verify_source(files):
    need(set(files) == MEMBERS and all(0 < len(raw) <= 32 * 1024 for raw in files.values()), 'source member scope differs')
    for name, pin in FROZEN.items():
        need(sha(files[name]) == pin, f'published {name} differs')
    manifest = json.loads(files['WORKER_MANIFEST.json'])['files']
    for name in WORKERS:
        spec = manifest[name]
        need(len(files[name]) == spec['bytes'] and sha(files[name]) == spec['SHA256'], f'original v2 {name} differs')


def qualify_pilot(raw):
    need(type(raw) is bytes and 0 < len(raw) <= 128 * 1024, 'original .634 evidence required')
    result = json.loads(raw)
    row = result.get('accounting') or []
    need(result.get('request') == PILOT and result.get('terminal') is True
         and result.get('device_qualified') is True and result.get('valid_cards') == result.get('planned_cards') == 8
         and len(row) == 7 and row[:4] == ['1103606.634', 'O01-hip-pinned-duplex-pilot-v2', 'COMPLETED', '0:0']
         and row[5] == EXCLUDED and 'gres/dcu=8' in row[6].split(',')
         and not result.get('failures'), 'original .634 pilot has no independently qualified eight-card result')
    return sha(raw)


def shards(selected):
    groups = defaultdict(list)
    for node, details in sorted(selected.items()):
        groups[details['parent'], details['zone']].append(node)
    out = {}
    for (parent, zone), nodes in sorted(groups.items()):
        for offset in range(0, len(nodes), 4):
            label = f'duplex-{REQUEST[:12]}-{parent}-{zone}-{offset // 4:02d}'
            out[label] = dict(parent=parent, zone=zone, nodes=nodes[offset:offset + 4],
                              request=sha((REQUEST + ':' + label).encode())[:32])
    return out


def scope(plan, owners):
    need(plan['request'] == REQUEST and math.isfinite(plan['epoch'])
         and re.fullmatch(r'[a-f0-9]{40}', plan['publication'])
         and all(HEX.fullmatch(plan[key]) for key in ('pilot_SHA256', 'snapshot_SHA256', 'source_SHA256')),
         'campaign identity differs')
    selected = plan['selected']
    need(1 <= len(selected) <= 79 and EXCLUDED not in selected and plan['shards'] == shards(selected),
         'empty, duplicate or already-consumed node inventory')
    for node, result in selected.items():
        parent, zone = result['parent'], result['zone']
        need(NODE.fullmatch(node) and parent in PARENTS and zone in ZONES and owners[node] == (parent, zone)
             and re.fullmatch(parent + r'\.[0-9]+', result['before_CPU_step'])
             and len(result['PCI']) == len(set(result['PCI'])) == 8 and all(PCI.fullmatch(p) for p in result['PCI']),
             'node does not bind its original CPU-only admission or ownership')


def command(target, run, plan, label, unit):
    size, nodes = len(unit['nodes']), unit['nodes']
    runtime = [APPTAINER, 'exec', '--rocm', '--no-home', '--pwd', '/',
               '--bind', str(run) + ':' + str(run), '--bind', str(target) + ':' + str(target) + ':ro',
               '--bind', '/opt/hyhal:/opt/hyhal:ro', str(ROOT / 'containers/dtk26-sandbox-r1'),
               '/bin/bash', '--noprofile', '--norc', '-c',
               'set -eo pipefail; source /opt/dtk/env.sh; exec python3 -I -S -B -u "$@"',
               'hip-duplex-fleet', str(target / 'worker.py'), str(run / 'nodes'),
               str(plan['epoch'] + 180), str(run / 'pci.json')]
    return ['srun', '--jobid=' + unit['parent'], '--job-name=' + label, '--exact', '--nodes=' + str(size),
            '--ntasks=' + str(size), '--ntasks-per-node=1', '--nodelist=' + ','.join(nodes), '--immediate=5',
            '--kill-on-bad-exit=0', '--cpus-per-task=12', '--mem=12288M', '--gres=dcu:8', '--time=00:02:30',
            '--output=' + str(run / (label+'-%N.stdout')), '--error=' + str(run / (label+'-%N.stderr')), *runtime]


def original(target):
    files = {name: read(target / name, 32 * 1024) for name in MEMBERS}
    verify_source(files)
    plan = json.loads(read(target / 'plan.json'))
    need(source_identity(files) == plan['source_SHA256'], 'original staged source digest differs')
    sys.path.insert(0, str(target))
    import sync_dispatch
    owners = sync_dispatch.node_map(json.loads(files['ledger.json']))
    scope(plan, owners)
    return plan, owners, sync_dispatch


def dispatch(target, plan, owners, scheduler):
    epoch, groups = plan['epoch'], plan['shards']
    need(0 <= time.time() - epoch < 30, 'one finite campaign epoch expired')
    need(not scheduler.account(groups), 'one or more original labels exist')
    used = {unit['parent'] for unit in groups.values()}
    occupied = {parent:scheduler.allocation(parent, sorted(n for n, (j, _) in owners.items() if j == parent))
                for parent in used}
    need((ROOT / 'containers/dtk26-sandbox-r1').is_dir() and os.access(APPTAINER, os.X_OK), 'pinned sandbox missing')
    run = ROOT / 'results' / RUN
    run.mkdir(mode=0o700, exist_ok=False)
    for directory in ('nodes', 'attempts'):
        (run / directory).mkdir(mode=0o700)
    save(run / 'intent.json', dict(plan=plan, replay_allowed=False))
    save(run / 'pci.json', {node:item['PCI'] for node,item in plan['selected'].items()})
    children = {}
    for label, unit in groups.items():
        outcome = run / 'attempts' / f'{label}.json'
        if (set(unit['nodes']) & occupied[unit['parent']]) or time.time() >= epoch + 45:
            save(outcome, dict(attempted=False, reason='new occupied node or bounded admission elapsed'))
            continue
        argv = command(target, run, plan, label, unit)
        save(outcome, dict(attempted=True, argv=argv))
        try:
            children[label] = subprocess.Popen(argv, stdin=subprocess.DEVNULL, start_new_session=True)
        except OSError:
            continue
    save(run / 'srun-pids.json', {label: proc.pid for label, proc in children.items()})
    for proc in children.values():
        try:
            proc.wait(timeout=max(.01, epoch + 210 - time.time()))
        except subprocess.TimeoutExpired:
            break
    return dict(request=REQUEST, planned_cards=8*len(plan['selected']), srun_client_count=len(children),
                observed_accounting=scheduler.account(groups), replay_allowed=False)


def validate_node(directory, node, step, pci, deadline):
    artifacts, verified = {}, set()
    recordings = []
    def get(name):
        b = read(directory / name)
        artifacts[name] = sha(b)
        return b
    for card in range(8):
        try:
            payload = json.loads(get(f'card-{card}.json'))
            progress = [json.loads(line) for line in get(f'card-{card}.progress.jsonl').splitlines()]
            identity = payload['PCI']
            need(payload['card'] == card and payload['step'] == step and payload['node'] == node
                 and payload['normal_release'] is True and payload['error'] is None
                 and identity in pci and identity not in verified, 'worker identity differs')
            expected = []
            for case, size in zip(payload['cases'], SIZES, strict=True):
                need(case['bytes'] == size and case['passed'] is True
                     and len(set(case['source_SHA256'])) == len(case['source_SHA256']) == 2
                     and all(HEX.fullmatch(pin) for pin in case['source_SHA256']), 'case scope differs')
                for ordinal, (arm, mode) in enumerate(zip(case['arms'], ARMS, strict=True)):
                    need(arm['mode'] == mode and arm['warm_completed'] == 3 and len(arm['samples']) == 12
                         and arm['payload_and_guards_passed'] is True, 'arm scope differs')
                    for row in arm['samples']:
                        need(row['combined_bytes'] == 16*size and type(row['begin_ns']) is type(row['end_ns']) is int
                             and row['end_ns'] > row['begin_ns']
                             and all(math.isfinite(row[k]) and row[k] > 0 for k in ('h2d_stream_ms','d2h_stream_ms')),
                             'sample invalid')
                    expected.append(dict(bytes=size, arm_index=ordinal, result=arm))
            need(progress == expected, 'original arm journal differs')
            verified.add(identity)
            recordings.append(payload['cases'])
        except (OSError, KeyError, ValueError, TypeError):
            pass
    try:
        intent = json.loads(get('intent.json'))
        terminal = json.loads(get('terminal.json'))
        need(all(row['node'] == node and row['step'] == step and row['planned_cards'] == 8
                 for row in (intent, terminal)) and set(intent['PCI']) == set(pci)
             and intent['HIP_SHA256'] == HIP_PIN and intent['deadline'] == deadline
             and intent['sizes'] == list(SIZES) and intent['arm_order'] == list(ARMS)
             and intent['warm'] == 3 and intent['samples'] == 12 and intent['copies_per_direction'] == 8
             and terminal['normal_release_count'] == terminal['qualified_cards'] == 8
             and terminal['cooperative_stop_requested'] is False, 'node completion differs')
        valid = len(verified)
        return dict(valid_cards=valid, node_qualified=valid == 8, raw_SHA256=artifacts,
                    node_rates=timing(recordings) if valid == 8 else None)
    except (OSError, KeyError, ValueError, TypeError):
        return dict(valid_cards=len(verified), node_qualified=False, raw_SHA256=artifacts)


def timing(recordings):
    """Recompute directional stream and 8-card wall statistics from original samples."""
    bound = []
    for case_index, size in enumerate(SIZES):
        rates = {mode: [] for mode in ('serial','overlap')}
        direction = {mode: {'h2d_stream_ms': [], 'd2h_stream_ms': []} for mode in rates}
        for arm_index, mode in enumerate(ARMS):
            for sample in range(12):
                active = [cases[case_index]['arms'][arm_index]['samples'][sample] for cases in recordings]
                span = max(x['end_ns'] for x in active) - min(x['begin_ns'] for x in active)
                rates[mode].append(sum(x['combined_bytes'] for x in active) / span)
                for direction_name, readings in direction[mode].items():
                    readings.extend(x[direction_name] for x in active)
        median = {mode: statistics.median(values) for mode, values in rates.items()}
        bound.append(dict(bytes=size, node_GB_s=median, overlap_over_serial=median['overlap']/median['serial'],
                          median_card_stream_ms={mode:{name:statistics.median(values) for name,values in dirs.items()}
                                                 for mode,dirs in direction.items()}))
    return bound


def reconcile(plan, scheduler):
    run = ROOT / 'results' / RUN
    accounting = scheduler.account(plan['shards'])
    observations = {}
    for label, unit in plan['shards'].items():
        nodes, parent = unit['nodes'], unit['parent']
        info = dict(request=unit['request'], planned_cards=8*len(nodes), state='UNKNOWN', valid_cards=0,
                    independent_post_required=True, placement_by_parent={parent:{node:8 for node in nodes}})
        observations[label] = info
        row = accounting.get(label)
        attempt = run / 'attempts' / f'{label}.json'
        if attempt.is_file():
            info['original_attempt_SHA256'] = sha(read(attempt))
        if not row:
            continue
        need(len(row) == 7 and row[1] == label and re.fullmatch(parent + r'\.[0-9]+', row[0])
             and set(scheduler.hosts(row[5])) == set(nodes), 'original Slurm scope differs')
        state = row[2].split()[0].rstrip('+')
        info.update(state=state, accounting=row)
        if state not in scheduler.TERMINAL:
            continue
        info['terminal'] = True
        tres = dict(item.split('=', 1) for item in row[6].split(',') if '=' in item)
        resources = (tres.get('gres/dcu') == str(8*len(nodes)) and tres.get('cpu') == str(12*len(nodes))
                     and tres.get('mem') in (f'{12*len(nodes)}G', f'{12288*len(nodes)}M')
                     and row[4].isdecimal() and 0 < int(row[4]) <= 150)
        info['resources_match'] = resources and tres.get('node') == str(len(nodes))
        info['nodes'] = {node:validate_node(run / 'nodes' / node, node, row[0],
                                            plan['selected'][node]['PCI'], plan['epoch']+180) for node in nodes}
        info['valid_cards'] = sum(item['valid_cards'] for item in info['nodes'].values())
        info['device_qualified'] = (row[2:4] == ['COMPLETED','0:0'] and info['resources_match']
                                    and all(item['node_qualified'] for item in info['nodes'].values()))
    return dict(request=REQUEST, planned_cards=8*len(plan['selected']), observations=observations,
                valid_cards=sum(info['valid_cards'] for info in observations.values()), replay_allowed=False)


def receive(payload):
    os.umask(0o077)
    plan = payload['plan']
    need(0 <= time.time() - plan['epoch'] < 25, 'original receive absolute window expired')
    files = {}
    for name, (pin, encoded) in payload['files'].items():
        need(name in MEMBERS and len(encoded) <= 46000, 'source receive bound')
        raw = base64.b64decode(encoded, validate=True)
        need(sha(raw) == pin, 'source member changed in transit')
        files[name] = raw
    verify_source(files)
    need(source_identity(files) == plan['source_SHA256'], 'source identity changed in transit')
    target = ROOT / 'incoming' / RUN
    target.mkdir(mode=0o700, exist_ok=False)
    for name, raw in files.items():
        with (target / name).open('xb') as output:
            output.write(raw)
    save(target / 'plan.json', plan)
    child = subprocess.run([PYTHON, '-I','-S','-B','-u',str(target/'controller.py'),'--dispatch'],
                           stdin=subprocess.DEVNULL, timeout=max(.01, plan['epoch']+225-time.time()))
    return child.returncode


def main():
    if 'PAYLOAD' in globals():
        return receive(globals()['PAYLOAD'])
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--dispatch', action='store_true')
    group.add_argument('--reconcile', action='store_true')
    args = parser.parse_args()
    need(HERE == ROOT / 'incoming' / RUN, 'expected original staged controller')
    plan, owners, scheduler = original(HERE)
    result = dispatch(HERE, plan, owners, scheduler) if args.dispatch else reconcile(plan, scheduler)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
