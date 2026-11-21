"""Sample assigned DCUs inside existing workload steps and join a common fleet window."""

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import time

from sample_training80_busy import INTERVAL_S, MAX_LAG_S, PCI, POINTS, node_map, require, require_pins


MODES = {'workload', 'idle_gres_sample'}


def hex_token(value, length):
    return isinstance(value, str) and len(value) == length and re.fullmatch('[0-9a-f]+', value) is not None


def check_binding(binding, node, step):
    cards = binding['pci']
    require(binding['node'] == node and binding['step'] == step and re.fullmatch(r'[0-9]+\.[0-9]+', step)
            and hex_token(binding['request'], 32) and binding['mode'] in MODES and hex_token(binding['owner_receipt_SHA256'], 64)
            and re.fullmatch(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}', binding['boot_id']),
            'sampling must bind its original node, numbered step and device-owner receipt')
    require(type(binding['scheduled_epoch']) in (int, float) and math.isfinite(binding['scheduled_epoch'])
            and binding['scheduled_epoch'] > 0 and type(cards) is list and 0 < len(cards) == len(set(cards)) <= 8
            and all(isinstance(pci, str) and PCI.fullmatch(pci) for pci in cards), 'invalid sample window or PCI subset')


def read_points(drm, cards, scheduled, *, clock=time.time, sleep=time.sleep):
    paths = {}
    wanted = set(cards)
    for entry in sorted(drm.glob('renderD*')):
        device = entry / 'device'
        pci = device.resolve(strict=True).name.lower()
        if pci in wanted:
            require(pci not in paths, 'duplicate owned render device')
            paths[pci] = device / 'gpu_busy_percent'
    require(set(paths) == wanted, 'owned device disappeared from DRM inventory')
    rows = []
    for point in range(POINTS):
        planned = scheduled + point*INTERVAL_S
        delay = planned-clock()
        if delay > 0:
            sleep(delay)
        opened = clock()
        require(planned <= opened <= planned+MAX_LAG_S, 'read missed the common start time')
        values = {pci: int(paths[pci].read_text()) for pci in cards}
        closed = clock()
        require(opened <= closed <= planned+MAX_LAG_S and all(0 <= value <= 100 for value in values.values()),
                'read exceeded the common window or returned invalid busy')
        rows.append(dict(point=point, opened_epoch=opened, closed_epoch=closed, busy=values))
    return rows


def sample(binding, binding_sha, node, step, boot_id, *, read=read_points, clock=time.time):
    check_binding(binding, node, step)
    require(hex_token(binding_sha, 64) and isinstance(boot_id, str) and boot_id == binding['boot_id'] and boot_id,
            'sampling boot or source identity differs from assigned device receipt')
    report = dict(binding_SHA256=binding_sha, node=node, step=step, boot_id=boot_id, mode=binding['mode'],
                  request=binding['request'], scheduled_epoch=binding['scheduled_epoch'], pci=binding['pci'],
                  owner_receipt_SHA256=binding['owner_receipt_SHA256'],
                  counter_kind='sysfs_gpu_busy_percent', sampler_submitted_GPU_compute=False, rows=[], error=None)
    try:
        now = clock()
        require(0 < binding['scheduled_epoch'] - now <= 60, 'assigned worker did not enter before the sampling deadline')
        report['rows'] = read(Path('/sys/class/drm'), binding['pci'], binding['scheduled_epoch'])
    except (OSError, ValueError) as error:
        report['error'] = type(error).__name__
    return report


def aggregate(owners, pins, plan, reports):
    require_pins(owners, pins)
    requested = plan['request']
    scheduled = plan['scheduled_epoch']
    require(hex_token(requested, 32) and type(scheduled) in (int,float) and math.isfinite(scheduled)
            and scheduled > 0, 'invalid original common sampling window')
    bindings = plan['bindings']
    all_cards = {(node, pci) for node, pcis in pins.items() for pci in pcis}
    planned_cards = set()
    for digest, binding in bindings.items():
        require(hex_token(digest, 64), 'expected an exact source binding SHA-256')
        node, step = binding['node'], binding['step']
        check_binding(binding, node, step)
        keys = {(node,pci) for pci in binding['pci']}
        require(node in owners and step.split('.')[0] == owners[node][0] and binding['request'] == requested
                and binding['scheduled_epoch'] == scheduled and keys <= all_cards and planned_cards.isdisjoint(keys),
                'foreign or overlapping PCI assignment in original sampling plan')
        planned_cards.update(keys)

    by_card, groups, failures, sampled_workload = {}, {}, {}, set()
    expected_group_samples = {}
    for node, (_,group) in owners.items():
        expected_group_samples[group] = expected_group_samples.get(group,0) + len(pins[node])*POINTS
    require(set(reports) <= set(bindings), 'unexpected or mismatched source report')
    for digest, report in reports.items():
        binding = bindings[digest]
        require(report['binding_SHA256'] == digest and all(report[key] == binding[key] for key in
                ('request','scheduled_epoch','node','step','boot_id','mode','pci','owner_receipt_SHA256'))
                and report['counter_kind'] == 'sysfs_gpu_busy_percent'
                and report['sampler_submitted_GPU_compute'] is False, 'original sampling report identity differs')
        if report['error'] is not None:
            require(report['rows'] == [], 'failed sample exposed unvalidated partial rows')
            failures[digest] = report['error']
            continue
        node = binding['node']
        cards = set(binding['pci'])
        if binding['mode']=='workload':
            sampled_workload.update((node,pci) for pci in cards)
        require(len(report['rows']) == POINTS, 'missing original sampling point')
        for point, row in enumerate(report['rows']):
            planned = scheduled + point*INTERVAL_S
            require(row['point'] == point and planned <= row['opened_epoch'] <= row['closed_epoch'] <= planned+MAX_LAG_S
                    and set(row['busy']) == cards and all(type(v) is int and 0 <= v <= 100 for v in row['busy'].values()),
                    'incorrect common time window or assigned card counter')
            for pci, value in row['busy'].items():
                by_card.setdefault((node,pci), []).append(value)
                groups.setdefault(owners[node][1], []).append(value)
    values = [v for rows in by_card.values() for v in rows]
    complete = set(by_card) == all_cards
    mean = sum(values)/len(values) if complete else None
    return dict(request=requested, scheduled_epoch=scheduled, interval_s=INTERVAL_S, points=POINTS,
                expected_DCU=len(all_cards), planned_DCU=len(planned_cards), observed_DCU=len(by_card),
                sampled_workload_DCU=len(sampled_workload), clock_sync_independently_measured=False,
                observed_samples=len(values), expected_samples=len(all_cards)*POINTS, complete_coverage=complete,
                fleet_mean_busy_percent=mean, strictly_above_80_in_this_window=None if mean is None else mean > 80,
                by_group={group: dict(expected_samples=expected, observed_samples=len(groups.get(group,())),
                                     observed_mean_busy_percent=sum(groups[group])/len(groups[group]) if group in groups else None)
                          for group, expected in sorted(expected_group_samples.items())}, failed_sources=failures,
                missing_cards=[dict(node=node,pci=pci) for node,pci in sorted(all_cards-set(by_card))],
                sustained_utilization_qualified=False)


def read_frozen(path, expected=None):
    with path.open('rb') as stream:
        raw = stream.read(4*1024**2+1)
    require(0 < len(raw) <= 4*1024**2, 'original sampling metadata exceeds bound')
    digest = sha256(raw).hexdigest()
    require(expected is None or digest == expected, 'original metadata SHA-256 differs')
    return json.loads(raw), digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode',required=True)
    worker = modes.add_parser('sample')
    worker.add_argument('binding',type=Path)
    worker.add_argument('binding_sha256')
    worker.add_argument('output',type=Path)
    reader = modes.add_parser('aggregate')
    for name in ('ledger','pins','plan','report_dir','output'):
        reader.add_argument(name,type=Path)
    reader.add_argument('plan_sha256')
    args = parser.parse_args()
    if args.mode == 'sample':
        binding, digest = read_frozen(args.binding,args.binding_sha256)
        step = os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        def expired(*_):
            raise TimeoutError('in-step busy sample expired')
        signal.signal(signal.SIGALRM,expired)
        signal.setitimer(signal.ITIMER_REAL, 68)
        result = sample(binding,digest,socket.gethostname().split('.')[0],step,boot)
        ok = result['error'] is None
    else:
        plan, plan_sha = read_frozen(args.plan,args.plan_sha256)
        ledger, _ = read_frozen(args.ledger,plan['ledger_SHA256'])
        pins, _ = read_frozen(args.pins,plan['pci_pins_SHA256'])
        owners = node_map(ledger)
        require_pins(owners, pins)
        require(all(hex_token(digest,64) for digest in plan['bindings']), 'invalid binding receipt path')
        reports = {digest:read_frozen(args.report_dir/(digest+'.json'))[0] for digest in plan['bindings']
                   if (args.report_dir/(digest+'.json')).is_file()}
        result = aggregate(owners,pins,plan,reports)
        result['plan_SHA256'] = plan_sha
        ok = result['complete_coverage']
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,allow_nan=False)
        stream.write('\n')
    return int(not ok)


if __name__ == '__main__':
    raise SystemExit(main())
