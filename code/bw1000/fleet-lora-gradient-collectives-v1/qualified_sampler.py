"""Sample and aggregate the 640-DCU allocation's sysfs busy counters."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import socket
import time

PARENTS = {'1087656', '1103606'}
POINTS = 3
INTERVAL_S = 2
MAX_LAG_S = 1.0
PCI = re.compile(r'[0-9a-f]{4,8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]\Z')


def require(ok, detail):
    if not ok:
        raise ValueError(detail)


def node_map(ledger):
    owners = {}
    counts = {parent: 0 for parent in PARENTS}
    for group, row in ledger['groups'].items():
        require(set(row['allocations']) == PARENTS, 'parent set differs from the allocation ledger')
        for parent, nodes in row['allocations'].items():
            for node in nodes:
                require(node not in owners, 'node assigned twice')
                owners[node] = (parent, group)
                counts[parent] += 1
    require(set(counts.values()) == {40} and len(owners) == 80, 'expected two disjoint 40-node jobs')
    return owners


def require_pins(owners, pins):
    require(set(pins) == set(owners), 'device pins do not cover all expected nodes')
    for names in pins.values():
        require(len(names) == len(set(names)) == 8 and all(PCI.fullmatch(p) for p in names),
                'expected eight unique PCI identifiers per node')


def collect(drm, expected, start, read_clock=time.time, sleep=time.sleep):
    paths = {}
    for entry in sorted(drm.glob('renderD*')):
        device = entry / 'device'
        pci = device.resolve(strict=True).name.lower()
        require(pci not in paths, 'duplicate render device PCI address')
        paths[pci] = device
    require(set(paths) == set(expected), 'visible DRM device PCI set changed')
    rows = []
    for point in range(POINTS):
        planned = start + point * INTERVAL_S
        delay = planned - read_clock()
        if delay > 0:
            sleep(delay)
        opened = read_clock()
        busy = {pci: int((paths[pci] / 'gpu_busy_percent').read_text()) for pci in expected}
        closed = read_clock()
        require(planned <= opened <= closed <= planned + MAX_LAG_S, 'counter read missed the sampling window')
        require(all(0 <= value <= 100 for value in busy.values()), 'invalid sysfs percentage')
        rows.append(dict(point=point, opened_epoch=opened, closed_epoch=closed, busy=busy))
    return rows


def aggregate(owners, pins, by_node, start):
    require_pins(owners, pins)
    require(set(by_node) <= set(owners), 'unknown node in sample set')
    missing = sorted(set(owners) - set(by_node))
    groups = {}
    values = []
    steps = {}
    for node, report in by_node.items():
        parent, group = owners[node]
        require(report['node'] == node and re.fullmatch(parent + r'\.\d+',report['step']),
                'sample did not originate in expected parent')
        steps.setdefault(parent,set()).add(report['step'])
        require(len(report['rows']) == POINTS, 'missing sample point')
        group_values = groups.setdefault(group, [])
        for i, row in enumerate(report['rows']):
            planned = start + i * INTERVAL_S
            require(row['point'] == i and planned <= row['opened_epoch'] <= row['closed_epoch'] <= planned + MAX_LAG_S,
                    'sample does not match the common wall-clock window')
            require(set(row['busy']) == set(pins[node]) and
                    all(type(x) is int and 0 <= x <=100 for x in row['busy'].values()), 'invalid PCI coverage or busy')
            group_values.extend(row['busy'].values())
            values.extend(row['busy'].values())
    require(all(len(ids) == 1 for ids in steps.values()), 'mixed Slurm steps within an allocation')
    complete = not missing
    return dict(expected_DCU=640, observed_DCU=len(by_node)*8, missing_nodes=missing,
                complete=complete, points=POINTS, interval_s=INTERVAL_S, period_s=(POINTS-1)*INTERVAL_S,
                scheduled_epoch=start, sample_count=len(values), expected_samples=640*POINTS,
                fleet_mean_busy_percent=sum(values)/len(values) if complete else None,
                observed_devices_mean_busy_percent=sum(values)/len(values) if values else None,
                steps={p:sorted(ids) for p,ids in steps.items()},
                groups={g:dict(samples=len(rows),observed_devices_mean_busy_percent=sum(rows)/len(rows))
                        for g,rows in sorted(groups.items())},
                scope='Sysfs gpu_busy_percent over three shared wall-clock points in the recorded window.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('sample','aggregate'))
    p.add_argument('ledger',type=Path)
    p.add_argument('pci_pins',type=Path)
    p.add_argument('scheduled_epoch',type=float)
    p.add_argument('node_dir',type=Path)
    p.add_argument('--summary',type=Path)
    args = p.parse_args()
    require(math.isfinite(args.scheduled_epoch) and args.scheduled_epoch > 0, 'invalid sampling epoch')
    owners = node_map(json.loads(args.ledger.read_bytes()))
    pins = json.loads(args.pci_pins.read_bytes())
    if args.mode == 'sample':
        node = socket.gethostname().split('.')[0]
        parent = os.environ['SLURM_JOB_ID']
        step = os.environ['SLURM_STEP_ID']
        require(node in owners and parent == owners[node][0] and step.isdecimal(), 'unexpected Slurm ownership')
        data = dict(node=node,step=parent+'.'+step,
                    rows=collect(Path('/sys/class/drm'),pins[node],args.scheduled_epoch))
        with (args.node_dir / (node + '.json')).open('x',encoding='utf-8') as f:
            json.dump(data,f,allow_nan=False)
        return
    require(args.summary is not None, 'summary path required')
    by_node = {}
    for node in owners:
        path = args.node_dir/(node+'.json')
        if path.is_file():
            by_node[node] = json.loads(path.read_bytes())
    result = aggregate(owners,pins,by_node,args.scheduled_epoch)
    with args.summary.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(result,f,indent=2,allow_nan=False)
        f.write('\n')
    return 0 if result['complete'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
