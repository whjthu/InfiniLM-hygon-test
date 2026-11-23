"""Render one admitted pilot or sharded follow-up through the existing single parent."""

import argparse
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re
import time


PARENT = '1103606'
NODE_PATTERN = re.compile(r'f(?:12|13)r[1-4]n[0-9]{2}')
EXCLUDED = {'f13r2n01','f13r2n02','f13r2n03'}
BUILD_REQUEST = 'f564ba1c74c94b5aa6bd890874595d24'
PILOT_REQUEST = '0f8bc49a2a514f4187487a07dd9f0055'
ROOT = PurePosixPath('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
APPTAINER = '/public/software/apps/apptainer/1.3.4/bin/apptainer'
SANDBOX = str(ROOT/'containers/dtk26-sandbox-r1')
LEDGER = Path(__file__).resolve().parents[4]/'coordination/training/training80-active-node-task-map-v1.json'
LEDGER_SHA = '36361b92773f8eda412132ea176c29e814e4620ea8392ff287e68d854d67f5c9'


def need(condition, why):
    if not condition:
        raise ValueError(why)


def canonical(path):
    path = PurePosixPath(path)
    need(path.is_absolute() and path.is_relative_to(ROOT) and '..' not in path.parts, 'expected canonical project path')
    return str(path)


def known_nodes():
    raw = LEDGER.read_bytes()
    need(sha256(raw).hexdigest()==LEDGER_SHA,'current known surviving allocation ledger differs')
    groups = json.loads(raw)['groups']
    return {node for group in groups.values() for node in group['allocations'].get(PARENT, [])}


def pilot_gate(pilot):
    original, row = pilot['terminal'], pilot['accounting']
    need(pilot['step']==row[0]==original['step'] and re.fullmatch(PARENT+r'\.\d+',row[0])
         and row[2:4]==['COMPLETED','0:0'] and row[5]==original['node']
         and row[6] in ('cpu=12,gres/dcu=8,mem=16G,node=1','cpu=12,gres/dcu=8,mem=16384M,node=1')
         and original['request']==PILOT_REQUEST and original['complete'] is True
         and original['valid_cards']==8 and original['completed_arms']==32 and original['actual_pairs']==16
         and original['sharding_candidate'] is True
         and pilot['independent_post']['verified'] is True
         and pilot['independent_post']['node']==original['node']
         and re.fullmatch(PARENT+r'\.\d+',pilot['independent_post']['step'])
         and pilot['independent_post']['step']!=row[0]
         and re.fullmatch('[0-9a-f]{64}',pilot['independent_post']['SHA256']),
         'new numerical/performance pilot or independent physical release missing')


def derive(value, now=None):
    """Return launch argv only. Root owns publication, fresh admission and the queue."""
    now = time.time() if now is None else now
    step = value['kind']
    request = value['request']
    package, campaign = (canonical(value[key]) for key in ('package','campaign'))
    reservations = value['nodes']
    need(step in ('pilot','followup') and value['project_owned'] is True and value['fresh_protected_excluded'] is True
         and re.fullmatch('[0-9a-f]{32}',request) and 0<=now-value['epoch']<15
         and value['deadline_epoch']==value['epoch']+150 and campaign.endswith('-'+request),
         'wrong once-only campaign identity or already-expired epoch')
    nodes = sorted(reservations)
    need(nodes and set(nodes)<=known_nodes()-EXCLUDED and all(NODE_PATTERN.fullmatch(node) for node in nodes),
         'non-surviving parent or protected Deep node selected')
    if step=='pilot':
        need(request==PILOT_REQUEST and len(nodes)==1 and nodes[0]=='f12r2n13','fixed small pilot scope differs')
    else:
        need(request not in (BUILD_REQUEST,PILOT_REQUEST) and 1<=len(nodes)<=32,'new sharded scope differs')
        pilot_gate(value['pilot'])
    for node in nodes:
        record = reservations[node]
        need(record['node']==node and record['parent']==PARENT and record['request']==request
             and re.fullmatch(PARENT+r'\.\d+',record['pre_step'])
             and record['root_admitted'] is True and record['epoch']==value['epoch']
             and record['deadline_epoch']==value['deadline_epoch']
             and re.fullmatch('[0-9a-f]{64}',record['scope_SHA256']),
             'node admission differs from the intended shard')
    build = canonical(value['build_receipt'])
    sha = value['build_receipt_SHA256']
    need(re.fullmatch('[0-9a-f]{64}',sha),'unbound target build receipt')
    allocator = [APPTAINER,'exec','--rocm','--no-home','--pwd','/',
                 '--bind',f'{ROOT}:/refactor:ro','--bind',f'{package}:/headshare:ro',
                 '--bind',f'{campaign}:/scope','--bind',f'{build}:/build-receipt.json:ro',
                 SANDBOX,'/bin/bash','--noprofile','--norc','-c']
    work = (
        'set -euo pipefail; source /opt/dtk/env.sh; node=$(hostname -s); '
        'scope_sha=; for binding in $2; do '
        'if [[ "${binding%%:*}" == "$node" ]]; then scope_sha="${binding#*:}"; fi; done; test -n "$scope_sha"; '
        'exec /usr/bin/python3 -B -u /headshare/device.py --scope "/scope/$node.json" '
        '--scope-sha256 "$scope_sha" --out "/scope/result-$node" '
        '--build-receipt /build-receipt.json --build-receipt-sha256 "$1"'
    )
    shards=[]
    for index in range(0,len(nodes),4):
        subset=nodes[index:index+4]
        count=str(len(subset))
        label=f'GLM474-fresh-kv-h2-{request[:12]}-{index//4:02d}'
        scope_pins=' '.join(node+':'+reservations[node]['scope_SHA256'] for node in subset)
        argv=['srun','--jobid='+PARENT,'--job-name='+label,'--exact','--immediate=5','--time=00:03:00',
              '--nodes='+count,'--ntasks='+count,'--ntasks-per-node=1','--nodelist='+','.join(subset),
              '--kill-on-bad-exit=0','--cpus-per-task=12','--mem=16384M','--gres=dcu:8',
              '--output='+campaign+'/'+label+'-%N.stdout','--error='+campaign+'/'+label+'-%N.stderr',
              *allocator,work,'headshare',sha,scope_pins]
        shards.append(dict(label=label,nodes=subset,argv=argv,request=sha256((request+':'+label).encode()).hexdigest()[:32],
                           planned_cards=8*len(subset),requires_fresh_queue_reservation=True,
                           reconcile_only_on_unknown=True))
    return dict(request=request,parent=PARENT,shards=shards,planned_nodes=len(nodes),planned_cards=8*len(nodes),
                deadline_epoch=value['deadline_epoch'],submissions_performed=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root_admitted_plan',type=Path)
    print(json.dumps(derive(json.loads(parser.parse_args().root_admitted_plan.read_bytes())),sort_keys=True))
