"""Two independently accountable O01 srun steps, sharing the original arm grid."""

import argparse
import base64
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
import zlib


ROOT = Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
HERE = Path(globals().get('__file__','<stdin>')).resolve().parent
REQUEST = '1f31f75efa1c9568e8f6b3173a383d8f'
RUN = 'O01-resident-BF16-graph64-32card-v1-' + REQUEST
APPTAINER = '/public/software/apps/apptainer/1.3.4/bin/apptainer'
HOST_PYTHON = '/usr/local/python3.12/bin/python3'
NANO = 10**9
HIP_SHA256 = 'b125b6b4153f5d25dd8e445b43b2f57b0eab9b6305eb765430a82a4359e1870b'
TERMINAL = {'COMPLETED','FAILED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
REUSED = {
    'contention_worker.py': '4fe6194201848a0b065abdde41beea71400e02aac2b3c64f985b7a9470d305ad',
    'duplex_worker.py': '48310c24b6a0582a65dd8b5fa02d216784a5ae3b9dd57c570aed13b7fe6adb7f',
    'hip_runtime.py': 'e551c1faa38e6472be7861a572354c72311a9b21d4ad2e73d7bb0a3fa403d534',
    'node_sample.py': 'e10a17f576da010cd50937a8f91eac1c8f66098dde6d887c8373d00a615c8931',
    'qualified_sampler.py': '09fc94b7f527c159bd489bacbf7718dad8615ba61cd3f361a44f441da14a0430',
    'schedule.py': 'ead627410a66ae1b278a31fb45d30f1b44b8c366c47eed74ad3fd5855fb62824',
    'report.py': 'da4237c96a2ab250715961f308be5866f354ff967e7b3f0842d97954090ad6bc',
    'sync_dispatch.py': '3c13176628661ddc2474fa96b15f37da1c33cba49ad5ecf1493a8711587b7aaa',
    'admission.py': 'fc89becc1b18bbab71e9a0c1dcdb05482b00643f7dff8cdfcbe1f2dd265291e5',
    'ledger.json': '36361b92773f8eda412132ea176c29e814e4620ea8392ff287e68d854d67f5c9',
}
FILES = set(REUSED)|{'worker.py','scope.py','controller.py','SCOPE.json','SOURCE_FILES.json'}


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha(raw):
    return sha256(raw).hexdigest()


def read(path, limit=128*1024):
    need(path.resolve(strict=True)==path and path.is_file(),'original file is not canonical')
    with path.open('rb') as stream:
        raw = stream.read(limit+1)
    need(0<len(raw)<=limit,'original file is outside its byte envelope')
    return raw


def save(path, value):
    with path.open('x',encoding='utf8') as stream:
        json.dump(value,stream,sort_keys=True,allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def source_identity(files):
    return sha(json.dumps({name:sha(raw) for name,raw in sorted(files.items())},separators=(',',':')).encode())


def source_verify(files):
    need(set(files)==FILES and all(0<len(raw)<=32*1024 for raw in files.values()),'staged source membership differs')
    pins = json.loads(files['SOURCE_FILES.json'])['files']
    need(set(pins)==FILES-{'SOURCE_FILES.json'},'manifest member names differ')
    for name,spec in pins.items():
        raw = files[name]
        need(sha(raw)==spec['SHA256'] and len(raw)==spec['bytes'],name+' differs from manifest')
    need(all(sha(files[name])==expected for name,expected in REUSED.items()),'frozen shared module differs')


def command(staged, result, parent, nodes, label):
    need(parent in ('1087656','1103606') and len(set(nodes))==len(nodes)==2,'expected two independent worker nodes')
    runtime = [APPTAINER,'exec','--rocm','--no-home','--pwd','/',
               '--bind',str(result)+':'+str(result),'--bind',str(staged)+':'+str(staged)+':ro',
               '--bind','/opt/hyhal:/opt/hyhal:ro',str(ROOT/'containers/dtk26-sandbox-r1'),
               '/bin/bash','--noprofile','--norc','-c',
               'set -eo pipefail; source /opt/dtk/env.sh; exec python3 -B -u "$@"',
               'bf16-graph64-32card',str(staged/'worker.py'),str(result/'nodes'),str(result/'worker-plan.json')]
    return ['srun','--jobid='+parent,'--job-name='+label,'--exact','--nodes=2','--ntasks=2',
            '--ntasks-per-node=1','--nodelist='+','.join(nodes),'--immediate=5','--kill-on-bad-exit=0',
            '--cpus-per-task=12','--mem=65536M','--gres=dcu:8','--time=00:02:20',
            '--output='+str(result/(label+'-%N.stdout')),'--error='+str(result/(label+'-%N.stderr')),*runtime]


def dispatch(target, plan, groups, owners, scheduler, label_for):
    epoch = plan['campaign_epoch']
    need(0<=time.time()-epoch<25,'campaign startup expired')
    labels = {label_for(parent):parent for parent in groups}
    need(not scheduler.account(labels),'original label already exists; reconcile')
    for parent,nodes in groups.items():
        available_parent_nodes = sorted(node for node,(job,_) in owners.items() if job==parent)
        need(not set(nodes)&scheduler.allocation(parent,available_parent_nodes),
             'a planned node acquired an active step; original 32-card scope is retained')
    need((ROOT/'containers/dtk26-sandbox-r1').is_dir() and os.access(APPTAINER,os.X_OK),'qualified Torch sandbox missing')
    result = ROOT/'results'/RUN
    result.mkdir(mode=0o700,exist_ok=False)
    (result/'nodes').mkdir(mode=0o700)
    save(result/'worker-plan.json',plan)
    save(result/'intent.json',dict(plan=plan,groups=groups,labels=labels,planned_cards=32,replay_allowed=False))
    owned = {}
    for label,parent in labels.items():
        if time.time()>=epoch+40:
            save(result/(label+'.attempt.json'),dict(attempted=False,reason='fixed startup deadline'))
            continue
        argv = command(target,result,parent,groups[parent],label)
        save(result/(label+'.attempt.json'),dict(attempted=True,argv=argv))
        try:
            owned[label] = subprocess.Popen(argv,stdin=subprocess.DEVNULL,start_new_session=True)
        except OSError:
            continue
    save(result/'srun-pids.json',{label:child.pid for label,child in owned.items()})
    for child in owned.values():
        try:
            child.wait(timeout=max(.01,epoch+166-time.time()))
        except subprocess.TimeoutExpired:
            break
    return dict(request=REQUEST,planned_cards=32,client_count=len(owned),accounting=scheduler.account(labels),
                replay_allowed=False)


def bound_node(directory,node,row,plan,compare):
    intent_raw = read(directory/'intent.json')
    original = json.loads(intent_raw)
    start = plan['shared_start_epoch']
    names = ('eager','graph','graph','eager')
    expected_grid = [[name,round((start+offset)*NANO),round((start+offset+8)*NANO)]
                     for name,offset in zip(names,(0,12,24,36),strict=True)]
    need(original['step']==row[0] and original['node']==node and original['request']==REQUEST
         and original['grid']==expected_grid and set(original['selected_PCI'])==set(plan['selected'][node]['PCI']),
         'node intent differs from original dispatch binding')
    validated = compare(directory)
    need(validated['raw_SHA256'].get('intent.json')==sha(intent_raw),'original node intent changed during read')
    if validated['passed']:
        for card in range(8):
            filename=f'card-{card}.json'
            raw=read(directory/filename)
            if (sha(raw)!=validated['raw_SHA256'].get(filename) or
                    json.loads(raw).get('provider_SHA256')!=HIP_SHA256):
                validated['passed']=False
                validated['failures'].append('original HIP provider differs for '+filename)
    return validated


def reconcile(plan, groups, scheduler, label_for, compare):
    result = ROOT/'results'/RUN
    labels = {label_for(parent):parent for parent in groups}
    account = scheduler.account(labels)
    nodes_total = {node:dict(valid_cards=0,completed_arms=0,actual_busy_card_samples=0,passed=False)
                   for nodes in groups.values() for node in nodes}
    shards = {}
    for label,parent in labels.items():
        nodes, row = groups[parent], account.get(label)
        item = dict(parent=parent,nodes=nodes,state='UNKNOWN',terminal=False,
                    independent_post_required=True,device_qualified=False)
        shards[label] = item
        if not row:
            continue
        need(len(row)==7 and row[1]==label and re.fullmatch(parent+r'\.[0-9]+',row[0])
             and set(scheduler.hosts(row[5]))==set(nodes),'original accounting does not bind its two nodes')
        state = row[2].split()[0].rstrip('+')
        item.update(state=state,terminal=state in TERMINAL,accounting=row)
        if not item['terminal']:
            continue
        tres = dict(part.split('=',1) for part in row[6].split(',') if '=' in part)
        item['exact_16_DCU_allocation'] = (tres.get('cpu')=='24' and tres.get('node')=='2' and tres.get('gres/dcu')=='16'
                                          and tres.get('mem') in ('128G','131072M') and row[4].isdecimal()
                                          and 0<int(row[4])<=140)
        for node in nodes:
            try:
                nodes_total[node] = bound_node(result/'nodes'/node,node,row,plan,compare)
            except (OSError,ValueError,KeyError,TypeError) as error:
                nodes_total[node]['failure'] = str(error)[:120]
        item['device_qualified'] = (row[2:4]==['COMPLETED','0:0'] and item['exact_16_DCU_allocation']
                                    and all(nodes_total[n]['passed'] for n in nodes))
    output = dict(request=REQUEST,expected_nodes=4,planned_cards=32,expected_arms=128,expected_busy_card_samples=512,
                  valid_cards=sum(v['valid_cards'] for v in nodes_total.values()),
                  completed_arms=sum(v['completed_arms'] for v in nodes_total.values()),
                  actual_busy_card_samples=sum(v['actual_busy_card_samples'] for v in nodes_total.values()),
                  shards=shards,nodes=nodes_total,device_qualified=all(v['device_qualified'] for v in shards.values()),
                  clock_inter_node_bounded=False,collective_was_executed=False,independent_post_required=True,replay_allowed=False)
    if output['device_qualified']:
        ratios = [ratio for n in nodes_total.values() for pair in n['per_card_graph_over_eager_pairs'].values() for ratio in pair]
        output['median_graph_over_eager_across_card_pairs'] = statistics.median(ratios)
    return output


def unpack(packed):
    files = {}
    for name,(expected,encoded) in packed.items():
        need(name in FILES and len(encoded)<=46000,'staged file encoding outside envelope')
        decoder=zlib.decompressobj()
        raw=decoder.decompress(base64.b64decode(encoded,validate=True),32769)
        need(len(raw)<=32768 and decoder.eof and not decoder.unused_data,'staged file compression outside envelope')
        need(sha(raw)==expected,'payload member SHA differs')
        files[name]=raw
    source_verify(files)
    return files


def receive(payload):
    os.umask(0o077)
    plan = payload['plan']
    need(plan['request']==REQUEST and 0<=time.time()-plan['campaign_epoch']<25,'original receive epoch expired')
    files=unpack(payload['files'])
    need(source_identity(files)==plan['source_SHA256'],'source SHA differs')
    target=ROOT/'incoming'/RUN
    target.mkdir(mode=0o700,exist_ok=False)
    for name,raw in files.items():
        with (target/name).open('xb') as stream:
            stream.write(raw)
    save(target/'plan.json',plan)
    result=subprocess.run([HOST_PYTHON,'-I','-S','-B','-u',str(target/'controller.py'),'--dispatch'],
                          stdin=subprocess.DEVNULL,timeout=max(.01,plan['campaign_epoch']+180-time.time()))
    return result.returncode


def main():
    if 'PAYLOAD' in globals():
        return receive(globals()['PAYLOAD'])
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--dispatch',action='store_true')
    group.add_argument('--reconcile',action='store_true')
    args=parser.parse_args()
    need(HERE==ROOT/'incoming'/RUN,'unique staged controller required')
    files={name:read(HERE/name,32*1024) for name in FILES}
    source_verify(files)
    plan=json.loads(read(HERE/'plan.json'))
    need(source_identity(files)==plan['source_SHA256'],'original source envelope differs')
    sys.path.insert(0,str(HERE))
    from scope import frozen_ledger,shard_label,validate_plan
    import sync_dispatch
    from report import compare
    owners=sync_dispatch.node_map(frozen_ledger(files['ledger.json'],json.loads(files['ledger.json'])))
    groups=validate_plan(plan)
    outcome=(dispatch(HERE,plan,groups,owners,sync_dispatch,shard_label) if args.dispatch
             else reconcile(plan,groups,sync_dispatch,shard_label,compare))
    print(json.dumps(outcome,sort_keys=True,allow_nan=False))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
