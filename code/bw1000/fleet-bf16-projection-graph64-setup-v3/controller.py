"""Finite Graph64 expansion. Cooperative stop inspection never sends a signal itself."""

import argparse
import base64
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
import zlib


ROOT=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
HERE=Path(globals().get('__file__','<stdin>')).resolve().parent
REQUEST='ce9cd75da7d17f811e78e0f88a9d963d'
RUN='O01-resident-BF16-graph64-setup-v3-'+REQUEST
HOST_PYTHON='/usr/local/python3.12/bin/python3'
APPTAINER='/public/software/apps/apptainer/1.3.4/bin/apptainer'
REUSED={
    'graph32_controller.py':'389fa359a7c685d7db59424c950fe836a26414b928911b5dd09c72efd33b06d9',
    'report.py':'da4237c96a2ab250715961f308be5866f354ff967e7b3f0842d97954090ad6bc',
    'sync_dispatch.py':'3c13176628661ddc2474fa96b15f37da1c33cba49ad5ecf1493a8711587b7aaa',
    'admission.py':'fc89becc1b18bbab71e9a0c1dcdb05482b00643f7dff8cdfcbe1f2dd265291e5',
    'ledger.json':'36361b92773f8eda412132ea176c29e814e4620ea8392ff287e68d854d67f5c9',
}
FILES=set(REUSED)|{'worker.py','scope.py','controller.py','SCOPE.json','SOURCE_FILES.json',
                   'contention_worker.py','duplex_worker.py','hip_runtime.py','schedule.py',
                   'node_sample.py','qualified_sampler.py','setup_gate.py'}
HIP_SHA='b125b6b4153f5d25dd8e445b43b2f57b0eab9b6305eb765430a82a4359e1870b'


def need(ok,reason):
    if not ok:
        raise ValueError(reason)


def sha(raw):
    return sha256(raw).hexdigest()


def identity(files):
    return sha(json.dumps({n:sha(raw) for n,raw in sorted(files.items())},separators=(',',':')).encode())


def verify_source(files):
    need(set(files)==FILES and all(0<len(raw)<=32768 for raw in files.values()),'staged file scope differs')
    manifest=json.loads(files['SOURCE_FILES.json'])['files']
    need(set(manifest)==FILES-{'SOURCE_FILES.json'},'staged filename manifest differs')
    for name,spec in manifest.items():
        need(len(files[name])==spec['bytes'] and sha(files[name])==spec['SHA256'],name+' source differs')
    need(all(sha(files[name])==pin for name,pin in REUSED.items()),'frozen reused logic differs')


def unpack(packed):
    files={}
    for name,(pin,data) in packed.items():
        need(name in FILES and len(data)<=46000,'encoded source scope differs')
        inflater=zlib.decompressobj()
        raw=inflater.decompress(base64.b64decode(data,validate=True),32769)
        need(len(raw)<=32768 and inflater.eof and not inflater.unused_data and sha(raw)==pin,
             'original source bytes differ or compression envelope exceeded')
        files[name]=raw
    verify_source(files)
    return files


def proc_identity(pid):
    path=Path('/proc')/str(pid)
    stat=(path/'stat').read_text().rsplit(')',1)[1].split()
    argv=(path/'cmdline').read_bytes().split(b'\0')[:-1]
    argv=[os.fsdecode(item) for item in argv]
    need(path.stat().st_uid==os.getuid() and argv and Path(argv[0]).name=='srun','owned srun process differs')
    return dict(pid=pid,start_ticks=int(stat[19]),uid=os.getuid(),argv=argv)


def command(target,run,label,unit):
    parent,nodes=unit['parent'],unit['nodes']
    entries=str(len(nodes))
    runtime=[APPTAINER,'exec','--rocm','--no-home','--pwd','/',
             '--bind',str(run)+':'+str(run),'--bind',str(target)+':'+str(target)+':ro',
             '--bind','/opt/hyhal:/opt/hyhal:ro',str(ROOT/'containers/dtk26-sandbox-r1'),
             '/bin/bash','--noprofile','--norc','-c',
             'set -eo pipefail; source /opt/dtk/env.sh; exec python3 -B -u "$@"',
             'graph64-fleet',str(target/'worker.py'),str(run/'nodes'),str(run/'worker-plan.json')]
    return ['srun','--jobid='+parent,'--job-name='+label,'--exact','--nodes='+entries,'--ntasks='+entries,
            '--ntasks-per-node=1','--nodelist='+','.join(nodes),'--immediate=5','--kill-on-bad-exit=0',
            '--cpus-per-task=12','--mem=65536M','--gres=dcu:8','--time=00:02:20',
            '--output='+str(run/(label+'-%N.stdout')),'--error='+str(run/(label+'-%N.stderr')),*runtime]


def dispatch(target,plan,owners,scheduler,save):
    epoch,shards=plan['campaign_epoch'],plan['shards']
    need(0<=time.time()-epoch<25,'original launch epoch expired')
    need(not scheduler.account(shards),'original label exists; reconcile')
    parents={unit['parent'] for unit in shards.values()}
    occupied={parent:scheduler.allocation(parent,sorted(n for n,(p,_) in owners.items() if p==parent))
              for parent in sorted(parents)}
    need((ROOT/'containers/dtk26-sandbox-r1').is_dir() and os.access(APPTAINER,os.X_OK),'pinned Torch sandbox missing')
    run=ROOT/'results'/RUN
    run.mkdir(mode=0o700,exist_ok=False)
    for child in ('nodes','attempts'):
        (run/child).mkdir(mode=0o700)
    save(run/'worker-plan.json',plan)
    save(run/'intent.json',dict(plan=plan,replay_allowed=False,planned_cards=8*len(plan['selected'])))
    children={}
    for label,unit in shards.items():
        attempt=run/'attempts'/label
        attempt.mkdir(mode=0o700)
        common=dict(request=unit['request'],parent=unit['parent'],zone=unit['zone'],nodes=unit['nodes'],label=label)
        if set(unit['nodes'])&occupied[unit['parent']] or time.time()>=epoch+40:
            save(attempt/'intent.json',dict(common,attempted=False,reason='new scheduler conflict or startup elapsed'))
            continue
        argv=command(target,run,label,unit)
        save(attempt/'intent.json',dict(common,attempted=True,argv=argv))
        try:
            child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,start_new_session=True)
        except OSError as error:
            save(attempt/'spawn-error.json',dict(type=type(error).__name__))
            continue
        children[label]=child
        try:
            saved=proc_identity(child.pid)
            need(saved['argv']==argv,'owned srun argv differs')
            save(attempt/'owner.json',dict(common,**saved))
        except (OSError,ValueError) as error:
            save(attempt/'owner-unavailable.json',dict(type=type(error).__name__))
    for child in children.values():
        try:
            child.wait(timeout=max(.01,epoch+166-time.time()))
        except subprocess.TimeoutExpired:
            break
    return dict(request=REQUEST,planned_cards=8*len(plan['selected']),planned_shards=len(shards),
                observed_accounting=scheduler.account(shards),client_count=len(children),replay_allowed=False)


def account_scope(row,label,unit,scheduler):
    nodes=unit['nodes']
    need(len(row)==7 and row[1]==label and re.fullmatch(unit['parent']+r'\.[0-9]+',row[0])
         and set(scheduler.hosts(row[5]))==set(nodes),'original Slurm shard or node scope differs')
    tres=dict(field.split('=',1) for field in row[6].split(',') if '=' in field)
    n=len(nodes)
    return (tres.get('node')==str(n) and tres.get('cpu')==str(12*n) and tres.get('gres/dcu')==str(8*n)
            and tres.get('mem') in (f'{64*n}G',f'{65536*n}M'))


def worker_intent(path,node,step,plan,read):
    raw=read(path)
    row=json.loads(raw)
    start=plan['shared_start_epoch']
    grid=[[mode,round((start+offset)*10**9),round((start+offset+8)*10**9)]
          for mode,offset in zip(('eager','graph','graph','eager'),(0,12,24,36),strict=True)]
    need((row['request'],row['node'],row['step'])==(REQUEST,node,step) and row['grid']==grid
         and len(row['selected_PCI'])==8 and set(row['selected_PCI'])==set(plan['selected'][node]['PCI']),
         'original worker intent does not bind the shard, schedule or fresh PCI')
    return sha(raw)


def phase_diagnostic(raw,node,row,start):
    report=json.loads(raw)
    need((report['request'],report['node'],report['step'])==(REQUEST,node,row[0]),
         'setup report does not bind the original worker')
    grid=report['grid']
    expected=[[mode,start+offset*10**9,start+(offset+8)*10**9]
              for mode,offset in zip(('eager','graph','graph','eager'),(0,12,24,36),strict=True)]
    need(grid==expected and len(report['arm_gates'])==4 and type(report['process_enter_epoch_ns']) is int and
         report['process_enter_epoch_ns']<start and sorted(card['card'] for card in report['cards'])==list(range(8)),
         'setup report does not keep eight planned cards and fixed windows')
    gates=[report['setup_gate'],*report['arm_gates']]
    all_released=True
    for gate,cutoff in zip(gates,[start-2*10**9,*[item[1]-200_000_000 for item in grid]],strict=True):
        arrivals=gate['arrived_epoch_ns']
        need(gate['cutoff_epoch_ns']==cutoff and gate['expected_cards']==8
             and set(arrivals)<=set('01234567') and all(type(value) is int and value<cutoff for value in arrivals.values()),
             'graph phase readiness altered its cutoff or owner denominator')
        released=gate['released_epoch_ns']
        if released is None:
            all_released=False
        else:
            need(gate['first_failure'] is None and len(arrivals)==8 and max(arrivals.values())==released<cutoff,
                 'graph phase released without eight timely participants')
    return dict(report,all_released=all_released,raw_SHA256=sha(raw))


def bound_node(path,node,row,plan,read,compare):
    original=worker_intent(path/'intent.json',node,row[0],plan,read)
    audit=compare(path)
    need(audit['raw_SHA256'].get('intent.json')==original,'worker intent changed during audit')
    if audit['passed']:
        for card in range(8):
            name=f'card-{card}.json'
            raw=read(path/name)
            if sha(raw)!=audit['raw_SHA256'].get(name) or json.loads(raw).get('provider_SHA256')!=HIP_SHA:
                audit['passed']=False
                audit['failures'].append('original HIP provider differs for '+name)
    try:
        setup=phase_diagnostic(read(path/'setup-summary.json'),node,row,round(plan['shared_start_epoch']*10**9))
        audit['setup_phase']=setup
        if audit['passed']:
            need(setup['all_released'],'timed audit requires all node-local phase gates')
    except (OSError,ValueError,KeyError,TypeError) as error:
        audit['passed']=False
        audit['failures'].append('setup phase '+str(error)[:120])
    return audit


def reconcile(plan,scheduler,read,compare):
    run=ROOT/'results'/RUN
    shards=plan['shards']
    actual=scheduler.account(shards)
    nodes={name:dict(valid_cards=0,completed_arms=0,actual_busy_card_samples=0,passed=False) for name in plan['selected']}
    result={}
    for label,unit in shards.items():
        item=dict(unit,state='UNKNOWN',terminal=False,device_qualified=False,independent_post_required=True,
                  planned_cards=len(unit['nodes'])*8)
        result[label]=item
        attempt=run/'attempts'/label/'intent.json'
        if attempt.is_file():
            raw=read(attempt)
            item['attempt_SHA256']=sha(raw)
            item['attempted']=json.loads(raw)['attempted']
        row=actual.get(label)
        if not row:
            continue
        exact=account_scope(row,label,unit,scheduler)
        state=row[2].split()[0].rstrip('+')
        item.update(accounting=row,state=state,terminal=state in scheduler.TERMINAL,resources_match=exact)
        if not item['terminal']:
            continue
        for name in unit['nodes']:
            try:
                nodes[name]=bound_node(run/'nodes'/name,name,row,plan,read,compare)
            except (OSError,ValueError,KeyError,TypeError) as error:
                nodes[name]['failure']=str(error)[:120]
        item['device_qualified']=(exact and row[2:4]==['COMPLETED','0:0'] and row[4].isdecimal()
                                  and 0<int(row[4])<=140 and all(nodes[n]['passed'] for n in unit['nodes']))
    count=len(nodes)
    overall=dict(request=REQUEST,planned_nodes=count,planned_cards=count*8,planned_arms=count*32,
                 planned_busy_card_samples=count*128,shards=result,nodes=nodes,
                 valid_cards=sum(item['valid_cards'] for item in nodes.values()),
                 completed_arms=sum(item['completed_arms'] for item in nodes.values()),
                 actual_busy_card_samples=sum(item['actual_busy_card_samples'] for item in nodes.values()),
                 device_qualified=all(shard['device_qualified'] for shard in result.values()),
                 clock_inter_node_bounded=False,collective_executed=False,replay_allowed=False)
    if overall['device_qualified']:
        ratios=[x for node in nodes.values() for pair in node['per_card_graph_over_eager_pairs'].values() for x in pair]
        overall['median_graph_over_eager_card_pairs']=statistics.median(ratios)
    return overall


def inspect_stop(plan,label,request,step,owners,scheduler,read,fields):
    unit=plan['shards'].get(label)
    need(unit is not None and unit['request']==request and re.fullmatch(unit['parent']+r'\.[0-9]+',step),
         'stop request is not an exact original numbered shard')
    run=ROOT/'results'/RUN
    path=run/'attempts'/label
    observed=json.loads(read(path/'intent.json'))
    owner=json.loads(read(path/'owner.json'))
    expected=dict(request=request,parent=unit['parent'],zone=unit['zone'],nodes=unit['nodes'],label=label)
    need(observed['attempted'] is True and all(observed[k]==owner[k]==v for k,v in expected.items()),
         'owner receipt differs from the original shard intent')
    local=proc_identity(owner['pid'])
    need(all(local[k]==owner[k] for k in ('pid','start_ticks','uid','argv'))
         and local['argv']==observed['argv'],'owned srun PID/startticks/command changed')
    all_nodes=sorted(n for n,(parent,_) in owners.items() if parent==unit['parent'])
    need(set(unit['nodes'])<=scheduler.allocation(unit['parent'],all_nodes),'parent no longer owns exact active shard')
    rows=scheduler.query('scontrol','show','step',step,'-o')
    need(len(rows)==1,'live exact step missing or ambiguous')
    live=fields(rows[0])
    need(live.get('StepId')==step and set(scheduler.hosts(live['NodeList']))==set(unit['nodes']),
         'live numbered step scope differs')
    row=scheduler.account({label}).get(label)
    need(row is not None and row[0]==step and row[2].split()[0]=='RUNNING'
         and account_scope(row,label,unit,scheduler),'actual accounting does not show the owned running shard')
    intents={node:worker_intent(run/'nodes'/node/'intent.json',node,step,plan,read) for node in unit['nodes']}
    need(proc_identity(owner['pid'])==local,'owned srun changed during stop inspection')
    return dict(request=request,label=label,step=step,owner=local,worker_intent_SHA256=intents,
                signal_argv=['scancel','--signal=USR1',step],read_only=True,
                recheck_before_signal=True,independent_terminal_and_post_required=True)


def receive(payload):
    os.umask(0o077)
    plan=payload['plan']
    need(plan['request']==REQUEST and 0<=time.time()-plan['campaign_epoch']<25,'original receive epoch expired')
    files=unpack(payload['files'])
    need(identity(files)==plan['source_SHA256'],'source pin does not match campaign')
    target=ROOT/'incoming'/RUN
    target.mkdir(mode=0o700,exist_ok=False)
    for name,raw in files.items():
        with (target/name).open('xb') as output:
            output.write(raw)
    with (target/'plan.json').open('x',encoding='utf8') as out:
        json.dump(plan,out,sort_keys=True,allow_nan=False)
        out.flush()
        os.fsync(out.fileno())
    result=subprocess.run([HOST_PYTHON,'-I','-S','-B','-u',str(target/'controller.py'),'--dispatch'],
                          stdin=subprocess.DEVNULL,timeout=max(.01,plan['campaign_epoch']+180-time.time()))
    return result.returncode


def main():
    if 'PAYLOAD' in globals():
        return receive(globals()['PAYLOAD'])
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--dispatch',action='store_true')
    modes.add_argument('--reconcile',action='store_true')
    modes.add_argument('--inspect-stop',nargs=3,metavar=('LABEL','SHARD_REQUEST','STEP'))
    args=parser.parse_args()
    need(HERE==ROOT/'incoming'/RUN,'unique staged controller required')
    files={name:(HERE/name).read_bytes() for name in FILES}
    verify_source(files)
    plan=json.loads((HERE/'plan.json').read_bytes())
    need(identity(files)==plan['source_SHA256'],'original source envelope differs')
    sys.path.insert(0,str(HERE))
    from graph32_controller import read,save
    from admission import fields
    from scope import ledger,validate_plan
    import sync_dispatch
    from report import compare

    owners=ledger(files['ledger.json'])
    validate_plan(plan,owners)
    if args.dispatch:
        result=dispatch(HERE,plan,owners,sync_dispatch,save)
    elif args.reconcile:
        result=reconcile(plan,sync_dispatch,read,compare)
    else:
        result=inspect_stop(plan,*args.inspect_stop,owners,sync_dispatch,read,fields)
    print(json.dumps(result,sort_keys=True,allow_nan=False))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
