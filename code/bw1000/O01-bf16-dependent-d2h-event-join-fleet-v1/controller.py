"""One finite eligible-fleet projection and pinned D2H interaction. Cooperative stop inspection never sends a signal itself."""

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
REQUEST='53ec0f52789d442fa6a9c48b00c759e1'
RUN='O01-BF16-dependent-d2h-event-join-fleet-v1-'+REQUEST
HOST_PYTHON='/usr/local/python3.12/bin/python3'
CONTROLLER_SECONDS=545
SLURM_TIME='00:08:25'
SLURM_SECONDS=505
APPTAINER='/public/software/apps/apptainer/1.3.4/bin/apptainer'
REUSED={
    'graph32_controller.py':'389fa359a7c685d7db59424c950fe836a26414b928911b5dd09c72efd33b06d9',
    'report.py':'0f5f81c7fb307e633507ca15967e8ce3d6d4c50bf5a8d549b957d983883dc4ce',
    'sync_dispatch.py':'3c13176628661ddc2474fa96b15f37da1c33cba49ad5ecf1493a8711587b7aaa',
    'admission.py':'fc89becc1b18bbab71e9a0c1dcdb05482b00643f7dff8cdfcbe1f2dd265291e5',
    'ledger.json':'36361b92773f8eda412132ea176c29e814e4620ea8392ff287e68d854d67f5c9',
}
FILES=set(REUSED)|{'worker.py','scope.py','controller.py','SCOPE.json','SOURCE_FILES.json',
                   'contention_worker.py','duplex_worker.py','hip_runtime.py','schedule.py',
                   'node_sample.py','qualified_sampler.py','setup_gate.py','contract.py','dependent.py','transfer.py'}
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
             'dependent-projection',str(target/'worker.py'),str(run/'nodes'),str(run/'worker-plan.json')]
    return ['srun','--jobid='+parent,'--job-name='+label,'--exact','--nodes='+entries,'--ntasks='+entries,
            '--ntasks-per-node=1','--nodelist='+','.join(nodes),'--immediate=5','--kill-on-bad-exit=0',
            '--cpus-per-task=12','--mem=65536M','--gres=dcu:8','--time='+SLURM_TIME,
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
    for label,unit in sorted(shards.items(), key=lambda item: plan['cohort_by_shard'][item[0]]):
        attempt=run/'attempts'/label
        attempt.mkdir(mode=0o700)
        common=dict(request=unit['request'],parent=unit['parent'],zone=unit['zone'],nodes=unit['nodes'],label=label,
                    cohort=plan['cohort_by_shard'][label])
        if set(unit['nodes'])&occupied[unit['parent']] or time.time()>=epoch+40:
            save(attempt/'intent.json',dict(common,attempted=False,reason='new scheduler conflict or startup elapsed'))
            continue
        argv=command(target,run,label,unit)
        save(attempt/'intent.json',dict(common,attempted=True,argv=argv, spawn_intent_epoch_ns=time.time_ns()))
        try:
            child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,start_new_session=True)
        except OSError as error:
            save(attempt/'spawn-error.json',dict(type=type(error).__name__))
            continue
        children[label]=child
        try:
            saved=proc_identity(child.pid)
            need(saved['argv']==argv,'owned srun argv differs')
            save(attempt/'owner.json',dict(common,spawn_observed_epoch_ns=time.time_ns(),**saved))
        except (OSError,ValueError) as error:
            save(attempt/'owner-unavailable.json',dict(type=type(error).__name__))
    for child in children.values():
        try:
            child.wait(timeout=max(.01,epoch+527-time.time()))
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
    from contract import cohort_for, grid
    cohort=cohort_for(plan,node)
    expected_grid=[list(window) for window in grid(plan,cohort)]
    need((row['request'],row['node'],row['step'])==(REQUEST,node,step) and row['grid']==expected_grid and row['cohort']==cohort
         and len(row['selected_PCI'])==8 and set(row['selected_PCI'])==set(plan['selected'][node]['PCI']),
         'original worker intent does not bind the shard, schedule or fresh PCI')
    return sha(raw)


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
    return audit


def startup_observation(path,node,row,plan,read):
    """Preserve visible phases when a worker never completed Torch/HIP admission."""
    from contract import cohort_for
    cohort=cohort_for(plan,node)
    seen={}
    prior=0
    for phase in ('process-enter','stop-handler-ready','setup-enter','torch-import-begin','torch-import-end'):
        try:
            raw=read(path/(phase+'.json'))
        except FileNotFoundError:
            break
        marker=json.loads(raw)
        need((marker['request'],marker['node'],marker['step'],marker['cohort'])==(REQUEST,node,row[0],cohort)
             and type(marker['epoch_ns']) is int and marker['epoch_ns']>=max(1,prior),
             'original startup phase differs')
        prior=marker['epoch_ns']
        seen[phase]={'epoch_ns':prior,'SHA256':sha(raw)}
    return seen


def reconcile(plan,scheduler,read,compare):
    run=ROOT/'results'/RUN
    shards=plan['shards']
    from contract import ORDER, PROBES
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
            attempt_intent=json.loads(raw)
            item['attempted']=attempt_intent['attempted']
            if item['attempted']:
                need((attempt_intent['request'],attempt_intent['label'],attempt_intent['cohort']) ==
                     (unit['request'],label,plan['cohort_by_shard'][label]),'local attempt identity differs')
                item['spawn_intent_epoch_ns']=attempt_intent['spawn_intent_epoch_ns']
        row=actual.get(label)
        if not row:
            continue
        exact=account_scope(row,label,unit,scheduler)
        state=row[2].split()[0].rstrip('+')
        item.update(accounting=row,state=state,terminal=state in scheduler.TERMINAL,resources_match=exact)
        for name in unit['nodes']:
            try:
                nodes[name]['startup']=startup_observation(run/'nodes'/name,name,row,plan,read)
            except (OSError,ValueError,KeyError,TypeError) as error:
                nodes[name]['startup_failure']=str(error)[:120]
        if not item['terminal']:
            continue
        for name in unit['nodes']:
            try:
                nodes[name].update(bound_node(run/'nodes'/name,name,row,plan,read,compare))
            except (OSError,ValueError,KeyError,TypeError) as error:
                nodes[name]['failure']=str(error)[:120]
        item['device_qualified']=(exact and row[2:4]==['COMPLETED','0:0'] and row[4].isdecimal()
                                  and 0<int(row[4])<=SLURM_SECONDS and all(nodes[n]['passed'] for n in unit['nodes']))
    count=len(nodes)
    overall=dict(request=REQUEST,campaign_epoch=plan["campaign_epoch"],planned_nodes=count,planned_cards=count*8,planned_arms=count*8*len(ORDER),
                 planned_busy_card_samples=count*8*len(ORDER)*len(PROBES),shards=result,nodes=nodes,
                 valid_cards=sum(item['valid_cards'] for item in nodes.values()),
                 completed_arms=sum(item['completed_arms'] for item in nodes.values()),
                 actual_busy_card_samples=sum(item['actual_busy_card_samples'] for item in nodes.values()),
                 device_qualified=all(shard['device_qualified'] for shard in result.values()),
                 clock_inter_node_bounded=False,collective_executed=False,replay_allowed=False,
                 common_window_inter_node_overlap_ns=None,
                 common_expected_nodes=count, common_node_intervals=sum(
                     item.get('common_window_epoch_ns') is not None for item in nodes.values()))
    qualified_nodes=[node for shard in result.values() if shard['device_qualified'] for node in shard['nodes']]
    ratios=[value for name in qualified_nodes
            for pair in nodes[name]['per_card_event_join_over_host_two_fences_pairs'].values() for value in pair]
    overall['paired_comparison']=dict(qualified_nodes=len(qualified_nodes), measured_pairs=len(ratios),
                                      planned_pairs=count*8*2,
                                      median_event_join_over_host_two_fences=statistics.median(ratios) if ratios else None)
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
    readiness={}
    for node in unit['nodes']:
        path=run/'nodes'/node
        try:
            pin=worker_intent(path/'intent.json',node,step,plan,read)
            readiness[node]={'kind':'worker_intent','SHA256':pin}
        except FileNotFoundError:
            stages=startup_observation(path,node,row,plan,read)
            need('stop-handler-ready' in stages,'cooperative handler is not confirmed on every original node')
            readiness[node]={'kind':'stop_handler_ready','SHA256':stages['stop-handler-ready']['SHA256']}
    need(proc_identity(owner['pid'])==local,'owned srun changed during stop inspection')
    return dict(request=request,label=label,step=step,owner=local,worker_stop_readiness=readiness,
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
                          stdin=subprocess.DEVNULL,timeout=max(.01,plan['campaign_epoch']+CONTROLLER_SECONDS-time.time()))
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
