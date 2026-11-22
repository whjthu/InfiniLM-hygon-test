"""One HOST keeper: fresh pre, one native GPU step, independent post/accounting."""
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import sys
import time

from held_child import cleanup_held
from slurm_accounting import wait_terminal,is_terminal
from contract import *

HOST_PYTHON='/usr/local/python3.12/bin/python3'
APPTAINER='/public/software/apps/apptainer/1.3.4/bin/apptainer'


def query(argv,clock):
    return subprocess.check_output(argv,text=True,timeout=min(10,clock.left(WHOLE,10))).strip()


def rows(clock):
    text=query(['sacct','-n','-P','-j',PARENT,'-o',','.join(COLUMNS).replace('JobName','JobName%120')],clock)
    return parse_accounting(text)


def one_row(label,clock):
    found=[r for r in rows(clock) if r[1]==label]
    need(len(found)<=1,'duplicate exact original Slurm label')
    if found:
        need(numbered(found[0]) and found[0][5]==NODE,'label belongs to wrong parent/numbered node')
    return found[0] if found else None


def admission(clock):
    parent=query(['squeue','-h','-j',PARENT,'-o','%T|%N'],clock).split('|')
    need(len(parent)==2 and parent[0]=='RUNNING','parent allocation not running')
    need(NODE in query(['scontrol','show','hostnames',parent[1]],clock).splitlines(),'fixed target not in allocation')
    parent_rows=[r for r in rows(clock) if r[0]==PARENT]
    need(len(parent_rows)==1 and parent_rows[0][2]=='RUNNING','parent accounting missing or ambiguous')
    # Same step-specific two-column CLI as the qualified shared Swift keeper;
    # parse every nonempty row, including rejection of truncated/extra columns.
    for handle,hosts in parse_step_membership(query(['squeue','--steps','-h','-j',PARENT,'-o','%i|%N'],clock)):
        need(NODE not in query(['scontrol','show','hostnames',hosts],clock).splitlines(),
             'fixed target has existing owner; never cancel it')
    for phase in ('pre','gpu','post','read'):
        need(one_row(LABEL+'-'+phase,clock) is None,'original step label already exists; reconcile only')

def container_command():
    runtime=R/'results'
    mounts=[(ROOT,ROOT,'rw'),(ASSET,ASSET,'rw'),(D,D,'ro'),
        (runtime/'training-gemma-original-model-v1',runtime/'training-gemma-original-model-v1','ro'),
        (DOMAIN,DOMAIN,'ro'),
        (runtime/'training-gemma-usgs72-continuous25-supervisor-v2',runtime/'training-gemma-usgs72-continuous25-supervisor-v2','ro'),
        (runtime/'training-gemma-generic64-control25-v1',runtime/'training-gemma-generic64-control25-v1','ro'),
        (runtime/'training-gemma-generic64-control25-supervisor-v1',runtime/'training-gemma-generic64-control25-supervisor-v1','ro'),
        (runtime/'training-qwen-usgs-multievent36-effect-inputs-v1',runtime/'training-qwen-usgs-multievent36-effect-inputs-v1','ro'),
        (runtime/'training-vlm-swift-offline-runtime-v1',Path('/swift-runtime'),'ro'),
        (runtime/'training-minicpm-hygon-processor-r2/cpu/runtime-site',Path('/minicpm-transformers'),'ro'),
        (Path('/opt/hyhal'),Path('/opt/hyhal'),'ro')]
    argv=[APPTAINER,'exec','--rocm','--no-home','--pwd','/']
    for source,target,mode in mounts:argv+=['--bind',f'{source}:{target}:{mode}']
    py='/minicpm-transformers:/swift-runtime/site:'+str(DOMAIN/'framework')+':'+str(D)
    command=['python3','-B','-u',str(D/'keeper.py')]
    shell='set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh; export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1; export PYTHONPATH='+shlex.quote(py)+'; exec '+shlex.join(command)
    return [*argv,str(R/'containers/dtk26-sandbox-r1'),'/bin/bash','--noprofile','--norc','-c',shell,'gemma-dev36-one-dcu']

def phase_passed(phase,rc,error,held,status):
    row=status['row']
    return bool(rc==0 and error is None and held.get('reaped') is True and held.get('termination_needed') is False
        and status['terminal_visible'] and numbered(row) and row[1]==LABEL+'-'+phase and
        row[2:4]==['COMPLETED','0:0'] and row[5]==NODE and resources_match(phase,row[6]))

def execute_step(phase,command,clock):
    name=LABEL+'-'+phase;need(one_row(name,clock) is None,'original step already exists; no replay')
    reserve={'pre':800,'gpu':160,'post':95,'read':15}[phase]
    seconds=min({'pre':60,'gpu':GPU,'post':60,'read':60}[phase],clock.left(WHOLE,reserve))
    argv=step_argv(phase,command,seconds);slurm_seconds=int(seconds//60)*60
    save(phase+'-step-intent.json',dict(request_id=REQUEST,epoch=clock.epoch,phase=phase,argv=argv,
        GPU_cards=1 if phase=='gpu' else 0,submitted_at=time.time(),slurm_seconds=slurm_seconds,original_label=name))
    child=None;anchor={};error=None;held={};old=signal.getsignal(signal.SIGTERM)
    try:
        with (ROOT/(phase+'.log')).open('xb') as log:
            child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            anchor=dict(PID=child.pid,PGID=child.pid,start_new_session=True)
            fields=Path('/proc',str(child.pid),'stat').read_text().rsplit(') ',1)[1].split()
            save(phase+'-srun-owner.json',dict(**anchor,startticks=int(fields[19]),host=socket.gethostname(),
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip()))
            child.wait(timeout=min(slurm_seconds,clock.left(WHOLE,max(10,reserve-10))))
    except BaseException as e:error=dict(type=type(e).__name__,reason=str(e)[:512])
    finally:
        signal.signal(signal.SIGTERM,signal.SIG_IGN)
        try:held=cleanup_held(child,anchor,time.time()+min(12,max(0,clock.epoch+WHOLE-10-time.time())))
        except BaseException as e:error=dict(type=type(e).__name__,phase='held_srun_cleanup',previous=error)
        signal.signal(signal.SIGTERM,old)
    status=wait_terminal(lambda:one_row(name,clock),time.monotonic()+min(35,clock.left(WHOLE,5)),interval=2,max_observations=18)
    rc=child.returncode if child is not None else None
    value=dict(request_id=REQUEST,phase=phase,returncode=rc,error=error,held=held,srun_invoked=child is not None,
        actual_numbered_steps=1 if numbered(status['row']) else None,accounting_columns=COLUMNS,terminal=status,
        passed=phase_passed(phase,rc,error,held,status))
    save(phase+'-accounting.json',value);return value

def main():
    source=verify('GPU');intent,ipin=stable_json(ROOT/'intent.json')
    need(intent['request_id']==REQUEST and intent['dispatch_request_id']==DISPATCH_REQUEST and intent['source_manifest']==source,
         'original intent/package mismatch')
    clock=Clock(intent['epoch']);results={};error=None;before=None;after=None;native=None
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('original900 epoch')))
    signal.setitimer(signal.ITIMER_REAL,clock.left())
    fields=Path('/proc/self/stat').read_text().rsplit(') ',1)[1].split()
    save('controller-owner.json',dict(PID=os.getpid(),startticks=int(fields[19]),host=socket.gethostname(),
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),request_id=REQUEST,epoch=clock.epoch))
    try:
        admission(clock)
        need(not ASSET.exists(),'effect scope already exists; exact old attempt reconciliation only')
        ASSET.mkdir(mode=0o700)
        sys.path.insert(0,str(CORE))
        from bind_ready import bind
        binding_pin=bind(ASSET/'effect-binding.json')
        save('ready-binding.json',dict(core_source=core_source(),original_binding=binding_pin))
        clock.left(WHOLE,800)
        results['pre']=execute_step('pre',[HOST_PYTHON,'-I','-S','-B','-c',
            'import sys,runpy;sys.path.insert(0,'+repr(str(D))+');sys.argv=["devices.py","pre"];runpy.run_path('+repr(str(D/'devices.py'))+',run_name="__main__")'],clock)
        need(results['pre']['passed'],'original pre failed/unknown')
        before,_=stable_json(ROOT/'before.json')
        results['gpu']=execute_step('gpu',[HOST_PYTHON,'-I','-S','-B','-c',
            'import sys,runpy;sys.path.insert(0,'+repr(str(D))+');runpy.run_path('+repr(str(D/'gpu_entry.py'))+',run_name="__main__")'],clock)
        need(results['gpu']['terminal']['terminal_visible'],'original GPU terminal UNKNOWN; hold reservation')
    except BaseException as e:error=dict(type=type(e).__name__,reason=str(e)[:512])
    finally:
        if results.get('gpu',{}).get('terminal',{}).get('terminal_visible'):
            try:
                results['post']=execute_step('post',[HOST_PYTHON,'-I','-S','-B','-c',
                    'import sys,runpy;sys.path.insert(0,'+repr(str(D))+');sys.argv=["devices.py","post"];runpy.run_path('+repr(str(D/'devices.py'))+',run_name="__main__")'],clock)
                after,_=stable_json(ROOT/'after.json')
                need(before and before['pci']==after['pci'] and before['boot_id']==after['boot_id'],'independent post inventory differs')
            except BaseException as e:error=dict(type=type(e).__name__,phase='post',reason=str(e)[:512],previous=error)
        if (ROOT/'native-outcome.json').exists():native,_=stable_json(ROOT/'native-outcome.json')
        release=bool(results.get('gpu',{}).get('terminal',{}).get('terminal_visible') and results.get('post',{}).get('passed') and
            after and after['owned_processes_remaining']==[] and after['kfd_holder_count']==0)
        producer=bool(error is None and native and native['passed'] is True and len(results)==3 and all(v['passed'] for v in results.values()))
        out=dict(request_id=REQUEST,dispatch_request_id=DISPATCH_REQUEST,epoch=clock.epoch,deadline=clock.epoch+WHOLE,
            source_manifest=source,core_source_manifest=core_source(),publication=intent['publication'],results=results,
            error=error,engineering_completed=producer,release_permitted=release,planned_each=36,planned_total=108,
            native_outcome=pin(ROOT/'native-outcome.json') if native else None,before=pin(ROOT/'before.json') if before else None,
            after=pin(ROOT/'after.json') if after else None,actual_updates=0,Model_GO=False,formal17=False)
        save('outcome.json',out)
    read_result=None;independent=None
    if results.get('gpu',{}).get('terminal',{}).get('terminal_visible') and results.get('post',{}).get('terminal',{}).get('terminal_visible'):
        try:
            command=[HOST_PYTHON,'-I','-S','-B','-c','import sys,runpy;sys.path.insert(0,'+repr(str(D))+');sys.argv=["read_terminal.py","--epoch",'+repr(str(clock.epoch))+',"--source-manifest-sha",'+repr(source['SHA256'])+'];runpy.run_path('+repr(str(D/'read_terminal.py'))+',run_name="__main__")']
            read_result=execute_step('read',command,clock)
            if (ASSET/'independent-terminal.json').exists():independent,_=stable_json(ASSET/'independent-terminal.json')
        except BaseException as e:error=dict(type=type(e).__name__,phase='independent_read',reason=str(e)[:512],previous=error)
    complete=bool(producer and read_result and read_result['passed'] and independent and independent['passed'] is True)
    summary=dict(request_id=REQUEST,dispatch_request_id=DISPATCH_REQUEST,source_manifest=source,core_source_manifest=core_source(),
        epoch=clock.epoch,deadline=clock.epoch+WHOLE,engineering_completed=complete,release_permitted=release,
        planned_each=36,planned_total=108,actual_predictions=sum(v.get('completed',0) for v in native['actual_arm_counts'].values())
        if native and native.get('actual_arm_counts') else None,independent_read=read_result,
        independent_terminal=pin(ASSET/'independent-terminal.json') if independent else None,error=error,
        actual_updates=0,dev_gold_read=False,blind_read=False,Model_GO=False,formal17=False)
    save('dispatch-receipt.json',summary)
    print(json.dumps({k:summary[k] for k in ('engineering_completed','release_permitted','planned_total','actual_predictions','actual_updates','dev_gold_read')}),flush=True)
    return 0 if complete else 1

if __name__=='__main__':raise SystemExit(main())
