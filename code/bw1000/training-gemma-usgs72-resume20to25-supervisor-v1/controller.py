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
    for phase in ('pre','gpu','post'):
        need(one_row(LABEL+'-'+phase,clock) is None,'original step label already exists; reconcile only')


def container_command():
    runtime=R/'results'
    mounts=[(ROOT,ROOT,'rw'),(ASSET,ASSET,'rw'),(D,D,'ro'),(CORE,CORE,'ro'),
        (ASSET/'continuous',ASSET/'continuous','ro'),
        (ASSET/'continuous-independent-terminal.json',ASSET/'continuous-independent-terminal.json','ro'),
        (CONTINUOUS_ROOT,CONTINUOUS_ROOT,'ro'),
        (runtime/'training-gemma-original-model-v1',runtime/'training-gemma-original-model-v1','ro'),
        (runtime/'training-gemma-usgs-render72-native-CPU-v1/data',runtime/'training-gemma-usgs-render72-native-CPU-v1/data','ro'),
        (runtime/'training-gemma-usgs-render72-exposure25-resume20-v1',runtime/'training-gemma-usgs-render72-exposure25-resume20-v1','ro'),
        (runtime/'training-vlm-swift-offline-runtime-v1',Path('/swift-runtime'),'ro'),
        (runtime/'training-minicpm-hygon-processor-r2/cpu/runtime-site',Path('/minicpm-transformers'),'ro'),
        (Path('/opt/hyhal'),Path('/opt/hyhal'),'ro')]
    argv=[APPTAINER,'exec','--rocm','--no-home','--pwd','/']
    for source,target,mode in mounts:
        argv+=['--bind',f'{source}:{target}:{mode}']
    py='/minicpm-transformers:/swift-runtime/site:'+str(ASSET/'framework')+':'+str(D)
    command=['python3','-B','-u',str(D/'supervisor.py')]
    shell='set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh; export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1; export PYTHONPATH='+shlex.quote(py)+'; exec '+shlex.join(command)
    return [*argv,str(R/'containers/dtk26-sandbox-r1'),'/bin/bash','--noprofile','--norc','-c',shell,'gemma-resume20to25']


def execute_step(phase,command,clock):
    name=LABEL+'-'+phase
    need(one_row(name,clock) is None,'step has prior execution; no replay')
    nominal={'pre':60,'gpu':GPU,'post':90}[phase]
    reserve={'pre':2300,'gpu':240,'post':40}[phase]
    seconds=min(nominal,clock.left(WHOLE,reserve))
    argv=step_argv(phase,command,seconds)
    save(phase+'-step-intent.json',dict(request_id=REQUEST,epoch=clock.epoch,phase=phase,argv=argv,
        GPU_cards=8 if phase=='gpu' else 0,original_label=name))
    child=None;anchor={};error=None;held={};status=None
    old=signal.getsignal(signal.SIGTERM)
    try:
        with (ROOT/(phase+'.log')).open('xb') as log:
            child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            anchor=dict(PID=child.pid,PGID=child.pid,start_new_session=True)
            owner=Path('/proc',str(child.pid),'stat').read_text().rsplit(') ',1)[1].split()
            save(phase+'-srun-owner.json',dict(**anchor,startticks=int(owner[19]),host=socket.gethostname(),
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip()))
            child.wait(timeout=min(seconds,clock.left(WHOLE,reserve-20)))
    except BaseException as e:
        error=dict(type=type(e).__name__,reason=str(e)[:512])
    finally:
        signal.signal(signal.SIGTERM,signal.SIG_IGN)
        try:
            held=cleanup_held(child,anchor,time.time()+min(12,max(0,clock.epoch+WHOLE-40-time.time())))
        except BaseException as e:
            error=dict(type=type(e).__name__,phase='held_srun_cleanup',previous=error)
        signal.signal(signal.SIGTERM,old)
    status=wait_terminal(lambda:one_row(name,clock),time.monotonic()+min(40,clock.left(WHOLE,20)),
                         interval=2,max_observations=20)
    row=status['row']
    rc=child.returncode if child is not None else None
    passed=bool(rc==0 and error is None and held.get('reaped') is True
        and held.get('termination_needed') is False and status['terminal_visible']
        and numbered(row) and row[2:4]==['COMPLETED','0:0'] and row[5]==NODE
        and allocated(row[6])==(8 if phase=='gpu' else 0))
    value=dict(request_id=REQUEST,phase=phase,returncode=rc,error=error,held=held,
        srun_invoked=child is not None,actual_numbered_steps=1 if row else None,
        accounting_columns=COLUMNS,terminal=status,passed=passed)
    save(phase+'-accounting.json',value)
    return value


def main():
    source=verify('GPU')
    intent,intent_pin=stable_json(ROOT/'intent.json')
    need(intent['request_id']==REQUEST and intent['dispatch_request_id']==DISPATCH_REQUEST
         and intent['source_manifest']==source,'original intent differs from operative package')
    clock=Clock(intent['epoch']);results={};error=None;before=None;after=None
    def expired(*_):raise TimeoutError('original whole epoch stop')
    signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,clock.left(WHOLE))
    stat=Path('/proc/self/stat').read_text().rsplit(') ',1)[1].split()
    save('controller-owner.json',dict(PID=os.getpid(),startticks=int(stat[19]),host=socket.gethostname(),
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),request_id=REQUEST,epoch=clock.epoch))
    try:
        admission(clock)
        stage,_=stable_json(ASSET/'stage-complete.json',STAGE_PIN)
        need(stage['passed'] is True and stage['training_request_id']==REQUEST,'actual845 stage prerequisite missing')
        prior_continuous()
        need(not (ASSET/'resumed').exists(),'resumed output already exists; no replay')
        results['pre']=execute_step('pre',[HOST_PYTHON,'-B',str(D/'devices.py'),'pre'],clock)
        need(results['pre']['passed'],'independent pre step failed/unknown')
        before,_=stable_json(ROOT/'before.json')
        need(before['step']==results['pre']['terminal']['row'][0],'pre original actual step binding differs')
        results['gpu']=execute_step('gpu',container_command(),clock)
        need(results['gpu']['terminal']['terminal_visible'],'original GPU terminal unknown; keep reservation')
        entry,_=stable_json(ROOT/'gpu-step-entry.json')
        need(entry['request_id']==REQUEST and entry['intent']==intent_pin
             and entry['step']==results['gpu']['terminal']['row'][0],'GPU original step/intent binding differs')
    except BaseException as e:
        error=dict(type=type(e).__name__,reason=str(e)[:512])
    finally:
        # Post is independent and is attempted after any known GPU terminal,
        # including native failure; UNKNOWN terminal never authorizes release.
        if results.get('gpu',{}).get('terminal',{}).get('terminal_visible'):
            try:
                results['post']=execute_step('post',[HOST_PYTHON,'-B',str(D/'devices.py'),'post'],clock)
                after,_=stable_json(ROOT/'after.json')
                need(before and after['pci']==before['pci'] and after['boot_id']==before['boot_id'],
                     'post PCI/boot differs from fresh original pre')
            except BaseException as e:
                error=dict(type=type(e).__name__,phase='post',previous=error,reason=str(e)[:512])
        native=None
        if (ROOT/'native-outcome.json').exists():
            native,_=stable_json(ROOT/'native-outcome.json')
        engineering=bool(error is None and native and native['passed'] is True
            and len(results)==3 and all(r['passed'] for r in results.values())
            and after and after['owned_processes_remaining']==[] and after['kfd_holder_count']==0)
        out=dict(request_id=REQUEST,dispatch_request_id=DISPATCH_REQUEST,epoch=clock.epoch,deadline=clock.epoch+WHOLE,source_manifest=source,
            publication=intent['publication'],engineering_completed=engineering,
            release_permitted=engineering,error=error,results=results,
            planned_updates=5,original_continuous_updates=25,actual_updates=native.get('actual_updates') if native else None,
            native_outcome=pin(ROOT/'native-outcome.json') if native else None,
            before=pin(ROOT/'before.json') if before else None,after=pin(ROOT/'after.json') if after else None,
            model_qualified=False,formal17=False,effect_qualified=False,resume_invoked=True,
            exact_paired_recovery=native.get('exact_paired_recovery') if native else None,
            pair_comparison=pin(ROOT/'pair-comparison.json') if (ROOT/'pair-comparison.json').exists() else None)
        save('outcome.json',out)
        print(json.dumps({k:out[k] for k in ('engineering_completed','release_permitted','actual_updates','resume_invoked','exact_paired_recovery')}),flush=True)
    return 0 if engineering else 1


if __name__=='__main__':raise SystemExit(main())
