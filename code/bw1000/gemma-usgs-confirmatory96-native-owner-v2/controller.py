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
from authority import verify_authority,verify_physical
from asset_witness import ORIGINAL_SOURCE, ORIGINAL_INPUTS, ORIGINAL_AUDIT

HOST_PYTHON='/usr/local/python3.12/bin/python3'
APPTAINER='/public/software/apps/apptainer/1.3.4/bin/apptainer'


def query(argv,clock):
    return subprocess.check_output(argv,text=True,timeout=min(10,clock.left(WHOLE,10))).strip()


def rows(clock):
    text=query(accounting_argv(),clock)
    return parse_accounting(text)


def one_row(label,clock):
    need(label.startswith(LABEL+'-'),'exact scoped accounting label required')
    return exact_label(rows(clock),label[len(LABEL)+1:])


def check_completed_pre(clock,pre,before,owner_pin,fresh):
    current,current_pin=stable_json(ROOT/'intent.json',owner_pin)
    reread,pre_pin=stable_json(ROOT/'pre-accounting.json')
    snapshot,snapshot_pin=stable_json(ROOT/'before.json')
    started,started_pin=stable_json(ROOT/'pre-step-intent.json')
    need(current_pin==owner_pin and current['request_id']==REQUEST and current['epoch']==clock.epoch
         and current['dispatch_request_id']==DISPATCH_REQUEST and current['label']==LABEL,
         'post-pre handshake must use current original intent')
    need(pre==reread and pre['request_id']==REQUEST and pre['phase']=='pre' and pre['passed'] is True
         and pre['actual_numbered_steps']==1 and pre['srun_invoked'] is True and pre['accounting_columns']==COLUMNS
         and phase_passed('pre',pre['returncode'],pre['error'],pre['held'],pre['terminal']),
         'exact own durable successful pre accounting required')
    stored=pre['terminal']['row']
    need(fresh==stored and accounting_window(fresh,started),'fresh original nine-column pre accounting differs')
    end=datetime.fromisoformat(fresh[8]).replace(tzinfo=timezone.utc).timestamp()
    need(-1<=time.time()-end<=90,'exact pre terminal snapshot is stale')
    need(started['owner_intent']==owner_pin and started['request_id']==REQUEST and started['phase']=='pre'
         and started['epoch']==clock.epoch and started['original_label']==LABEL+'-pre'
         and started['GPU_cards']==0,'exact own pre step intent differs')
    need(before==snapshot and snapshot['request_id']==REQUEST and snapshot['phase']=='pre'
         and snapshot['epoch']==clock.epoch and snapshot['owner_intent']==owner_pin and snapshot['phase_intent']==started_pin
         and snapshot['node']==NODE and snapshot['step']==stored[0]
         and snapshot['master_port']==PORT and snapshot['master_port_available'] is True,
         'exact own before snapshot differs')
    verify_physical(current['root_authority'],snapshot)
    save('pre-admission.json',dict(request_id=REQUEST,epoch=clock.epoch,owner_intent=owner_pin,
         phase_intent=started_pin,before=snapshot_pin,stored_pre_accounting=pre_pin,accounting_columns=COLUMNS,row=stored))


def admission(clock,*,pre=None,before=None,owner_pin=None):
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
    for line in query(['scontrol','show','step',PARENT,'--oneliner'],clock).splitlines():
        row=dict(atom.split('=',1) for atom in line.split() if '=' in atom)
        step=row.get('StepId','')
        if step in (PARENT+'.batch',PARENT+'.extern') or not step:
            continue
        need(numbered([step]),'unclassified controller step')
        state=row.get('State')
        if state in ('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL'):
            continue
        need(row.get('NodeList') not in (None,'','(null)'),'unclassified controller step nodes')
        need(NODE not in query(['scontrol','show','hostnames',row['NodeList']],clock).splitlines(),
             'controller still shows an active holder on n12')
    observed=rows(clock)
    pre_row=exact_label(observed,'pre')
    for phase in ('gpu','post','read'):
        need(exact_label(observed,phase) is None,'original step label already exists; reconcile only')
    if pre is None:
        need(before is owner_pin is None and pre_row is None,'first admission forbids every existing phase label')
    else:
        check_completed_pre(clock,pre,before,owner_pin,pre_row)

def container_command():
    runtime=R/'results'
    from core.input_bundle import TARGET as inputs
    mounts=[(ROOT,ROOT,'rw'),(ASSET,ASSET,'rw'),(D,D,'ro'),
        (runtime/'training-gemma-original-model-v1',runtime/'training-gemma-original-model-v1','ro'),
        (DOMAIN,DOMAIN,'ro'),
        (runtime/'training-gemma-usgs72-continuous25-supervisor-v2',runtime/'training-gemma-usgs72-continuous25-supervisor-v2','ro'),
        (runtime/'training-gemma-usgs72-question-only-control25-v1',runtime/'training-gemma-usgs72-question-only-control25-v1','ro'),
        (runtime/'training-gemma-usgs72-question-only-control25-v1-native',runtime/'training-gemma-usgs72-question-only-control25-v1-native','ro'),
        (ORIGINAL_SOURCE,ORIGINAL_SOURCE,'ro'), (inputs,inputs,'ro'),
        (ORIGINAL_AUDIT,ORIGINAL_AUDIT,'ro'),
        (runtime/'training-vlm-swift-offline-runtime-v1',Path('/swift-runtime'),'ro'),
        (runtime/'training-minicpm-hygon-processor-r2/cpu/runtime-site',Path('/minicpm-transformers'),'ro'),
        (Path('/opt/hyhal'),Path('/opt/hyhal'),'ro'),
        (ROOT/'miopen-private-v4/private-home',Path('/miopen-private-home-v4'),'rw'),
        (ROOT/'miopen-private-v4/provider/libMIOpen.so.1.0',Path('/opt/dtk/lib/libMIOpen.so.1.0'),'ro')]
    argv=[APPTAINER,'exec','--rocm','--no-mount','home,cwd','--pwd','/']
    for source,target,mode in mounts:argv+=['--bind',f'{source}:{target}:{mode}']
    py='/minicpm-transformers:/swift-runtime/site:'+str(DOMAIN/'framework')+':'+str(D)
    entry=D/'keeper.py'
    command=['python3','-B','-u',str(entry)]
    shell='set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh; unset PYTHONPYCACHEPREFIX; export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1; export PYTHONPATH='+shlex.quote(py)+'; test -z "${LD_PRELOAD:-}${LD_AUDIT:-}${LD_PROFILE:-}"; export HOME=/miopen-private-home-v4 MIOPEN_USER_DB_PATH=/miopen-private-home-v4/.config/miopen/user-db MIOPEN_CUSTOM_CACHE_DIR=/miopen-private-home-v4/.cache/miopen; exec '+shlex.join(command)
    return [*argv,str(R/'containers/dtk26-sandbox-r1'),'/bin/bash','--noprofile','--norc','-c',shell,'gemma-blind36-one-dcu']

def phase_passed(phase,rc,error,held,status):
    row=status['row']
    return bool(rc==0 and error is None and held.get('reaped') is True and held.get('termination_needed') is False
        and status['terminal_visible'] and numbered(row) and row[1]==LABEL+'-'+phase and
        row[2:4]==['COMPLETED','0:0'] and row[5]==NODE and resources_match(phase,row[6]))

def execute_step(phase,command,clock):
    name=LABEL+'-'+phase;need(one_row(name,clock) is None,'original step already exists; no replay')
    reserve={'pre':GPU+190,'gpu':240,'post':95,'read':15}[phase]
    seconds=min({'pre':60,'gpu':GPU,'post':60,'read':60}[phase],clock.left(WHOLE,reserve))
    argv=step_argv(phase,command,seconds);slurm_seconds=int(seconds//60)*60
    _,owner_pin=stable_json(ROOT/'intent.json')
    step_intent=dict(request_id=REQUEST,epoch=clock.epoch,phase=phase,argv=argv,
        GPU_cards=1 if phase=='gpu' else 0,submitted_at=time.time(),slurm_seconds=slurm_seconds,original_label=name,
        owner_intent=owner_pin)
    save(phase+'-step-intent.json',step_intent)
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
        passed=phase_passed(phase,rc,error,held,status) and accounting_window(status['row'],step_intent))
    save(phase+'-accounting.json',value);return value

def main():
    source=verify('GPU');intent,ipin=stable_json(ROOT/'intent.json')
    need(intent['request_id']==REQUEST and intent['dispatch_request_id']==DISPATCH_REQUEST and intent['source_manifest']==source,
         'original intent/package mismatch')
    verify_authority(intent['root_authority'],source,intent['epoch'],intent['publication'])
    clock=Clock(intent['epoch']);results={};error=None;before=None;after=None;native=None
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('original900 epoch')))
    signal.setitimer(signal.ITIMER_REAL,clock.left())
    fields=Path('/proc/self/stat').read_text().rsplit(') ',1)[1].split()
    save('controller-owner.json',dict(PID=os.getpid(),startticks=int(fields[19]),host=socket.gethostname(),
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),request_id=REQUEST,epoch=clock.epoch))
    try:
        admission(clock)
        need(ASSET.is_dir() and ASSET.resolve(strict=True)==ASSET and not any(ASSET.iterdir()),
             'new inference result namespace must start empty')
        sys.path.insert(0,str(CORE))
        from bind_ready import bind
        binding_pin=bind(ASSET/'effect-binding.json')
        save('ready-binding.json',dict(core_source=core_source(),original_binding=binding_pin))
        clock.left(WHOLE,GPU+190)
        results['pre']=execute_step('pre',[HOST_PYTHON,'-I','-S','-B','-c',
            'import sys,runpy;sys.path.insert(0,'+repr(str(D))+');sys.argv=["devices.py","pre"];runpy.run_path('+repr(str(D/'devices.py'))+',run_name="__main__")'],clock)
        need(results['pre']['passed'],'original pre failed/unknown')
        before,_=stable_json(ROOT/'before.json')
        verify_physical(intent['root_authority'],before)
        admission(clock,pre=results['pre'],before=before,owner_pin=ipin)
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
                post_intent,post_pin=stable_json(ROOT/'post-step-intent.json')
                post_row=results['post']['terminal']['row']
                gpu_row=results['gpu']['terminal']['row']
                gpu_intent,_=stable_json(ROOT/'gpu-step-intent.json')
                need(before and before['pci']==after['pci'] and before['boot_id']==after['boot_id']
                     and after['owner_intent']==ipin and after['phase_intent']==post_pin
                     and after['request_id']==REQUEST and after['phase']=='post' and after['step']==post_row[0]
                     and accounting_window(gpu_row,gpu_intent) and accounting_window(post_row,post_intent)
                     and gpu_row[8]<=post_row[7],'independent post inventory/phase chronology differs')
            except BaseException as e:
                after=None
                error=dict(type=type(e).__name__,phase='post',reason=str(e)[:512],previous=error)
        if (ROOT/'native-outcome.json').exists():native,_=stable_json(ROOT/'native-outcome.json')
        release=bool(results.get('gpu',{}).get('terminal',{}).get('terminal_visible') and results.get('post',{}).get('passed') and
            after and after['owned_processes_remaining']==[] and after['kfd_holder_count']==0)
        producer=bool(error is None and native and native['passed'] is True and len(results)==3 and all(v['passed'] for v in results.values()))
        out=dict(request_id=REQUEST,dispatch_request_id=DISPATCH_REQUEST,epoch=clock.epoch,deadline=clock.epoch+WHOLE,
            source_manifest=source,core_source_manifest=core_source(),publication=intent['publication'],results=results,
            error=error,engineering_completed=producer,release_permitted=release,planned_each=96,planned_total=288,
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
        planned_each=96,planned_total=288,actual_predictions=sum(v.get('completed',0) for v in native['actual_arm_counts'].values())
        if native and native.get('actual_arm_counts') else None,independent_read=read_result,
        independent_terminal=pin(ASSET/'independent-terminal.json') if independent else None,error=error,
        planned_slots_accounted=288 if complete else None, planned_slots_unresolved=0 if complete else 288,
        actual_updates=0,blind_gold_read=False,dev_inputs_read=False,Model_GO=False,formal17=False)
    save('dispatch-receipt.json',summary)
    print(json.dumps({k:summary[k] for k in ('engineering_completed','release_permitted','planned_total','actual_predictions','actual_updates','blind_gold_read')}),flush=True)
    return 0 if complete else 1

if __name__=='__main__':raise SystemExit(main())
