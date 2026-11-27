"""One HOST keeper: fresh pre, one native GPU step, independent post/accounting."""
from contextlib import ExitStack, contextmanager
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
from short_mount import SOURCE as SHORT_SOURCE, SHORT, host_create_once
from slurm_accounting import wait_terminal,is_terminal
from contract import *
from root_guard import admit_r41, fields
from utc_scope import utc_control_argv

HOST_PYTHON='/usr/local/python3.12/bin/python3'
APPTAINER='/public/software/apps/apptainer/1.3.4/bin/apptainer'


def query(argv,clock):
    return subprocess.check_output(utc_control_argv(argv),text=True,timeout=min(10,clock.left(WHOLE,10))).strip()


def rows(clock):
    from root_guard import holder_accounting
    return parse_accounting("\n".join("|".join(row[:7]) for row in holder_accounting(query,clock)))


def one_row(label,clock):
    found=[r for r in rows(clock) if r[1]==label]
    need(len(found)<=1,'duplicate exact original Slurm label')
    if found:
        need(numbered(found[0]) and found[0][5]==NODE,'label belongs to wrong parent/numbered node')
    return found[0] if found else None


def admission(clock, expected_pre=None):
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
        observed=one_row(LABEL+'-'+phase,clock)
        expected=expected_pre if phase=='pre' else None
        need(observed==expected,'R41_EXACT_PHASE_HISTORY_BEFORE_DISPATCH')


def container_command():
    host_create_once()
    runtime=R/'results'
    r39=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-gemma-usgs-r39-two-updates-v1/6d14e9df9832432b97f1f46f032b49f3'); r26=runtime/'training-gemma-usgs-r26-native-image-CPU-v1/537844da9e9b470993209b74ad45c7f7/data'
    mounts=[(ROOT,ROOT,'rw'),(SHORT_SOURCE,SHORT,'rw'),(ASSET,ASSET,'ro'),(D,D,'ro'),(CORE,CORE,'ro'),(r39,r39,'ro'),(r26,r26,'ro'),
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
    shell='set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh; unset PYTHONPYCACHEPREFIX; export TMPDIR=/tmp/gr09ece93d TMP=/tmp/gr09ece93d TEMP=/tmp/gr09ece93d; export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1; export PYTHONPATH='+shlex.quote(py)+'; exec '+shlex.join(command)
    return [*argv,str(R/'containers/dtk26-sandbox-r1'),'/bin/bash','--noprofile','--norc','-c',shell,'gemma-continuous25']


@contextmanager
def shared_ordinary_lock():
    import fcntl
    import stat

    path=R/'results'/'.training80-dcu-standby.lock'
    ensure_existing=os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC
    fd=os.open(path,ensure_existing)
    try:
        info=os.fstat(fd)
        before=path.lstat()
        need(stat.S_ISREG(info.st_mode) and (info.st_dev,info.st_ino)==(before.st_dev,before.st_ino),
             'R41_EXACT_EXISTING_ORDINARY_SHARED_LOCK')
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        after=path.lstat()
        need((info.st_dev,info.st_ino)==(after.st_dev,after.st_ino),'R41_ORDINARY_LOCK_REPLACED')
        yield
    finally:
        os.close(fd)


def exact_gpu_tres(raw):
    entries=raw.split(',')
    pairs=[entry.split('=',1) for entry in entries]
    if len(pairs)!=4 or any(len(item)!=2 for item in pairs):
        return False
    found=dict(pairs)
    return (found.keys()=={'cpu','gres/dcu','mem','node'}
            and (found['cpu'],found['gres/dcu'],found['node'])==('64','8','1')
            and found['mem'] in ('224000M','218.75G'))


def exact_gpu_terminal(row):
    return bool(numbered(row) and is_terminal(row) and row[1]==LABEL+'-gpu' and row[5]==NODE
                and exact_gpu_tres(row[6]))


def observe_gpu_handoff(clock):
    name=LABEL+'-gpu'
    command=['scontrol','show','step',PARENT,'-o']
    deadline=time.monotonic()+min(10,clock.left(WHOLE,240))
    while time.monotonic()<deadline:
        remaining=deadline-time.monotonic()
        if remaining<=0:
            break
        out=subprocess.check_output(command,timeout=min(2,remaining),text=True)
        need(len(out)<=2*1024**2,'R41_GPU_HANDOFF_CONTROLLER_OUTPUT_BOUND')
        matches=[item for line in out.splitlines() if (item:=fields(line)).get('Name')==name]
        need(len(matches)<=1,'R41_GPU_HANDOFF_DUPLICATE_CONTROLLER_LABEL')
        if matches:
            item=matches[0]
            handle=item.get('StepId','')
            need(re.fullmatch(PARENT+r'\.[0-9]+',handle) is not None
                 and item.get('NodeList')==NODE and exact_gpu_tres(item.get('TRES','')),
                 'R41_GPU_HANDOFF_WRONG_NUMERIC_NODE_OR_RESOURCES')
            if item.get('State') in ('RUNNING','COMPLETING'):
                save('gpu-shared-lock-handoff.json',dict(request_id=REQUEST,epoch=clock.epoch,
                    node=NODE,step=handle,tres=item['TRES'],observed='CONTROLLER_EXACT_NUMBERED_VISIBLE'))
                return handle
        if not matches:
            # An immediate terminal may have vanished from the live controller.
            remaining=deadline-time.monotonic()
            if remaining<=0:
                break
            acct=['sacct','-n','-P','-j',PARENT,'-o',','.join(COLUMNS).replace('JobName','JobName%120')]
            raw=subprocess.check_output(acct,timeout=min(2,remaining),text=True)
            need(len(raw)<=2*1024**2,'R41_GPU_HANDOFF_ACCOUNTING_OUTPUT_BOUND')
            reported=[row for row in parse_accounting(raw.strip()) if row[1]==name]
            need(len(reported)<=1,'R41_GPU_HANDOFF_DUPLICATE_ACCOUNTING_LABEL')
            if reported and is_terminal(reported[0]):
                row=reported[0]
                need(exact_gpu_terminal(row),'R41_GPU_HANDOFF_WRONG_TERMINAL_IDENTITY')
                save('gpu-shared-lock-handoff.json',dict(request_id=REQUEST,epoch=clock.epoch,
                    node=NODE,step=row[0],tres=row[6],observed='ACCOUNTED_EXACT_TERMINAL_CONSUMPTION'))
                return row[0]
        time.sleep(max(0,min(.25,deadline-time.monotonic())))
    raise ValueError('R41_GPU_HANDOFF_UNKNOWN_RECONCILE_ONLY')


def compute_source_command(clock, seconds):
    from base64 import b64decode
    import zlib
    from contract import PORT, TRAINING_REQUEST
    claim,_=stable_json(D/'CLAIM.json',cap=16384)
    approval,actual=stable_json(R/'incoming'/(LABEL+'-authority')/(REQUEST+'.json'),cap=16384)
    need(actual['SHA256']==claim['placement_authority_SHA256'] and approval['request_id']==REQUEST,
         'R120_COMPUTE_PREFLIGHT_APPROVAL_FILE')
    now=time.time()
    source=verify('GPU')
    target=dict(R=R.as_posix(),PARENT=PARENT,NODE=NODE,PORT=PORT,LABEL=LABEL,
                TRAINING_REQUEST=TRAINING_REQUEST,DISPATCH_REQUEST=DISPATCH_REQUEST)
    attrs=dict(root=D.as_posix(),manifest=source,target=target,issued=approval['issued_epoch'],
               pre_deadline=now+min(seconds,60))
    program=zlib.decompress(b64decode(b'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',validate=True)).decode()
    return [HOST_PYTHON,'-I','-S','-B','-c',program,json.dumps(attrs,sort_keys=True,allow_nan=False)]


def execute_step(phase,command,clock,on_handoff=None):
    name=LABEL+'-'+phase
    need(one_row(name,clock) is None,'step has prior execution; no replay')
    nominal={'pre':60,'gpu':GPU,'post':90}[phase]
    reserve={'pre':2300,'gpu':240,'post':40}[phase]
    seconds=min(nominal,clock.left(WHOLE,reserve))
    if phase=='pre':
        command=compute_source_command(clock,seconds)
    argv=step_argv(phase,command,seconds)
    need(phase!='gpu' or callable(on_handoff),'R41_SHARED_LOCK_HANDOFF_REQUIRED')
    save(phase+'-step-intent.json',dict(request_id=REQUEST,epoch=clock.epoch,phase=phase,argv=argv,
        GPU_cards=8 if phase=='gpu' else 0,original_label=name))
    child=None;anchor={};error=None;held={};status=None;handoff_step=None
    old=signal.getsignal(signal.SIGTERM)
    try:
        with (ROOT/(phase+'.log')).open('xb') as log:
            child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            anchor=dict(PID=child.pid,PGID=child.pid,start_new_session=True)
            owner=Path('/proc',str(child.pid),'stat').read_text().rsplit(') ',1)[1].split()
            save(phase+'-srun-owner.json',dict(**anchor,startticks=int(owner[19]),host=socket.gethostname(),
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip()))
            if phase=='gpu':
                handoff_step=observe_gpu_handoff(clock)
                on_handoff()
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
        if phase=='gpu' and handoff_step is None:
            save('gpu-shared-lock-unconfirmed.json',dict(request_id=REQUEST,epoch=clock.epoch,
                ordinary_exclusion_retained=True,original_intent_consumed=True,
                state='UNKNOWN_NO_REPLAY',held_srun_reaped=held.get('reaped')))
            on_handoff()
    status=wait_terminal(lambda:one_row(name,clock),time.monotonic()+min(40,clock.left(WHOLE,20)),
                         interval=2,max_observations=20)
    row=status['row']
    if phase=='gpu' and status['terminal_visible'] and (
            not exact_gpu_terminal(row) or (handoff_step is not None and row[0]!=handoff_step)):
        status.update(row=None,terminal_visible=False)
        row=None
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
    try:
        stage,_=stable_json(ASSET/'stage-complete.json',STAGE_PIN)
        need(stage['passed'] is True and stage['training_request_id']=='247fa3682a114542ac3640fbc726aaae','actual845 stage prerequisite missing')
        from native_worker import trust_cpu_stage
        trust_cpu_stage()
        with ExitStack() as lock_stack:
            lock_stack.enter_context(shared_ordinary_lock())
            admit_r41(query, rows, clock)
            admission(clock)
            need(not (ROOT/'continuous').exists(),'R41 continuous already attempted; reconcile instead')
            stat=Path('/proc/self/stat').read_text().rsplit(') ',1)[1].split()
            save('controller-owner.json',dict(PID=os.getpid(),startticks=int(stat[19]),host=socket.gethostname(),
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),request_id=REQUEST,
                epoch=clock.epoch,ordinary_exclusion_required_until_independent_post=True))
            results['pre']=execute_step('pre',[HOST_PYTHON,'-B',str(D/'devices.py'),'pre'],clock)
            need(results['pre']['passed'],'independent pre step failed/unknown')
            before,_=stable_json(ROOT/'before.json')
            pre_row=results['pre']['terminal']['row']
            need(before['step']==pre_row[0],'pre original actual step binding differs')
            admit_r41(query, rows, clock)
            admission(clock, expected_pre=pre_row)
            results['gpu']=execute_step('gpu',container_command(),clock,on_handoff=lock_stack.close)
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
            release_permitted=bool(after and before and after['pci']==before['pci'] and after['boot_id']==before['boot_id']
                and after['owned_processes_remaining']==[] and after['kfd_holder_count']==0
                and results.get('post',{}).get('passed') and results.get('gpu',{}).get('terminal',{}).get('terminal_visible')),
            error=error,results=results,
            planned_updates=2,nominal_scheduler_steps=25,actual_updates=native.get('actual_updates') if native else None,
            native_outcome=pin(ROOT/'native-outcome.json') if native else None,
            before=pin(ROOT/'before.json') if before else None,after=pin(ROOT/'after.json') if after else None,
            model_qualified=False,formal17=False,effect_qualified=False,resume_invoked=False)
        save('outcome.json',out)
        print(json.dumps({k:out[k] for k in ('engineering_completed','release_permitted','actual_updates','resume_invoked')}),flush=True)
    return 0 if engineering else 1


if __name__=='__main__':raise SystemExit(main())
