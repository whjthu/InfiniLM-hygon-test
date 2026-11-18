"""One unique 2CPU/4GiB/0GPU Slurm step; source-only until root publication/adoption."""
import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
SOURCE=Path(__file__).resolve().parent
if str(SOURCE) not in sys.path:sys.path.insert(0,str(SOURCE))
from held_child import cleanup_held
from slurm_accounting import is_terminal, wait_terminal
from pending_placeholder import fields, versions, classify

R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results')
LABEL='training-hunyuanvideo-execution-mount-SHA-CPU-v3'
RUN=R/LABEL
COLUMNS=['JobIDRaw','JobName','State','ExitCode','ElapsedRaw','NodeList','AllocTRES']


def need(value,reason):
    if not value:raise ValueError(reason)


def save(path,value):
    with path.open('x') as stream:
        json.dump(value,stream,sort_keys=True,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())


def budget(epoch,mono,now,now_mono,reserve=0):
    remaining=min(epoch+900-now,mono+900-now_mono)-reserve
    need(math.isfinite(remaining) and remaining>0,'ORIGINAL900_EPOCH_EXHAUSTED')
    return remaining


def step_budget(remaining):
    seconds=60*(min(840,int(remaining)-60)//60)
    need(seconds>=60,'NO_WHOLE_MINUTE_CPU_BUDGET')
    return seconds


def main():
    p=argparse.ArgumentParser()
    for n in ('publication','source-manifest-sha256','request-id'):p.add_argument('--'+n,required=True)
    p.add_argument('--epoch',required=True,type=float);args=p.parse_args()
    need(args.epoch<=time.time()<args.epoch+60,'INSTALL_COUNTS_IN_ORIGINAL_EPOCH')
    mono=time.monotonic()-(time.time()-args.epoch)
    def left(reserve=0):return budget(args.epoch,mono,time.time(),time.monotonic(),reserve)
    def hard_exit(*_):os._exit(124)
    def interrupted(*_):
        signal.signal(signal.SIGALRM,hard_exit)
        signal.setitimer(signal.ITIMER_REAL,max(.001,min(args.epoch+900-time.time(),mono+900-time.monotonic())))
        raise TimeoutError('ORIGINAL_CPU_SIGNAL_OR_STOP_WINDOW')
    for kind in (signal.SIGTERM,signal.SIGHUP,signal.SIGINT,signal.SIGALRM):signal.signal(kind,interrupted)
    signal.setitimer(signal.ITIMER_REAL,max(.001,min(args.epoch+850-time.time(),mono+850-time.monotonic())))
    need(re.fullmatch('[0-9a-f]{40}',args.publication) and re.fullmatch('[0-9a-f]{32}',args.request_id),'ACTUAL_PUB_REQUEST')
    source=Path(__file__).resolve().parent
    need(source==RUN/'source' and source.resolve()==source and RUN.stat().st_uid==os.getuid()==154459,'OWN_NAMESPACE_SOURCE')
    raw=(source/'SOURCE_FILES.json').read_bytes();need(hashlib.sha256(raw).hexdigest()==args.source_manifest_sha256,'ACTUAL_PUBLISHED_MF')
    for name,pin in json.loads(raw)['files'].items():
        left(60);path=source/name
        need('..' not in Path(name).parts and not path.is_symlink(),'SOURCE_PATH')
        raw=path.read_bytes();need(len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['SHA256'],'SOURCE_MEMBER')
    plan=json.loads((source/'PLAN.json').read_bytes());claim=json.loads((source/'CLAIM.json').read_bytes())
    need(claim['root_GO'] is claim['CPU_GO'] is True and claim['GPU_GO'] is False
         and claim['max_invocations']==1 and claim['parent']=='1103606' and claim['node']=='f13r3n13','EXPLICIT_ONCE_CPU_ADOPTION')
    need(not any((RUN/name).exists() for name in ('intent.json','outcome.json','owned-srun.json','worker-intent.json','weight-receipt.json')),'ORIGINAL_NAMESPACE_EXISTS_RECONCILE_NO_REPLAY')
    def query(argv,deadline=None):
        seconds=min(12,left(5))
        if deadline is not None:seconds=min(seconds,deadline-time.monotonic())
        need(seconds>0,'QUERY_WINDOW_EXHAUSTED')
        return subprocess.check_output(argv,text=True,timeout=seconds).splitlines()
    def accounting(deadline=None):
        rows=[line.split('|') for line in query(['sacct','-n','-P','-j','1103606','-o','JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES'],deadline)]
        need(all(len(row)==7 for row in rows),'ACCOUNTING_COLUMNS')
        matches=[row for row in rows if row[1]==LABEL];need(len(matches)<=1,'LABEL_NOT_UNIQUE')
        return matches[0] if matches else None
    need(accounting() is None,'ORIGINAL_LABEL_EXISTS_NO_REPLAY')
    version_evidence=versions(query(['scontrol','--version']),query(['srun','--version']),query(['scontrol','show','config']))
    prior=json.loads((source/'prior-v1-terminal.json').read_bytes())
    need(hashlib.sha256((source/'prior-v1-terminal.json').read_bytes()).hexdigest()==
         '00c73016b009ecb7528bf7b9407e23827fa53469aff5879879242a9322bed809','PUBLISHED_V1_TERMINAL_PIN')
    need(prior['request_id']=='69127d3b80624d95a477120d52abdcd2' and prior['original_state']=='REMOTE_PYTHON_EXIT1_BEFORE_NEW_CPU_SRUN'
         and prior['new_CPU_step_submissions']==prior['weight_body_bytes_read']==0,'QUALIFIED_V1_PRELAUNCH_FAILURE')
    prior2_raw=(source/'prior-v2-terminal.json').read_bytes()
    need(hashlib.sha256(prior2_raw).hexdigest()=='2a83b2327357c59aff6b264678f1fc731eb67295111b392a332fed49e8162366','QUALIFIED_CPU189_FAILURE_PIN')
    prior2=json.loads(prior2_raw)
    need(prior2['independent_failure_metadata_verified'] is True and prior2['model_or_hash_passed'] is False
         and prior2['context']['request_id']=='487949ac4113400f9e9a96c48d248efe'
         and prior2['original_state']=='SLURM_STEP_FAILED_PRE_WEIGHT_PRIORITY_GATE'
         and prior2['actual_CPU_step_submissions']==1 and prior2['logical_weight_body_bytes_read']==0,'QUALIFIED_CPU189_PRE_HASH_FAILURE')
    jobs=query(['scontrol','show','job','1103606','--oneliner']);need(len(jobs)==1,'ONE_PARENT')
    job=fields(jobs[0]);owner=re.fullmatch(r'.*\(([0-9]+)\)',job.get('UserId',''))
    need(job.get('JobState')=='RUNNING' and job.get('NumNodes')=='40'
         and 'gres/dcu=320' in job.get('AllocTRES','').split(',') and owner and int(owner[1])==os.getuid(),'OWN_RUNNING40_PARENT')
    need(datetime.fromisoformat(job.get('EndTime','')).timestamp()>=args.epoch+900,'PARENT_ENDTIME_COVERS_ORIGINAL900')
    hosts=query(['scontrol','show','hostnames',job['NodeList']]);need(len(hosts)==len(set(hosts))==40 and 'f13r3n13' in hosts,'EXACT_ASSIGNED_NODE')
    step_observations=[]
    for line in query(['scontrol','show','step','1103606','--oneliner']):
        if not line.strip():continue
        row=fields(line);step=row.get('StepId','')
        if step in ('1103606.batch','1103606.extern'):continue
        step_hosts=query(['scontrol','show','hostnames',row['NodeList']]) if row.get('NodeList') else []
        classification=classify(row,hosts,step_hosts,os.getuid())
        step_observations.append(dict(original_fields=row,expanded_step_hosts=step_hosts,classification=classification,
            reason='Official24.05.8 pending-notification sentinel; no resource allocation path'
                if classification=='PENDING_NOTIFICATION_PLACEHOLDER_NO_ALLOCATED_TRES' else 'Original numbered-step conflict guard',
            original_request_state='UNKNOWN_NOT_INFERRED_FROM_STEP_CLASSIFICATION'))
        if classification=='NUMBERED_STEP_REQUIRES_NODE_CONFLICT_CHECK':
            need('f13r3n13' not in step_hosts,'NODE_CARRIES_ACTIVE_STEP')
    slurm_seconds=step_budget(left())
    work_deadline=min(args.epoch+780,time.time()+slurm_seconds-20)
    need(work_deadline>time.time()+10,'NO_HASH_WORK_BUDGET')
    command=['srun','--jobid=1103606','--job-name='+LABEL,'--exact','--nodes=1','--ntasks=1','--nodelist=f13r3n13',
             '--cpus-per-task=2','--mem=4096M','--gres=none','--immediate=5','--time='+str(slurm_seconds//60),
             '/usr/local/python3.12/bin/python3','-I','-S','-B',str(source/'worker.py'),'--request-id',args.request_id,
             '--publication',args.publication,'--source-manifest-sha256',args.source_manifest_sha256,
             '--epoch',str(args.epoch),'--work-deadline',str(work_deadline)]
    # -I omits this script's sibling import directory. Worker explicitly adds only this exact source root.
    save(RUN/'intent.json',dict(request_id=args.request_id,publication=args.publication,epoch=args.epoch,deadline=args.epoch+900,
        source_manifest_SHA256=args.source_manifest_sha256,node='f13r3n13',parent='1103606',CPU=2,memory_MiB=4096,GPU=0,
        slurm_seconds=slurm_seconds,work_deadline=work_deadline,command=command,
        current_version_evidence=version_evidence,original_step_observations=step_observations,
        expanded_parent_hosts=hosts,
        original_TDB_request_state='UNKNOWN_NO_CANCEL_NO_RESERVATION_RELEASE'))
    child=None;anchor=None;timed_out=False;held=None;failure=None
    try:
        with (RUN/'worker.stdout').open('xb') as out,(RUN/'worker.stderr').open('xb') as err:
            child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
            anchor=dict(PID=child.pid,PGID=child.pid,start_new_session=True,request_id=args.request_id)
            save(RUN/'owned-srun.json',anchor)  # Strong unreaped Popen anchor, before any proc read.
            child.wait(timeout=min(left(60),slurm_seconds+5))
    except subprocess.TimeoutExpired:
        timed_out=True
    except BaseException as error:
        failure=type(error).__name__
    finally:
        for kind in (signal.SIGTERM,signal.SIGHUP,signal.SIGINT):signal.signal(kind,signal.SIG_IGN)
        signal.signal(signal.SIGALRM,hard_exit)
        signal.setitimer(signal.ITIMER_REAL,max(.001,min(args.epoch+900-time.time(),mono+900-time.monotonic())))
        held=cleanup_held(child,anchor,min(args.epoch+870,time.time()+max(0,min(args.epoch+900-time.time(),mono+900-time.monotonic()))))
    terminal_end=min(mono+895,time.monotonic()+35)
    accounting_errors=[];rows_seen=[]
    def observe():
        row=accounting(terminal_end);rows_seen.append(row);return row
    try:
        observed=wait_terminal(observe,terminal_end,interval=1,max_observations=20)
    except (OSError,ValueError,subprocess.SubprocessError) as error:
        accounting_errors.append(type(error).__name__)
        observed=dict(terminal_visible=False,row=None,observations=rows_seen)
    row=observed['row'];worker=None;pin=None
    path=RUN/'weight-receipt.json'
    if is_terminal(row) and path.is_file() and not path.is_symlink() and path.stat().st_size<=65536:
        raw=path.read_bytes();worker=json.loads(raw);pin=dict(relative_path=LABEL+'/weight-receipt.json',bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())
    tres=dict(x.split('=',1) for x in row[6].split(',') if '=' in x) if row else {}
    passed=bool(not timed_out and failure is None and child and child.returncode==0 and held['reaped'] is True
        and is_terminal(row) and row[2:4]==['COMPLETED','0:0'] and row[5]=='f13r3n13'
        and tres.get('cpu')=='2' and tres.get('mem') in ('4G','4096M') and not any(k.startswith('gres/') and v!='0' for k,v in tres.items())
        and worker and worker['passed'] is True and worker['request_id']==args.request_id and worker['CPU_step']==row[0])
    outcome=dict(request_id=args.request_id,publication=args.publication,epoch=args.epoch,deadline=args.epoch+900,
        source_manifest_SHA256=args.source_manifest_sha256,passed=passed,worker_receipt=pin,held_child=held,
        current_step_outcome='TERMINAL' if is_terminal(row) else 'UNKNOWN',actual_accounting=row,accounting_columns=COLUMNS,
        observations=observed['observations'],accounting_errors=accounting_errors,timed_out=timed_out,failure=failure,
        root_independent_terminal_verified=False,GPU_GO=False)
    save(RUN/'outcome.json',outcome)
    print(json.dumps(outcome,allow_nan=False));return 0 if passed else 1


if __name__=='__main__':raise SystemExit(main())
