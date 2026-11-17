"""One published CPU processor scope, bounded by its original 600-second epoch."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

from budget import command, left, terminal_timeout
from runtime_gate import ROOT,RUN,LABEL,FRAMEWORK,MODEL,read,require,exact_data,runtime,cpu330,pin
from slurm_accounting import is_terminal,wait_terminal

COLUMNS='JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES'


def save(path,value):
    with path.open('x',encoding='utf8') as stream:json.dump(value,stream,indent=2,allow_nan=False)


def fields(line):
    return dict(word.split('=',1) for word in line.split() if '=' in word)


def accounting_for(query,parent,label):
    require(parent in ('1087656','1103606'),'unexpected accounting parent')
    rows=[x.split('|') for x in query('sacct','-n','-P','-j',parent,'-o',COLUMNS)]
    require(all(len(row)==7 for row in rows),'accounting schema changed')
    matches=[row for row in rows if row[1]==label]
    require(len(matches)<=1,'duplicate original step label')
    return matches[0] if matches else None


def runtime_bind_source(host):
    if host==Path('/opt/hyhal'):
        target=Path('/usr/local/hyhal')
        require(host.resolve(strict=True)==target and target.is_dir() and not target.is_symlink(),
                'system hyhal alias identity changed')
        return target
    require(host.is_dir() and not host.is_symlink(),'required read-only runtime bind missing')
    return host


def main(publication,expected_claim):
    epoch,started=time.time(),time.monotonic()
    deadline,mono_end=epoch+600,started+600
    def exhausted(*_):raise TimeoutError('original HOST CPU epoch exhausted')
    signal.signal(signal.SIGALRM,exhausted)
    signal.setitimer(signal.ITIMER_REAL,max(0.001,min(deadline-time.time(),mono_end-time.monotonic())))
    source=Path(__file__).resolve().parent
    require(source==RUN/'source' and not RUN.is_symlink(),'unexpected installed namespace')
    require(re.fullmatch('[0-9a-f]{40}',publication) and re.fullmatch('[0-9a-f]{64}',expected_claim),
            'invalid publication or claim SHA')
    require(hashlib.sha256((source/'CLAIM.json').read_bytes()).hexdigest()==expected_claim,'claim differs')
    from verify_package import verify
    verify(source)
    # Exclusive lock is never removed, including prelaunch/transport failure.
    (RUN/'launch.lock').mkdir(mode=0o700)
    save(RUN/'launch-state.json',dict(publication=publication,claim_SHA256=expected_claim,epoch=epoch,
        deadline=deadline,maximum_invocations=1,replay_allowed=False,GPU_cards=0))
    require(not any((RUN/n).exists() for n in ('intent.json','outcome.json','owned-srun.json','step.log','model','work')),
            'existing run requires reconciliation')
    terminal_end=None
    def remaining(reserve=0):return left(deadline,mono_end,time.time(),time.monotonic(),reserve)
    def query(*argv):
        timeout=min(12,remaining()) if terminal_end is None else terminal_timeout(
            deadline,mono_end,terminal_end,time.time(),time.monotonic())
        return subprocess.check_output(argv,text=True,timeout=timeout).strip().splitlines()
    def accounting(parent,label):return accounting_for(query,parent,label)
    try:
        require(accounting('1103606',LABEL) is None,'original scope already has accounting')
        qualified330=cpu330(source,accounting)
        jobrows=query('scontrol','show','job','1103606','--oneliner')
        require(len(jobrows)==1,'parent allocation unavailable')
        job=fields(jobrows[0])
        require(job.get('JobState')=='RUNNING' and job.get('NumNodes')=='40'
                and 'gres/dcu=320' in job.get('AllocTRES','').split(','),'parent allocation changed')
        require('f13r1n11' in query('scontrol','show','hostnames',job['NodeList']),'node outside parent')
        for line in query('scontrol','show','step','1103606','--oneliner'):
            row=fields(line);step=row.get('StepId','')
            if step in ('1103606.extern','1103606.batch'):continue
            require(re.fullmatch(r'1103606\.[0-9]+',step),'unknown active step')
            require('f13r1n11' not in query('scontrol','show','hostnames',row['NodeList']),
                    'node already carries active step')
        dataset=exact_data(source,RUN/'data')
        providers=runtime(source)
        remaining(90)
        model=RUN/'model';model.mkdir()
        for name,expected in providers['processor'].items():
            data=pin(MODEL/name,expected)
            with (model/name).open('xb') as stream:stream.write(data)
        work=RUN/'work';work.mkdir()
        binds=[(source,source,'ro'),(RUN/'data',RUN/'data','ro'),(model,model,'ro'),(work,work,'rw'),
               (FRAMEWORK,FRAMEWORK,'ro'),
               (ROOT/'results/training-vlm-swift-offline-runtime-v1','/swift-runtime','ro'),
               (ROOT/'results/training-qwen-vision014-CPU-v2/vision','/vision','ro'),
               (ROOT/'results/training-minicpm-hygon-processor-r2/cpu/runtime-site','/minicpm-transformers','ro'),
               (Path('/opt/hyhal'),'/opt/hyhal','ro')]
        container=['/public/software/apps/apptainer/1.3.4/bin/apptainer','exec','--contain','--no-home','--pwd','/']
        for host,target,mode in binds:
            actual=runtime_bind_source(host)
            container+=['--bind',f'{actual}:{target}:{mode}']
        argv,budget=command(container,str(source/'inside.py'),
            dict(framework=str(FRAMEWORK),container=str(ROOT/'containers/dtk26-sandbox-r1')),
            LABEL,deadline,mono_end,time.time(),time.monotonic())
        # One actual argv budget function is used here and in local boundary checks.
        save(RUN/'intent.json',dict(publication=publication,claim_SHA256=expected_claim,epoch=epoch,deadline=deadline,
            command=argv,budget=budget,CPU330=qualified330,data=dataset,actual_weights_hashed=0,
            maximum_invocations=1,GPU_cards=0,replay_allowed=False))
    except BaseException as error:
        save(RUN/'outcome.json',dict(passed=False,phase='PRELAUNCH',error_type=type(error).__name__,
            epoch=epoch,deadline=deadline,actual_new_Slurm_steps=0,planned_rows={s:144 for s in ('train','dev','blind')},
            GPU_cards=0,current_step_outcome='NOT_SUBMITTED',replay_allowed=False))
        raise
    rc,timed_out,child=None,False,None
    owned=None
    with (RUN/'step.log').open('xb') as log:
        try:
            child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                proc=Path('/proc',str(child.pid),'stat').read_text().rsplit(') ',1)[1].split()
                owned=dict(PID=child.pid,PGID=int(proc[2]),startticks=int(proc[19]))
                save(RUN/'owned-srun.json',owned)
            except FileNotFoundError:save(RUN/'owned-srun.json',dict(PID=child.pid,exited_before_identity_observation=True))
            try:rc=child.wait(timeout=min(budget['client_seconds'],remaining(40)))
            except subprocess.TimeoutExpired:timed_out=True
        except OSError:timed_out=True
    cleanup=[]
    if child is not None and child.poll() is None and owned is not None:
        for sig in (signal.SIGTERM,signal.SIGKILL):
            try:
                proc=Path('/proc',str(child.pid),'stat').read_text().rsplit(') ',1)[1].split()
                require(int(proc[2])==owned['PGID']==owned['PID'] and int(proc[19])==owned['startticks'],
                        'owned client identity changed')
                os.killpg(owned['PGID'],sig);cleanup.append(int(sig))
                child.wait(timeout=min(5,remaining(35)))
                break
            except subprocess.TimeoutExpired:continue
            except (OSError,ValueError):break
    errors=[]
    def observe():
        try:return accounting('1103606',LABEL)
        except (OSError,ValueError,subprocess.SubprocessError) as error:
            errors.append(type(error).__name__);return None
    terminal_end=min(mono_end-10,time.monotonic()+35)
    visibility=wait_terminal(observe,terminal_end,interval=2,max_observations=18)
    row=visibility['row'];receipt=None;native=None;receipt_errors=[]
    for name,key in [('CPU-receipt.json','receipt'),('native/progress.json','native')]:
        path=RUN/'work'/name
        if path.is_file() and not path.is_symlink() and path.stat().st_size<=262144:
            try:
                value=read(path)
                if key=='receipt':receipt=value
                else:native=value
            except (OSError,ValueError) as error:receipt_errors.append(dict(file=name,error_type=type(error).__name__))
    passed=bool(rc==0 and not timed_out and not cleanup and row and is_terminal(row)
        and row[2:4]==['COMPLETED','0:0'] and row[5]=='f13r1n11' and row[0].startswith('1103606.')
        and 'gres/dcu=' not in row[6] and 'gres/gpu=' not in row[6]
        and not receipt_errors and receipt and receipt.get('passed') is True and receipt.get('deadline')==deadline
        and native and native.get('state')=='NATIVE_CPU_INPUT_AND_TRAIN_COLLATOR_PASS'
        and native.get('completed_rows')=={s:144 for s in ('train','dev','blind')}
        and native.get('actual_native_encode_calls')==576 and native.get('media_SHA_verified')==48
        and native.get('model_weights_loaded') is False and native.get('actual_GPU_cards')==native.get('actual_updates')==0)
    log=RUN/'step.log'
    with log.open('rb') as stream:bounded=stream.read(4*1024**2)
    result=dict(passed=passed,epoch=epoch,deadline=deadline,observed_epoch=time.time(),rc=rc,client_timeout=timed_out,
        owned_client_cleanup_signals=cleanup,accounting_columns=COLUMNS,accounting=visibility,accounting_errors=errors,
        CPU_receipt=receipt,native_progress=native,receipt_errors=receipt_errors,
        planned_rows={s:144 for s in ('train','dev','blind')},
        log_bytes=log.stat().st_size,bounded_log_SHA256=hashlib.sha256(bounded).hexdigest(),
        GPU_cards=0,actual_updates=0,current_step_outcome='TERMINAL' if row and is_terminal(row) else 'UNKNOWN',
        formal_training_acceptance=False,semantic_readability='UNKNOWN',replay_allowed=False)
    save(RUN/'outcome.json',result)
    print(json.dumps(dict(passed=passed,current_step_outcome=result['current_step_outcome'],GPU_cards=0)),flush=True)
    return 0 if passed else 1


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--publication',required=True)
    p.add_argument('--claim-sha256',required=True);a=p.parse_args()
    raise SystemExit(main(a.publication,a.claim_sha256))
