"""Bounded single-node metadata diagnosis; no fleet/counter collection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import signal
import socket
import subprocess
import time
from admission import need,parent_gate,step_gate
from slurm_accounting import wait_terminal

D=Path(__file__).resolve().parent;ROOT=D.parent
PARENT='1087656';NODE='a01r4n14'
COLUMNS='JobIDRaw,JobName%120,State,ExitCode,ElapsedRaw,NodeList,AllocCPUS,ReqMem,AllocTRES'
H=lambda b:hashlib.sha256(b).hexdigest()


def save(name,value):
    raw=(json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode();need(len(raw)<=32768,'SMALL_DIAGNOSTIC')
    with (ROOT/name).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    return dict(bytes=len(raw),SHA256=H(raw))


def memory_bytes(text):
    m=re.fullmatch(r'(\d+)([KMGTP]?)',text);need(m is not None,'TRES_MEMORY_SCHEMA')
    return int(m[1])*({'':1,'K':1024,'M':1024**2,'G':1024**3,'T':1024**4,'P':1024**5}[m[2]])


def main():
    p=argparse.ArgumentParser();p.add_argument('--epoch',type=float,required=True)
    p.add_argument('--request-id',required=True);p.add_argument('--publication',required=True);a=p.parse_args()
    need(re.fullmatch('[0-9a-f]{32}',a.request_id) and re.fullmatch('[0-9a-f]{40}',a.publication),'CONTEXT')
    need(0<=time.time()-a.epoch<45,'ORIGINAL_EPOCH')
    deadline=a.epoch+90;mono=time.monotonic()+deadline-time.time()
    def left(reserve=0):
        n=min(deadline-time.time(),mono-time.monotonic())-reserve;need(n>0,'ORIGINAL_DEADLINE');return n
    def stop(_s,_f):raise TimeoutError('ORIGINAL_WORK_DEADLINE')
    for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,stop)
    signal.setitimer(signal.ITIMER_REAL,left(30))
    affinity=sorted(os.sched_getaffinity(0)-{0,1});need(len(affinity)>=2,'CONTROLLER_CPU2');os.sched_setaffinity(0,affinity[:2])
    resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3))
    from verify_package import verify
    verify(D,adopted=True)
    label='training80-stage-CPU-v1-'+a.request_id
    ticks=lambda pid:int(Path('/proc/%d/stat'%pid).read_text().rsplit(')',1)[1].split()[19])
    ctx=dict(request_id=a.request_id,publication=a.publication,epoch=a.epoch,deadline=deadline,host=socket.gethostname(),
             boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),PID=os.getpid(),startticks=ticks(os.getpid()),
             uid=os.getuid(),parent=PARENT,node=NODE,label=label)
    save('intent.json',ctx)
    result=dict(context=ctx,original_sampler_passed=False,diagnostic_receipt_verified=False,stage_receipt=None,
                actual_accounting=None,accounting_columns=COLUMNS.split(','),failures={},GPU_busy_reads=0)
    child=None;cleanup=dict(reaped=False,TERM=False,KILL=False,errors=[])
    def query(*argv,stop_at=None):
        timeout=min(5,left(15))
        if stop_at is not None:timeout=min(timeout,max(.001,stop_at-time.monotonic()))
        r=subprocess.run(argv,capture_output=True,text=True,timeout=timeout);need(r.returncode==0,'QUERY_FAILED')
        need(len(r.stdout.encode())<=256*1024,'QUERY_BOUND');return r.stdout.strip().splitlines()
    def account(window=None):
        matches=[r.split('|') for r in query('sacct','-n','-P','-j',PARENT,'--format='+COLUMNS,stop_at=window)
                 if len(r.split('|'))==9 and r.split('|')[1]==label]
        need(len(matches)<=1,'DUPLICATE_EXACT_LABEL');return matches[0] if matches else None
    try:
        ledger=json.loads((D/'ledger.json').read_bytes());nodes=sorted(n for g in ledger['groups'].values() for n in g['allocations'][PARENT])
        need(len(nodes)==len(set(nodes))==40 and NODE in ledger['groups']['T01']['allocations'][PARENT],'EXACT_LEDGER_NODE')
        raw=query('scontrol','show','job',PARENT,'-o');need(len(raw)==1,'EXACT_PARENT')
        result['parent_gate']=parent_gate(raw[0],PARENT,os.getuid(),nodes,lambda expr:query('scontrol','show','hostnames',expr))
        end=time.mktime(time.strptime(result['parent_gate']['EndTime'],'%Y-%m-%dT%H:%M:%S'));need(end>=deadline,'PARENT_LIFETIME')
        result['step_gate']=step_gate(query('scontrol','show','step',PARENT,'-o'),PARENT)
        need(account() is None,'NEW_LABEL_DEDUP');need(left()>=45,'NO_LATE_STEP')
        step_origin=time.time()
        command=['srun','--jobid='+PARENT,'--job-name='+label,'--exact','--immediate=5','--nodes=1','--ntasks=1',
                 '--nodelist='+NODE,'--cpus-per-task=1','--mem=256M','--gres=none','--time=00:01:00',
                 '/usr/local/python3.12/bin/python3','-I','-S','-B',str(D/'stage_worker.py'),'--epoch',str(a.epoch),
                 '--step-origin',str(step_origin),'--request-id',a.request_id,'--output',str(ROOT/'stage-receipt.json')]
        with (ROOT/'worker.stdout').open('xb') as out,(ROOT/'worker.stderr').open('xb') as err:
            child=subprocess.Popen(command,stdout=out,stderr=err,start_new_session=True)
            need(left(30)>0,'DEADLINE_AFTER_HELD_POPEN')
            save('owned-srun.json',dict(context=ctx,PID=child.pid,PGID=child.pid,startticks=ticks(child.pid),step_origin=step_origin))
            child.wait(timeout=min(25,left(30)))
    except BaseException as exc:result['failures']['controller']=dict(type=type(exc).__name__,reason_SHA256=H(str(exc).encode()))
    finally:
        for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,signal.SIG_IGN)
        signal.signal(signal.SIGALRM,lambda *_:os._exit(124))
        signal.setitimer(signal.ITIMER_REAL,max(.001,min(deadline-time.time(),mono-time.monotonic())))
        if child is not None:
            try:
                if child.poll() is None:
                    cleanup['TERM']=True;os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=max(0,min(2,deadline-time.time()-20)))
                    except subprocess.TimeoutExpired:
                        cleanup['KILL']=True;os.killpg(child.pid,signal.SIGKILL)
                child.wait(timeout=max(0,min(3,deadline-time.time()-17)));cleanup['reaped']=True
            except (OSError,subprocess.TimeoutExpired) as exc:cleanup['errors'].append(type(exc).__name__)
            cleanup['returncode']=child.poll()
    result['cleanup']=cleanup
    try:
        if child is not None:
            end=min(mono-8,time.monotonic()+8)
            observed=wait_terminal(lambda:account(end),end,interval=1,max_observations=3)
            result['accounting_observations']=observed;row=observed.get('row');result['actual_accounting']=row
            if row is not None:
                receipt=ROOT/'stage-receipt.json';need(not receipt.is_symlink() and receipt.stat().st_size<=32768,'SMALL_EXACT_STAGE_RECEIPT')
                raw=receipt.read_bytes();value=json.loads(raw)
                need(value['request_id']==a.request_id and value['epoch']==a.epoch and value['node']==NODE and value['step']==row[0], 'STAGE_RECEIPT_IDENTITY')
                result['stage_receipt']=dict(bytes=len(raw),SHA256=H(raw),value=value)
                tres=dict(x.split('=',1) for x in row[8].split(',') if '=' in x)
                need(row[5]==NODE and row[6]=='1' and tres.get('cpu')=='1' and tres.get('node')=='1'
                     and memory_bytes(tres.get('mem',''))==256*1024**2
                     and all(int(v)==0 for k,v in tres.items() if k.startswith('gres/')),'ACTUAL_CPU1_MEM256M_ZERO_GRES')
                need(int(row[4])<=30,'ACTUAL_STEP_ELAPSED_EXCEEDS30')
                result['diagnostic_receipt_verified']=bool(observed['terminal_visible']
                    and row[2:4]==['COMPLETED','0:0'] and child.returncode==0
                    and cleanup['reaped'] and not cleanup['errors'] and not cleanup['TERM'] and not cleanup['KILL']
                    and value['diagnostic_complete'])
    except BaseException as exc:result['failures']['audit']=dict(type=type(exc).__name__,reason_SHA256=H(str(exc).encode()))
    result['finished']=time.time();pin=save('outcome.json',result)
    print(json.dumps(dict(result=result,outcome_pin=pin),separators=(',',':'),allow_nan=False),flush=True)
    return 0 if result['diagnostic_receipt_verified'] and not result['failures'] else 2


if __name__=='__main__':raise SystemExit(main())
