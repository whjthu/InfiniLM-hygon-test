"""One fresh180s normal640-DCU allocation counter window without GPU computation, no service and no fake GPU work."""
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
from admission import need, parent_gate, step_gate
from qualified_sampler import aggregate, node_map
from distribution import describe
from slurm_accounting import wait_terminal

D=Path(__file__).resolve().parent
ROOT=D.parent
COLUMNS='JobIDRaw,JobName%120,State,ExitCode,ElapsedRaw,NodeList,AllocCPUS,ReqMem,AllocTRES'
H=lambda b:hashlib.sha256(b).hexdigest()


def save(path,value):
    raw=(json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()
    need(len(raw)<=4*1024**2,'CONTROLLER_METADATA_CAP')
    with path.open('xb') as f:
        f.write(raw);f.flush();os.fsync(f.fileno())
    return dict(bytes=len(raw),SHA256=H(raw))


def ticks(pid):
    return int(Path('/proc/%d/stat'%pid).read_text().rsplit(')',1)[1].split()[19])


def memory_bytes(text):
    m=re.fullmatch(r'(\d+)([KMGTP]?)',text);need(m is not None,'TRES_MEMORY_SCHEMA')
    return int(m[1])*({'':1,'K':1024,'M':1024**2,'G':1024**3,'T':1024**4,'P':1024**5}[m[2]])

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--epoch',type=float,required=True)
    p.add_argument('--request-id',required=True)
    p.add_argument('--publication',required=True)
    a=p.parse_args()
    need(re.fullmatch('[0-9a-f]{32}',a.request_id) and re.fullmatch('[0-9a-f]{40}',a.publication),'CONTEXT')
    need(0<=time.time()-a.epoch<130,'ORIGINAL_EPOCH')
    deadline=a.epoch+180
    mono=time.monotonic()+deadline-time.time()
    def left(reserve=0):
        remaining=min(deadline-time.time(),mono-time.monotonic())-reserve
        need(remaining>0,'ORIGINAL_DEADLINE')
        return remaining
    def stop(_s,_f):
        raise TimeoutError('ORIGINAL_WORK_WINDOW')
    catchable=[signal.SIGTERM,signal.SIGINT,signal.SIGHUP]
    for s in catchable:
        signal.signal(s,stop)
    signal.signal(signal.SIGALRM,stop)
    signal.setitimer(signal.ITIMER_REAL,left(35))
    allowed=sorted(os.sched_getaffinity(0)-{0,1})
    need(len(allowed)>=2,'CPU2_EXCLUDING0_1')
    os.sched_setaffinity(0,allowed[:2])
    inherited=os.getpriority(os.PRIO_PROCESS,0);requested=max(inherited,10)
    os.nice(requested-inherited)
    resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3))
    from verify_package import verify
    verify(D,adopted=True)
    ledger=json.loads((D/'ledger.json').read_bytes());owners=node_map(ledger)
    parents={job:sorted(n for n,(p,_) in owners.items() if p==job) for job in ('1087656','1103606')}
    context=dict(request_id=a.request_id,publication=a.publication,epoch=a.epoch,deadline=deadline,
                 PID=os.getpid(),startticks=ticks(os.getpid()),host=socket.gethostname(),uid=os.getuid(),
                 boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),cpu_affinity=allowed[:2],
                 inherited_nice=inherited,requested_nice=requested,actual_nice=os.getpriority(os.PRIO_PROCESS,0))
    save(ROOT/'controller-intent.json',context)
    result=dict(context=context,qualified=False,current_driver_busy=None,planned_nodes=80,planned_DCU=640,
                expected_samples=1920,points=3,interval_s=2,steps={},failures={},accounting_columns=COLUMNS.split(','),
                admission={},no_GPU_work_started=True,GPU_compute=False,normal_DCU_allocation_requested=640,old34e_state='UNKNOWN_UNCHANGED_NO_REPLAY',
                clock_sync='UNKNOWN_NOT_INDEPENDENTLY_MEASURED',
                scheduled_window='nominal_shared_wall_epoch_3points_2s_spacing')
    children={};cleanup={};observed={};pins={};labels={}
    def query(*argv,stop_at=None):
        timeout=min(8,left(20))
        if stop_at is not None:
            timeout=min(timeout,max(.001,stop_at-time.monotonic()))
        r=subprocess.run(argv,capture_output=True,text=True,timeout=timeout)
        need(r.returncode==0,'METADATA_QUERY_FAILED')
        need(len(r.stdout.encode())<=256*1024,'QUERY_OUTPUT_BOUND')
        return r.stdout.strip().splitlines()
    def expand(expression):
        need(bool(expression),'EMPTY_HOST_EXPRESSION')
        return query('scontrol','show','hostnames',expression)
    def account(job,label,window):
        rows=query('sacct','-n','-P','-j',job,'--format='+COLUMNS,stop_at=window)
        matches=[r.split('|') for r in rows if len(r.split('|'))==9 and r.split('|')[1]==label]
        need(len(matches)<=1,'DUPLICATE_EXACT_LABEL')
        return matches[0] if matches else None
    scheduled=None
    try:
        for job,nodes in parents.items():
            rows=query('scontrol','show','job',job,'-o');need(len(rows)==1,'EXACT_PARENT_ROW')
            parent=parent_gate(rows[0],job,os.getuid(),nodes,expand)
            # Parsing uses the login host's own configured local timezone.
            end=time.mktime(time.strptime(parent['EndTime'],'%Y-%m-%dT%H:%M:%S'))
            need(end>=deadline,'PARENT_LIFETIME_COVERS_ORIGINAL_EPOCH')
            current=step_gate(query('scontrol','show','step',job,'-o'),job)
            result['admission'][job]=dict(parent=parent,current_steps=current,expanded_nodes=nodes)
            labels[job]='training80-current-busy-v2-'+a.request_id+'-'+job
            need(account(job,labels[job],None) is None,'ORIGINAL_LABEL_ALREADY_HAS_ACCOUNTING')
        need(len(set(parents['1087656'])|set(parents['1103606']))==80,'EIGHTY_UNIQUE')
        need(left()>=85,'NO_NEW_STEP_WITHOUT60_PLUS_CLEANUP_BUDGET')
        samples=ROOT/'nodes';samples.mkdir(mode=0o700)
        scheduled=time.time()+22
        need(scheduled+5<deadline-35,'COMMON_WINDOW_WITHIN_ORIGINAL_DEADLINE')
        result['scheduled_epoch']=scheduled
        save(ROOT/'admission.json',dict(context=context,parents=result['admission'],labels=labels,scheduled_epoch=scheduled))
        for job,nodes in parents.items():
            need(left()>=80,'LATE_SECOND_STEP_NOT_SUBMITTED')
            argv=['srun','--jobid='+job,'--job-name='+labels[job],'--exact','--immediate=5','--nodes=40','--ntasks=40',
                  '--ntasks-per-node=1','--nodelist='+','.join(nodes),'--cpus-per-task=1','--mem=256M','--gres=dcu:8',
                  '--time=00:01:00','--kill-on-bad-exit=0','/usr/local/python3.12/bin/python3','-I','-S','-B','-c',
                  "import runpy,sys;sys.path.insert(0,sys.argv.pop(1));sys.argv=sys.argv[1:];runpy.run_path(sys.argv[0],run_name='__main__')",
                  str(D),str(D/'node_sample.py'),
                  '--ledger',str(D/'ledger.json'),'--output',str(samples),'--epoch',str(a.epoch),'--scheduled',str(scheduled),
                  '--request-id',a.request_id]
            with (ROOT/(job+'.stdout')).open('xb') as stdout,(ROOT/(job+'.stderr')).open('xb') as stderr:
                child=subprocess.Popen(argv,stdout=stdout,stderr=stderr,start_new_session=True)
                children[job]=child  # Held Popen precedes any fallible identity read/write.
                need(left(35)>0,'DEADLINE_AFTER_POPEN')
                save(ROOT/(job+'-owned-srun.json'),dict(context=context,PID=child.pid,PGID=child.pid,
                                                       startticks=ticks(child.pid),label=labels[job]))
        for job,child in children.items():
            wait=min(left(35),scheduled+14-time.time())
            need(wait>0,'COMMON_WINDOW_CONTROLLER_WAIT_EXHAUSTED')
            child.wait(timeout=wait)
            if child.returncode:
                result['failures'][job]=dict(srun_rc=child.returncode)
    except BaseException as exc:
        result['failures']['controller']=dict(type=type(exc).__name__,reason_SHA256=H(str(exc).encode()))
    finally:
        for s in catchable:
            signal.signal(s,signal.SIG_IGN)
        signal.signal(signal.SIGALRM,lambda *_:os._exit(124))
        signal.setitimer(signal.ITIMER_REAL,max(.001,min(deadline-time.time(),mono-time.monotonic())))
        for job,child in children.items():
            entry=dict(held_PID=child.pid,TERM=False,KILL=False,reaped=False,errors=[]);cleanup[job]=entry
            try:
                if child.poll() is None:
                    entry['TERM']=True
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=max(0,min(5,deadline-time.time()-20)))
                    except subprocess.TimeoutExpired:
                        entry['KILL']=True;os.killpg(child.pid,signal.SIGKILL)
                child.wait(timeout=max(0,min(3,deadline-time.time()-18)));entry['reaped']=True
            except (OSError,subprocess.TimeoutExpired) as exc:
                entry['errors'].append(type(exc).__name__)
            entry['returncode']=child.poll()
    result['owned_cleanup']=cleanup
    try:
        for job in children:
            # All query/sleep time shrinks from the original epoch. No new12s epoch.
            terminal_deadline=min(mono-6,time.monotonic()+8)
            wait=wait_terminal(lambda:account(job,labels[job],terminal_deadline),terminal_deadline,interval=1,max_observations=3)
            result['steps'][job]=wait
        for node,(job,_) in owners.items():
            path=ROOT/'nodes'/(node+'.json')
            if not path.exists():
                continue
            need(not path.is_symlink() and path.stat().st_size<=32768,'NODE_RECEIPT_BOUND')
            raw=path.read_bytes();row=json.loads(raw)
            need(row['node']==node and row['request_id']==a.request_id and row['epoch']==a.epoch,'NODE_RECEIPT_CONTEXT')
            if row['error_type'] is not None:
                result['failures'][node]=dict(type=row['error_type'],reason_SHA256=row['error_reason_SHA256'])
                continue
            accounting=result['steps'].get(job,{}).get('row')
            need(accounting and row['step']==accounting[0],'SAVED_NODE_ACTUAL_STEP')
            observed[node]=row;pins[node]=sorted(row['PCI_KFD'])
        if scheduled is not None:
            distribution=describe(owners,pins,observed,scheduled)
            card_pin=save(ROOT/'per-card-window-private.json',distribution.pop('per_card_private'))
            result['card_window_distribution']=dict(distribution,per_card_private=card_pin)
            if set(observed)==set(owners):
                summary=aggregate(owners,pins,observed,scheduled)
            else:
                # No synthetic PCI/card identities for missing nodes, and no full-fleet mean.
                summary=dict(complete=False,expected_DCU=640,observed_DCU=len(observed)*8,
                             missing_nodes=sorted(set(owners)-set(observed)),fleet_mean_busy_percent=None,
                             expected_samples=1920,observed_points=sum(len(r['rows']) for r in observed.values()),
                             points=3,interval_s=2,period_s=4,scheduled_epoch=scheduled)
            resources=True
            for job,wait in result['steps'].items():
                row=wait.get('row');tres=dict(x.split('=',1) for x in (row[8].split(',') if row else []) if '=' in x)
                resources=resources and bool(wait['terminal_visible'] and row[2:4]==['COMPLETED','0:0']
                    and set(expand(row[5]))==set(parents[job]) and row[6]=='40'
                    and tres.get('cpu')=='40' and tres.get('node')=='40' and memory_bytes(tres.get('mem',''))==40*256*1024**2
                    and tres.get('gres/dcu')=='320' and all(int(v)==0 for k,v in tres.items() if k.startswith('gres/') and k!='gres/dcu'))
            qualified=bool(len(children)==2 and len(result['steps'])==2 and resources and not result['failures']
                           and all(c['reaped'] and not c['errors'] and not c['TERM'] and not c['KILL'] for c in cleanup.values())
                           and summary['complete'])
            if not qualified:
                summary['fleet_mean_busy_percent']=None
            result['card_window_distribution']['qualified_window']=qualified
            result.update(summary=summary,qualified=qualified,current_driver_busy=summary['fleet_mean_busy_percent'] if qualified else None)
    except BaseException as exc:
        result['failures']['aggregation']=dict(type=type(exc).__name__,reason_SHA256=H(str(exc).encode()))
    result.update(finished=time.time(),node_receipts_observed=len(observed),missing_nodes=sorted(set(owners)-set(observed)),
                  limitation='Only this3point4s sysfs driver-busy window; allocation/CPUsteps are not training or SMoccupancy')
    outcome_pin=save(ROOT/'outcome.json',result)
    bounded={k:result[k] for k in ('qualified','current_driver_busy','planned_nodes','planned_DCU','expected_samples',
             'points','interval_s','steps','failures','accounting_columns','no_GPU_work_started','old34e_state',
             'GPU_compute','normal_DCU_allocation_requested','owned_cleanup','node_receipts_observed','missing_nodes','limitation','clock_sync','scheduled_window')}
    bounded.update(outcome=outcome_pin,summary=result.get('summary'),request_id=a.request_id)
    bounded['card_window_distribution']=result.get('card_window_distribution')
    raw=json.dumps(bounded,separators=(',',':'),allow_nan=False)
    need(len(raw.encode())<=32768,'BOUNDED_FLEET_SUMMARY')
    print(raw,flush=True)
    return 0 if result['qualified'] else 2


if __name__=='__main__':
    raise SystemExit(main())
