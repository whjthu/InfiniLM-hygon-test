import os,re,signal,subprocess,time
from pathlib import Path
from common import D,P,RUN,need,save,load,own,budget
from held_child import cleanup_held
COLS='JobIDRaw,JobName%100,State%50,ExitCode,ElapsedRaw,NodeList%500,AllocTRES%200'
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
def main(context):
    epoch=context['epoch'];left=budget(epoch,113);need(time.time()<epoch+25,'EARLY_DISPATCH_BUDGET')
    owner=own();save(RUN/'host-owner.json',owner)
    child=None;anchor=None;failure=None;held=None;row=None;accounting_errors=[]
    def query(argv):
        r=subprocess.run(argv,stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=min(5,left(5)))
        need(r.returncode==0 and len(r.stdout)<=1048576,'BOUNDED_ALLOCATION_QUERY');return r.stdout.strip().splitlines()
    def accounting():
        rows=[v for line in query(['sacct','-n','-P','-j',P['parent'],'-o',COLS]) if len(v:=line.split('|'))==7 and v[1]==P['label']]
        need(len(rows)<=1,'ONCE_LABEL_AMBIGUOUS');return rows[0] if rows else None
    def interrupt(*_):raise TimeoutError('ORIGINAL_HOST_STOP')
    for s in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP,signal.SIGALRM):signal.signal(s,interrupt)
    signal.setitimer(signal.ITIMER_REAL,max(.001,epoch+95-time.time()))
    save(RUN/'control-state.json',dict(context,exec_request_attempted=False))
    try:
        q=query(['scontrol','show','job',P['parent'],'--oneliner']);need(len(q)==1,'PARENT_RECORD')
        f=dict(v.split('=',1) for v in q[0].split() if '=' in v)
        match=re.fullmatch(r'[^()]+\((\d+)\)',f.get('UserId',''))
        need(match and int(match[1])==os.getuid() and f['JobState']=='RUNNING','NUMERIC_PARENT_OWNER_RUNNING')
        need(P['node'] in query(['scontrol','show','hostnames',f['NodeList']]),'CURRENT_NODE_MEMBERSHIP')
        need(accounting() is None,'ONCE_LABEL_ALREADY_USED_NO_REPLAY')
        for line in query(['scontrol','show','step',P['parent'],'--oneliner']):
            step=dict(v.split('=',1) for v in line.split() if '=' in v);suffix=step.get('StepId','').rsplit('.',1)[-1]
            if suffix not in ('batch','extern'):
                need('NodeList' in step and P['node'] not in query(['scontrol','show','hostnames',step['NodeList']]),'NODE_ACTIVE_OR_UNKNOWN_STEP')
        step_end=time.time()+60;need(step_end<=epoch+90,'ACTUAL60_WITHIN_ORIGINAL_EPOCH')
        argv=['srun','--jobid='+P['parent'],'--job-name='+P['label'],'--exact','--immediate=5','--cpu-bind=cores','-N1','-n1','-w',P['node'],'-c2','--mem=2048M','--gres=none','--time=00:01:00','/usr/local/python3.12/bin/python3','-B',str(D/'node_probe.py'),'--epoch',str(epoch),'--step-end',str(step_end)]
        save(RUN/'step-intent.json',dict(context,command=argv,step_end=step_end,Slurm_seconds=60,owner=owner))
        with (RUN/'srun.stdout.private').open('xb') as out,(RUN/'srun.stderr.private').open('xb') as err:
            save(RUN/'control-state.json',dict(context,exec_request_attempted=True))
            child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
            anchor=dict(PID=child.pid,PGID=child.pid,start_new_session=True,supervisor=owner);save(RUN/'held-srun.json',anchor)
            child.wait(timeout=min(epoch+95-time.time(),left(18)))
    except BaseException as error:failure=dict(type=type(error).__name__,reason=str(error)[:256])
    finally:
        for s in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(s,signal.SIG_IGN)
        signal.signal(signal.SIGALRM,lambda *_:os._exit(124));signal.setitimer(signal.ITIMER_REAL,max(.001,epoch+115-time.time()))
        try:held=cleanup_held(child,anchor,epoch+105)
        except BaseException as error:held=dict(reaped=False,error_type=type(error).__name__)
    while time.time()<epoch+112:
        try:row=accounting()
        except Exception as error:accounting_errors.append(type(error).__name__);break
        if row and row[2].split()[0] in TERMINAL:break
        time.sleep(min(.5,max(0,epoch+112-time.time())))
    node=load(RUN/'node-result.json') if (RUN/'node-result.json').exists() else None
    terminal=bool(row and row[2].split()[0] in TERMINAL)
    verified=bool(failure is None and held and held.get('reaped') and not held.get('termination_needed') and child and child.returncode==0
        and terminal and row[2:4]==['COMPLETED','0:0'] and row[5]==P['node'] and set(row[6].split(','))=={'cpu=2','mem=2G','node=1'}
        and node and node['diagnostic_complete'] and node['step']==row[0])
    result=dict(context,diagnostic_receipt_verified=verified,frontend_ready=bool(verified and node['frontend_ready']),failure=failure,
        held_srun=held,accounting_columns=COLS,accounting=row,accounting_errors=accounting_errors,node=node,
        execution_state='TERMINAL' if terminal else 'UNKNOWN',GPU_GO=False,Model_GO=False,replay_allowed=False)
    save(RUN/'outcome.json',result);signal.setitimer(signal.ITIMER_REAL,0);return 0 if verified else 1
