import json,os,re,signal,subprocess,time
from datetime import datetime
from pathlib import Path
from common import D,P,RUN,need,save,load,own,budget
from held_child import cleanup_held
COLS='JobIDRaw,JobName%100,State%50,ExitCode,ElapsedRaw,NodeList%500,AllocTRES%200'
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
def main(context):
    epoch=context['epoch'];left=budget(epoch,115);need(time.time()<epoch+35,'EARLY_DISPATCH_BUDGET')
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
    signal.setitimer(signal.ITIMER_REAL,max(.001,epoch+98-time.time()))
    save(RUN/'control-state.json',dict(context,exec_request_attempted=False))
    try:
        q=query(['scontrol','show','job',P['parent'],'--oneliner']);need(len(q)==1,'PARENT_RECORD')
        f=dict(v.split('=',1) for v in q[0].split() if '=' in v)
        match=re.fullmatch(r'[^()]+\((\d+)\)',f.get('UserId',''))
        need(match and int(match[1])==os.getuid() and f['JobState']=='RUNNING','NUMERIC_PARENT_OWNER_RUNNING')
        need(datetime.fromisoformat(f['EndTime']).timestamp()>=epoch+120,'PARENT_REMAINING_LEASE')
        need(P['node'] in query(['scontrol','show','hostnames',f['NodeList']]),'CURRENT_NODE_MEMBERSHIP')
        need(accounting() is None,'ONCE_LABEL_ALREADY_USED_NO_REPLAY')
        for line in query(['scontrol','show','step',P['parent'],'--oneliner']):
            step=dict(v.split('=',1) for v in line.split() if '=' in v);suffix=step.get('StepId','').rsplit('.',1)[-1]
            if suffix not in ('batch','extern'):
                need('NodeList' in step and P['node'] not in query(['scontrol','show','hostnames',step['NodeList']]),'NODE_ACTIVE_OR_UNKNOWN_STEP')
        seconds=60;need(time.time()+seconds<=epoch+95,'FULL60S_STEP_REMAINING_NO_SUBMISSION')
        step_end=time.time()+seconds;need(step_end<=epoch+95,'SLURM_WITHIN_ORIGINAL_EPOCH')
        work_deadline=min(epoch+80,step_end-10);need(work_deadline-time.time()>=10,'MINIMUM_WORK_WINDOW_BEFORE_SRUN')
        save(RUN/'node-context.json',dict(context,step_end=step_end,work_deadline=work_deadline))
        argv=['srun','--jobid='+P['parent'],'--job-name='+P['label'],'--exact','--immediate=5','--cpu-bind=cores','-N1','-n1','-w',P['node'],'-c4','--mem=2048M','--gres=none','--time=00:'+str(seconds//60).zfill(2)+':'+str(seconds%60).zfill(2),'/usr/local/python3.12/bin/python3','-B',str(D/'node_entry.py')]
        save(RUN/'step-intent.json',dict(context,command=argv,step_end=step_end,Slurm_seconds=seconds,work_deadline=work_deadline,owner=owner))
        with (RUN/'srun.stdout.private').open('xb') as out,(RUN/'srun.stderr.private').open('xb') as err:
            save(RUN/'control-state.json',dict(context,exec_request_attempted=True))
            child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
            anchor=dict(PID=child.pid,PGID=child.pid,start_new_session=True,supervisor=owner);save(RUN/'held-srun.json',anchor)
            child.wait(timeout=min(epoch+98-time.time(),left(18)))
    except BaseException as error:failure=dict(type=type(error).__name__,reason_SHA256=__import__('hashlib').sha256(str(error).encode()).hexdigest())
    finally:
        for s in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(s,signal.SIG_IGN)
        signal.signal(signal.SIGALRM,lambda *_:os._exit(124));signal.setitimer(signal.ITIMER_REAL,max(.001,epoch+117-time.time()))
        try:held=cleanup_held(child,anchor,epoch+105)
        except BaseException as error:held=dict(reaped=False,error_type=type(error).__name__)
    while time.time()<epoch+110:
        try:row=accounting()
        except Exception as error:accounting_errors.append(type(error).__name__);break
        if row and row[2].split()[0] in TERMINAL:break
        time.sleep(min(.5,max(0,epoch+110-time.time())))
    node=None
    try:
        if (RUN/'worker/outcome.json').exists():node=load(RUN/'worker/outcome.json')
    except Exception as error:
        if failure is None:failure=dict(type=type(error).__name__,reason='NODE_RESULT_READ_FAILED')
    terminal=bool(row and row[2].split()[0] in TERMINAL)
    verified=bool(failure is None and held and held.get('reaped') and not held.get('termination_needed') and child and child.returncode==0
        and terminal and row[2:4]==['COMPLETED','0:0'] and row[5]==P['node'] and set(row[6].split(','))=={'cpu=4','mem=2G','node=1'}
        and node and node['passed'] is True and node['explicit_body_consumed']==node['body_written']==1048576 and node['whole_SHA_verified'] is False and node['step']==row[0] and all(node[k]==context[k] for k in ('request_id','publication','epoch','source_manifest_SHA256')))
    result=dict(context,passed=verified,range_prefix_producer_verified=verified,all26_assets_qualified=False,failure=failure,
        held_srun=held,accounting_columns=COLS,accounting=row,accounting_errors=accounting_errors,node=node,
        execution_state='TERMINAL' if terminal else ('NOT_SUBMITTED' if child is None and load(RUN/'control-state.json')['exec_request_attempted'] is False else 'UNKNOWN'),GPU_GO=False,Model_GO=False,replay_allowed=False)
    save(RUN/'outcome.json',result)
    independent=None;consumer_error=None
    if terminal and node is not None:
        try:
            need(time.time()<epoch+112,'ORIGINAL_READER_WINDOW_REQUIRED')
            save(RUN/'consumer-intent.json',dict(context,max_invocations=1,asset_body_reads=0,fresh_sacct_calls=1))
            c=subprocess.run([__import__('sys').executable,'-B',str(D/'read_terminal.py'),'--context',str(RUN/'node-context.json')],stdin=subprocess.DEVNULL,capture_output=True,timeout=max(.001,min(5,epoch+115-time.time())))
            need(len(c.stdout)+len(c.stderr)<=262144,'CONSUMER_LOG_BOUND')
            (RUN/'consumer.stdout.private').write_bytes(c.stdout);(RUN/'consumer.stderr.private').write_bytes(c.stderr)
            independent=load(RUN/'independent-result.json');need(c.returncode==0 and independent['independent_terminal_metadata_verified'] is True,'INDEPENDENT_TERMINAL_READER')
        except BaseException as error:consumer_error=dict(type=type(error).__name__)
    accepted=bool(verified and independent and independent.get('range_HTTP_prefix_qualified') is True and consumer_error is None)
    closure=dict(context,passed=accepted,producer_passed=verified,independent_terminal_metadata_verified=bool(independent and independent.get('independent_terminal_metadata_verified')),range_HTTP_prefix_qualified=accepted,consumer_error=consumer_error,whole_SHA_verified=False,Model_GO=False,GPU_GO=False,training_GO=False,replay_allowed=False,finished_epoch=time.time())
    save(RUN/'closure.json',closure)
    summary=json.dumps(dict(context=context,closure=closure,producer=result,independent=independent),allow_nan=False)
    need(len(summary.encode())<=262144,'RETURNED_METADATA_BOUND')
    print(summary,flush=True)
    signal.setitimer(signal.ITIMER_REAL,0);return 0 if accepted else 1
