"""Exactly one fresh CPU step, one900s epoch; no original three-stream replay or polling."""
import json,os,re,signal,subprocess,threading,time
from datetime import datetime
from common import D,P,need,save,own,budget,control,load,sha,canonical
from dependency_gate import validate_snapshot
from held_child import cleanup_held
COLS='JobIDRaw,JobName%100,State%50,ExitCode,ElapsedRaw,NodeList%500,AllocTRES%200'
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}

def main(context,snapshot):
    validate_snapshot(snapshot);need(sha(canonical(snapshot))==context['dependency_snapshot_SHA256'],'EXACT_ORIGINAL_PASS_SNAPSHOT');epoch=context['epoch'];left=budget(epoch,P['budget']['producer_accounting_stop']);root=D.parent;owner=own();need(owner['UID']==154459 and context['deadline']==epoch+900,'NEW900_OWNER_CONTEXT')
    child=None;anchor=None;held=None;row=None;worker=None;error=None;attempted=False;reader=None;stdout=bytearray()
    def stop(*_):raise TimeoutError('ORIGINAL_ASSEMBLY_HOST_STOP')
    for s in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(s,stop)
    signal.setitimer(signal.ITIMER_REAL,max(.001,epoch+P['budget']['host_stop']-time.time()))
    def query(argv):
        r=subprocess.run(argv,stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=min(5,left(5)));need(r.returncode==0 and len(r.stdout)+len(r.stderr)<=1048576,'BOUNDED_FRESH_NEW_SCOPE_QUERY');return r.stdout.strip().splitlines()
    def accounting():
        rows=[v for line in query(['sacct','-n','-P','-j',P['parent'],'-o',COLS]) if len(v:=line.split('|'))==7 and v[1]==P['label']];need(len(rows)<=1,'NEW_LABEL_ALREADY_USED_OR_AMBIGUOUS');return rows[0] if rows else None
    save(root/'host-intent.json',dict(context,owner=owner,exec_request_attempted=False))
    try:
        need(time.time()<epoch+P['budget']['admission_end'],'EARLY_NEW_CPU_ADMISSION')
        lines=query(['scontrol','show','job',P['parent'],'--oneliner']);need(len(lines)==1,'ONE_PARENT');fields=dict(x.split('=',1) for x in lines[0].split() if '='in x);uid=re.fullmatch(r'[^()]+\((\d+)\)',fields.get('UserId',''))
        need(uid and int(uid[1])==os.getuid() and fields['JobState']=='RUNNING' and datetime.fromisoformat(fields['EndTime']).timestamp()>=epoch+900,'FRESH_RUNNING_OWN_PARENT_LEASE900')
        need(P['node'] in query(['scontrol','show','hostnames',fields['NodeList']]) and accounting() is None,'EXACT_NODE_AND_UNUSED_LABEL')
        for line in query(['scontrol','show','step',P['parent'],'--oneliner']):
            f=dict(x.split('=',1) for x in line.split() if '='in x);suffix=f.get('StepId','').rsplit('.',1)[-1]
            if suffix not in ('batch','extern'):need('NodeList'in f and P['node'] not in query(['scontrol','show','hostnames',f['NodeList']]),'FIXED_NODE_CONFLICT_OR_UNKNOWN')
        need(time.time()<epoch+P['budget']['admission_end'],'ADMISSION_END_NO_SUBMIT')
        seconds=int((epoch+P['budget']['step_end_max']-time.time()-15)//60)*60;need(seconds>=600,'INSUFFICIENT_ORIGINAL_STEP_BUDGET');step_end=time.time()+seconds;need(step_end<=epoch+790,'SHRUNK_STEP_END')
        h,rem=divmod(seconds,3600);m,s=divmod(rem,60);t=f'{h:02d}:{m:02d}:{s:02d}';node_context=dict(context,step_end=step_end);save(root/'node-context.json',node_context)
        argv=['srun','--jobid='+P['parent'],'--job-name='+P['label'],'--exact','--immediate=5','--cpu-bind=cores','-N1','-n1','-w',P['node'],'-c4','--mem=4096M','--gres=none','--time='+t,'/usr/local/python3.12/bin/python3','-B',str(D/'node_entry.py')]
        save(root/'step-intent.json',dict(context,command=argv,Slurm_seconds=seconds,step_end=step_end,owner=owner))
        with (root/'srun.stderr.private').open('xb') as err:
            attempted=True;save(root/'host-control.json',dict(context,exec_request_attempted=True));child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=err,start_new_session=True);anchor=dict(PID=child.pid,PGID=child.pid,start_new_session=True,supervisor=owner);save(root/'held-srun.json',anchor)
            reader_error=[]
            def drain():
                try:
                    while True:
                        b=os.read(child.stdout.fileno(),16384)
                        if not b:break
                        need(len(stdout)+len(b)<=65536,'SMALL_WORKER_TERMINAL_BOUND');stdout.extend(b)
                except BaseException as e:reader_error.append(type(e).__name__)
            reader=threading.Thread(target=drain,daemon=True);reader.start()
            while child.poll() is None:
                left(30);need(not reader_error and (root/'srun.stderr.private').stat().st_size<=262144,'WORKER_LOG_BOUND')
                try:child.wait(timeout=min(.5,left(30)))
                except subprocess.TimeoutExpired:pass
            reader.join(timeout=min(3,left(20)));need(not reader.is_alive() and not reader_error and stdout,'BOUNDED_WORKER_TERMINAL');worker=json.loads(stdout);need(all(worker[k]==v for k,v in context.items()),'ORIGINAL_WORKER_CONTEXT');save(root/'worker-terminal.json',worker)
    except BaseException as e:error=dict(type=type(e).__name__,reason=sha(str(e).encode()))
    finally:
        signal.signal(signal.SIGALRM,lambda *_:os._exit(124));signal.setitimer(signal.ITIMER_REAL,max(.001,epoch+P['budget']['client_end']-time.time()))
        for s in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(s,signal.SIG_IGN)
        try:held=cleanup_held(child,anchor,epoch+P['budget']['held_cleanup'])
        except BaseException as e:held=dict(reaped=False,error_type=type(e).__name__)
        if reader is not None:reader.join(timeout=max(0,min(2,epoch+P['budget']['held_cleanup']-time.time())))
    while attempted and time.time()<epoch+P['budget']['producer_accounting_stop']:
        try:row=accounting()
        except BaseException as e:
            if error is None:error=dict(type=type(e).__name__,reason='NEW_ACCOUNTING_FAILED')
            break
        if row and row[2].split()[0] in TERMINAL:break
        time.sleep(min(.5,max(0,epoch+P['budget']['producer_accounting_stop']-time.time())))
    terminal=bool(row and row[2].split()[0] in TERMINAL);passed=bool(error is None and child and child.returncode==0 and worker and worker['passed'] is True and held and held.get('reaped') and not held.get('termination_needed') and row and row[2:4]==['COMPLETED','0:0'] and row[5]==P['node'] and set(row[6].split(','))=={'cpu=4','mem=4G','node=1'} and worker['step']==row[0])
    result=dict(context,passed=passed,worker=worker,host_owner=owner,held_srun=held,accounting_columns=COLS,accounting=row,error=error,exec_request_attempted=attempted,execution_state='TERMINAL' if terminal else ('UNKNOWN' if attempted else 'NOT_SUBMITTED'),finished_epoch=time.time(),GPU_GO=False,Model_GO=False,training_GO=False,replay_allowed=False)
    save(root/'outcome.json',result);control(result);signal.setitimer(signal.ITIMER_REAL,0);return 0 if passed else 1
