"""One fresh fixed compute receiver; login only passes bounded RAM bytes and small metadata."""
import os,queue,re,signal,subprocess,threading,time
from datetime import datetime
from common import D,P,need,save,load,own,budget,control,member,verify_marker,exact_fd,line_fd
from held_child import cleanup_held
from protocol import unpack
from control_evidence import persist_bytes,read_control_bytes
from step_visibility import observe_actual_step
COLS='JobIDRaw,JobName%100,State%50,ExitCode,ElapsedRaw,NodeList%500,AllocTRES%200'
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}

def main(context):
    root=D.parent;row_plan=member(context);epoch=context['epoch'];left=budget(epoch,P['budget']['producer_accounting_stop']);need(time.time()<epoch+180,'EARLY_SINGLE_DISPATCH')
    owner=own();need(owner['UID']==154459,'TARGET_OWNER');save(root/'host-owner.json',owner)
    child=None;anchor=None;held=None;failure=None;row=None;endpoint=None;step_attempted=False;reads=[];observed_compute_ready=None;step_queries=[];compute_control_count=0
    pump=dict(read_bytes=0,write_bytes=0,read_calls=0,write_calls=0,read_blocked_seconds=0.,write_blocked_seconds=0.,control_read_bytes=0,control_write_bytes=0,EOF=False,error=None,completed=False,blocks_completed=0)
    pump_thread=None
    host_phase_costs=dict(progress_receipt_persist=dict(seconds=0.,calls=0))
    def query(argv):
        r=subprocess.run(argv,stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=min(5,left(5)));need(r.returncode==0 and len(r.stdout)<=1048576,'BOUNDED_ALLOCATION_QUERY');return r.stdout.strip().splitlines()
    def accounting():
        values=[v for s in query(['sacct','-n','-P','-j',row_plan['parent'],'-o',COLS]) if len(v:=s.split('|'))==7 and v[1]==row_plan['label']]
        need(len(values)<=1,'EXACT_LABEL_DEDUP');return values[0] if values else None
    def receive_line(fd):
        nonlocal compute_control_count
        q=queue.Queue()
        def read():
            try:q.put(('ok',read_control_bytes(fd)))
            except BaseException as e:q.put(('error',type(e).__name__))
        t=threading.Thread(target=read,daemon=True);t.start();reads.append(t)
        kind,raw=q.get(timeout=min(45,left(35)));ordinal=compute_control_count;compute_control_count+=1
        if kind=='ok':persist_bytes(root,'compute-control-%02d.private'%ordinal,raw,65537,context)
        else:save(root/('compute-control-%02d-error.json'%ordinal),dict(context,error_type=raw))
        need(kind=='ok' and 0<len(raw)<=65536 and raw.endswith(b'\n'),'COMPUTE_CONTROL_LINE');import json;return json.loads(raw)
    def interrupt(*_):raise TimeoutError('ORIGINAL_HOST_STOP')
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP,signal.SIGALRM):signal.signal(sig,interrupt)
    signal.setitimer(signal.ITIMER_REAL,max(.001,epoch+P['budget']['host_stop']-time.time()))
    save(root/'control-state.json',dict(context,exec_request_attempted=False))
    try:
        lines=query(['scontrol','show','job',row_plan['parent'],'--oneliner']);need(len(lines)==1,'PARENT_RECORD');f=dict(v.split('=',1) for v in lines[0].split() if '=' in v)
        uid=re.fullmatch(r'[^()]+\((\d+)\)',f.get('UserId',''));need(uid and int(uid[1])==os.getuid() and f['JobState']=='RUNNING','OWN_PARENT_RUNNING')
        lease_end=datetime.fromisoformat(f['EndTime']).timestamp();need(lease_end>=epoch+86400,'ORIGINAL86400_PARENT_LEASE');parent_admission=dict(fields=f,lease_end_epoch=lease_end,observed_epoch=time.time())
        need(row_plan['node'] in query(['scontrol','show','hostnames',f['NodeList']]),'FIXED_NODE_MEMBERSHIP');need(accounting() is None,'ONCE_LABEL_ALREADY_USED')
        for value in query(['scontrol','show','step',row_plan['parent'],'--oneliner']):
            step=dict(v.split('=',1) for v in value.split() if '=' in v);suffix=step.get('StepId','').rsplit('.',1)[-1]
            if suffix not in ('batch','extern'):need('NodeList'in step and row_plan['node'] not in query(['scontrol','show','hostnames',step['NodeList']]),'FIXED_NODE_BUSY_OR_UNKNOWN')
        need(time.time()<epoch+180,'ADMISSION_WITHIN180')
        need(verify_marker(context)['value']['context']=={k:context[k] for k in ['publication','request_id','epoch','deadline','source_manifest_SHA256','namespace_nonce']},'ORIGINAL_COMMON_MARKER')
        seconds=int((epoch+P['budget']['step_end_max']-time.time()-30)//60)*60
        need(seconds>=84600,'INSUFFICIENT_ORIGINAL_STEP_BUDGET_NO_SUBMIT')
        step_end=time.time()+seconds;need(step_end<=epoch+P['budget']['step_end_max'],'STEP_END_WITHIN_ORIGINAL85020')
        hours,rem=divmod(seconds,3600);minutes,seconds_tail=divmod(rem,60);time_arg=f'{hours:02d}:{minutes:02d}:{seconds_tail:02d}'
        node_context=dict(context,step_end=step_end,payload_deadline=min(epoch+P['budget']['payload_stop'],step_end-180));save(root/'node-context.json',node_context)
        argv=['srun','--jobid='+row_plan['parent'],'--job-name='+row_plan['label'],'--exact','--immediate=5','--cpu-bind=cores','-N1','-n1','-w',row_plan['node'],'-c4','--mem=4096M','--gres=none','--time='+time_arg,'/usr/local/python3.12/bin/python3','-B',str(D/'node_entry.py')]
        save(root/'step-intent.json',dict(context,command=argv,Slurm_seconds=seconds,step_end=step_end,payload_deadline=node_context['payload_deadline'],owner=owner))
        with (root/'srun.stderr.private').open('xb') as err:
            step_attempted=True;save(root/'control-state.json',dict(context,exec_request_attempted=True),replace=True)
            child=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,start_new_session=True)
            anchor=dict(PID=child.pid,PGID=child.pid,start_new_session=True,supervisor=owner);save(root/'held-srun.json',anchor)
            ready=receive_line(child.stdout.fileno());need(ready['stage']=='READY' and ready['mode']=='receiver' and ready['owner']['host']==row_plan['node'] and ready['owner']['UID']==154459 and ready['allocated_CPU']==4 and ready['worker_CPU']==2 and len(ready['CPU_affinity'])==2,'ACTUAL_COMPUTE_READY')
            need(all(ready[k]==context[k] for k in ('request_id','publication','epoch','deadline','source_manifest_SHA256','namespace_nonce','file_id')),'COMPUTE_SAME_ORIGINAL_CONTEXT')
            need(re.fullmatch(re.escape(row_plan['parent'])+r'\.\d+',ready.get('step','')),'EXACT_OBSERVED_COMPUTE_STEP')
            observed_compute_ready=ready;save(root/'compute-ready-observed.json',ready)
            actual_fields=observe_actual_step(root,context,row_plan,ready['step'],step_queries)
            need(actual_fields['StepId']==ready['step'] and actual_fields['NodeList']==row_plan['node'],'EXACT_FRESH_STEP_NODE')
            tm=re.fullmatch(r'(?:(\d+)-)?(\d+):(\d{2}):(\d{2})',actual_fields['TimeLimit']);need(tm is not None,'FINITE_ACTUAL_SLURM_HH_MM_SS')
            days,hours_actual,minutes_actual,seconds_actual=[int(v or 0) for v in tm.groups()];actual_limit=days*86400+hours_actual*3600+minutes_actual*60+seconds_actual
            actual_start=datetime.fromisoformat(actual_fields['StartTime']).timestamp();actual_end=actual_start+actual_limit
            need(actual_limit==seconds and epoch-1<=actual_start<=time.time()+1 and actual_end<=epoch+85020,'ACTUAL_STEP_END_WITHIN_ORIGINAL85020')
            admission=dict(parent=parent_admission,step_fields=actual_fields,requested_seconds=seconds,time_arg=time_arg,actual_start_epoch=actual_start,actual_limit_seconds=actual_limit,actual_end_epoch=actual_end,observed_epoch=time.time())
            save(root/'compute-ready.json',ready);control(dict(ready,target_host_owner=owner,allocated_CPU=4,worker_CPU=2,allocated_memory_MiB=4096,GRES=0,fresh_admission=admission))
            def forward():
                try:
                    need(line_fd(0)==b'GO\n','ROOT_RELEASE')
                    need(os.write(child.stdin.fileno(),b'GO\n')==3,'EXACT_GO_TO_COMPUTE');pump['control_read_bytes']=pump['control_write_bytes']=3
                    def rd(f,n):
                        mark=time.monotonic();b=os.read(f,min(n,P['chunk_bytes']));pump['read_blocked_seconds']+=time.monotonic()-mark;pump['read_calls']+=1;pump['read_bytes']+=len(b);return b
                    def put(b):
                        n=0
                        while n<len(b):
                            mark=time.monotonic();v=os.write(child.stdin.fileno(),memoryview(b)[n:n+P['chunk_bytes']]);pump['write_blocked_seconds']+=time.monotonic()-mark;pump['write_calls']+=1;need(v>0,'COMPUTE_STDIN_SHORT_WRITE');n+=v;pump['write_bytes']+=v
                    for index in range(row_plan['blocks']):
                        header=exact_fd(0,128,rd);meta=unpack(header,row_plan,index);put(header);remaining=meta['compressed_bytes']
                        while remaining:
                            b=rd(0,min(P['chunk_bytes'],remaining));need(b,'ROOT_COMPRESSED_BLOCK_EOF');put(b);remaining-=len(b)
                        pump['blocks_completed']+=1
                    need(not os.read(0,1),'ROOT_TRAILING_BYTES');pump['EOF']=True;pump['completed']=True
                except BaseException as e:pump['error']=dict(type=type(e).__name__,reason_SHA256=__import__('hashlib').sha256(str(e).encode()).hexdigest())
                finally:
                    try:child.stdin.close()
                    except OSError:pass
            pump_thread=threading.Thread(target=forward,daemon=True);pump_thread.start()
            terminal_queue=queue.Queue()
            def drain_terminal():
                try:
                    raw=read_control_bytes(child.stdout.fileno());persist_bytes(root,'compute-control-terminal.private',raw,65537,context)
                    need(0<len(raw)<=65536 and raw.endswith(b'\n'),'COMPUTE_TERMINAL_CONTROL_LINE')
                    terminal_queue.put(('ok',__import__('json').loads(raw)))
                except BaseException as e:terminal_queue.put(('error',type(e).__name__))
            terminal_thread=threading.Thread(target=drain_terminal,daemon=True);terminal_thread.start();reads.append(terminal_thread)
            last_progress_time=time.monotonic();last_progress_bytes=0
            while child.poll() is None:
                need((root/'srun.stderr.private').stat().st_size<=262144,'BOUNDED_SRUN_LOG');left(40)
                progress_bytes_now=pump['write_bytes']
                if time.monotonic()-last_progress_time>=P['progress_seconds'] or progress_bytes_now-last_progress_bytes>=P['progress_bytes']:
                    mark=time.monotonic()
                    try:save(root/'host-pump-progress.json',dict(context,observed_epoch=time.time(),pump=pump,host_phase_costs=host_phase_costs,login_payload_file_bytes=0),replace=True)
                    finally:
                        host_phase_costs['progress_receipt_persist']['seconds']+=time.monotonic()-mark
                        host_phase_costs['progress_receipt_persist']['calls']+=1
                    last_progress_time=time.monotonic();last_progress_bytes=progress_bytes_now
                try:child.wait(timeout=min(.5,left(40)))
                except subprocess.TimeoutExpired:pass
            kind,endpoint=terminal_queue.get(timeout=min(5,left(35)));need(kind=='ok','COMPUTE_TERMINAL_CONTROL');save(root/'receiver-terminal.json',endpoint)
    except BaseException as e:failure=dict(type=type(e).__name__,reason_SHA256=__import__('hashlib').sha256(str(e).encode()).hexdigest())
    finally:
        for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,signal.SIG_IGN)
        signal.signal(signal.SIGALRM,lambda *_:os._exit(124));signal.setitimer(signal.ITIMER_REAL,max(.001,epoch+85920-time.time()))
        try:held=cleanup_held(child,anchor,epoch+P['budget']['held_cleanup'])
        except BaseException as e:held=dict(reaped=False,error_type=type(e).__name__)
        if pump_thread is not None:pump_thread.join(timeout=max(0,min(2,epoch+P['budget']['held_cleanup']-time.time())))
    while step_attempted and time.time()<epoch+P['budget']['producer_accounting_stop']:
        try:row=accounting()
        except Exception as e:
            if failure is None:failure=dict(type=type(e).__name__,reason='ACCOUNTING_QUERY_FAILED')
            break
        if row and row[2].split()[0] in TERMINAL:break
        time.sleep(min(.5,max(0,epoch+P['budget']['producer_accounting_stop']-time.time())))
    terminal=bool(row and row[2].split()[0] in TERMINAL)
    passed=bool(failure is None and child and child.returncode==0 and held and held.get('reaped') and not held.get('termination_needed') and endpoint and endpoint['passed'] is True and terminal and row[2:4]==['COMPLETED','0:0'] and row[5]==row_plan['node'] and set(row[6].split(','))=={'cpu=4','mem=4G','node=1'} and endpoint['step']==row[0] and pump['completed'] and pump['error'] is None and pump['read_bytes']==pump['write_bytes']==endpoint['wire_processed_bytes'] and pump['blocks_completed']==row_plan['blocks'] and not (pump_thread and pump_thread.is_alive()))
    result=dict(context,stage='TARGET_TERMINAL',passed=passed,endpoint=endpoint,pump=pump,host_phase_costs=host_phase_costs,observed_compute_ready=observed_compute_ready,step_visibility_queries=step_queries,metadata_persist_semantics='Progress save cost includes JSON/write/fsync/rename, distinct from payload forwarding; final pump counters retained in this outcome',host_owner=owner,held_srun=held,accounting_columns=COLS,accounting=row,failure=failure,execution_state='TERMINAL' if terminal else ('NOT_SUBMITTED' if not step_attempted else 'UNKNOWN'),old14_accesses=0,old_partial_accesses=0,hardlinks=0,login_payload_file_bytes=0,GPU_GO=False,Model_GO=False,training_GO=False,replay_allowed=False,finished_epoch=time.time())
    save(root/'outcome.json',result);control(result);signal.setitimer(signal.ITIMER_REAL,0);return 0 if passed else 1
