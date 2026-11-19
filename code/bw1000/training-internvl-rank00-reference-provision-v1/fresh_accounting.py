"""Exactly one fresh ordinary Hygon SSH/sacct query in the original900s epoch."""
import hashlib,json,math,re,runpy,subprocess,time
from pathlib import Path
HELPER=Path('C:/Users/WangHaojie/.codex/visualizations/2026/10/12/01a130bd-3b97-73c2-8ecf-eda5151f1ba7/glm-paired-readonly-connection.py')
HELPER_SHA='3f00e0eac9b40eb3fee7d788776389bc22c3e87d20de0441312f04a2566178d1'
COLS='JobIDRaw,JobName%100,State%50,ExitCode,ElapsedRaw,NodeList%500,AllocTRES%200'
REMOTE=r"""
import hashlib,json,math,os,signal,socket,subprocess,sys,time
record=dict(context=CONTEXT,query_calls=0,passed=False,columns=COLUMNS,requested_step=STEP,requested_label=LABEL,remote_writes=0,asset_body_reads=0,old_receipt_reads=0,Slurm_submissions=0,observed_begin_epoch=time.time())
try:
    remaining=READ_DEADLINE-time.time()
    if not (math.isfinite(remaining) and 0<remaining<=30 and CONTEXT['deadline']==CONTEXT['epoch']+900 and READ_DEADLINE<=CONTEXT['epoch']+892):raise ValueError('EXACT_ORIGINAL_CONSUMER_DEADLINE')
    def stop(*_):raise TimeoutError('FRESH_ACCOUNTING_DEADLINE')
    signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,remaining)
    cores=sorted(set(os.sched_getaffinity(0))-{0,1})
    if len(cores)<2:raise ValueError('TWO_READ_CPU')
    os.sched_setaffinity(0,cores[:2]);nice=os.getpriority(os.PRIO_PROCESS,0)
    if nice<10:os.nice(10-nice)
    fields=open('/proc/self/stat').read().rsplit(') ',1)[1].split()
    record['owner']=dict(PID=os.getpid(),UID=os.getuid(),startticks=int(fields[19]),host=socket.gethostname(),boot_id=open('/proc/sys/kernel/random/boot_id').read().strip(),CPU_affinity=cores[:2])
    if os.getuid()!=154459:raise ValueError('EXPECTED_HYGON_UID')
    record['query_calls']=1
    q=subprocess.run(['sacct','-n','-P','-j',STEP,'-o',COLUMNS],stdin=subprocess.DEVNULL,capture_output=True,timeout=min(8,max(.001,READ_DEADLINE-time.time()-1)))
    record.update(query_returncode=q.returncode,stdout_bytes=len(q.stdout),stderr_bytes=len(q.stderr),stdout_SHA256=hashlib.sha256(q.stdout).hexdigest(),stderr_SHA256=hashlib.sha256(q.stderr).hexdigest())
    if q.returncode!=0 or len(q.stdout)+len(q.stderr)>1048576:raise ValueError('BOUNDED_FRESH_SACCT')
    lines=q.stdout.decode('utf8').strip().splitlines();rows=[v for line in lines if len(v:=line.split('|'))==7 and v[0]==STEP and v[1]==LABEL]
    if len(rows)!=1:raise ValueError('ONE_EXACT_FRESH_STEP_LABEL')
    record.update(passed=True,actual_row=rows[0],actual_rows_observed=len(lines))
except BaseException as e:record['error']=dict(type=type(e).__name__,reason_SHA256=hashlib.sha256(str(e).encode()).hexdigest())
record['observed_end_epoch']=time.time();signal.setitimer(signal.ITIMER_REAL,0)
print(json.dumps(record,sort_keys=True,allow_nan=False),flush=True)
sys.exit(0 if record['passed'] else 1)
"""

def need(v,w):
    if not v:raise ValueError(w)

def sha(b):return hashlib.sha256(b).hexdigest()

def save(path,value):
    with path.open('x',encoding='utf8') as f:json.dump(value,f,sort_keys=True,allow_nan=False);f.write('\n')

def query_once(run,context,plan,step):
    begin=time.time();anchor=time.monotonic();read_end=min(begin+25,context['epoch']+892);cleanup_end=min(begin+29,context['epoch']+894)
    child=None;stdout=stderr=b'';failure=None;timed_out=False;forced=False;reaped=False;record=None
    result=dict(passed=False,actual_ordinary_SSH_attempts=0,actual_sacct_calls=None,remote_state='NOT_STARTED',remote_writes=0,asset_body_reads=0,old_receipt_reads=0,Slurm_submissions=0,context=context,columns=COLS,step=step,label=plan['label'])
    def remaining(end):return min(end-time.time(),anchor+end-begin-time.monotonic())
    try:
        need(math.isfinite(begin) and context['deadline']==context['epoch']+900 and remaining(read_end)>=10,'INSUFFICIENT_ORIGINAL_CONSUMER_TIME_NO_QUERY')
        need(re.fullmatch(re.escape(plan['target_parent'])+r'\.\d+',step),'EXACT_ORIGINAL_STEP')
        need(not (run/'fresh-accounting-attempt.json').exists(),'FRESH_QUERY_ALREADY_ATTEMPTED_NO_REPLAY')
        raw=HELPER.read_bytes();need(sha(raw)==HELPER_SHA,'QUALIFIED_READONLY_SSH_PROVIDER')
        base=runpy.run_path(str(HELPER))['ssh'];argv=[base[0],'-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=5',*base[1:],'/usr/local/python3.12/bin/python3 -I -S -B -']
        need(len(subprocess.list2cmdline(argv))<30000,'WINDOWS_CREATEPROCESS_BOUND')
        prefix='CONTEXT='+repr(context)+'\nREAD_DEADLINE='+repr(read_end)+'\nSTEP='+repr(step)+'\nLABEL='+repr(plan['label'])+'\nCOLUMNS='+repr(COLS)+'\n'
        body=(prefix+REMOTE).encode();need(len(body)<=16384 and remaining(read_end)>=8,'BOUNDED_FRESH_READ_BODY_AND_TIME')
        save(run/'fresh-accounting-attempt.json',dict(context,step=step,label=plan['label'],columns=COLS,SSH_Popen_about_to_call=True,read_deadline=read_end,cleanup_deadline=cleanup_end,body_bytes=len(body),body_SHA256=sha(body),max_SSH_attempts=1,max_sacct_calls=1,remote_outcome='UNKNOWN',replay_allowed=False))
        child=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE);result['actual_ordinary_SSH_attempts']=1
        stdout,stderr=child.communicate(input=body,timeout=remaining(read_end))
    except subprocess.TimeoutExpired as e:timed_out=True;failure=dict(type=type(e).__name__);stdout=e.output or b'';stderr=e.stderr or b''
    except BaseException as e:failure=dict(type=type(e).__name__,reason_SHA256=sha(str(e).encode()))
    finally:
        if child is not None:
            if child.poll() is None:
                forced=True
                try:child.kill()
                except ProcessLookupError:pass
            left=remaining(cleanup_end)
            if left>0:
                try:stdout,stderr=child.communicate(timeout=min(3,left))
                except subprocess.TimeoutExpired as e:stdout=e.output or stdout;stderr=e.stderr or stderr
                except BaseException as e:
                    if failure is None:failure=dict(type=type(e).__name__)
            reaped=child.poll() is not None
    if len(stdout)+len(stderr)<=1048576:
        with (run/'fresh-accounting.stdout.private').open('xb') as f:f.write(stdout)
        with (run/'fresh-accounting.stderr.private').open('xb') as f:f.write(stderr)
    else:failure=dict(type='FreshAccountingLogBound')
    if child is not None:
        result['remote_state']='RETURNED' if reaped and not forced and not timed_out and child.returncode not in(None,255) else 'UNKNOWN'
    if result['remote_state']=='RETURNED' and len(stdout)<=262144:
        try:
            record=json.loads(stdout);need(record['context']==context and record['columns']==COLS and record['requested_step']==step and record['requested_label']==plan['label'],'FRESH_QUERY_CONTEXT')
            need(record['observed_begin_epoch']>=context['epoch'] and record['observed_end_epoch']<=read_end and record['query_calls'] in(0,1),'ACTUAL_FRESH_QUERY_BOUNDARY')
            result['sacct_invocation_attempts']=record['query_calls'];result['actual_sacct_calls']=record['query_calls'] if 'query_returncode' in record else None
        except (ValueError,KeyError,TypeError) as e:
            if failure is None:failure=dict(type=type(e).__name__,reason_SHA256=sha(str(e).encode()))
    result.update(passed=bool(failure is None and child and child.returncode==0 and record and record.get('passed') is True and record.get('query_calls')==1 and result['remote_state']=='RETURNED'),record=record,failure=failure,transport=dict(returncode=child.returncode if child else None,timed_out=timed_out,forced_local_SSH_kill=forced,held_SSH_reaped=reaped),read_deadline=read_end,cleanup_deadline=cleanup_end,finished_epoch=time.time(),elapsed_seconds=time.monotonic()-anchor,stdout_pin=dict(bytes=len(stdout),SHA256=sha(stdout)),stderr_pin=dict(bytes=len(stderr),SHA256=sha(stderr)),replay_allowed=False)
    save(run/'fresh-accounting-result.json',result)
    return result
