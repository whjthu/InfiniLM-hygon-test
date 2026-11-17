"""Injected constants only; one owned installation and original terminal read."""
import base64,hashlib,json,math,os,resource,signal,socket,stat,subprocess,time
from pathlib import Path
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
W=R/'results/GLM474-history-value-only-common-AB-CPU-v1'
O=R/'results/GLM474-history-value-only-common-AB-provider-remainder-v1'
TARGET='/refactor/results/GLM474-history-value-only-common-AB-provider-remainder-v1'
H=lambda b:hashlib.sha256(b).hexdigest()
anchor=time.monotonic();initial=READER_EPOCH+180-time.time();child=None;owned=False
summary=dict(audit_request_id=AUDIT_REQUEST,original_request_id=ORIGINAL_REQUEST,passed=False,state='UNKNOWN',replay_allowed=False)
def need(ok,code):
    if not ok:raise ValueError(code)
def left():
    n=min(READER_EPOCH+180-time.time(),initial-(time.monotonic()-anchor));need(math.isfinite(n) and n>2,'ORIGIN_EXHAUSTED');return n
def alarm(*unused):raise TimeoutError('ORIGIN_EXHAUSTED')
def save(p,q):
    with p.open('x',encoding='utf8') as f:json.dump(q,f,allow_nan=False);f.write('\n')
def stable(p,cap):
    left();s=p.stat();need(stat.S_ISREG(s.st_mode) and s.st_size<=cap,'SMALL_FILE_BOUND')
    with p.open('rb') as f:
        a=os.fstat(f.fileno());b=f.read(cap+1);z=os.fstat(f.fileno())
    ident=lambda x:(x.st_dev,x.st_ino,x.st_size,x.st_mtime_ns,x.st_ctime_ns)
    need(len(b)<=cap and ident(s)==ident(a)==ident(z)==ident(p.stat()),'SMALL_FILE_CHANGED');return b
def reap():
    if child is not None:
        if child.poll() is None:
            need(os.getpgid(child.pid)==child.pid,'OWN_CONTAINER_SESSION')
            os.killpg(child.pid,signal.SIGKILL)
        child.wait(timeout=3)
try:
    need(0<initial<=180,'ORIGIN_INVALID');signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,initial)
    resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3));cores=sorted(os.sched_getaffinity(0));need(cores,'CPU_AFFINITY');os.sched_setaffinity(0,set(cores[:4]))
    # Read-only original admission before installation or a container invocation.
    original=stable(W/'outcome.json',16*1024**2);need(H(original)==OUTCOME_SHA,'ORIGINAL_OUTCOME_PIN');out=json.loads(original)
    controlraw=stable(W/'control-state.json',65536);control=json.loads(controlraw)
    need(H(controlraw)==out.get('control_state_SHA256'),'ORIGINAL_CONTROL_PIN')
    need(control.get('request_id')==ORIGINAL_REQUEST and control.get('publication')==ORIGINAL_PUBLICATION and control.get('epoch')==ORIGINAL_EPOCH and control.get('source_manifest_SHA256')=='47aada58ed7e9c9068d5a5c5694a4f32512207618aeb4f71c54966691657eb14' and control.get('claim_SHA256')=='5cf8581d5046fdcf50388042b77d2d5221c101e1cefb72cd0d551f10428fb9a4','ORIGINAL_CONTEXT')
    row=out.get('accounting',{}).get('row')
    terminal=bool(row and len(row)==7 and row[0].startswith('1103606.') and row[1]=='GLM474-history-value-only-common-AB-CPU-v1' and row[2].split()[0] in {'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'})
    notsubmitted=control.get('exec_request_attempted') is False and out.get('current_step_outcome')=='NOT_SUBMITTED' and out.get('control_checkpoint_verified') is True
    need(terminal or notsubmitted,'ORIGINAL_NOT_TERMINAL')
    need(not O.exists(),'ONCE_NAMESPACE_EXISTS');os.umask(0o077);O.mkdir();owned=True;(O/'src').mkdir()
    context=dict(audit_request_id=AUDIT_REQUEST,reader_epoch=READER_EPOCH,reader_deadline=READER_EPOCH+180,reader_publication=PUBLICATION,source_manifest_SHA256=SOURCE_SHA,claim_SHA256=CLAIM_SHA,original_request_id=ORIGINAL_REQUEST,original_publication=ORIGINAL_PUBLICATION,original_epoch=ORIGINAL_EPOCH,original_outcome_SHA256=OUTCOME_SHA)
    save(O/'install-intent.json',context)
    total=0
    for name,pin in FILES.items():
        left();p=Path(name);need(len(p.parts)==1 and not p.is_absolute(),'SOURCE_MEMBER')
        b=base64.b64decode(pin['b64'],validate=True);total+=len(b);need(total<=256*1024 and len(b)==pin['bytes'] and H(b)==pin['SHA256'],'SOURCE_PIN')
        with (O/'src'/name).open('xb') as f:f.write(b)
    mfraw=stable(O/'src/SOURCE_FILES.json',65536);need(H(mfraw)==SOURCE_SHA,'MF_PIN');mf=json.loads(mfraw)
    need(set(FILES)==set(mf)|{'SOURCE_FILES.json'},'EXACT_MEMBERS')
    for name,pin in mf.items():
        b=stable(O/'src'/name,256*1024);need(H(b)==pin['SHA256'] and len(b)==pin['bytes'],'MEMBER_PIN')
    c=stable(O/'src/CLAIM.json',65536);need(H(c)==CLAIM_SHA,'CLAIM_PIN');claim=json.loads(c)
    need(claim['ROOT_GO'] and claim['CPU_read_GO'] and claim['maximum_invocations']==1 and not claim['GPU_GO'] and not claim['Native_compile_GO'] and OUTCOME_SHA==claim['original_outcome_SHA256'],'CLAIM_NOT_APPROVED')
    argv=['/public/software/apps/apptainer/1.3.4/bin/apptainer','exec','--rocm','--cleanenv','--no-home','--pwd','/',
      '--bind',str(R)+':/refactor:ro','--bind',str(O)+':'+TARGET+':rw','--bind','/opt/hyhal:/opt/hyhal:ro',
      '--env','CUDA_VISIBLE_DEVICES=-1,HIP_VISIBLE_DEVICES=-1,ROCR_VISIBLE_DEVICES=-1',str(R/'containers/dtk26-sandbox-r1'),
      '/usr/bin/python3','-I','-S','-B','-u',TARGET+'/src/read_original.py','--request-id',ORIGINAL_REQUEST,'--publication',ORIGINAL_PUBLICATION,
      '--epoch',str(ORIGINAL_EPOCH),'--reader-epoch',str(READER_EPOCH),'--reader-manifest-sha256',SOURCE_SHA,'--outcome-sha256',OUTCOME_SHA]
    save(O/'exec-intent.json',dict(context,argv=argv,CPU_affinity=sorted(os.sched_getaffinity(0)),address_space_hard_limit_bytes=3*1024**3,apptainer_setup_implicit_SDK_driver='UNKNOWN'))
    with (O/'container.stdout.private').open('xb') as stdout,(O/'container.stderr.private').open('xb') as stderr:
        try:
            left();child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True)
            try:
                row=Path('/proc',str(child.pid),'stat').read_text().rsplit(') ',1)[1].split()
                save(O/'owned-container.json',dict(PID=child.pid,PGID=int(row[2]),startticks=int(row[19]),host=socket.gethostname(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip()))
            except FileNotFoundError:save(O/'owned-container.json',dict(PID=child.pid,exited_before_identity_observation=True))
            rc=child.wait(timeout=max(1,left()-5))
        finally:reap()
    report=stable(O/'report-private.json',262144);value=json.loads(report)
    need(value['request_id']==ORIGINAL_REQUEST and value['publication']==ORIGINAL_PUBLICATION and value['original_epoch']==ORIGINAL_EPOCH,'REPORT_CONTEXT')
    logs={}
    for name in ('container.stdout.private','container.stderr.private'):
        b=stable(O/name,128*1024);logs[name]=dict(bytes=len(b),SHA256=H(b))
    summary.update(state='RETURNED',rc=rc,passed=value['passed'],execution_state=value['execution_state'],report_bytes=len(report),report_SHA256=H(report),report_b64=base64.b64encode(report).decode(),logs=logs,owned_child_reaped=child.returncode is not None)
except BaseException as error:
    summary.update(error_type=type(error).__name__,error_code=str(error)[:120] if isinstance(error,ValueError) else 'TRANSPORT_OR_BUDGET_FAILURE')
finally:
    signal.setitimer(signal.ITIMER_REAL,0)
    try:reap()
    except BaseException as error:summary['cleanup_error_type']=type(error).__name__
    small={k:v for k,v in summary.items() if k!='report_b64'}
    if owned:save(O/'transport-outcome.json',small)
    print(json.dumps(summary,allow_nan=False),flush=True)
