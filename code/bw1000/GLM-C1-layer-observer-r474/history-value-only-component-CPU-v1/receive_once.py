"""Futureonce bootstrap. False means an original bound controlcheckpoint exists."""
import base64,hashlib,json,math,os,sys,time
from pathlib import Path
need=lambda ok:None if ok else (_ for _ in ()).throw(ValueError('BOOTSTRAP_SCOPE'))
W=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/GLM474-history-value-only-component-CPU-v1')


def verified_control(path,expected):
    try:
        raw=Path(path).read_bytes();value=json.loads(raw)
        if not all(value.get(k)==expected[k] for k in ('request_id','publication','epoch','source_manifest_SHA256','claim_SHA256')):return None,None
        if type(value.get('exec_request_attempted')) is not bool:return None,None
        return value,hashlib.sha256(raw).hexdigest()
    except (OSError,ValueError,TypeError,KeyError):return None,None


owned=False;rc=1
expected=dict(request_id=REQUEST,publication=PUBLICATION,epoch=EPOCH,deadline=EPOCH+300,source_manifest_SHA256=SOURCE_SHA,claim_SHA256=CLAIM_SHA)
try:
    need(math.isfinite(EPOCH) and 0<EPOCH<=time.time()<EPOCH+300)
    os.umask(0o077);need(not W.exists());W.mkdir(mode=0o700);owned=True
    control=dict(expected,exec_request_attempted=False,phase='SOURCE_INSTALL_NOT_SUBMITTED',updated_epoch=time.time())
    tmp=W/'control-state.initial.json'
    with tmp.open('x',encoding='utf8') as f:
        json.dump(control,f,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,W/'control-state.json')
    directory=os.open(str(W),os.O_RDONLY)
    try:os.fsync(directory)
    finally:os.close(directory)
    D=W/'src';D.mkdir()
    with (W/'install-intent.json').open('x') as f:json.dump(expected,f)
    total=0
    for name,row in FILES.items():
        need(time.time()<EPOCH+300);p=Path(name);need(not p.is_absolute() and '..' not in p.parts)
        raw=base64.b64decode(row['b64'],validate=True);total+=len(raw)
        need(total<=512*1024 and len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['SHA256'])
        dest=D/name;dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('xb') as f:f.write(raw)
    need(hashlib.sha256((D/'SOURCE_FILES.json').read_bytes()).hexdigest()==SOURCE_SHA and hashlib.sha256((D/'CLAIM.json').read_bytes()).hexdigest()==CLAIM_SHA)
    sys.path.insert(0,str(D));from verify_package import verify
    verify(D);c=json.loads((D/'CLAIM.json').read_bytes());need(c['ROOT_GO'] and c['CPU_GO'] and c['Native_compile_GO'] and not c['GPU_GO'])
    namespace={'__name__':'_owned_cpu_dispatch','__file__':str(D/'dispatch_once.py')}
    exec(compile((D/'dispatch_once.py').read_bytes(),str(D/'dispatch_once.py'),'exec'),namespace)
    rc=namespace['main'](PUBLICATION,CLAIM_SHA)
except BaseException as error:
    control,control_sha=verified_control(W/'control-state.json',expected) if owned else (None,None)
    attempted=control['exec_request_attempted'] if control is not None else None
    state='NOT_SUBMITTED' if attempted is False else 'UNKNOWN'
    failure=dict(request_id=REQUEST,publication=PUBLICATION,epoch=EPOCH,passed=False,error_type=type(error).__name__,error_reason=str(error)[:240],error_message_SHA256=hashlib.sha256(str(error)[:4096].encode()).hexdigest(),exec_request_attempted=attempted,control_checkpoint_verified=control is not None,control_state_SHA256=control_sha,current_step_outcome=state,replay_allowed=False,GPU_GO=False,Model_GO=False)
    if owned:
        try:
            name='outcome.json' if not (W/'outcome.json').exists() else 'bootstrap-failure.json'
            with (W/name).open('x',encoding='utf8') as f:json.dump(failure,f,indent=2,allow_nan=False);f.write('\n')
            failure['failure_receipt_SHA256']=hashlib.sha256((W/name).read_bytes()).hexdigest()
        except OSError:failure['failure_receipt_write_failed']=True
    print(json.dumps(failure,allow_nan=False),flush=True)
raise SystemExit(rc)
