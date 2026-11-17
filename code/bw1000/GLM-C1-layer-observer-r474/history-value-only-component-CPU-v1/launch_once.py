"""Once published CPU-only A/B build; source-only until ROOT/CPU/compile GO adopted."""
import argparse,base64,hashlib,json,math,re,runpy,subprocess,time,uuid
from pathlib import Path
D=Path(__file__).resolve().parent;G=D.parents[3]
L=G/'local-only/GLM474-history-value-only-component-CPU-v1-original'
HELPER=Path('C:/Users/WangHaojie/.codex/visualizations/2026/10/12/01a130bd-3b97-73c2-8ecf-eda5151f1ba7/glm-paired-readonly-connection.py')
HELPER_SHA='3f00e0eac9b40eb3fee7d788776389bc22c3e87d20de0441312f04a2566178d1'
H=lambda b:hashlib.sha256(b).hexdigest()

def need(ok,code):
    if not ok:raise ValueError(code)

def save(p,v):
    with p.open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False)

def classify_execution(returncode,documents):
    if returncode==255 or len(documents)!=1:return 'UNKNOWN'
    value=documents[0]
    if value.get('current_step_outcome')=='NOT_SUBMITTED' and value.get('exec_request_attempted') is False and value.get('control_checkpoint_verified') is True and re.fullmatch('[0-9a-f]{64}',value.get('control_state_SHA256') or ''):
        return 'NOT_SUBMITTED'
    if value.get('current_step_outcome')=='TERMINAL' and value.get('exec_request_attempted') is True and value.get('control_checkpoint_verified') is True:
        return 'RETURNED'
    return 'UNKNOWN'


def main(a):
    epoch=time.time();anchor=time.monotonic();deadline=epoch+300
    def left(reserve=0):
        n=min(deadline-time.time(),300-(time.monotonic()-anchor))-reserve
        need(math.isfinite(n) and n>0,'ORIGINAL_CLIENT_EPOCH_EXHAUSTED');return n
    need(re.fullmatch('[0-9a-f]{40}',a.publication) and all(re.fullmatch('[0-9a-f]{64}',v) for v in (a.source_manifest_sha256,a.claim_sha256)),'PUBLICATION_PINS')
    need(not L.exists(),'ONCE_NAMESPACE_EXISTS');L.mkdir(mode=0o700)
    request=uuid.uuid4().hex;q=dict(request_id=request,publication=a.publication,epoch=epoch,deadline=deadline,attempted=False,state='NOT_ATTEMPTED',replay_allowed=False,GPU_GO=False,Model_GO=False)
    save(L/'intent.json',dict(q,source_manifest_SHA256=a.source_manifest_sha256,claim_SHA256=a.claim_sha256))
    stdout=stderr=b''
    try:
        from verify_package import verify
        verify(D);git=['git','--no-optional-locks','-c','safe.directory='+G.as_posix(),'-C',str(G)]
        def show(path):return subprocess.check_output(git+['show',a.publication+':'+path],timeout=min(15,left(60)),stderr=subprocess.DEVNULL)
        rel=D.relative_to(G).as_posix();mfraw=(D/'SOURCE_FILES.json').read_bytes();need(H(mfraw)==a.source_manifest_sha256 and show(rel+'/SOURCE_FILES.json')==mfraw,'PUBLISHED_MF')
        mf=json.loads(mfraw);blobs={'SOURCE_FILES.json':mfraw}
        for name,p in mf.items():
            b=(D/name).read_bytes();need(H(b)==p['SHA256'] and len(b)==p['bytes'] and show(rel+'/'+name)==b,'PUBLISHED_MEMBER');blobs[name]=b
        need(not subprocess.check_output(git+['status','--porcelain','--untracked-files=all','--',rel],timeout=min(15,left(60))),'OWN_SOURCE_DIRTY')
        c=json.loads(blobs['CLAIM.json']);need(H(blobs['CLAIM.json'])==a.claim_sha256 and c['ROOT_GO'] and c['CPU_GO'] and c['Native_compile_GO'] and not c['GPU_GO'] and not c['Model_GO'],'PUBLISHED_CPU_CLAIM')
        need(H(HELPER.read_bytes())==HELPER_SHA,'CONNECTION_HELPER_PIN');ssh=runpy.run_path(str(HELPER))['ssh']
        files={n:dict(bytes=len(b),SHA256=H(b),b64=base64.b64encode(b).decode()) for n,b in blobs.items()}
        payload=('EPOCH='+repr(epoch)+'\nREQUEST='+repr(request)+'\nPUBLICATION='+repr(a.publication)+'\nSOURCE_SHA='+repr(a.source_manifest_sha256)+'\nCLAIM_SHA='+repr(a.claim_sha256)+'\nFILES='+repr(files)+'\n').encode()+blobs['receive_once.py']
        need(len(payload)<=1024*1024,'SOURCE_TRANSPORT_CAP');save(L/'exec-intent.json',dict(q,attempted=True,payload_bytes=len(payload),payload_SHA256=H(payload)));q['attempted']=True
        argv=[ssh[0],'-o','StrictHostKeyChecking=yes',*ssh[1:],'/usr/local/python3.12/bin/python3 -I -S -B -u -']
        done=subprocess.run(argv,input=payload,capture_output=True,timeout=left(5));stdout,stderr=done.stdout,done.stderr
        need(len(stdout)<=131072 and len(stderr)<=65536,'RETURNED_OUTPUT_CAP');docs=[]
        for line in stdout.splitlines():
            try:v=json.loads(line)
            except (ValueError,UnicodeError):continue
            if isinstance(v,dict) and v.get('request_id')==request and 'passed' in v:docs.append(v)
        q.update(rc=done.returncode,state=classify_execution(done.returncode,docs))
        if len(docs)==1:q['summary']=docs[0]
    except subprocess.TimeoutExpired as e:stdout,stderr=e.stdout or b'',e.stderr or b'';q.update(state='UNKNOWN',error_type='TimeoutExpired')
    except BaseException as e:q.update(error_type=type(e).__name__,state='UNKNOWN' if q['attempted'] else 'NOT_ATTEMPTED')
    (L/'stdout.private').write_bytes(stdout);(L/'stderr.private').write_bytes(stderr)
    q.update(stdout_bytes=len(stdout),stdout_SHA256=H(stdout),stderr_bytes=len(stderr),stderr_SHA256=H(stderr),elapsed_s=time.monotonic()-anchor);save(L/'receipt.json',q);print(json.dumps(q,allow_nan=False))
    return 0 if q.get('summary',{}).get('passed') and q['state']=='RETURNED' else 2

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--publication',required=True);p.add_argument('--source-manifest-sha256',required=True);p.add_argument('--claim-sha256',required=True);raise SystemExit(main(p.parse_args()))
