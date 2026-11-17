"""Single published ordinary-SSH read-only reconciliation; no original replay."""
import argparse,hashlib,json,math,re,runpy,subprocess,time,uuid
from pathlib import Path
D=Path(__file__).resolve().parent;G=D.parents[3];L=G/'local-only/GLM474-common-AB-original-reconcile-v1'
HELPER=Path('C:/Users/WangHaojie/.codex/visualizations/2026/10/12/01a130bd-3b97-73c2-8ecf-eda5151f1ba7/glm-paired-readonly-connection.py')
HELPER_SHA='3f00e0eac9b40eb3fee7d788776389bc22c3e87d20de0441312f04a2566178d1'
H=lambda b:hashlib.sha256(b).hexdigest()
def need(ok,code):
    if not ok:raise ValueError(code)
def save(path,value):
    with path.open('x',encoding='utf8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def main(a):
    epoch=time.time();start=time.monotonic();request=uuid.uuid4().hex
    def left():
        n=min(epoch+90-time.time(),90-(time.monotonic()-start));need(math.isfinite(n) and n>0,'ORIGIN_EXHAUSTED');return n
    need(re.fullmatch('[0-9a-f]{40}',a.publication) and all(re.fullmatch('[0-9a-f]{64}',v) for v in (a.source_manifest_sha256,a.claim_sha256)),'CLI_PINS')
    need(not L.exists(),'ONCE_NAMESPACE_EXISTS');L.mkdir(mode=0o700)
    q=dict(reconcile_request_id=request,epoch=epoch,deadline=epoch+90,publication=a.publication,attempted=False,state='NOT_ATTEMPTED',replay_allowed=False)
    save(L/'intent.json',dict(q,source_manifest_SHA256=a.source_manifest_sha256,claim_SHA256=a.claim_sha256));out=err=b''
    try:
        git=['git','--no-optional-locks','-c','safe.directory='+G.as_posix(),'-C',str(G)];rel=D.relative_to(G).as_posix()
        def show(n):return subprocess.check_output(git+['show',a.publication+':'+rel+'/'+n],timeout=min(10,left()),stderr=subprocess.DEVNULL)
        raw=(D/'SOURCE_FILES.json').read_bytes();need(H(raw)==a.source_manifest_sha256 and show('SOURCE_FILES.json')==raw,'PUBLISHED_MF');mf=json.loads(raw)
        for name,pin in mf.items():
            need(len(Path(name).parts)==1 and not Path(name).is_absolute(),'MEMBER_PATH');b=(D/name).read_bytes();need(len(b)==pin['bytes'] and H(b)==pin['SHA256'] and show(name)==b,'PUBLISHED_MEMBER')
        need(not subprocess.check_output(git+['status','--porcelain','--untracked-files=all','--',rel],timeout=min(10,left())),'OWN_SOURCE_DIRTY')
        raw=(D/'CLAIM.json').read_bytes();c=json.loads(raw);need(H(raw)==a.claim_sha256 and c['ROOT_GO'] and c['read_only_GO'] and c['maximum_invocations']==1 and not c['GPU_GO'],'CLAIM_NOT_APPROVED')
        need(H(HELPER.read_bytes())==HELPER_SHA,'CONNECTION_PIN');ssh=runpy.run_path(str(HELPER))['ssh']
        payload=('EPOCH='+repr(epoch)+'\nREQUEST='+repr(request)+'\n').encode()+(D/'reconcile.py').read_bytes();need(len(payload)<=32768,'PAYLOAD_BOUND')
        save(L/'exec-intent.json',dict(q,attempted=True,payload_bytes=len(payload),payload_SHA256=H(payload)));q['attempted']=True
        done=subprocess.run([ssh[0],'-o','StrictHostKeyChecking=yes',*ssh[1:],'/usr/local/python3.12/bin/python3 -I -S -B -u -'],input=payload,capture_output=True,timeout=left()+5);out,err=done.stdout,done.stderr
        need(len(out)<=131072 and len(err)<=65536,'OUTPUT_BOUND');docs=[]
        for line in out.splitlines():
            try:value=json.loads(line)
            except (ValueError,UnicodeError):continue
            if isinstance(value,dict) and value.get('reconcile_request_id')==request:docs.append(value)
        need(len(docs)==1,'RETURN_COUNT');q.update(rc=done.returncode,summary=docs[0],state='RETURNED' if done.returncode==0 else 'UNKNOWN')
        save(L/'result.json',docs[0])
    except subprocess.TimeoutExpired as error:out,err=error.stdout or b'',error.stderr or b'';q.update(error_type='TimeoutExpired',state='UNKNOWN')
    except BaseException as error:q.update(error_type=type(error).__name__,state='UNKNOWN' if q['attempted'] else 'NOT_ATTEMPTED')
    (L/'stdout.private').write_bytes(out);(L/'stderr.private').write_bytes(err)
    q.update(stdout_bytes=len(out),stdout_SHA256=H(out),stderr_bytes=len(err),stderr_SHA256=H(err),elapsed_s=time.monotonic()-start);save(L/'receipt.json',q);print(json.dumps(q,allow_nan=False))
    return 0 if q['state']=='RETURNED' else 2
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--publication',required=True);p.add_argument('--source-manifest-sha256',required=True);p.add_argument('--claim-sha256',required=True);raise SystemExit(main(p.parse_args()))
