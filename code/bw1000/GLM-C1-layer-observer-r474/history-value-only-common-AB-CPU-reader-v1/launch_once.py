"""Root-adopted once reader only. Never launches a build or Slurm step."""
import argparse,base64,hashlib,json,math,re,runpy,subprocess,time,uuid
from pathlib import Path
D=Path(__file__).resolve().parent;G=D.parents[3]
L=G/'local-only/GLM474-history-value-only-common-AB-CPU-reader-v1-original'
HELPER=Path('C:/Users/WangHaojie/.codex/visualizations/2026/10/12/01a130bd-3b97-73c2-8ecf-eda5151f1ba7/glm-paired-readonly-connection.py')
HELPER_SHA='3f00e0eac9b40eb3fee7d788776389bc22c3e87d20de0441312f04a2566178d1'
ORIGINAL_REQUEST='094e92db7d214e318e2f08358dd06fd8';ORIGINAL_EPOCH=1794933214.2382555
ORIGINAL_PUBLICATION='6a1b000069379d8fe46f1086bb9c64d725a6d7e3'
H=lambda b:hashlib.sha256(b).hexdigest()
def need(ok,code):
    if not ok:raise ValueError(code)
def save(p,q):
    with p.open('x',encoding='utf8') as f:json.dump(q,f,indent=2,allow_nan=False);f.write('\n')
def main(a):
    epoch=time.time();anchor=time.monotonic()
    def left(reserve=0):
        n=min(epoch+180-time.time(),180-(time.monotonic()-anchor))-reserve;need(math.isfinite(n) and n>0,'READER_ORIGIN_EXHAUSTED');return n
    need(re.fullmatch('[0-9a-f]{40}',a.publication) and all(re.fullmatch('[0-9a-f]{64}',x) for x in (a.source_manifest_sha256,a.claim_sha256,a.outcome_sha256)),'CLI_PINS')
    need(not L.exists(),'ONCE_NAMESPACE_EXISTS');L.mkdir(mode=0o700)
    request=uuid.uuid4().hex;q=dict(audit_request_id=request,original_request_id=ORIGINAL_REQUEST,original_epoch=ORIGINAL_EPOCH,original_publication=ORIGINAL_PUBLICATION,epoch=epoch,deadline=epoch+180,publication=a.publication,attempted=False,state='NOT_ATTEMPTED',replay_allowed=False)
    save(L/'intent.json',dict(q,source_manifest_SHA256=a.source_manifest_sha256,claim_SHA256=a.claim_sha256,original_outcome_SHA256=a.outcome_sha256))
    stdout=stderr=b''
    try:
        git=['git','--no-optional-locks','-c','safe.directory='+G.as_posix(),'-C',str(G)];rel=D.relative_to(G).as_posix()
        def show(name):return subprocess.check_output(git+['show',a.publication+':'+rel+'/'+name],timeout=min(12,left(30)),stderr=subprocess.DEVNULL)
        mfraw=(D/'SOURCE_FILES.json').read_bytes();need(H(mfraw)==a.source_manifest_sha256 and show('SOURCE_FILES.json')==mfraw,'PUBLISHED_MF')
        mf=json.loads(mfraw);blobs={'SOURCE_FILES.json':mfraw}
        for name,pin in mf.items():
            need(len(Path(name).parts)==1 and not Path(name).is_absolute(),'SOURCE_MEMBER');b=(D/name).read_bytes()
            need(len(b)==pin['bytes'] and H(b)==pin['SHA256'] and show(name)==b,'PUBLISHED_MEMBER');blobs[name]=b
        need(sum(map(len,blobs.values()))<=256*1024,'SOURCE_BYTES_BOUND')
        need(not subprocess.check_output(git+['status','--porcelain','--untracked-files=all','--',rel],timeout=min(12,left(30))),'OWN_SOURCE_DIRTY')
        claim=json.loads(blobs['CLAIM.json']);need(H(blobs['CLAIM.json'])==a.claim_sha256 and claim['ROOT_GO'] and claim['CPU_read_GO'] and claim['maximum_invocations']==1 and not claim['GPU_GO'] and not claim['Native_compile_GO'],'PUBLISHED_CLAIM')
        need(H(HELPER.read_bytes())==HELPER_SHA,'CONNECTION_PIN');ssh=runpy.run_path(str(HELPER))['ssh']
        files={n:dict(bytes=len(b),SHA256=H(b),b64=base64.b64encode(b).decode()) for n,b in blobs.items()}
        constants=dict(READER_EPOCH=epoch,AUDIT_REQUEST=request,PUBLICATION=a.publication,SOURCE_SHA=a.source_manifest_sha256,CLAIM_SHA=a.claim_sha256,ORIGINAL_REQUEST=ORIGINAL_REQUEST,ORIGINAL_PUBLICATION=ORIGINAL_PUBLICATION,ORIGINAL_EPOCH=ORIGINAL_EPOCH,OUTCOME_SHA=a.outcome_sha256,FILES=files)
        payload=('\n'.join(k+'='+repr(v) for k,v in constants.items())+'\n').encode()+blobs['receive_once.py'];need(len(payload)<=512*1024,'PAYLOAD_BOUND')
        save(L/'exec-intent.json',dict(q,attempted=True,payload_bytes=len(payload),payload_SHA256=H(payload)));q['attempted']=True
        argv=[ssh[0],'-o','StrictHostKeyChecking=yes',*ssh[1:],'/usr/local/python3.12/bin/python3 -I -S -B -u -']
        done=subprocess.run(argv,input=payload,capture_output=True,timeout=left()+15);stdout,stderr=done.stdout,done.stderr
        need(time.monotonic()-anchor<=200 and len(stdout)<=6*1024**2 and len(stderr)<=128*1024,'TRANSPORT_BOUND')
        rows=[]
        for line in stdout.splitlines():
            try:v=json.loads(line)
            except (ValueError,UnicodeError):continue
            if isinstance(v,dict) and v.get('audit_request_id')==request:rows.append(v)
        need(len(rows)==1,'RETURN_SUMMARY_COUNT');v=rows[0]
        if 'report_b64' in v:
            raw=base64.b64decode(v.pop('report_b64'),validate=True);need(len(raw)<=4*1024**2 and len(raw)==v['report_bytes'] and H(raw)==v['report_SHA256'],'REPORT_TRANSFER_PIN')
            parsed=json.loads(raw);need(parsed['request_id']==ORIGINAL_REQUEST and parsed['original_epoch']==ORIGINAL_EPOCH and parsed['publication']==ORIGINAL_PUBLICATION,'REPORT_ORIGINAL_CONTEXT')
            with (L/'report-private.json').open('xb') as f:f.write(raw)
        q.update(rc=done.returncode,summary=v,state='RETURNED' if done.returncode==0 and v.get('state')=='RETURNED' else 'UNKNOWN')
    except subprocess.TimeoutExpired as error:stdout,stderr=error.stdout or b'',error.stderr or b'';q.update(state='UNKNOWN',error_type='TimeoutExpired')
    except BaseException as error:q.update(error_type=type(error).__name__,state='UNKNOWN' if q['attempted'] else 'NOT_ATTEMPTED')
    (L/'stdout.private').write_bytes(stdout);(L/'stderr.private').write_bytes(stderr)
    q.update(stdout_bytes=len(stdout),stdout_SHA256=H(stdout),stderr_bytes=len(stderr),stderr_SHA256=H(stderr),elapsed_s=time.monotonic()-anchor);save(L/'receipt.json',q);print(json.dumps(q,allow_nan=False))
    return 0 if q['state']=='RETURNED' else 2
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--publication',required=True);p.add_argument('--source-manifest-sha256',required=True);p.add_argument('--claim-sha256',required=True);p.add_argument('--outcome-sha256',required=True)
    raise SystemExit(main(p.parse_args()))
