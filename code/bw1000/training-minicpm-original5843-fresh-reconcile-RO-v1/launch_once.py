"""One ordinary published readonly SSH request; root adopts GO before invocation."""
import argparse,ast,hashlib,json,os,re,runpy,subprocess,time,uuid
from pathlib import Path
D=Path(__file__).resolve().parent;G=D.parents[2]
HELPER=Path('C:/Users/WangHaojie/.codex/visualizations/2026/10/12/01a130bd-3b97-73c2-8ecf-eda5151f1ba7/glm-paired-readonly-connection.py')
HELPER_SHA='3f00e0eac9b40eb3fee7d788776389bc22c3e87d20de0441312f04a2566178d1'
PRIOR=[('results/bw1000/2026-11-12/training-minicpm-first-boundary-original-request-execution-unknown.json',6355,'3be650db2c90451a5ab06abb5bd5bd22b4f1244be1fed3f55d6b924a36cebf2d'),
       ('results/bw1000/2026-11-17/training-minicpm-first-boundary-original-successful-readonly-reconciliation.json',899,'b4598fd5d61e369740d6aae4eec2e8bdd153d8dc14e2d23d154f4d48f5912acb')]
def digest(b):return hashlib.sha256(b).hexdigest()
def need(ok,code):
    if not ok:raise ValueError(code)
def main():
    epoch=time.time();mono=time.monotonic()
    parser=argparse.ArgumentParser();parser.add_argument('--publication',required=True);a=parser.parse_args()
    def left():
        v=min(epoch+42-time.time(),mono+42-time.monotonic());need(v>0,'ORIGINAL42_CLIENT_STOP');return v
    def git(*args):
        q=subprocess.run(['git','--no-optional-locks','-c','safe.directory='+str(G),'-C',str(G),*args],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=min(8,left()))
        need(q.returncode==0,'PUBLISHED_GIT_ADMISSION');return q.stdout
    need(re.fullmatch('[0-9a-f]{40}',a.publication),'PUBLICATION')
    claim_raw=(D/'CLAIM.json').read_bytes();claim=json.loads(claim_raw)
    need(claim['root_GO'] is True and claim['READ_GO'] is True and claim['max_invocations']==1
         and claim['GPU_GO'] is False and claim['Model_GO'] is False,'ROOT_READ_ONLY_ADOPTED')
    need(git('rev-parse','HEAD').decode().strip()==a.publication,'LOCAL_HEAD')
    need(git('remote','get-url','origin').decode().strip()=='git@github.com:whjthu/InfiniLM-hygon-test.git','ORDINARY_ORIGIN')
    need(not git('diff','--cached','--name-only').strip(),'EMPTY_INDEX')
    head=git('ls-remote','origin','refs/heads/bw1000-experiments').decode().split()
    need(head==[a.publication,'refs/heads/bw1000-experiments'],'ORDINARY_REMOTE_HEAD')
    rel=D.relative_to(G).as_posix();need(not git('status','--porcelain','--',rel).strip(),'SOURCE_CLEAN')
    mf_raw=(D/'SOURCE_FILES.json').read_bytes();mf=json.loads(mf_raw)['files']
    need({p.name for p in D.iterdir() if p.is_file()}==set(mf)|{'SOURCE_FILES.json'},'EXACT_INVENTORY')
    for name in list(mf)+['SOURCE_FILES.json']:
        raw=(D/name).read_bytes();need(git('cat-file','blob',a.publication+':'+rel+'/'+name)==raw,'PUBLISHED_SOURCE_BYTES')
        if name in mf:need(len(raw)==mf[name]['bytes'] and digest(raw)==mf[name]['SHA256'],'LOCAL_SOURCE_PIN')
        if name.endswith('.py'):ast.parse(raw)
    for path,size,pin in PRIOR:
        raw=git('cat-file','blob',a.publication+':'+path);need(len(raw)==size and digest(raw)==pin,'PUBLISHED_ORIGINAL_AND_PREVIOUS_RECONCILIATION')
    need(digest(HELPER.read_bytes())==HELPER_SHA,'EXACT_EXISTING_SSH_HELPER')
    left();argv=runpy.run_path(str(HELPER))['ssh']
    argv=[argv[0],'-o','StrictHostKeyChecking=yes',*argv[1:],'/usr/local/python3.12/bin/python3 -I -S -B -u -']
    L=G/'local-only'/(D.name+'-original');need(not L.exists(),'NO_REPLAY_NAMESPACE');L.mkdir(mode=0o700)
    request=uuid.uuid4().hex
    def save(name,value):
        with (L/name).open('xb') as f:f.write((json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode());f.flush();os.fsync(f.fileno())
    intent=dict(request_id=request,publication=a.publication,epoch=epoch,deadline=epoch+45,source_manifest_SHA256=digest(mf_raw),claim_SHA256=digest(claim_raw),original_request=claim['original_request'],original_namespace=claim['original_namespace'],parent=claim['parent'],helper_SHA256=HELPER_SHA)
    save('intent.json',intent);child=None;out=b'';err=b'';error=None;attempted=False;forced=False;reaped=False
    try:
        from build_body import make_body
        body=make_body(a.publication,request,epoch);left()
        save('attempt.json',dict(intent,exec_request_attempted=True,body_bytes=len(body),body_SHA256=digest(body)));attempted=True
        child=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        save('held-SSH.json',dict(PID=child.pid,request_id=request,held_Popen=True))
        out,err=child.communicate(body,timeout=left())
    except BaseException as e:error=dict(type=type(e).__name__,message_SHA256=digest(str(e).encode()))
    finally:
        if child is not None:
            try:
                if child.poll() is None:child.kill();forced=True
            except BaseException as e:error=error or dict(type=type(e).__name__,phase='HELD_KILL')
            try:
                reserve=max(0,min(2,mono+44-time.monotonic()))
                if reserve>0:out,err=child.communicate(timeout=reserve)
                reaped=child.poll() is not None
            except BaseException as e:error=error or dict(type=type(e).__name__,phase='HELD_REAP')
        else:reaped=True
        bounded=len(out)<=131072 and len(err)<=65536
        (L/'stdout.private').write_bytes(out[:131072]);(L/'stderr.private').write_bytes(err[:65536])
        remote=None
        if bounded and not error and child is not None and child.returncode==0 and reaped and not forced:
            try:
                remote=json.loads(out);need(remote['request_id']==request and remote['publication']==a.publication and remote['epoch']==epoch
                    and remote['original_request']==claim['original_request'] and remote['parent']==claim['parent'],'RETURN_CONTEXT')
            except BaseException as e:error=dict(type=type(e).__name__,phase='RETURN_PARSE')
        state='KNOWN_LOCAL_PRE_SSH_FAILURE' if not attempted else ('RETURNED' if child is not None and child.returncode not in (None,255) and reaped and not forced and not error and bounded else 'UNKNOWN')
        result=dict(intent,state=state,exec_request_attempted=attempted,returncode=child.returncode if child else None,error=error,held_reaped=reaped,held_forced_kill=forced,
            stdout=dict(bytes=len(out),SHA256=digest(out),stored_bytes=min(len(out),131072)),stderr=dict(bytes=len(err),SHA256=digest(err),stored_bytes=min(len(err),65536)),output_within_bound=bounded,
            elapsed_monotonic=time.monotonic()-mono,original_execution_state=remote.get('original_execution_state','UNKNOWN') if remote else 'UNKNOWN',replay_allowed=False,release_allowed=False)
        save('receipt.json',result)
    print(json.dumps({k:result[k] for k in ('request_id','state','returncode','held_reaped','original_execution_state','replay_allowed')}))
    return 0 if remote is not None and not error and reaped and bounded and not forced else 1
if __name__=='__main__':raise SystemExit(main())
