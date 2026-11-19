"""One published GPU evaluation request; all preparation/transport share the first5400 epoch."""
import argparse,hashlib,json,os,re,subprocess,time,uuid
from pathlib import Path
D=Path(__file__).resolve().parent;G=D.parents[2]
def canonical(v):return (json.dumps(v,sort_keys=True,separators=(',',':'))+'\n').encode()
def digest(v):return hashlib.sha256(v).hexdigest()
def need(ok,code):
    if not ok:raise ValueError(code)
SSH=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','StrictHostKeyChecking=yes','-F','C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline/ssh_config','a100-27','/home/whj/nvidia-baseline/env-vlm/bin/python -I -S -B -u -']
def main():
    epoch=time.time();mono=time.monotonic()
    p=argparse.ArgumentParser();p.add_argument('--publication',required=True);a=p.parse_args()
    def left():
        v=min(epoch+5395-time.time(),mono+5395-time.monotonic());need(v>0,'ORIGINAL5395_CLIENT_STOP');return v
    def git(*args):
        q=subprocess.run(['git','--no-optional-locks','-c','safe.directory='+str(G),'-C',str(G),*args],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=min(15,left()))
        need(q.returncode==0,'PUBLISHED_GIT_ADMISSION');return q.stdout
    from verify_package import verify
    verified=verify(budget=left);need(verified['root_GO'] is True and verified['GPU_GO'] is True,'ROOT_GPU_ADOPTED')
    need(re.fullmatch('[0-9a-f]{40}',a.publication),'PUBLICATION')
    need(git('rev-parse','HEAD').decode().strip()==a.publication,'LOCAL_HEAD')
    need(git('remote','get-url','origin').decode().strip()=='git@github.com:whjthu/InfiniLM-hygon-test.git','EXACT_ORDINARY_ORIGIN')
    need(not git('diff','--cached','--name-only').strip(),'EMPTY_INDEX')
    line=git('ls-remote','origin','refs/heads/bw1000-experiments').decode().split();need(len(line)==2 and line[0]==a.publication and line[1]=='refs/heads/bw1000-experiments','ORDINARY_REMOTE_EQUALS_PUBLICATION')
    rel=D.relative_to(G).as_posix();need(not git('status','--porcelain','--',rel).strip(),'SOURCE_CLEAN')
    mf=json.loads((D/'SOURCE_FILES.json').read_bytes())['files']
    for name in list(mf)+['SOURCE_FILES.json']:
        left();need(git('cat-file','blob',a.publication+':'+rel+'/'+name)==(D/name).read_bytes(),'PUBLISHED_SOURCE_EXACT')
    plan=json.loads((D/'INSTALL_PLAN.json').read_bytes());core=json.loads((D/'core/PLAN.json').read_bytes())
    q=core['native150_public'];audit=git('cat-file','blob',a.publication+':'+q['path'])
    need(len(audit)==q['bytes'] and digest(audit)==q['SHA256'],'ACTUAL_NATIVE150_PUBLISHED')
    L=G/'local-only'/(plan['label']+'-original-dispatch');need(not L.exists(),'NO_REPLAY');L.mkdir(mode=0o700)
    request=uuid.uuid4().hex
    def save(name,obj):
        with (L/name).open('xb') as f:f.write(canonical(obj));f.flush();os.fsync(f.fileno())
    intent=dict(request_id=request,publication=a.publication,epoch=epoch,deadline=epoch+5400,source_manifest_SHA256=digest((D/'SOURCE_FILES.json').read_bytes()),claim_SHA256=digest((D/'CLAIM.json').read_bytes()),namespace=plan['namespace'],planned_generations=300)
    save('intent.json',intent);child=None;attempted=False;out=b'';err=b'';error=None;reaped=False;timed_out=False
    try:
        from build_body import make_body
        body=make_body(a.publication,request,epoch).encode()
        with (L/'body.private').open('xb') as f:f.write(body)
        left();save('attempt.json',dict(intent,exec_request_attempted=True,body_bytes=len(body),body_SHA256=digest(body)));attempted=True
        child=subprocess.Popen(SSH,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        save('held-SSH.json',dict(PID=child.pid,request_id=request,held_Popen=True))
        out,err=child.communicate(body,timeout=left())
    except BaseException as e:
        error=dict(type=type(e).__name__,message_SHA256=digest(str(e).encode()));timed_out=isinstance(e,subprocess.TimeoutExpired)
    finally:
        if child is not None:
            try:
                if child.poll() is None:child.kill()
            except BaseException as e:error=error or dict(type=type(e).__name__,phase='HELD_KILL')
            try:
                reserve=max(0,min(3,mono+5398-time.monotonic()))
                if reserve>0:out,err=child.communicate(timeout=reserve)
                reaped=child.poll() is not None
            except BaseException as e:error=error or dict(type=type(e).__name__,phase='HELD_REAP')
        else:reaped=True
        (L/'stdout.private').write_bytes(out);(L/'stderr.private').write_bytes(err)
        remote=None
        if child and child.returncode==0 and not error:
            try:remote=json.loads(out.decode().splitlines()[-1]);need(remote['request_id']==request,'RETURN_CONTEXT')
            except BaseException as e:error=dict(type=type(e).__name__,phase='RETURN_PARSE')
        state='KNOWN_LOCAL_PRE_SSH_FAILURE' if not attempted else ('RETURNED' if child and child.returncode not in (None,255) and not error else 'UNKNOWN')
        record=dict(intent,state=state,exec_request_attempted=attempted,returncode=child.returncode if child else None,error=error,timeout=timed_out,held_reaped=reaped,elapsed_monotonic=time.monotonic()-mono,remote=remote,stdout=dict(bytes=len(out),SHA256=digest(out)),stderr=dict(bytes=len(err),SHA256=digest(err)),replay_allowed=False)
        save('receipt.json',record)
    print(json.dumps(dict(request_id=request,state=state,passed=bool(remote and remote.get('engineering_completed') is True),held_reaped=reaped)))
    return 0 if remote and remote.get('engineering_completed') is True and reaped and not error else 1
if __name__=='__main__':raise SystemExit(main())
