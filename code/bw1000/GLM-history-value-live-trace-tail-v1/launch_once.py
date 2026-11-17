"""Published one-shot CPU diagnostic. Preparation does not execute this entry."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import shlex
import subprocess
import time
import uuid

D = Path(__file__).resolve().parent
G = D.parents[2]
LABEL = 'GLM188-live-trace-tail-v1'
HELPER = Path('C:/Users/WangHaojie/.codex/visualizations/2026/10/12/01a130bd-3b97-73c2-8ecf-eda5151f1ba7/glm-paired-readonly-connection.py')
HELPER_SHA = '3f00e0eac9b40eb3fee7d788776389bc22c3e87d20de0441312f04a2566178d1'
H = lambda b: hashlib.sha256(b).hexdigest()


def main():
    epoch = time.time(); end = time.monotonic()+30
    p = argparse.ArgumentParser()
    p.add_argument('--publication', required=True)
    p.add_argument('--manifest-sha256', required=True)
    a = p.parse_args()
    assert re.fullmatch('[0-9a-f]{40}',a.publication)
    def left():
        n=end-time.monotonic()
        if n<=0: raise TimeoutError('ORIGINAL30')
        return n
    git=['git','-c','safe.directory='+G.as_posix(),'-C',str(G),'--no-optional-locks']
    package=D.relative_to(G).as_posix()
    def published(name):
        return subprocess.check_output(git+['show',a.publication+':'+package+'/'+name],timeout=min(3,left()))
    raw=published('SOURCE_FILES.json')
    assert raw==(D/'SOURCE_FILES.json').read_bytes() and H(raw)==a.manifest_sha256
    files={'SOURCE_FILES.json':base64.b64encode(raw).decode()}
    for name,pin in json.loads(raw)['files'].items():
        assert Path(name).name==name
        b=published(name)
        assert b==(D/name).read_bytes() and len(b)==pin['bytes'] and H(b)==pin['SHA256']
        files[name]=base64.b64encode(b).decode()
    assert not subprocess.check_output(git+['status','--porcelain','--untracked-files=all','--',package],timeout=min(3,left()))
    claim=json.loads(base64.b64decode(files['CLAIM.json']))
    assert claim['ROOT_GO'] is True and claim['CPU_GO'] is True and claim['GPU_GO'] is False
    assert claim['label']==LABEL and claim['max_invocations']==1
    assert H(HELPER.read_bytes())==HELPER_SHA
    ssh=runpy.run_path(str(HELPER))['ssh']
    local=G/'local-only'/LABEL
    local.mkdir(mode=0o700)  # Fixed namespace: no retry/new request after any attempt.
    request=uuid.uuid4().hex
    intent=dict(request=request,publication=a.publication,epoch=epoch,deadline=epoch+30,
                source_manifest_SHA256=a.manifest_sha256,replay_allowed=False)
    def save(name,q):
        with (local/name).open('x',encoding='utf8') as f:
            json.dump(q,f,allow_nan=False);f.flush();os.fsync(f.fileno())
    save('intent.json',intent)
    bootstrap="""import base64,hashlib,json,os,pathlib,signal,sys,time
q=json.loads(sys.stdin.buffer.readline(131073));assert len(json.dumps(q))<131072
left=q['epoch']+28-time.time();assert 0<left<=28
signal.signal(signal.SIGALRM,lambda *_:os._exit(124));signal.setitimer(signal.ITIMER_REAL,left)
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/GLM188-live-trace-tail-v1')
r.mkdir(mode=0o700);s=r/'src';s.mkdir(mode=0o700)
mf=base64.b64decode(q['files']['SOURCE_FILES.json']);assert hashlib.sha256(mf).hexdigest()==q['source_manifest_SHA256']
pins=json.loads(mf)['files'];assert set(q['files'])==set(pins)|{'SOURCE_FILES.json'}
for n,v in q['files'].items():
 assert pathlib.Path(n).name==n
 b=base64.b64decode(v,validate=True)
 if n in pins:assert len(b)==pins[n]['bytes'] and hashlib.sha256(b).hexdigest()==pins[n]['SHA256']
 with (s/n).open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
with (r/'install-intent.json').open('x') as f:json.dump({k:v for k,v in q.items() if k!='files'},f);f.flush();os.fsync(f.fileno())
os.execv('/usr/local/python3.12/bin/python3',['python3','-I','-S','-B',str(s/'read_tail.py'),'control',str(q['epoch']),q['request'],q['publication']])
"""
    wire=json.dumps(dict(intent,files=files),separators=(',',':')).encode()+b'\n'
    assert len(wire)<=131072
    command='/usr/local/python3.12/bin/python3 -I -S -B -u -c '+shlex.quote(bootstrap)
    child=None;stdout=b'';error=None
    save('exec-intent.json',dict(intent,payload_SHA256=H(wire),dispatch_attempted=True))
    with (local/'stderr.private').open('xb') as err:
        try:
            child=subprocess.Popen([ssh[0],'-o','StrictHostKeyChecking=yes',*ssh[1:],command],
                                   stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err)
            stdout,_=child.communicate(wire,timeout=max(.01,left()-1))
        except BaseException as e:
            error=type(e).__name__
        finally:
            if child is not None and child.poll() is None: child.kill()
            if child is not None:
                try:
                    extra,_=child.communicate(timeout=max(.01,end-time.monotonic()))
                    if not stdout:stdout=extra
                except subprocess.TimeoutExpired:pass
    (local/'stdout.private').write_bytes(stdout)
    result=dict(intent,error_type=error,returncode=child.returncode if child else None,
                stdout_bytes=len(stdout),stdout_SHA256=H(stdout),state='UNKNOWN')
    if child is not None and child.returncode==0 and len(stdout)<=131072:
        try:
            observed=json.loads(stdout)
            assert observed['request']==request and observed['publication']==a.publication
            result.update(state='RETURNED',remote=observed)
        except (ValueError,KeyError,AssertionError):pass
    save('receipt.json',result)
    print(json.dumps(result,allow_nan=False))


if __name__=='__main__':
    main()
