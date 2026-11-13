"""Relay the exact public Hugging Face revision over the existing Hygon SSH route."""
import ast
import hashlib
import json
from pathlib import Path
import queue
import shlex
import subprocess
import threading
import time
import urllib.request
import uuid

D = Path(__file__).resolve().parent
B = D.parent
G = Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test')
git = ['git','-c','safe.directory='+G.as_posix(),'-C',str(G)]
claim = json.loads((D/'source-publication-receipt.json').read_bytes())['commit']
task = json.loads(subprocess.check_output(git+['show',claim+':coordination/tasks/training-hunyuanvideo-public-transformer-relay-v1.json']))
assert task['transfer_GO'] is True
blobs = {}
for name, digest in task['source_SHA256'].items():
    raw = subprocess.check_output(git+['show',claim+':'+task['source']+'/'+name])
    assert hashlib.sha256(raw).hexdigest()==digest
    blobs[name]=raw
assert Path(__file__).read_bytes()==blobs['relay_once.py']
plan = json.loads(blobs['PLAN.json'])
assert not (D/'launch-intent.json').exists()
tree = ast.parse((B/'read_extra40_approval_original_handles_once.py').read_bytes())
i = next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Assign)
         and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='ssh')
ns = {'__file__':__file__}
exec(compile(ast.Module(body=tree.body[:i+1],type_ignores=[]),'existing-hygon-ssh','exec'),ns)
request=uuid.uuid4().hex
epoch=time.time()
anchor=time.monotonic()
deadline=epoch+plan['seconds']
prefix=f'PLAN={plan!r}\nDEADLINE={deadline!r}\nCLAIM={claim!r}\nREQUEST={request!r}\n'
ssh=ns['ssh']
ssh[-1]=ssh[-1].removesuffix(' -')+' -c '+shlex.quote(prefix+blobs['receive.py'].decode())
def dump(name, value):
    with (D/name).open('x',encoding='utf8') as f:
        json.dump(value,f,indent=2)
dump('launch-intent.json',dict(claim=claim,request_id=request,epoch=epoch,deadline=deadline))
sent=0
error=None
value=None
with (D/'ssh.stderr.private').open('xb') as err:
    proc=subprocess.Popen(ssh,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err)
    ready=queue.Queue()
    threading.Thread(target=lambda:ready.put(proc.stdout.readline()),daemon=True).start()
    timer=threading.Timer(plan['seconds'],lambda:proc.kill() if proc.poll() is None else None)
    timer.daemon=True
    timer.start()
    try:
        ack=json.loads(ready.get(timeout=30))
        assert ack==dict(claim=claim,request_id=request,deadline=deadline,ready=True)
        for name,pin in plan['files'].items():
            url=plan['url']+name
            req=urllib.request.Request(url,headers={'Accept-Encoding':'identity'})
            with urllib.request.urlopen(req,timeout=60) as stream:
                assert stream.status==200 and int(stream.headers['Content-Length'])==pin['bytes']
                for block in iter(lambda:stream.read(1024**2),b''):
                    proc.stdin.write(block)
                    sent+=len(block)
                    delay=max(0,sent/plan['rate_cap']-(time.monotonic()-anchor))
                    if time.time()+delay+15>=deadline:
                        raise TimeoutError('transfer deadline')
                    if delay:time.sleep(delay)
        proc.stdin.close()
        raw=proc.stdout.read(16385)
        assert len(raw)<=16384
        value=json.loads(raw)
        assert value['claim']==claim and value['request_id']==request
    except Exception as e:
        error=type(e).__name__
        if proc.stdin and not proc.stdin.closed:
            try:proc.stdin.close()
            except OSError:pass
    finally:
        try:proc.wait(timeout=15)
        except subprocess.TimeoutExpired:proc.kill();proc.wait()
        timer.cancel()
result=dict(request_id=request,claim=claim,rc=proc.returncode,bytes_sent=sent,error=error,receiver=value,
    passed=bool(value and value.get('passed') is True and not error and proc.returncode==0))
dump('transfer-outcome.json',result)
print(json.dumps(result))
raise SystemExit(0 if result['passed'] else 1)
