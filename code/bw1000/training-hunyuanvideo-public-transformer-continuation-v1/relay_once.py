"""Send only the official public suffix from a verified offset."""
import ast
import hashlib
import http.client
import json
from pathlib import Path
import queue
import shlex
import subprocess
import threading
import time
import urllib.request
import uuid

D=Path(__file__).resolve().parent
B=D.parent
G=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test')
git=['git','-c','safe.directory='+G.as_posix(),'-C',str(G)]
claim=json.loads((D/'source-publication-receipt.json').read_bytes())['commit']
task=json.loads(subprocess.check_output(git+['show',claim+':coordination/tasks/training-hunyuanvideo-public-transformer-continuation-v1.json']))
assert task['transfer_GO'] and not (D/'launch-intent.json').exists()
blobs={}
for name,digest in task['source_SHA256'].items():
    raw=subprocess.check_output(git+['show',claim+':'+task['source']+'/'+name])
    assert hashlib.sha256(raw).hexdigest()==digest
    blobs[name]=raw
assert Path(__file__).read_bytes()==blobs['relay_once.py']
plan=json.loads(blobs['PLAN.json'])
tree=ast.parse((B/'read_extra40_approval_original_handles_once.py').read_bytes())
i=next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Assign)
       and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='ssh')
ns={'__file__':__file__}
exec(compile(ast.Module(body=tree.body[:i+1],type_ignores=[]),'existing-hygon-ssh','exec'),ns)

def fetch(start, end):
    length=end-start+1
    request=urllib.request.Request(plan['url'],headers={
        'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'})
    for attempt in range(4):
        if time.monotonic()+45>=mono_deadline or time.time()+45>=deadline:
            raise TimeoutError('no range budget')
        try:
            with urllib.request.urlopen(request,timeout=30) as r:
                if r.status!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{plan["total"]}':
                    raise ValueError('invalid public Content-Range')
                data=r.read(length+1)
            if len(data)!=length:
                raise EOFError('incomplete public range')
            return data
        except (OSError,EOFError,http.client.IncompleteRead):
            if attempt==3:
                raise
            time.sleep(1<<attempt)

epoch=time.time()
start_mono=time.monotonic()
deadline=epoch+plan['seconds']
mono_deadline=start_mono+plan['seconds']
request=uuid.uuid4().hex
base=dict(claim=claim,request_id=request,deadline=deadline)
prefix=f'PLAN={plan!r}\nDEADLINE={deadline!r}\nCLAIM={claim!r}\nREQUEST={request!r}\n'
ssh=ns['ssh']
ssh[-1]=ssh[-1].removesuffix(' -')+' -c '+shlex.quote(prefix+blobs['receive.py'].decode())
with (D/'launch-intent.json').open('x') as f:
    json.dump(dict(**base,epoch=epoch),f)
sent=0
error=None
value=None
with (D/'ssh.stderr.private').open('xb') as err:
    proc=subprocess.Popen(ssh,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err)
    ready=queue.Queue()
    threading.Thread(target=lambda:ready.put(proc.stdout.readline(16385)),daemon=True).start()
    timer=threading.Timer(plan['seconds'],lambda:proc.kill() if proc.poll() is None else None)
    timer.daemon=True
    timer.start()
    try:
        ack=json.loads(ready.get(timeout=60))
        assert ack==dict(**base,ready=True,offset=plan['offset'])
        with (D/'receiver-ready.json').open('x') as f:
            json.dump(ack,f)
        for start in range(plan['offset'],plan['total'],plan['range_bytes']):
            data=fetch(start,min(start+plan['range_bytes'],plan['total'])-1)
            for offset in range(0,len(data),1024**2):
                block=data[offset:offset+1024**2]
                proc.stdin.write(block)
                sent+=len(block)
                delay=max(0,sent/plan['rate_cap']-(time.monotonic()-start_mono))
                if time.monotonic()+delay+15>=mono_deadline:
                    raise TimeoutError('closing budget exhausted')
                if delay:
                    time.sleep(delay)
        proc.stdin.close()
        value=json.loads(proc.stdout.readline(16385))
        assert all(value[k]==v for k,v in base.items())
    except Exception as e:
        error=type(e).__name__
        if not proc.stdin.closed:
            try:
                proc.stdin.close()
            except OSError:
                pass
    finally:
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        timer.cancel()
passed=bool(not error and proc.returncode==0 and value and value.get('passed') is True
            and value.get('whole_SHA256')==plan['full_SHA256'] and sent==plan['total']-plan['offset'])
result=dict(**base,passed=passed,bytes_sent=sent,rc=proc.returncode,error=error,receiver=value)
with (D/'transfer-outcome.json').open('x') as f:
    json.dump(result,f)
print(json.dumps(result))
raise SystemExit(0 if passed else 1)
