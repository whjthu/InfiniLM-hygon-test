"""Send only the official public suffix from a verified offset."""
import hashlib
import http.client
import json
from pathlib import Path
import queue
import shlex
import subprocess
import sys
import threading
import time
import urllib.request
import uuid

D = Path(__file__).resolve().parent
G, claim, arm, attempt_dir = sys.argv[1:]
G = Path(G).resolve(strict=True)
A = Path(attempt_dir).resolve(strict=True)
git = ['git', '-c', 'safe.directory=' + G.as_posix(), '-C', str(G)]
task = json.loads(subprocess.check_output(git + ['show', claim + ':coordination/tasks/training-hunyuanvideo-public-resume-v2.json']))
assert task['transfer_GO'] and arm in task['assets'] and not (A / 'launch-intent.json').exists()
blobs = {}
for name, digest in task['source_SHA256'].items():
    raw = subprocess.check_output(git + ['show', claim + ':' + task['source'] + '/' + name])
    assert hashlib.sha256(raw).hexdigest() == digest
    blobs[name] = raw
assert Path(__file__).read_bytes() == blobs['relay_once.py']
plan = json.loads(blobs['PLAN.json'])['assets'][arm]
config_path = G.parent / 'infra-code-research/config/project.json'
c = json.loads(config_path.read_bytes())['cluster']
assert Path(c['ssh_key']).name == 'hygon-261111.txt'
known_hosts = (G / c['known_hosts']).resolve(strict=True)
ssh = ['ssh.exe', '-T', '-i', c['ssh_key'], '-p', str(c['ssh_port']),
       '-o', 'IdentitiesOnly=yes', '-o', 'StrictHostKeyChecking=yes',
       '-o', 'UserKnownHostsFile=' + str(known_hosts), '-o', 'BatchMode=yes',
       '-o', 'ConnectTimeout=10', c['ssh_host']]

def fetch(start, end):
    length=end-start+1
    request=urllib.request.Request(plan['url']+'?download=true&range='+str(start),headers={
        'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'})
    for attempt in range(7):
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
            if attempt==6:
                raise
            time.sleep(min(30,1<<attempt))

epoch=time.time()
start_mono=time.monotonic()
deadline=epoch+plan['seconds']
mono_deadline=start_mono+plan['seconds']
request=uuid.uuid4().hex
base=dict(claim=claim,request_id=request,deadline=deadline)
prefix=f'PLAN={plan!r}\nDEADLINE={deadline!r}\nCLAIM={claim!r}\nREQUEST={request!r}\n'
ssh.append('/usr/local/python3.12/bin/python3 -B -u -c '+shlex.quote(prefix+blobs['receive.py'].decode()))
with (A/'launch-intent.json').open('x') as f:
    json.dump(dict(**base,epoch=epoch),f)
sent=0
error=None
value=None
with (A/'ssh.stderr.private').open('xb') as err:
    proc=subprocess.Popen(ssh,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err)
    ready=queue.Queue()
    threading.Thread(target=lambda:ready.put(proc.stdout.readline(16385)),daemon=True).start()
    timer=threading.Timer(plan['seconds'],lambda:proc.kill() if proc.poll() is None else None)
    timer.daemon=True
    timer.start()
    try:
        ack=json.loads(ready.get(timeout=120))
        assert ack==dict(**base,ready=True,offset=plan['offset'])
        with (A/'receiver-ready.json').open('x') as f:
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
with (A/'transfer-outcome.json').open('x') as f:
    json.dump(result,f)
print(json.dumps(result))
raise SystemExit(0 if passed else 1)
