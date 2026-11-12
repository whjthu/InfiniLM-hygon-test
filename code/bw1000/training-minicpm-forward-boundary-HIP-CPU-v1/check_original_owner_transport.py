"""Small readonly diagnostic: distinguish current SSH closure from reader size."""
from pathlib import Path
import hashlib,json,subprocess
D=Path(__file__).resolve().parent;B=D.parent
remote=b'''
from pathlib import Path
import hashlib,json,socket
W=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-minicpm-forward-boundary-HIP-CPU-v1')
intent=json.loads((W/'intent.json').read_bytes());owner=json.loads((W/'owned-srun.json').read_bytes());host=socket.gethostname();match=None
if host==owner['host']:
 p=Path('/proc/%d/stat'%owner['PID']);match=p.exists() and int(p.read_text().split(') ',1)[1].split()[19])==owner['startticks']
raw=(W/'outcome.json').read_bytes();out=json.loads(raw)
print(json.dumps(dict(passed=bool(out['passed']),current_HOST=host,owner=owner,host_matches=host==owner['host'],original_PID_identity_present=match,claim=intent['claim'],request_id=intent['request_id'],outcome_SHA256=hashlib.sha256(raw).hexdigest(),accounting=out['accounting'],CPU_tests=out['CPU-result.json']['value']['tests_run'],GPU_cards=0,new_Slurm_steps=0)))
'''
ns={'__file__':str(D/'reuse.py')};s=(B/'i03-deep-c4-history-staging-v1/read_source_once.py').read_text(encoding='utf8');exec(compile(s.split('REMOTE=',1)[0],'existing_ordinary_SSH','exec'),ns)
with (D/'owner-transport-intent.json').open('x') as f:json.dump(dict(payload_bytes=len(remote),payload_SHA256=hashlib.sha256(remote).hexdigest(),read_only=True,CPU_relaunch=False),f,indent=2)
r=subprocess.run(ns['ssh'],input=remote,capture_output=True,timeout=45)
for n,raw in [('owner-transport.stdout.private.json',r.stdout),('owner-transport.stderr.private.txt',r.stderr)]:
 with (D/n).open('xb') as f:f.write(raw)
v=dict(transport_rc=r.returncode,payload_bytes=len(remote),stdout_SHA256=hashlib.sha256(r.stdout).hexdigest(),stderr_SHA256=hashlib.sha256(r.stderr).hexdigest(),category='Connection closed' if b'connection closed' in r.stderr.lower() else None)
if r.returncode==0:v['value']=json.loads(r.stdout)
with (D/'owner-transport-result.json').open('x') as f:json.dump(v,f,indent=2)
value=v.get('value',{});print(json.dumps(dict(transport_rc=r.returncode,category=v['category'],small_payload_bytes=len(remote),CPU_tests=value.get('CPU_tests'),host_matches=value.get('host_matches'),original_PID_identity_present=value.get('original_PID_identity_present'),new_Slurm_steps=0,GPU_cards=0)))
