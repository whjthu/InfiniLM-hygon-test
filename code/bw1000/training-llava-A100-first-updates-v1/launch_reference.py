"""Invoke the original published once-only8A100 reference after fresh admission."""
from pathlib import Path
import base64,hashlib,json,re,subprocess,sys,time,uuid
D=Path(__file__).resolve().parent;G=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test')
tag=sys.argv[1];assert re.fullmatch('[a-z0-9-]{1,40}',tag)
claim=json.loads((D/'source-publication-receipt.json').read_bytes())['commit'];preflight=json.loads((D/('preflight-'+tag+'-result.json')).read_bytes())
assert preflight['claim']==claim and preflight['transport_rc']==0 and preflight['value']['passed'] and 0<=time.monotonic()-preflight['ended_monotonic']<=60
git=['git','-c','safe.directory='+G.as_posix(),'-C',str(G)]
def blob(n):return subprocess.check_output(git+['show',claim+':'+n])
task=json.loads(blob('coordination/tasks/training-llava-A100-first-updates-v1.json'));assert task['GPU_GO'] and task['max_native_GPU_launches']==1 and task['actual_GPU_launches']==0
sources={}
for name,pin in task['source_SHA256'].items():
 raw=blob(task['source']+'/'+name);assert hashlib.sha256(raw).hexdigest()==pin;sources[name]=raw
files={n:dict(b64=base64.b64encode(sources[n]).decode(),bytes=len(sources[n]),SHA256=task['source_SHA256'][n]) for n in task['remote_files']}
request=uuid.uuid4().hex;code=('CLAIM='+repr(claim)+'\nREQUEST='+repr(request)+'\nFILES='+repr(files)+'\n').encode()+sources['receive_reference.py']
with (D/'launch-intent.json').open('x') as f:json.dump(dict(claim=claim,request_id=request,payload_SHA256=hashlib.sha256(code).hexdigest(),exec_request_attempted=True,replay_allowed=False,GPU_cards=8,preflight_tag=tag),f,indent=2)
A=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline');ssh=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(A/'ssh_config'),'a100-27','python3 -B -u -']
try:r=subprocess.run(ssh,input=code,capture_output=True,timeout=960);rc=r.returncode;stdout=r.stdout;stderr=r.stderr;unknown=rc==255
except subprocess.TimeoutExpired as error:rc=None;stdout=error.stdout or b'';stderr=error.stderr or b'';unknown=True
for name,raw in [('original-stdout.private.json',stdout),('original-stderr.private.txt',stderr)]:
 with (D/name).open('xb') as f:f.write(raw)
docs=[]
for line in stdout.splitlines():
 try:docs.append(json.loads(line))
 except ValueError:pass
acks=[v for v in docs if v.get('stage')=='REMOTE_SOURCE_INSTALLED' and v.get('request_id')==request];intents=[v for v in docs if v.get('stage')=='REMOTE_INSTALL_INTENT' and v.get('request_id')==request];final=[v for v in docs if 'supervisor_rc' in v and v.get('request_id')==request]
receipt=dict(claim=claim,request_id=request,transport_rc=rc,remote_exec_acknowledged=bool(intents),remote_source_installed_acknowledged=bool(acks),remote_outcome='UNKNOWN' if unknown else 'ORIGINAL_RETURNED',stdout_SHA256=hashlib.sha256(stdout).hexdigest(),stderr_SHA256=hashlib.sha256(stderr).hexdigest(),replay_allowed=False)
if final:assert len(final)==1;receipt['value']=final[0]
with (D/'transport-outcome.json').open('x') as f:json.dump(receipt,f,indent=2)
v=receipt.get('value',{});o=v.get('outcome.json',{}).get('value',{});print(json.dumps(dict(transport_rc=rc,remote_exec_acknowledged=bool(intents),remote_source_installed_acknowledged=bool(acks),supervisor_rc=v.get('supervisor_rc'),native_returncode=o.get('native_returncode'),keeper_passed=o.get('keeper_passed'),native25_rank_records=o.get('native25_rank_records'),capture_rank_records=o.get('capture_rank_records'),prelaunch=v.get('prelaunch-result.json',{}).get('value',{}).get('state'),remote_outcome=receipt['remote_outcome'])))
