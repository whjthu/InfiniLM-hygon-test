"""Invoke only the published original native Qwen GPU reference,once."""
from pathlib import Path
import base64,hashlib,json,subprocess,uuid
D=Path(__file__).resolve().parent;G=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test');claim=json.loads((D/'source-publication-receipt.json').read_bytes())['commit'];git=['git','-c','safe.directory='+G.as_posix(),'-C',str(G)]
def blob(name):return subprocess.check_output(git+['show',claim+':'+name])
task=json.loads(blob('coordination/tasks/training-qwen-A100-first-updates-v1.json'));assert task['GPU_GO'] and task['max_native_GPU_launches']==1 and task['actual_GPU_launches']==0;sources={}
for name,pin in task['source_SHA256'].items():
 raw=blob(task['source']+'/'+name);assert hashlib.sha256(raw).hexdigest()==pin;sources[name]=raw
files={name:dict(b64=base64.b64encode(sources[name]).decode(),bytes=len(sources[name]),SHA256=task['source_SHA256'][name]) for name in task['remote_files']};request=uuid.uuid4().hex;code=('CLAIM='+repr(claim)+'\nREQUEST='+repr(request)+'\nFILES='+repr(files)+'\n').encode()+sources['receive_reference.py']
with (D/'launch-intent.json').open('x') as f:json.dump(dict(claim=claim,request_id=request,payload_SHA256=hashlib.sha256(code).hexdigest(),exec_request_attempted=True,replay_allowed=False,GPU_cards=8),f,indent=2)
A=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline');ssh=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(A/'ssh_config'),'a100-26','python3 -B -u -']
try:r=subprocess.run(ssh,input=code,capture_output=True,timeout=960);rc=r.returncode;stdout=r.stdout;stderr=r.stderr;unknown=rc==255
except subprocess.TimeoutExpired as e:rc=None;stdout=e.stdout or b'';stderr=e.stderr or b'';unknown=True
(D/'original-stdout.private.json').write_bytes(stdout);(D/'original-stderr.private.txt').write_bytes(stderr);documents=[]
for line in stdout.splitlines():
 try:documents.append(json.loads(line))
 except ValueError:pass
acks=[x for x in documents if x.get('stage')=='REMOTE_SOURCE_INSTALLED' and x.get('request_id')==request];final=[x for x in documents if 'supervisor_rc' in x and x.get('request_id')==request];result=dict(claim=claim,request_id=request,transport_rc=rc,remote_exec_acknowledged=bool(acks),remote_outcome='UNKNOWN' if unknown else 'ORIGINAL_RETURNED',stdout_SHA256=hashlib.sha256(stdout).hexdigest(),stderr_SHA256=hashlib.sha256(stderr).hexdigest(),replay_allowed=False)
if final:assert len(final)==1;result['value']=final[0]
(D/'transport-outcome.json').write_bytes((json.dumps(result,indent=2)+'\n').encode());v=result.get('value',{});o=v.get('outcome.json',{}).get('value',{});print(json.dumps(dict(transport_rc=rc,remote_exec_acknowledged=bool(acks),supervisor_rc=v.get('supervisor_rc'),native_returncode=o.get('native_returncode'),keeper_passed=o.get('keeper_passed'),native25_rank_records=o.get('native25_rank_records'),capture_rank_records=o.get('capture_rank_records'),prelaunch=v.get('prelaunch-result.json',{}).get('value',{}).get('state'),remote_outcome=result['remote_outcome'])))
