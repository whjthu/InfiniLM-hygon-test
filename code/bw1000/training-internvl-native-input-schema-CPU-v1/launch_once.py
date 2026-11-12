"""Invoke only the published Qwen original-A100 CPU structural scope once."""
from pathlib import Path
import base64,hashlib,json,subprocess,uuid
D=Path(__file__).resolve().parent
G=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test')
claim=json.loads((D/'source-publication-receipt.json').read_bytes())['commit']
git=['git','-c','safe.directory='+G.as_posix(),'-C',str(G)]
def blob(name):return subprocess.check_output(git+['show',claim+':'+name])
task=json.loads(blob('coordination/tasks/training-internvl-native-input-schema-CPU-v1.json'))
assert task['CPU_GO'] and not task['GPU_GO'] and task['actual_CPU_launches']==0
sources={}
for name,pin in task['source_SHA256'].items():
 raw=blob(task['source']+'/'+name);assert hashlib.sha256(raw).hexdigest()==pin;sources[name]=raw
files={remote_name:dict(b64=base64.b64encode(sources[source_name]).decode(),bytes=len(sources[source_name]),SHA256=task['source_SHA256'][source_name]) for remote_name,source_name in task['remote_files'].items()}
request=uuid.uuid4().hex
payload=('CLAIM='+repr(claim)+'\nREQUEST='+repr(request)+'\nFILES='+repr(files)+'\n').encode()+sources['receive_once.py']
with (D/'launch-intent.json').open('x',encoding='utf8') as f:json.dump(dict(claim=claim,request_id=request,payload_SHA256=hashlib.sha256(payload).hexdigest(),exec_request_attempted=True,replay_allowed=False,GPU_cards=0),f,indent=2)
A=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline')
ssh=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(A/'ssh_config'),'a100-27','python3 -B -u -']
try:
 r=subprocess.run(ssh,input=payload,capture_output=True,timeout=330);rc=r.returncode;stdout=r.stdout;stderr=r.stderr;unknown=rc==255
except subprocess.TimeoutExpired as error:
 rc=None;stdout=error.stdout or b'';stderr=error.stderr or b'';unknown=True
(D/'original-stdout.private.json').write_bytes(stdout);(D/'original-stderr.private.txt').write_bytes(stderr)
documents=[]
for line in stdout.splitlines():
 try:documents.append(json.loads(line))
 except ValueError:pass
acks=[x for x in documents if x.get('stage')=='REMOTE_SOURCE_INSTALLED' and x.get('request_id')==request]
final=[x for x in documents if 'returncode' in x and x.get('request_id')==request]
result=dict(claim=claim,request_id=request,transport_rc=rc,remote_exec_acknowledged=bool(acks),remote_outcome='UNKNOWN' if unknown else 'ORIGINAL_RETURNED',stdout_SHA256=hashlib.sha256(stdout).hexdigest(),stderr_SHA256=hashlib.sha256(stderr).hexdigest(),replay_allowed=False)
if final:
 assert len(final)==1;result['value']=final[0]
(D/'transport-outcome.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
v=result.get('value',{});c=v.get('CPU_result') or {}
print(json.dumps(dict(transport_rc=rc,remote_exec_acknowledged=bool(acks),remote_outcome=result['remote_outcome'],passed=v.get('passed'),returncode=v.get('returncode'),phase=c.get('phase'),error_type=c.get('error_type'),trainable_tensors=c.get('native_trainable_tensors'),trainable_parameters=c.get('native_trainable_parameters'),owned_group_reaped=v.get('owned_group_reaped'),GPU_cards=0)))
