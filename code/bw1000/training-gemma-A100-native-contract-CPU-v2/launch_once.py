"""Invoke the published CPU contract once,without launching a native training run."""
from pathlib import Path
import base64,hashlib,json,subprocess,uuid
B=Path(__file__).resolve().parents[2];D=B/'training-gemma-A100-native-contract-CPU-v2';F=B/'hygon-training-integration';G=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test');claim=json.loads((D/'source-publication-receipt.json').read_bytes())['commit'];git=['git','-c','safe.directory='+G.as_posix(),'-C',str(G)]
def blob(name):return subprocess.check_output(git+['show',claim+':'+name])
task=json.loads(blob('coordination/tasks/training-gemma-A100-native-contract-CPU-v2.json'));assert task['CPU_GO'] and not task['GPU_GO'];files={};sources={}
for name,pin in task['source_SHA256'].items():
 raw=blob(task['source']+'/'+name);assert hashlib.sha256(raw).hexdigest()==pin;sources[name]=raw
 if name!='CLAIM.json':files['src/'+name]=dict(b64=base64.b64encode(raw).decode(),bytes=len(raw),SHA256=pin)
request=uuid.uuid4().hex;payload=('CLAIM='+repr(claim)+'\nREQUEST='+repr(request)+'\nFILES='+repr(files)+'\n').encode()+sources['receive_once.py']
with (D/'launch-intent.json').open('x') as f:json.dump(dict(claim=claim,request_id=request,payload_SHA256=hashlib.sha256(payload).hexdigest(),exec_request_attempted=True,replay_allowed=False,GPU_cards=0),f)
A=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline');ssh=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(A/'ssh_config'),'a100-26','python3 -B -u -']
try:r=subprocess.run(ssh,input=payload,capture_output=True,timeout=320);rc=r.returncode;stdout=r.stdout;stderr=r.stderr;unknown=rc==255
except subprocess.TimeoutExpired as e:rc=None;stdout=e.stdout or b'';stderr=e.stderr or b'';unknown=True
(D/'original-stdout.private.json').write_bytes(stdout);(D/'original-stderr.private.txt').write_bytes(stderr);result=dict(claim=claim,request_id=request,transport_rc=rc,remote_outcome='UNKNOWN' if unknown else 'ORIGINAL_RETURNED',stdout_SHA256=hashlib.sha256(stdout).hexdigest(),replay_allowed=False)
if not unknown and stdout:result['value']=json.loads(stdout)
(D/'transport-outcome.json').write_bytes((json.dumps(result,indent=2)+'\n').encode());v=result.get('value',{});c=v.get('CPU_result') or {};print(json.dumps(dict(returncode=rc,passed=v.get('passed'),phase=c.get('phase'),error_type=c.get('error_type',v.get('error_type')),message=c.get('message'),trainable_tensors=c.get('native_trainable_tensors'),trainable_parameters=c.get('native_trainable_parameters'),wall_seconds=v.get('wall_seconds'))))
if not v.get('passed'):raise SystemExit(1)
