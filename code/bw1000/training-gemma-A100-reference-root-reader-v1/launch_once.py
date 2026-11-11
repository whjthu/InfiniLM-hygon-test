"""Execute this published original-host CPU reader once; never launch a model."""
from pathlib import Path
import base64,hashlib,json,subprocess,uuid
B=Path(__file__).resolve().parents[2];D=B/'training-gemma-A100-reference-root-reader-v1';G=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test');claim=json.loads((D/'source-publication-receipt.json').read_bytes())['commit'];git=['git','-c','safe.directory='+G.as_posix(),'-C',str(G)]
def blob(name):return subprocess.check_output(git+['show',claim+':'+name])
task=json.loads(blob('coordination/tasks/training-gemma-A100-reference-root-reader-v1.json'));assert task['CPU_GO'] and not task['GPU_GO'];sources={}
for name,pin in task['source_SHA256'].items():
 raw=blob(task['source']+'/'+name);assert hashlib.sha256(raw).hexdigest()==pin;sources[name]=raw
files={n:dict(b64=base64.b64encode(sources[n]).decode(),bytes=len(sources[n]),SHA256=task['source_SHA256'][n]) for n in task['remote_files']};request=uuid.uuid4().hex;code=('CLAIM='+repr(claim)+'\nREQUEST='+repr(request)+'\nFILES='+repr(files)+'\n').encode()+sources['receive_once.py']
with (D/'launch-intent.json').open('x') as f:json.dump(dict(claim=claim,request_id=request,payload_SHA256=hashlib.sha256(code).hexdigest(),exec_request_attempted=True,replay_allowed=False,GPU_cards=0),f,indent=2)
A=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline');a=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(A/'ssh_config'),'a100-26','python3 -B -u -']
try:r=subprocess.run(a,input=code,capture_output=True,timeout=720);rc=r.returncode;stdout=r.stdout;stderr=r.stderr;unknown=rc==255
except subprocess.TimeoutExpired as e:rc=None;stdout=e.stdout or b'';stderr=e.stderr or b'';unknown=True
(D/'original-stdout.private.json').write_bytes(stdout);(D/'original-stderr.private.txt').write_bytes(stderr);result=dict(claim=claim,request_id=request,transport_rc=rc,remote_outcome='UNKNOWN' if unknown else 'ORIGINAL_RETURNED',stdout_SHA256=hashlib.sha256(stdout).hexdigest(),replay_allowed=False)
if not unknown and stdout:result['value']=json.loads(stdout)
(D/'transport-outcome.json').write_bytes((json.dumps(result,indent=2)+'\n').encode());v=result.get('value',{});r=v.get('independent-reference.json',{}).get('value',{});failure=v.get('failure.json',{}).get('value',{});print(json.dumps(dict(transport_rc=rc,passed=v.get('passed'),reader_returncode=v.get('reader_returncode'),artifacts=r.get('tensor_artifacts'),native25_ranks=r.get('original25_completed_ranks'),error_type=failure.get('error_type',v.get('error_type')),validated_artifacts=v.get('audit-progress.json',{}).get('value',{}).get('validated_artifacts'),wall_seconds=v.get('wall_seconds'))))
