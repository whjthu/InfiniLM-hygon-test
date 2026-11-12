from pathlib import Path
import base64,hashlib,json,subprocess,uuid
D=Path(__file__).resolve().parents[1];B=D.parent;G=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test');A=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline');claim=json.loads((D/'source-publication-receipt.json').read_bytes())['commit'];request=uuid.uuid4().hex;git=['git','-c','safe.directory='+G.as_posix(),'-C',str(G)]
def blob(n):return subprocess.check_output(git+['show',claim+':'+n])
t=json.loads(blob('coordination/tasks/training-hymotion-A100-transformer-meta-CPU-v1.json'));assert t['CPU_GO'] and not t['GPU_GO'];source={}
for n,pin in t['source_SHA256'].items():
 raw=blob(t['source']+'/'+n);assert hashlib.sha256(raw).hexdigest()==pin;source[n]=raw
files={n:dict(b64=base64.b64encode(source[n]).decode(),bytes=len(source[n]),SHA256=t['source_SHA256'][n]) for n in t['remote_files']};payload=('CLAIM='+repr(claim)+'\nREQUEST='+repr(request)+'\nFILES='+repr(files)+'\n').encode()+source['receive_once.py']
with (D/'launch-intent.json').open('x') as f:json.dump(dict(claim=claim,request_id=request,payload_SHA256=hashlib.sha256(payload).hexdigest(),exec_request_attempted=True,remote_exec_acknowledged=None,GPU_cards=0,weights_or_cache_tensor_bytes_read=0),f,indent=2)
ssh=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(A/'ssh_config'),'a100-26','python3 -B -u -']
try:r=subprocess.run(ssh,input=payload,capture_output=True,timeout=270);rc=r.returncode;out=r.stdout;err=r.stderr
except subprocess.TimeoutExpired as e:rc=None;out=e.stdout or b'';err=e.stderr or b''
for n,raw in [('original.stdout.private.json',out),('original.stderr.private.txt',err)]:
 with (D/n).open('xb') as f:f.write(raw)
docs=[]
for line in out.splitlines():
 try:docs.append(json.loads(line))
 except (ValueError,UnicodeDecodeError):pass
acks=[x for x in docs if x.get('stage')=='REMOTE_INTENT_CREATED' and x.get('claim')==claim and x.get('request_id')==request];ends=[x for x in docs if x.get('claim')==claim and x.get('request_id')==request and 'returncode' in x];value=ends[0] if len(ends)==1 else None
record=dict(claim=claim,request_id=request,transport_rc=rc,remote_exec_acknowledged=bool(acks),value=value,remote_outcome='UNKNOWN' if rc in (None,255) else 'ORIGINAL_RETURNED',stdout_SHA256=hashlib.sha256(out).hexdigest(),stderr_SHA256=hashlib.sha256(err).hexdigest(),replay_allowed=False)
with (D/'transport-outcome.json').open('x') as f:json.dump(record,f,indent=2)
q=value.get('CPU-result.json',{}).get('value',{}) if value else {};print(json.dumps(dict(transport_rc=rc,remote_exec_acknowledged=bool(acks),passed=value.get('passed') if value else None,phase=q.get('phase'),error_type=q.get('error_type'),trainable_tensors=q.get('trainable_tensors'),trainable_parameters=q.get('trainable_parameters'),CUDA_initialized=q.get('CUDA_initialized'),Model_GO=False)))
