"""Send the one published OFF/OFF request; never replay an unknown outcome."""
from pathlib import Path
import base64,hashlib,json,subprocess,time,uuid
D=Path(__file__).resolve().parent;G=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test')
commit=json.loads((D/'source-publication-receipt.json').read_bytes())['commit'];pre=json.loads((D/'preflight-result.json').read_bytes())
assert pre['commit']==commit and pre['transport_rc']==0 and pre['value']['passed']
# The actual dispatch repeats Slurm dedup and a fresh Node boot/monotonic/PCI/KFD/port lease.
git=['git','-c','safe.directory='+G.as_posix(),'-C',str(G)]
task=json.loads(subprocess.check_output(git+['show',commit+':coordination/tasks/deep-i03-offoff-repeatability-v1.json']));claim=task['claim'];files={}
assert task['actual_GPU_launches']==0 and claim['gpu_step_count_max']==1 and task['GPU_GO']
for name in claim['files_SHA256']:
 raw=subprocess.check_output(git+['show',commit+':'+task['source']+'/'+name]);assert hashlib.sha256(raw).hexdigest()==claim['files_SHA256'][name];files[name]=base64.b64encode(raw).decode()
wire=json.dumps(dict(claim=claim,claim_sha256=task['claim_SHA256'],files=files),separators=(',',':')).encode();assert len(wire)<1024**2
dispatcher=base64.b64decode(files['pilot_dispatch_remote.py'],validate=True)
request=uuid.uuid4().hex;payload=('WIRE='+repr(base64.b64encode(wire).decode())+'\nCLAIM='+repr(commit)+'\nREQUEST='+repr(request)+'\n').encode()+dispatcher
with (D/'launch-intent.json').open('x') as f:json.dump(dict(claim_commit=commit,request_id=request,payload_SHA256=hashlib.sha256(payload).hexdigest(),exec_request_attempted=True,replay_allowed=False,GPU_step_max=1,GPU_cards=8),f,indent=2)
source=(D.parent/'i03-deep-c4-history-staging-v1/read_source_once.py').read_text(encoding='utf8');ns={'__file__':str(D/'reuse.py')};exec(compile(source.split('REMOTE=',1)[0],'existing_ordinary_cluster_transport','exec'),ns)
try:r=subprocess.run(ns['ssh'],input=payload,capture_output=True,timeout=7600);rc=r.returncode;stdout=r.stdout;stderr=r.stderr;unknown=rc==255
except subprocess.TimeoutExpired as failure:rc=None;stdout=failure.stdout or b'';stderr=failure.stderr or b'';unknown=True
for n,v in [('launch.stdout.private.jsonl',stdout),('launch.stderr.private.txt',stderr)]:
 with (D/n).open('xb') as f:f.write(v)
docs=[]
for line in stdout.splitlines():
 try:docs.append(json.loads(line))
 except ValueError:pass
acks=[v for v in docs if v.get('stage')=='REMOTE_INTENT_CREATED' and v.get('request_id')==request]
final=[v for v in docs if v.get('stage')=='CONTROLLER_TERMINAL' and v.get('request_id')==request]
out=dict(claim_commit=commit,request_id=request,transport_rc=rc,remote_exec_acknowledged=bool(acks),remote_outcome='ORIGINAL_RETURNED' if final and not unknown else 'UNKNOWN',stdout_SHA256=hashlib.sha256(stdout).hexdigest(),stderr_SHA256=hashlib.sha256(stderr).hexdigest(),replay_allowed=False)
if final:assert len(final)==1;out['value']=final[0]['result']
with (D/'launch-outcome.json').open('x') as f:json.dump(out,f,indent=2)
v=out.get('value',{});print(json.dumps(dict(transport_rc=rc,remote_exec_acknowledged=bool(acks),remote_outcome=out['remote_outcome'],passed=v.get('passed'),actual_GPU_srun_launched=v.get('actual_GPU_srun_launched'),returncode=v.get('returncode'),accounting=v.get('original_GPU_accounting'),post_step=(v.get('post') or {}).get('step'))))
