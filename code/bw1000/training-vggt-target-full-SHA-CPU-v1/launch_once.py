"""Invoke the one published independent target read through existing ordinary SSH."""
from pathlib import Path
import hashlib,json,subprocess,uuid
D=Path(__file__).resolve().parent;G=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test');old=D.parent/'training-vggt-original-public-model-v1'
claim=json.loads((D/'source-publication-receipt.json').read_bytes())['commit'];git=['git','-c','safe.directory='+G.as_posix(),'-C',str(G)]
pre=json.loads((D/'preflight-result.json').read_bytes());assert pre['transport_rc']==0 and pre['value']['passed'] and pre['value']['namespace_absent']
task=json.loads(subprocess.check_output(git+['show',claim+':coordination/tasks/training-vggt-target-full-SHA-CPU-v1.json']));assert task['CPU_GO'] and task['actual_CPU_launches']==0 and not task['GPU_GO']
code=subprocess.check_output(git+['show',claim+':'+task['source']+'/verify_target_remote.py']);assert hashlib.sha256(code).hexdigest()==task['source_SHA256']['verify_target_remote.py']
original=json.loads((old/'transfer-outcome.json').read_bytes());assert original['passed'] and original['source_rc']==original['destination_rc']==0
request=uuid.uuid4().hex;payload=('CLAIM='+repr(claim)+'\nREQUEST='+repr(request)+'\nORIGINAL='+repr(original)+'\n').encode()+code
with (D/'launch-intent.json').open('x') as f:json.dump(dict(claim=claim,request_id=request,payload_SHA256=hashlib.sha256(payload).hexdigest(),exec_request_attempted=True,replay_allowed=False,GPU_cards=0),f,indent=2)
source=(D.parent/'i03-deep-c4-history-staging-v1/read_source_once.py').read_text(encoding='utf8');ns={'__file__':str(D/'reuse.py')};exec(compile(source.split('REMOTE=',1)[0],'existing_ordinary_cluster_transport','exec'),ns)
try:r=subprocess.run(ns['ssh'],input=payload,capture_output=True,timeout=330);rc=r.returncode;stdout=r.stdout;stderr=r.stderr;unknown=rc==255
except subprocess.TimeoutExpired as e:rc=None;stdout=e.stdout or b'';stderr=e.stderr or b'';unknown=True
for name,raw in [('original-stdout.private.jsonl',stdout),('original-stderr.private.txt',stderr)]:
 with (D/name).open('xb') as f:f.write(raw)
docs=[]
for line in stdout.splitlines():
 try:docs.append(json.loads(line))
 except ValueError:pass
acks=[v for v in docs if v.get('stage')=='REMOTE_HASH_INTENT' and v.get('request_id')==request];final=[v for v in docs if v.get('stage')=='REMOTE_HASH_TERMINAL' and v.get('request_id')==request]
out=dict(claim=claim,request_id=request,transport_rc=rc,remote_exec_acknowledged=bool(acks),remote_outcome='ORIGINAL_RETURNED' if final and not unknown else 'UNKNOWN',stdout_SHA256=hashlib.sha256(stdout).hexdigest(),stderr_SHA256=hashlib.sha256(stderr).hexdigest(),replay_allowed=False)
if final:assert len(final)==1;out['value']=final[0]['result']
with (D/'transport-outcome.json').open('x') as f:json.dump(out,f,indent=2)
v=out.get('value',{});print(json.dumps(dict(transport_rc=rc,remote_exec_acknowledged=bool(acks),remote_outcome=out['remote_outcome'],passed=v.get('passed'),file_bytes_read=v.get('file_bytes_read'),hash_seconds=v.get('hash_seconds'),GPU_GO=False)))
