from pathlib import Path
import base64,hashlib,json,subprocess,uuid
D=Path(__file__).resolve().parents[1];B=D.parent;G=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test');claim=json.loads((D/'source-publication-receipt.json').read_bytes())['commit'];request=uuid.uuid4().hex

def blob(path):return subprocess.check_output(['git','-c','safe.directory='+G.as_posix(),'-C',str(G),'show',claim+':'+path])
task=json.loads(blob('coordination/tasks/training-sdxl-backward-audit-CPU-v1.json'));assert task['CPU_GO'] and task['GPU_GO'] is False;sources={}
for name,pin in task['source_SHA256'].items():
 raw=blob(task['source']+'/'+name);assert hashlib.sha256(raw).hexdigest()==pin;sources[name]=raw
files={n:dict(b64=base64.b64encode(sources[n]).decode(),bytes=len(sources[n]),SHA256=task['source_SHA256'][n]) for n in task['remote_files']}
payload=('CLAIM='+repr(claim)+'\nREQUEST='+repr(request)+'\nFILES='+repr(files)+''+'\n').encode()+sources['receive_once.py']
ns={'__file__':str(D/'reuse.py')};pre=(B/'i03-deep-c4-history-staging-v1/read_source_once.py').read_text(encoding='utf8');exec(compile(pre.split('REMOTE=',1)[0],'existing_cluster_ssh','exec'),ns)
with (D/'launch-intent.json').open('x') as f:json.dump(dict(request_id=request,claim=claim,payload_SHA256=hashlib.sha256(payload).hexdigest(),exec_request_attempted=True,remote_exec_acknowledged=None,GPU_cards=0,raw_tensor_transfer_bytes=0,scope='Unique published CPU-only LoRA backward observer subset5 from unified330 framework; actual9 realTorch/PEFT tiny CPU tests only. No production data or model weight values. Unknown execution is reconciled, never replayed.'),f,indent=2)
try:r=subprocess.run(ns['ssh'],input=payload,capture_output=True,timeout=330);rc=r.returncode;stdout=r.stdout;stderr=r.stderr;timed_out=False
except subprocess.TimeoutExpired as error:rc=None;stdout=error.stdout or b'';stderr=error.stderr or b'';timed_out=True
for name,raw in [('original.stdout',stdout),('original.stderr',stderr)]:
 with (D/name).open('xb') as f:f.write(raw)
documents=[]
for line in stdout.splitlines():
 try:documents.append(json.loads(line))
 except (ValueError,UnicodeDecodeError):pass
acks=[x for x in documents if x.get('stage')=='REMOTE_INTENT_CREATED' and x.get('request_id')==request and x.get('claim')==claim];outcomes=[x for x in documents if x.get('claim')==claim and 'accounting' in x]
out=dict(request_id=request,claim=claim,exec_request_attempted=True,remote_exec_acknowledged=bool(acks),transport_rc=rc,transport_timeout=timed_out,remote_outcome='UNKNOWN' if timed_out or rc in (None,255) else 'ORIGINAL_RETURNED',stdout_SHA256=hashlib.sha256(stdout).hexdigest(),stderr_SHA256=hashlib.sha256(stderr).hexdigest(),replay_allowed=False)
if acks:assert len(acks)==1;out['ack']=acks[0]
if outcomes:assert len(outcomes)==1;out['value']=outcomes[0]
with (D/'transport-outcome.json').open('x') as f:json.dump(out,f,indent=2)
v=out.get('value',{});result=v.get('CPU-result.json',{}).get('value',{});print(json.dumps(dict(transport_rc=rc,remote_exec_acknowledged=bool(acks),passed=v.get('passed'),accounting=v.get('accounting'),tests=result.get('tests'),phase=result.get('phase'),error_type=result.get('error_type'),CUDA_initialized=result.get('CUDA_initialized'),Model_GO=False)))
