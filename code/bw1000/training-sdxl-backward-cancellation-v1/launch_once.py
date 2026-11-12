from pathlib import Path
import base64,hashlib,json,subprocess,uuid
D=Path(__file__).resolve().parents[1];B=D.parent;G=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test');claim=json.loads((D/'source-publication-receipt.json').read_bytes())['commit'];request=uuid.uuid4().hex

def blob(path):return subprocess.check_output(['git','-c','safe.directory='+G.as_posix(),'-C',str(G),'show',claim+':'+path])
task=json.loads(blob('coordination/tasks/training-sdxl-backward-cancellation-v1.json'));assert task['GPU_GO'] and task['GPU_cards_max']==1 and task['raw_payload_cross_host_transfer_bytes']==0;sources={}
for name,pin in task['source_SHA256'].items():
 raw=blob(task['source']+'/'+name);assert hashlib.sha256(raw).hexdigest()==pin;sources[name]=raw
files={name:dict(b64=base64.b64encode(sources[name]).decode(),SHA256=task['source_SHA256'][name]) for name in task['remote_files']}
payload=('CLAIM='+repr(claim)+'\nREQUEST='+repr(request)+'\nFILES='+repr(files)+'\n').encode()+sources['receive_once.py']
pre=(B/'i03-deep-c4-history-staging-v1/read_source_once.py').read_text(encoding='utf8');ns={'__file__':str(D/'reuse.py')};exec(compile(pre.split('REMOTE=',1)[0],'existing_cluster_ssh','exec'),ns)
with (D/'launch-intent.json').open('x') as f:json.dump(dict(request_id=request,claim_commit=claim,payload_SHA256=hashlib.sha256(payload).hexdigest(),exec_request_attempted=True,remote_exec_acknowledged=None,GPU_cards=1,whole_seconds=1200,new_SSH_service=False,new_poller_service=False,replay_allowed=False,scope='Unique additive LoRA-B native backward/cancellation diagnostic; unchanged original replay/math/optimizer/update gate and no tensor transfer. Timeout/unknown result requires exact request/intent/PID/startticks/Slurm reconciliation, never replay.'),f,indent=2)
try:r=subprocess.run(ns['ssh'],input=payload,capture_output=True,timeout=60);rc=r.returncode;stdout=r.stdout;stderr=r.stderr;timed_out=False
except subprocess.TimeoutExpired as error:rc=None;stdout=error.stdout or b'';stderr=error.stderr or b'';timed_out=True
for name,raw in [('original.stdout',stdout),('original.stderr',stderr)]:
 with (D/name).open('xb') as f:f.write(raw)
documents=[]
for line in stdout.splitlines():
 try:documents.append(json.loads(line))
 except (ValueError,UnicodeDecodeError):pass
acks=[v for v in documents if v.get('claim_commit')==claim and v.get('request_id')==request and 'owned_controller' in v]
out=dict(request_id=request,claim_commit=claim,exec_request_attempted=True,remote_exec_acknowledged=bool(acks),transport_rc=rc,transport_timeout=timed_out,remote_outcome='UNKNOWN' if timed_out or rc!=0 else 'ORIGINAL_BOOTSTRAP_RETURNED',stdout_SHA256=hashlib.sha256(stdout).hexdigest(),stderr_SHA256=hashlib.sha256(stderr).hexdigest(),replay_allowed=False)
if acks:assert len(acks)==1;out['value']=acks[0]
with (D/'transport-outcome.json').open('x') as f:json.dump(out,f,indent=2)
value=out.get('value',{});owner=value.get('owned_controller',{});print(json.dumps(dict(request_id=request,transport_rc=rc,remote_exec_acknowledged=bool(acks),owned_controller={k:owner.get(k) for k in ('PID','PGID','startticks','host','epoch','deadline')},controller_ready=value.get('controller_ready') is not None,controller_returncode=value.get('controller_returncode'),GPU_execution_status=value.get('GPU_execution_status'),replay_allowed=False)))
