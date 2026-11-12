"""One ordinary SSH submission of the already-published extra40 resource scope."""
from pathlib import Path
import hashlib,json,subprocess,uuid
D=Path(__file__).resolve().parent;B=D.parent
G=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test')
claim=json.loads((D/'source-publication-receipt.json').read_bytes())['commit']
git=['git','-c','safe.directory='+G.as_posix(),'-C',str(G)]
task=json.loads(subprocess.check_output(git+['show',claim+':coordination/tasks/training-extra40-resource-expansion-v1.json']))
assert task['allocation_GO'] and task['max_sbatch_submissions']==1 and task['actual_sbatch_submissions']==0
source=subprocess.check_output(git+['show',claim+':'+task['source']+'/submit_extra40.py'])
assert hashlib.sha256(source).hexdigest()==task['source_SHA256']['submit_extra40.py']
request=uuid.uuid4().hex;payload=('CLAIM='+repr(claim)+'\nREQUEST='+repr(request)+'\n').encode()+source
with (D/'launch-intent.json').open('x') as f:json.dump(dict(claim=claim,request_id=request,payload_SHA256=hashlib.sha256(payload).hexdigest(),exec_request_attempted=True,replay_allowed=False),f,indent=2)
ns={'__file__':str(D/'reuse.py')};old=(B/'i03-deep-c4-history-staging-v1/read_source_once.py').read_text(encoding='utf8');exec(compile(old.split('REMOTE=',1)[0],'existing_cluster_SSH','exec'),ns)
try:r=subprocess.run(ns['ssh'],input=payload,capture_output=True,timeout=195);rc=r.returncode;out=r.stdout;err=r.stderr
except subprocess.TimeoutExpired as error:rc=None;out=error.stdout or b'';err=error.stderr or b''
for n,raw in [('submission-stdout.private.jsonl',out),('submission-stderr.private.txt',err)]:
 with (D/n).open('xb') as f:f.write(raw)
docs=[]
for line in out.splitlines():
 try:docs.append(json.loads(line))
 except ValueError:pass
ack=any(v.get('stage')=='SUBMISSION_INTENT_SAVED' and v.get('request_id')==request for v in docs)
final=[v for v in docs if v.get('request_id')==request and 'sbatch_rc' in v]
result=dict(claim=claim,request_id=request,transport_rc=rc,remote_exec_acknowledged=ack,
    stdout_SHA256=hashlib.sha256(out).hexdigest(),stderr_SHA256=hashlib.sha256(err).hexdigest(),
    remote_outcome='UNKNOWN' if rc in (None,255) else 'ORIGINAL_RETURNED',replay_allowed=False)
if final:assert len(final)==1;result['value']=final[0]
with (D/'transport-outcome.json').open('x') as f:json.dump(result,f,indent=2)
v=result.get('value',{})
print(json.dumps(dict(transport_rc=rc,request_id=request,remote_exec_acknowledged=ack,remote_outcome=result['remote_outcome'],
    job_id=v.get('job_id'),new_job=v.get('new_job'),allocation_confirmed=v.get('allocation_confirmed'),
    total_unique_nodes=v.get('total_unique_nodes'),total_Slurm_allocated_DCU=v.get('total_Slurm_allocated_DCU'),replay_allowed=False)))
