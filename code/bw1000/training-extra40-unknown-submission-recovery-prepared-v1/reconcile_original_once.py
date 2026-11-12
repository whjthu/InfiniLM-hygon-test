"""Readonly reconcile only the original extra40 request and its exact Slurm label."""
from pathlib import Path
import hashlib,json,subprocess,time
D=Path(__file__).resolve().parent;B=D.parent
intent=json.loads((D/'launch-intent.json').read_bytes())
remote=('CLAIM='+repr(intent['claim'])+'\nREQUEST='+repr(intent['request_id'])+'\n').encode()+br'''
from pathlib import Path
import datetime,hashlib,json,os,socket,subprocess
W=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-extra40-resource-expansion-v1');LABEL='training-extra40-long-20261112-v1'
def q(args):
 r=subprocess.run(args,capture_output=True,text=True,timeout=20);assert len(r.stdout.encode())<2*1024**2
 return dict(rc=r.returncode,stdout=r.stdout,stderr=r.stderr)
files={}
for n in ('intent.json','submission-attempt.json','sbatch-result.json','submitted-job.json','outcome.json'):
 p=W/n
 if p.is_file():
  assert p.stat().st_size<2*1024**2;b=p.read_bytes();v=json.loads(b)
  if 'request_id' in v:assert v['request_id']==REQUEST
  if 'claim' in v:assert v['claim']==CLAIM
  files[n]=dict(bytes=len(b),SHA256=hashlib.sha256(b).hexdigest(),value=v)
queue=q(['squeue','-h','-u',str(os.getuid()),'-o','%i|%100j|%T|%D|%l|%N'])
assert queue['rc']==0
rows=[r.split('|') for r in queue['stdout'].splitlines() if len(r.split('|'))>=2 and r.split('|')[1].strip()==LABEL]
history=q(['sacct','-X','-n','-P','-u',str(os.getuid()),'-S','2026-11-12','--format=JobIDRaw,JobName%100,State,ExitCode,AllocNodes,NodeList'])
assert history['rc']==0
hist=[r.split('|') for r in history['stdout'].splitlines() if len(r.split('|'))>=2 and r.split('|')[1].strip()==LABEL]
old=q(['scontrol','show','job','-o','1087656']);assert old['rc']==0
field={p.split('=',1)[0]:p.split('=',1)[1] for p in old['stdout'].split() if '=' in p}
print(json.dumps(dict(observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),request_id=REQUEST,claim=CLAIM,
 namespace_exists=W.exists(),files=files,matching_queue=rows,matching_accounting=hist,
 old_parent={k:field.get(k) for k in ['JobId','JobState','NumNodes','AllocTRES','NodeList','TimeLimit','EndTime']},
 original_owner_host_matches=bool(files.get('intent.json',{}).get('value',{}).get('owner',{}).get('host')==socket.gethostname()),
 new_submission_attempted=False,replay_allowed=False)))
'''
ns={'__file__':str(D/'reuse.py')};s=(B/'i03-deep-c4-history-staging-v1/read_source_once.py').read_text(encoding='utf8');exec(compile(s.split('REMOTE=',1)[0],'existing_cluster_SSH','exec'),ns)
folder=D/'reconciliation';folder.mkdir(exist_ok=True);tag=str(time.time_ns())
start=time.monotonic();r=subprocess.run(ns['ssh'],input=remote,capture_output=True,timeout=75)
for n,b in [('stdout.private.json',r.stdout),('stderr.private.txt',r.stderr)]:
 with (folder/(tag+'.'+n)).open('xb') as f:f.write(b)
result=dict(request_id=intent['request_id'],claim=intent['claim'],transport_rc=r.returncode,RTT_seconds=time.monotonic()-start,
 payload_SHA256=hashlib.sha256(remote).hexdigest(),stdout_SHA256=hashlib.sha256(r.stdout).hexdigest(),stderr_SHA256=hashlib.sha256(r.stderr).hexdigest(),new_submission_attempted=False,replay_allowed=False)
if r.returncode==0:result['value']=json.loads(r.stdout)
with (folder/(tag+'-receipt.json')).open('x') as f:json.dump(result,f,indent=2)
v=result.get('value',{});print(json.dumps(dict(receipt=tag+'-receipt.json',transport_rc=r.returncode,namespace_exists=v.get('namespace_exists'),persistent_files=list(v.get('files',{})),matching_queue=v.get('matching_queue'),matching_accounting=v.get('matching_accounting'),old_parent=v.get('old_parent'),new_submission_attempted=False)))
