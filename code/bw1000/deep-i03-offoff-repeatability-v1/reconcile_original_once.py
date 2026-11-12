"""Read only the original request namespace and exact Slurm labels; never replay."""
from pathlib import Path
import hashlib,json,subprocess,time
D=Path(__file__).resolve().parent
intent=json.loads((D/'launch-intent.json').read_bytes())
remote=('REQUEST='+repr(intent['request_id'])+'\nCOMMIT='+repr(intent['claim_commit'])+'\n').encode()+b'''
from pathlib import Path
import datetime,hashlib,json,subprocess
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results')
label='i03-deep-offoff-repeatability-v1';W=R/(label+'-controller')
def query(args):
 r=subprocess.run(args,capture_output=True,text=True,timeout=15);assert r.returncode==0 and len(r.stdout)<1024**2;return r.stdout
files={}
for name in ('intent.json','gpu-intent.json','owned-srun.json','before.json','post.json','outcome.json'):
 p=W/name
 if not p.is_file():files[name]=dict(exists=False);continue
 raw=p.read_bytes();assert len(raw)<1024**2
 value=json.loads(raw)
 if name=='intent.json':assert value['request_id']==REQUEST and value['claim_commit']==COMMIT
 files[name]=dict(exists=True,bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest(),value=value)
rows=[]
for line in query(['sacct','-n','-P','-j','1087656','--format=JobIDRaw,JobName%100,State%50,ExitCode,ElapsedRaw,NodeList%500']).splitlines():
 fields=[v.strip() for v in line.split('|')]
 if len(fields)>=6 and fields[1] in (label,label+'-pre',label+'-post'):rows.append(dict(zip(['step','label','state','exit_code','elapsed_seconds','node'],fields[:6])))
active=query(['squeue','--steps','-h','-j','1087656','-w','f12r3n12','-o','%i|%j|%N']).splitlines()
print(json.dumps(dict(observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),request_id=REQUEST,
 namespace_exists=W.exists(),private_results_exists=(R/(label+'-private')).exists(),files=files,accounting_columns=['step','label','state','exit_code','elapsed_seconds','node'],
 exact_label_accounting=rows,target_active_steps=active,replay_allowed=False)))
'''
source=(D.parent/'i03-deep-c4-history-staging-v1/read_source_once.py').read_text(encoding='utf8');ns={'__file__':str(D/'reuse.py')};exec(compile(source.split('REMOTE=',1)[0],'existing_ordinary_cluster_transport','exec'),ns)
start=time.monotonic();r=subprocess.run(ns['ssh'],input=remote,capture_output=True,timeout=90)
result=dict(request_id=intent['request_id'],transport_rc=r.returncode,RTT_seconds=time.monotonic()-start,payload_SHA256=hashlib.sha256(remote).hexdigest(),stdout_SHA256=hashlib.sha256(r.stdout).hexdigest(),stderr_SHA256=hashlib.sha256(r.stderr).hexdigest(),replay_allowed=False)
if r.returncode==0:result['value']=json.loads(r.stdout)
for n,v in [('reconcile-first.stdout.private.json',r.stdout),('reconcile-first.stderr.private.txt',r.stderr)]:
 with (D/n).open('xb') as f:f.write(v)
with (D/'reconcile-first-result.json').open('x') as f:json.dump(result,f,indent=2)
v=result.get('value',{});print(json.dumps(dict(transport_rc=r.returncode,namespace_exists=v.get('namespace_exists'),private_results_exists=v.get('private_results_exists'),accounting=v.get('exact_label_accounting'),active_steps=v.get('target_active_steps'),replay_allowed=False)))
