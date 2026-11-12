"""Two scheduler test-only requests; no job is submitted or cancelled."""
from pathlib import Path
import hashlib,json,subprocess,time
D=Path(__file__).resolve().parent;B=D.parent
remote=br'''
import datetime,json,subprocess
base=['sbatch','--test-only','--nodes=40','--ntasks=40','--ntasks-per-node=1','--cpus-per-task=64',
 '--gres=dcu:8','--mem=240000M','--time=30-00:00:00','--partition=hx1hdnormal01',
 '--account=aclsylqidf','--qos=user_aclsylqidf','--wrap=/bin/true']
results=[]
for flag in ['--exclusive','--exclusive=user']:
 r=subprocess.run(base+[flag],capture_output=True,text=True,timeout=30)
 assert len(r.stdout)<65536 and len(r.stderr)<65536
 results.append(dict(flag=flag,rc=r.returncode,stdout=r.stdout,stderr=r.stderr))
print(json.dumps(dict(observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),results=results,
 nodes=40,DCU=320,requested_CPU=2560,new_jobs_submitted=0,job_updates=0,old_jobs_cancelled=0)))
'''
ns={'__file__':str(D/'reuse.py')};s=(B/'i03-deep-c4-history-staging-v1/read_source_once.py').read_text(encoding='utf8');exec(compile(s.split('REMOTE=',1)[0],'existing_cluster_SSH','exec'),ns)
start=time.monotonic();r=subprocess.run(ns['ssh'],input=remote,capture_output=True,timeout=75)
for n,b in [('admission-test-only.private.json',r.stdout),('admission-test-only-stderr.private.txt',r.stderr)]:
 with (D/n).open('xb') as f:f.write(b)
result=dict(transport_rc=r.returncode,RTT_seconds=time.monotonic()-start,stdout_SHA256=hashlib.sha256(r.stdout).hexdigest(),stderr_SHA256=hashlib.sha256(r.stderr).hexdigest(),new_jobs_submitted=0)
if r.returncode==0:result['value']=json.loads(r.stdout)
with (D/'admission-test-only-result.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
