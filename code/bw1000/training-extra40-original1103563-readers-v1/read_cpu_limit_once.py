"""Diagnose actual inherited group CPU admission limit; no job updates."""
from pathlib import Path
import hashlib,json,subprocess,time
D=Path(__file__).resolve().parent;B=D.parent
remote=br'''
import datetime,json,subprocess
def q(args):
 r=subprocess.run(args,capture_output=True,text=True,timeout=20);assert len(r.stdout.encode())<=1024**2
 return dict(rc=r.returncode,stdout=r.stdout,stderr=r.stderr)
print(json.dumps(dict(observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 account_associations=q(['sacctmgr','-n','-P','show','assoc','where','account=aclsylqidf','format=ID,Cluster,Account,User,ParentName,GrpTRES%120,GrpJobs,MaxJobs,MaxWall']),
 account=q(['sacctmgr','-n','-P','show','account','where','name=aclsylqidf','format=Account,Description,Organization,ParentName']),
 jobs={n:q(['scontrol','show','job','-o',n]) for n in ['1087656','1103563']},
 original_node=q(['scontrol','show','node','-o','f11r4n09']),new_sbatch_submissions=0,job_updates=0)))
'''
ns={'__file__':str(D/'reuse.py')};s=(B/'i03-deep-c4-history-staging-v1/read_source_once.py').read_text(encoding='utf8');exec(compile(s.split('REMOTE=',1)[0],'existing_cluster_SSH','exec'),ns)
start=time.monotonic();r=subprocess.run(ns['ssh'],input=remote,capture_output=True,timeout=100)
for n,b in [('cpu-limit.private.json',r.stdout),('cpu-limit-stderr.private.txt',r.stderr)]:
 with (D/n).open('xb') as f:f.write(b)
receipt=dict(transport_rc=r.returncode,RTT_seconds=time.monotonic()-start,stdout_SHA256=hashlib.sha256(r.stdout).hexdigest(),stderr_SHA256=hashlib.sha256(r.stderr).hexdigest(),new_sbatch_submissions=0)
if r.returncode==0:
 v=json.loads(r.stdout);receipt['observed_UTC']=v['observed_UTC'];receipt['associations']=v['account_associations'];receipt['account']=v['account'];receipt['jobs']={}
 for n,q in v['jobs'].items():
  p={x.split('=',1)[0]:x.split('=',1)[1] for x in q['stdout'].split() if '=' in x}
  receipt['jobs'][n]={k:p.get(k) for k in ['JobId','JobState','Reason','NumNodes','NumCPUs','NumTasks','CPUs/Task','ReqTRES','AllocTRES','OverSubscribe','Shared','TimeLimit','NodeList']}
 p={x.split('=',1)[0]:x.split('=',1)[1] for x in v['original_node']['stdout'].split() if '=' in x}
 receipt['original_node']={k:p.get(k) for k in ['NodeName','CPUTot','CPUAlloc','RealMemory','AllocTRES','CfgTRES','State']}
with (D/'cpu-limit-read-result.json').open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt))
