"""Independently read the actual original/new job handles; never submit or cancel."""
from pathlib import Path
import hashlib,json,re,subprocess,sys,time
D=Path(__file__).resolve().parent;B=D.parent
tag=sys.argv[1];assert re.fullmatch('[a-z0-9-]{1,30}',tag)
recovery=json.loads((D/'recovery-transport-outcome.json').read_bytes());v=recovery['value']
assert recovery['remote_outcome']=='ORIGINAL_RETURNED' and recovery['transport_rc']==0 and v['sbatch_rc']==0
job=v['job_id'];assert job=='1103563'
remote=('CLAIM='+repr(recovery['claim'])+'\nREQUEST='+repr(recovery['request_id'])+'\nJOB='+repr(job)+'\n').encode()+br'''
from pathlib import Path
import datetime,hashlib,json,os,subprocess
W=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-extra40-resource-expansion-v1')
def q(args):
 r=subprocess.run(args,capture_output=True,text=True,timeout=20)
 assert r.returncode==0 and len(r.stdout.encode())<4*1024**2
 return r.stdout
def fields(raw):return {x.split('=',1)[0]:x.split('=',1)[1] for x in raw.split() if '=' in x}
files={}
for n in ('recovery-intent.json','intent.json','submission-attempt.json','sbatch-result.json','submitted-job.json','outcome.json'):
 p=W/n;assert p.is_file() and p.stat().st_size<2*1024**2;b=p.read_bytes();files[n]=dict(bytes=len(b),SHA256=hashlib.sha256(b).hexdigest(),value=json.loads(b))
assert files['recovery-intent.json']['value']['original_request_id']=='42f2a1e31ffc476d8b03cf27082a79c2'
assert files['recovery-intent.json']['value']['recovery_claim']==CLAIM and files['recovery-intent.json']['value']['recovery_request_id']==REQUEST
assert files['submitted-job.json']['value']==dict(job_id=JOB,request_id=REQUEST,claim=CLAIM)
old=fields(q(['scontrol','show','job','-o','1087656']));new=fields(q(['scontrol','show','job','-o',JOB]))
assert old['JobState']=='RUNNING' and old['NumNodes']=='40' and 'gres/dcu=320' in old['AllocTRES']
assert new['JobName']=='training-extra40-long-20261112-v1' and new['Comment']=='extra40-'+REQUEST
assert new['TimeLimit']=='30-00:00:00' and new['Account']=='aclsylqidf'
keys=['JobId','JobName','JobState','Reason','NumNodes','NumCPUs','ReqTRES','AllocTRES','TresPerNode','TimeLimit','StartTime','EndTime','NodeList','Partition','Account','QOS','Comment']
result=dict(observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),old_parent={k:old.get(k) for k in keys},
 new_parent={k:new.get(k) for k in keys},allocation_confirmed=False,confirmed_total_nodes=40,confirmed_total_allocated_DCU=320,
 original_files=files,new_sbatch_submissions=0,old_jobs_cancelled=0,GPU_busy_measured=False)
old_nodes=q(['scontrol','show','hostnames',old['NodeList']]).split();assert len(old_nodes)==len(set(old_nodes))==40
result['old_nodes']=old_nodes
queue=q(['squeue','-h','-u',str(os.getuid()),'-o','%i|%100j|%T|%D|%l|%R'])
result['own_queue']=[x.split('|') for x in queue.splitlines()]
result['new_job_start_estimate']=q(['squeue','--start','-h','-j',JOB,'-o','%i|%T|%S|%R']).strip()
if new['JobState']=='RUNNING':
 nodes=q(['scontrol','show','hostnames',new['NodeList']]).split()
 assert len(nodes)==len(set(nodes))==40 and not set(nodes)&set(old_nodes) and 'gres/dcu=320' in new['AllocTRES']
 node_rows=[]
 for line in q(['scontrol','show','node','-o',','.join(nodes)]).splitlines():
  p=fields(line);node_rows.append({k:p.get(k) for k in ['NodeName','State','CPUTot','CPUAlloc','RealMemory','AllocMem','Gres','CfgTRES','AllocTRES','Partitions']})
 assert len(node_rows)==40 and {p['NodeName'] for p in node_rows}==set(nodes)
 assert all('gres/dcu=8' in p['CfgTRES'] and 'gres/dcu=8' in p['AllocTRES'] for p in node_rows)
 result.update(new_nodes=nodes,new_node_metadata=node_rows,allocation_confirmed=True,confirmed_total_nodes=80,confirmed_total_allocated_DCU=640)
print(json.dumps(result))
'''
ns={'__file__':str(D/'reuse.py')};s=(B/'i03-deep-c4-history-staging-v1/read_source_once.py').read_text(encoding='utf8');exec(compile(s.split('REMOTE=',1)[0],'existing_cluster_SSH','exec'),ns)
start=time.monotonic();r=subprocess.run(ns['ssh'],input=remote,capture_output=True,timeout=90)
for n,b in [('stdout.private.json',r.stdout),('stderr.private.txt',r.stderr)]:
 with (D/(tag+'-allocation-'+n)).open('xb') as f:f.write(b)
result=dict(transport_rc=r.returncode,RTT_seconds=time.monotonic()-start,request_id=recovery['request_id'],job_id=job,
 stdout_SHA256=hashlib.sha256(r.stdout).hexdigest(),stderr_SHA256=hashlib.sha256(r.stderr).hexdigest(),new_sbatch_submissions=0)
if r.returncode==0:result['value']=json.loads(r.stdout)
with (D/(tag+'-allocation-result.json')).open('x') as f:json.dump(result,f,indent=2)
v=result.get('value',{});print(json.dumps(dict(transport_rc=r.returncode,new_parent=v.get('new_parent'),allocation_confirmed=v.get('allocation_confirmed'),
 confirmed_total_nodes=v.get('confirmed_total_nodes'),confirmed_total_allocated_DCU=v.get('confirmed_total_allocated_DCU'),new_job_start_estimate=v.get('new_job_start_estimate'),new_sbatch_submissions=0)))
