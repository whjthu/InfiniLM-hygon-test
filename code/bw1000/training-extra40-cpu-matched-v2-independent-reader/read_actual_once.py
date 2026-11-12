"""Independent original1103563 terminal and actual new1103606 allocation read."""
from pathlib import Path
import hashlib,json,re,subprocess,sys,time
D=Path(__file__).resolve().parent;B=D.parent;tag=sys.argv[1];assert re.fullmatch('[a-z0-9-]{1,30}',tag)
transport=json.loads((D/'transport-outcome.json').read_bytes());v=transport['value']
assert transport['transport_rc']==0 and v['job_id']=='1103606' and v['sbatch_rc']==0
remote=('CLAIM='+repr(transport['claim'])+'\nREQUEST='+repr(transport['request_id'])+'\n').encode()+br'''
from pathlib import Path
import datetime,hashlib,json,os,subprocess
W=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-extra40-cpu-matched-v2')
def q(args):
 r=subprocess.run(args,capture_output=True,text=True,timeout=20);assert r.returncode==0 and len(r.stdout)<4*1024**2
 return r.stdout
def fields(raw):return {x.split('=',1)[0]:x.split('=',1)[1] for x in raw.split() if '=' in x}
files={}
for n in ('intent.json','cancel-attempt.json','cancel-result.json','original1103563-terminal.json','submission-attempt.json','sbatch-result.json','submitted-job.json','outcome.json'):
 p=W/n;assert p.is_file() and p.stat().st_size<2*1024**2;b=p.read_bytes();files[n]=dict(bytes=len(b),SHA256=hashlib.sha256(b).hexdigest(),value=json.loads(b))
assert files['submitted-job.json']['value']==dict(job_id='1103606',claim=CLAIM,request_id=REQUEST)
old_terminal=[r.split('|') for r in q(['sacct','-X','-n','-P','-j','1103563','--format=JobIDRaw,State,ExitCode,AllocNodes,NodeList']).splitlines() if r.split('|')[0]=='1103563']
assert len(old_terminal)==1 and old_terminal[0][1].startswith('CANCELLED') and old_terminal[0][3]=='0'
assert files['original1103563-terminal.json']['value']['row']==old_terminal[0]
old=fields(q(['scontrol','show','job','-o','1087656']));new=fields(q(['scontrol','show','job','-o','1103606']))
assert old['JobState']=='RUNNING' and old['NumNodes']=='40' and 'gres/dcu=320' in old['AllocTRES']
assert new['JobName']=='training-extra40-cpu-matched-v2' and new['Comment']=='extra40-v2-'+REQUEST
assert new['TimeLimit']=='30-00:00:00' and new['OverSubscribe']=='USER' and new['NumCPUs']=='2560'
keys=['JobId','JobName','JobState','Reason','NumNodes','NumCPUs','ReqTRES','AllocTRES','OverSubscribe','TimeLimit','StartTime','EndTime','NodeList','Partition','Account','QOS','Comment']
old_nodes=q(['scontrol','show','hostnames',old['NodeList']]).split();assert len(old_nodes)==len(set(old_nodes))==40
result=dict(observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),original_files=files,
 previous_job_terminal=dict(job_id='1103563',accounting_columns=['JobIDRaw','State','ExitCode','AllocNodes','NodeList'],row=old_terminal[0]),
 old_parent={k:old.get(k) for k in keys},new_parent={k:new.get(k) for k in keys},old_nodes=old_nodes,
 allocation_confirmed=False,confirmed_total_nodes=40,confirmed_total_DCU=320,confirmed_total_CPU=2560,
 new_sbatch_submissions=0,original_parent_cancelled=False,GPU_busy_measured=False)
result['start_estimate']=q(['squeue','--start','-h','-j','1103606','-o','%i|%T|%S|%R']).strip()
if new['JobState']=='RUNNING':
 nodes=q(['scontrol','show','hostnames',new['NodeList']]).split();assert len(nodes)==len(set(nodes))==40 and not set(nodes)&set(old_nodes)
 assert 'gres/dcu=320' in new['AllocTRES'] and 'cpu=2560' in new['AllocTRES']
 node_metadata=[]
 for line in q(['scontrol','show','node','-o',','.join(nodes)]).splitlines():
  p=fields(line);node_metadata.append({k:p.get(k) for k in ['NodeName','State','CPUTot','CPUAlloc','RealMemory','AllocMem','Gres','CfgTRES','AllocTRES','Partitions']})
 assert len(node_metadata)==40 and {p['NodeName'] for p in node_metadata}==set(nodes)
 assert all('gres/dcu=8' in p['CfgTRES'] and 'gres/dcu=8' in p['AllocTRES'] for p in node_metadata)
 result.update(allocation_confirmed=True,new_nodes=nodes,new_node_metadata=node_metadata,confirmed_total_nodes=80,confirmed_total_DCU=640,confirmed_total_CPU=5120)
print(json.dumps(result))
'''
ns={'__file__':str(D/'reuse.py')};s=(B/'i03-deep-c4-history-staging-v1/read_source_once.py').read_text(encoding='utf8');exec(compile(s.split('REMOTE=',1)[0],'existing_cluster_SSH','exec'),ns)
start=time.monotonic();r=subprocess.run(ns['ssh'],input=remote,capture_output=True,timeout=90)
for n,b in [('stdout.private.json',r.stdout),('stderr.private.txt',r.stderr)]:
 with (D/(tag+'-allocation-'+n)).open('xb') as f:f.write(b)
result=dict(transport_rc=r.returncode,RTT_seconds=time.monotonic()-start,job_id='1103606',
 stdout_SHA256=hashlib.sha256(r.stdout).hexdigest(),stderr_SHA256=hashlib.sha256(r.stderr).hexdigest(),new_sbatch_submissions=0)
if r.returncode==0:result['value']=json.loads(r.stdout)
with (D/(tag+'-allocation-result.json')).open('x') as f:json.dump(result,f,indent=2)
v=result.get('value',{});print(json.dumps(dict(transport_rc=r.returncode,new_parent=v.get('new_parent'),allocation_confirmed=v.get('allocation_confirmed'),
 confirmed_total_nodes=v.get('confirmed_total_nodes'),confirmed_total_DCU=v.get('confirmed_total_DCU'),confirmed_total_CPU=v.get('confirmed_total_CPU'),
 start_estimate=v.get('start_estimate'),previous_job_terminal=v.get('previous_job_terminal'),new_sbatch_submissions=0)))
