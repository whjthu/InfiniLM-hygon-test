"""One bounded draft-only cost residency, with independent before/post KFD probes."""
from pathlib import Path
import json,hashlib,subprocess,os,time,re,datetime,argparse,socket
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');D=Path(__file__).resolve().parent
LABEL='Kimi-C1-long-draft-cost-r324';NODE='a02r3n15';W=R/'results'/(LABEL+'-watch.json')
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def validate_scope(g):
 assert g.get('execution_selected')is True,'Concrete published cost card must be selected'
 assert g['GPU_parent']==910371 and g['GPU_node']==NODE and g['CPU_parent']==910407 and g['CPU_node']=='a14r2n20'
 for k,v in {'cards':1,'target_cards':0,'GPU_max_s':600,'CPU_max_s':900,'warmups':2,'queries':8,'CPUs':8,'mem_MiB':65536}.items():assert type(g[k])is int and g[k]==v
 assert g['context_points']==[1024,8192,32760]and all(type(x)is int for x in g['context_points'])
 assert g['one_attempt']is True and g['profile']is False and g['synthetic_cost_only']is True
 assert re.fullmatch('[0-9a-f]{40}',g['publication_commit']) and re.fullmatch('[0-9a-f]{64}',g['source_manifest_SHA256'])
 return g
def verify_sources(g):
 validate_scope(g);assert sha(D/'SOURCE_MANIFEST.json')==g['source_manifest_SHA256'];m=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());assert all(sha(D/n)==h for n,h in m['files_SHA256'].items())
 c=R/'incoming/Kimi-C1-long-draft-r323';assert sha(c/'SOURCE_MANIFEST.json')==m['consumer_manifest_SHA256'];cm=json.loads((c/'SOURCE_MANIFEST.json').read_bytes());assert all(sha(c/n)==h for n,h in cm['files_SHA256'].items())
 cp=R/'candidates/Kimi-C1-long-draft-CPU-r323/CPU-prepared.json';assert sha(cp)==m['CPU637_receipt_SHA256'];q=json.loads(cp.read_bytes());assert q['completed']and q['passed']and q['actual_step']=='910407.637'
 p=R/'incoming/DFlash2-trained-manifest-r9019.json';assert sha(p)==m['trained_manifest_SHA256'];tm=json.loads(p.read_bytes());assert all(sha(R/'incoming'/n)==h for n,h in tm['incoming_source_SHA256'].items())
 assert sha(R/'incoming/resource_wave_local_r3097_outer.sh')==m['outer_wrapper_SHA256']
 return m
def clean_env():return{k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_','CUDA_','HIP_','ROCR_','GPU_'))or k in ['SLURM_CONF','SLURM_CONF_SERVER']}
def main():
 p=argparse.ArgumentParser();p.add_argument('--execution-go',required=True);a=p.parse_args();g=json.loads(Path(a.execution_go).read_bytes());verify_sources(g);assert not W.exists()
 assert os.environ['SLURM_JOB_ID']=='910407' and socket.gethostname().split('.')[0]=='a14r2n20';env=clean_env()
 for job,node in [(910371,NODE),(910407,'a14r2n20')]:
  raw=subprocess.check_output(['scontrol','show','job',str(job),'-o'],env=env,text=True,timeout=25);assert 'UserId=aclsylqidf('in raw and 'JobState=RUNNING'in raw
  nodes=re.search(r'(?:^| )NodeList=([^ ]+)',raw).group(1);assert node in subprocess.check_output(['scontrol','show','hostnames',nodes],env=env,text=True,timeout=15).splitlines()
 q={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':False,'GPU_submitted':False,'CPU_keeper_actual':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'source_manifest_SHA256':g['source_manifest_SHA256'],'publication_commit':g['publication_commit'],'component_qualified':False,'scope':'One traineddraft, syntheticexpandedfeatures; no target, acceptance or Model speed qualification'}
 def save():
  q['observed_UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();tmp=W.with_suffix('.tmp');tmp.write_text(json.dumps(q,indent=2)+'\n');tmp.replace(W)
 def rows():return [x.split('|')for x in subprocess.check_output(['sacct','-S','2026-10-31','-j','910371','-n','-P','-o','JobID,JobName%120,State,ExitCode,NodeList%100'],env=env,text=True,timeout=25).splitlines()]
 def no_foreign(ignore=()):
  for x in rows():
   if re.fullmatch(r'910371[.]\d+',x[0])and x[0]not in ignore and x[2].split()[0].rstrip('+')not in TERMINAL:
    assert NODE not in subprocess.check_output(['scontrol','show','hostnames',x[4]],env=env,text=True,timeout=15).splitlines(),'Active foreign step on draft node'
 def idle(suffix):
  args=['srun','--jobid=910371','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c1','--gres=none','--mem=256M','-w',NODE,'-t','00:01:00','--job-name='+LABEL+'-'+suffix,'python3',str(D/'idle.py')]
  z=subprocess.run(args,env=env,capture_output=True,text=True,timeout=100);v=[json.loads(x)for x in z.stdout.splitlines()if x.startswith('{')];assert z.returncode==0 and len(v)==1 and v[0]['passed']and v[0]['node']==NODE and not v[0]['kfd_pids']
  return v[0]
 save();proc=None
 try:
  assert not any(x[1]==LABEL for x in rows()),'Follow existing exact label; no replay';no_foreign();before=idle('before');q['fresh_before']=before;save()
  until=time.monotonic()+20
  while True:
   found=[x for x in rows()if x[0]==before['step']]
   if found and found[0][1:]==[LABEL+'-before','COMPLETED','0:0',NODE]:break
   assert time.monotonic()<until,'Own CPU probe terminal accounting unresolved';time.sleep(2)
  no_foreign({before['step']});epoch=time.time();q['GPU_start_UTC_unix']=epoch;save()
  args=['srun','--jobid=910371','--overlap','--exact','--immediate=30','--cpu-bind=none','--kill-on-bad-exit=1','-N1','-n1','-c8','--gres=dcu:1','--mem=65536M','-w',NODE,'-t','00:10:00','--job-name='+LABEL,'--output='+str(R/'logs'/(LABEL+'-%J.log')),'--export=ALL,WAVE_INSIDE_R2323=Kimi-C1-long-draft-cost-r324/GPU_inside.sh,KIMI_DRAFT_COST_MANIFEST_SHA='+g['source_manifest_SHA256']+',KIMI_DRAFT_COST_GPUSTART_UNIX='+str(epoch),'bash',str(R/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
  with(R/'logs'/(LABEL+'-srun-client.log')).open('x')as log:proc=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  q.update(GPU_submitted=True,srun_PID=proc.pid,state='FOLLOW_SAME_SINGLE_GPU_HANDLE');save();end=time.monotonic()+690
  while time.monotonic()<end:
   matched=[x for x in rows()if x[1]==LABEL];assert len(matched)<=1
   if matched:
    row=matched[0];assert row[4]==NODE and re.fullmatch(r'910371[.]\d+',row[0]);assert q.get('actual_GPU',row[0])==row[0];q.update(actual_GPU=row[0],last_Slurm=row);save()
    if row[2].split()[0].rstrip('+')in TERMINAL:q['terminal']=row;break
   if proc.poll()is not None:q.setdefault('srun_exited_at',time.monotonic());assert time.monotonic()-q['srun_exited_at']<60,'srun exited, accounting unresolved; follow only'
   time.sleep(15)
  assert 'terminal'in q,'GPU terminal unresolved; follow exact handle, no replay'
  q['post']=idle('post');q['post_empty']=True;save();path=R/'results'/(LABEL+'-'+q['actual_GPU']+'.json');until=time.monotonic()+30
  while not path.exists()and time.monotonic()<until:time.sleep(2)
  if path.exists():
   data=json.loads(path.read_bytes());q.update(report_path=str(path),report_SHA256=sha(path),report=data)
   q['component_qualified']=bool(q['terminal'][2:4]==['COMPLETED','0:0']and data['completed']and data['passed']and data['normal_close']and len(data['cases'])==3 and all(x['normal_retired_closed']and len(x['samples'])==8 for x in data['cases']))
  q.update(completed=True,state='COMPONENT_COMPLETE_PENDING_ROOT_COST_AUDIT'if q['component_qualified']else'TERMINAL_FAIL_POST_CLEAN');save()
 except BaseException as e:q.update(error=repr(e),state='FAILED_BEFORE_GPU'if not q['GPU_submitted']else'FOLLOW_SAME_HANDLE_OR_TERMINAL_FAILURE_NO_REPLAY');save();raise
if __name__=='__main__':main()
