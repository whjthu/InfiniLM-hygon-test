"""One node TP8 component only, one published source, bounded cleanup, no replay."""
from pathlib import Path
import json,hashlib,os,subprocess,re,time,datetime,argparse
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');S=R/'incoming/GLM-selected-KV-on-demand-capacity-r408/gpu-v1'
LABEL='GLM-selected-KV-on-demand-capacity-r408-v1';NODE='a02r3n16';PARENT='910371';W=R/'results'/(LABEL+'-watch.json')
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
host=lambda p:R/Path(p).relative_to('/refactor')if str(p).startswith('/refactor/')else Path(p)
def main():
 a=argparse.ArgumentParser();a.add_argument('--publication-commit',required=True);a.add_argument('--manifest-sha',required=True);a=a.parse_args()
 assert re.fullmatch('[0-9a-f]{40}',a.publication_commit)and os.environ['SLURM_JOB_ID']=='910407'
 assert sha(S/'GPU_MANIFEST.json')==a.manifest_sha;m=json.loads((S/'GPU_MANIFEST.json').read_bytes());assert m['GPU_GO']is True
 assert all(sha(S/n)==h for n,h in m['source_SHA256'].items())
 p=host(m['CPU_receipt']);assert sha(p)==m['CPU_receipt_SHA256']
 c=json.loads(p.read_bytes());assert c['completed']and sha(host(c['library']))==c['library_SHA256']==m['bridge_SHA256']
 assert c['passed']and c['fresh_M_before_compile']and c['actual_HOST_contract']['passed'];assert c['actual_CPU_consumer']['passed']
 assert m['CPU_source_manifest_SHA256']==c['source_manifest_SHA256']
 env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','CUDA_','HIP_','ROCR_','GPU_'))or k in ['SLURM_CONF','SLURM_CONF_SERVER']}
 parent=subprocess.check_output(['scontrol','show','job',PARENT,'-o'],text=True,env=env,timeout=25);assert 'JobState=RUNNING'in parent and 'UserId=aclsylqidf('in parent
 nodes=re.search(r'(?:^| )NodeList=([^ ]+)',parent).group(1);assert NODE in subprocess.check_output(['scontrol','show','hostnames',nodes],text=True,env=env,timeout=15).splitlines()
 assert not W.exists();watch=dict(completed=False,passed=False,publication_commit=a.publication_commit,manifest_SHA256=a.manifest_sha,CPU_keeper=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],GPU_submitted=False,GPU_max_s=600,no_retry=True,Model_loaded=False);proc=None
 def save():watch['observed_UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();t=W.with_suffix('.tmp');t.write_text(json.dumps(watch,indent=2)+'\n');t.replace(W)
 def rows():return [s.split('|')for s in subprocess.check_output(['sacct','-S','2026-11-01','-j',PARENT,'-n','-P','-o','JobIDRaw,JobName%100,State,ExitCode,NodeList%100'],text=True,env=env,timeout=25).splitlines()]
 def exact():
  found=[v for v in rows()if v[1]==LABEL];assert len(found)<=1
  for v in found:assert re.fullmatch(PARENT+r'[.]\d+',v[0])and v[4]==NODE
  return found[0]if found else None
 def idle(phase):
  cmd=['srun','--jobid='+PARENT,'--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c1','-w',NODE,'--gres=none','--mem=512M','-t','00:01:00','--job-name='+LABEL+'-'+phase,'python3',str(S/'idle.py')]
  p=subprocess.run(cmd,env=env,text=True,capture_output=True,timeout=100);v=[json.loads(s)for s in p.stdout.splitlines()if s.startswith('{')]
  assert p.returncode==0 and len(v)==1 and v[0]['node']==NODE and v[0]['KFD_available']and not v[0]['kfd_pids']and not v[0]['matching_component_processes']and not v[0]['port43882_listening'];return v[0]
 def dedup(ignore=()):
  for v in rows():
   if re.fullmatch(PARENT+r'[.]\d+',v[0])and v[0]not in ignore and v[2].split()[0].rstrip('+')not in TERMINAL:
    assert NODE not in subprocess.check_output(['scontrol','show','hostnames',v[4]],text=True,env=env,timeout=15).splitlines()
 try:
  assert exact()is None;dedup();watch['fresh_before']=idle('before');dedup((watch['fresh_before']['actual_step'],));save()
  wrapper=S/'outer.sh';assert sha(wrapper)==m['source_SHA256']['outer.sh']
  epoch=time.time();watch['GPU_epoch_unix']=epoch;watch['GPU_deadline_unix']=epoch+600
  export='ALL,WAVE_INSIDE_R2323=GLM-selected-KV-on-demand-capacity-r408/gpu-v1/GPU_inside.sh,GLM408_GPU_SOURCE_SHA256='+a.manifest_sha+',GLM408_GPU_START='+str(epoch)
  cmd=['srun','--jobid='+PARENT,'--overlap','--exact','--immediate=30','--cpu-bind=none','--kill-on-bad-exit=0','-N1','-n1','--ntasks-per-node=1','-c32','-w',NODE,'--gres=dcu:8','--mem=64000M','-t','00:10:00','--job-name='+LABEL,'--export='+export,'--output='+str(R/'logs'/(LABEL+'-%J-rank%t.log')),'bash',str(wrapper),'probe']
  gpu_env=dict(env,WAVE_INSIDE_R2323='GLM-selected-KV-on-demand-capacity-r408/gpu-v1/GPU_inside.sh',GLM408_GPU_SOURCE_SHA256=a.manifest_sha,GLM408_GPU_START=str(epoch))
  with(R/'logs'/(LABEL+'-srun-client.log')).open('x')as f:proc=subprocess.Popen(cmd,env=gpu_env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  watch.update(GPU_submitted=True,srun_PID=proc.pid);save()
  while time.time()<epoch+620:
   try:row=exact();watch.pop('observation_error',None)
   except(subprocess.CalledProcessError,subprocess.TimeoutExpired,OSError)as exc:watch['observation_error']=repr(exc);save();time.sleep(10);continue
   if row:
    watch.update(actual_GPU=row[0],last_Slurm=row);save()
    if row[2].split()[0].rstrip('+')in TERMINAL:watch['terminal']=row;break
   time.sleep(10)
  else:raise TimeoutError('Original600second GPU deadline, no replay')
  watch['srun_exit_code']=proc.wait(timeout=20);watch['post']=idle('post')
  owners=[];until=time.monotonic()+40
  for rank in range(8):
   p=R/'results'/('GLM408-'+watch['actual_GPU']+'-rank%d.json'%rank)
   while True:
    try:q=json.loads(p.read_bytes())
    except(FileNotFoundError,json.JSONDecodeError):q=None
    if q is not None and q.get('completed')is True:break
    assert time.monotonic()<until,'Bounded report visibility unresolved; samehandle no replay'
    time.sleep(2)
   assert q.get('passed')is True and q.get('normal')is True and q['actual_step']==watch['actual_GPU']and q['rank']==rank
   assert q['GPU_source_manifest_SHA256']==a.manifest_sha and q['library_SHA256']==m['bridge_SHA256']
   assert q['normal_destroy']and q['Runtime_weak_expired']and q['Torch_after_destroy']and q['policy_closed']and q['measurement_calls']==8 and q['warm_calls']==4 and q['directK_diagnostics']==2 and len(q['records'])==12
   owners.append(dict(rank=rank,path=str(p),SHA256=sha(p),PID=q['PID'],physical_PCI=q['physical_PCI']))
  assert len({q['physical_PCI']for q in owners})==8
  watch.update(owners=owners,normal_all8=True,completed=True)
  watch['passed']=all(q is not None for q in owners)and row[2]=='COMPLETED'and row[3]=='0:0'and watch['srun_exit_code']==0
  save();assert watch['passed'],watch
 except BaseException as e:
  watch.update(error=repr(e),passed=False,controller_finished=True)
  if 'terminal'in watch or not watch['GPU_submitted']:watch['completed']=True
  save()
  # Observation/source/consumer failure never cancels a still-owned live job.
  # Slurm600s and inside original540s remain authoritative; follow samehandle.
  # Only known exact owned row after original hard bound may receive TERM.
  if watch['GPU_submitted']and time.time()>=watch.get('GPU_deadline_unix',float('inf')):
   try:
    row=exact()
    if row and row[0]==watch.get('actual_GPU')and row[2].split()[0].rstrip('+')not in TERMINAL:subprocess.run(['scancel','--signal=TERM',row[0]],env=env,check=True,timeout=25)
   except(subprocess.CalledProcessError,subprocess.TimeoutExpired,OSError)as exc:watch['terminal_observation_error']=repr(exc);save()
  raise
if __name__=='__main__':main()
