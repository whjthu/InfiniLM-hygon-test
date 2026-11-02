"""One draft fullphase raw-history comparison, bounded cleanup, no replay."""
from pathlib import Path
import json,hashlib,os,subprocess,re,time,datetime,argparse
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');S=R/'incoming/Kimi-raw-history-cost-r386'
LABEL='Kimi-raw-history-cost-r386';NODE='a02r3n15';PARENT='910371';W=R/'results'/(LABEL+'-watch.json')
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
host=lambda p:R/Path(p).relative_to('/refactor')if str(p).startswith('/refactor/')else Path(p)
def main():
 a=argparse.ArgumentParser();a.add_argument('--publication-commit',required=True);a.add_argument('--manifest-sha',required=True);a=a.parse_args();cleanup_deadline=time.monotonic()+645
 assert re.fullmatch('[0-9a-f]{40}',a.publication_commit)and os.environ['SLURM_JOB_ID']=='910407'
 assert sha(S/'GPU_MANIFEST.json')==a.manifest_sha;m=json.loads((S/'GPU_MANIFEST.json').read_bytes());assert m['GPU_GO']is True
 assert all(sha(S/n)==h for n,h in m['files_SHA256'].items())
 cp=host(m['CPU_receipt']);assert sha(cp)==m['CPU_receipt_SHA256'];q=json.loads(cp.read_bytes());assert q['completed']and q['passed']and q['GPU']==0 and not q['Native_dlopen']and q['source_manifest_SHA256']==m['CPU_source_manifest_SHA256']
 assert all(sha(host(n))==h for n,h in m['dependencies_SHA256'].items())
 env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','CUDA_','HIP_','ROCR_','GPU_'))or k in ['SLURM_CONF','SLURM_CONF_SERVER']}
 parent=subprocess.check_output(['scontrol','show','job',PARENT,'-o'],text=True,env=env,timeout=25);assert 'JobState=RUNNING'in parent and 'UserId=aclsylqidf('in parent
 nodes=re.search(r'(?:^| )NodeList=([^ ]+)',parent).group(1);assert NODE in subprocess.check_output(['scontrol','show','hostnames',nodes],text=True,env=env,timeout=15).splitlines()
 assert not W.exists();watch=dict(completed=False,passed=False,publication_commit=a.publication_commit,manifest_SHA256=a.manifest_sha,CPU_keeper=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],GPU_submitted=False,GPU_max_s=600,no_retry=True,target_Model_loaded=False,draft_checkpoint_loads_planned=1);proc=None
 def save():watch['observed_UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();t=W.with_suffix('.tmp');t.write_text(json.dumps(watch,indent=2)+'\n');t.replace(W)
 def rows():return [s.split('|')for s in subprocess.check_output(['sacct','-S','2026-11-01','-j',PARENT,'-n','-P','-o','JobIDRaw,JobName%100,State,ExitCode,NodeList%100'],text=True,env=env,timeout=25).splitlines()]
 def exact():
  found=[v for v in rows()if v[1]==LABEL];assert len(found)<=1
  for v in found:assert re.fullmatch(PARENT+r'[.]\d+',v[0])and v[4]==NODE
  return found[0]if found else None
 def idle(phase):
  cmd=['srun','--jobid='+PARENT,'--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c1','-w',NODE,'--gres=none','--mem=512M','-t','00:01:00','--job-name='+LABEL+'-'+phase,'python3',str(S/'idle.py')]
  p=subprocess.run(cmd,env=env,text=True,capture_output=True,timeout=min(100,max(1,cleanup_deadline-time.monotonic())));v=[json.loads(s)for s in p.stdout.splitlines()if s.startswith('{')]
  assert p.returncode==0 and len(v)==1 and v[0]['node']==NODE and v[0]['KFD_available']and not v[0]['kfd_pids']and not v[0]['matching_component_processes']and all(v[0]['ports_bindable'].values());return v[0]
 def dedup(ignore=()):
  for v in rows():
   if re.fullmatch(PARENT+r'[.]\d+',v[0])and v[0]not in ignore and v[2].split()[0].rstrip('+')not in TERMINAL:
    assert NODE not in subprocess.check_output(['scontrol','show','hostnames',v[4]],text=True,env=env,timeout=15).splitlines()
 try:
  assert exact()is None;dedup();watch['fresh_before']=idle('before');dedup((watch['fresh_before']['actual_step'],));save()
  wrapper=R/'incoming/resource_wave_local_r3097_outer.sh';assert sha(wrapper)==m['outer_wrapper_SHA256']
  epoch=time.time();watch['GPU_epoch_unix']=epoch;watch['GPU_deadline_unix']=epoch+600
  export='ALL,WAVE_INSIDE_R2323=Kimi-raw-history-cost-r386/GPU_inside.sh,KIMI_R386_GPU_MANIFEST='+a.manifest_sha+',KIMI_R386_GPU_DEADLINE='+str(epoch+570)
  cmd=['srun','--jobid='+PARENT,'--overlap','--exact','--immediate=30','--cpu-bind=none','--kill-on-bad-exit=0','-N1','-n1','--ntasks-per-node=1','-c8','-w',NODE,'--gres=dcu:1','--mem=65536M','-t','00:10:00','--job-name='+LABEL,'--export='+export,'--output='+str(R/'logs'/(LABEL+'-%J-rank%t.log')),'bash',str(wrapper),'probe']
  with(R/'logs'/(LABEL+'-srun-client.log')).open('x')as f:proc=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  watch.update(GPU_submitted=True,srun_PID=proc.pid);save()
  while time.time()<epoch+610 and time.monotonic()<cleanup_deadline-15:
   row=exact()
   if row:
    watch.update(actual_GPU=row[0],last_Slurm=row);save()
    if row[2].split()[0].rstrip('+')in TERMINAL:watch['terminal']=row;break
   time.sleep(10)
  else:raise TimeoutError('Original600second GPU deadline, no replay')
  watch['srun_exit_code']=proc.wait(timeout=20);watch['post']=idle('post')
  p=R/'results'/('Kimi386-raw-history-'+watch['actual_GPU']+'.json');until=min(time.monotonic()+90,cleanup_deadline-2)
  while True:
   q=json.loads(p.read_bytes())if p.exists()else None
   if q is not None and (q.get('completed')or q.get('error')):break
   assert row[2]=='COMPLETED'and row[3]=='0:0'and watch['srun_exit_code']==0,'Failed GPU has no final report; preserve failure'
   assert time.monotonic()<until,'Bounded final report visibility expired'
   time.sleep(2)
  watch.update(report_path=str(p),report_SHA256=sha(p),normal_runtime_close=q.get('normal_Runtime_close'),normal_session_close=q.get('normal_session_close'),driver_completed=q.get('completed'),driver_error=q.get('error'),completed=True)
  watch['passed']=q.get('completed')is True and q.get('passed')is True and watch['normal_runtime_close']and watch['normal_session_close']and row[2]=='COMPLETED'and row[3]=='0:0'and watch['srun_exit_code']==0
  save();assert watch['passed'],watch
 except BaseException as e:
  watch.update(error=repr(e));save()
  row=exact()if watch['GPU_submitted']else None
  if row and row[2].split()[0].rstrip('+')not in TERMINAL:subprocess.run(['scancel',row[0]],env=env,check=True,timeout=25)
  if proc is not None:
   try:watch['srun_exit_code']=proc.wait(timeout=min(20,max(1,cleanup_deadline-time.monotonic())))
   except Exception as close:watch['process_cleanup_error']=repr(close)
   if 'post'not in watch and 'post_after_failure'not in watch:
    try:watch['post_after_failure']=idle('post')
    except Exception as close:watch['post_error']=repr(close)
   save()
  watch['completed']=True;save()
  raise
if __name__=='__main__':main()
