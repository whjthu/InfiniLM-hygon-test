"""One GPU600s, four declared fresh children. Read loss never cancels a run."""
from pathlib import Path
import json,hashlib,os,subprocess,re,time,datetime,argparse,traceback
from observation import observe,deadline_owned_active,terminal
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');S=R/'incoming/Kimi-draft-query-warm-cost-r379/gpu-v1';LABEL='Kimi-draft-query-warm-cost-r379-v1';NODE='a02r3n03';PARENT='910371';W=R/'results'/(LABEL+'-watch.json')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
host=lambda p:R/Path(p).relative_to('/refactor')if str(p).startswith('/refactor/')else Path(p)
def main():
 p=argparse.ArgumentParser();p.add_argument('--publication-commit',required=True);p.add_argument('--manifest-sha',required=True);a=p.parse_args();assert re.fullmatch('[0-9a-f]{40}',a.publication_commit)and os.environ['SLURM_JOB_ID']=='910407'
 assert sha(S/'GPU_MANIFEST.json')==a.manifest_sha;m=json.loads((S/'GPU_MANIFEST.json').read_bytes());assert m['GPU_GO']is True;assert all(sha(S/n)==h for n,h in m['files_SHA256'].items())and all(sha(host(n))==h for n,h in m['dependencies_SHA256'].items())
 cp=host(m['CPU_receipt']);assert sha(cp)==m['CPU_receipt_SHA256'];cpu=json.loads(cp.read_bytes());assert cpu['completed']and cpu['passed']and cpu['GPU']==0 and not cpu['Native_dlopen']and cpu['actual_CPU']=='910407.762'and cpu['source_manifest_SHA256']==m['CPU_source_manifest_SHA256']
 cm=host(m['CPU_source_manifest']);assert sha(cm)==m['CPU_source_manifest_SHA256'];cm=json.loads(cm.read_bytes());assert all(sha(S/n)==cm['files_SHA256'][n]for n in ['GPU_driver.py','child_driver.py','warm_plan.py'])
 env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_','CUDA_','HIP_','ROCR_','GPU_'))or k in ['SLURM_CONF','SLURM_CONF_SERVER']}
 parent=subprocess.check_output(['scontrol','show','job',PARENT,'-o'],text=True,env=env,timeout=25);assert 'JobState=RUNNING'in parent and 'UserId=aclsylqidf('in parent;nodes=re.search(r'(?:^| )NodeList=([^ ]+)',parent).group(1);assert NODE in subprocess.check_output(['scontrol','show','hostnames',nodes],text=True,env=env,timeout=15).splitlines()
 assert not W.exists();watch=dict(completed=False,passed=False,publication_commit=a.publication_commit,manifest_SHA256=a.manifest_sha,CPU_keeper=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],GPU_submitted=False,GPU_max_s=600,no_retry=True,target_Model_loaded=False,planned_fresh_draft_loads=4);proc=None;epoch=None
 def save():
  watch['observed_UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();tmp=W.with_suffix('.tmp');tmp.write_text(json.dumps(watch,indent=2)+'\n');tmp.replace(W)
 def rows():return [v.split('|')for v in subprocess.check_output(['sacct','-S','2026-11-02','-j',PARENT,'-n','-P','-o','JobIDRaw,JobName%100,State,ExitCode,NodeList%100'],text=True,env=env,timeout=25).splitlines()]
 def exact():
  found=[v for v in rows()if v[1]==LABEL];assert len(found)<=1
  for v in found:assert re.fullmatch(PARENT+r'[.]\d+',v[0])and v[4]==NODE
  return found[0]if found else None
 def idle(phase):
  cmd=['srun','--jobid='+PARENT,'--overlap','--exact','--immediate=15','--cpu-bind=none','-N1','-n1','-c1','-w',NODE,'--gres=none','--mem=512M','-t','00:01:00','--job-name='+LABEL+'-'+phase,'python3',str(S/'idle.py')];z=subprocess.run(cmd,env=env,text=True,capture_output=True,timeout=75);v=[json.loads(l)for l in z.stdout.splitlines()if l.startswith('{')];assert z.returncode==0 and len(v)==1 and v[0]['node']==NODE and v[0]['KFD_available']and not v[0]['kfd_pids']and not v[0]['matching_component_processes']and all(v[0]['ports_bindable'].values());return v[0]
 def dedup(ignore=()):
  for v in rows():
   if re.fullmatch(PARENT+r'[.]\d+',v[0])and v[0]not in ignore and not terminal(v):assert NODE not in subprocess.check_output(['scontrol','show','hostnames',v[4]],text=True,env=env,timeout=15).splitlines()
 try:
  assert exact()is None;dedup();watch['fresh_before']=idle('before');dedup((watch['fresh_before']['actual_step'],));save();wrapper=R/'incoming/resource_wave_local_r3097_outer.sh';assert sha(wrapper)==m['outer_wrapper_SHA256']
  epoch=time.time();watch.update(GPU_epoch_unix=epoch,GPU_deadline_unix=epoch+600)
  env.update(WAVE_INSIDE_R2323='Kimi-draft-query-warm-cost-r379/gpu-v1/GPU_inside.sh',KIMI_R379_GPU_MANIFEST=a.manifest_sha,KIMI_R379_GPU_DEADLINE=str(epoch+570))
  cmd=['srun','--jobid='+PARENT,'--overlap','--exact','--immediate=15','--cpu-bind=none','--kill-on-bad-exit=0','-N1','-n1','--ntasks-per-node=1','-c8','-w',NODE,'--gres=dcu:1','--mem=65536M','-t','00:10:00','--job-name='+LABEL,'--export=ALL','--output='+str(R/'logs'/(LABEL+'-%J-rank%t.log')),'bash',str(wrapper),'probe']
  with(R/'logs'/(LABEL+'-srun-client.log')).open('x')as log:proc=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  watch.update(GPU_submitted=True,srun_PID=proc.pid);save()
  try:row=observe(exact,watch,save,epoch+620)
  except TimeoutError:
   owned=deadline_owned_active(exact,watch,epoch+600,time.time());watch['deadline_cancel_owned_step']=owned
   if owned:subprocess.run(['scancel',owned],env=env,check=True,timeout=25)
   save();raise
  watch['terminal']=row;watch['srun_exit_code']=proc.wait(timeout=30);watch['post']=idle('post')
  report=R/'results'/('Kimi379-four-process-'+watch['actual_GPU'])/'result.json';until=time.monotonic()+75
  while True:
   try:q=json.loads(report.read_bytes())
   except (FileNotFoundError,json.JSONDecodeError):q=None
   if q is not None and q.get('completed'):break
   assert time.monotonic()<until,'Bounded original report visibility; no reload';time.sleep(2)
  watch.update(completed=True,report_path=str(report),report_SHA256=sha(report),driver_error=q.get('error'),actual_children=len(q.get('children',[])),normal_all_children=all(c.get('normal_session_close')and c.get('normal_Runtime_close')for c in q.get('children',[])))
  watch['passed']=q.get('passed')is True and len(q.get('children',[]))==4 and watch['normal_all_children']and row[2]=='COMPLETED'and row[3]=='0:0'and watch['srun_exit_code']==0;save();assert watch['passed'],watch
 except BaseException as e:
  watch.update(error=repr(e),traceback=traceback.format_exc());save()
  # Never cancel here: arbitrary parsing/read/observer errors are not a deadline.
  raise
if __name__=='__main__':main()
