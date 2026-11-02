"""Single bounded two-card component batch; each rank independent, no TP model."""
import pathlib,json,hashlib,os,subprocess,time,re,argparse,datetime,traceback
R=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');S=R/'incoming/Kimi-draft-KV-guarded-FP16-r380/gpu-v1';LABEL='Kimi-draft-KV-guarded-FP16-r380-v1';NODE='a02r3n07';W=R/'results'/(LABEL+'-watch.json')
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--publication-commit',required=True);p.add_argument('--manifest-sha',required=True);a=p.parse_args();assert re.fullmatch('[0-9a-f]{40}',a.publication_commit)
 assert os.environ['SLURM_JOB_ID']=='910407';assert sha(S/'GPU_MANIFEST.json')==a.manifest_sha;m=json.loads((S/'GPU_MANIFEST.json').read_bytes());assert all(sha(S/n)==h for n,h in m['files_SHA256'].items());assert all(sha(R/pathlib.Path(p).relative_to('/refactor'))==h for p,h in m['production_dependencies_SHA256'].items())
 cp=R/'candidates/Kimi-draft-KV-guarded-FP16-r380-cpu-v1/CPU-result.json';assert sha(cp)==m['CPU761_receipt_SHA256'];c=json.loads(cp.read_bytes());assert c['passed']and c['actual_CPU']=='910407.761'and not c['GPU_initialized']and not c['Model_loaded']
 env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','CUDA_','HIP_','ROCR_','GPU_'))or k in ['SLURM_CONF','SLURM_CONF_SERVER']}
 parent=subprocess.check_output(['scontrol','show','job','910371','-o'],env=env,text=True,timeout=20);assert 'JobState=RUNNING'in parent and 'UserId=aclsylqidf('in parent;nodes=re.search(r'(?:^| )NodeList=([^ ]+)',parent).group(1);assert NODE in subprocess.check_output(['scontrol','show','hostnames',nodes],env=env,text=True,timeout=10).splitlines()
 assert not W.exists();watch=dict(completed=False,publication_commit=a.publication_commit,manifest_SHA256=a.manifest_sha,CPU_keeper_actual=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],GPU_submitted=False,GPU_cards=2,GPU_max_s=600,CPU_max_s=900,no_retry=True,Model_loaded=False)
 def save():
  watch['observed_UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();tmp=W.with_suffix('.tmp');tmp.write_text(json.dumps(watch,indent=2)+'\n');tmp.replace(W)
 def rows():return [v.split('|')for v in subprocess.check_output(['sacct','-S','2026-11-02','-j','910371','-n','-P','-o','JobID,JobName%100,State,ExitCode,NodeList%100'],env=env,text=True,timeout=20).splitlines()]
 def exact():
  found=[v for v in rows()if v[1]==LABEL];assert len(found)<=1
  for v in found:assert re.fullmatch(r'910371[.]\d+',v[0])and v[4]==NODE
  return found[0]if found else None
 def no_intersection(ignore=()):
  for v in rows():
   if re.fullmatch(r'910371[.]\d+',v[0])and v[0]not in ignore and v[2].split()[0].rstrip('+')not in TERMINAL:assert NODE not in subprocess.check_output(['scontrol','show','hostnames',v[4]],env=env,text=True,timeout=10).splitlines()
 def idle(suffix):
  cmd=['srun','--jobid=910371','--overlap','--exact','--immediate=15','--cpu-bind=none','-N1','-n1','-c1','-w',NODE,'--gres=none','--mem=512M','-t','00:01:00','--job-name='+LABEL+'-'+suffix,'python3',str(S/'idle.py')];z=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=60);vs=[json.loads(v)for v in z.stdout.splitlines()if v.startswith('{')];assert z.returncode==0 and len(vs)==1 and vs[0]['node']==NODE and vs[0]['KFD_available']and not vs[0]['kfd_pids']and not vs[0]['matching_component_processes'];return vs[0]
 try:
  assert exact()is None;no_intersection();watch['fresh_before']=idle('before');no_intersection((watch['fresh_before']['actual_step'],));save()
  wrapper=R/'incoming/resource_wave_local_r3097_outer.sh';assert sha(wrapper)=='d3ce6c3196d5728a5f97d459d070fb7720df76d25cadc6be8af0dbb6eca75dce'
  env.update(WAVE_INSIDE_R2323='Kimi-draft-KV-guarded-FP16-r380/gpu-v1/GPU_inside.sh',KIMI380_GPU_MANIFEST_SHA=a.manifest_sha,KIMI380_GPUSTART_UNIX=str(time.time()))
  cmd=['srun','--jobid=910371','--overlap','--exact','--immediate=15','--cpu-bind=none','--kill-on-bad-exit=0','-N1','-n2','--ntasks-per-node=2','-c4','-w',NODE,'--gres=dcu:2','--mem=64000M','-t','00:10:00','--job-name='+LABEL,'--export=ALL','--output='+str(R/'logs'/(LABEL+'-%J-rank%t.log')),'bash',str(wrapper),'probe']
  with(R/'logs'/(LABEL+'-srun-client.log')).open('x')as log:proc=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  watch.update(GPU_submitted=True,srun_client_PID=proc.pid);save();deadline=time.monotonic()+660
  while time.monotonic()<deadline:
   row=exact()
   if row:
    watch.update(actual_GPU=row[0],last_Slurm=row);save()
    if row[2].split()[0].rstrip('+')in TERMINAL:watch['terminal']=row;break
   time.sleep(3)
  else:
   row=exact()
   if row and row[2].split()[0].rstrip('+')not in TERMINAL:subprocess.run(['scancel',row[0]],env=env,check=True,timeout=20)
   raise TimeoutError('One published GPU600s scope; no replay')
  watch['srun_client_exit']=proc.wait(timeout=30);watch['post']=idle('post');watch['reports']=[]
  for rank in (0,1):
   p=R/'results'/f"Kimi380-gpu-v1-{watch['actual_GPU']}-rank{rank}.json";assert p.exists();q=json.loads(p.read_bytes());watch['reports'].append(dict(rank=rank,path=str(p),SHA256=sha(p),passed=q.get('passed'),normal_close=q.get('normal_close'),cases=len(q.get('cases',[])),error=q.get('error')))
  watch.update(completed=True,passed=watch['srun_client_exit']==0 and all(v['passed']and v['normal_close']and v['cases']==4 for v in watch['reports']));save();assert watch['passed']
 except BaseException as e:watch.update(error=repr(e),traceback=traceback.format_exc());save();raise
if __name__=='__main__':main()
