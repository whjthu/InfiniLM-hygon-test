"""One8rank independent component batch, published source,600s GPU hard bound."""
import pathlib,json,hashlib,os,subprocess,time,re,argparse,datetime
R=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');S=R/'incoming/Deep-HC-epilogue-r358/GPU-r1';LABEL='Deep-HC-epilogue-r358-r1';NODE='a02r4n07';W=R/'results'/(LABEL+'-watch.json')
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
host=lambda p:R/pathlib.Path(p).relative_to('/refactor')if str(p).startswith('/refactor/')else pathlib.Path(p)
def main():
 p=argparse.ArgumentParser();p.add_argument('--publication-commit',required=True);p.add_argument('--manifest-sha',required=True);p.add_argument('--preflight-only',action='store_true');a=p.parse_args();assert re.fullmatch('[0-9a-f]{40}',a.publication_commit)and re.fullmatch('[0-9a-f]{64}',a.manifest_sha)
 assert os.environ['SLURM_JOB_ID']=='910407';assert sha(S/'GPU_MANIFEST.json')==a.manifest_sha;m=json.loads((S/'GPU_MANIFEST.json').read_bytes());assert all(sha(S/n)==h for n,h in m['files_SHA256'].items())
 cp=R/'candidates/Deep-HC-epilogue-r358-cpu-r1/CPU-build.json';assert sha(cp)==m['CPU_receipt_SHA256'];c=json.loads(cp.read_bytes());assert c['completed']and c['actual_step']=='910407.696'and c['actual_fixture_groups']==8 and not c['GPU_initialized']and not c['Native_dlopen'];assert sha(host(c['library']))==c['library_SHA256']and sha(host(c['Native_path']))==c['Native_SHA256']=='afc6f58b71ef7a51657deae452da86a31afbfa6e79d36d442ccdd26d56c0e6f2'
 assert sha(cp.with_name('CPU-fixtures.json'))==c['CPU_fixture_receipt_SHA256']
 env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','CUDA_','HIP_','ROCR_','GPU_'))or k in ['SLURM_CONF','SLURM_CONF_SERVER']}
 parent=subprocess.check_output(['scontrol','show','job','910374','-o'],env=env,text=True,timeout=25);assert 'JobState=RUNNING'in parent and 'UserId=aclsylqidf('in parent;nodes=re.search(r'(?:^| )NodeList=([^ ]+)',parent).group(1);assert NODE in subprocess.check_output(['scontrol','show','hostnames',nodes],env=env,text=True,timeout=15).splitlines()
 if a.preflight_only:
  print(json.dumps(dict(passed=True,CPU_only=True,preflight_actual=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],manifest_SHA256=a.manifest_sha,host_namespace_paths_checked=True,GPU_submitted=False)));return
 assert not W.exists();watch=dict(completed=False,publication_commit=a.publication_commit,manifest_SHA256=a.manifest_sha,CPU_keeper_actual=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],GPU_submitted=False,GPU_max_s=600,no_retry=True,Model_loaded=False)
 def save():watch['observed_UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();tmp=W.with_suffix('.tmp');tmp.write_text(json.dumps(watch,indent=2)+'\n');tmp.replace(W)
 def rows():return [v.split('|')for v in subprocess.check_output(['sacct','-S','2026-11-01','-j','910374','-n','-P','-o','JobID,JobName%100,State,ExitCode,NodeList%100'],env=env,text=True,timeout=25).splitlines()]
 def exact():
  found=[v for v in rows()if v[1]==LABEL];assert len(found)<=1
  for v in found:assert re.fullmatch(r'910374[.]\d+',v[0])and v[4]==NODE
  return found[0]if found else None
 def no_intersection(ignore=()):
  for v in rows():
   if re.fullmatch(r'910374[.]\d+',v[0])and v[0]not in ignore and v[2].split()[0].rstrip('+')not in TERMINAL:assert NODE not in subprocess.check_output(['scontrol','show','hostnames',v[4]],env=env,text=True,timeout=15).splitlines()
 def idle(suffix):
  cmd=['srun','--jobid=910374','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c1','-w',NODE,'--gres=none','--mem=512M','-t','00:02:00','--job-name='+LABEL+'-'+suffix,'python3',str(S/'idle.py')];z=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=160);values=[json.loads(v)for v in z.stdout.splitlines()if v.startswith('{')];assert z.returncode==0 and len(values)==1 and values[0]['node']==NODE and values[0]['KFD_available']and not values[0]['kfd_pids'];return values[0]
 try:
  assert exact()is None;no_intersection();watch['fresh_before']=idle('before');no_intersection((watch['fresh_before']['actual_step'],));save();wrapper=R/'incoming/Deep-C1-current-profile-r353/repair-r1/controller-v2/resource_wave_host_alias_outer.sh';assert sha(wrapper)=='b6cd3ac7aa1829a39434d8dc186deed6c993323d1ac894727072701570c0825d'
  export='ALL,WAVE_INSIDE_R2323=Deep-HC-epilogue-r358/GPU-r1/GPU_inside.sh,DEEP358_MANIFEST_SHA='+a.manifest_sha
  cmd=['srun','--jobid=910374','--overlap','--exact','--immediate=30','--cpu-bind=none','--kill-on-bad-exit=0','-N1','-n8','--ntasks-per-node=8','-c4','-w',NODE,'--gres=dcu:8','--mem=64000M','-t','00:10:00','--job-name='+LABEL,'--export='+export,'--output='+str(R/'logs'/(LABEL+'-%J-rank%t.log')),'bash',str(wrapper),'probe']
  with(R/'logs'/(LABEL+'-srun-client.log')).open('x')as log:proc=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  watch.update(GPU_submitted=True,srun_client_PID=proc.pid);save();end=time.monotonic()+720
  while time.monotonic()<end:
   row=exact()
   if row:
    watch.update(actual_GPU=row[0],last_Slurm=row);save()
    if row[2].split()[0].rstrip('+')in TERMINAL:watch['terminal']=row;break
   time.sleep(10)
  else:
   row=exact()
   if row and row[2].split()[0].rstrip('+')not in TERMINAL:subprocess.run(['scancel',row[0]],env=env,check=True,timeout=25)
   raise TimeoutError('Original oneGPUstep deadline; no replay')
  watch['post']=idle('post');reports=[]
  for rank in range(8):
   p=R/'results'/f"Deep358-GPU-r1-{watch['actual_GPU']}-rank{rank}.json"
   if p.exists():
    q=json.loads(p.read_bytes());reports.append(dict(rank=rank,path=str(p),SHA256=sha(p),passed=q.get('passed'),normal_close=q.get('normal_close'),error=q.get('error'),CPU_affinity=q.get('CPU_affinity_after_first_baseline'),physical_HIP=q.get('hardware',{}).get('visible_physical_HIP')))
  successful=[v for v in reports if v['passed']and v['normal_close']];watch.update(completed=True,reports=reports,valid_ranks=len(successful),all8_passed=len(successful)==8)
  if len(successful)==8:
   assert {v['physical_HIP']for v in successful}=={str(i)for i in range(8)}
   masks=[set(v['CPU_affinity'])for v in successful];assert all(len(v)==4 for v in masks)and all(not masks[i]&masks[j]for i in range(8)for j in range(i))
  save()
 except BaseException as exc:watch.update(error=repr(exc));save();raise
if __name__=='__main__':main()
