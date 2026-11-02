"""Published one-shot a314 functionality qualification: GPU180s/CPUkeeper300s."""
import pathlib,json,hashlib,os,subprocess,time,re,argparse,datetime,traceback
R=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');S=R/'incoming/Deep-HC-Native-dispatch-qual-r378/gpu-v1';LABEL='Deep-HC-Native-dispatch-qual-r378-v1';NODE='a02r4n04';W=R/'results'/(LABEL+'-watch.json')
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
host=lambda p:R/pathlib.Path(p).relative_to('/refactor')if str(p).startswith('/refactor/')else pathlib.Path(p)
def main():
 p=argparse.ArgumentParser();p.add_argument('--publication-commit',required=True);p.add_argument('--manifest-sha',required=True);a=p.parse_args();assert re.fullmatch('[0-9a-f]{40}',a.publication_commit)
 assert os.environ['SLURM_JOB_ID']=='910407';assert sha(S/'GPU_MANIFEST.json')==a.manifest_sha;m=json.loads((S/'GPU_MANIFEST.json').read_bytes());assert all(sha(S/n)==h for n,h in m['files_SHA256'].items())
 cp=R/'candidates/Deep-HC-Native-dispatch-qual-r378-cpu-v1/CPU-build.json';assert sha(cp)==m['CPU757_receipt_SHA256'];c=json.loads(cp.read_bytes());assert c['passed']and c['actual_step']=='910407.757'and not c['Native_dlopen']and not c['GPU_initialized'];assert sha(host(c['bridge_library']))==m['bridge_SHA256']and sha(host(c['Native_path']))==m['Native_SHA256']
 for n,h in c['Native50_objects_SHA256'].items():assert sha(host(n))==h
 env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','CUDA_','HIP_','ROCR_','GPU_'))or k in ['SLURM_CONF','SLURM_CONF_SERVER']}
 env.update(WAVE_INSIDE_R2323='Deep-HC-Native-dispatch-qual-r378/gpu-v1/GPU_inside.sh',DEEP378_GPU_MANIFEST_SHA=a.manifest_sha)
 parent=subprocess.check_output(['scontrol','show','job','910374','-o'],env=env,text=True,timeout=20);assert 'JobState=RUNNING'in parent and 'UserId=aclsylqidf('in parent;nodes=re.search(r'(?:^| )NodeList=([^ ]+)',parent).group(1);assert NODE in subprocess.check_output(['scontrol','show','hostnames',nodes],env=env,text=True,timeout=10).splitlines()
 assert not W.exists();watch=dict(completed=False,publication_commit=a.publication_commit,manifest_SHA256=a.manifest_sha,CPU_keeper_actual=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],GPU_submitted=False,GPU_max_s=180,CPU_max_s=300,no_retry=True,Model_loaded=False,performance_experiment=False)
 def save():
  watch['observed_UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();tmp=W.with_suffix('.tmp');tmp.write_text(json.dumps(watch,indent=2)+'\n');tmp.replace(W)
 def rows():return [v.split('|')for v in subprocess.check_output(['sacct','-S','2026-11-02','-j','910374','-n','-P','-o','JobID,JobName%100,State,ExitCode,NodeList%100'],env=env,text=True,timeout=15).splitlines()]
 def exact():
  found=[v for v in rows()if v[1]==LABEL];assert len(found)<=1
  for v in found:assert re.fullmatch(r'910374[.]\d+',v[0])and v[4]==NODE
  return found[0]if found else None
 def no_intersection(ignore=()):
  for v in rows():
   if re.fullmatch(r'910374[.]\d+',v[0])and v[0]not in ignore and v[2].split()[0].rstrip('+')not in TERMINAL:assert NODE not in subprocess.check_output(['scontrol','show','hostnames',v[4]],env=env,text=True,timeout=10).splitlines()
 def idle(suffix):
  cmd=['srun','--jobid=910374','--overlap','--exact','--immediate=15','--cpu-bind=none','-N1','-n1','-c1','-w',NODE,'--gres=none','--mem=512M','-t','00:01:00','--job-name='+LABEL+'-'+suffix,'python3',str(S/'idle.py')];z=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=45);vs=[json.loads(v)for v in z.stdout.splitlines()if v.startswith('{')];assert z.returncode==0 and len(vs)==1 and vs[0]['node']==NODE and vs[0]['KFD_available']and not vs[0]['kfd_pids']and not vs[0]['matching_component_processes'];return vs[0]
 try:
  assert exact()is None;no_intersection();watch['fresh_before']=idle('before');no_intersection((watch['fresh_before']['actual_step'],));save()
  wrapper=R/'incoming/Deep-C1-current-profile-r353/repair-r1/controller-v2/resource_wave_host_alias_outer.sh';assert sha(wrapper)=='b6cd3ac7aa1829a39434d8dc186deed6c993323d1ac894727072701570c0825d'
  cmd=['srun','--jobid=910374','--overlap','--exact','--immediate=15','--cpu-bind=none','--kill-on-bad-exit=0','-N1','-n1','--ntasks-per-node=1','-c32','-w',NODE,'--gres=dcu:8','--mem=64000M','-t','00:03:00','--job-name='+LABEL,'--export=ALL','--output='+str(R/'logs'/(LABEL+'-%J-rank%t.log')),'bash',str(wrapper),'probe']
  with(R/'logs'/(LABEL+'-srun-client.log')).open('x')as log:proc=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  watch.update(GPU_submitted=True,srun_client_PID=proc.pid);save();deadline=time.monotonic()+200
  while time.monotonic()<deadline:
   row=exact()
   if row:
    watch.update(actual_GPU=row[0],last_Slurm=row);save()
    if row[2].split()[0].rstrip('+')in TERMINAL:watch['terminal']=row;break
   time.sleep(3)
  else:
   row=exact()
   if row and row[2].split()[0].rstrip('+')not in TERMINAL:subprocess.run(['scancel',row[0]],env=env,check=True,timeout=15)
   raise TimeoutError('Original GPU180s step deadline; no replay')
  watch['srun_client_exit']=proc.wait(timeout=15);watch['post']=idle('post');rp=R/'results'/f"Deep378-gpu-v1-{watch['actual_GPU']}.json";assert rp.exists();q=json.loads(rp.read_bytes());watch.update(completed=True,report_path=str(rp),report_SHA256=sha(rp),passed=q.get('passed'),actual_calls=q.get('actual_calls'),normal_close_all8=len(q.get('protocol',{}).get('closes',[]))==8 and all(v.get('normal_close')and v.get('Runtime_expired')==1 for v in q.get('protocol',{}).get('closes',[])),threads_joined=q.get('protocol',{}).get('threads_joined'),error=q.get('error'));save();assert watch['srun_client_exit']==0 and watch['passed']and watch['normal_close_all8']and watch['threads_joined']
 except BaseException as e:watch.update(error=repr(e),traceback=traceback.format_exc());save();raise
if __name__=='__main__':main()
