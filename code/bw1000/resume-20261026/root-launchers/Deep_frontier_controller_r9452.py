import pathlib,json,hashlib,os,subprocess,time,datetime,argparse,re
R=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');D=R/'incoming/Deep-real-batch-frontier-r158';label='Deep-real-batch-frontier-r158';parent='910374';node='a02r4n05';ap=argparse.ArgumentParser();ap.add_argument('--publication-commit',required=True);opt=ap.parse_args();assert re.fullmatch('[0-9a-f]{40}',opt.publication_commit)
env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_','CUDA_','HIP_','ROCR_','GPU_'))};sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();record=R/'results/Deep-real-frontier-controller-r9452.json';assert not record.exists();q={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'CPU_keeper':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'source_commit':opt.publication_commit,'state':'validating','GPU_submitted':False,'restarts':0}
def save():q['UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();record.write_text(json.dumps(q,indent=2))
def rows():return [s.split('|')for s in subprocess.check_output(['sacct','-S','2026-10-28','-j',parent,'-n','-P','-o','JobID,JobName%110,State%40,ExitCode,NodeList%100'],env=env,text=True,timeout=30).splitlines()]
def idle(suffix):
 a=['srun','--jobid='+parent,'--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c1','-w',node,'--gres=none','--mem=512M','-t','1','--job-name='+label+'-'+suffix,'python3',str(R/'incoming/verify_gpu_idle_r5738.py')];z=subprocess.run(a,env=env,capture_output=True,text=True,timeout=70);assert z.returncode==0;v=[json.loads(s)for s in z.stdout.splitlines()if s.startswith('{')];assert len(v)==1 and v[0]['node']==node and not v[0]['kfd_pids'];return v
save();p=None
try:
 cp=R/'candidates/Deep-real-batch-frontier-r158/CPU-static.json';assert sha(cp)=='a911a84303a53607076f85eb09b61194b9236985bc7c7740fe6a7d62153db4ae';c=json.loads(cp.read_bytes());assert c['completed']and c['CPU_only']and not c['Native_loaded']and c['actual_step']=='910407.500';mp=D/'manifest-r158.json';assert sha(mp)=='38f43425325ffec3a11b0d843fde502fc73908c198bc8b36ca2f92bb464740f4';m=json.loads(mp.read_bytes())
 for name,h in m['source_SHA256'].items():assert sha(D/name)==h,name
 rs=rows();assert not any(v[1]==label for v in rs)
 for v in rs:
  if v[0].startswith(parent+'.')and v[0].split('.')[-1].isdigit()and v[2].split()[0].rstrip('+')not in ['COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED']:
   assert node not in subprocess.check_output(['scontrol','show','hostnames',v[4]],env=env,text=True,timeout=20).splitlines()
 q['before']=idle('before-r9452');save();go={'GPU_authorized':True,'source_published':True,'publication_branch':'bw1000-experiments','publication_commit':opt.publication_commit,'source_manifest_SHA256':sha(mp),'CPU_receipt_SHA256':sha(cp),'scope':'One8card ordinary ac937 same64 naturalEOS B32/8/16/4 frontier, noModeloptimizationgainclaim from singleorder'};gp=D/'root-GO-r158.json';assert not gp.exists();gp.write_text(json.dumps(go,indent=2))
 a=['srun','--jobid='+parent,'--overlap','--exact','--immediate=30','--cpu-bind=none','--kill-on-bad-exit=1','-N1','-n1','-c32','-w',node,'--gres=dcu:8','--mem=128000M','-t','60','--job-name='+label,'--output='+str(R/'logs'/(label+'-%J.log')),'env','WAVE_INSIDE_R2323='+label+'/GPU_inside_r158.sh','bash',str(R/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
 with(R/'logs/Deep-real-frontier-client-r9452.log').open('x')as log:p=subprocess.Popen(a,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 q.update(state='running_unique_GPU_follow_handle',GPU_submitted=True,GPU_client_PID=p.pid);save();t=time.monotonic()
 while p.poll()is None and time.monotonic()-t<3900:
  matching=[v for v in rows()if v[1]==label];assert len(matching)<=1
  if matching:q.update(actual_handle=matching[0][0],last_actual_row=matching[0])
  save();time.sleep(10)
 assert p.poll()is not None,'Observation bound; no automatic retry';q.update(GPU_client_returncode=p.returncode,state='GPU_client_terminal',actual_rows=[v for v in rows()if v[1]==label]);save();q['post']=idle('post-r9452');q['post_empty']=True;save()
except BaseException as e:q.update(error_class=type(e).__name__,error=repr(e),state='failed_before_GPU'if p is None else'unknown_GPU_follow_same_handle');save();raise
finally:save()
