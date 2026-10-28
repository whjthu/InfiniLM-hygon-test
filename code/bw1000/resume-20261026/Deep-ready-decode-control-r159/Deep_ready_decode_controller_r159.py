import pathlib,json,hashlib,os,subprocess,time,datetime,argparse,re
R=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');D=R/'incoming/Deep-ready-decode-ABBA-r159';label='Deep-ready-decode-ABBA-r159';parent='910374';node='a02r4n05';ap=argparse.ArgumentParser();ap.add_argument('--publication-commit',required=True);ap.add_argument('--cpu-receipt-sha',required=True);opt=ap.parse_args();assert re.fullmatch('[0-9a-f]{40}',opt.publication_commit)and re.fullmatch('[0-9a-f]{64}',opt.cpu_receipt_sha)
env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_','CUDA_','HIP_','ROCR_','GPU_'))};sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();record=R/'results/Deep-ready-decode-controller-r159.json';assert not record.exists();q={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'CPU_keeper':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'source_commit':opt.publication_commit,'state':'validating','GPU_submitted':False,'restarts':0}
def save():q['UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();record.write_text(json.dumps(q,indent=2))
def rows():return [s.split('|')for s in subprocess.check_output(['sacct','-S','2026-10-28','-j',parent,'-n','-P','-o','JobID,JobName%110,State%40,ExitCode,NodeList%100'],env=env,text=True,timeout=30).splitlines()]
def idle(suffix):
 assert sha(R/'incoming/verify_gpu_idle_r5738.py')=='45b7d14902965ca8569000a8c20882d589c7b2468f365e96d0cb552cf6f57b53'
 a=['srun','--jobid='+parent,'--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c1','-w',node,'--gres=none','--mem=512M','-t','1','--job-name='+label+'-'+suffix,'python3',str(R/'incoming/verify_gpu_idle_r5738.py')];z=subprocess.run(a,env=env,capture_output=True,text=True,timeout=70);assert z.returncode==0;v=[json.loads(s)for s in z.stdout.splitlines()if s.startswith('{')];assert len(v)==1 and v[0]['node']==node and not v[0]['kfd_pids'];return v
save();p=None
try:
 cp=R/'candidates/Deep-ready-decode-ABBA-r159/CPU-static.json';assert sha(cp)==opt.cpu_receipt_sha;c=json.loads(cp.read_bytes());assert c['completed']and c['CPU_only']and not c['Native_loaded']and re.fullmatch(r'[0-9]+\.[0-9]+',c['actual_step']);mp=D/'manifest-r159.json';assert sha(mp)=='eba657d67ce50a3d7ce58a091064031e0e147235e5fbc851744166438f2a8282';m=json.loads(mp.read_bytes())
 for name,h in m['source_SHA256'].items():assert sha(D/name)==h,name
 assert c['source_manifest_SHA256']==sha(mp)and c['Native_SHA256']=='ac9370c8f188bfda8555718c52690b914883fc6b02b9267cbd279be3ecdd3835'
 q.update(CPU_receipt_SHA256=sha(cp),CPU_static_actual_step=c['actual_step'],CPU_static_node=c.get('node'),source_manifest_SHA256=sha(mp));save()
 rs=rows();assert not any(v[1]==label for v in rs)
 for v in rs:
  if v[0].startswith(parent+'.')and v[0].split('.')[-1].isdigit()and v[2].split()[0].rstrip('+')not in ['COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED']:
   assert node not in subprocess.check_output(['scontrol','show','hostnames',v[4]],env=env,text=True,timeout=20).splitlines()
 q['before']=idle('before-r159');save();go={'GPU_authorized':True,'source_published':True,'publication_branch':'bw1000-experiments','publication_commit':opt.publication_commit,'source_manifest_SHA256':sha(mp),'CPU_receipt_SHA256':sha(cp),'scope':'One8card ordinary ac937 B32 same64 QA ABBA fixed ready-decode; firsttoken from prefill, no dynamic service or math change'};gp=D/'root-GO-r159.json';assert not gp.exists();gp.write_text(json.dumps(go,indent=2))
 a=['srun','--jobid='+parent,'--overlap','--exact','--immediate=30','--cpu-bind=none','--kill-on-bad-exit=1','-N1','-n1','-c32','-w',node,'--gres=dcu:8','--mem=128000M','-t','60','--job-name='+label,'--output='+str(R/'logs'/(label+'-%J.log')),'env','WAVE_INSIDE_R2323='+label+'/GPU_inside_r159.sh','bash',str(R/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
 with(R/'logs/Deep-ready-decode-client-r159.log').open('x')as log:p=subprocess.Popen(a,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 q.update(state='running_unique_GPU_follow_handle',GPU_submitted=True,GPU_client_PID=p.pid);save();t=time.monotonic()
 while p.poll()is None and time.monotonic()-t<3900:
  matching=[v for v in rows()if v[1]==label];assert len(matching)<=1
  if matching:q.update(actual_handle=matching[0][0],last_actual_row=matching[0])
  save();time.sleep(10)
 assert p.poll()is not None,'Observation bound; no automatic retry';q.update(GPU_client_returncode=p.returncode,state='GPU_client_terminal',actual_rows=[v for v in rows()if v[1]==label]);save();q['post']=idle('post-r159');q['post_empty']=True;save()
 matches=q['actual_rows'];assert len(matches)==1 and re.fullmatch(r'910374\.[0-9]+',matches[0][0]);handle=matches[0][0];q['actual_handle']=handle;rp=R/'results'/('Deep-ready-decode-ABBA-r159-'+handle+'.json');q['result_path']=str(rp)
 if rp.is_file():
  data=json.loads(rp.read_bytes());q['result_SHA256']=sha(rp);q['summary']={k:data.get(k)for k in ['completed','grid_complete','performance_qualified','engine_worker_shutdown_completed','outcome','error','shutdown_error']};q['ABBA_trials']=len(data.get('measurement_trials',[]));q['Native_execution_passed']=bool(p.returncode==0 and matches[0][2:4]==['COMPLETED','0:0']and data.get('engine_worker_shutdown_completed'));q['measurement_complete']=bool(data.get('grid_complete')and data.get('performance_qualified'));q['optimization_adopted']=False
 q['state']='terminal_followed_no_automatic_replay';save()
except BaseException as e:q.update(error_class=type(e).__name__,error=repr(e),state='failed_before_GPU'if p is None else'unknown_GPU_follow_same_handle');save();raise
finally:save()
