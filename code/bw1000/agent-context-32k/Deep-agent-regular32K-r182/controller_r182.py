"""One prepared32K resident run. Root publication/GO, owned handles, no retry."""
import pathlib,json,hashlib,subprocess,os,time,re,datetime,traceback
R=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');D=R/'incoming/Deep-agent-regular32K-r182';LABEL='Deep-agent-regular32K-r182';NODE='a02r4n01';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();W=R/(LABEL+'-watch.json');assert not W.exists()
env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_','CUDA_','HIP_','ROCR_','GPU_'))};w={'GPU_submitted':False,'actual_handle':None,'completed':False,'restarts':0,'CPU_keeper_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'rows':1,'capacity':32768,'cards':8};proc=None;actual=None
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
def save():
 w['UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();tmp=W.with_suffix('.tmp');tmp.write_text(json.dumps(w,indent=2));tmp.replace(W)
def account():
 rows=[x.split('|')for x in subprocess.check_output(['sacct','-S','2026-10-28','-j','910374','-n','-P','-o','JobID,JobName%120,State,ExitCode,NodeList%100'],env=env,text=True,timeout=30).splitlines()];return [x for x in rows if x[1]==LABEL]
def idle(suffix):
 p=R/'incoming/verify_gpu_idle_r5738.py';assert sha(p)=='45b7d14902965ca8569000a8c20882d589c7b2468f365e96d0cb552cf6f57b53'
 argv=['srun','--jobid=910374','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c1','-w',NODE,'--gres=none','--mem=512M','-t','00:02:00','--job-name='+LABEL+'-'+suffix,'python3',str(p)];z=subprocess.run(argv,env=env,text=True,capture_output=True,timeout=160);rows=[json.loads(x)for x in z.stdout.splitlines()if x.startswith('{')];assert z.returncode==0 and len(rows)==1 and rows[0]['node']==NODE and rows[0]['kfd_pids']==[] and rows[0]['CUDA_not_initialized_by_probe'];return rows[0]
save()
try:
 m=json.loads((D/'manifest-r182.json').read_bytes());g=json.loads((D/'root-GO-r182.json').read_bytes());assert g['GPU_authorized']is True and g['source_published']is True and g['source_manifest_SHA256']==sha(D/'manifest-r182.json')and re.fullmatch('[0-9a-f]{40}',g['publication_commit'])and g['publication_branch']=='bw1000-experiments'
 for p,h in m['source_SHA256'].items():assert sha(D/p)==h,p
 b=json.loads((D/'Native-binding.json').read_bytes());assert b['CPU_receipt_SHA256']=='17e94923049bfe910a3cb4810fb69d9c9e4bdebc9265ad2c21d5eaa65c2d39ac'and b['library_SHA256']=='232fb7e2566f7d1961fa433d9e45660bfe760ec726676dcb7f1f406d857c801f';cp=R/'candidates/Deep-agent-regular32K-r182/CPU-preflight.json'
 outer=R/'incoming/resource_wave_local_r3097_outer.sh';assert sha(outer)=='d3ce6c3196d5728a5f97d459d070fb7720df76d25cadc6be8af0dbb6eca75dce'
 if not cp.exists():
  # Necessary private-overlay/official-tokenizer data preparation only.
  prep=['srun','--jobid=910374','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c8','-w',NODE,'--gres=none','--mem=16384M','-t','00:06:00','--job-name='+LABEL+'-prepare','--output='+str(R/'logs'/(LABEL+'-prepare-%J.log')),'env','WAVE_INSIDE_R2323=Deep-agent-regular32K-r182/CPU_inside_r182.sh','bash',str(outer),'probe'];z=subprocess.run(prep,env=env,text=True,capture_output=True,timeout=350);w['necessary_CPU_prepare']={'returncode':z.returncode,'elapsed_outputs':z.stdout[-2000:]};save();assert z.returncode==0
 c=json.loads(cp.read_bytes());assert c['completed']and c['CPU_only']and not c['Torch_imported']and c['source_manifest_SHA256']==sha(D/'manifest-r182.json')and c['Native_binding_SHA256']==sha(D/'Native-binding.json')and c['new_Native_SHA256']==b['library_SHA256']
 assert not account(),'Existing exact run label: follow the same task; never restart'
 parent=subprocess.check_output(['scontrol','show','job','910374','-o'],env=env,text=True,timeout=20);assert 'JobState=RUNNING'in parent and 'UserId=aclsylqidf('in parent
 w.update(source_manifest_SHA256=sha(D/'manifest-r182.json'),Native_binding_SHA256=sha(D/'Native-binding.json'),CPU_prepare_receipt_SHA256=sha(cp),new_Native_SHA256=b['library_SHA256'],publication_commit=g['publication_commit'],fresh_before=idle('before'));save()
 args=['srun','--jobid=910374','--overlap','--exact','--immediate=30','--cpu-bind=none','--kill-on-bad-exit=1','-N1','-n1','-c32','-w',NODE,'--gres=dcu:8','--mem=128000M','-t','04:15:00','--job-name='+LABEL,'--output='+str(R/'logs'/(LABEL+'-%J.log')),'env','WAVE_INSIDE_R2323=Deep-agent-regular32K-r182/GPU_inside_r182.sh','bash',str(outer),'probe']
 with(R/'logs'/(LABEL+'-client.log')).open('x')as log:proc=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 w.update(GPU_submitted=True,client_PID=proc.pid,state='following_one_new_Native32K_regular_then_long_output');save();deadline=time.monotonic()+15600
 while time.monotonic()<deadline:
  rows=account();assert len(rows)<=1
  if rows:
   row=rows[0];assert actual is None or actual==row[0];actual=row[0];w.update(actual_handle=actual,Slurm=row)
   if row[2].split()[0].rstrip('+')in TERMINAL:w['terminal']=row;save();break
  save();time.sleep(15)
 else:raise TimeoutError('Owned one-run observer deadline; no automatic retry')
 w.update(post=idle('post'),post_empty=True)
 result=R/'results'/('Deep-agent-regular32K-r182-'+actual+'.json');w['result_path']=str(result)
 if result.is_file():
  q=json.loads(result.read_bytes());w.update(result_SHA256=sha(result),summary={k:q.get(k)for k in ['completed','passed','normal_server_shutdown','normal_Native_shutdown','capacity','rows','actual_preload_create','owner_exit_observation']})
 log=R/'logs'/(LABEL+'-'+actual+'.log');text=log.read_text(errors='replace');w['normal_Native_marker_count']=text.count('DEEP_REGULAR_R182_NATIVE_NORMAL');w['completed']=True;w['scope']='Actual new Native32K regular agent and long-output evidence; no performance/benchmark-wide quality claim';save()
except BaseException as exc:
 w.update(error=repr(exc),traceback=traceback.format_exc());save()
 if actual is None and w['GPU_submitted']:
  try:
   rows=account()
   if len(rows)==1:actual=rows[0][0];w.update(actual_handle=actual,discovered_during_error=rows[0]);save()
   else:w['owned_handle_status']='UNKNOWN; root must resolve submitted exact label, no unsafe cancellation';save()
  except BaseException as e:w['owned_handle_lookup_error']=repr(e);save()
 if actual is not None:
  rows=account()
  if len(rows)==1 and rows[0][0]==actual and rows[0][2].split()[0].rstrip('+')not in TERMINAL:
   w['before_owned_cancel']=rows[0];save();subprocess.run(['scancel',actual],env=env,check=True,timeout=30)
   deadline=time.monotonic()+180
   while time.monotonic()<deadline:
    rows=account()
    if rows and rows[0][0]==actual and rows[0][2].split()[0].rstrip('+')in TERMINAL:w['terminal_after_owned_cancel']=rows[0];break
    time.sleep(5)
  try:w.update(error_post=idle('error-post'),post_empty=True)
  except BaseException as e:w['post_error']=repr(e)
  save()
 raise
