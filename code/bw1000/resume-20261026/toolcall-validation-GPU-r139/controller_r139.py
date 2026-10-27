"""CPU keeper: qualify actual CPU receipt; fresh idle; exactly one GPU task."""
import pathlib,json,hashlib,subprocess,os,time,datetime,traceback,re
R=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');D=R/'incoming/toolcall-validation-GPU-r139';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();label='toolcall-Qwen06-GPU-r139';watch=R/'results/toolcall-Qwen06-r139-watch.json';assert not watch.exists()
env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_','CUDA_','HIP_','ROCR_','GPU_'))or k in ['SLURM_CONF','SLURM_CONF_SERVER']};q={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'CPU_keeper':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'GPU_submitted':False,'passed':False,'actual_handle':None};actual=None
def save():
 q['heartbeat_UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();p=watch.with_suffix('.tmp');p.write_text(json.dumps(q,indent=2));p.replace(watch)
def account(name):
 rows=[s.split('|')for s in subprocess.check_output(['sacct','-S','2026-10-27','-j','910374','-n','-P','-o','JobID,JobName%110,State,ExitCode,NodeList%100'],env=env,text=True,timeout=25).splitlines()];return [r for r in rows if r[1]==name]
def idle(suffix):
 name=label+'-'+suffix;a=['srun','--jobid=910374','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c1','-w','a02r4n01','--gres=none','--mem=512M','-t','00:02:00','--job-name='+name,'python3',str(R/'incoming/verify_gpu_idle_r5738.py')];p=subprocess.run(a,env=env,text=True,capture_output=True,timeout=160);rows=[json.loads(s)for s in p.stdout.splitlines()if s.startswith('{')];assert p.returncode==0 and len(rows)==1 and rows[0]['node']=='a02r4n01'and not rows[0]['kfd_pids'];return rows[0]
active={'RUNNING','PENDING','CONFIGURING','COMPLETING','SUSPENDED'};norm=lambda s:s.split()[0].rstrip('+');save()
try:
 m=json.loads((D/'GPU-source-manifest.json').read_bytes());g=json.loads((D/'root-GO-after-CPU.json').read_bytes());assert g['GPU_authorized']is True and g['source_published']is True and re.fullmatch('[0-9a-f]{40}',g['publication_commit'])and g['publication_branch']=='bw1000-experiments'and g['GPU_manifest_SHA256']==sha(D/'GPU-source-manifest.json')
 for n,h in m['source_SHA256'].items():assert sha(D/n)==h,n
 cp=R/'candidates/toolcall-Qwen06-r138/CPU-preflight.json';assert sha(cp)==g['CPU_receipt_SHA256'];c=json.loads(cp.read_bytes());assert c['completed']and c['CPU_only']and c['static_only']and c['static_overlay_AST_passed']and not c['standard_server_imported']and c['source_manifest_SHA256']==m['CPU_source_manifest_SHA256']and not c['GPU_initialized']
 assert re.fullmatch(r'910374\.\d+',c['actual_step']);assert c['packages_imported_by_this_preflight']==['stdlib_only'];q['CPU_receipt_SHA256']=sha(cp);q['CPU_scope']='STATIC_ONLY_RUNTIME_IMPORT_PENDING_GPU'
 assert not account(label),'unique task previously exists: inspect, never restart'
 parent=subprocess.check_output(['scontrol','show','job','910374','-o'],env=env,text=True,timeout=20);assert 'UserId=aclsylqidf('in parent and 'JobState=RUNNING'in parent
 q['fresh_before']=idle('before');q['ready_before_GPU']=True;save()
 outer=R/'incoming/resource_wave_local_r3097_outer.sh';assert sha(outer)=='d3ce6c3196d5728a5f97d459d070fb7720df76d25cadc6be8af0dbb6eca75dce'
 args=['srun','--jobid=910374','--overlap','--exact','--immediate=30','--cpu-bind=none','--kill-on-bad-exit=1','-N1','-n1','-c8','-w','a02r4n01','--gres=dcu:1','--mem=32768M','-t','00:25:00','--job-name='+label,'--output='+str(R/'logs'/(label+'-%J.log')),'env','WAVE_INSIDE_R2323=toolcall-validation-GPU-r139/GPU_inside_r139.sh','bash',str(outer),'probe']
 with(R/'logs'/(label+'-client.log')).open('x')as f:p=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 q.update(GPU_submitted=True,client_PID=p.pid);save();deadline=time.monotonic()+1570
 while time.monotonic()<deadline:
  rows=account(label);assert len(rows)<=1
  if rows:
   row=rows[0];assert actual is None or actual==row[0];actual=row[0];q.update(actual_handle=actual,Slurm=row)
   if norm(row[2])not in active:q['terminal']=row;save();break
  save();time.sleep(10)
 else:raise TimeoutError('same owned handle watch expired; no automatic retry')
 q['post']=idle('post');q['post_empty']=True;rp=R/'results'/('toolcall-Qwen06-r139-'+actual+'.json');q['result_path']=str(rp);q['result_SHA256']=sha(rp);res=json.loads(rp.read_bytes());q['summary']={k:res.get(k)for k in ['completed','passed','cases','actual_real_tool_executions','normal_server_shutdown','error','client_evidence_directory','actual_server_DSO_SHA256','runtime_import_qualified','preweight_smoke_SHA256','preweight_smoke_exit','preweight_smoke_log']};q['passed']=q['terminal'][2:4]==['COMPLETED','0:0']and res['passed']and res['normal_server_shutdown'];q['completed']=True;save();assert q['passed']
except BaseException as e:
 q.update(error=repr(e),traceback=traceback.format_exc());save()
 if q['GPU_submitted']and actual:
  rows=account(label)
  if len(rows)==1 and rows[0][0]==actual and norm(rows[0][2])in active:
   q['before_precise_owned_cancel']=rows[0];save();subprocess.run(['scancel',actual],env=env,check=True,timeout=25)
   deadline=time.monotonic()+150
   while time.monotonic()<deadline:
    rows=account(label)
    if rows and rows[0][0]==actual and norm(rows[0][2])not in active:q['terminal_after_owned_cancel']=rows[0];save();break
    time.sleep(5)
  try:q['error_post']=idle('errorpost');q['error_post_empty']=True;save()
  except BaseException as cleanup_error:q['error_post_failure']=repr(cleanup_error);save()
 raise
