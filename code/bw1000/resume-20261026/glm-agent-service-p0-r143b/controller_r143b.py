"""One resident16card GLM service under CPU keeper; root source publication GO."""
import argparse,datetime,hashlib,json,os,pathlib,re,socket,subprocess,time,traceback,uuid
R=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');D=R/'incoming/GLM-agent-service-r143b';LABEL='GLM-agent-service-r143b';PARENT='910376';NODES=['a02r4n12','a02r4n13'];sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
ap=argparse.ArgumentParser();ap.add_argument('--manifest-sha',required=True);ap.add_argument('--go-file',required=True);o=ap.parse_args();assert socket.gethostname()=='a14r2n20'and os.environ['SLURM_JOB_ID']=='910407'
assert sha(D/'GPU-control-manifest-r143b.json')==o.manifest_sha;m=json.loads((D/'GPU-control-manifest-r143b.json').read_bytes());g=json.loads(pathlib.Path(o.go_file).read_bytes());assert g['GPU_authorized']is True and g['source_published']is True and g['GPU_manifest_SHA256']==o.manifest_sha and g['publication_branch']=='bw1000-experiments'and re.fullmatch('[0-9a-f]{40}',g['publication_commit'])
for n,h in m['source_SHA256'].items():assert sha(D/n)==h,n
assert sha(D/'source-manifest-r143b.json')==m['CPU_source_manifest_SHA256'];sm=json.loads((D/'source-manifest-r143b.json').read_bytes())
for n,h in sm['source_SHA256'].items():assert sha(D/n)==h,n
cp=R/'candidates/GLM-agent-service-r143b/CPU-prepare.json';assert sha(cp)==g['CPU_receipt_SHA256'];c=json.loads(cp.read_bytes());assert c['completed']and c['CPU_only']and c['static_only']and not c['GPU_initialized']and not c['native_model_constructed']and c['source_manifest_SHA256']==m['CPU_source_manifest_SHA256']
env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_'))and k not in['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL']}
watch=R/(LABEL+'-watch.json');record=R/(LABEL+'-launch.json');assert not watch.exists()and not record.exists();known=None;terminal=None;process=None;TERM={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE'}
run=LABEL+'-'+uuid.uuid4().hex[:16];w={'GPU_submitted':False,'actual_handle':None,'Native_lifecycle_qualified':False,'HTTP_tool_qualified':False,'run':run,'CPU_keeper_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'publication_commit':g['publication_commit'],'GPU_manifest_SHA256':o.manifest_sha,'CPU_receipt_SHA256':sha(cp),'model_path':c['model_path'],'cards':16,'TP':8,'PP':2,'rows':32,'context':640,'restarts':0}
def save():
 w['UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();temp=watch.with_suffix('.tmp');temp.write_text(json.dumps(w,indent=2));temp.replace(watch)
def account():return [s.split('|')for s in subprocess.check_output(['sacct','-S','2026-10-27','-j',PARENT,'-n','-P','-o','JobID,JobName%110,State,ExitCode,NodeList%110'],env=env,text=True,timeout=30).splitlines()]
def norm(s):return s.split()[0].rstrip('+')
def idle(tag):
 p=R/'incoming/verify_gpu_idle_r5738.py';assert sha(p)=='45b7d14902965ca8569000a8c20882d589c7b2468f365e96d0cb552cf6f57b53'
 a=['srun','--jobid='+PARENT,'--overlap','--exact','--immediate=30','--cpu-bind=none','-N2','-n2','--ntasks-per-node=1','-c1','-w',','.join(NODES),'--gres=none','--mem=512M','-t','00:02:00','--job-name='+LABEL+'-'+tag,'python3',str(p)];z=subprocess.run(a,env=env,text=True,capture_output=True,timeout=170);rows=[json.loads(s)for s in z.stdout.splitlines()if s.startswith('{')];assert z.returncode==0 and len(rows)==2 and {x['node']for x in rows}==set(NODES)and all(not x['kfd_pids']for x in rows),(z.returncode,rows,z.stderr);return rows
with record.open('x')as f:json.dump(w,f,indent=2)
save()
try:
 rows=account();assert any(x[0]==PARENT and norm(x[2])=='RUNNING'for x in rows)and not any(x[1]==LABEL for x in rows)
 for x in rows:
  if x[0].startswith(PARENT+'.')and x[0].split('.')[-1].isdigit()and norm(x[2])not in TERM:
   current=subprocess.check_output(['scontrol','show','hostnames',x[4]],env=env,text=True,timeout=20).splitlines();assert not set(current)&set(NODES),x
 w['fresh_before']=idle('before');w['watch_ready_before_GPU']=True;save()
 a=['srun','--jobid='+PARENT,'--overlap','--exact','--immediate=30','--cpu-bind=none','--kill-on-bad-exit=1','-N2','-n2','--ntasks-per-node=1','-c64','-w',','.join(NODES),'--gres=dcu:8','--mem=128000M','-t','02:30:00','--job-name='+LABEL,'--output='+str(R/'logs'/(LABEL+'-%J-%t.log')),'env','GLM_SHARED_PREFIX_VALIDATE_R5814=0','GLM_PREFLIGHT_ONLY_R5194=0','GLM_DECODE_ROW_DEDUP_R5386=0','GLM_RUN_R5040='+run,'GLM_BOUNDED_ROPE_R4279=1','GLM_SHARED_CONTROL_R4279=1','GLM_PP_SIZE_R3603=2','GLM_CACHE_ROWS_R3627=32','GLM_CACHE_CAP_R3627=640','GLM_NETWORK_TRIAL=shca','GLM_PIPELINE_TAIL_NODE=a02r4n13','GLM_EXACT_BANKS_R4188=1','GLM_CHANNELS_R4363=0','GLM_VARIANT_R4363=layer77-5783','INFINILM_OWNER_CPU_WIDTH_R2392=4','WAVE_INSIDE_R2323=GLM-agent-service-r143b/GPU_inside_r143b.sh','USE_WEIGHT_CACHE_R3604=0','GLM_SERVICE_HOST_R143=127.0.0.1','GLM_SERVICE_PORT_R143=43843','bash',str(R/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
 with(R/'logs'/(LABEL+'-client.log')).open('x')as log:process=subprocess.Popen(a,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 w.update(GPU_submitted=True,client_PID=process.pid,args=a,state='following_one_unique_service_handle');save();start=time.monotonic()
 while time.monotonic()-start<9250:
  matches=[x for x in account()if x[1]==LABEL];assert len(matches)<=1
  if matches:
   row=matches[0];assert known is None or known==row[0];known=row[0];w.update(actual_handle=known,Slurm=row)
   if norm(row[2])in TERM:terminal=row;w['terminal']=row;save();break
  p=R/'results'/('GLM-agent-service-r143b-ready-'+run+'.json')
  if p.exists():
   ready=json.loads(p.read_bytes());assert ready['actual_step']==known and ready['Native_DSO_SHA256']==sm['Native_DSO_SHA256']and ready['context']==640;w['service_ready']=ready;w['client_execution']='Run CPU-only Slurm step on a02r4n13 against http://127.0.0.1:43843; P0 client independent frozen artifact.';w['normal_stop_marker']=str(R/'control'/('GLM-agent-service-r143b-stop-'+run+'.json'))
  save();time.sleep(10)
 else:raise TimeoutError('Observe same handle; do not automatically restart')
 w['post']=idle('post');w['post_empty']=True;results=[]
 for stage in[0,1]:
  for rank in range(8):
   p=R/'results'/f'glm-agent-service-r143b-{run}-stage{stage}-rank{rank}.json';raw=p.read_bytes();q=json.loads(raw);results.append({'stage':stage,'rank':rank,'path':str(p),'SHA256':hashlib.sha256(raw).hexdigest(),'completed':q.get('completed'),'normal_reset_close':q.get('normal_reset_close'),'worker_shutdown_completed':q.get('worker_shutdown_completed'),'requests':len(q.get('service_requests',[]))});assert q['completed']and q['normal_reset_close']and q['worker_shutdown_completed']and q['native_service_loop_completed']
   if stage==1 and rank==0:assert q['normal_HTTP_shutdown']
 w['owner_results']=results;w['Native_lifecycle_qualified']=terminal[2:4]==['COMPLETED','0:0'];w['state']='terminal_checked_P0_client_results_separate';save()
except BaseException as error:
 w.update(error=repr(error),traceback=traceback.format_exc(),state='failed_before_GPU'if process is None else'follow_same_actual_handle');save()
 if terminal:
  try:w['error_post']=idle('errorpost');save()
  except BaseException as ce:w['post_error']=repr(ce);save()
 raise
