"""Single original32-owner allocation under existing CPU keeper; never SSH/restart."""
import argparse,datetime,hashlib,json,os,pathlib,re,socket,subprocess,time,traceback,uuid
R=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');D=R/'incoming/GLM-PP4-allocation-launch-r188';S=R/'incoming/GLM-PP4-capacity-r186';PARENT='910376';NODES=['a02r4n12','a02r4n13','a02r4n14','a02r4n15'];LABEL='GLM-PP4-allocation-r188';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
a=argparse.ArgumentParser();a.add_argument('--manifest-sha',required=True);a.add_argument('--publication-commit',required=True);a.add_argument('--native-cpu-receipt-path',default='/refactor/candidates/GLM-PP4-registry-repair-r184b/CPU-build.json');a.add_argument('--native-cpu-receipt-sha',required=True);o=a.parse_args()
assert socket.gethostname().split('.')[0]=='a14r2n20'and os.environ['SLURM_JOB_ID']=='910407'
assert re.fullmatch('[0-9a-f]{40}',o.publication_commit)and re.fullmatch('[0-9a-f]{64}',o.native_cpu_receipt_sha)
assert not(R/'GLM-PP4-allocation-r188-intent.json').exists()and not(R/'GLM-PP4-allocation-r188-watch.json').exists()
assert sha(D/'launch-manifest-r188.json')==o.manifest_sha;m=json.loads((D/'launch-manifest-r188.json').read_bytes())
for n,h in m['files'].items():assert sha(D/n)==h,n
assert sha(S/'source-manifest-r186.json')==m['allocation_source_manifest_SHA256'];sm=json.loads((S/'source-manifest-r186.json').read_bytes())
for n,h in sm['files'].items():assert sha(S/n)==h,n
assert o.native_cpu_receipt_path=='/refactor/candidates/GLM-PP4-registry-repair-r184b/CPU-build.json'and o.native_cpu_receipt_sha=='6aa62f788aad6080027d665f30d7c9824c07cb1866de905397fef15c56b26529'
# This CPU keeper runs the one necessary recursive metadata preflight in the same
# container build context. No rocm/devices, no dlopen, no separate Slurm job.
app='/public/software/apps/apptainer/1.3.4/bin/apptainer';oldroot=pathlib.Path('/public/home/aclsylqidf/hygon-inference-20261013')
cpuargs=[app,'exec','--home',str(oldroot/'build-deps/root')+':/root','--bind',str(oldroot)+':/old:ro','--bind',str(R)+':/refactor','--bind','/opt/hyhal:/opt/hyhal:ro','--bind','/public/home/aclsylqidf/shared:/shared:ro',str(R/'containers/dtk26-sandbox-r1'),'/bin/bash','--noprofile','--norc','/refactor/incoming/GLM-PP4-capacity-r186/CPU_inside_r186.sh']
cpurun=subprocess.run(cpuargs,capture_output=True,text=True,timeout=210)
assert cpurun.returncode==0,('Native ancestry CPU preflight',cpurun.stderr[-512:])
cp=S/'CPU-result-r186.json';o.cpu_receipt_sha=sha(cp);c=json.loads(cp.read_bytes());assert c['completed']and c['CPU_only']and not c['GPU_initialized']and not c['dlopen']and c['source_manifest_SHA256']==m['allocation_source_manifest_SHA256']and c['parent_metadata']['library_SHA256']=='038d4ff2e7aee2e6db80f691dd885222f3e3160e25ae9d0e269296daf9596c00'

assert sha(R/'incoming/resource_wave_local_r3097_outer.sh')==m['outer_SHA256']
env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_'))and k not in['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL']}
TERM={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE'}
watch=R/(LABEL+'-watch.json');intent=R/(LABEL+'-intent.json');assert not watch.exists()and not intent.exists();known=terminal=process=None
run=LABEL+'-'+uuid.uuid4().hex[:16];name=run;w={'run':run,'GPU_submitted':False,'actual_handle':None,'publication_commit':o.publication_commit,'CPU_receipt_SHA256':sha(cp),'launch_manifest_SHA256':o.manifest_sha,'allocation_source_manifest_SHA256':m['allocation_source_manifest_SHA256'],'CPU_keeper_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'parent':PARENT,'nodes':NODES,'TP':8,'PP':4,'cards':32,'context':32768,'rows':1,'weights_loaded':False,'forward_called':False,'restarts':0,'qualified':False}
def save():
 w['UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();tmp=watch.with_suffix('.tmp');tmp.write_text(json.dumps(w,indent=2));tmp.replace(watch)
def norm(s):return s.split()[0].rstrip('+')
def account():return [s.split('|')for s in subprocess.check_output(['sacct','-S','2026-10-27','-j',PARENT,'-n','-P','-o','JobID,JobName%160,State,ExitCode,NodeList%110'],env=env,text=True,timeout=30).splitlines()]
def check_steps(ignore_completed_probe=()):
 rows=account();assert any(x[0]==PARENT and norm(x[2])=='RUNNING'for x in rows)
 for x in rows:
  if x[0].startswith(PARENT+'.')and x[0].split('.')[-1].isdigit()and norm(x[2])not in TERM:
   # Only our synchronous CPU-only probe has returned0; sacct may lag its exit.
   if x[0]in ignore_completed_probe:continue
   nodes=subprocess.check_output(['scontrol','show','hostnames',x[4]],env=env,text=True,timeout=20).splitlines();assert not set(nodes)&set(NODES),x
 return {'parent_RUNNING':True,'target_nodes_active_steps':[],'observed_rows':len(rows)}
def idle(tag):
 args=['srun','--jobid='+PARENT,'--overlap','--exact','--immediate=30','--cpu-bind=none','-N4','-n4','--ntasks-per-node=1','-c1','-w',','.join(NODES),'--gres=none','--mem=512M','-t','00:02:00','--job-name='+name+'-'+tag,'python3',str(D/'resource_probe_r188.py')]
 z=subprocess.run(args,env=env,text=True,capture_output=True,timeout=170);rows=[json.loads(s)for s in z.stdout.splitlines()if s.startswith('{')]
 assert z.returncode==0 and len(rows)==4 and len({x['actual_step']for x in rows})==1 and {x['node']for x in rows}==set(NODES)and all(x['CPU_only']and not x['GPU_runtime_initialized']and x['KFD_empty']and x['port43843_bindable']for x in rows),(z.returncode,rows,z.stderr[:512]);return rows
with intent.open('x')as f:json.dump(w,f,indent=2)
save()
try:
 w['resource_step_check']=check_steps();w['fresh_before']=idle('before');w['resource_step_check_after_probe']=check_steps({x['actual_step']for x in w['fresh_before']});save()
 args=['srun','--jobid='+PARENT,'--overlap','--exact','--immediate=30','--cpu-bind=none','--kill-on-bad-exit=1','-N4','-n4','--ntasks-per-node=1','-c64','-w',','.join(NODES),'--gres=dcu:8','--mem=128000M','-t','00:12:00','--job-name='+name,'--output='+str(R/'logs'/(name+'-%J-%t.log')),'env','GLM_SHARED_PREFIX_VALIDATE_R5814=0','GLM_PREFLIGHT_ONLY_R5194=0','GLM_DECODE_ROW_DEDUP_R5386=0','GLM_BOUNDED_ROPE_R4279=1','GLM_SHARED_CONTROL_R4279=1','GLM_PP_SIZE_R3603=4','GLM_CACHE_ROWS_R3627=1','GLM_CACHE_CAP_R3627=32768','GLM_NETWORK_TRIAL=shca','GLM_PROJECTED_HOST_R4504=0','GLM_PIPELINE_TAIL_NODE=a02r4n15','GLM_EXACT_BANKS_R4188=1','GLM_CHANNELS_R4363=0','GLM_VARIANT_R4363=layer77-5783','INFINILM_OWNER_CPU_WIDTH_R2392=4','WAVE_INSIDE_R2323=GLM-PP4-allocation-launch-r188/allocation_inside_r188.sh','USE_WEIGHT_CACHE_R3604=0','GLM_ALLOCATION_PUBLICATION_COMMIT_R188='+o.publication_commit,'GLM_ALLOCATION_CPU_RECEIPT_SHA256_R188='+o.cpu_receipt_sha,'GLM_ALLOCATION_GO_R186=/refactor/control/'+name+'-GO.json','bash',str(R/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
 with(R/'logs'/(name+'-client.log')).open('x')as log:process=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 w.update(GPU_submitted=True,client_PID=process.pid,state='following_single_allocation_handle');save();start=time.monotonic()
 while time.monotonic()-start<950:
  matches=[x for x in account()if x[1]==name];assert len(matches)<=1
  if matches:
   row=matches[0];assert known is None or known==row[0];known=row[0];w.update(actual_handle=known,Slurm=row)
   if norm(row[2])in TERM:terminal=row;w['terminal']=row;save();break
  save();time.sleep(5)
 else:raise TimeoutError('Same allocation handle observation expired; no resubmission')
 w['post']=idle('post');w['post_empty']=True;save()
 assert norm(terminal[2])=='COMPLETED'and terminal[3]=='0:0',terminal
 receipts=[];net=[]
 for stage in range(4):
  log=R/'logs'/(name+'-'+known+'-'+str(stage)+'.log');text=log.read_text(errors='replace');normal=text.count('GLM_ALLOCATION_R186_NORMAL '+str(stage));ib=text.count('Using network IBext_v8');fallback=text.count('Using network Socket')
  net.append({'stage':stage,'node':NODES[stage],'path':str(log),'SHA256':sha(log),'normal_markers':normal,'IBext_v8_mentions':ib,'Socket_network_mentions':fallback});assert normal==1 and ib>=1 and fallback==0,net[-1]
  for rank in range(8):
   p=R/'results'/f'GLM-allocation-r186-{known}-stage{stage}-rank{rank}.json';q=json.loads(p.read_bytes())
   assert q['completed']and q['normal_destroy']and q['normal_shutdown']and q['Torch_postclose_liveness']and q['all32_constructor_memory_floor_passed']and q['world32_integer_SUM_exact']and not q['weights_loaded']and not q['forward_called']and not q['finish_called']
   assert q['Native_SHA256']=='038d4ff2e7aee2e6db80f691dd885222f3e3160e25ae9d0e269296daf9596c00'and q['source_manifest_SHA256']==m['allocation_source_manifest_SHA256']and q['CPU_receipt_SHA256']==o.cpu_receipt_sha
   receipts.append({'stage':stage,'rank':rank,'path':str(p),'SHA256':sha(p),'completed':True,'memory_after_create':q['memory_after_create'],'memory_after_trim':q['memory_after_trim'],'memory_after_shutdown':q['memory_after_shutdown'],'constructor_logical_census':q['constructor_logical_census'],'actual_partition':q['actual_partition']})
 w.update(owner_results=receipts,network_and_normal=net,qualified=True,qualification_scope='Original32-owner PP4/32K constructor capacity/lifecycle only; no weights,finish,forward or realagent',state='allocation_terminal_qualified_loaded_fit_UNKNOWN');save()
except BaseException as exc:
 w.update(error_class=type(exc).__name__,error=str(exc)[:768],state='failed_before_GPU'if process is None else'follow_same_actual_handle_no_restart');save()
 if terminal and not w.get('post_empty'):
  try:w['error_post']=idle('errorpost');w['post_empty']=True;save()
  except BaseException as ce:w['post_error_class']=type(ce).__name__;save()
 raise
