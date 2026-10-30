"""One loaded32-card GLM regular-agent validation. Root publication/CPU pins."""
import argparse,datetime,hashlib,json,os,pathlib,re,socket,subprocess,time,traceback,uuid
from service_budget_r204 import OBSERVER_SECONDS
R=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');D=R/'incoming/GLM-prefill-paired-r241';LABEL='GLM-prefill-paired-r241';PARENT='910376';NODES=['a02r4n12','a02r4n13','a02r4n14','a02r4n15'];sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
a=argparse.ArgumentParser();a.add_argument('--manifest-sha',required=True);a.add_argument('--publication-commit',required=True);a.add_argument('--cpu-receipt-sha',required=True);o=a.parse_args();assert socket.gethostname().split('.')[0]=='a14r2n20'and os.environ['SLURM_JOB_ID']=='910407'
assert re.fullmatch('[0-9a-f]{40}',o.publication_commit)and re.fullmatch('[0-9a-f]{64}',o.cpu_receipt_sha)
assert sha(D/'source-manifest-r204.json')==o.manifest_sha;m=json.loads((D/'source-manifest-r204.json').read_bytes())
for n,h in m['source_SHA256'].items():assert sha(D/n)==h,n
cp=R/'candidates/GLM-prefill-paired-r241/CPU-prepare.json';assert sha(cp)==o.cpu_receipt_sha;c=json.loads(cp.read_bytes());assert c['completed']and c['CPU_only']and c['static_only']and not c['GPU_initialized']and not c['native_model_constructed']and c['source_manifest_SHA256']==o.manifest_sha
assert c['Native_DSO_SHA256']=='038d4ff2e7aee2e6db80f691dd885222f3e3160e25ae9d0e269296daf9596c00'and c['allocation795_proof']['actual_allocation_handle']=='910376.795'and c['allocation795_proof']['normal_owner_count']==32
fixture=R/'results/GLM-regular-fixture-r178a-910407.539/CPU-sizing.json';assert sha(fixture)=='ed2de5e3a0035655ae89f38c65359acf567821b5d254279eebc60847d5f89cbf'
env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_'))and k not in['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL']};TERM={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE'}
watch=R/(LABEL+'-watch.json');intent=R/(LABEL+'-intent.json');assert not watch.exists()and not intent.exists();known=terminal=process=None
run=LABEL+'-'+uuid.uuid4().hex[:16];name=run;w={'run':run,'GPU_submitted':False,'actual_handle':None,'publication_commit':o.publication_commit,'CPU_receipt_SHA256':sha(cp),'source_manifest_SHA256':o.manifest_sha,'Native_SHA256':c['Native_DSO_SHA256'],'CPU_keeper_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'parent':PARENT,'nodes':NODES,'TP':8,'PP':4,'cards':32,'rows':1,'context':32768,'normal_output_budget':4096,'thinking_output_budget':8192,'restarts':0,'Native_lifecycle_qualified':False,'regular_agent_qualified':False}
def save():
 w['UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();p=watch.with_suffix('.tmp');p.write_text(json.dumps(w,indent=2));p.replace(watch)
def norm(s):return s.split()[0].rstrip('+')
def account():return [s.split('|')for s in subprocess.check_output(['sacct','-S','2026-10-27','-j',PARENT,'-n','-P','-o','JobID,JobName%160,State,ExitCode,NodeList%110'],env=env,text=True,timeout=30).splitlines()]
def check_steps(ignore=()):
 rows=account();assert any(x[0]==PARENT and norm(x[2])=='RUNNING'for x in rows)
 for x in rows:
  if x[0].startswith(PARENT+'.')and x[0].split('.')[-1].isdigit()and norm(x[2])not in TERM and x[0]not in ignore:
   nodes=subprocess.check_output(['scontrol','show','hostnames',x[4]],env=env,text=True,timeout=20).splitlines();assert not set(nodes)&set(NODES),x
 return {'parent_RUNNING':True,'foreign_active_target_steps':[]}
def idle(tag):
 args=['srun','--jobid='+PARENT,'--overlap','--exact','--immediate=30','--cpu-bind=none','-N4','-n4','--ntasks-per-node=1','-c1','-w',','.join(NODES),'--gres=none','--mem=512M','-t','00:02:00','--job-name='+name+'-'+tag,'python3',str(D/'resource_probe_r204.py')]
 z=subprocess.run(args,env=env,text=True,capture_output=True,timeout=170);rows=[json.loads(s)for s in z.stdout.splitlines()if s.startswith('{')];assert z.returncode==0 and len(rows)==4 and len({r['actual_step']for r in rows})==1 and {r['node']for r in rows}==set(NODES)and all(r['CPU_only']and not r['GPU_runtime_initialized']and r['KFD_empty']and r['port43843_bindable']for r in rows),(z.returncode,rows,z.stderr[-512:]);return rows
def stop_owned_step():
 global terminal
 if known is None:return False
 rows=[v for v in account()if v[0]==known];assert len(rows)==1 and rows[0][1]==name
 if norm(rows[0][2])in TERM:terminal=rows[0];return False
 w['forced_owned_step_fail_stop']=True;save()
 for sig,seconds in [('TERM',30),('KILL',120)]:
  subprocess.run(['scancel','--signal='+sig,known],env=env,check=True,timeout=20)
  end=time.monotonic()+seconds
  while time.monotonic()<end:
   rows=[v for v in account()if v[0]==known];assert len(rows)==1 and rows[0][1]==name
   if norm(rows[0][2])in TERM:terminal=rows[0];w['terminal']=terminal;save();return True
   time.sleep(2)
 raise TimeoutError('Confirmed owned step did not become terminal after bounded fail-stop')

with intent.open('x')as f:json.dump(w,f,indent=2)
save()
try:
 w['resource_check']=check_steps();w['fresh_before']=idle('before');w['after_CPUprobe_check']=check_steps({r['actual_step']for r in w['fresh_before']});save()
 args=['srun','--jobid='+PARENT,'--overlap','--exact','--immediate=30','--cpu-bind=none','--kill-on-bad-exit=1','-N4','-n4','--ntasks-per-node=1','-c64','-w',','.join(NODES),'--gres=dcu:8','--mem=128000M','-t','06:10:00','--job-name='+name,'--output='+str(R/'logs'/(name+'-%J-%t.log')),'env','GLM_PAIR_MANIFEST_SHA256_R241='+o.manifest_sha,'GLM_SHARED_PREFIX_VALIDATE_R5814=0','GLM_PREFLIGHT_ONLY_R5194=0','GLM_DECODE_ROW_DEDUP_R5386=0','GLM_RUN_R5040='+run,'GLM_BOUNDED_ROPE_R4279=1','GLM_SHARED_CONTROL_R4279=1','GLM_PP_SIZE_R3603=4','GLM_CACHE_ROWS_R3627=1','GLM_CACHE_CAP_R3627=32768','GLM_NETWORK_TRIAL=shca','GLM_PROJECTED_HOST_R4504=0','GLM_PIPELINE_TAIL_NODE=a02r4n15','GLM_EXACT_BANKS_R4188=1','GLM_CHANNELS_R4363=0','GLM_VARIANT_R4363=layer77-5783','INFINILM_OWNER_CPU_WIDTH_R2392=4','WAVE_INSIDE_R2323=GLM-prefill-paired-r241/GPU_inside_r204.sh','USE_WEIGHT_CACHE_R3604=0','GLM_SERVICE_HOST_R143=127.0.0.1','GLM_SERVICE_PORT_R143=43843','GLM_REGULAR_FIXTURE_RECEIPT_R180=/refactor/results/GLM-regular-fixture-r178a-910407.539/CPU-sizing.json','GLM_REGULAR_FIXTURE_RECEIPT_SHA256_R180=ed2de5e3a0035655ae89f38c65359acf567821b5d254279eebc60847d5f89cbf','bash',str(R/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
 with(R/'logs'/(name+'-client.log')).open('x')as log:process=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 w.update(GPU_submitted=True,client_PID=process.pid,state='following_one_unique_once_load_service_handle');save();started=time.monotonic();stop_observed=None
 while time.monotonic()-started<OBSERVER_SECONDS:
  matches=[r for r in account()if r[1]==name];assert len(matches)<=1
  if matches:
   row=matches[0];assert known is None or known==row[0];known=row[0];w.update(actual_handle=known,Slurm=row)
   if norm(row[2])in TERM:terminal=row;w['terminal']=row;save();break
  ready=R/'results'/('GLM-prefill-paired-r241-ready-'+run+'.json')
  if ready.exists():
   q=json.loads(ready.read_bytes());assert q['actual_step']==known and q['Native_DSO_SHA256']==c['Native_DSO_SHA256']and(q['TP'],q['PP'],q['cards'],q['context'],q['rows'])==(8,4,32,32768,1);w['ready_present']=True;w['ready_SHA256']=sha(ready)
  wrapper=R/'results'/('GLM-prefill-paired-r241-client-'+run+'-wrapper.json')
  if known and wrapper.exists():
   cw=json.loads(wrapper.read_bytes())
   if cw.get('completed'):
    if stop_observed is None:stop_observed=time.monotonic()
    if time.monotonic()-stop_observed>(600 if cw.get('passed')else 120):
     stop_owned_step();raise TimeoutError('Owned service did not drain after terminal CPU supervisor')
  save();time.sleep(10)
 else:raise TimeoutError('Same service handle; do not automatically restart')
 w['post']=idle('post');w['post_empty']=True;save();owners=[];logs=[]
 for stage in range(4):
  lp=R/'logs'/(name+'-'+known+'-'+str(stage)+'.log');text=lp.read_text(errors='replace');normal=text.count('GLM_REGULAR_SERVICE_R204_NORMAL '+str(stage));ib=text.count('Using network IBext_v8');fallback=text.count('Using network Socket');logs.append({'stage':stage,'path':str(lp),'SHA256':sha(lp),'normal_markers':normal,'IBext_v8_mentions':ib,'Socket_network_mentions':fallback});assert normal==1 and ib>=1 and fallback==0,logs[-1]
  for rank in range(8):
   p=R/'results'/f'glm-agent-service-r204-{run}-stage{stage}-rank{rank}.json';q=json.loads(p.read_bytes());assert q['completed']and q['normal_reset_close']and q['worker_shutdown_completed']and q['native_service_loop_completed']and q['post_close_Torch_liveness']and q['library_sha256']==c['Native_DSO_SHA256'];owners.append({'stage':stage,'rank':rank,'path':str(p),'SHA256':sha(p),'completed':True,'service_requests':len(q['service_requests']),'minimum_driver_free_bytes':q.get('minimum_driver_free_bytes')})
   if stage==3 and rank==0:assert q['normal_HTTP_shutdown']
 client=R/'results'/('GLM-prefill-paired-r241-client-'+run);sq=json.loads((client/'result.json').read_bytes());wrapper=json.loads(client.with_name(client.name+'-wrapper.json').read_bytes());assert sq['completed']and sq['passed']and sq['attempts']==sq['completed_HTTP_200']==18 and wrapper['completed']and wrapper['passed']and wrapper['single_HTTP_child_reaped']
 plan=[1,2]+[1,2,2,1]*4;signatures=[]
 for stage in range(4):
  for rank in range(8):
   p=R/'results'/f'glm-agent-service-r204-{run}-stage{stage}-rank{rank}.json';q=json.loads(p.read_bytes());assert q['paired_actual_step']==known and q['paired_run']==run and q['paired_source_binding_r241']['source_manifest_SHA256']==o.manifest_sha and q['paired_source_binding_r241']['all_source_files_bound_before_Torch']
   assert not q.get('automatic_CPU_client_group_forced_stop') and q['normal_shutdown']and len(q['service_requests'])==18
   rows=q['service_requests'];assert [v['pair_index']for v in rows]==list(range(18))and[v['prefill_control_window']for v in rows]==plan
   assert len({v['actual_prompt_token_ids_SHA256']for v in rows})==1
   assert all(v['prompt_tokens']==8900 and v['requested_output_budget']==4096 and v['generated_natural_EOS']and v['original_reset_all32_completed']for v in rows)
   signatures.append([(v['pair_index'],v['prefill_control_window'],v['actual_prompt_token_ids_SHA256'],v['generated_token_ids_SHA256'],v['generated_tokens'],v['reason'])for v in rows])
 assert all(v==signatures[0]for v in signatures)
 assert norm(terminal[2])=='COMPLETED'and terminal[3]=='0:0'and not w.get('forced_owned_step_fail_stop')
 w.update(completed=True,owner_results=owners,network_and_normal=logs,Native_lifecycle_qualified=True,paired_measurement_conditions_qualified=True,regular_agent_qualified=False,paired_client_result_SHA256=sha(client/'result.json'),paired_wrapper_SHA256=sha(client.with_name(client.name+'-wrapper.json')),state='terminal_checked_paired_same_prompt_all32_EOS_reset_exit_and_cleanup');save()

except BaseException as error:
 w.update(error_class=type(error).__name__,error=str(error)[:768],state='failed_before_GPU'if process is None else'follow_same_actual_handle_no_restart');save()
 if process is not None and terminal is None and known is not None:
  try:stop_owned_step()
  except BaseException as se:w['fail_stop_error_class']=type(se).__name__;save()
 if terminal and not w.get('post_empty'):
  try:w['error_post']=idle('errorpost');w['post_empty']=True;save()
  except BaseException as ce:w['post_error_class']=type(ce).__name__;save()
 raise
