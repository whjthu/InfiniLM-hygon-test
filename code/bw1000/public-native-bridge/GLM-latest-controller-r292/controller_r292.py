"""One separately authorized original32card residency. No retry/reload or import-only GPU job."""
import argparse,pathlib,json,hashlib,subprocess,os,time,re,datetime,socket
R=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');D=pathlib.Path(__file__).resolve().parent;S=R/'incoming/GLM-latest-full-owner-r290';LABEL='GLM-latest99-full-Model-r292';NODES=['a02r4n12','a02r4n13','a02r4n14','a02r4n15'];W=R/'results'/(LABEL+'-watch.json');TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'};sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
OWNER_SHA='903053b1f6d8031bc03c698c7bd9498a8cab8b0802ff19f4c88a7b4d698e7674';CPU_SHA='901f34e710ae963d8e76e81581090c832a89f3d551e67d78ef02debd7a8ed270'
def validate_scope(g):
 if g.get('GPU_authorized')is not True:raise PermissionError('Explicit additional GLM GPU execution GO required')
 assert g['publication_branch']=='bw1000-experiments'and re.fullmatch('[0-9a-f]{40}',g['publication_commit']);assert g['owner_manifest_SHA256']==OWNER_SHA and re.fullmatch('[0-9a-f]{64}',g['controller_manifest_SHA256']);assert g['GPU_parent']==910376 and g['GPU_nodes']==NODES and g['CPU_keeper_parent']==910407 and g['CPU_keeper_node']=='a14r2n20'
 for k in ['GPU_parent','CPU_keeper_parent','cards','TP','PP','context','ordinary_budget','thinking_budget','max_HTTP','max_Native','GPU_max_s','Native_max_s','CPU_keeper_max_s']:assert type(g[k])is int
 assert(g['cards'],g['TP'],g['PP'],g['context'],g['ordinary_budget'],g['thinking_budget'],g['max_HTTP'],g['max_Native'])==(32,8,4,32768,4096,8192,121,120)
 assert g['GPU_max_s']==29400 and g['Native_max_s']==28800 and g['CPU_keeper_max_s']==32400 and g['one_attempt']is True and g['no_retry_reload']is True
 return g

def verify_sources(g):
 validate_scope(g);assert sha(D/'SOURCE_MANIFEST.json')==g['controller_manifest_SHA256'];m=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());assert all(sha(D/n)==h for n,h in m['files_SHA256'].items());assert sha(S/'SOURCE_MANIFEST.json')==OWNER_SHA;sm=json.loads((S/'SOURCE_MANIFEST.json').read_bytes());assert all(sha(S/n)==h for n,h in sm['files_SHA256'].items());cp=R/'candidates/GLM-owner-entry-r291/CPU-prepared.json';assert sha(cp)==CPU_SHA;c=json.loads(cp.read_bytes());assert c['completed']and c['passed']and c['CPU_only']and c['GPU']==c['Model']==c['production_HTTP']==0 and c['owner_source_manifest_SHA256']==OWNER_SHA and c['actual_preweight_guarded_import_and_original_deadline_PASS']and len(c['cases'])==14 and all(v['passed']for v in c['cases'])and c['cleanup_before_thread_allocation']['errors']==[] and c['cleanup_parent_exits_TERM_child_requires_group_KILL']['passed']
 assert sha(R/'incoming/resource_wave_local_r3097_outer.sh')=='d3ce6c3196d5728a5f97d459d070fb7720df76d25cadc6be8af0dbb6eca75dce'

def clean_env():return{k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_','CUDA_','HIP_','ROCR_','GPU_'))or k in ['SLURM_CONF','SLURM_CONF_SERVER']}
def parent(parent,node,env):
 z=subprocess.check_output(['scontrol','show','job',str(parent),'-o'],env=env,text=True,timeout=25);assert 'UserId=aclsylqidf('in z and 'JobState=RUNNING'in z;nodes=re.search(r'(?:^| )NodeList=([^ ]+)',z).group(1);expanded=subprocess.check_output(['scontrol','show','hostnames',nodes],env=env,text=True,timeout=15).splitlines();assert node in expanded

def validate_account_row(row,expanded):
 assert re.fullmatch(r'910376[.]\d+',row[0])and row[1]==LABEL and len(expanded)==4 and set(expanded)==set(NODES),'Accounting row is not the exact owned parent/label/nodes'

def main():
 p=argparse.ArgumentParser();p.add_argument('--execution-go',required=True);a=p.parse_args();g=json.loads(pathlib.Path(a.execution_go).read_bytes());verify_sources(g);assert os.environ['SLURM_JOB_ID']=='910407'and socket.gethostname().split('.')[0]=='a14r2n20';assert not W.exists();env=clean_env();parent(910407,'a14r2n20',env);[parent(910376,node,env)for node in NODES]
 w={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':False,'GPU_submitted':False,'actual_GPU':None,'CPU_keeper_actual':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'publication_commit':g['publication_commit'],'owner_manifest_SHA256':OWNER_SHA,'controller_manifest_SHA256':g['controller_manifest_SHA256'],'scope':'Latest99 complete normal+multicall+fixedGSM32+fixedMMLU32 same residency; bufferedHTTP no incrementalSLO claim','no_retry_reload':True,'GPU_max_s':29400,'CPU_keeper_max_s':32400,'Model_qualified':False};proc=None;actual=None
 def save():
  w['observed_UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();tmp=W.with_suffix('.tmp');tmp.write_text(json.dumps(w,indent=2)+chr(10));tmp.replace(W)
 def account():
  rows=[x.split('|')for x in subprocess.check_output(['sacct','-S','2026-10-31','-j','910376','-n','-P','-o','JobID,JobName%120,State,ExitCode,NodeList%100'],env=env,text=True,timeout=25).splitlines()];rows=[v for v in rows if v[1]==LABEL];assert len(rows)<=1;
  for row in rows:validate_account_row(row,subprocess.check_output(['scontrol','show','hostnames',row[4]],env=env,text=True,timeout=15).splitlines())
  return rows[0]if rows else None
 def no_foreign_steps(ignore=()):
  rows=[x.split('|')for x in subprocess.check_output(['sacct','-S','2026-10-31','-j','910376','-n','-P','-o','JobID,JobName%120,State,ExitCode,NodeList%100'],env=env,text=True,timeout=25).splitlines()]
  for row in rows:
   if re.fullmatch(r'910376[.]\d+',row[0])and row[0]not in ignore and row[2].split()[0].rstrip('+')not in TERMINAL:
    nodes=subprocess.check_output(['scontrol','show','hostnames',row[4]],env=env,text=True,timeout=15).splitlines();assert not set(nodes)&set(NODES),'Active foreign step intersects target nodes'
  return True
 def idle(suffix):
  args=['srun','--jobid=910376','--overlap','--exact','--immediate=30','--cpu-bind=none','-N4','-n4','--ntasks-per-node=1','-c1','-w',','.join(NODES),'--gres=none','--mem=512M','-t','00:02:00','--job-name='+LABEL+'-'+suffix,'python3',str(D/'resource_probe_r204.py')];z=subprocess.run(args,env=env,text=True,capture_output=True,timeout=160);rows=[json.loads(x)for x in z.stdout.splitlines()if x.startswith('{')];assert z.returncode==0 and len(rows)==4 and {r['node']for r in rows}==set(NODES)and len({r['actual_step']for r in rows})==1 and all(r['CPU_only']and not r['GPU_runtime_initialized']and r['KFD_empty']and r['port43843_bindable']for r in rows);return rows
 save()
 try:
  assert account()is None,'Existing exact label must be followed, never replayed';no_foreign_steps();w['fresh_before']=idle('before');no_foreign_steps({r['actual_step']for r in w['fresh_before']});save();start=time.monotonic();epoch=time.time();w['GPU_start_UTC_unix']=epoch;save()
  export='ALL,WAVE_INSIDE_R2323=GLM-latest-full-owner-r290/GPU_inside_r290.sh,GLM_R290_GPUSTART_UNIX='+str(epoch)+',GLM_SOURCE_MANIFEST_SHA='+OWNER_SHA
  args=['srun','--jobid=910376','--overlap','--exact','--immediate=30','--cpu-bind=none','--kill-on-bad-exit=1','-N4','-n4','--ntasks-per-node=1','-c64','-w',','.join(NODES),'--gres=dcu:8','--mem=128000M','-t','08:10:00','--job-name='+LABEL,'--output='+str(R/'logs'/(LABEL+'-%J-%t.log')),'--export='+export,'env','GLM_SHARED_PREFIX_VALIDATE_R5814=0','GLM_PREFLIGHT_ONLY_R5194=0','GLM_DECODE_ROW_DEDUP_R5386=0','GLM_RUN_R5040='+LABEL,'GLM_BOUNDED_ROPE_R4279=1','GLM_SHARED_CONTROL_R4279=1','GLM_PP_SIZE_R3603=4','GLM_CACHE_ROWS_R3627=1','GLM_CACHE_CAP_R3627=32768','GLM_NETWORK_TRIAL=shca','GLM_PROJECTED_HOST_R4504=0','GLM_PIPELINE_TAIL_NODE=a02r4n15','GLM_EXACT_BANKS_R4188=1','GLM_CHANNELS_R4363=0','GLM_VARIANT_R4363=layer77-5783','INFINILM_OWNER_CPU_WIDTH_R2392=4','USE_WEIGHT_CACHE_R3604=0','GLM_SERVICE_HOST_R143=127.0.0.1','GLM_SERVICE_PORT_R143=43843','GLM_REGULAR_FIXTURE_RECEIPT_R180=/refactor/results/GLM-regular-fixture-r178a-910407.539/CPU-sizing.json','GLM_REGULAR_FIXTURE_RECEIPT_SHA256_R180=ed2de5e3a0035655ae89f38c65359acf567821b5d254279eebc60847d5f89cbf','bash',str(R/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
  with(R/'logs'/(LABEL+'-srun-client.log')).open('x')as log:proc=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  w.update(GPU_submitted=True,srun_client_PID=proc.pid,state='FOLLOW_SAME_SINGLE_OWNED_GPU_HANDLE');save();end=start+30000
  while time.monotonic()<end:
   try:row=account();w.pop('account_observation_error',None)
   except(subprocess.TimeoutExpired,subprocess.CalledProcessError,OSError)as e:w['account_observation_error']=repr(e);save();time.sleep(15);continue
   if row:
    assert actual is None or actual==row[0];actual=row[0];w.update(actual_GPU=actual,last_Slurm=row)
    if row[2].split()[0].rstrip('+')in TERMINAL:w['terminal']=row;save();break
   if proc.poll()is not None:
    w.setdefault('srun_exit_observed_monotonic',time.monotonic());w['srun_exit_code']=proc.poll()
    if time.monotonic()-w['srun_exit_observed_monotonic']>60:raise RuntimeError('srun exited; actual accounting still unresolved; follow exact label, no restart')
   save();time.sleep(15)
  else:
   row=account()
   if row and row[0]==actual and row[2].split()[0].rstrip('+')not in TERMINAL:
    w['owned_runtime_over_authorized_GPU_bound']=True;save();subprocess.run(['scancel','--signal=TERM',actual],env=env,check=True,timeout=25)
   raise TimeoutError('Original GPU runtime bound exceeded; follow same owned handle to terminal, no replay')
  w['post']=idle('post');w['post_empty']=True;owners=[];signatures=[];logs=[]
  for stage in range(4):
   log=R/'logs'/(LABEL+'-'+actual+'-'+str(stage)+'.log');text=log.read_text(errors='replace');normal=text.count('GLM_REGULAR_SERVICE_R204_NORMAL '+str(stage));assert normal==1 and 'Using network IBext_v8'in text and 'Using network Socket'not in text;logs.append({'stage':stage,'SHA256':sha(log),'normal_marker':normal,'IBext_v8':True,'Socket_fallback':False})
   for rank in range(8):
    p=R/'results'/('GLM-latest-full-owner-r290-'+LABEL+'-stage'+str(stage)+'-rank'+str(rank)+'.json');q=json.loads(p.read_bytes());assert q['completed']and q['worker_shutdown_completed']and q['post_close_Torch_liveness']and q['normal_reset_close']and q['native_service_loop_completed']and q['normal_shutdown']and q['library_sha256']=='038d4ff2e7aee2e6db80f691dd885222f3e3160e25ae9d0e269296daf9596c00'and q['latest_protocol_binding_r290']['source_manifest_SHA256']==OWNER_SHA and q['actual_producer']==actual
    rows=q['service_requests'];assert len(rows)==q['full_requests_completed']==120 and all(r['original_reset_all32_completed']for r in rows);assert[r['prefill_control_window']for r in rows]==[2]*102+[1,2]+[1,2,2,1]*4
    pair=rows[102:];assert all(r['prompt_tokens']==8900 and r['requested_output_budget']==4096 and r['generated_natural_EOS']and r['reason']=='eos_token'for r in pair)and len({r['actual_prompt_token_ids_SHA256']for r in pair})==1
    signatures.append([(r['actual_prompt_token_ids_SHA256'],r['generated_token_ids_SHA256'],r['generated_tokens'],r['reason'])for r in rows]);owners.append({'stage':stage,'rank':rank,'path':str(p),'SHA256':sha(p),'normal_exit':True,'Native_requests':120})
    if stage==3 and rank==0:assert q['normal_HTTP_shutdown']and not q['cleanup']['forced_cleanup']and not q['cleanup']['errors']
  assert all(x==signatures[0]for x in signatures),'All32 token/EOS/reset signatures differ'
  p=R/'results'/('GLM-latest-full-owner-r290-supervisor-'+LABEL+'.json');client=json.loads(p.read_bytes());assert client['completed']and client['passed']and client['actual_HTTP']==121 and client['actual_Native']==120 and client['actual_GPU_handle']==actual
  assert w['terminal'][2:4]==['COMPLETED','0:0'];w.update(completed=True,state='FULL_LATEST_NORMAL_MULTICALL_COARSE_PILOTS_AND_PAIR_PASS_PENDING_ROOT_PERF_AUDIT',Model_qualified=True,owner_results=owners,network_and_normal=logs,CPU_supervisor_SHA256=sha(p),actual_HTTP=121,actual_Native=120,all32_token_EOS_reset_signatures_equal=True,client_incremental_metrics_qualified=False,fullbenchmark_or_no_regression_qualified=False);save()
 except BaseException as e:
  w.update(error=repr(e),state='FAILED_BEFORE_GPU'if not w['GPU_submitted']else'FOLLOW_SAME_HANDLE_NO_RETRY');save()
  # An observation failure never means a job died. Slurm/inside absolute bounds
  # remain in force; root follows the recorded handle/unique label to terminal.
  raise
if __name__=='__main__':main()
