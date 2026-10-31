"""One bounded current239/Native247 residency. CPU owned, no automatic retries."""
import argparse,datetime,hashlib,json,os,pathlib,secrets,socket,subprocess,time,traceback,sys
D=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(D/'launcher'))
from resident_supervisor_r279 import ResidentP0Supervisor
R=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');SRC=R/'incoming/Kimi-latest-service-r275'
LABEL='Kimi-latest99-supplement-Model-r279';NODES=['a14r2n08','a14r2n09','a14r2n10','a14r2n12'];RUN='kimi-latest99-supplement-r279-20261031';MODEL_SHA='f72bbc3fcb0971e395121ba297eae08080d67eab16c7eb9aefb101eaa1057671';NATIVE_SHA='ec239cf5d60fa4c00d6f2168beb2289b13151063f1f068a83e3cf64d8b9c99e4'
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
active={'RUNNING','PENDING','CONFIGURING','COMPLETING','SUSPENDED','RESIZING','REQUEUED','SIGNALING','STAGE_OUT'}
norm=lambda s:s.split()[0].rstrip('+')
p=argparse.ArgumentParser();p.add_argument('--execution-go',required=True);a=p.parse_args()
assert os.environ['SLURM_JOB_ID']=='910407'and socket.gethostname()=='a14r2n20'
go=json.loads(pathlib.Path(a.execution_go).read_bytes());assert go['execution_authorized'] and go['one_GPU_attempt'] and go['GPU_max_s']==29400 and go['CPU_keeper_max_s']==32400 and go['maximum_HTTP']==41 and go['no_retries']
assert sha(D/'SOURCE_MANIFEST.json')==go['controller_manifest_SHA256'];mf=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());assert all(sha(D/n)==h for n,h in mf['files_SHA256'].items())
assert sha(SRC/'service-source-manifest-r275.json')=='2e48f0fdd66f018908d0fe76d08dab7d2ae7202ae4a82b63eced7a3f26497678';sm=json.loads((SRC/'service-source-manifest-r275.json').read_bytes());assert all(sha(SRC/n)==h for n,h in sm['files_SHA256'].items());assert sha(SRC/'runtime/model-manifest-r9183.json')==MODEL_SHA
receipt=R/'candidates/Kimi-latest-service-r267/CPU-prepared.json';assert sha(receipt)=='cdc289834cf1dbbbeb8fa616b4d62da9af0aa6195cdef311bc8d5e4a5f9b2846';q=json.loads(receipt.read_bytes());assert q['completed']and q['passed']and q['CPU_only']and not q['GPU']and not q['Native_dlopen']and not q['Model_or_weights_loaded'];assert q['source_manifest_SHA256']=='9118cf2606f141719a55ef7f96c718f6c3cd8bc9b2fdce44f7994408af025dc5'and q['model_manifest_SHA256']=='47dd4d38187f70441f54451b4d897617c228b72c5edd15ea761dfddbc785564e'
base=R/'candidates/Kimi-latest-entry-r266/CPU-build.json';assert sha(base)=='5e75533f6ec5b5b58c5d068ae9a32953e09e4c67ccae01b8a68d36a42d316533';b=json.loads(base.read_bytes());assert b['completed']and not b['GPU']and not b['Native_dlopen']and b['native_SHA256']==NATIVE_SHA
for name,h in b['static_install']['public_files_SHA256'].items():assert sha(R/name[len('/refactor/'):])==h,name
fixture=R/'candidates/Kimi-latest-service-r267/fixed-real-request-r267.json';assert sha(fixture)=='9c488dcb625b72e9dfb3d0b001ec95878b4d0d1366785c1acb84e44f1ad0b2f9'
cp=R/'candidates/Kimi-supplement-client-r278/CPU-prepared.json';assert sha(cp)=='390786b3324d88da2ada5f34c1847e708cb68c3d4a805062f4195e5592a03ab4';cq=json.loads(cp.read_bytes());assert cq['passed']and cq['completed']and cq['production_HTTP']==0 and cq['source_manifest_SHA256']=='1940c5f683b97cb0f246002781bc4787824e42268ad7f80843ef2042369e4b50'
run=RUN;base_port=19743;gpu_label=LABEL;wp=R/'results'/(LABEL+'-watch.json');assert not wp.exists();nonce=secrets.token_hex(24)
env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_','CUDA_','HIP_','ROCR_','GPU_'))or k in ['SLURM_CONF','SLURM_CONF_SERVER']}
w={'run':run,'CPU_actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'GPU_submitted':False,'actual_handle':None,'completed':False,'scope':'LATEST99_SUPPLEMENT41_MULTICALL_GSM32_FINITE_C1_PROFILE_NORMAL34_INHERITED513','publication_commit':go['publication_commit'],'cards':32,'TP':16,'PP':2,'no_draft':True,'GPU_max_seconds':29400,'maximum_HTTP':41,'source_manifest_SHA256':go['controller_manifest_SHA256'],'source_qualifications_not_Model':True};actual=None;proc=None;supervisor=None

def save():
    w['heartbeat_UTC'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    t = wp.with_suffix('.tmp')
    t.write_text(json.dumps(w, indent=2))
    t.replace(wp)

def account(name):
    rows = [s.split('|') for s in subprocess.check_output(['sacct', '-S', '2026-10-27', '-j', '910380', '-n', '-P', '-o', 'JobID,JobName%120,State,ExitCode,NodeList%120'], env=env, text=True, timeout=25).splitlines()]
    matches = [s for s in rows if s[1] == name]
    assert len(matches) <= 1
    return matches[0] if matches else None

def terminal(name, handle):
    deadline = time.monotonic() + 90
    while True:
        row = account(name)
        if row:
            assert row[0] == handle
            if norm(row[2]) in TERMINAL:
                return row
        if time.monotonic() > deadline:
            raise TimeoutError('Follow same accounting handle; no retry')
        time.sleep(2)

def cpu_step(name, inside, nodes, cpus, mem, timeout):
    args = ['srun', '--jobid=910380', '--overlap', '--exact', '--immediate=30', '--cpu-bind=none', '-N' + str(len(nodes)), '-n' + str(len(nodes)), '--ntasks-per-node=1', '-c' + str(cpus), '--gres=none', '--mem=' + str(mem) + 'M', '-w', ','.join(nodes), '-t', '00:05:00', '--job-name=' + name, '--export=ALL,GLOBAL_MODEL_MODE_R2354=shca,GLOBAL_OFFSET_R2354=0,WAVE_INSIDE_R2323=Kimi-supplement-controller-r279/' + inside, 'bash', str(R / 'incoming/resource_wave_local_r3097_outer.sh'), 'probe']
    z = subprocess.run(args, env=env, text=True, capture_output=True, timeout=timeout)
    rows = [json.loads(s) for s in z.stdout.splitlines() if s.startswith('{')]
    assert z.returncode == 0 and len(rows) == len(nodes), ('CPUstep_failed', z.returncode, z.stderr[-2000:])
    handles = {s['actual_step'] for s in rows}
    assert len(handles) == 1
    term = terminal(name, handles.pop())
    assert term[2:4] == ['COMPLETED', '0:0']
    return {'rows': rows, 'terminal': term}

def idle(tag):
    name = LABEL + '-' + tag
    args = ['srun', '--jobid=910380', '--overlap', '--exact', '--immediate=30', '--cpu-bind=none', '-N4', '-n4', '--ntasks-per-node=1', '-c1', '--gres=none', '--mem=512M', '-w', ','.join(NODES), '-t', '00:02:00', '--job-name=' + name, 'python3', str(D / 'launcher/idle_service_r223.py')]
    z = subprocess.run(args, env=env, text=True, capture_output=True, timeout=160)
    rows = [json.loads(s) for s in z.stdout.splitlines() if s.startswith('{')]
    assert z.returncode == 0 and len(rows) == 4 and ({x['node'] for x in rows} == set(NODES)) and all((x['passed'] and (not x['kfd_pids']) and x['service_port_free'] for x in rows))
    handles = {x['step'] for x in rows}
    assert len(handles) == 1
    term = terminal(name, handles.pop())
    assert term[2:4] == ['COMPLETED', '0:0']
    return {'rows': rows, 'terminal': term}

save()
try:
 for parent in ['910380','910407']:
  parent_state=subprocess.check_output(['scontrol','show','job',parent,'-o'],env=env,text=True,timeout=25);assert 'UserId=aclsylqidf('in parent_state and 'JobState=RUNNING'in parent_state
 assert account(gpu_label)is None
 w['CPU_native_closure']=cpu_step(LABEL+'-imports','runtime/preflight_inside_r279.sh',NODES,64,8192,340);assert all(x['passed']and x['CPU_only']and not x['GPU']and not x['dlopen']for x in w['CPU_native_closure']['rows']);save()
 w['fresh_before']=idle('before');head=next(x['ib0_ipv4'][0]for x in w['fresh_before']['rows']if x['node']==NODES[0]);save()
 gpu_started=time.monotonic();gpu_start_unix=time.time();w['GPU_start_UTC_unix']=gpu_start_unix;save()
 shared={'KIMI_R205_GPUSTART_UNIX':str(gpu_start_unix),'KIMI_USE_RAMCACHE_R3679':'0','KIMI_REDUCE_POLICY':'0','INFINILM_OWNER_CPU_WIDTH_R2392':'4','INFINILM_DENSE_PREFILL_F32_R2475':'1','INFINILM_MXFP4_TILE_TP_R2559':'0','GLOBAL_OFFSET_R2354':'0','GLOBAL_RUN_R2354':run,'GLOBAL_PORT_R2354':str(base_port),'KIMI_ENGINE_HEAD':head,'KIMI_ENGINE_NONCE':nonce,'KIMI_TP_NODES':'2','KIMI_PP_SIZE':'2','GLOBAL_MODEL_MODE_R2354':'shca','KIMI_DRAFT_CARD_COUNT_R9142':'0','KIMI_ENTRY_R9143':'1','KIMI_ENTRY_MANIFEST_R9143':'/refactor/incoming/Kimi-latest-service-r275/runtime/model-manifest-r9183.json','KIMI_CAPACITY_ONLY_R9145':'0','KIMI_AGENT_HTTP_PORT_R144D':'43844','WAVE_INSIDE_R2323':'Kimi-latest-service-r275/runtime/target_inside.sh'}
 args=['srun','--jobid=910380','--overlap','--exact','--immediate=30','--cpu-bind=none','--kill-on-bad-exit=1','-N4','-n4','--ntasks-per-node=1','-c64','--gres=dcu:8','--mem=128000M','-w',','.join(NODES),'-t','08:10:00','--job-name='+gpu_label,'--export=ALL,'+','.join(k+'='+v for k,v in shared.items()),'--output='+str(R/'logs'/(gpu_label+'-%J-%t.log')),'bash',str(R/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
 with(R/'logs'/(LABEL+'-GPU-client.log')).open('x')as f:proc=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 supervisor=ResidentP0Supervisor(root=R,source=D,run=run,version='r275',env=env,account=account,gpu_started=gpu_started,gpu_started_unix=gpu_start_unix)
 w.update(GPU_submitted=True,client_PID=proc.pid,no_argv_nonce_persisted=True);save();deadline=gpu_started+30000
 while time.monotonic()<deadline:
  row=account(gpu_label)
  if row:
   assert actual is None or actual==row[0];actual=row[0];w.update(actual_handle=actual,Slurm=row)
   if norm(row[2])in TERMINAL:w['terminal']=row;save();break
  if proc.poll() is not None:
   w.setdefault('GPU_srun_exit_observed_clock',time.monotonic());w['GPU_srun_exitcode']=proc.poll()
   if time.monotonic()>=w['GPU_srun_exit_observed_clock']+60:
    raise TimeoutError('Owned GPU srun exited without accounting terminal after60s; follow recorded handle, no retry')
  supervisor.tick(actual);w['automatic_client']=supervisor.snapshot();ready=R/'results'/('Kimi-latest-service-ready-r275-'+run+'.json')
  if ready.exists():
   q=json.loads(ready.read_bytes());assert q['ready']and q['service_node']==NODES[0];w.update(service_ready=q,service_ready_SHA256=sha(ready))
  save();time.sleep(10)
 else:raise TimeoutError('Observation bound: follow the same actual handle; no resubmission')
 w['inherited_normal_GPU']='910380.513';w['normal34_repeated']=False;w['same_run75_claim']=False
 w['post_KFD']=idle('post');w['post_empty']=True;w['nodes']=[];signatures=[]
 for i in range(4):
  p=R/'results'/('Kimi-latest-service-r275-'+run+'-node%d.json'%i);q=json.loads(p.read_bytes());normal=bool(q['completed']and q['all8_owned_workers_shutdown_completed']and q['protocol_before_destroy_passed']and q['agent_HTTP_normal_shutdown']);w['nodes'].append({'node':i,'path':str(p),'SHA256':sha(p),'normal8':normal,'requests':len(q.get('agent_requests',[]))});assert normal
  assert q['manifest_SHA256']==MODEL_SHA and q['library_sha256']==NATIVE_SHA
  text=(R/'logs'/(gpu_label+'-'+actual+'-'+str(i)+'.log')).read_text(errors='replace');assert 'Using network IBext_v8'in text and 'Using network Socket'not in text and text.count('KIMI_MODERN_BATCH_R9143_NORMAL')==1
  signature=[]
  for index,item in enumerate(q['agent_requests']):
   assert item['admitted_index']==index and item['prefill_mode']==(['W512']*41)[index]
   assert item['normal_reset']and item['input_tokens']+item['output_budget']<=32768
   signature.append([item['request_id'],item['client_session'],item['input_tokens'],item['output_budget'],item['generated_ids'],item['finish'],item['history_render_audit']['Native_prompt_token_ids_SHA256']])
  signatures.append(hashlib.sha256(json.dumps(signature,separators=(',',':')).encode()).hexdigest())
 assert len(set(signatures))==1;w['all4_actual_request_prompt_output_signatures_equal']=True;w['prompt_output_signature_SHA256']=signatures[0]
 head_report=json.loads((R/'results'/('Kimi-latest-service-r275-'+run+'-node0.json')).read_bytes());rows=head_report['agent_requests'];w['actual_admissions']=len(rows)
 w['Native_backend_serial_proven']=all(a['finished_monotonic']<=b['started_monotonic']for a,b in zip(rows,rows[1:]));assert w['Native_backend_serial_proven']
 w['completed']=True;w['normal_service_lifecycle']=w['terminal'][2:4]==['COMPLETED','0:0'];w['automatic_client']=supervisor.snapshot();w['client_passed']=supervisor.state['client_P0_passed'];w['Model_qualified']=bool(w['normal_service_lifecycle']and w['client_passed']and len(rows)==41);w['state']='PASS_PENDING_ROOT_INDEPENDENT_AUDIT'if w['Model_qualified']else'SUPPLEMENT_FAILED_PRIOR_NORMAL_QUALIFICATION_RETAINED';save()
except TimeoutError as e:w.update(error=repr(e),state='unknown_follow_same_handle_no_resubmit');save();raise
except BaseException as e:
 w.update(error=str(e).replace(nonce,'[REDACTED_NONCE]'),state='failed_before_GPU'if proc is None else'owned_GPU_contract_failure');save()
 if actual:
  row=account(gpu_label)
  if row and row[0]==actual and norm(row[2])in active:subprocess.run(['scancel','--signal=TERM',actual],env=env,check=True,timeout=25);w['terminal']=terminal(gpu_label,actual);save()
  if w.get('terminal'):
   try:w['error_post']=idle('errorpost');w['post_empty']=True
   except BaseException as ce:w['cleanup_error']=repr(ce)
   save()
 raise
finally:
 if supervisor is not None:
  try:supervisor.close_owned_client(terminal)
  finally:w['automatic_client']=supervisor.snapshot();save()
