"""One CPU-owned 32-card resident service, never a throughput optimization run."""
import argparse,datetime,hashlib,json,os,pathlib,secrets,socket,subprocess,time,traceback
from resident_p0_supervisor_r146 import ResidentP0Supervisor
R=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');D=R/'incoming/Kimi-agent-service-Model-r189';LABEL='Kimi-agent-service-r189';NODES=['a14r2n08','a14r2n09','a14r2n10','a14r2n12'];sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--manifest-sha',required=True);p.add_argument('--go-file',required=True);a=p.parse_args();assert os.environ['SLURM_JOB_ID']=='910407'and socket.gethostname()=='a14r2n20'
assert sha(D/'service-source-manifest-r189.json')==a.manifest_sha;m=json.loads((D/'service-source-manifest-r189.json').read_bytes());go=json.loads(pathlib.Path(a.go_file).read_bytes());assert go['actual_device_step']=='910380.462'and go['fresh_device_post_step']=='910380.465'and go['device_and_post_reviewed']is True and go['device_compact_receipt_SHA256']==m['root_device_evidence_required']['root_compact_receipt_SHA256'];assert go['GPU_authorized']is True and go['source_published']is True and go['source_manifest_SHA256']==a.manifest_sha and len(go['publication_commit'])==40 and go['publication_branch']=='bw1000-experiments'
for n,h in m['source_SHA256'].items():assert sha(D/n)==h,n
client_spec=json.loads((D/"client/client-launch-r189.json").read_bytes());assert client_spec["status"]=="FROZEN_CPU_REVIEWED" and client_spec["GPU_GO"] is False
run=json.loads((D/'launcher/run-context-public.json').read_bytes())['run'];base_port=m['base_port'];gpu_label=LABEL+'-'+run;wp=R/'results'/(LABEL+'-watch.json');assert not wp.exists();nonce=secrets.token_hex(24);env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_'))or k in ['SLURM_CONF','SLURM_CONF_SERVER']}
for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES']:env.pop(k,None)
w={'run':run,'CPU_actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'GPU_submitted':False,'actual_handle':None,'completed':False,'P0_passed':None,'scope':'FORMAL32K_WITH4K8K_OUTPUT_NOT_SMOKE','publication_commit':go['publication_commit'],'source_manifest_SHA256':a.manifest_sha,'cards':32,'no_draft':True};actual=None;proc=None;supervisor=None
active={'RUNNING','PENDING','CONFIGURING','COMPLETING','SUSPENDED'};norm=lambda s:s.split()[0].rstrip('+')
def save():
 w['heartbeat_UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();t=wp.with_suffix('.tmp');t.write_text(json.dumps(w,indent=2));t.replace(wp)
def account(name):
 rows=[s.split('|')for s in subprocess.check_output(['sacct','-S','2026-10-27','-j','910380','-n','-P','-o','JobID,JobName%120,State,ExitCode,NodeList%120'],env=env,text=True,timeout=25).splitlines()];matches=[s for s in rows if s[1]==name];assert len(matches)<=1;return matches[0]if matches else None
def terminal(name,handle):
 deadline=time.monotonic()+90
 while True:
  row=account(name)
  if row:
   assert row[0]==handle
   if norm(row[2])not in active:return row
  if time.monotonic()>deadline:raise TimeoutError('Follow same accounting handle; no retry')
  time.sleep(2)
def cpu_step(name,inside,nodes,cpus,mem,timeout):
 args=['srun','--jobid=910380','--overlap','--exact','--immediate=30','--cpu-bind=none','-N'+str(len(nodes)),'-n'+str(len(nodes)),'--ntasks-per-node=1','-c'+str(cpus),'--gres=none','--mem='+str(mem)+'M','-w',','.join(nodes),'-t','00:05:00','--job-name='+name,'--export=ALL,GLOBAL_MODEL_MODE_R2354=shca,GLOBAL_OFFSET_R2354=0,WAVE_INSIDE_R2323=Kimi-agent-service-Model-r189/'+inside,'bash',str(R/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
 z=subprocess.run(args,env=env,text=True,capture_output=True,timeout=timeout);rows=[json.loads(s)for s in z.stdout.splitlines()if s.startswith('{')];assert z.returncode==0 and len(rows)==len(nodes),('CPUstep_failed',z.returncode,z.stderr[-2000:]);handles={s['actual_step']for s in rows};assert len(handles)==1;term=terminal(name,handles.pop());assert term[2:4]==['COMPLETED','0:0'];return {'rows':rows,'terminal':term}
def idle(tag):
 name=LABEL+'-'+tag;args=['srun','--jobid=910380','--overlap','--exact','--immediate=30','--cpu-bind=none','-N4','-n4','--ntasks-per-node=1','-c1','--gres=none','--mem=512M','-w',','.join(NODES),'-t','00:02:00','--job-name='+name,'python3',str(D/'launcher/idle_service_r189.py')];z=subprocess.run(args,env=env,text=True,capture_output=True,timeout=160);rows=[json.loads(s)for s in z.stdout.splitlines()if s.startswith('{')];assert z.returncode==0 and len(rows)==4 and {x['node']for x in rows}==set(NODES)and all(x['passed']and not x['kfd_pids']and x['service_port_free']for x in rows);handles={x['step']for x in rows};assert len(handles)==1;term=terminal(name,handles.pop());assert term[2:4]==['COMPLETED','0:0'];return {'rows':rows,'terminal':term}
save()
try:
 parent=subprocess.check_output(['scontrol','show','job','910380','-o'],env=env,text=True,timeout=25);assert 'UserId=aclsylqidf('in parent and 'JobState=RUNNING'in parent;assert account(gpu_label)is None
 w['CPU_overlay']=cpu_step(LABEL+'-prepare','runtime/CPU_prepare_inside_r189.sh',[NODES[0]],1,8192,340);assert w['CPU_overlay']['rows'][0]['completed']and not w['CPU_overlay']['rows'][0]['GPU'];save()
 w['CPU_native_closure']=cpu_step(LABEL+'-imports','runtime/preflight_inside.sh',NODES,64,8192,340);assert all(x['passed']and x['CPU_only']and not x['GPU']and not x['dlopen']for x in w['CPU_native_closure']['rows']);save()
 w['fresh_before']=idle('before');head=next(x['ib0_ipv4'][0]for x in w['fresh_before']['rows']if x['node']==NODES[0]);w['watch_ready_before_GPU']=True;save()
 shared={'KIMI_USE_RAMCACHE_R3679':'0','KIMI_REDUCE_POLICY':'0','INFINILM_OWNER_CPU_WIDTH_R2392':'4','INFINILM_DENSE_PREFILL_F32_R2475':'1','INFINILM_MXFP4_TILE_TP_R2559':'0','GLOBAL_OFFSET_R2354':'0','GLOBAL_RUN_R2354':run,'GLOBAL_PORT_R2354':str(base_port),'KIMI_ENGINE_HEAD':head,'KIMI_ENGINE_NONCE':nonce,'KIMI_TP_NODES':'2','KIMI_PP_SIZE':'2','GLOBAL_MODEL_MODE_R2354':'shca','KIMI_DRAFT_CARD_COUNT_R9142':'0','KIMI_ENTRY_R9143':'1','KIMI_ENTRY_MANIFEST_R9143':'/refactor/incoming/Kimi-agent-service-Model-r189/runtime/model-manifest-r9183.json','KIMI_CAPACITY_ONLY_R9145':'0','KIMI_AGENT_HTTP_PORT_R144D':'43844','WAVE_INSIDE_R2323':'Kimi-agent-service-Model-r189/runtime/target_inside.sh'}
 args=['srun','--jobid=910380','--overlap','--exact','--immediate=30','--cpu-bind=none','--kill-on-bad-exit=1','-N4','-n4','--ntasks-per-node=1','-c64','--gres=dcu:8','--mem=128000M','-w',','.join(NODES),'-t','02:30:00','--job-name='+gpu_label,'--export=ALL,'+','.join(k+'='+v for k,v in shared.items()),'--output='+str(R/'logs'/(gpu_label+'-%J-%t.log')),'bash',str(R/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
 with(R/'logs'/(LABEL+'-GPU-client.log')).open('x')as f:proc=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 gpu_started=time.monotonic();supervisor=ResidentP0Supervisor(root=R,source=D,run=run,version='r189',env=env,account=account,gpu_started=gpu_started)
 w.update(GPU_submitted=True,client_PID=proc.pid,command_scope='owned single Slurmstep; no argv/private nonce persisted');save();deadline=gpu_started+8800
 while time.monotonic()<deadline:
  row=account(gpu_label)
  if row:
   assert actual is None or actual==row[0];actual=row[0];w.update(actual_handle=actual,Slurm=row)
   if norm(row[2])not in active:w['terminal']=row;save();break
  supervisor.tick(actual);w['automatic_P0_client']=supervisor.snapshot()
  ready=R/'results'/('Kimi-agent-service-ready-r189-'+run+'.json')
  if ready.exists():
   q=json.loads(ready.read_bytes());assert q['ready']and q['service_node']==NODES[0];w.update(service_ready=q,service_ready_SHA256=sha(ready))
  save();time.sleep(10)
 else:raise TimeoutError('Observation bound; follow exact GPU handle, no resubmission')
 w['post_KFD']=idle('post');w['post_empty']=True;w['nodes']=[]
 for i in range(4):
  p=R/'results'/('Kimi-agent-service-r189-'+run+'-node%d.json'%i);q=json.loads(p.read_bytes());normal=bool(q['completed']and q['all8_owned_workers_shutdown_completed']and q['protocol_before_destroy_passed']and q['agent_HTTP_normal_shutdown']);w['nodes'].append({'node':i,'path':str(p),'SHA256':sha(p),'normal8':normal,'served_requests':len(q.get('agent_requests',[]))});assert normal
  text=(R/'logs'/(gpu_label+'-'+actual+'-'+str(i)+'.log')).read_text(errors='replace');assert 'Using network IBext_v8'in text and 'Using network Socket'not in text and text.count('KIMI_MODERN_BATCH_R9143_NORMAL')==1
 # The final q is node3; read only node0 request fields into the public watch.
 head_report=json.loads((R/'results'/('Kimi-agent-service-r189-'+run+'-node0.json')).read_bytes())
 fields=['request_id','client_session','accepted_monotonic','active_requests_at_accept','started_monotonic','finished_monotonic','normal_reset','input_tokens','output_budget','finish']
 admissions=[{k:item.get(k) for k in fields}for item in head_report['agent_requests']]
 w['server_admissions']=admissions
 positive=[item for item in admissions if item['client_session']in ['P1-HTTP','P1-SSE']]
 w['actual_server_C2_admission_overlap']=any(a['client_session']=='P1-HTTP'and b['client_session']=='P1-SSE'and max(a['accepted_monotonic'],b['accepted_monotonic'])<min(a['finished_monotonic'],b['finished_monotonic'])for a in positive for b in positive)
 ordered=sorted(admissions,key=lambda item:item['started_monotonic'])
 w['Native_backend_serial_proven']=all(a['finished_monotonic']<=b['started_monotonic']for a,b in zip(ordered,ordered[1:]))
 w['all_actual_request_caps_preserved']=all(item['normal_reset']and item['input_tokens']<32768 and item['output_budget']<=8192 and item['input_tokens']+item['output_budget']<=32768 for item in admissions)
 assert w['actual_server_C2_admission_overlap']and w['Native_backend_serial_proven']and w['all_actual_request_caps_preserved']
 w['completed']=True;w['normal_service_lifecycle']=w['terminal'][2:4]==['COMPLETED','0:0'];w['automatic_P0_client']=supervisor.snapshot();w['P0_passed']=supervisor.state['client_P0_passed'];w['regular_and_long_output_passed']=supervisor.state['client_P0_passed'];w['P1_passed']=supervisor.state['client_P0_passed'];w['scope']='Native service lifecycle plus unique frozen CPU client; root audits exact backend identity and witnesses';save();assert w['normal_service_lifecycle']and w['P1_passed']
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
  finally:w["automatic_P0_client"]=supervisor.snapshot();save()
