"""One CPU-owned cross-allocation run; source pins and ready watch precede GPU Popen."""
import common,os,json,subprocess,pathlib,time,datetime,socket,traceback,hashlib,secrets,re
r=common.ROOT;src=common.SRC;env,removed=common.clean_env();pin=json.loads((src/'model-source-pins.json').read_text());mp=r/'incoming/Kimi-agent-service-Model-r144c/runtime/model-manifest-r9183.json';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads(mp.read_text());context=json.loads((src/'run-context-public.json').read_bytes());mapping=dict(context,nonce=secrets.token_hex(24),passed=True,completed=True,GPU_submitted=False,head_ib0_IPv4=None);run=mapping['run'];hp=r/'results'/('Kimi-sparse-profile-model-r136b-'+run+'.json');assert not hp.exists()
w={'schema':1,'run':run,'CPU_actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'hostname':socket.gethostname(),'PID':os.getpid(),'ready':False,'completed':False,'GPU_submitted':False,'manifest_SHA256':sha(mp),'groups':[],'observation_errors':[],'removed_child_env_names':removed};started=time.monotonic();procs={};streams=[]
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
def save():w['heartbeat_UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();common.atomic(hp,w)
def norm(row):return row[2].split()[0].rstrip('+')
def validate():
 proof=json.loads((src.parent/'fixed-data-r136b.json').read_bytes());assert proof['GSM_unique']==32 and proof['case_count']==12 and proof['no_new_tokenization']
 assert mapping['passed']and mapping['completed']and not mapping['GPU_submitted'];assert sha(mp)==pin['manifest_SHA256'];assert m['nodes']==common.RES['target_nodes'];assert m['topology']=={'tp_nodes':2,'pp':2,'logical_capacity':8};assert m['native_DSO_SHA256']==pin['native_DSO_SHA256'];assert common.RES['logical_capacity']==8 and common.RES['state_pool_rows']==17 and common.RES['KV_pages']==68
 for p,h in pin['files'].items():assert sha(pathlib.Path(p.replace('/refactor',str(r),1)))==h,p
 for p,h in m['source_SHA256'].items():assert sha(pathlib.Path(p.replace('/refactor',str(r),1)))==h,p
 for g in common.RES['target_groups']:common.verify_parent(g,env)
 common.verify_parent({'parent':common.RES['draft']['parent'],'nodes':[common.RES['draft']['node']]},env)

def idle(group,name):
 a=['srun','--jobid='+group['parent'],'--overlap','--exact','--cpu-bind=none','-N'+str(len(group['nodes'])),'-n'+str(len(group['nodes'])),'--ntasks-per-node=1','-c1','--gres=none','--mem=512M','-w',','.join(group['nodes']),'-t','00:02:00','--job-name='+name,'python3',str(r/'incoming/Kimi_Q0_idle_r8066.py')]
 z=subprocess.run(a,env=env,text=True,capture_output=True,timeout=160);rows=[json.loads(s)for s in z.stdout.splitlines()if s.startswith('{')];assert z.returncode==0 and len(rows)==len(group['nodes']) and {x['node']for x in rows}==set(group['nodes']) and all(x['passed']and not x['kfd_pids']for x in rows),(z.returncode,rows,z.stderr);handles={x['step']for x in rows};assert len(handles)==1;handle=handles.pop();actual=common.terminal(group['parent'],name,handle,env);assert actual[2:4]==['COMPLETED','0:0'];return {'rows':rows,'actual_terminal':actual}
def spawn(v,a):
 fo=(r/'logs'/(v['name']+'-client.log')).open('x');streams.append(fo);p=subprocess.Popen(a,env=env,stdin=subprocess.DEVNULL,stdout=fo,stderr=subprocess.STDOUT);procs[v['name']]=p;v.update(command_scope='ownedSlurmstep; no argv/environment persisted',client_PID=p.pid,submitted_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
def observe(v):
 row=common.account(v['parent'],v['name'],env)
 if row:
  assert row[0].startswith(v['parent']+'.')and row[0].split('.')[1].isdigit();assert v.get('actual_handle',row[0])==row[0];v['actual_handle']=row[0];v['actual_row']=row
  if norm(row)in TERMINAL:v['terminal']=row
 p=procs.get(v['name'])
 if p and p.poll()is not None:v['client_returncode']=p.returncode
 return row

def signal_owned(v,reason):
 if v.get('terminal')or v.get('signal'):return
 row=common.account(v['parent'],v['name'],env);assert row and row[0]==v['actual_handle']and row[1]==v['name']and run in v['name'];assert row[0].startswith(v['parent']+'.')and row[0].split('.')[1].isdigit();v['signal']={'reason':reason,'exact_handle':row[0],'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()};save();z=subprocess.run(['scancel','--signal=TERM',row[0]],env=env,text=True,capture_output=True,timeout=25);v['signal'].update(returncode=z.returncode,stderr=z.stderr[-1000:]);save()

def capacity():
 out=[]
 for n in range(4):
  p=r/'results'/('Kimi-agent-service-capacity-r144c-'+run+'-node'+str(n)+'.json')
  if not p.exists():return None
  assert p.stat().st_size<8192;v=json.loads(p.read_text());g=w['groups'][n//4]
  if not g.get('actual_handle'):return None
  assert v['run']==run and v['node']==n and v['host']==m['nodes'][n] and v['actual_step']==g['actual_handle'];assert v['manifest_SHA256']==pin['manifest_SHA256']and v['native_DSO_SHA256']==pin['native_DSO_SHA256'];assert v['capacity_gate_passed']and not v['capacity_only_mode'];assert v['phase']=='loading_weights'and not v['weight_payload_opened_at_publish']and v['inference_forward_calls_at_publish']==0;assert v['all_target_min_free_bytes']>common.RES['minimum_free_bytes'];assert v['layout']['PP_stage']==n//2 and v['layout']['TP']==16 and v['layout']['PP']==2;assert v['PP_ranges']==m['static_memory']['PP_ranges'];assert v['geometry']=={'max_rows':16,'pages':68,'page_size':256,'KDA_pool_rows':17,'pages_per_request':4,'max_total_tokens':1024};assert v['actual_target_owners']==32 and v['actual_shard_head_geometry_validated'];assert v['globalnode']==n;assert v['model_weight_keys_validated']and v['actual_all93_coverage_validated'];assert v['actual_engine_key_stage_range']==m['static_memory']['PP_ranges'][n//2];out.append(v)
 return out

def start_draft():
 d=common.RES['draft'];v={'parent':d['parent'],'nodes':[d['node']],'name':'Kimi-sparse-profile-draft-'+run};w['draft']=v;exp='ALL,KIMI_DRAFT_HOST_R9046='+w['draft_host']+',KIMI_DRAFT_PORT_R9046='+str(mapping['base_port']+9000)+',KIMI_DRAFT_NONCE_R9046='+mapping['nonce']+',WAVE_INSIDE_R2323=Kimi-agent-service-Model-r144c/runtime/draft_inside.sh';a=['srun','--jobid='+v['parent'],'--overlap','--exact','--cpu-bind=none','--kill-on-bad-exit=1','-N1','-n1','-c8','--gres=dcu:1','--mem=32768M','-w',d['node'],'-t','02:00:00','--job-name='+v['name'],'--export='+exp,'--output='+str(r/'logs'/(v['name']+'-%J.log')),'bash',str(r/'incoming/resource_wave_local_r3097_outer.sh'),'probe'];spawn(v,a)

def native_error(v):
 # Only fixed own per-task logs; no glob or foreign job matching.
 if not v.get('actual_handle'):return None
 for local in range(4):
  p=r/'logs'/(v['name']+'-'+v['actual_handle']+'-'+str(local)+'.log')
  if not p.exists():continue
  with p.open('rb')as f:f.seek(0,2);size=f.tell();f.seek(max(0,size-131072));raw=f.read()
  for line in raw.decode(errors='replace').splitlines():
   if 'Traceback (most recent call last)'in line or 'exception during forward'in line or 'I192launchcode='in line:return {'path':str(p),'line':line,'tail_SHA256':hashlib.sha256(raw).hexdigest()}
 return None
save()
try:
 validate();w['preflight_pins_passed']=True
 g=common.RES['target_groups'][0];prelabel='Kimi-agent-service-r144c-imports';a=['srun','--jobid='+g['parent'],'--overlap','--exact','--cpu-bind=none','-N4','-n4','--ntasks-per-node=1','-c64','--gres=none','--mem=8192M','-w',','.join(g['nodes']),'-t','00:05:00','--job-name='+prelabel,'--export=ALL,GLOBAL_OFFSET_R2354=0,GLOBAL_MODEL_MODE_R2354=shca,WAVE_INSIDE_R2323=Kimi-agent-service-Model-r144c/runtime/preflight_inside.sh','bash',str(r/'incoming/resource_wave_local_r3097_outer.sh'),'probe'];z=subprocess.run(a,env=env,text=True,capture_output=True,timeout=340);rows=[json.loads(x)for x in z.stdout.splitlines()if x.startswith('{')];assert z.returncode==0 and len(rows)==4 and all(x['passed']and x['CPU_only']and not x['GPU']and not x['dlopen']and x['event_exports_strong_T']and x['manifest_SHA256']==pin['manifest_SHA256']for x in rows),(z.returncode,rows,z.stderr[-1000:]);handles={x['actual_step']for x in rows};assert len(handles)==1;pt=common.terminal(g['parent'],prelabel,handles.pop(),env);assert pt[2:4]==['COMPLETED','0:0'];w['actual_CPU_closure']={'rows':rows,'terminal':pt};save()
 queue=r/'results/Kimi-sparse-profile-queue-r136b';queue.mkdir(exist_ok=True)
 for path in sorted((r/'incoming/Kimi-agent-service-Model-r144c/queue').glob('q*.json')):
  target=queue/path.name;raw=path.read_bytes()
  if target.exists():assert target.read_bytes()==raw
  else:
   tmp=target.with_suffix('.publish');tmp.write_bytes(raw);os.link(tmp,target);tmp.unlink()
 w['fixed_queue_prepopulated_before_GPU']=len(list(queue.glob('q*.json')));save()
 for i,g in enumerate(common.RES['target_groups']):w.setdefault('fresh_before',[]).append(idle(g,'Kimi-sparse-profile-model-before-r136b-g'+str(i)));save()
 d=common.RES['draft'];w['draft_fresh_before']=idle({'parent':d['parent'],'nodes':[d['node']]},'Kimi-sparse-profile-model-draft-before-r136b');w['draft_host']=w['draft_fresh_before']['rows'][0]['ib0_ipv4'][0];mapping['head_ib0_IPv4']=next(x['ib0_ipv4'][0]for group in w['fresh_before']for x in group['rows']if x['node']==m['nodes'][0]);w['head_ib0_IPv4']=mapping['head_ib0_IPv4'];w['ready']=True;w['watch_ready_before_GPU']=True;save()
 # Both groups submitted before waiting; actual SLURM_NTASKS remains four per allocation.
 for i,g in enumerate(common.RES['target_groups']):
  v={'parent':g['parent'],'nodes':g['nodes'],'offset':g['offset'],'name':'Kimi-sparse-profile-g'+str(i)+'-'+run};w['groups'].append(v)
  shared={'KIMI_USE_RAMCACHE_R3679':'0','KIMI_REDUCE_POLICY':'0','INFINILM_OWNER_CPU_WIDTH_R2392':'4','INFINILM_DENSE_PREFILL_F32_R2475':'1','INFINILM_MXFP4_TILE_TP_R2559':'0','GLOBAL_OFFSET_R2354':str(g['offset']),'GLOBAL_RUN_R2354':run,'GLOBAL_PORT_R2354':str(mapping['base_port']),'KIMI_ENGINE_HEAD':mapping['head_ib0_IPv4'],'KIMI_ENGINE_NONCE':mapping['nonce'],'KIMI_TP_NODES':'2','KIMI_PP_SIZE':'2','GLOBAL_MODEL_MODE_R2354':'shca','KIMI_FEATURE_PORT_R9046':str(mapping['base_port']+8000),'KIMI_DRAFT_PORT_R9046':str(mapping['base_port']+9000),'KIMI_DRAFT_HOST_R9046':w['draft_host'],'KIMI_DRAFT_NONCE_R9046':mapping['nonce'],'KIMI_ENTRY_R9143':'1','KIMI_ENTRY_MANIFEST_R9143':'/refactor/incoming/Kimi-agent-service-Model-r144c/runtime/model-manifest-r9183.json','KIMI_TRUE_DRAFT_BATCH_R9142':'1','KIMI_DRAFT_CARD_COUNT_R9142':'1','KIMI_CAPACITY_ONLY_R9145':'0','WAVE_INSIDE_R2323':'Kimi-agent-service-Model-r144c/runtime/target_inside.sh'}
  a=['srun','--jobid='+g['parent'],'--overlap','--exact','--cpu-bind=none','--kill-on-bad-exit=1','-N4','-n4','--ntasks-per-node=1','-c64','--gres=dcu:8','--mem=128000M','-w',','.join(g['nodes']),'-t','02:00:00','--job-name='+v['name'],'--export=ALL,'+','.join(k+'='+val for k,val in shared.items()),'--output='+str(r/'logs'/(v['name']+'-%J-%t.log')),'bash',str(r/'incoming/resource_wave_local_r3097_outer.sh'),'probe'];spawn(v,a);w['GPU_submitted']=True;save()
 while True:
  try:
   for v in w['groups']:observe(v)
   failed=next((v for v in w['groups']if v.get('terminal')and v['terminal'][2:4]!=['COMPLETED','0:0']),None)
   if not failed:
    failed=next((v for v in w['groups']if native_error(v)),None)
    if failed:failed['native_error']=native_error(failed)
   if failed:
    w['explicit_target_failure']=failed['name']
    for v in w['groups']:
     if v.get('actual_handle'):signal_owned(v,'explicit same-run target failure '+failed['name'])
   if not w.get('draft')and not failed:
    ready=capacity()
    if ready:w['all4_capacity_projections']=ready;w['capacity_passed_before_draft']=True;save();start_draft()
   if w.get('draft'):
    observe(w['draft']);dh=w['draft'].get('actual_handle')
    if dh:
     dp=r/'results'/('Kimi-PP4-cache-draft-r9202-'+dh+'.json')
     if dp.exists():
      dr=json.loads(dp.read_text());w['draft']['status']=dr.get('status');w['draft']['report']=str(dp)
      if dr.get('status')=='ready':w['draft']['ready_seen']=True
    if w['draft'].get('terminal')and w['draft']['terminal'][2:4]!=['COMPLETED','0:0']:
     for v in w['groups']:
      if v.get('actual_handle'):signal_owned(v,'explicit same-run draft failure')
   if all(v.get('terminal')for v in w['groups']):break
  except (OSError,ValueError,subprocess.TimeoutExpired,subprocess.CalledProcessError)as e:w['observation_errors'].append({'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'error':repr(e)});w['observation_errors']=w['observation_errors'][-20:]
  w['resident_queue_policy']='frozen12 C8W8 GSM32 OFFONOFF, originalcleanup150, no newquality fromrepeats';save();time.sleep(10)
 w['all_target_terminal']=True;save()
 if w.get('draft'):
  v=w['draft'];deadline=time.monotonic()+45
  while not v.get('terminal')and time.monotonic()<deadline:observe(v);save();time.sleep(3)
  if not v.get('terminal'):signal_owned(v,'all same-run targets terminal without draft normal close')
  v['terminal']=common.terminal(v['parent'],v['name'],v['actual_handle'],env,seconds=130);save()
 for i,g in enumerate(common.RES['target_groups']):w.setdefault('post_KFD',[]).append(idle(g,'Kimi-sparse-profile-model-post-r136b-g'+str(i)));save()
 if w.get('draft'):w['draft_post_KFD']=idle({'parent':d['parent'],'nodes':[d['node']]},'Kimi-sparse-profile-model-draft-post-r136b')
 w['post_KFD_passed']=True;save();audit_env=dict(env);audit_env.update(SLURM_JOB_ID=os.environ['SLURM_JOB_ID'],SLURM_STEP_ID=os.environ['SLURM_STEP_ID']);z=subprocess.run(['python3',str(src/'model_audit.py'),str(hp)],env=audit_env,text=True,capture_output=True,timeout=200);w['audit']={'returncode':z.returncode,'stdout':z.stdout[-5000:],'stderr':z.stderr[-3000:]};w['completed']=True;w['passed']=z.returncode==0;save()
except BaseException as e:
 w['error']=repr(e);w['traceback']=traceback.format_exc();w['follow_same_handles_no_resubmit']=bool(w['GPU_submitted']);save()
 # A controller contract error must not leave the peer allocation waiting.
 if w['GPU_submitted']:
  owned=w['groups']+([w['draft']]if w.get('draft')else[])
  for v in owned:
   try:
    observe(v)
    if v.get('actual_handle'):signal_owned(v,'same-run controller contract exception')
   except BaseException as ce:v['cleanup_error']=repr(ce)
  save()
  for v in owned:
   if v.get('actual_handle'):
    try:v['terminal']=common.terminal(v['parent'],v['name'],v['actual_handle'],env,seconds=180)
    except BaseException as ce:v['cleanup_terminal_error']=repr(ce)
   save()
  if all(v.get('terminal')for v in w['groups']):
   for i,g in enumerate(common.RES['target_groups']):
    try:w.setdefault('error_post_KFD',[]).append(idle(g,'Kimi-sparse-profile-model-errorpost-r136b-g'+str(i)))
    except BaseException as ce:w.setdefault('error_post_errors',[]).append(repr(ce))
   if w.get('draft')and w['draft'].get('terminal'):
    try:w['error_draft_post_KFD']=idle({'parent':d['parent'],'nodes':[d['node']]},'Kimi-sparse-profile-model-draft-errorpost-r136b')
    except BaseException as ce:w['error_draft_post_error']=repr(ce)
  save()
 raise
finally:
 for f in streams:f.close()
print(json.dumps(w))
if not w.get('passed'):raise SystemExit(1)
