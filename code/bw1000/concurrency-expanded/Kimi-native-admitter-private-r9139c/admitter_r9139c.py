import pathlib,json,sys,time,datetime,subprocess,os,socket,hashlib,resource,gc,fcntl,statistics,threading
from queue_io_r9139 import canonical,publish
from selection_r9139 import select,summarize
base=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014').resolve();cfg=json.loads(pathlib.Path(sys.argv[1]).read_text());assert cfg['target_handle']=='910380.407'and cfg['draft_handle']=='910371.528';run=cfg['run'];inc=pathlib.Path(__file__).resolve().parent;eos=set(json.loads((base/cfg['eos_relative']).read_text())['eos_token_ids']);plan=json.loads((inc/'heldout32-native-plan-r9129.json').read_text());step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];assert step.startswith('910407.');out=base/'results'/('Kimi-native-admitter-r9139c-'+run+'.json');hp=out.with_suffix('.heartbeat.json');lock=(base/'results'/('Kimi-native-admitter-r9139c-'+run+'.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
child_env=dict(os.environ)
for key in list(child_env):
 if((key.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_'))and key not in ['SLURM_CONF','SLURM_CONF_SERVER'])or key in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES']):child_env.pop(key)
UTC=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(path,data):
 if path==hp and threading.current_thread().name!='admitter-heartbeat':return
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2));tmp.replace(path)
assert not out.exists()and not hp.exists();state={'run':run,'target_handle':cfg['target_handle'],'draft_handle':cfg['draft_handle'],'CPU_step':step,'hostname':socket.gethostname(),'PID':os.getpid(),'status':'WAIT_INITIAL11','heartbeat_count':0,'new_GPU_jobs':0,'CUDA_initialized':False,'selection':None,'published':[],'observations':[]};write(out,state);print(json.dumps({k:state[k]for k in ['CPU_step','hostname','PID']}),flush=True);started=time.monotonic();lastSlurm=-999.;elapsed_tick=started;elapsed=0;terminal=False;queueempty=None;cache={};previous_counts={};last_progress=started;oldsignature=None;summaries_cache={};last_read={};queue=base/cfg['queue'];terminal_seen=None
def read_report_random(path):
 with path.open('rb',buffering=0) as f:
  os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_RANDOM)
  chunks=[]
  while True:
   chunk=f.read(65536)
   if not chunk:break
   chunks.append(chunk)
  return b''.join(chunks).decode('utf8')
def heartbeat_loop():
 while True:
  try:
   h={k:state.get(k)for k in ['UTC','CPU_step','hostname','PID','status','heartbeat_count','queueempty_since','max_RSS_KiB','remaining_conservative_s']};h['heartbeat_write_UTC']=UTC();write(hp,h)
  except Exception:pass
  time.sleep(5)
threading.Thread(target=heartbeat_loop,name='admitter-heartbeat',daemon=True).start()
commands_cache={}
selection_source=inc/'selected-actual-r9139b.json';state['selection']=json.loads(selection_source.read_text());assert state['selection']['chosen']['C']==4 and state['selection']['chosen']['W']==4 and state['selection']['chosen']['I192'];state['selection_provenance']={'source':str(selection_source),'SHA256':hashlib.sha256(selection_source.read_bytes()).hexdigest(),'same_run_original11_owneractivation':True}
try:
 probe=queue/('.admitter-preflight-'+os.urandom(6).hex()+'.json');cmd={'preflight':True,'CPU_step':step};assert publish(probe,cmd)and not publish(probe,cmd);probe.unlink();state['atomic_publish_same_queue_FS_passed']=True
 while time.monotonic()-started<5500:
  tick=time.monotonic();state['heartbeat_count']+=1;initial=[];current=[];fresh=True;meta=[];readerrors=[]
  for node,relative in enumerate(cfg['reports']):
   path=base/'results'/('Kimi-admitter-projection-r9139c-node%d.json'%node)
   try:
    doc=json.loads(read_report_random(path));assert (datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(doc['_projection']['UTC'])).total_seconds()<25,'staleprojection';assert doc['node']==node;cs=doc.get('batch_cases',[]);n=len(cs);assert n>=previous_counts.get(node,0),'nonmonotonicreport';previous_counts[node]=n;entry={k:doc.get(k)for k in ['node','phase','active_batch_progress','error','shutdown_error']};entry.update(case_count=n,mtime=path.stat().st_mtime);meta.append(entry)
    mini={'node':node,'error':doc.get('error'),'shutdown_error':doc.get('shutdown_error'),'batch_cases':[]}
    if node==2:
     mini['batch_cases']=cs[:11]
     for case in cs:
      label=case['label']
      if label not in summaries_cache:summaries_cache[label]=case['admitter_summary'];assert summaries_cache[label]['label']==label
      current.append(summaries_cache[label])
     state['command_rejections']=(doc.get('command_rejections')or[])
    else:mini['batch_cases']=[{k:case[k]for k in ['label','state_stats_after64','I192_counts_delta8x2']}for case in cs[:11]]
    initial.append(mini);cache[node]=entry
    # Emitterinitial11 refs only survive until selection thisiteration, not fourrawreports.
    del doc,cs
   except (OSError,ValueError,KeyError,AssertionError,TypeError)as error:fresh=False;readerrors.append({'node':node,'error':repr(error)})
  signature=json.dumps(meta,sort_keys=True)
  if signature!=oldsignature:last_progress=tick;oldsignature=signature
  state.update(UTC=UTC(),report_fresh_all=fresh,read_errors=readerrors,owners=meta or list(cache.values()),last_model_progress_age_s=tick-last_progress,observed_complete_cohorts=len(current),max_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
  if tick-lastSlurm>=20:
   try:
    lines=subprocess.check_output(['sacct','-j',cfg['target_handle']+','+cfg['draft_handle'],'-n','-P','-o','JobID,State,ExitCode,ElapsedRaw,NodeList'],env=child_env,text=True,timeout=15);state['actual_sacct']=lines;row=next(v.split('|')for v in lines.splitlines()if v.split('|')[0]==cfg['target_handle']);elapsed=int(row[3]);elapsed_tick=time.monotonic();terminal=row[1].split()[0].rstrip('+')not in ['RUNNING','PENDING','COMPLETING','CONFIGURING'];lastSlurm=tick
   except (OSError,ValueError,StopIteration,subprocess.SubprocessError)as e:state['sacct_error']=repr(e);fresh=False
  remaining=5000-(elapsed+time.monotonic()-elapsed_tick);state['remaining_conservative_s']=remaining;model_error=any(m.get('error')or m.get('shutdown_error')for m in meta)
  if fresh and len(initial)==4 and all(len(r['batch_cases'])>=11 for r in initial)and not state['selection']and not model_error and not terminal:
   try:state['selection']=select(initial,eos);state['selection_UTC']=UTC();state['status']='SELECTED_ADMITTING';write(inc/'selected-actual-r9139c.json',state['selection'])
   except (AssertionError,ValueError,KeyError,TypeError)as e:
    state['selection_error']=repr(e);state['status']='ORDERED_STOP_NO_VALID_SMOKE_OR_ACTIVATION';publish(queue/'q9999-admitter-stop-r9139.json',{'schema':1,'action':'stop','reason':'initial11 no eligible naturallyEOS_correct_state_or_activation; preserveevidence_nopromotion'})
  del initial;gc.collect()
  done={c['label']for c in current};rejected={x['id']for x in state.get('command_rejections',[])};existing={};stop_present=False;queue_ok=True
  try:
   for path in queue.glob('q*.json'):
    cmd=json.loads(path.read_text());stop_present|=cmd.get('action')=='stop'
    if cmd.get('action')=='case':
     key=cmd['id'];assert key not in existing or existing[key]==cmd;existing[key]=cmd
  except (OSError,ValueError,KeyError,AssertionError)as e:queue_ok=False;state['queue_read_error']=repr(e)
  pending=[c for key,c in existing.items()if key not in done|rejected];state['pending_command_ids']=[c['id']for c in pending]
  if len(current)>=11 and not pending:
   if queueempty is None:queueempty={'UTC':UTC(),'monotonic':tick}
  else:queueempty=None
  state['queueempty_since']=queueempty
  if state['selection']and fresh and queue_ok and not model_error and not terminal:
   chosen=state['selection']['chosen'];C,W,fast=chosen['C'],chosen['W'],chosen['I192'];variant=next(p for p in plan['plans']if p['C']==C and p['state_width']==W);commands=[]
   cachekey=(C,W,fast)
   if cachekey not in commands_cache:
    for item in variant['files']:
     src=inc/pathlib.PurePosixPath(item['path'].replace(chr(92),'/'));raw=src.read_bytes();assert hashlib.sha256(raw).hexdigest()==item['SHA256'];cmd=json.loads(raw);cmd['I192']=bool(fast);commands.append(cmd)
    commands_cache[cachekey]=commands
   commands=commands_cache[cachekey]
   for cmd in commands:
    if cmd['id']in existing:assert existing[cmd['id']]==cmd,'sameIDdifferentparameters'
   completed_pairs=sum(all(cmd['id']in done for cmd in commands[i:i+2])for i in range(0,len(commands),2));state.update(selected_C=C,selected_W=W,selected_I192=fast,heldout_complete_pairs=completed_pairs,heldout_complete_questions=completed_pairs*C,planned_questions=32,heldout_rejected_cohorts=[c['id']for c in commands if c['id']in rejected])
   selected_samples=[c for c in current if c['label'].startswith('heldout32-C%d-'%C)];state['heldout_summary']={'cohorts':len(selected_samples),'naturallyEOS_correct_requests':sum(c['completed_correct_requests']for c in selected_samples),'diagnostic_decode_tokens':sum(c['all_decode_tokens']for c in selected_samples),'good_decode_tokens':sum(c['completed_correct_decode_tokens']for c in selected_samples),'all33cards':33,'native_not_HTTP':True}
   state['heldout_by_mode']={}
   for mode in ['ar','state']:
    arm=[c for c in selected_samples if c['strategy']==mode];users=[u for c in arm for u in c['users']];walls=sum(c['window_s']for c in arm);rates0=[u['native_user_firstlast_tok_s']for u in users if u['native_user_firstlast_tok_s']is not None];state['heldout_by_mode'][mode]={'attempted_requests':len(users),'correct':sum(u['correct']for u in users),'naturalEOS':sum(u['naturalEOS']for u in users),'cap':sum(u['captruncated']for u in users),'parsefail':sum(u['parsefail']for u in users),'naturalEOS_correct_above15':sum(u['native_SLO_above15']for u in users),'peruser_min':min(rates0)if rates0 else None,'peruser_median':statistics.median(rates0)if rates0 else None,'peruser_max':max(rates0)if rates0 else None,'completed_correct_goodput_pooled_per33cards':sum(c['completed_correct_decode_tokens']for c in arm)/walls/33 if walls else None,'pooled_independent_cohorts_not_HTTP':True}
   known=done|rejected|set(existing);nextpair=next((commands[i:i+2]for i in range(0,len(commands),2)if not all(c['id']in known for c in commands[i:i+2])),None)
   scoped={mode:[c for c in current if c['strategy']==mode and c['C']==C and c['W']==(1 if mode=='ar'else W)and(mode=='ar'or c['I192']==fast)]for mode in ['ar','state']}
   rates={m:min(rate for c in cs for u in c['users']for rate in [u['native_user_firstlast_tok_s'],u.get('native_conservative_decode_tok_s')]if(rate or 0)>0)for m,cs in scoped.items()};pre=max(c['prefill_s']for cs in scoped.values()for c in cs)
   # Use slowest observedperuser and wholecycle costs, never addnestedRPC timers.
   cycle={m:max(c['max_cycle_s']for c in cs)for m,cs in scoped.items()}
   def estimate(cmd):
    mode=cmd['strategy'];cap=cmd.get('cap',256);return 1.25*cap/rates[mode]+max(5.,pre)
   state['budget_rate_floor']=rates;state['budget_max_wholecycle_s_diagnostic_not_repeated_coldspike']=cycle;state['budget_decode_safety_factor']=1.25
   missing=[]if nextpair is None else[c for c in nextpair if c['id']not in done|rejected|set(existing)];required=sum(estimate(c)for c in pending)+sum(estimate(c)for c in missing)+150;state['next_admission_required_s']=required
   if stop_present:state['status']='DRAINING_EXISTING_STOP'
   elif nextpair is not None and len(pending)<=2 and remaining>required:
    for cmd in missing:
     index=next(i for i,c in enumerate(commands)if c['id']==cmd['id']);path=queue/('q2000-heldout-C%d-%04d.json'%(C,index));created=publish(path,cmd);state['published'].append({'id':cmd['id'],'path':str(path),'SHA256':hashlib.sha256(canonical(cmd)).hexdigest(),'created':created,'UTC':UTC()})
    state['status']='ADMITTING_PAIRED_HELDOUT'
   elif not pending and(nextpair is None or remaining<=required):
    reason='allplanned32pairedcommandscompleted_or_rejected'if nextpair is None else'budgetinsufficientfornextpairedcohort_preserve150s';publish(queue/'q9999-admitter-stop-r9139.json',{'schema':1,'action':'stop','reason':reason});state['status']='ORDERED_STOP';state['stop_reason']=reason
   elif nextpair is not None and remaining<=required:state['status']='BUDGET_WAIT_DRAIN_CURRENT_PAIRS'
   state['unexecuted_command_ids']=[c['id']for c in commands if c['id']not in done|rejected]
  if model_error:state['status']='MODEL_ERROR_NO_NEW_ADMISSION'
  if terminal:
   if terminal_seen is None:terminal_seen=tick
   state['status']='TARGET_TERMINAL_NO_ADMISSION'
   try:
    watch=json.loads((base/cfg['watch']).read_text());state['final_watch']={k:v for k,v in watch.items()if k not in ['audit_stdout','audit_stderr']}
    if'audit_exit'in watch or tick-terminal_seen>180:state['status']='TERMINAL_OBSERVED';write(out,state);write(hp,{k:state[k]for k in ['UTC','CPU_step','hostname','PID','status','heartbeat_count','queueempty_since','max_RSS_KiB']});break
   except (OSError,ValueError)as e:state['watch_read_error']=repr(e)
  write(out,state);write(hp,{k:state[k]for k in ['UTC','CPU_step','hostname','PID','status','heartbeat_count','queueempty_since','max_RSS_KiB','remaining_conservative_s']});time.sleep(max(.2,5-(time.monotonic()-tick)))
 else:state['status']='BOUNDED_CPU_EXIT_NO_GPU_SIGNAL';write(out,state)
except BaseException as e:state['status']='CPU_ADMITTER_ERROR_NO_GPU_SIGNAL';state['error']=repr(e);state['UTC']=UTC();write(out,state);write(hp,{k:state[k]for k in ['UTC','CPU_step','hostname','PID','status','heartbeat_count']});raise
