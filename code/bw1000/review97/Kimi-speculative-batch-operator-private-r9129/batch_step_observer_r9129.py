"""CPU-only Slurm-tracked batch resident observer. Never launch/cancel a GPU job."""
import pathlib,json,sys,time,datetime,subprocess,os,socket,hashlib,gc
base=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014').resolve();cfg=json.loads(pathlib.Path(sys.argv[1]).read_text());assert cfg['schema']==1 and cfg['target_handle'].startswith('910380.');run=cfg['run'];assert all(c.isalnum()or c in '-_'for c in run);out=base/'results'/('Kimi-batch-observer-r9129-'+run+'.json');heartbeat=out.with_suffix('.heartbeat.json');job=os.environ['SLURM_JOB_ID'];step=job+'.'+os.environ['SLURM_STEP_ID'];assert job=='910406';record={'run':run,'target_handle':cfg['target_handle'],'draft_handle':cfg['draft_handle'],'observer_step':step,'observer_host':socket.gethostname(),'observer_pid':os.getpid(),'CUDA_initialized':False,'status':'observing','heartbeat_count':0,'last_known_cases':[]};lastknown={};last_progress=time.monotonic();signature=None;queueempty=None;terminal_seen=None;started=time.monotonic();last_slurm=-999.;terminal=False
stamp=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(path,data):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2));tmp.replace(path)
def safe_path(relative):
 p=(base/relative).resolve();assert p.is_relative_to(base);return p
assert not out.exists()and not heartbeat.exists();write(out,record);print(json.dumps({'observer_step':step,'hostname':socket.gethostname(),'pid':os.getpid(),'target_handle':cfg['target_handle']}),flush=True)
try:
 while time.monotonic()-started<5500:
  tick=time.monotonic();probes={};owners=[];current_cases=None
  for node,relative in enumerate(cfg['reports']):
   p=safe_path(relative);probe={'fresh':False}
   try:
    stat=p.stat();data=json.loads(p.read_text());assert data['node']==node;minimal={k:data.get(k)for k in ['node','phase','active_batch_progress','completed','error','shutdown_error','loaded_keys','parameter_keys','loading_seconds','serving_protocol_complete','all8_owned_workers_shutdown_completed']};minimal['case_count']=len(data.get('batch_cases',[]));minimal['mtimeUTC']=datetime.datetime.fromtimestamp(stat.st_mtime,datetime.timezone.utc).isoformat();minimal['file_bytes']=stat.st_size
    old=lastknown.get(node)
    if old and minimal['case_count']<old['case_count']:probe['reason']='nonmonotonic read; lastknown retained'
    else:lastknown[node]=minimal;probe['fresh']=True
    if node==cfg.get('emitter',2)and probe['fresh']:
     current_cases=[]
     for case in data.get('batch_cases',[]):
      users=[{k:u.get(k)for k in ['logical','useful_decode_tokens','user_firstlast_tok_s','decode_user_tok_s','EOS','truncated','native_TTFT_s','ITL_P95_s','ITL_P99_s','quality']}for u in case['request_metrics']]
      current_cases.append({k:case.get(k)for k in ['label','strategy','configured_C','configured_width','I192_enabled','aggregate_decode_tok_s','aggregate_decode_tok_s_per_all33cards','decode_common_window_s','total_useful_decode_tokens','I192_counts_delta8x2']}|{'users':users})
    del data
   except (OSError,ValueError,KeyError,AssertionError)as e:probe['reason']=repr(e)
   probes[str(node)]=probe
   if node in lastknown:owners.append(lastknown[node])
  if current_cases is not None:
   if len(current_cases)>=len(record['last_known_cases']):record['last_known_cases']=current_cases
  now_signature=json.dumps([owners,[(v['label'],v['aggregate_decode_tok_s'])for v in record['last_known_cases']]],sort_keys=True)
  if now_signature!=signature:last_progress=tick;signature=now_signature
  completed={v['label']for v in record['last_known_cases']};queue=safe_path(cfg['queue']);pending=[];queue_errors=[]
  for path in queue.glob('q*.json'):
   try:
    c=json.loads(path.read_text())
    if c.get('action')=='case'and c.get('id')not in completed:pending.append(c.get('id'))
   except (OSError,ValueError)as e:queue_errors.append({'file':path.name,'reason':repr(e)})
  initial_finished=len(completed)>=cfg.get('initial_cohorts',11)
  if initial_finished and not pending:
   if queueempty is None:queueempty={'monotonic':tick,'UTC':stamp()}
  else:queueempty=None
  record.update(UTC=stamp(),heartbeat_count=record['heartbeat_count']+1,probe_freshness=probes,owners=owners,pending_command_ids=pending,queue_read_errors=queue_errors,queueempty_since=queueempty,no_progress_s=tick-last_progress,possible_stall=bool((pending or any(str(v.get('phase','')).startswith('batch:')for v in owners))and tick-last_progress>120),cards=cfg['cards'])
  if tick-last_slurm>30:
   last_slurm=tick
   try:record['actual_sacct']=subprocess.check_output(['sacct','-j',cfg['target_handle']+','+cfg['draft_handle'],'-n','-P','-o','JobID,State,ExitCode,ElapsedRaw,NodeList'],text=True,timeout=20)
   except (OSError,subprocess.SubprocessError)as e:record['sacct_read_error']=repr(e)
  try:
   w=json.loads(safe_path(cfg['watch']).read_text());assert w['actual_handle']==cfg['target_handle'];record['watch']={k:v for k,v in w.items()if k not in ['audit_stdout','audit_stderr']};terminal=bool(w.get('terminal'))
   if terminal and terminal_seen is None:terminal_seen=tick
  except (OSError,ValueError,KeyError,AssertionError)as e:record['watch_read_error']=repr(e)
  write(out,record);write(heartbeat,{k:record[k]for k in ['UTC','run','target_handle','observer_step','observer_host','observer_pid','status','heartbeat_count','queueempty_since','no_progress_s']});gc.collect()
  if terminal and('audit_exit'in record.get('watch',{})or tick-terminal_seen>180):record['status']='terminal_observed';write(out,record);break
  time.sleep(max(.1,5-(time.monotonic()-tick)))
 else:record['status']='bounded_observer_timeout_no_GPU_signal';write(out,record)
except BaseException as e:record['status']='CPU_observer_error_no_GPU_signal';record['error']=repr(e);write(out,record);raise
