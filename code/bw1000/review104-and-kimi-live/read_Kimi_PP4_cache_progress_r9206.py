import pathlib,json,datetime,hashlib,subprocess,os
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';run='shard-cache-r9206-1eeada521ff4f579';out={'observedUTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'run':run};p=r/'results'/('Kimi-shard-PP4-cache-model-r9206-'+run+'.json');w=json.loads(p.read_bytes());out['health']={k:w[k]for k in ['CPU_actual_step','hostname','PID','heartbeat_UTC','ready','capacity_passed_before_draft','completed','passed','error','observation_errors','audit','post_KFD_passed']if k in w};out['groups']=[{k:v for k,v in g.items()if k!='args'}for g in w.get('groups',[])];out['draft']={k:v for k,v in w.get('draft',{}).items()if k!='args'};out['nodes']=[]
for n in range(8):
 p=r/'results'/('Kimi-shard-PP4-cache-r9206-'+run+'-node%d.json'%n);v={'node':n,'exists':p.exists()}
 if p.exists():
  raw=p.read_bytes();q=json.loads(raw);v.update(rawSHA256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),mtime=p.stat().st_mtime)
  for key in ['phase','loaded_keys','loading_elapsed_seconds','loading_seconds','completed','error','traceback','all8_owned_workers_shutdown_completed','serving_protocol_complete','head_after_finish','host_loader_phase_profile','memory_after_finish']:
   if key in q:v[key]=q[key]
  for key in ['after_finish_memory_gate','last_memory_guard']:
   if key in q:v[key]={k:q[key][k]for k in ['label','minimum_free_bytes','required_free_bytes']}
  v['completed_case_count']=len(q.get('batch_cases',[]))
  if n==6:
   v['cases']=[]
   for c in q.get('batch_cases',[]):
    z={k:c.get(k)for k in ['label','strategy','draft_projected_cache','configured_C','configured_width','decode_common_window_s','aggregate_decode_tok_s','aggregate_decode_tok_s_per_card','total_useful_decode_tokens','prefill_all_complete_s','all_serving_GPU_cards','host_observer_decode_window']};z['users']=[{k:u.get(k)for k in ['logical','useful_tokens','useful_decode_tokens','user_firstlast_tok_s','decode_user_tok_s','ITL_P95_s','ITL_P99_s','EOS','truncated']}|{'sample_index':u.get('quality',{}).get('sample_index'),'quality_correct':u.get('quality',{}).get('correct'),'answer':u.get('quality',{}).get('extracted')}for u in c.get('request_metrics',[])];v['cases'].append(z)
  pp=p.with_suffix('.progress.json')
  if pp.exists():v['progress']=json.loads(pp.read_bytes())
 out['nodes'].append(v)
env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM','PMI','CUDA','HIP','ROCR'))};z=subprocess.run(['sacct','-S','2026-10-25','-j','910407,910375,910381,910371','-n','-P','-o','JobIDRaw,JobName%95,State,ExitCode,Elapsed'],env=env,text=True,capture_output=True,timeout=30);out['sacct']=[s for s in z.stdout.splitlines()if run in s or'Kimi-PP4-cache-Model-keeper-r9206'in s];print(json.dumps(out,ensure_ascii=False))
