import pathlib,json,hashlib,datetime
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';p=r/'Kimi-fixed-cohort-service-r8973-launch.json';d=json.loads(p.read_bytes());run=d['run'];out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'run':run,'launch':{k:d[k]for k in ['state','watcher_pid','Model_srun_pid','Model_srun_returncode','error']if k in d},'launch_SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'owners':[]}
p=r/'Kimi-fixed-cohort-service-r8973-watch.json'
if p.exists():out['watch']=json.loads(p.read_bytes());out['watch_SHA256']=hashlib.sha256(p.read_bytes()).hexdigest()
for node in range(4):
 p=r/'results'/f'kimi-fixed-cohort-service-r8973-{run}-node{node}.json'
 if not p.exists():continue
 q=json.loads(p.read_bytes());o={'node':node,'SHA256':hashlib.sha256(p.read_bytes()).hexdigest()};o.update({k:q[k]for k in ['phase','completed','error','shutdown_error','loading_seconds','loading_elapsed_seconds','loaded_keys','parameter_keys','all8_owned_workers_shutdown_completed','preload_memory_gate','geometry_memory_gate']if k in q});o['QA_count']=len(q.get('QA',[]));o['cases']=[{k:z[k]for k in ['label','B','prefill_wall_seconds','decode_wall_seconds','aggregate_decode_tok_s','end_to_end_output_tok_s','service_wall_seconds','matched_baseline_complete_IDs']if k in z}for z in q.get('serving_cases',[])if z['label'].startswith('B64')];c=q.get('fixed_cohort_interleave',{});o['candidate']={k:c[k]for k in ['status','service_wall_seconds','end_to_end_output_tok_s','complete_frozen_IDs_EOS_match','first_differences','job_count']if k in c};out['owners'].append(o)
p=r/'results'/f'Kimi-fixed-cohort-service-r8973-audit-{run}.json'
if p.exists():out['audit']=json.loads(p.read_bytes());out['audit_SHA256']=hashlib.sha256(p.read_bytes()).hexdigest()
print(json.dumps(out))
