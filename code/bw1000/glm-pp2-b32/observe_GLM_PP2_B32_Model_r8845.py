import pathlib,json,subprocess,datetime
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';launch=json.loads((r/'glm-transient-Model-r8845-launch.json').read_text());run=launch['run'];out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'CPU_read_only':True,'run':run,'PID':launch['pid'],'watch_PID':launch['watch_pid']}
p=r/'results/GLM-transient-Model-watch-r8845.json';out['watch']=json.loads(p.read_text())if p.exists()else None
s=subprocess.check_output(['sacct','-S','2026-10-24','-j','910376','-n','-P','-o','JobID,JobName%80,State,ExitCode,Elapsed,NodeList%100'],text=True,timeout=30)
out['actual']=[v.split('|')for v in s.splitlines()if 'glm-transient-Model-r8845|'in v]
out['phases']={}
for stage in(0,1):
 for rank in range(8):
  p=r/'results'/f'GLM-Model-phase-r8845-{run}-stage{stage}-rank{rank}.jsonl';out['phases'][f'{stage}:{rank}']=p.read_text().splitlines()[-3:]if p.exists()else[]
out['logtails']={}
for row in out['actual']:
 for node in(0,1):
  p=r/'logs'/f'glm-transient-Model-r8845-{row[0]}-{node}.log'
  if p.exists():
   with p.open('rb')as f:f.seek(max(0,p.stat().st_size-5000));out['logtails'][str(node)]=f.read().decode(errors='replace')
out['owner_summaries']={}
for stage in(0,1):
 p=r/'results'/f'glm-transient-Model-r8845-{run}-stage{stage}-rank0.json'
 if p.exists():
  q=json.loads(p.read_text());out['owner_summaries'][str(stage)]={k:q.get(k)for k in['completed','worker_shutdown_completed','normal_numeric_rejection','cohort_phases_completed','actual_owner_weight_load_wall_seconds','resident_candidate_rejected','resident_rejection_detail','cross_geometry_ID_failure']}
  out['owner_summaries'][str(stage)]['cases']=[{k:c.get(k)for k in['phase','batch','micro','elapsed_seconds','tail_prefill_CAPI_return_seconds','aggregate_useful_decode_tokens_s','cross_geometry_full_IDs_passed','same_micro_reference_passed','memory_free_after_phase']}for c in q.get('multi_batch_cases',[])]
print(json.dumps(out))
