import pathlib,json,subprocess,datetime
r=pathlib.Path.home()/'hygon-inference-refactor-20261014'
launch=json.loads((r/'glm-transient-Model-r8868-launch.json').read_text());run=launch['run']
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'CPU_read_only':True,'run':run,'PID':launch['pid'],'watch_PID':launch['watch_pid']}
p=r/'results/GLM-transient-Model-watch-r8868.json';out['watch']=json.loads(p.read_text())if p.exists()else None
s=subprocess.check_output(['sacct','-S','2026-10-24','-j','910371','-n','-P','-o','JobID,JobName%80,State,ExitCode,Elapsed,NodeList%100'],text=True,timeout=30)
out['actual']=[v.split('|')for v in s.splitlines()if 'glm-transient-Model-r8868|'in v]
out['phases']={};out['owner_summaries']={}
for a in('A','B'):
 for stage in(0,1):
  for rank in range(8):
   p=r/'results'/f'GLM-Model-phase-r8868-{run}-{a}-stage{stage}-rank{rank}.jsonl'
   out['phases'][f'{a}:{stage}:{rank}']=p.read_text().splitlines()[-2:]if p.exists()else[]
  p=r/'results'/f'glm-transient-Model-r8868-{run}-{a}-stage{stage}-rank0.json'
  if p.exists():
   q=json.loads(p.read_text());v={k:q.get(k)for k in('completed','worker_shutdown_completed','resident_candidate_rejected','resident_rejection_detail','actual_owner_weight_load_wall_seconds','cohort_phases_completed')}
   v['cases']=[{k:c.get(k)for k in('phase','batch','micro','elapsed_seconds','cross_geometry_full_IDs_passed','aggregate_useful_decode_tokens_s')}for c in q.get('multi_batch_cases',[])]
   out['owner_summaries'][f'{a}:{stage}']=v
out['logtails']={}
for row in out['actual']:
 for node in range(4):
  p=r/'logs'/f'glm-transient-Model-r8868-{row[0]}-{node}.log'
  if p.exists():
   with p.open('rb')as f:f.seek(max(0,p.stat().st_size-3000));out['logtails'][str(node)]=f.read().decode(errors='replace')
p=r/'results'/f'GLM-DP-common-clock-r8868-{run}.json'
if p.exists():
 q=json.loads(p.read_text());out['clock']={k:v for k,v in q.items()if k not in('records','token_packets')}
print(json.dumps(out))
