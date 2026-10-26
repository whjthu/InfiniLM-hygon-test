import pathlib,json,subprocess
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';label='GLM-arrival-Model-r9260';q={'launch':None,'watch':None,'Slurm':[],'progress':{},'report':None,'logs':{}}
for key in ['launch','watch']:
 f=r/(label+'-'+key+'.json')
 if f.exists():q[key]=json.loads(f.read_bytes())
rows=[x.split('|')for x in subprocess.check_output(['sacct','-S','2026-10-24','-j','910376','-n','-P','-o','JobID,JobName%100,State%40,ExitCode,Elapsed,NodeList%120'],text=True,timeout=30).splitlines()];q['Slurm']=[x for x in rows if x[1]in [label,'GLM-arrival-keeper-r9260',label+'-before',label+'-post']]
if q['launch']:
 run=q['launch']['run']
 for stage in [0,1]:
  f=r/'results'/f'GLM-arrival-progress-r9260-{run}-s{stage}-r0.json'
  if f.exists():q['progress'][str(stage)]=json.loads(f.read_bytes())
  f=r/'results'/f'GLM-Model-phase-r9183-{run}-stage{stage}-rank0.jsonl'
  if f.exists():q.setdefault('phases',{})[str(stage)]=f.read_text(errors='replace').splitlines()[-8:]
 f=r/'results'/f'glm-arrival-r9260-{run}-stage1-rank0.json'
 if f.exists():
  a=json.loads(f.read_bytes());q['report']={k:a.get(k)for k in ['completed','warm_CCL_progress_passed','benchmark_complete','normal_reset_close','minimum_driver_free_bytes','error','remaining_process_seconds']};q['report']['episodes']=[{k:e.get(k)for k in ['label','chunked','complete','metrics','task_scores']}for e in a.get('episodes',[])];q['report']['warmups']=[{k:e.get(k)for k in ['label','complete','cap_seconds']}for e in a.get('warmups',[])];q['report']['native_calls']=len(a.get('native_events',[]))
 handle=(q['watch']or{}).get('actual_handle')
 if handle:
  for rank in [0,1]:
   f=r/'logs'/f'{label}-{handle}-{rank}.log'
   if f.exists():
    with f.open('rb')as h:h.seek(max(0,f.stat().st_size-7000));q['logs'][str(rank)]=h.read().decode(errors='replace')
print(json.dumps(q))
