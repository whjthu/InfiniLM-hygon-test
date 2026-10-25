import pathlib,json,hashlib
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';launch=json.loads((r/'glm-transient-Model-r8855-launch.json').read_text());run=launch['run'];q={'CPU_only':True,'GPU':False,'dlopen':False,'run':run,'owners':[]}
for stage in [0,1]:
 p=r/'results'/f'glm-transient-Model-r8855-{run}-stage{stage}-rank0.json';d=json.loads(p.read_text());rows=[]
 for c in d['multi_batch_cases']:
  ev=c.get('events',[]);rows.append({'phase':c['phase'],'saved_effective_policy':c.get('routed_native_policy_r5123'),'delta':c.get('native_operator_count_delta'),'prefill_events':[{'flat':e.get('flat_tokens'),'policy':e.get('native_policy'),'before':e.get('native_before'),'after':e.get('native_after')}for e in ev if e['round']==0]})
 q['owners'].append({'stage':stage,'rank':0,'path':str(p),'SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'normal':d.get('worker_shutdown_completed'),'completed':d.get('completed'),'cases':rows})
print(json.dumps(q))
