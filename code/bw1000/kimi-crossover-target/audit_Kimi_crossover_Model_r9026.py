import pathlib,json,hashlib,re
r=pathlib.Path.home()/'hygon-inference-refactor-20261014'
name='Kimi-crossover-TP16PP2-r9026';launch=json.loads((r/(name+'-launch.json')).read_text());run=launch['run'];w=json.loads((r/(name+'-watch.json')).read_text());m=json.loads((r/'incoming/Kimi-crossover-manifest-r9026.json').read_text())
out={'diagnostic_complete':False,'Model_performance_claim':False,'speculative_serving_qualified':False,'run':run,'owners':[]}
try:
 reports=[]
 for node in range(4):
  p=r/'results'/f'kimi-crossover-r9026-{run}-node{node}.json';q=json.loads(p.read_text());reports.append(q)
  out['owners'].append({'node':node,'normal8':q.get('all8_owned_workers_shutdown_completed'),'error':q.get('error'),'SHA256':hashlib.sha256(p.read_bytes()).hexdigest()})
  assert q['completed'] and q['all8_owned_workers_shutdown_completed'] and not q.get('error') and not q.get('shutdown_error')
  assert q['library_sha256']==m['native_DSO_SHA256'] and q['crossover_complete'] and q['performance_claim'] is False and q['allposition_finalIDs_EOS_pass'] is False
  arms=q['crossover_arms'];assert [a['arm'] for a in arms]==list('ABCDE')
  assert all(len(a['actual_ids'])==32 and len(a['expected_A_ids'])==32 and sum(a['widths'])==32 for a in arms)
  assert all(a['real_acceptance_claim'] is False for a in arms)
  expected_calls=33+33+26+12+5
  assert len(q['crossover_calls'])==expected_calls
  assert all(z['arrays']['flags_argument']==3 and z['arrays']['count_argument']==1 for z in q['crossover_calls'])
  if node==2:
   for arm in 'ABCDE':
    calls=[z for z in q['crossover_calls'] if z['arm']==arm]
    head=[h for z in calls for owner in z['owners'] for h in owner['head'] if owner['node']==2]
    assert {h['input_position'] for h in head}=={536,537},(arm,head)
    traces=[v for z in calls for owner in z['owners'] for v in owner['trace_after']['records'] if owner['node'] in (0,2)]
    assert {536,537}<={p for v in traces for p in v['positions']}
    assert {19,37,66,78,90}<={v['layer'] for v in traces if v['point']=='decoder'}
 assert all(q['crossover_arms']==reports[0]['crossover_arms'] for q in reports)
 assert w['post_KFD_passed'] and w['terminal']['State']=='COMPLETED' and w['terminal']['ExitCode']=='0:0'
 handle=w['terminal']['JobID'];logs=[]
 for node in range(4):
  p=r/'logs'/f'Kimi-crossover-TP16PP2-{run}-{handle}-{node}.log';raw=p.read_bytes();s=raw.decode(errors='replace')
  assert 'Using network IBext_v8' in s and 'Using network Socket' not in s
  assert not any(re.search(r'rocBLAS.*(?:error|709)|Traceback \(most recent|HIP error|Segmentation fault|terminate called',line,re.I) for line in s.splitlines())
  logs.append({'node':node,'SHA256':hashlib.sha256(raw).hexdigest()})
 out.update(diagnostic_complete=True,all32normal=True,logs=logs,arms=reports[2]['crossover_arms'],scope='Actual Math0 fixed causal crossover; finite differences retained. No EOS/replay/page255/serving/speculative throughput qualification.')
except BaseException as e:out['error']=repr(e);raise
finally:
 p=r/'results'/f'{name}-audit-{run}.json';assert not p.exists();p.write_text(json.dumps(out,indent=2));print(json.dumps(out))
