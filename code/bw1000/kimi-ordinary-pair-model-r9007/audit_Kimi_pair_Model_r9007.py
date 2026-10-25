import pathlib,json,hashlib,re
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';name='Kimi-TP16PP2-I192-r9007';launch=json.loads((r/(name+'-launch.json')).read_text());run=launch['run'];w=json.loads((r/(name+'-watch.json')).read_text());m=json.loads((r/'incoming/Kimi-TP16PP2-I192-manifest-r9007.json').read_text());out={'qualified':False,'Model_performance_claim':False,'run':run,'owners':[]}
try:
 reports=[]
 for node in range(4):
  p=r/'results'/f'kimi-pair-Model-r9007-{run}-node{node}.json';q=json.loads(p.read_text());out['owners'].append({'node':node,'normal8':q.get('all8_owned_workers_shutdown_completed'), 'error':q.get('error'),'shutdown_error':q.get('shutdown_error'),'SHA256':hashlib.sha256(p.read_bytes()).hexdigest()});reports.append(q)
  assert q['completed'] and q['all8_owned_workers_shutdown_completed'] and not q.get('error') and not q.get('shutdown_error'),out['owners'][-1]
  assert q['library_sha256']==m['native_DSO_SHA256'] and q['allposition_finalIDs_EOS_pass'] and q['actual_page255_crossing_tested'] and q['actual_EOS_skip_tested']
  cases=q['pair_cases'];assert len([z for z in cases if z['label'].startswith('page255-prefix')])==8
  assert all(z['complete_IDs_firstEOS_equal_samecontext_AR']for z in cases if z['mode']=='paired_target_recomputed_greedy')
  assert {z['epoch']for z in q['QA']}=={'before','after'} and all(z['complete_frozen_IDs_exact']for z in q['QA'])
  assert q['cache_geometry']==dict(max_rows=16,pages=68,page_size=256,KDA_pool_rows=17,pages_per_request=4,max_total_tokens=1024)
 assert all([(z['label'],z['generated_ids'])for z in q['pair_cases']]==[(z['label'],z['generated_ids'])for z in reports[0]['pair_cases']]for q in reports)
 assert w['post_KFD_passed'] and w['terminal']['State']=='COMPLETED' and w['terminal']['ExitCode']=='0:0'
 logs=[];handle=w['terminal']['JobID']
 for node in range(4):
  p=r/'logs'/f'Kimi-TP16PP2-I192-{run}-{handle}-{node}.log';raw=p.read_bytes();s=raw.decode(errors='replace');assert 'Using network IBext_v8'in s and 'Using network Socket'not in s
  assert not any(re.search(r'rocBLAS.*(?:error|709)|Traceback \(most recent|HIP error|Segmentation fault|terminate called',line,re.I)for line in s.splitlines());logs.append({'node':node,'SHA256':hashlib.sha256(raw).hexdigest()})
 phases=reports[2]['paired_phase_calls'];out.update(qualified=True,all32normal=True,logs=logs,phase_counts={p:sum(z['phase']==p for z in phases)for p in ['ordinary','prefill','verify','replay','continuation']},phase_maxhost_CAPI_seconds={p:sum(z['CAPI_maxhost_seconds']for z in phases if z['phase']==p)for p in ['ordinary','prefill','verify','replay','continuation']},scope='One loaded currentTP16PP2/Math0target,originalordinarybaselinevsoraclecandidate diagnostic;completeIDsfirstEOS/commitreplay/actualpageboundary;no trainedacceptance/specspeedclaim')
except BaseException as e:out['error']=repr(e);raise
finally:
 p=r/'results'/f'{name}-audit-{run}.json';assert not p.exists();p.write_text(json.dumps(out,indent=2));print(json.dumps(out))
