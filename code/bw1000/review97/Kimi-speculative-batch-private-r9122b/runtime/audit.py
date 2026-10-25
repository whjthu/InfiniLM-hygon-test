"""Independent one-run native-batch audit. No historical token identity requirement."""
import pathlib,json,hashlib,re,time,math
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';name='Kimi-batch-TP16PP2-r9122b';launch=json.loads((r/(name+'-launch.json')).read_bytes());run=launch['run'];watch=json.loads((r/(name+'-watch.json')).read_bytes());m=json.loads((r/'incoming/Kimi-speculative-batch-Model-r9122b/runtime/target-manifest-r9122b.json').read_bytes());q={'qualified_runtime':False,'production_adopted':False,'quality_no_regression_proven':False,'run':run,'owners':[]}
try:
 reports=[]
 for node in range(4):
  p=r/'results'/f'kimi-batch-r9122b-{run}-node{node}.json';d=json.loads(p.read_bytes());reports.append(d);q['owners'].append({'node':node,'normal8':d.get('all8_owned_workers_shutdown_completed'),'error':d.get('error'),'shutdown_error':d.get('shutdown_error'),'SHA256':hashlib.sha256(p.read_bytes()).hexdigest()});assert d['completed']and d['all8_owned_workers_shutdown_completed']and d['serving_protocol_complete']and not d.get('error')and not d.get('shutdown_error');assert d['library_sha256']==m['native_DSO_SHA256'];assert d['batch_cases']
 cases=reports[2]['batch_cases'];reference=[(z['label'],[(v['logical'],v['generated_ids'])for v in z['request_metrics']])for z in cases]
 for d in reports:assert [(z['label'],[(v['logical'],v['generated_ids'])for v in z['request_metrics']])for z in d['batch_cases']]==reference,'same-run request/token transport differs'
 for c in cases:
  assert c['actual_native_batch']and c['configured_C']<=8 and c['configured_C']*c['configured_width']<=16 and c['decode_common_window_s']>0 and math.isfinite(c['aggregate_decode_tok_s']);assert len(c['request_metrics'])==c['configured_C'];assert len({v['logical']for v in c['request_metrics']})==c['configured_C']
  for v in c['request_metrics']:assert len(v['generated_ids'])==v['generated_tokens']and(v['EOS']or v['truncated'])and 0<=v['useful_decode_tokens']<=v['generated_tokens']
  if not c['I192_enabled']:assert c['I192_counts_delta8x2']==[[0,0]]*8
  else:assert all(row[0]>=0 and row[1]==16*row[0]for row in c['I192_counts_delta8x2'])
 assert watch['post_KFD_passed']and watch['terminal']['State']=='COMPLETED'and watch['terminal']['ExitCode']=='0:0';handle=watch['terminal']['JobID'];logs=[]
 for node in range(4):
  p=r/'logs'/f'Kimi-batch-TP16PP2-{run}-{handle}-{node}.log';raw=p.read_bytes();s=raw.decode(errors='replace');assert 'Using network IBext_v8'in s and 'Using network Socket'not in s;assert not any(re.search(r'rocBLAS.*(?:error|709)|Traceback \(most recent|HIP error|Segmentation fault|terminate called',line,re.I)for line in s.splitlines());logs.append({'node':node,'SHA256':hashlib.sha256(raw).hexdigest()})
 deadline=time.monotonic()+60
 while True:
  launch=json.loads((r/(name+'-launch.json')).read_bytes())
  if launch.get('draft_post'):break
  assert time.monotonic()<deadline,'draft cleanup receipt pending';time.sleep(1)
 assert launch['draft_srun_returncode']==0 and launch['draft_post']['passed']and not launch['draft_post']['kfd_pids'];dp=pathlib.Path(launch['draft_report']);draft=json.loads(dp.read_bytes());assert draft['status']=='COMPLETED'and draft['normal_close']and draft['remaining_request_keys_before_close']==[];dlog=r/'logs'/(launch['draft_jobname']+'-'+launch['draft_handle']+'.log');assert dlog.read_text().count('KIMI_BATCH_DRAFT_R9128_NORMAL')==1
 q.update(qualified_runtime=True,all32normal=True,draft_normal=True,draft_handle=launch['draft_handle'],draft_result_SHA256=hashlib.sha256(dp.read_bytes()).hexdigest(),logs=logs,cases=cases,actual_concurrency_coverage=sorted({z['configured_C']for z in cases}),scope='True sharedweights nativepackedbatch, perrequestEOS/context; native emitterclock notHTTP; GSM8Ksmoke pilot notstandardbenchmark; C16/32unqualified')
except BaseException as e:q['error']=repr(e);raise
finally:
 p=r/'results'/f'{name}-audit-{run}.json';assert not p.exists();p.write_text(json.dumps(q,indent=2));print(json.dumps(q))
