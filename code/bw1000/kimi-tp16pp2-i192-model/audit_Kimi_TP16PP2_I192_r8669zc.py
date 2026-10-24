"""Independent terminal CPU audit; no Engine/KFD calls, no inherited TPS."""
import pathlib,json,hashlib,math,re
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';name='Kimi-TP16PP2-I192-r8669zc';launch=json.loads((r/(name+'-launch.json')).read_text());run=launch['run'];w=json.loads((r/(name+'-watch.json')).read_text());m=json.loads((r/'incoming/Kimi-TP16PP2-I192-manifest-r8669zc.json').read_text());out={'run':run,'qualified':False,'all32normal':False,'scope':m['scope'],'terminal':w['terminal'],'post_KFD_passed':w['post_KFD_passed'],'owners':[]}
try:
 original_failures=[]
 for node in range(4):
  p=r/'results'/f'kimi-TP16PP2-I192-r8669zc-{run}-node{node}.json'
  if not p.is_file():original_failures.append({'node':node,'phase':'missing_report','error':'Modelreportmissing'});continue
  q=json.loads(p.read_text())
  if q.get('error')or q.get('shutdown_error')or not q.get('completed'):
   original_failures.append({'node':node,'phase':q.get('phase'),'error':q.get('error'),'shutdown_error':q.get('shutdown_error'),'normal8':q.get('all8_owned_workers_shutdown_completed',False)})
 out['original_Model_failures']=original_failures
 if original_failures:raise AssertionError(('original_Model_failed_before_qualification',original_failures))
 reports=[];labels=['TP16-B16M16-O-full-reference128','TP16-B16M16-N-qualification128','TP16-B16M16-O-ordinary0','TP16-B16M16-N-ordinary1','TP16-B16M16-N-ordinary2','TP16-B16M16-O-ordinary3']
 for node in range(4):
  p=r/'results'/f'kimi-TP16PP2-I192-r8669zc-{run}-node{node}.json';q=json.loads(p.read_text());reports.append(q);out['owners'].append({'node':node,'normal8':q.get('all8_owned_workers_shutdown_completed',False),'completed':q['completed'],'report_SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'error':q.get('error'),'shutdown_error':q.get('shutdown_error')})
  if q.get('error')or q.get('shutdown_error')or not q.get('completed'):
   out.setdefault('original_Model_failures',[]).append({'node':node,'phase':q.get('phase'),'error':q.get('error'),'shutdown_error':q.get('shutdown_error'),'normal8':q.get('all8_owned_workers_shutdown_completed',False)})
   raise AssertionError(('original_Model_failed_before_qualification',node,q.get('phase'),q.get('error'),q.get('shutdown_error')))
  assert q['node']==node and q['PP_stage']==node//2 and q['nodes']==4 and q['TP']==16 and q['PP']==2
  assert q['layout']['TP_node_rank']==node%2 and q['layout']['sampling_node']==2
  assert q['cache_geometry']==dict(max_rows=16,pages=68,page_size=256,KDA_pool_rows=17,pages_per_request=4,max_total_tokens=1024)
  expected=[45,47][node//2];assert q['compact_workspace_constructor_counts']==[[expected,expected*4468480]]*8
  assert q['preload_memory_gate']['minimum_free_bytes']>4*1024**3 and q['geometry_memory_gate']['minimum_free_bytes']>4*1024**3
  assert len(q['postQA_reset_memory']['before'])==8 and len(q['postQA_reset_memory']['after'])==8
  assert q['library_sha256']==m['native_DSO_SHA256'];assert q['serving_protocol_complete']and not q.get('error')and not q.get('shutdown_error')
  cases=[z for z in q['serving_cases']if z['label'].startswith('TP16-B16M16-')];assert[z['label']for z in cases]==labels
  assert all(z['B']==16 and z['frontend_cohort_M']==16 and z['policy']==1 for z in cases)
  gold=cases[0]['generated_ids'];assert all(z['generated_ids']==gold for z in cases);assert cases[0]['matched_baseline_complete_IDs']is None and all(z['matched_baseline_complete_IDs']is True for z in cases[1:]);assert [z['I192_candidate_ON']for z in cases]==[False,True,False,True,True,False]
  assert {z['epoch']for z in q['QA']}=={'before','after'}and all(z['complete_frozen_IDs_exact']and z['policy']==0 for z in q['QA'])
  assert all(z['delta']==[[0]*4]*8 and z['I192_delta']==[z['I192_expected']]*8 for z in q['counter_dispatch']);assert any(z['candidate']and z['N']==16 and z['I192_expected'][0]>0 for z in q['counter_dispatch'])
  assert all(z['generated_ids']==[x for x in reports[0]['serving_cases']if x['label']==z['label']][0]['generated_ids']for z in cases)
  assert q['planned_resident_memory']['layers']==([[0,46],[46,93]][node//2]);out.setdefault('memory_phases',[]).append({'node':node,**{k:q.get(k)for k in ['planned_resident_memory','memory_after_create','memory_after_finish','last_forward_memory','geometry_memory_gate','loading_seconds','memory_after_finish_domains']}})
 out['all32normal']=all(z['normal8']and z['completed']for z in out['owners']);assert out['all32normal']and w['post_KFD_passed']and w['terminal']['State']=='COMPLETED'and w['terminal']['ExitCode']=='0:0'
 emitter=reports[2];ordinary=[z for z in emitter['serving_cases']if '-ordinary'in z['label']];assert len(ordinary)==4
 out['ordinary']=[{k:z[k]for k in ['label','I192_candidate_ON','aggregate_decode_tok_s','decode_tok_s_per_card','decode_wall_seconds','prefill_wall_seconds','end_to_end_output_tok_s','TTFT_seconds','per_request_first_last_tok_s','per_request_max_ITL_seconds','instrumented_baseline','user_clock']}for z in ordinary]
 for z in ordinary:
  assert not z['instrumented_baseline'];assert math.isclose(z['useful_decode_outputs']/z['decode_wall_seconds'],z['aggregate_decode_tok_s'],rel_tol=1e-12)
 logs=[];handle=w['terminal']['JobID']
 for node in range(4):
  matches=list((r/'logs').glob(f'Kimi-TP16PP2-I192-{run}-{handle}-{node}.log'));assert len(matches)==1;raw=matches[0].read_bytes();text=raw.decode(errors='replace');assert 'Using network IBext_v8'in text and 'Using network Socket'not in text
  assert not any(re.search(r'rocBLAS.*(?:error|709)|Traceback \(most recent|HIP error|Segmentation fault|terminate called',line,re.I)for line in text.splitlines());logs.append({'node':node,'SHA256':hashlib.sha256(raw).hexdigest(),'actual_IBext_channels':sum('via NET/IBext_v8'in line for line in text.splitlines()),'GDR':'UNKNOWN'})
 out['network_logs']=logs;out['qualified']=True
except BaseException as e:out['error']=repr(e);raise
finally:
 p=r/'results'/f'{name}-audit-{run}.json';assert not p.exists();p.write_text(json.dumps(out,indent=2));print(json.dumps({k:out.get(k)for k in ['run','qualified','all32normal','error']}))
