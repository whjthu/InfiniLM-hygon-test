"""Terminal CPU audit for the finite wholeModel diagnostic, no GPU calls."""
import pathlib,json,hashlib,math,re
from analyze_Kimi_Model_events_r8669zq import analyze
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';name='Kimi-Model-events-r8669zq';launch=json.loads((r/(name+'-launch.json')).read_text());run=launch['run'];w=json.loads((r/(name+'-watch.json')).read_text());m=json.loads((r/'incoming/Kimi-Model-events-manifest-r8669zq.json').read_text());out={'run':run,'qualified':False,'all32normal':False,'level':'WHOLE_MODEL_CURRENT_BASELINE_DIAGNOSTIC','terminal':w['terminal'],'post_KFD_passed':w['post_KFD_passed'],'owners':[]}
try:
 reports=[];failures=[]
 for node in range(4):
  p=r/'results'/f'kimi-Model-events-r8669zq-{run}-node{node}.json'
  if not p.is_file():failures.append({'node':node,'error':'missingModelreport'});continue
  q=json.loads(p.read_text());reports.append(q);out['owners'].append({'node':node,'normal8':q.get('all8_owned_workers_shutdown_completed',False),'completed':q.get('completed',False),'report_SHA256':hashlib.sha256(p.read_bytes()).hexdigest()})
  if q.get('error')or q.get('shutdown_error')or not q.get('completed'):failures.append({'node':node,'phase':q.get('phase'),'error':q.get('error'),'shutdown_error':q.get('shutdown_error')})
 out['original_Model_failures']=failures;assert not failures,failures
 labels=['TP16-B16M16-O-full-reference128','TP16-B16M16-current-ordinary-before','TP16-B16M16-owner-profile128','TP16-B16M16-current-ordinary-after']
 for node,q in enumerate(reports):
  assert(q['TP'],q['PP'],q['PP_stage'],q['layout']['TP_node_rank'])==(16,2,node//2,node%2)
  assert q['cache_geometry']==dict(max_rows=16,pages=68,page_size=256,KDA_pool_rows=17,pages_per_request=4,max_total_tokens=1024)
  expected=[45,47][node//2];assert q['compact_workspace_constructor_counts']==[[expected,expected*4468480]]*8
  assert q['preload_memory_gate']['minimum_free_bytes']>4*1024**3 and q['geometry_memory_gate']['minimum_free_bytes']>4*1024**3
  assert q['frozen648_original_reference']['matched_full_IDs_EOS']and q['library_sha256']==m['native_DSO_SHA256']
  cases=[z for z in q['serving_cases']if z['label'].startswith('TP16-B16M16-')];assert[z['label']for z in cases]==labels
  gold=cases[0]['generated_ids'];assert all(z['generated_ids']==gold and z['B']==z['frontend_cohort_M']==16 for z in cases)
  assert all(len(v)==128 for v in gold);assert [z['I192_candidate_ON']for z in cases]==[False,True,True,True]
  assert [z['timed_decode_token_only']for z in cases]==[False,True,True,True]
  assert [z['instrumented_baseline']for z in cases]==[False,False,True,False]
  assert all(z['matched_baseline_complete_IDs']is True for z in cases[1:])
  assert len(q['QA'])==8 and all(z['complete_frozen_IDs_exact']for z in q['QA'])and {z['epoch']for z in q['QA']}=={'before','after'}
  assert all(z['delta']==[[0]*4]*8 and z['I192_delta']==[z['I192_expected']]*8 for z in q['counter_dispatch'])
  assert all(z['generated_ids']==next(x['generated_ids']for x in reports[0]['serving_cases']if x['label']==z['label'])for z in cases)
 out['phase_summary']=analyze(reports);out['all32normal']=all(z['normal8']and z['completed']for z in out['owners']);assert out['all32normal']and w['post_KFD_passed']and w['terminal']['State']=='COMPLETED'and w['terminal']['ExitCode']=='0:0'
 emitter=reports[2];out['ordinary_anchors']=[{k:z[k]for k in ['label','aggregate_decode_tok_s','decode_tok_s_per_card','prefill_wall_seconds','decode_wall_seconds','TTFT_seconds','per_request_first_last_tok_s','per_request_max_ITL_seconds','end_to_end_output_tok_s','user_clock']}for z in emitter['serving_cases']if '-ordinary-'in z['label']];assert len(out['ordinary_anchors'])==2
 out['interleaved_status']=emitter.get('interleaved_finaltokens_status');out['interleaved']=[{k:z.get(k)for k in ['label','completed','matched_independent_complete_IDs_EOS','first_differences','TTFT_seconds','per_request_first_last_tok_s','per_request_max_ITL_seconds','service_wall_seconds','end_to_end_output_tok_s','post_all_prefill_decode_tok_s','timing_scope','geometry']}for z in emitter.get('interleaved_cases',[])];out['interleaved_Model_adoption']=False
 out['network_logs']=[]
 for node in range(4):
  p=r/'logs'/f'Kimi-Model-events-{run}-{w["terminal"]["JobID"]}-{node}.log';raw=p.read_bytes();text=raw.decode(errors='replace');assert 'Using network IBext_v8'in text and 'Using network Socket'not in text
  out['network_logs'].append({'node':node,'SHA256':hashlib.sha256(raw).hexdigest(),'actual_IBext_channels':sum('via NET/IBext_v8'in v for v in text.splitlines()),'GDR':'UNKNOWN'})
 out['qualified']=True;out['scope']='Baselinefull128+QA/32normal/actualownerphase schema qualified; diagnostic streamspans notactivekernels/newModelgain. Interleaved status separately retainsnegativeIDs; noautomaticpromotion.'
except BaseException as e:out['error']=repr(e);raise
finally:
 p=r/'results'/f'{name}-audit-{run}.json';assert not p.exists();p.write_text(json.dumps(out,indent=2));print(json.dumps({k:out.get(k)for k in ['run','qualified','all32normal','interleaved_status','error']}))
