"""Finite terminal audit for cap16 A/C schedule test. CPU only."""
import pathlib,json,hashlib
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';name='Kimi-interleaved-service-r8669aaa';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
launch=json.loads((r/(name+'-launch.json')).read_text());run=launch['run'];w=json.loads((r/(name+'-watch.json')).read_text());m=json.loads((r/'incoming/Kimi-interleaved-service-manifest-r8669aaa.json').read_text());out={'run':run,'qualified':False,'all32normal':False,'level':'FINITE_MIXED_SCHEDULE_MODEL_QUALIFICATION','terminal':w.get('terminal'),'post_KFD_passed':w.get('post_KFD_passed',False),'owners':[]}
try:
 reports=[];failures=[]
 for node in range(4):
  p=r/'results'/f'kimi-interleaved-service-r8669aaa-{run}-node{node}.json'
  if not p.is_file():failures.append({'node':node,'error':'missingModelreport'});continue
  q=json.loads(p.read_text());reports.append(q);out['owners'].append({'node':node,'normal8':q.get('all8_owned_workers_shutdown_completed',False),'completed':q.get('completed',False),'report_SHA256':sha(p)})
  if q.get('error')or q.get('shutdown_error')or not q.get('completed'):failures.append({'node':node,'phase':q.get('phase'),'error':q.get('error'),'shutdown_error':q.get('shutdown_error')})
 out['original_Model_failures']=failures;assert not failures,failures
 chat_path=r/'results/kimi-chat-cases-r20.json';assert sha(chat_path)==m['remote_evidence_sha256']['results/kimi-chat-cases-r20.json'];chat=json.loads(chat_path.read_text());expectedQA={(epoch,0,task)for epoch in ['before','after']for task in chat['cases']};assert len(expectedQA)==16
 for node,q in enumerate(reports):
  assert(q['TP'],q['PP'],q['PP_stage'],q['layout']['TP_node_rank'])==(16,2,node//2,node%2)
  assert q['cache_geometry']==dict(max_rows=16,pages=68,page_size=256,KDA_pool_rows=17,pages_per_request=4,max_total_tokens=1024)
  moes=[45,47][node//2];assert q['compact_workspace_constructor_counts']==[[moes,moes*4468480]]*8
  assert q['preload_memory_gate']['minimum_free_bytes']>4*1024**3 and q['geometry_memory_gate']['minimum_free_bytes']>4*1024**3
  assert q['frozen648_original_reference']['matched_full_IDs_EOS']and q['library_sha256']==m['native_DSO_SHA256']
  cases=[z for z in q['serving_cases']if z['label']=='TP16-B16M16-A-current-ordinary128'];assert len(cases)==1
  a=cases[0];assert a['B']==a['frontend_cohort_M']==16 and a['policy']==1 and a['I192_candidate_ON']and a['timed_decode_token_only']and not a['instrumented_baseline'];gold=a['generated_ids'];assert len(gold)==16 and all(len(v)==128 for v in gold)
  assert len(q['QA'])==16 and all(z['complete_frozen_IDs_exact']for z in q['QA'])and {(z['epoch'],z['policy'],z['task'])for z in q['QA']}==expectedQA
  assert all(z['delta']==[[0]*4]*8 and z['I192_delta']==[z['I192_expected']]*8 for z in q['counter_dispatch'])
  assert gold==next(z['generated_ids']for z in reports[0]['serving_cases']if z['label']==a['label'])
  for z in q.get('interleaved_cases',[]):
   assert z['completed']and len(z['generated_ids'])==16 and all(len(v)==128 for v in z['generated_ids'])
   assert z['generated_ids']==next(v['generated_ids']for v in reports[0]['interleaved_cases']if v['label']==z['label'])
   assert z['global_physical_rows']==list(range(1,17))and z['input_lengths']==[512]*16
   assert z['generated_ids']==gold or z['first_differences'] # Mismatch is retained, never self-gold.
 out['all32normal']=all(z['normal8']and z['completed']for z in out['owners']);assert out['all32normal']and w['post_KFD_passed']and w['terminal']['State']=='COMPLETED'and w['terminal']['ExitCode']=='0:0'
 emitter=reports[2];out['A_ordinary']=[{k:z[k]for k in ['label','generated_ids','aggregate_decode_tok_s','decode_tok_s_per_card','prefill_wall_seconds','decode_wall_seconds','TTFT_seconds','per_request_first_last_tok_s','per_request_max_ITL_seconds','end_to_end_output_tok_s','user_clock']}for z in emitter['serving_cases']if z['label']=='TP16-B16M16-A-current-ordinary128']
 out['candidate_status']=emitter.get('interleaved_finaltokens_status');out['repeat_status']=emitter.get('interleaved_repeat_status');out['C_cases']=[{k:z.get(k)for k in ['label','completed','generated_ids','matched_independent_complete_IDs_EOS','first_differences','TTFT_seconds','per_request_first_last_tok_s','per_request_max_ITL_seconds','service_wall_seconds','end_to_end_output_tok_s','end_to_end_output_tok_s_per_card','post_all_prefill_decode_tok_s','timing_scope','geometry']}for z in emitter.get('interleaved_cases',[])]
 out['network_logs']=[]
 for node in range(4):
  p=r/'logs'/f'Kimi-interleaved-service-{run}-{w["terminal"]["JobID"]}-{node}.log';text=p.read_text(errors='replace');assert 'Using network IBext_v8'in text and 'Using network Socket'not in text
  out['network_logs'].append({'node':node,'SHA256':sha(p),'actual_IBext_channels':sum('via NET/IBext_v8'in v for v in text.splitlines()),'GDR':'UNKNOWN'})
 out['baseline_and_lifecycle_qualified']=True
 out['qualified']=out['candidate_status']=='PASS'and all(z['matched_independent_complete_IDs_EOS']for z in out['C_cases'])
 out['scope']='One A/C finite cap16 schedule comparison and optional C repeat, not balancedABBA/newfastestclaim. C changes geometry; only fullIDs/EOS+QA qualify. Ordinaryuserfirstlast/TTFT/maxITL/E2E are emitterready, notclient; mixed earlydecode not puredecode benchmark.'
 assert out['qualified'],('candidate_not_qualified',out['candidate_status'],out['repeat_status'])
except BaseException as e:out['error']=repr(e);raise
finally:
 p=r/'results'/f'{name}-audit-{run}.json';assert not p.exists();p.write_bytes((json.dumps(out,indent=2)+'\n').encode());print(json.dumps({k:out.get(k)for k in ['run','qualified','all32normal','candidate_status','error']}))
