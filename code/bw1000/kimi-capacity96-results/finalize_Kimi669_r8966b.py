"""Offline actual-report audit/compact; no Engine, CUDA, job, QA or KFD calls."""
import pathlib,json,hashlib,math,statistics
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();run='capacity96service8669aaj-f09f62e8537c5a81';wpath=r/'Kimi-capacity96-service-r8669aaj-watch.json';w=json.loads(wpath.read_text());old=r/'results'/f'Kimi-capacity96-service-r8669aaj-audit-{run}.json'
out={'qualified':False,'actual_handle':'910375.669','run':run,'terminal':w['terminal'],'watch_SHA256':sha(wpath),'original_audit_exit':w['audit_exit'],'original_audit_SHA256':sha(old),'original_audit_preserved':True,'audit_correction':'OriginalL11expects8 QA records; actualchat8cases*before_after=16. Offlineonlyvalidateactualcasekeys/epochs/count, noQA rerun/Slurm/GPU/KFD.','owners':[],'cases':[]}
chat=json.loads((r/'results/kimi-chat-cases-r20.json').read_text());frozen_path=r/'incoming/r4917-Kimi-two-geometry-frozen-gold.json';assert sha(frozen_path)=='ab686988a0952fe094d76773902e784e5d4be577c44c57c23493d702cc1d5625';frozen=json.loads(frozen_path.read_text());eos=set(frozen['eos_token_ids']);prompts=frozen['natural_input_ids'];gold=next(v for v in frozen['geometries']if(v['B'],v['M'])==(64,16))['fixed_complete_IDs'];assert all(gold[i]==gold[i%16]for i in range(64));reports=[]
try:
 for node in range(4):
  p=r/'results'/f'kimi-capacity96-service-r8669aaj-{run}-node{node}.json';q=json.loads(p.read_text());reports.append(q)
  owner={'node':node,'raw_report_SHA256':sha(p),'completed':q.get('completed'),'normal8':q.get('all8_owned_workers_shutdown_completed'),'error':q.get('error'),'shutdown_error':q.get('shutdown_error'),'loading_seconds':q['loading_seconds'],'loader_flushes':q['loader_flushes'],'CPU_copy_stats_after_loading':q['CPU_copy_stats_after_loading'],'QA_count':len(q['QA']),'minimum_finish_free_bytes':min(x[0]for x in q['memory_after_finish']),'preload_gate':q['preload_memory_gate'],'postQA_gate':q['geometry_memory_gate'],'cache_geometry':q['cache_geometry']};out['owners'].append(owner)
  assert q['node']==node and q['PP_stage']==node and q['TP']==8 and q['PP']==4 and q['nodes']==4
  assert owner['completed']and owner['normal8']and not owner['error']and not owner['shutdown_error']
  assert q['cache_geometry']==dict(max_rows=96,pages=291,page_size=256,KDA_pool_rows=97,pages_per_request=3,max_total_tokens=768)
  assert q['library_sha256']=='f6f10c2b243e0336074b872859e6ca6acb08effdcd56d86b9ac4df1bf2ddb5e9'
  actual={(x['epoch'],x['task'],x['policy'])for x in q['QA']};expected={(epoch,task,0)for epoch in ('before','after')for task in chat['cases']};assert actual==expected and len(q['QA'])==len(expected)==16
  assert all(x['complete_frozen_IDs_exact']for x in q['QA'])
  known=json.loads((r/'results/modern-kimi-quality-fp32-r20-910375.77-summary.json').read_text())
  for epoch,task,policy in sorted(expected):
   label='QA-'+epoch+'-p0-'+task;rows=[z for z in q['serving_cases']if z['label']==label];assert len(rows)==1
   z=rows[0];key='sort'if task.startswith('sort_probe')else task
   original=[v['generated_ids'][0]for v in known['cases']if v['task']==key and v['grouped']]
   assert z['B']==1 and z['policy']==0 and z['generated_ids'][0]in original
   assert all(not any(t in eos for t in ids[:-1])and(len(ids)==128 or ids[-1]in eos)for ids in z['generated_ids'])
  owner['QA_actual_generated_IDs_independently_rechecked']=16

  assert owner['preload_gate']['minimum_free_bytes']>4*1024**3 and owner['postQA_gate']['minimum_free_bytes']>4*1024**3
  cases=[v for v in q['serving_cases']if v['label'].startswith(('B64-','B96-'))];assert [v['B']for v in cases]==[64,96]
  for z in cases:
   B=z['B'];expected_ids=[gold[i%16]for i in range(B)];assert z['generated_ids']==expected_ids and z['matched_baseline_complete_IDs'] is True
   assert z['max_output_tokens']==128 and all(len(v)==128 or v[-1]in eos for v in z['generated_ids'])and all(not any(x in eos for x in v[:-1])for v in z['generated_ids'])
   assert z['input_lengths']==[512]*B and z['frontend_cohort_M']==16 and z['policy']==1 and z['all_service_cards']==32 and not z['instrumented_baseline']
  assert all(x['delta']==[x['expected']]*8 for x in q['counter_dispatch'])
  assert any(x['N']==16 and x['policy']==1 and x['expected'][0]>0 and x['expected'][1]==16*x['expected'][0]and x['expected'][2:]==[0,0]for x in q['counter_dispatch'])
 for B in (64,96):
  arms=[next(v for v in q['serving_cases']if v['B']==B and v['label'].startswith('B'))for q in reports];z=arms[3];assert all(v['generated_ids']==z['generated_ids']for v in arms)
  useful=sum(sum(x not in eos for x in row[1:])for row in z['generated_ids']);assert math.isclose(useful/z['decode_wall_seconds'],z['aggregate_decode_tok_s'],rel_tol=1e-12)
  values={k:z[k]for k in ['label','B','frontend_cohort_M','aggregate_decode_tok_s','decode_tok_s_per_card','decode_wall_seconds','prefill_wall_seconds','end_to_end_output_tok_s','service_wall_seconds','useful_all_outputs','useful_decode_outputs','user_clock']}
  values.update(input_tokens_per_s=B*512/z['prefill_wall_seconds'],E2E_per_card=z['end_to_end_output_tok_s']/32,user_mean_tok_s=statistics.mean(v for v in z['per_request_first_last_tok_s']if v is not None),user_min_tok_s=min(v for v in z['per_request_first_last_tok_s']if v is not None),user_max_tok_s=max(v for v in z['per_request_first_last_tok_s']if v is not None),TTFT_mean_seconds=statistics.mean(z['TTFT_seconds']),TTFT_min_seconds=min(z['TTFT_seconds']),TTFT_max_seconds=max(z['TTFT_seconds']),ITL_max_seconds=max(v for v in z['per_request_max_ITL_seconds']if v is not None),full128_rows=sum(len(v)==128 for v in z['generated_ids']),EOS_rows=sum(v[-1]in eos for v in z['generated_ids']),full_IDs_JSON_SHA256=hashlib.sha256(json.dumps(z['generated_ids'],separators=(',',':')).encode()).hexdigest(),complete_frozen_mapped_IDs_EOS_match=True,all4frontend_IDagreement=True)
  out['cases'].append(values)
 assert w['terminal']['State']=='COMPLETED'and w['terminal']['ExitCode']=='0:0'and w['post_KFD_passed']and len(w['post_KFD'])==4 and all(v['passed']and not v['kfd_pids']for v in w['post_KFD'])
 out['unique_post']=[{'node':v['node'],'step':v['step'],'passed':v['passed'],'kfd_pids':v['kfd_pids']}for v in w['post_KFD']];out['all32normal']=True;out['qualified']=True;out['distinct_prompts']=len({tuple(x)for x in prompts});out['limits']='OneB64thenB96 order/noABBA, sameI512O128 two promptfamilies repeated; expliciti%16 historicalM16cohort-positiongold, noti%2. Counter provesproviderbranch, notGPUoccupancy. NoHTTP/clientclock/DSpark/profile/bestservingclaim. Guards unchanged; coldpass wasnotactivationproof butbothactualarmscompleted.'
 a,b=out['cases'];out['B96_vs_B64_percent']={k:(b[k]/a[k]-1)*100 for k in ['aggregate_decode_tok_s','decode_tok_s_per_card','end_to_end_output_tok_s','user_mean_tok_s','TTFT_mean_seconds','ITL_max_seconds']}
except BaseException as exc:out['error']=repr(exc);raise
finally:
 p=r/'results/Kimi669-B64-B96-offline-final-r8966b.json';assert not p.exists();p.write_bytes((json.dumps(out,indent=2)+'\n').encode());print(json.dumps({'SHA256':sha(p),'result':out}))
