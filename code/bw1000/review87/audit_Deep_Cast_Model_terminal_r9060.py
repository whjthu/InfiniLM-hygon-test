import pathlib,json,hashlib,math
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';p=r/'results/deep-scalar-Cast-model-r9060-910374.1018.json';q=json.loads(p.read_bytes());w=json.loads((r/'deep-scalar-Cast-model-r9060-watch.json').read_bytes());assert w['terminal']and w['qualified_terminal_result'];assert q['completed']and q['engine_worker_shutdown_completed']and q['quality_passed']and q['performance_qualified']and not q['performance_adopted'];ref=json.loads((r/'incoming/Deep-B32-samegeom-reference-r7269.json').read_bytes());gold=json.loads((r/'results/native-engine-allselected-r816-910373.442.json').read_bytes());trials=q['owner_trials_r9060'];assert [t['label']for t in trials]==['A0','C1','C2','A3'];assert [t['cast_arm']for t in trials]==[0,1,1,0]and [t['sink_arm']for t in trials]==[1]*4;assert all(t['packed_batch_arm']==2 for t in trials)
for t in trials:
 arm=t['cast_arm'];assert t['complete']and t['full_IDs_EOS_exact']and t['generated_ids']==ref['generated_ids']and len(t['generated_ids'])==32 and all(len(x)==128 for x in t['generated_ids']);assert t['first_EOS_positions']==[None]*32
 assert t['cast_counter_delta']==([[1376,174752,173376]]*8 if arm else[[0,0,0]]*8)
 assert t['sink_counter_delta']==[[66496,4227584,461952]]*8;assert t['accepted_decode_fused_counter_delta']==[5461]*8;assert t['scratch_counter_delta']==[[0,344,0,1032]]*8
 assert t['packed_counter_delta']==[[344,0,176064,66496,528448,1894144-(173376 if arm else 0),1032]]*8
 assert len(t['cast_prefill_chunk_checks'])==8 and all(x['counter_delta']==([[172,21844,21672]]*8 if arm else[[0,0,0]]*8)for x in t['cast_prefill_chunk_checks'])
 assert [(x['step'],x['chunk'],x['rows'])for x in t['CAPI_windows']]==[(0,a,4)for a in range(0,32,4)]+[(s,0,32)for s in range(1,128)]
 pre=sum(x['seconds']for x in t['CAPI_windows']if x['step']==0);dec=sum(x['seconds']for x in t['CAPI_windows']if x['step']>0);assert math.isclose(pre,t['native_prefill_seconds'],rel_tol=1e-12)and math.isclose(dec,t['native_decode_seconds'],rel_tol=1e-12)
 for stamps,m in zip(t['request_completion_offsets_seconds'],t['request_metrics']):
  assert len(stamps)==128 and all(math.isfinite(x)for x in stamps)and all(b>a for a,b in zip(stamps,stamps[1:]));assert math.isclose(127/(stamps[-1]-stamps[0]),m['decode_tok_s'],rel_tol=1e-12)and math.isclose(stamps[0],m['TTFT_seconds'],rel_tol=1e-12)
for epoch,oldepoch in [('before','before'),('after_restore','after')]:
 rows=q['chat_cases_by_epoch'][epoch];assert len(rows)==6
 for t in rows:
  g=next(x for x in gold['chat_cases_by_epoch'][oldepoch]if x['phase']==t['phase']);assert t['generated_ids']==g['generated_ids']and t['stopped_at_eos']and t['baseline_ids_exact']and t['QA_quality_passed']and t['frozen816_full_IDs_exact']
assert q['final_A_budget']['status']=='COMPLETED'and q['final_A_budget']['threshold_seconds']==400;assert q['owner_affinity']['verified']and q['owner_affinity']['width']==4;assert sorted(v['device']for v in q['owner_affinity']['actual_owner_report']['owners'])==list(range(8));assert w['post_KFD']['passed']and not w['post_KFD']['kfd_pids'];assert q['independent_per_owner_normal_markers'].startswith('NOT_RETAINED')
out=dict(independent_audit_passed=True,hardware='BW1000',actual_handle='910374.1018',source_report_SHA256=hashlib.sha256(p.read_bytes()).hexdigest(),source_report_bytes=p.stat().st_size,watch=w,report=q)
print(json.dumps(out))
