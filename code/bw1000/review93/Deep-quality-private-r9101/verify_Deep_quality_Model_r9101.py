import pathlib,json,sys,hashlib
p=pathlib.Path(sys.argv[1]);q=json.loads(p.read_bytes());assert q['completed']and q['engine_worker_shutdown_completed']and q['collection_finished'];assert q['quality_complete']and q['candidate_exercised']and q['all_candidate_groups_hit'];assert len(q['quality_trials'])==16
for mode in [1,2]:
 ts=[v for v in q['quality_trials']if v['mode']==mode];assert len(ts)==8;assert len({v['id']for t in ts for v in t['records']})==32
 for t in ts:
  assert t['finite_readonly_canary_all_calls']and not t['budget_stop']and t['finished_pad_excluded'];assert len(t['records'])==4
  if mode==1:assert all(v[0]>0 and v[3]>0 and v[4]>0 for v in t['packed_prefill_counter_delta'])
  for v in t['records']:assert v['generated_ids']and v['score']['reference_token_equality_required']==False and v['stop_reason']in ['EOS','max_new_tokens']
assert q['paired_quality']['n']==32
print(json.dumps(dict(passed=True,step=q['step'],actual32_task_scores=True,quality_noninferiority_proven=False,engine_destroy_and_join=True,independent_owner_normal_markers='NOT_RETAINED;originalenginejoincontract',SHA256=hashlib.sha256(p.read_bytes()).hexdigest())))
