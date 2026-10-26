import pathlib,json,sys,hashlib
p=pathlib.Path(sys.argv[1]);q=json.loads(p.read_bytes());assert q['completed']and q['engine_worker_shutdown_completed']and q['collection_finished'];assert q['performance_qualified'];assert q['planned_count']==16 and len(set(q['planned_ids']))==16;assert q['admitted_count']+len(q['bilateral_capacity_rejections'])==16
for t in q['warmup_trials']+q['quality_trials']:
 assert t['finite_readonly_canary_all_calls']and t['all_prefill_packed_hit'];assert all(w['all8_C4short_verified']and w['all8_precision_contract_verified']for w in t['windows'])
for t in q['warmup_trials']:assert t['diagnostic_only_excluded_from_quality_performance']
assert all(t['B']<=4 and not t['budget_stop']for t in q['quality_trials'])
if q['second8_not_executed']:assert q['first8_stop']['stop']and q['outcome']=='PREDECLARED_DATA_STOP'
else:assert q['grid_complete']
assert set(q['arms']['0']['ids'])==set(q['arms']['1']['ids']);assert q['quality_noninferiority_proven']is False
print(json.dumps({'closed':True,'step':q['step'],'replica':q['replica'],'planned':16,'admitted':q['admitted_count'],'executed_unique':len(q['arms']['0']['ids']),'outcome':q['outcome'],'quality_noninferiority_proven':False,'SHA256':hashlib.sha256(p.read_bytes()).hexdigest()}))
