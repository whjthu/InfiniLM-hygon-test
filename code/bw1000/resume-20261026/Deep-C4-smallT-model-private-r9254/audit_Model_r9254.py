import pathlib,json,sys,hashlib
q=json.loads(pathlib.Path(sys.argv[1]).read_bytes());assert q['completed']and q['engine_worker_shutdown_completed']and q['collection_finished']and q['grid_complete'];assert len(q['warmup_trials'])==6 and len(q['quality_trials'])==28 and len(q['paired_cohorts'])==14
for t in q['warmup_trials']+q['quality_trials']+q['optional_B32_trials']:
 assert t['finite_readonly_canary_all_calls']and t['all_prefill_packed_hit']
 assert all(w['all8_C4short_verified']and w['all8_precision_contract_verified']for w in t['windows'])
 if t in q['quality_trials']:assert not t['budget_stop']and t['sinkarm']==1 and t['compact_EOS_rows']
assert all(t['diagnostic_only_excluded_from_quality_performance']for t in q['warmup_trials'])
for C in[1,2,4]:
 for arm in[0,1]:
  t=[x for x in q['quality_trials']if x['B']==C and x['smallt_arm']==arm];assert len(t)==8//C;assert {x['id']for z in t for x in z['records']}==set(q['first8_ids']);assert q['grid'][str(C)][str(arm)]['complete']
assert q['first_smallC_pair']['C']==1 and q['quality_noninferiority_proven']is False
print(json.dumps({'benchmark_closed':True,'step':q['step'],'pilot_independent_ids':8,'observed_quality_regressions':sum(len(v['quality_regressions'])for v in q['paired_cohorts']),'scope':'C1C2C4pairedfirst8closedcohortNativeCAPIclock,notHTTP/formalnodrop;Modeladoptionrequiresrootactualperf/qualityreview','SHA256':hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest()}))
