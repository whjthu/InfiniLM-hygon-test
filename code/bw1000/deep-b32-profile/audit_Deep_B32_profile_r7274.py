import pathlib,json,hashlib,sys,subprocess,math
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';tag=sys.argv[1];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows=[v.split('|')for v in subprocess.check_output(['sacct','-S','2026-10-23','-j',tag,'-n','-P','-o','JobID,State,ExitCode'],text=True).splitlines()];assert len(rows)==1 and rows[0][1:]==['COMPLETED','0:0']
p=r/'results'/f'deep-B32-profile-r7274-{tag}.json';q=json.loads(p.read_text());assert q['completed']and q['engine_worker_shutdown_completed']and q['formal_profile_complete']and not q['performance_adopted']
assert q['library_sha256']=='0e0512551483457af01944e9c9298442f499d959836a1c5b66dfa438a3c6c4ea'
launch=json.loads((r/'results/Deep-B32-profile-r7274-launch.json').read_text())
for n,h in launch['sources'].items():assert sha(r/'incoming'/n)==h
for key in ['owner_affinity_before_weight_load','owner_affinity_after_weight_load','owner_affinity']:
 a=q[key];assert a['verified']and a['width']==4;owners=a['actual_owner_report']['owners'];assert sorted(o['device']for o in owners)==list(range(8));masks=[]
 for o in owners:
  assert o['changes']==1;mask=list(map(int,o['verified_mask'].split(',')));assert len(mask)==4;masks+=mask
 assert len(masks)==len(set(masks))==32
ref=json.loads((r/'incoming/Deep-B32-samegeom-reference-r7269.json').read_text());trials=q['profile_B32_r7274'];assert [v['trial']for v in trials]==['warmup','baseline-r1','baseline-r2']
assert q['host_buffer_count']==1 and q['host_buffer_payload_bytes']==32*129280*2
summ=[]
for c in trials:
 assert c['generated_ids']==ref['generated_ids']and c['full4096_final_IDs_exact_reference']and not c['mismatches']and c['CPUargmax_all128']and c['fullvocab_checked_CAPI_windows']==135
 assert c['projection_counter_delta']==[127*43]*8
 windows=c['CAPI_windows'];assert len(windows)==135
 assert [(v['step'],v['chunk'],v['rows'])for v in windows]==[(0,a,4)for a in range(0,32,4)]+[(s,0,32)for s in range(1,128)]
 assert all(math.isfinite(v['seconds'])and v['seconds']>0 for v in windows)
 pre=sum(v['seconds']for v in windows if v['step']==0);dec=sum(v['seconds']for v in windows if v['step']>0)
 assert math.isclose(pre,c['native_prefill_seconds'],rel_tol=1e-12)and math.isclose(dec,c['native_decode_seconds'],rel_tol=1e-12)
 assert math.isclose(16384/pre,c['native_CAPI_prefill_input_tok_s'],rel_tol=1e-12)and math.isclose(4064/dec,c['native_CAPI_decode_output_tok_s'],rel_tol=1e-12)
 assert sum(bool(v['traced'])for v in windows)==int(c['trial']=='warmup')and c['formal_measurement']==(c['trial']!='warmup')
 summ.append({k:c[k]for k in ['trial','native_prefill_seconds','native_decode_seconds','native_CAPI_prefill_input_tok_s','native_CAPI_decode_output_tok_s','driver_decode_output_tok_s','CPU_validation_seconds_outside_CAPI']})
for epoch in ['before','after_restore']:
 assert q['finite_'+epoch+'_passed'];cs=q['chat_cases_by_epoch'][epoch];assert len(cs)==6 and all(v['QA_quality_passed']and v['frozen816_full_IDs_exact']and v['stopped_at_eos']for v in cs)
trace=q['critical_path_trace'];assert not trace['active']and len(trace['frames'])==1 and trace['frames'][0]['step']==64
tp=r/'results'/f'deep-B32-profile-r7274-{tag}';files=[{'path':str(f),'bytes':f.stat().st_size}for f in tp.rglob('*')if f.is_file()];assert files and any(v['bytes']>0 for v in files)
rates=[v['native_CAPI_decode_output_tok_s']for v in trials[1:]]
out={'completed':True,'source_report_SHA256':sha(p),'normal8owners':True,'samegeom_regression_all3_exact':True,'fresh_QA12_exact_EOS':True,'historical_QA18_retained_reference':ref['QA18_source'],'summary':summ,'decode_repeat_spread_fraction':(max(rates)-min(rates))/(sum(rates)/2),'trace_files':files,'attribution':'UNASSIGNED pending actual kernel/stack analysis; no per-layer API','scope':'Trace-off with profiling runtime present. CAPI includes mandatory full-logits/IDs D2H. CPU fullvocab validation outside CAPI but inside driverwall. Frozen679 samegeometry regression, not independent originalmath; not uninstrumented peak, no causal codegain.','performance_adopted':False}
p=r/'results'/f'deep-B32-profile-r7274-{tag}-audit.json';assert not p.exists();p.write_text(json.dumps(out,indent=2));print(json.dumps(out))
