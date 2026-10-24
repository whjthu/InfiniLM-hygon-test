import pathlib,json,hashlib,subprocess,sys,math
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';tag=sys.argv[1];p=r/'results'/f'Deep-HC-RMS-Model-r7385-{tag}.json';q=json.loads(p.read_text());assert q['completed']and q['engine_worker_shutdown_completed']and q['collection_before_destroy_passed']and not q['performance_adopted'];assert q['library_sha256']=='a01b553eb53fe16d4eb7d48e7bf34f92c867387928f9a2c4f9ac3341c4b7d2c8'
rows=[v.split('|')for v in subprocess.check_output(['sacct','-S','2026-10-23','-j',tag,'-n','-P','-o','JobID,State,ExitCode'],text=True).splitlines()];assert rows==[[tag,'COMPLETED','0:0']]
manifest=json.loads((r/'incoming/Deep-HC-RMS-Model-GPU-manifest-r7385.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for n,h in manifest['sources_SHA256'].items():assert sha(r/'incoming'/n)==h
for k in ['owner_affinity_before_weight_load','owner_affinity_after_weight_load','owner_affinity']:assert q[k]['verified']and q[k]['width']==4
for epoch in ['before','after_restore']:
 assert q['finite_'+epoch+'_passed'];assert len(q['chat_cases_by_epoch'][epoch])==6 and all(c['stopped_at_eos']and c['QA_quality_passed']and c['frozen816_full_IDs_exact']for c in q['chat_cases_by_epoch'][epoch])
trials=q['HC_RMS_Model_trials'];ref=json.loads((r/'incoming/Deep-B32-samegeom-reference-r7269.json').read_text());summ=[]
for c in trials:
 windows=c['CAPI_windows'];decode=sum(w['step']>0 for w in windows);assert [v[1]for v in c['HC_RMS_counter_delta']]==[86*decode*int(bool(c['flag']))]*8
 assert len(windows)==c['fullvocab_checked_CAPI_windows']and all(w['seconds']>0 and math.isfinite(w['seconds'])for w in windows)
 if c['completed128']:
  assert c['generated_ids']==ref['generated_ids']and c['full4096_final_IDs_exact_reference']and c['EOS_pattern_matches_reference']and c['CPUargmax_all_windows']and len(windows)==135
  assert c['same_load_OFF_exact']in[None,True]
 summ.append({k:c[k]for k in ['trial','flag','completed128','native_prefill_seconds','native_decode_seconds','native_CAPI_prefill_input_tok_s','native_CAPI_decode_output_tok_s','CPU_validation_seconds_outside_CAPI','HC_RMS_counter_delta','first_quality_failure']})
rejected=bool(q.get('candidate_rejected'));ratio=spread=None
if rejected:
 assert q['first_quality_failure']['flag']==1 and not q['formal_Model_complete']and trials[-1]['first_quality_failure'];status='CANDIDATE_FINAL_TOKEN_REJECTED_NORMAL_RESTORE'
else:
 assert q['formal_Model_complete']and [v['trial']for v in trials]==['OFF0','ON0','ON1','OFF1']and q['finite_candidate_ON_QA_passed']
 cs=q['chat_cases_by_epoch']['candidate_ON_QA'];assert len(cs)==6 and all(c['QA_quality_passed']and c['frozen816_full_IDs_exact']and c['stopped_at_eos']for c in cs)
 vals=[c['native_decode_seconds']for c in trials];ratio=(vals[0]+vals[3])/(vals[1]+vals[2]);spread=max(abs(vals[0]-vals[3])/((vals[0]+vals[3])/2),abs(vals[1]-vals[2])/((vals[1]+vals[2])/2));status='FULL_MODEL_FINAL_IDS_EOS_QA_PASSED'
out={'completed':True,'status':status,'Slurm':rows,'source_report_SHA256':sha(p),'normal8owners':True,'candidate_rejected':rejected,'first_quality_failure':q.get('first_quality_failure'),'summary':summ,'ABBA_decode_ratio':ratio,'paired_repeat_spread_fraction':spread,'benefit_exceeds_repeat_spread':ratio is not None and ratio-1>spread,'QA_restore_exact':True,'Model_performance_adopted':False,'scope':'OrdinaryfullModel CAPI includes mandatorylogits/IDsD2H; CPUfullvocabvalidation outsideCAPI insidedriverwall. Samegeometry679reference plus sameLoadOFF. No profiling or HTTP/clientservice claims; noGPUF64.'}
op=r/'results'/f'Deep-HC-RMS-Model-r7385-{tag}-audit.json';assert not op.exists();op.write_text(json.dumps(out,indent=2));print(json.dumps(out))
