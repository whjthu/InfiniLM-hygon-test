"""Local statistics over existing original495 audit; no new native/GPU calls."""
import pathlib,json,statistics,hashlib,datetime
r=pathlib.Path(__file__).resolve().parent;p=r/'r5828-GLM495-terminal-metadata.json';d=json.loads(p.read_text());a=d['original_audit'];w=d['watch']
assert a['qualified']and w['qualified']and w['audit_returncode']==0 and w['postKFD_passed']
assert w['jobs'][0][0]=='910376.495'and w['jobs'][0][2:4]==['COMPLETED','0:0']
assert len(d['owners'])==16 and all(x['normal_closed']for x in d['owners'])
assert d['ordinary_profile_samples']==d['window_samples']==0
q=d['tail_quality'];assert all(q[k]for k in ['completed','before_select_frozen_IDs_exact_global','after_select_frozen_IDs_exact_global','shared_validation_final_OFF','resident_final_OFF_empty','worker_shutdown_completed','post_close_Torch_liveness','planned_phase_accounting_completed','resident_candidate_quality_passed'])and q['after_select_shared_flag']==1
perf=[x for x in d['phases']if x['phase'].startswith('perf-')];assert len(perf)==8 and [x['arm']for x in perf]==['O','N','N','O']*2
assert all(x['same_micro_reference_passed']and x['cross_geometry_full_IDs_passed']and x['batch']==16 and x['micro']==8 and x['input_lengths']==[512]*16 and x['max_output_tokens']==128 for x in perf)
metric=lambda c,k:c['tail_ready_metrics'][k]
mean=lambda arm,fun:statistics.mean(fun(c)for c in perf if c['arm']==arm)
fields={'decode_tokens_s':lambda c:metric(c,'aggregate_useful_decode_tokens_s'),'decode_tokens_s_per_card':lambda c:metric(c,'decode_tokens_s_per_card'),'prefill_input_tokens_s':lambda c:c['prefill_input_tokens_per_service_second'],'E2E_output_tokens_s':lambda c:metric(c,'including_prefill_useful_tokens_s'),'mean_user_decode_tokens_s':lambda c:metric(c,'mean_request_useful_decode_tokens_s'),'max_user_decode_tokens_s':lambda c:max(metric(c,'request_useful_decode_tokens_s')),'TTFT_mean_seconds':lambda c:statistics.mean(metric(c,'TTFT_seconds_per_request')),'TTFT_max_seconds':lambda c:max(metric(c,'TTFT_seconds_per_request')),'ITL_median_seconds':lambda c:c['ITL_seconds']['median'],'ITL_P95_seconds':lambda c:c['ITL_seconds']['P95'],'ITL_P99_seconds':lambda c:c['ITL_seconds']['P99'],'ITL_max_seconds':lambda c:c['ITL_seconds']['max']}
summary={}
for name,fun in fields.items():
 O,N=mean('O',fun),mean('N',fun);summary[name]={'O_mean':O,'N_mean':N,'relative_percent':100*(N/O-1)}
rounds=[]
for repeat in (1,2):
 cs=[x for x in perf if f'-r{repeat}-'in x['phase']];O0,N0,N1,O1=cs
 before,after=metric(O0,'aggregate_useful_decode_tokens_s'),metric(O1,'aggregate_useful_decode_tokens_s')
 gain=a['ABBA'][str(repeat)]['aggregate_useful_decode_tokens_s']['gain_percent']
 rounds.append({'round':repeat,'arms':[{'phase':c['phase'],'decode_tokens_s':metric(c,'aggregate_useful_decode_tokens_s')}for c in cs],'paired_mean':a['ABBA'][str(repeat)],'forward_pair_gain_percent':100*(metric(N0,'aggregate_useful_decode_tokens_s')/before-1),'reverse_pair_gain_percent':100*(metric(N1,'aggregate_useful_decode_tokens_s')/after-1),'O_endpoint_drift_percent_of_mean':100*(after-before)/statistics.mean([before,after])})
counter={}
for stage in (0,1):
 owners=[x for x in d['owner_phase_counts']if x['stage']==stage];assert len(owners)==8
 for owner in owners:assert [c['phase']for c in owner['phases']]==[c['phase']for c in d['phases']]
 phases=owners[0]['phases']
 for owner in owners[1:]:assert owner['phases']==phases
 firstO=phases[0];firstN=phases[1];L=38 if stage==0 else 40;shared=firstO['shared_validation_counter_delta'][1]//(16*128)
 assert firstO['shared_validation_counter_delta']==[0,shared*16*128]
 assert firstN['shared_validation_counter_delta']==[shared*16*127,shared*16]
 expected_bytes=shared*16*sum(range(513,640))*8;assert firstN['avoided_selection_D2H_bytes']==expected_bytes
 counter[str(stage)]={'TP_owners':8,'stage_layers':L,'shared_layers':shared,'full_layers':L-shared,'each_O_per_owner_validation_row_counts':firstO['shared_validation_counter_delta'],'each_N_per_owner_validation_row_counts':firstN['shared_validation_counter_delta'],'each_N_source_exact_avoided_decode_selection_D2H_bytes_per_owner':expected_bytes,'each_N_avoided_bytes_all8_owners':8*expected_bytes,'bytes_scope':'Logical validation-copy payload only, source-derived from actual proof-bound P_i and actual branch counts; not physical PCIe transaction measurement or exclusive copy time.'}
out={'status':'QUALIFIED_SMALL_GAIN_TESTED_OPT_IN','producer':'910376.495','run':d['run'],'scope':'Actual GLM-5.3 / BW1000 gfx936 / TP8PP2 /16cards /ROWS16/B16/M8/cap640;16unique true512token prompts,maxO128. VON/TOFF/resident38/40/crossrequestOFF fixed. New defaultOFF shared entry validation proof reuse is the sole variable. 8ordinary phases O/N/N/O x2, noGPUprofile; only tested workload scope, not globalmaximum/online/15tok/s claim.',
 'quality':{'all16_normal_close':True,'all8_I512_full128_final_IDs_EOS_ownO_and_historical472':True,'short4_mixed_prefix_historical8219_continuation_passed':len(d['phases'])==9 and d['phases'][-1]['same_micro_reference_passed']and d['phases'][-1]['cross_geometry_full_IDs_passed'],'beforeOFF_afterON_frozen_B1QA':True,'final_flag_and_Model_policies_OFF':True,'original_unique_audit_SHA256':w['audit_SHA256'],'raw_CPU_argmax':a['raw_CPUargmax'],'postKFD':'496 bothnodes empty; noGPUinit','network':'Bothnodes actualIBext_v8, no Socket datafallback','logs':'strict0/7090','native_missing_proof_ON_direct_negative':'NOT_RUN as a standalone GPUcase; unchanged original fallback plus CPU gate truth-table/source contract, actual ON prefill fallback and actual mixed-prefix short/QA qualified.','middle_float_bits':'Diagnostic only, noGPUF64.'},
 'summary_arithmetic_phase_means':summary,'rounds':rounds,
 'drift':{'within_round_O_endpoint_percent':[x['O_endpoint_drift_percent_of_mean']for x in rounds],'first_to_last_O_percent':100*(metric(perf[-1],'aggregate_useful_decode_tokens_s')/metric(perf[0],'aggregate_useful_decode_tokens_s')-1),'interpretation':'Both paired rounds and all four direction-matched pairs positive. Each round gain exceeds its own O endpoint drift. Global O declines ~2%; time-balanced O/N/N/O controls linear drift but does not establish a universal maximum or statistical confidence from only4N/4O phases.'},
 'counter_and_copy_budget':counter,'each_N_source_exact_avoided_bytes_all16cards':sum(v['each_N_avoided_bytes_all8_owners']for v in counter.values()),
 'latency_scope':d['timing_scope'],'latency_limitation':'~39s maximum initial intertoken gap remains behind the second prefilling cohort in this fixed scheduler; candidate does not improve prefill/TTFT materially. Maxuser remains about2.4tok/s, far below15.',
 'time':{'preparation_seconds':'UNKNOWN (not instrumented)','CPU_build':'910406.184 COMPLETED0/14s; compile/link/nm elapsed11.975s, noCPUdlopen/GPU','Model_launch_UTC':'2026-10-24T04:54:57.033388+00:00','queue_seconds':'UNKNOWN','Slurm_elapsed_seconds':42*60+48,'max_owner_weight_load_wall_seconds':max(x['load_wall_seconds']for x in d['owners']),'ordinary_8phase_wall_seconds_sum':sum(c['elapsed_seconds']for c in perf),'short4_wall_seconds':d['phases'][-1]['elapsed_seconds'],'QA_only_seconds':d['QA_only_seconds'],'cleanup_only_seconds':'UNKNOWN; originalpostKFD UTC05:38:36 confirmedempty','last_original_watch_UTC':w['utc'],'timing_caution':'Do not sum overlapped16owner loads. CAPI wall includes waits, not GPUbusy; ITL/TTFT are server-emitter, noHTTP queue.'},
 'source':{'published_commit':'6fcfce8','branch':'bw1000-experiments','private_DSO_SHA256':'1af0b95df9db1c68f9e8c00124dbb0b331cd1b3650afd4506e3befa30f98be9d','native_SHA256':'f35956f532833ce964bf766c8a61b44b56fe79042921d86710ccd075101705d2','actual_PrefixProof_header_SHA256':'4df185ee83bc7706e524d99ffc24ac151142f003334a700d6db1197b82b013da','source_manifest_SHA256':'f0a6d7d05235795e302bd1b34b25b91549440b9c5e8a3756d1420b999baf1779','default':'OFF; ON only afterreset/drain and all16owner unanimous boundary; code/default not deployed or changed by this report.'},
 'Astra41_decision':'Promote as a small tested opt-in gain in this qualified B16M8 I512 scope; retain defaultOFF/fallback and avoid further flag sweeps or another load to polish this2% route. No prefill improvement claim. Root decides subsequent higher-occupancy route separately.',
 'allowner_report_SHA256':a['allowner_report_SHA256'],'original_audit_original_post_only':True,'input_SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'per_phase':d['phases']}
dest=r/'GLM495-qualified-shared-validation-r5829.json';assert not dest.exists();dest.write_text(json.dumps(out,indent=2),encoding='utf8');print(dest.name,hashlib.sha256(dest.read_bytes()).hexdigest());print(json.dumps(summary,indent=2))

