from pathlib import Path
import json,hashlib,statistics,datetime
r=Path(__file__).resolve().parent
p=r/'Deep-softmax-cast-Model-observed-root-b-r9002.json';d=json.loads(p.read_bytes());w=d['watch'];m=d['report']
assert w['terminal']['State']=='COMPLETED'and w['terminal']['ExitCode']=='0:0'
assert w['CPU_result_audit_returncode']==0 and w['qualified_terminal_result']
assert w['post_KFD']['passed']and w['post_KFD']['kfd_pids']==[]
assert m['completed']and m['quality_passed']and m['performance_qualified']and m['engine_worker_shutdown_completed']
ts=m['owner_trials_r9001'];assert [x['label']for x in ts]==['A0','C1','C2','A3']
rows=[]
for x in ts:
 assert x['complete']and x['full_IDs_EOS_exact']and len(x['generated_ids'])==32 and all(len(ids)==128 for ids in x['generated_ids'])
 assert x['scratch_counter_delta']==[[0,344,0,1032]]*8 and x['accepted_decode_fused_counter_delta']==[5461]*8
 q={k:x[k]for k in ['label','batch','micro','requested_outputs','complete','full_IDs_EOS_exact','packed_batch_arm','scratch_arm','native_prefill_seconds','native_prefill_input_tok_s','native_decode_seconds','native_decode_tok_s','native_decode_tok_s_per_card','driver_decode_seconds','driver_decode_tok_s','driver_wall_seconds','driver_E2E_tok_s','driver_E2E_tok_s_per_card','CPU_validation_seconds']}
 q['IDs_SHA256']=hashlib.sha256(json.dumps(x['generated_ids'],separators=(',',':')).encode()).hexdigest()
 q['userclock']={k:{'min':min(v[k]for v in x['request_metrics']),'mean':statistics.mean(v[k]for v in x['request_metrics']),'max':max(v[k]for v in x['request_metrics'])}for k in ['TTFT_seconds','decode_tok_s','ITL_p95_seconds','ITL_p99_seconds','max_gap_seconds']}
 q['scratch_counter_per_owner']=[0,344,0,1032];q['accepted_T32_decode_calls_per_owner']=5461
 q['packed_counter_per_owner']=x['packed_counter_delta']
 rows.append(q)
rates={}
for key in ['native_prefill_input_tok_s','native_decode_tok_s','driver_decode_tok_s','driver_E2E_tok_s']:
 a=statistics.mean(t[key]for t in [rows[0],rows[3]]);c=statistics.mean(t[key]for t in rows[1:3]);rates[key]={'A_mean':a,'C_mean':c,'C_over_A_percent':100*(c/a-1)}
pairs=[]
for c in rows[1:3]:
 for a in [rows[0],rows[3]]:pairs.append({'candidate':c['label'],'baseline':a['label'],'prefill_rate_gain_percent':100*(c['native_prefill_input_tok_s']/a['native_prefill_input_tok_s']-1),'prefill_latency_reduction_percent':100*(1-c['native_prefill_seconds']/a['native_prefill_seconds']),'driver_E2E_rate_gain_percent':100*(c['driver_E2E_tok_s']/a['driver_E2E_tok_s']-1),'native_decode_rate_change_percent_observed_not_kernel_gain':100*(c['native_decode_tok_s']/a['native_decode_tok_s']-1)})
qa={}
for epoch,cs in m['chat_cases_by_epoch'].items():
 assert len(cs)==6 and all(x['baseline_ids_exact']and x['stopped_at_eos']for x in cs)
 qa[epoch]={'cases':len(cs),'all_frozen_IDs_exact':True,'all_stopped_at_eos':True}
out={'hardware':'BW1000','level':'ACTUAL_MODEL','status':'COMPLETE_QUALITY_AND_PAIRED_PERFORMANCE_PASS','actual_handle':'910374.967','source_commit':'f236ca2efe3d1f9bb744102d900d907f2daa411f','DSO_SHA256':m['library_sha256'],'LM_source_commit':m['LM_source_commit'],'geometry':{'TP':8,'PP':1,'DCU':8,'B':32,'prefill_micro':4,'decode_micro':32,'input':512,'output':128},'loading_seconds':m['loading_seconds'],'trial_plan':'OneweightloadA/C/C/Afull128,modes0/2/2/0;Aaccepted923scratchON,CsamebutbatchonlySoftmaxCastwithoriginalperquery2DGemm;quantOFF/three scratchbackings/sink/orderedbanks/originaldrains andT32decode5461unchanged.','trials':rows,'balanced_rate_summary':rates,'each_C_vs_each_A':pairs,'A_endpoint_change_percent':{k:100*(rows[3][k]/rows[0][k]-1)for k in ['native_prefill_seconds','native_decode_seconds','driver_wall_seconds','native_prefill_input_tok_s','native_decode_tok_s','driver_E2E_tok_s']},'QA':qa,'QA_before_mode':m['QA_before_mode'],'QA_scope':m['QA_shortT_scope'],'minimum_runtime_boundary_free_bytes':m['minimum_runtime_boundary_free_bytes'],'terminal':w['terminal'],'original_audit_returncode':w['CPU_result_audit_returncode'],'unique_post':w['post_KFD'],'engine_worker_shutdown_completed':m['engine_worker_shutdown_completed'],'independent_per_owner8_normal_markers':'NOT_RETAINED;do notinventeightindividualmarkers fromenginejoin flag','shutdown_log_evidence':d['shutdown_log_evidence'],'performance_adopted':False,'new_decode_kernel_gain':False,'per_request_timing_scope':m['timing_scope'],'evidence':{'observation':p.name,'observation_SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'raw_report_SHA256':d['results_SHA256'],'raw_log_SHA256':d['logs_SHA256']},'limits':['Twofrozen512naturalpromptfamiliesrepeatedacross32rows,not32diverseonlineusers;CAPIreturnemitterclocknotHTTPclient.','Samejobpairedprefill/E2Egain,notnewabsolutepeakrecord. Historical923 Cmeanprefill138.202/decode66.535/driverE2E22.661 hasdifferenttime/native. CurrentCdecode~60.1 andE2E~22.48 doNOT exceedthatpriorabsolutedecode/E2E;donotpoolbestarms.','Cchangesprefillonly;observeddecodevariationnotdirectkernelgain. Physicalpeakunknown,singleuser15unmet.','Original955fullpackedfinaltokennegativeisnotrehabilitated;thiscandidatekeepsoriginal2Dmatrices.','Terminal0/audit0/engineworkerjoinflag/postemptyobserved;independenteightclosemarkersnotretained. No newGPUrunsolelyforlogging.']}
dest=r/'Deep967-softmax-cast-model-compact-r9003.json';assert not dest.exists();dest.write_bytes((json.dumps(out,ensure_ascii=False,indent=2)+'\n').encode())
print(json.dumps({'compact':dest.name,'SHA256':hashlib.sha256(dest.read_bytes()).hexdigest(),'balanced':rates,'QA':qa}))
