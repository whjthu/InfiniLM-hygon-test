"""CPU read of original570 reports/raw/audit/post; never calls Model or Slurm."""
import pathlib,json,hashlib,statistics,math
import numpy as np
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
wp=r/'results/GLM-transient-Model-watch-r8872.json';w=json.loads(wp.read_bytes());assert w['state'].startswith('terminal_')and w['postKFD_passed'];run=w['run'];mp=r/'incoming/GLM-CROSSREQUEST-Model-source-manifest-r8872.json';m=json.loads(mp.read_bytes());gold=json.loads((r/'incoming/GLM477-frozen-gold-r8844.json').read_bytes())
def stats(a):return {'min':min(a),'mean':statistics.mean(a),'max':max(a)}if a else None
def pct(a,p):
 a=sorted(a);z=(len(a)-1)*p;i=int(z);return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(z-i)
owners=[];tails=[];failure=None;transport=[];normal=0;before_ok=True;A_ok=True;reject_ok=True;destroy_ok=True;raw_proofs=[]
for stage in(0,1):
 for rank in range(8):
  p=r/'results'/f'glm-transient-Model-r8872-{run}-stage{stage}-rank{rank}.json';q=json.loads(p.read_bytes());cases=q['multi_batch_cases'];assert len(cases)==1;A=cases[0];assert A['prompt_chunk_tokens']==512 and A['batch']==16 and A['micro']==8
  full=A['generated_ids']==gold['generated_ids'][:16]and A['stopped_at_EOS']==gold['stopped_at_EOS'][:16];A_ok&=full
  old=json.loads((r/'results'/f'glm-fp8-cached-model-r999-910374.89-stage{3 if stage else 0}-rank{rank}.json').read_bytes());before=bool(q['before_select_frozen_IDs_exact_global'])and [x['generated_ids']for x in q['before_select_chat_cases']]==[x['generated_ids']for x in old['before_select_chat_cases']];before_ok&=before
  rejected=q['resident_rejection_detail']=='chunk_first_token_gold_mismatch'and q['normal_numeric_rejection']and q['resident_candidate_rejected']and q['after_select_QA_scope']=='SKIPPED_REJECTED_CANDIDATE'and q['after_select_shared_flag']==0;reject_ok&=bool(rejected)
  f=q['first_token_failure'];assert f['phase']=='perf-I512-B16-M8-C128-N-r0'and f['rows']==list(range(8))and f['expected']==[gold['generated_ids'][i][0]for i in range(8)]
  if failure is None:failure=f
  else:assert failure==f
  n=bool(q['completed']and q['worker_shutdown_completed']and q['post_close_Torch_liveness']);normal+=n
  cache=q['cache_after_destroy'];destroy=cache['created']==cache['destroyed']and cache['active']==cache['active_bytes']==0 and q['resident_final_OFF_empty']and q['shared_validation_final_OFF'];destroy_ok&=destroy
  transport.append(q['continuous_transport_closed'])
  owners.append({'stage':stage,'rank':rank,'file':p.name,'SHA256':sha(p),'normal':n,'A128goldEOS':full,'beforeQA':before,'Ctoken0_reject':bool(rejected),'destroy_empty':bool(destroy),'load_s':q['actual_owner_weight_load_wall_seconds']})
  if rank==0:tails.append(q)
tail=tails[1];A=tail['multi_batch_cases'][0];t=A['tail_ready_metrics'];gaps=sum(t['ITL_seconds_per_request'],[])
for entry in tail['raw_checkpoints']:
 p=r/'results'/entry['file'];b=p.read_bytes();assert sha(p)==entry['sha256'];a=np.frombuffer(b,dtype='<u2').reshape(entry['shape']);x=(a.astype(np.uint32)<<16).view(np.float32);assert np.isfinite(x).all();actual=x.argmax(axis=1).tolist();assert actual==entry['GPU_argmax']
 if entry['phase']==A['phase']:assert actual==[A['generated_ids'][i][entry['step']]for i in entry['rows']]
 else:
  assert entry['phase']==failure['phase']and entry['step']==0 and entry['rows']==failure['rows']and actual==failure['actual']
  rows=[]
  for j,row in enumerate(entry['rows']):
   best=int(actual[j]);want=int(failure['expected'][j]);top=np.argpartition(x[j],-2)[-2:];top=top[np.argsort(x[j,top])[::-1]]
   rows.append({'row':row,'actual':best,'gold':want,'mismatch':best!=want,'actual_minus_gold_logit':float(x[j,best]-x[j,want]),'top2':top.tolist(),'top1_minus_top2':float(x[j,top[0]]-x[j,top[1]])})
  raw_proofs.append({'file':p.name,'SHA256':sha(p),'bytes':len(b),'allfinite':True,'CPUargmax_matches_saved_GPU':True,'rows':rows})
assert len(raw_proofs)==1
arm={'phase':A['phase'],'B':16,'M':8,'input_each':512,'output_each':128,'full128goldEOS':A_ok,'prefill_s':A['tail_prefill_CAPI_return_seconds'],'prefill_input_TPS':A['prefill_input_tokens_per_service_second'],'post_allprefill_decode_TPS':t['aggregate_useful_decode_tokens_s'],'post_allprefill_decode_TPS_per_card':t['decode_tokens_s_per_card'],'driver_E2E_s':A['elapsed_seconds'],'driver_E2E_useful_TPS':t['secondary_phase_E2E_tokens_s'],'emitter_E2E_useful_TPS':t['including_prefill_useful_tokens_s'],'request_firstlast_TPS':t['request_useful_decode_tokens_s'],'request_firstlast_TPS_summary':stats(t['request_useful_decode_tokens_s']),'request_TTFT_s':t['TTFT_seconds_per_request'],'TTFT_s_summary':stats(t['TTFT_seconds_per_request']),'ITL_P95_s':pct(gaps,.95),'ITL_P99_s':pct(gaps,.99),'maximum_gap_s':max(gaps),'request_maxgap_s':t['maximum_gap_seconds_per_request'],'early_decode_outputs':A['early_decode_outputs']}
phase_files=[]
for stage in(0,1):
 vals=[]
 for rank in range(8):
  p=r/'results'/f'GLM-Model-phase-r8872-{run}-stage{stage}-rank{rank}.jsonl';rows=[json.loads(x)for x in p.read_text().splitlines()];phase_files.append({'file':p.name,'SHA256':sha(p)})
  vals.extend(rows)
 if stage==0:phase_summary=[]
 phase_summary.append({'stage':stage,'saved_free_not_peak':{ph:stats([x['free']for x in vals if x['phase']==ph])for ph in sorted({x['phase']for x in vals})}})
out={'hardware':'BW1000','level':'wholeModel','status':'NUMERICAL_NEGATIVE_ORIGINAL_AUDIT_FAILED','qualified':False,'producer':w['producer'],'run':run,'Slurm':w['jobs'],'sourceclaim':'892566ca4cc82d65aefe5e98c666c99522f75aaf','manifest_SHA256':sha(mp),'entry_SHA256':sha(r/'incoming/dispatch_GLM_resident_exact_Model_r8872.py'),'DSO_SHA256':m['model_SHA256'],'scope':'single16DCU TP8PP2/B16M8;one load A512full128 thenC128chunk;constructor32/cap640/exactbanks/fixedmask[38,40];same477first16distinct512prompts;prefix/vectorON;PV/tile2/quantreuseOFF;6208MiBadmission/.5GiBfloor;45minhard2600inner;no retry or extension',
 'A512':arm,'C128':{'completed_full128':False,'first_token_failure':failure,'mismatching_rows':[r for r,a,b in zip(failure['rows'],failure['actual'],failure['expected'])if a!=b],'stopped_before_second_cohort_prefill':True,'actual_interleaved_decode_calls':0,'actual_early_decode_outputs':0,'planned_early_decode_outputs_if_valid':24,'throughput_latency_tradeoff':'UNQUALIFIED_NOT_MEASURED; no C throughput, TTFT,ITL,maxgap,E2E comparison can be adopted','failure_raw_CPU_checks':raw_proofs,'source_cause':'UNKNOWN; source-feasible chunk API is insufficient to prove final numerical equivalence'},
 'original_watch':{'file':wp.name,'SHA256':sha(wp),'state':w['state'],'qualified':w.get('qualified'),'audit_returncode':w['audit_returncode'],'audit_stderr_tail':w.get('audit_stderr_tail'),'postKFD':w['postKFD'],'postKFD_passed':True},
 'independent_CPU_evidence':{'all16_reports_read':True,'normal_owners':normal,'A_full128goldEOS_allowners':A_ok,'beforeQA_original_fullIDs_allowners':before_ok,'allowners_same_C_first_token_failure':True,'explicit_rejection_restoreQA_scope_allowners':reject_ok,'Runtime_cache_destroyed_empty_allowners':destroy_ok,'no_extra_QA_forward_or_post':True,'transport_closed_report_unique':list({json.dumps(x,sort_keys=True):x for x in transport}.values()),'original_audit_not_overridden':True,'meaning':'Numerical negative and ownedGPU cleanup evidenced; whole packet remains unqualified/originalaudit1. Candidate case was begun but not logically ended at rejection.'},
 'owner_reports':owners,'phase_free_not_peak':phase_summary,'phase_source_files':phase_files,'measurement_contract':'A puredecode is after lastfirsttoken common anchor. For future validC,24 early outputs must be retained in E2E/user clocks and excluded from post-allprefill numerator. This run failed before any early decode. Peruser=(nonEOSn-1)/(last-first),notaggregate/B.','prior_historical_reference':'A39.156 is operational baseline only; no new gain; currentcandidatefailed.','performance_adopted':False}
print(json.dumps(out))
