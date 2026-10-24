"""Consume original495 terminal audit only; do not rerun audit or GPU work."""
import pathlib,json,hashlib,statistics,datetime
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');run='shared-5819-df76dacb54d8482aaaea'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
w=json.loads((r/'results/GLM-transient-Model-watch-r5819.json').read_text());assert w['run']==run and w['bound_actual_handle']=='910376.495'
assert w['state'] in ('terminal_audited','terminal_requires_review'),w['state']
out={'run':run,'step':'910376.495','read_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'watch':w}
if w.get('audit_returncode')==0:
 ap=r/'results/GLM-transient-Model-terminal-audit-r5819.json';assert sha(ap)==w['audit_SHA256'];a=json.loads(ap.read_text())
 out['original_audit']={k:a.get(k)for k in ['producer','qualified','Slurm','ABBA','preengine_records','allowner_report_SHA256','raw_CPUargmax','network_logs','candidate_rejected','selected_counts','scope']}
 out['owners']=[{k:v.get(k)for k in ['stage','rank','normal_closed','pinned_host_bytes','load_wall_seconds']}for v in a['owners']]
 out['owner_phase_counts']=[{'stage':v['stage'],'rank':v['rank'],'phases':[{k:c.get(k)for k in ['phase','arm','shared_validation_enabled','shared_validation_counter_delta','avoided_selection_D2H_bytes']}for c in v['phases']]}for v in a['owners']]
 ep=r/'results'/f'glm-transient-Model-r5819-{run}-stage1-rank0.json';assert sha(ep)==a['allowner_report_SHA256'][ep.name];q=json.loads(ep.read_text())
 payload=json.loads((r/'control'/f'glm-transient-Model-r5819-control-{run}-ids.json').read_text());eos=set(payload['chat_eos_token_ids'])
 def quantile(v,p):
  v=sorted(v);x=(len(v)-1)*p;i=int(x);return v[i]+(v[min(i+1,len(v)-1)]-v[i])*(x-i)
 def stats(v):return {'count':len(v),'min':min(v),'mean':statistics.mean(v),'median':statistics.median(v),'P95':quantile(v,.95),'P99':quantile(v,.99),'max':max(v)} if v else None
 out['tail_quality']={k:q.get(k)for k in ['completed','before_select_frozen_IDs_exact_global','after_select_frozen_IDs_exact_global','after_select_shared_flag','shared_validation_final_OFF','resident_final_OFF_empty','worker_shutdown_completed','post_close_Torch_liveness','planned_phase_accounting_completed','resident_candidate_quality_passed']}
 out['phases']=[]
 for c in q['multi_batch_cases']:
  times=c['tail_ready_times'];ids=c['generated_ids'];itl=[t[j]-t[j-1]for t,g in zip(times,ids)for j in range(1,len(t))if g[j]not in eos];initial=[t[1]-t[0]for t in times if len(t)>1]
  v={k:c.get(k)for k in ['phase','arm','reference_kind','batch','micro','input_lengths','max_output_tokens','effective_input_tokens','same_micro_reference_passed','cross_geometry_full_IDs_passed','stopped_at_EOS','elapsed_seconds','tail_ready_metrics','prefill_input_tokens_per_service_second','shared_validation_enabled','shared_validation_counter_delta','shared_validation_expected','shared_validation_source_exact_avoided_D2H_bytes']}
  v.update(ITL_seconds=stats(itl),initial_inter_token_gap_seconds=stats(initial),ordinary_no_profile=not c['phase'].startswith('prof-'),actual_forward_count=len(c['events']),first_native_start_realtime_ns=min(e['native_start_realtime_ns']for e in c['events']),last_native_end_realtime_ns=max(e['native_end_realtime_ns']for e in c['events']),owner_CAPI_wall_seconds_sum=sum(e['native_seconds']for e in c['events']),owner_CAPI_threadCPU_seconds_sum=sum(e['native_owner_thread_cpu_seconds']for e in c['events']))
  out['phases'].append(v)
 out['timing_scope']='ITL from retained emitter token ready timestamps, no HTTP or external queue measurement. First intertoken gap is separate to expose waiting behind the second prefill cohort. Native CAPI walls include dependencies/waits and are not GPUbusy; owner threadCPU overlaps wall. No new timing instrumentation.'
 out['ordinary_profile_samples']=len(q.get('native_pipeline_profile_samples_r8219',[]));out['window_samples']=len(q.get('selected_decode_window_r5819',[]))
 out['QA_only_seconds']='UNKNOWN: no exact before/after B1QA wall boundary retained; do not assign total residual to QA.'
print(json.dumps(out))
