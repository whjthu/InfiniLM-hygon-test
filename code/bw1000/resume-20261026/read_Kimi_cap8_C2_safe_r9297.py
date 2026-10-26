import pathlib,json,hashlib,subprocess,datetime,statistics
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';run='kimi-cap8-micro1-r9293-5efe17cbd6';p=r/'results'/('Kimi-cap8-micro1-r9293-'+run+'-node2.json');out={'schema':1,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'run':run,'same_handles_only':True,'sacct':[]}
for handle in ['910379.474','910371.586','910406.277']:
 raw=subprocess.check_output(['sacct','-S','2026-10-25','-j',handle,'-n','-P','-o','JobIDRaw,State,ExitCode,Elapsed'],text=True,timeout=25).splitlines();out['sacct'].extend(x.split('|')for x in raw if x.split('|')[0]==handle)
if p.exists():
 raw=p.read_bytes();d=json.loads(raw);out['report_SHA256']=hashlib.sha256(raw).hexdigest();out['report_bytes']=len(raw);out['report_mtime']=p.stat().st_mtime;out['status']={k:d.get(k)for k in ['node','phase','completed','error','shutdown_error','all8_owned_workers_shutdown_completed','serving_protocol_complete','manifest_SHA256','loaded_keys','loading_elapsed_seconds','create_seconds','loaded_keys_count','weight_payload_opened','screen_gate','pending_budget','resident_stop','command_rejections']}
 def compact_guard(g):
  if not isinstance(g,dict):return None
  return{k:g.get(k)for k in ['label','minimum_free_bytes','required_free_bytes']}|{'node_min_free_bytes':[min(x[0]for x in a['memory'])for a in g.get('owners',[])],'node_latest_min_free_bytes':[min(x[0]for x in a['latest_memory'])for a in g.get('owners',[])if a.get('latest_memory')],'sample_count':[a.get('sample_count')for a in g.get('owners',[])]}
 out['memory']={'afterfinish_min_free_bytes':min(x[0]for x in d.get('memory_after_finish',[]))if d.get('memory_after_finish')else None,'afterfinish_all32_gate':compact_guard(d.get('after_finish_memory_gate')),'last_guard':compact_guard(d.get('last_memory_guard_attempt')),'failed_guard':compact_guard(d.get('failed_memory_guard'))};out['cases']=[]
 for c in d.get('batch_cases',[]):
  v={k:c.get(k)for k in ['label','configured_C','configured_width','W2pair_enabled','adaptive_width','total_useful_decode_tokens','decode_common_window_s','aggregate_decode_tok_s','aggregate_decode_tok_s_per_card','all_serving_GPU_cards','prefill_all_complete_s','admission_native_monotonic','end_native_monotonic','W2_counts_delta8x2']};v['category']=c.get('command',{}).get('category');v['memory_minimum_case']=compact_guard(c.get('memory_minimum_case'));v['users']=[]
  for u in c['request_metrics']:
   q=u['quality'];v['users'].append({k:u.get(k)for k in ['logical','input_tokens','useful_tokens','useful_decode_tokens','first_ready_monotonic','last_ready_monotonic','completion_monotonic','user_firstlast_tok_s','native_TTFT_s','ITL_P50_s','ITL_P95_s','ITL_P99_s','EOS','truncated']}|{'quality':{k:q.get(k)for k in ['id','sample_index','task','domain','correct','extracted','expected']}})
  cy=c.get('cycles',[]);v['cycles_count']=len(cy);v['cycle_cost_totals']={k:sum(x.get(k,0)for x in cy)for k in ['verify_CAPI_maxhost_s','replay_CAPI_maxhost_s','draft_RPC_group_s','draft_RPC_sum_s','draft_query_sum_s','commit_local_s','cycle_local_wall_s']};v['prefill_calls']=[{k:x.get(k)for k in ['kind','logical','width','CAPI_maxhost_s','feature_harvest_local_s']}for x in c.get('prefill_calls',[])];out['cases'].append(v)
 out['case_count']=len(out['cases']);pp=p.with_suffix('.progress.json');out['progress']=json.loads(pp.read_bytes())if pp.exists()else None
hp=r/'results'/('Kimi-cap8micro1-A-model-r9293-'+run+'.json')
if hp.exists():
 h=json.loads(hp.read_bytes());out['keeper']={k:h.get(k)for k in ['CPU_actual_step','heartbeat_UTC','ready','completed','passed','error','capacity_passed_before_draft','post_KFD_passed']};out['draft_status']={k:h.get('draft',{}).get(k)for k in ['actual_handle','status','ready_seen','terminal']};out['lifecycle_scope']='same-run existing outputs only; no nonce/env/args/logtails exported'
print(json.dumps(out))
