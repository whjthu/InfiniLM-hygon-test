import pathlib,json,subprocess,datetime,hashlib,time
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';run='kimi-relaxed-A-r9258b-04e27169';p=r/'results'/('Kimi-relaxed-A-r9258-'+run+'-node2.json')
out={'schema':1,'observedUTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'run':run,'whole_run_terminal_qualified':False}
for attempt in range(4):
 try:raw=p.read_bytes();q=json.loads(raw);break
 except(OSError,ValueError):
  if attempt==3:raise
  time.sleep(.3)
out.update(original_emitter_SHA256=hashlib.sha256(raw).hexdigest(),original_bytes=len(raw),phase=q.get('phase'),completed=q.get('completed'),error=q.get('error'),traceback=q.get('traceback'),loaded_keys=q.get('loaded_keys'),loading_elapsed_seconds=q.get('loading_elapsed_seconds'),loading_seconds=q.get('loading_seconds'),host_loader_phase_profile=q.get('host_loader_phase_profile'),after_finish_memory_gate_min=(q.get('after_finish_memory_gate')or{}).get('minimum_free_bytes'),special_provenance_SHA256=(q.get('relaxed_special_provenance')or{}).get('provenance_SHA256'),cases=[])
for c in q.get('batch_cases',[]):
 if not c['label'].startswith(('pilot8-','repeat4-','held95-')):continue
 z={k:c.get(k)for k in ['label','relaxed_gap_epsilon','configured_C','configured_width','all_serving_GPU_cards','total_useful_decode_tokens','decode_common_window_s','aggregate_decode_tok_s','aggregate_decode_tok_s_per_card','prefill_all_complete_s','admission_native_monotonic']}
 z['memory_minimum_free_bytes']=(c.get('memory_minimum_case')or{}).get('minimum_free_bytes')
 z['users']=[{k:u.get(k)for k in ['generated_tokens','useful_tokens','useful_decode_tokens','first_ready_monotonic','last_ready_monotonic','completion_monotonic','native_TTFT_s','user_firstlast_tok_s','decode_user_tok_s','ITL_P50_s','ITL_P95_s','ITL_P99_s','ITL_max_s','EOS','truncated','quality']}for u in c['request_metrics']]
 z['cycles']=len(c['cycles']);z['cost_s']={k:sum(x.get(k,0)for x in c['cycles'])for k in ['verify_CAPI_maxhost_s','draft_RPC_group_s','replay_CAPI_maxhost_s','cycle_local_wall_s']}
 z['gap_gather_HOST_s']=sum((x.get('relaxed_gap')or{}).get('gather_wall_s',0)for x in c['cycles'])
 z['verified_relaxed']=sum(x.get('verified_relaxed',0)for x in c['cycles']);z['emitted_relaxed']=sum(x.get('emitted_relaxed',0)for x in c['cycles'])
 z['verified_accepted']=sum(x['accepted_proposals']for u in c['request_metrics']for x in u['cycles']);z['proposed']=sum(x['proposed']for u in c['request_metrics']for x in u['cycles']);z['useful_per_cycle']=c['total_useful_decode_tokens']/z['cycles']if z['cycles']else None
 z['emitted_accepted']=sum(x.get('acceptance',{}).get('emitted_accepted_proposals',0)for u in c['request_metrics']for x in u['cycles'])
 z['fallback_nonfinite_negative']=sum(x.get('acceptance',{}).get('fallback',0)for u in c['request_metrics']for x in u['cycles'])
 z['actual_state_stats_before64']=c.get('state_stats_before64');z['actual_state_stats_after64']=c.get('state_stats_after64')
 out['cases'].append(z)
out['command_rejections']=q.get('command_rejections',[]);out['pending_budget']=q.get('pending_budget',[])
hp=r/'results/Kimi-relaxed-conditional-r9259-health.json'
if hp.exists():
 h=json.loads(hp.read_bytes());out['operator']={k:h.get(k)for k in ['actual_CPU_step','heartbeatUTC','stage','completed','normal_stop_published','stop_reason','error','pilot_economics','repeat_economics','heldout_economics','remaining_autoStop_s']}
out['sacct']=[x.split('|')for x in subprocess.check_output(['sacct','-S','2026-10-26','-j','910379.466,910371.580,910407.419,910407.420','-n','-P','-o','JobIDRaw,State,ExitCode,ElapsedRaw'],text=True,timeout=25).splitlines()if x.split('|')[0]in['910379.466','910371.580','910407.419','910407.420']]
print(json.dumps(out))
