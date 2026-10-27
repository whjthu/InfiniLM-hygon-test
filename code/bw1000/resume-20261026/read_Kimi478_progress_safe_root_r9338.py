import pathlib,json,subprocess,datetime
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');run=json.loads((r/'incoming/Kimi-ctx2k-Model-r127/launcher/run-context-public.json').read_bytes())['run'];h=r/'results'/('Kimi-ctx2k-phase-A-model-r127-'+run+'.json');w=json.loads(h.read_bytes());out={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'run':run,'watch':{k:w.get(k)for k in ['ready','completed','passed','CPU_actual_step','heartbeat_UTC','post_KFD_passed']},'error_present':bool(w.get('error')),'cases':[]}
for node in [2]:
 p=r/'results'/('Kimi-ctx2k-r127-'+run+'-node'+str(node)+'.json');out['emitter_exists']=p.exists()
 if not p.exists():continue
 d=json.loads(p.read_bytes());out['emitter']={k:d.get(k)for k in ['phase','completed','all8_owned_workers_shutdown_completed','initial18_complete','screen_gate','resident_stop','pending_budget','command_rejections']};out['emitter']['error_present']=bool(d.get('error')or d.get('shutdown_error'))
 for c in d.get('batch_cases',[]):
  out['cases'].append({k:c.get(k)for k in ['label','configured_C','configured_width','observer_requested','W2pair_enabled','decode_common_window_s','total_useful_decode_tokens','aggregate_decode_tok_s','phase_observer_summary','phase_all_node_summaries','prefill_interference']})
  out['cases'][-1]['users']=[{'id':u['quality']['id'],'correct':u['quality']['correct'],'EOS':u['EOS'],'cap':u['truncated'],'N':u['useful_tokens'],'user_tok_s':u['user_firstlast_tok_s']}for u in c['request_metrics']]
rows=subprocess.check_output(['sacct','-S','2026-10-27','-j','910371.589,910379.478,910406.278','-n','-P','-o','JobID,JobName%110,State,ExitCode,Elapsed'],text=True,timeout=30).splitlines();out['Slurm']=[s.split('|')for s in rows if s.split('|')[0]in ['910371.589','910379.478','910406.278']]
out['scope']='LIVE_PARTIAL_ONLY_NOT_TERMINAL_QUALITY_OR_PERFORMANCE_QUALIFICATION; no nonce/argv/rawtoken export'
print(json.dumps(out))
