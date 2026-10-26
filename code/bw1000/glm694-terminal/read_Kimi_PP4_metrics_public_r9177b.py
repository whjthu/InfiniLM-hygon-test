import pathlib,json,subprocess,datetime,re,statistics,hashlib
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');run='pp4-r9154b-2b942746b681cfb9';hp=r/'results'/('Kimi-PP4-model-r9154b-'+run+'.json');h=json.loads(hp.read_bytes())if hp.exists()else{}
q={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'run':run,'runtime_handshake_values_excluded':True,'health':{k:h.get(k)for k in ['ready','completed','GPU_submitted','all_target_terminal','post_KFD_passed','manifest_SHA256','CPU_actual_step','heartbeat_UTC']},'owners':[],'groups':[]}
for g in h.get('groups',[]):q['groups'].append({k:g.get(k)for k in ['parent','nodes','offset','actual_handle','terminal']})
a=subprocess.check_output(['sacct','-S','2026-10-26','-j','910407,910375,910381,910371','-n','-P','-o','JobIDRaw,JobName%110,State%40,ExitCode,Elapsed,NodeList%120'],text=True,timeout=30);q['accounting']=[v.split('|')for v in a.splitlines()if run in v or 'Kimi-PP4-model-keeper-r9154b1'in v]
for i in range(8):
 p=r/'results'/('Kimi-modern-PP4-r9145b-'+run+'-node%d.json'%i);cp=r/'results'/('Kimi-PP4-capacity-r9145b-'+run+'-node%d.json'%i);o={'node':i}
 if p.exists()and i==6:
  raw_report=p.read_bytes();d=json.loads(raw_report);o.update({k:d.get(k)for k in ['phase','capacity_gate_passed','model_weight_keys_validated','weight_payload_opened','completed','preload_memory_gate','memory_after_create','memory_after_finish','all8_owned_workers_shutdown_completed']});o['has_error']=bool(d.get('error')or d.get('shutdown_error'));o['finished_cases']=[{'label':c.get('label'),'C':c.get('configured_C'),'W':c.get('configured_W'),'mode':c.get('mode')}for c in d.get('batch_cases',[])]
 if cp.exists():
  v=json.loads(cp.read_bytes());o['capacity_projection']={k:v.get(k)for k in ['phase','all64_min_free_bytes','required_free_bytes','model_weight_keys_validated','actual_engine_key_stage_range','actual_all93_coverage_validated']}
 if i//4<len(h.get('groups',[])):
  g=h['groups'][i//4];lp=r/'logs'/(g['name']+'-'+str(g.get('actual_handle'))+'-'+str(i%4)+'.log')
  if lp.exists():
   t=lp.read_text(errors='replace');o['network']={'actual_IBext_v8_selection_count':t.count('Using network IBext_v8'),'actual_Socket_selection_count':t.count('Using network Socket'),'normal_close_markers':t.count('KIMI_MODERN_BATCH_R9143_NORMAL'),'error_line_count':sum(bool(re.search(r'Traceback|HIP error|Segmentation fault',z))for z in t.splitlines()),'GDR_status':'UNKNOWN'}
 if i==6 and p.exists():
  o['source_report_SHA256']=hashlib.sha256(raw_report).hexdigest();o['case_metrics']=[]
  for c in d.get('batch_cases',[]):
   ms=c['request_metrics'];rs=[x['user_firstlast_tok_s']for x in ms if x.get('user_firstlast_tok_s')is not None];cycles=[z for m in ms for z in m.get('cycles',[])];proposed=sum(x['proposed']for x in cycles);accepted=sum(x['accepted_proposals']for x in cycles)
   o['case_metrics'].append({**{k:c.get(k)for k in ['label','strategy','configured_C','configured_width','I192_enabled','prefill_all_complete_s','decode_common_window_s','total_useful_decode_tokens','aggregate_decode_tok_s','all_serving_GPU_cards','aggregate_decode_tok_s_per_card']},'user_firstlast_min_median_max':[min(rs),statistics.median(rs),max(rs)]if rs else None,'naturalEOS':sum(bool(x['EOS'])for x in ms),'capped':sum(bool(x['truncated'])for x in ms),'correct_original_string_scorer':sum(bool(x.get('quality',{}).get('correct'))for x in ms),'sample_indices':[x.get('quality',{}).get('sample_index')for x in ms],'distinct_sample_count':len({x.get('quality',{}).get('sample_index')for x in ms}),'accepted_proposals':accepted,'proposed':proposed,'acceptance_rate':accepted/proposed if proposed else None,'draft_batch_RPC_total_s':sum(x.get('draft_RPC_group_s',0)for x in c['cycles']),'verify_CAPI_sum_s':sum(x.get('verify_CAPI_maxhost_s',0)for x in c['cycles']),'replay_CAPI_sum_s':sum(x.get('replay_CAPI_maxhost_s',0)for x in c['cycles']),'clock_scope':'native emitternode6 sameprocess monotonic; no HTTP; livepartialnotterminal'})
 q['owners'].append(o)
q['draft']={k:h.get('draft',{}).get(k)for k in ['actual_handle','terminal','submitted_UTC']};print(json.dumps(q))
