import gc,resource,os,time,datetime
def rss_bytes():
 try:return int(next(s.split()[1]for s in pathlib.Path('/proc/self/status').read_text().splitlines()if s.startswith('VmRSS:')))*1024
 except:return None
def compact_cases(cases):
 out=[]
 for c in cases:
  us=[]
  for u in c['request_metrics']:
   assert u['useful_decode_tokens']==max(0,u['useful_tokens']-1)
   assert u['EOS']or u['truncated']
   q=u['quality'];assert type(q['correct'])is bool
   if 'numeric_diagnostics'in q:assert q['correct']==q['numeric_diagnostics']['strict_correct']
   else:assert q['correct']==(q['extracted']==q['expected'])
   us.append({k:u.get(k)for k in ['logical','input_tokens','useful_tokens','useful_decode_tokens','first_ready_monotonic','last_ready_monotonic','native_TTFT_s','user_firstlast_tok_s','ITL_P50_s','ITL_P95_s','ITL_P99_s','EOS','truncated','completion_monotonic','quality','native_ready_events']})
  out.append({k:c[k]for k in ['label','configured_C','strategy','adaptive_width','W2pair_enabled','all_serving_GPU_cards','total_useful_decode_tokens','decode_common_window_s','aggregate_decode_tok_s','admission_native_monotonic','end_native_monotonic','prefill_all_complete_s','W2_counts_delta8x2']}|{'users':us,'phase_observer_summary':c.get('phase_observer_summary'),'prefill_interference':c.get('prefill_interference'),'observer_requested':c.get('observer_requested')})
 return out

MODE='A';ORIG_SHA='ecbb82fe8eda573a2c07f7555a8234e0769deb5d5868833a979eb61ef10f3bcf';NORMAL_OWNERS=32
import pathlib,json,sys,hashlib
hp=pathlib.Path(sys.argv[1]);w=json.loads(hp.read_bytes());r=pathlib.Path.home()/'hygon-inference-refactor-20261014';run=w['run'];assert all(v['terminal'][2:4]==['COMPLETED','0:0']for v in w['groups'])and w['draft']['terminal'][2:4]==['COMPLETED','0:0']and w['post_KFD_passed'];original=[];emitter_labels=set();case_summaries=[];RSS_by_node=[]
for node in range(4):
 p=r/'results'/('Kimi-ctx2k-r127-'+run+'-node%d.json'%node);raw=p.read_bytes();d=json.loads(raw);assert d['completed']and d['all8_owned_workers_shutdown_completed']and d['serving_protocol_complete']and not d.get('error')and not d.get('shutdown_error');assert d['manifest_SHA256']==w['manifest_SHA256'];assert all(c['all_serving_GPU_cards']==33 for c in d['batch_cases']);group=w['groups'][node//4];log=(r/'logs'/(group['name']+'-'+group['actual_handle']+'-'+str(node%4)+'.log')).read_text(errors='replace');assert log.count('KIMI_MODERN_BATCH_R9143_NORMAL')==1 and'Using network IBext_v8'in log and'Using network Socket'not in log;original.append({'node':node,'SHA256':hashlib.sha256(raw).hexdigest()});RSS_by_node.append({'node':node,'loaded_RSS_bytes':rss_bytes(),'bytes':len(raw)})
 if node==2:emitter_labels={c['label']for c in d['batch_cases']};case_summaries.extend(compact_cases(d['batch_cases']));execution_meta={k:d.get(k)for k in ['screen_gate','resident_stop','pending_budget','command_rejections','frontend_C32_services','frontend_C32_incomplete']}
 del d,raw,log;gc.collect();RSS_by_node[-1]['after_release_RSS_bytes']=rss_bytes()
dr=json.loads((r/'results'/('Kimi-PP4-cache-draft-r9202-'+w['draft']['actual_handle']+'.json')).read_bytes());assert dr['status']=='COMPLETED'and dr['normal_close'];expected={c['id']for c in (json.loads(p.read_bytes())for p in(r/'incoming/Kimi-ctx2k-Model-r127/queue').glob('q*.json'))if c.get('action')=='case'};got=emitter_labels;audit_result={'passed':True,'lifecycle_qualified':True,'initial18_complete':expected<=got,'cases':len(got),'expected_planned_ids':sorted(expected),'actual_case_ids':sorted(got),'original':original,'scope':'33card actual2k cap8 micro1 strictW8 C2observerOFFON + MMLU32 W2OFFON; no C16/C32; tokenidentitydiagnostic only; EOS/caps/quality separate','manifest_SHA256':w['manifest_SHA256']}
assert got<=expected
assert expected<=got or execution_meta.get('resident_stop')or execution_meta.get('pending_budget'),'unexecuted plannedcases require explicit reason ledger'
audit_result['unexecuted_planned_case_ids']=sorted(expected-got);audit_result['execution_metadata']=execution_meta
for post in w['post_KFD']+[w['draft_post_KFD']]:
 assert post['actual_terminal'][2:4]==['COMPLETED','0:0']and all(row['passed']and not row['kfd_pids']for row in post['rows'])
audit_result.update(schema=1,mode=MODE,actual_CPU_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],host=os.uname().nodename,old_original_checker_SHA256=ORIG_SHA,original_controller_audit=w.get('audit'),original_controller_step=w['CPU_actual_step'],original_lifecyclechecks_preserved=True,coveragegate_accounts_frozen_screen_or_budget_stop=True,readstrategy='same json parser/originalgates; read one node then del/gc; keep only scalar emitter case/user QA',new_performance_round=False,GPU=False,normal_target_owners=NORMAL_OWNERS,case_summaries=case_summaries,RSS_by_node=RSS_by_node,peak_CPU_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,post_KFD=w['post_KFD'],draft_post_KFD=w['draft_post_KFD'],finishedUTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
out=r/'results'/('Kimi-ctx2k-sequential-audit-r127-'+run+'.json');assert not out.exists();out.write_bytes((json.dumps(audit_result,indent=2)+'\n').encode());print(json.dumps({k:audit_result[k]for k in ['mode','passed','actual_CPU_step','normal_target_owners','peak_CPU_RSS_bytes']}))
