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
   us.append({k:u.get(k)for k in ['logical','input_tokens','useful_tokens','useful_decode_tokens','first_ready_monotonic','last_ready_monotonic','native_TTFT_s','user_firstlast_tok_s','ITL_P50_s','ITL_P95_s','ITL_P99_s','EOS','truncated','completion_monotonic','quality']})
  out.append({k:c[k]for k in ['label','configured_C','strategy','adaptive_width','W2pair_enabled','all_serving_GPU_cards','total_useful_decode_tokens','decode_common_window_s','aggregate_decode_tok_s','admission_native_monotonic','end_native_monotonic','prefill_all_complete_s','W2_counts_delta8x2']}|{'users':us})
 return out

MODE='A';ORIG_SHA='ecbb82fe8eda573a2c07f7555a8234e0769deb5d5868833a979eb61ef10f3bcf';NORMAL_OWNERS=32
import pathlib,json,sys,hashlib
hp=pathlib.Path(sys.argv[1]);w=json.loads(hp.read_bytes());r=pathlib.Path.home()/'hygon-inference-refactor-20261014';run=w['run'];assert all(v['terminal'][2:4]==['COMPLETED','0:0']for v in w['groups'])and w['draft']['terminal'][2:4]==['COMPLETED','0:0']and w['post_KFD_passed'];original=[];emitter_labels=set();case_summaries=[];RSS_by_node=[];profile_owners=[];profile_records=[]
for node in range(4):
 p=r/'results'/('Kimi-agent-service-r144d-'+run+'-node%d.json'%node);raw=p.read_bytes();d=json.loads(raw);assert d['completed']and d['all8_owned_workers_shutdown_completed']and d['serving_protocol_complete']and not d.get('error')and not d.get('shutdown_error');assert d['manifest_SHA256']==w['manifest_SHA256'];assert all(c['all_serving_GPU_cards']==33 for c in d['batch_cases']);group=w['groups'][node//4];log=(r/'logs'/(group['name']+'-'+group['actual_handle']+'-'+str(node%4)+'.log')).read_text(errors='replace');assert log.count('KIMI_MODERN_BATCH_R9143_NORMAL')==1 and'Using network IBext_v8'in log and'Using network Socket'not in log;original.append({'node':node,'SHA256':hashlib.sha256(raw).hexdigest()});RSS_by_node.append({'node':node,'loaded_RSS_bytes':rss_bytes(),'bytes':len(raw)})
 profiles=d.get('sparse_model_profiles',[]);profile_records.append({'node':node,'records':profiles})
 for profile in profiles:
  if profile.get('armed'):
   rawp=profile['raw_decoder_owner_events'];assert rawp['all8_published']and len(rawp['owners'])==8;profile_owners.extend([{'node':node,'label':profile['label'],'device':int(k),'stage':v['PP_stage']}for k,v in rawp['owners'].items()])
 if node==2:emitter_labels={c['label']for c in d['batch_cases']};case_summaries.extend(compact_cases(d['batch_cases']));execution_meta={k:d.get(k)for k in ['screen_gate','resident_stop','pending_budget','command_rejections','frontend_C32_services','frontend_C32_incomplete']}
 del d,raw,log;gc.collect();RSS_by_node[-1]['after_release_RSS_bytes']=rss_bytes()
dr=json.loads((r/'results'/('Kimi-PP4-cache-draft-r9202-'+w['draft']['actual_handle']+'.json')).read_bytes());assert dr['status']=='COMPLETED'and dr['normal_close'];expected={c['id']for c in (json.loads(p.read_bytes())for p in(r/'incoming/Kimi-agent-service-Model-r144d/queue').glob('q*.json'))if c.get('action')=='case'};got=emitter_labels;audit_result={'passed':True,'lifecycle_qualified':True,'matrix12_complete':expected<=got,'cases':len(got),'expected_planned_ids':sorted(expected),'actual_case_ids':sorted(got),'original':original,'scope':'33card cap8 micro1 currentC8W8 strict cacheOFF GSM32 OFFONOFF sparseprofile; no tokenidentitygate; EOS/caps/quality separate','manifest_SHA256':w['manifest_SHA256']}
assert got<=expected
assert expected<=got or execution_meta.get('resident_stop')or execution_meta.get('pending_budget'),'unexecuted plannedcases require explicit reason ledger'
audit_result['unexecuted_planned_case_ids']=sorted(expected-got);audit_result['execution_metadata']=execution_meta
for post in w['post_KFD']+[w['draft_post_KFD']]:
 assert post['actual_terminal'][2:4]==['COMPLETED','0:0']and all(row['passed']and not row['kfd_pids']for row in post['rows'])
windows={label:sum(x['label']==label for x in profile_owners)for label in {x['label']for x in profile_owners}};assert all(v==32 for v in windows.values());audit_result['profile_coverage']={'actual_T64_windows_all32owners':windows,'target_windows':4,'complete':len(windows)>=4,'missing_UNKNOWN':sorted({f'r136b-g{i}-ON'for i in range(4)}-set(windows))};audit_result['profiles_by_node']=profile_records
profiles={p['label']:p for nodep in profile_records if nodep['node']==2 for p in nodep['records']};brackets=[]
for i in range(4):
 labels=[f'r136b-g{i}-'+a for a in ['OFFbefore','ON','OFFafter']]
 if not all(label in profiles for label in labels):brackets.append({'group':i,'coverage':'UNKNOWN: incomplete cohort triplet'});continue
 before,on,after=[profiles[x]for x in labels]
 if not on.get('armed'):brackets.append({'group':i,'coverage':'UNKNOWN: no eligible actualT64 window'});continue
 index=on['actual_forward_map'][0]['actual_verify_index_zero_based'];triplet=[]
 for profile in [before,on,after]:triplet.append(next((x for x in profile['verify_host_samples']if x['verify_index_zero_based']==index and x['T']==64 and x['C']==8),None))
 if not all(triplet):brackets.append({'group':i,'coverage':'UNKNOWN: OFF/ON actual geometry not matched','verify_index':index});continue
 baseline=(triplet[0]['CAPI_maxhost_s']+triplet[2]['CAPI_maxhost_s'])/2;ratio=triplet[1]['CAPI_maxhost_s']/baseline;brackets.append({'group':i,'coverage':'actualT64 C8 matched verify index','verify_index':index,'before_ON_after_CAPI_maxhost_s':[x['CAPI_maxhost_s']for x in triplet],'observer_perturbation_pct':100*(ratio-1),'within3pct':ratio<=1.03,'not_optimization_gain':True})
audit_result['observer_OFF_ON_OFF_brackets']=brackets;audit_result['attribution_qualified']=audit_result['profile_coverage']['complete']and all(x.get('within3pct',False)for x in brackets)

audit_result.update(schema=1,mode=MODE,actual_CPU_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],host=os.uname().nodename,old_original_checker_SHA256=ORIG_SHA,original_controller_audit=w.get('audit'),original_controller_step=w['CPU_actual_step'],original_lifecyclechecks_preserved=True,coveragegate_accounts_frozen_screen_or_budget_stop=True,readstrategy='same json parser/originalgates; read one node then del/gc; keep only scalar emitter case/user QA',new_performance_round=False,GPU=False,normal_target_owners=NORMAL_OWNERS,case_summaries=case_summaries,RSS_by_node=RSS_by_node,peak_CPU_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,post_KFD=w['post_KFD'],draft_post_KFD=w['draft_post_KFD'],finishedUTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
out=r/'results'/('Kimi-sparse-profile-audit-r136b-'+run+'.json');assert not out.exists();out.write_bytes((json.dumps(audit_result,indent=2)+'\n').encode());print(json.dumps({k:audit_result[k]for k in ['mode','passed','actual_CPU_step','normal_target_owners','peak_CPU_RSS_bytes']}))
