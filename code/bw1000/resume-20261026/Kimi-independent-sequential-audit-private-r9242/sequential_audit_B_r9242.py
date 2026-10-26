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
   q=u['quality'];assert q['correct']==(q['extracted']==q['expected'])
   us.append({k:u.get(k)for k in ['logical','input_tokens','useful_tokens','useful_decode_tokens','first_ready_monotonic','last_ready_monotonic','native_TTFT_s','user_firstlast_tok_s','ITL_P50_s','ITL_P95_s','ITL_P99_s','EOS','truncated','quality']})
  out.append({k:c[k]for k in ['label','configured_C','strategy','adaptive_width','all_serving_GPU_cards','total_useful_decode_tokens','decode_common_window_s','aggregate_decode_tok_s']}|{'users':us})
 return out

MODE='B';ORIG_SHA='db62f72ce86f0a8135f2f80375b8ae5228e088c40f7e0907d7f529b3a4066d05';NORMAL_OWNERS=64
import pathlib,json,sys,hashlib
hp=pathlib.Path(sys.argv[1]);w=json.loads(hp.read_bytes());r=pathlib.Path.home()/'hygon-inference-refactor-20261014';run=w['run'];assert all(v['terminal'][2:4]==['COMPLETED','0:0']for v in w['groups'])and w['draft']['terminal'][2:4]==['COMPLETED','0:0']and w['post_KFD_passed'];original=[];emitter_labels=set();case_summaries=[];RSS_by_node=[]
for node in range(8):
 p=r/'results'/('Kimi-adaptive-B-r9225-'+run+'-node%d.json'%node);raw=p.read_bytes();d=json.loads(raw);assert d['completed']and d['all8_owned_workers_shutdown_completed']and d['serving_protocol_complete']and not d.get('error')and not d.get('shutdown_error');assert d['manifest_SHA256']==w['manifest_SHA256'];assert all(c['all_serving_GPU_cards']==65 for c in d['batch_cases']);group=w['groups'][node//4];log=(r/'logs'/(group['name']+'-'+group['actual_handle']+'-'+str(node%4)+'.log')).read_text(errors='replace');assert log.count('KIMI_MODERN_BATCH_R9143_NORMAL')==1 and'Using network IBext_v8'in log and'Using network Socket'not in log;original.append({'node':node,'SHA256':hashlib.sha256(raw).hexdigest()});RSS_by_node.append({'node':node,'loaded_RSS_bytes':rss_bytes(),'bytes':len(raw)})
 if node==6:emitter_labels={c['label']for c in d['batch_cases']};case_summaries.extend(compact_cases(d['batch_cases']))
 del d,raw,log;gc.collect();RSS_by_node[-1]['after_release_RSS_bytes']=rss_bytes()
dr=json.loads((r/'results'/('Kimi-PP4-cache-draft-r9202-'+w['draft']['actual_handle']+'.json')).read_bytes());assert dr['status']=='COMPLETED'and dr['normal_close'];expected={json.loads(p.read_bytes())['id']for p in(r/'incoming/Kimi-adaptive-Model-AB-r9225/runtime-B/queue').glob('q*.json')};got=emitter_labels;audit_result={'passed':True,'lifecycle_qualified':True,'initial9_complete':expected<=got,'cases':len(got),'expected_initial_ids':sorted(expected),'actual_case_ids':sorted(got),'original':original,'scope':'65card same resident AR/fixed/adaptive, later heldout boundedJSON mayextendmatrix; no tokenidentitygate; EOS/caps/quality separate','manifest_SHA256':w['manifest_SHA256']}
assert expected<=got
for post in w['post_KFD']+[w['draft_post_KFD']]:
 assert post['actual_terminal'][2:4]==['COMPLETED','0:0']and all(row['passed']and not row['kfd_pids']for row in post['rows'])
audit_result.update(schema=1,mode=MODE,actual_CPU_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],host=os.uname().nodename,old_original_checker_SHA256=ORIG_SHA,original_controller_audit=w.get('audit'),original_controller_step=w['CPU_actual_step'],source_checks_unchanged=True,readstrategy='same json parser/originalgates; read one node then del/gc; keep only scalar emitter case/user QA',new_performance_round=False,GPU=False,normal_target_owners=NORMAL_OWNERS,case_summaries=case_summaries,RSS_by_node=RSS_by_node,peak_CPU_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,post_KFD=w['post_KFD'],draft_post_KFD=w['draft_post_KFD'],finishedUTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
out=r/'results'/('Kimi-'+MODE+'-independent-sequential-audit-r9242.json');assert not out.exists();out.write_bytes((json.dumps(audit_result,indent=2)+'\n').encode());print(json.dumps({k:audit_result[k]for k in ['mode','passed','actual_CPU_step','normal_target_owners','peak_CPU_RSS_bytes']}))
