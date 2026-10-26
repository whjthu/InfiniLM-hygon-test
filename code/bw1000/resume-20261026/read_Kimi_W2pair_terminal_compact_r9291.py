import pathlib,json,hashlib,subprocess,datetime,statistics
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';run='kimi-W2pair-A-r9282-4108c1be3d';report=r/'results'/('Kimi-W2pair-A-r9282-'+run+'-node2.json');raw=report.read_bytes();d=json.loads(raw);out={'schema':1,'run':run,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'report_SHA256':hashlib.sha256(raw).hexdigest(),'report_bytes':len(raw),'report_mtime':report.stat().st_mtime,'phase':d.get('phase'),'completed':d.get('completed'),'error':d.get('error'),'screen_gate':d.get('screen_gate'),'case_count':len(d.get('batch_cases',[])),'cases':[]}
for parent,h in [('910379','910379.470'),('910371','910371.583'),('910406','910406.276')]:
 a=subprocess.check_output(['sacct','-S','2026-10-25','-j',h,'-n','-P','-o','JobIDRaw,JobName%110,State,ExitCode,Elapsed'],text=True,timeout=25);out.setdefault('sacct',[]).extend(x.split('|')for x in a.splitlines()if x.split('|')[0]==h)
for c in d.get('batch_cases',[]):
 v={k:c.get(k)for k in ['label','configured_C','configured_width','W2pair_enabled','adaptive_width','total_useful_decode_tokens','decode_common_window_s','aggregate_decode_tok_s','aggregate_decode_tok_s_per_card','all_serving_GPU_cards','prefill_all_complete_s','admission_native_monotonic','end_native_monotonic','host_observer_decode_window','memory_minimum_case','W2_counts_delta8x2']};v['category']=c.get('command',{}).get('category');v['users']=[]
 for u in c['request_metrics']:
  q=u['quality'];v['users'].append({k:u.get(k)for k in ['logical','input_tokens','useful_tokens','useful_decode_tokens','first_ready_monotonic','last_ready_monotonic','completion_monotonic','user_firstlast_tok_s','native_TTFT_s','ITL_P50_s','ITL_P95_s','ITL_P99_s','EOS','truncated']}|{'quality':{k:q.get(k)for k in ['id','sample_index','task','domain','correct','extracted','expected']}})
 # Keep existing numeric cost metadata, no raw logits/features/generatedtext.
 cycles=c.get('cycles',[]);v['cycle_count']=len(cycles)if isinstance(cycles,list)else cycles
 if isinstance(cycles,list)and cycles:
  v['first_cycle_numeric_shape']={k:vv for k,vv in cycles[0].items()if isinstance(vv,(int,float,bool,str,type(None)))}
  v['cycle_numeric_totals']={key:sum(x.get(key,0)for x in cycles if isinstance(x.get(key),(int,float))and not isinstance(x.get(key),bool))for key in cycles[0]if isinstance(cycles[0][key],(int,float))and not isinstance(cycles[0][key],bool)}
 out['cases'].append(v)
pp=report.with_suffix('.progress.json');out['tiny_progress']=json.loads(pp.read_bytes())if pp.exists()else None
# Preserve lifecycle diagnostics separately from provisional case performance.
out['lifecycle']={k:d.get(k)for k in ['serving_protocol_complete','all8_owned_workers_shutdown_completed','shutdown_error','resident_stop','pending_budget','command_rejections']};
out['memory_timeline']={'after_finish':d.get('memory_after_finish'),'after_finish_gate':d.get('after_finish_memory_gate'),'cases':[{'label':c['label'],'before':c.get('memory_before_case'),'minimum':c.get('memory_minimum_case'),'prefill_calls':c.get('prefill_calls')}for c in d.get('batch_cases',[])][-12:],'failed':d.get('failed_memory_guard')}
out['allnode_lifecycle']=[]
for n in range(4):
 np=r/'results'/('Kimi-W2pair-A-r9282-'+run+'-node%d.json'%n)
 if n==2:dd=d;rr=raw
 else:rr=np.read_bytes();dd=json.loads(rr)
 out['allnode_lifecycle'].append({'node':n,'report_SHA256':hashlib.sha256(rr).hexdigest(),'bytes':len(rr),**{k:dd.get(k)for k in ['error','phase','completed','all8_owned_workers_shutdown_completed','shutdown_error','serving_protocol_complete']},'failed_guard_label':dd.get('failed_memory_guard',{}).get('label'),'case_count':len(dd.get('batch_cases',[]))})
 if n!=2:del dd,rr
source=r/'incoming/Kimi-speculative-batch-Model-r9122c/pair_probe.cpp';sr=source.read_bytes();lines=sr.decode().splitlines();out['actual_capacity_source']={'path':str(source),'SHA256':hashlib.sha256(sr).hexdigest(),'create_batch_lines':[{'line':i+1,'text':v}for i,v in enumerate(lines)if 'create_batch_r9122c'in v or'capacity8/16/32'in v or'current pool17/8logical'in v or'3632 fixed capacity paired state context'in v]}
print(json.dumps(out))
