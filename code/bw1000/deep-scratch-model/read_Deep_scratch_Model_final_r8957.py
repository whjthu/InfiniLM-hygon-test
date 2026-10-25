import pathlib,json,hashlib,datetime,re,statistics,math
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';label='deep-scratch-model-r8956';d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Existing Model originalwatch/lognumericphases/report numeric summaries+SHA only;no rawlogs/IDs/timestamps/weights/activations exported','source_packet_manifest_SHA256':'69af10b6adfa84a22892b1336da0f0a9ed2d3fdaf50913b3ff7dc36cde982680'}
for key in ['launch','watch']:
 p=r/(label+'-'+key+'.json')
 if p.exists():
  z=json.loads(p.read_text());d[key]={k:v for k,v in z.items() if k not in ['args','fresh_before_KFD','CPU_result_audit_stdout','CPU_result_audit_stderr','error_warning_lines']}
  if key=='watch' and z.get('CPU_result_audit_stdout'):
   try:
    a=json.loads(z['CPU_result_audit_stdout']);d['original_audit']={k:v for k,v in a.items() if k not in ['trials','scope']}
   except ValueError:d['original_audit_parseable']=False
w=d.get('watch',{});h=w.get('actual_handle')
def dist(xs):
 xs=sorted(xs)
 def p(f):
  z=f*(len(xs)-1);i=int(z);return xs[i]+(z-i)*(xs[min(i+1,len(xs)-1)]-xs[i])
 return {'n':len(xs),'min':xs[0],'mean':statistics.mean(xs),'median':statistics.median(xs),'p95':p(.95),'p99':p(.99),'max':xs[-1]} if xs else None
if h:
 p=r/'results'/f'{label}-{h}.json'
 if p.exists():
  raw=p.read_bytes();q=json.loads(raw);d['report_SHA256']=hashlib.sha256(raw).hexdigest();d['report_bytes']=len(raw);d['report']={k:q.get(k) for k in ['phase','phase_window','phase_monotonic_seconds','phase_history','selected_native_returncode','error_phase','shutdown_entered','completed','collection_finished','engine_worker_shutdown_completed','error','shutdown_error','loading_seconds','finite_before_passed','finite_candidate_pack_passed','finite_after_restore_passed','outcome','quality_passed','performance_qualified','performance_adopted','keys','library_sha256','scope','timing_scope','input_scope','minimum_runtime_boundary_free_bytes','optional_final_A_budget','owner_event_capture_enabled']};d['trials']=[]
  for c in q.get('owner_trials_r8956',[]):
   t={k:v for k,v in c.items() if isinstance(v,(str,int,float,bool)) or v is None};t['userclock']={k:dist([x[k] for x in c.get('request_metrics',[])]) for k in ['TTFT_seconds','decode_tok_s','ITL_p95_seconds','ITL_p99_seconds','max_gap_seconds']};stamps=c.get('request_completion_offsets_seconds',[]);t['userclock']['E2E_seconds']=dist([a[-1] for a in stamps]);t['actual32requests_all128timestamps_finite_increasing']=len(stamps)==32 and all(len(a)==128 and all(math.isfinite(x) for x in a) and all(y>x for x,y in zip(a,a[1:])) for a in stamps);t['scratch_counts_per_column_ranges']=[[min(x),max(x)] for x in zip(*c.get('scratch_counter_delta',[]))];t['scratch_prefill_chunks_checked']=len(c.get('scratch_prefill_chunk_checks',[]));t['quant1_counts_per_column_ranges']=[[min(x),max(x)] for x in zip(*c.get('model_counter_delta',[]))];t['pack_counts_per_column_ranges']=[[min(x),max(x)] for x in zip(*c.get('pack_counter_delta',[]))];d['trials'].append(t)
  d['QA']={k:{'cases':len(v),'all_QA_passed':all(c.get('QA_quality_passed') for c in v),'all_frozen_IDs_exact':all(c.get('frozen816_full_IDs_exact') for c in v),'all_stopped_at_eos':all(c.get('stopped_at_eos') for c in v)} for k,v in q.get('chat_cases_by_epoch',{}).items()}
  d['affinity']={k:{'verified':q.get(k,{}).get('verified'),'width':q.get(k,{}).get('width'),'owner_devices':[x['device'] for x in q.get(k,{}).get('actual_owner_report',{}).get('owners',[])]} for k in ['owner_affinity_before_weight_load','owner_affinity_after_weight_load','owner_affinity']}
 p=r/'logs'/f'{label}-{h}.log'
 if p.exists():
  raw=p.read_bytes();d['log_SHA256']=hashlib.sha256(raw).hexdigest();d['log_bytes']=len(raw);text=raw.decode(errors='replace');loads=re.findall(r'NATIVE_ENGINE_LOAD\s+(\d+)\s+(\d+)\s+([\d.]+)',text);d['last_numeric_load_phase']=[int(loads[-1][0]),int(loads[-1][1]),float(loads[-1][2])] if loads else None;d['error_types']=[v for v in ['Traceback','AssertionError','RuntimeError','KeyError','TimeoutExpired','Segmentation fault'] if v in text]

if 'q' in locals():
 d['actual_scratch_counter_receipts']=[]
 for t in q.get('owner_trials_r8956',[]):
  row={'label':t['label'],'scratch_arm':t.get('scratch_arm'),'complete':t['complete'],'prefill_chunk_count':len(t.get('scratch_prefill_chunk_checks',[])),'whole_counter_ranges':[[min(v),max(v)]for v in zip(*t.get('scratch_counter_delta',[]))],'decode_fused_counts':t.get('accepted_decode_fused_counter_delta'),'failure':t.get('failure')}
  chunks=t.get('scratch_prefill_chunk_checks',[])
  row['prefill_chunk_all8_counter_ranges']=[{'chunk':x['chunk'],'T':x['total_tokens'],'ranges':[[min(v),max(v)]for v in zip(*x['counter_delta'])]}for x in chunks]
  d['actual_scratch_counter_receipts'].append(row)
print(json.dumps(d))
