"""Read original controller evidence; export only numeric summaries and hashes."""
import pathlib,json,hashlib,datetime,statistics,math
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';label='deep-activation-model-r8848b'
wp=r/(label+'-watch.json');w=json.loads(wp.read_text());assert w['actual_handle']=='910374.865'
rp=r/'results'/f'{label}-910374.865.json';q=json.loads(rp.read_text()) if rp.exists() else {}
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Existing Deep865 CPU summary only; no IDs, raw timestamps, logs, activations or weights exported; no new audit/device probe','watch':{k:v for k,v in w.items() if k not in ['CPU_result_audit_stdout','CPU_result_audit_stderr','error_warning_lines']},'files':{}}
mp=r/'incoming/Deep-activation-Model-manifest-r8848b.json';m=json.loads(mp.read_text())
paths=[wp,rp,mp,r/'logs'/f'{label}-910374.865.log',r/(label+'-launch.json')]+[r/'incoming'/n for n in m['incoming_source_SHA256']]+[r/n for n in m['existing_remote_SHA256']]
for p in paths:
 if p.exists():
  raw=p.read_bytes();out['files'][str(p.relative_to(r))]={'SHA256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
def dist(a):
 a=sorted(a)
 def pct(p):
  z=(len(a)-1)*p;i=int(z);return a[i]+(z-i)*(a[min(i+1,len(a)-1)]-a[i])
 return {'n':len(a),'min':a[0],'median':statistics.median(a),'p95':pct(.95),'p99':pct(.99),'max':a[-1]} if a else None
keys=['completed','collection_finished','engine_worker_shutdown_completed','error','shutdown_error','loading_seconds','finite_before_passed','finite_diagnostic_B_passed','finite_candidate_pack_passed','finite_after_restore_passed','outcome','quality_passed','performance_qualified','performance_adopted','keys','library_sha256','scope','timing_scope','input_scope','performance_scope','qualification_scope','B_scope','B_QA_seconds','QA_ON_vs_OFF_fullseq_exact','reference_SHA256']
out['report']={k:q.get(k) for k in keys};out['trials']=[]
for c in q.get('pack_trials_r8848b',[]):
 t={k:v for k,v in c.items() if isinstance(v,(str,int,float,bool)) or v is None}
 t['request_metric_summary']={k:dist([v[k] for v in c.get('request_metrics',[])]) for k in ['TTFT_seconds','decode_tok_s','ITL_p95_seconds','max_gap_seconds']}
 t['model_counter_per_column_ranges']=[[min(z),max(z)] for z in zip(*c.get('model_counter_delta',[]))]
 t['pack_counter_per_column_ranges']=[[min(z),max(z)] for z in zip(*c.get('pack_counter_delta',[]))]
 stamps=c.get('request_completion_offsets_seconds',[])
 t['actual32requests_all128timestamps_finite_increasing']=len(stamps)==32 and all(len(a)==128 and all(math.isfinite(x) for x in a) and all(y>x for x,y in zip(a,a[1:])) for a in stamps)
 gaps=[[y-x for x,y in zip(a,a[1:])] for a in stamps]
 t['request_E2E_seconds']=dist([a[-1] for a in stamps])
 t['request_ITL_p99_seconds']=dist([dist(a)['p99'] for a in gaps])
 t['pooled_request_ITL_seconds']=dist([x for a in gaps for x in a])
 out['trials'].append(t)
out['QA']={k:{'cases':len(v),'all_QA_passed':all(c.get('QA_quality_passed') for c in v),'all_frozen_full_IDs_exact':all(c.get('frozen816_full_IDs_exact') for c in v),'all_stopped_at_eos':all(c.get('stopped_at_eos') for c in v)} for k,v in q.get('chat_cases_by_epoch',{}).items()}
out['affinity']={k:{'verified':q.get(k,{}).get('verified'),'width':q.get(k,{}).get('width'),'owner_devices':[x['device'] for x in q.get(k,{}).get('actual_owner_report',{}).get('owners',[])]} for k in ['owner_affinity_before_weight_load','owner_affinity_after_weight_load','owner_affinity']}
a=w.get('CPU_result_audit_stdout','')
if a:
 try:
  z=json.loads(a);out['original_audit']={k:v for k,v in z.items() if k not in ['trials','scope']}
 except ValueError:out['original_audit']={'parseable':False}
print(json.dumps(out))
