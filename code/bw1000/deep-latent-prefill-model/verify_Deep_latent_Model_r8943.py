"""Pure CPU current Model terminal/result audit, no runtime/torch import."""
import pathlib,json,hashlib,sys,math
p=pathlib.Path(sys.argv[1]);q=json.loads(p.read_text());r=p.parents[1];m=json.loads((r/'incoming/Deep-latent-Model-manifest-r8943.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for n,h in m['incoming_source_SHA256'].items():assert sha(r/'incoming'/n)==h,n
assert q['completed']and q['engine_worker_shutdown_completed']and not q['performance_adopted'];assert q['library_sha256']==m['DSO_SHA256'];assert q['finite_before_passed']and q['finite_after_restore_passed']
ref=json.loads((r/'incoming/Deep-B32-samegeom-reference-r7269.json').read_text());trials=q['owner_trials_r8943'];assert trials and trials[0]['model_arm']==0
for c in trials:
 if not c['complete']:continue
 assert c['batch']==32 and c['micro']==32 and c['requested_outputs']==128 and c['generated_ids']==ref['generated_ids']and c['full_IDs_EOS_exact']and c['frozen814_and_fresh_A_full_IDs_exact'];expected=[(0,a,4)for a in range(0,32,4)]+[(s,0,32)for s in range(1,128)]
 assert [(x['step'],x['chunk'],x['rows'])for x in c['CAPI_windows']]==expected
 pre=sum(x['seconds']for x in c['CAPI_windows']if x['step']==0);dec=sum(x['seconds']for x in c['CAPI_windows']if x['step']>0);assert math.isclose(pre,c['native_prefill_seconds'],rel_tol=1e-12)and math.isclose(dec,c['native_decode_seconds'],rel_tol=1e-12)
 assert math.isclose(4064/dec,c['native_decode_tok_s'],rel_tol=1e-12)and math.isclose(16384/pre,c['native_prefill_input_tok_s'],rel_tol=1e-12)and math.isclose(4096/c['driver_wall_seconds'],c['driver_E2E_tok_s'],rel_tol=1e-12)
 for ts,v in zip(c['request_completion_offsets_seconds'],c['request_metrics']):
  assert len(ts)==128 and all(math.isfinite(x)for x in ts);gaps=[b-a for a,b in zip(ts,ts[1:])];assert min(gaps)>0 and ts[0]==v['TTFT_seconds'];assert math.isclose(127/(ts[-1]-ts[0]),v['decode_tok_s'],rel_tol=1e-12)
  z=sorted(gaps);t=.95*126;a=int(t);p95=z[a]+(t-a)*(z[min(a+1,126)]-z[a]);assert math.isclose(p95,v['ITL_p95_seconds'],rel_tol=1e-12)and max(gaps)==v['max_gap_seconds'];t=.99*126;a=int(t);p99=z[a]+(t-a)*(z[min(a+1,126)]-z[a]);assert math.isclose(p99,v['ITL_p99_seconds'],rel_tol=1e-12)
 flag=c['pack_mode'];assert len(c['pack_counter_delta'])==8
 for d in c['pack_counter_delta']:
  assert d[flag]==344 and d[1-flag]==0 and 0<d[2+flag]<=344*256 and d[3-flag]==0
  assert d[4]==2*d[2]and d[5]==2*d[1]and d[6]==2*d[2]and d[7]==d[1]and d[8+flag]==67633152 and d[9-flag]==0 and d[10]==5461 and d[11]==0
for c in trials:
 if c['complete']:
  assert c['pack_mode']==1 and len(c['model_counter_delta'])==8
  for v,pc in zip(c['model_counter_delta'],c['pack_counter_delta']):
   assert c['model_arm']==0 and v[0]==344 and v[1]==0 and v[2]==pc[3]and v[3:]==[0]*5
assert q['performance_qualified']and q['quality_passed']and len(trials)in[3,4] and all(c['complete']for c in trials)
assert [c['latent_arm']for c in trials]==[0,1,1,0][:len(trials)]and[c['label']for c in trials]==['A0','C1','C2','A3'][:len(trials)]
assert q['optional_final_A_budget']['threshold_seconds']==400 and q['optional_final_A_budget']['status']==('COMPLETED'if len(trials)==4 else'SKIPPED_BUDGET')
assert q['owner_event_windows']==[]and not q['owner_event_capture_enabled']
for c in trials:
 arm=c['latent_arm'];assert c['latent_counter_delta']==[[344,704512,22016]if arm else[0,0,0]]*8
 assert [v['chunk']for v in c['latent_prefill_chunk_checks']]==list(range(0,32,4))
 for v in c['latent_prefill_chunk_checks']:assert v['total_tokens']==2048 and v['counter_delta']==[[43,88064,2752]if arm else[0,0,0]]*8
assert q['minimum_runtime_boundary_free_bytes']>=512*1024**2 and all(v[0]>8*1024**3 for v in q['GPU_memory_after_create']) and all(v[0]>=6*1024**3 for s in q['device_memory_snapshots'] for v in s['free_total_bytes'])
for key in ['owner_affinity_before_weight_load','owner_affinity_after_weight_load','owner_affinity']:
 a=q[key];assert a['verified']and a['width']==4;owners=a['actual_owner_report']['owners'];assert sorted(x['device']for x in owners)==list(range(8));masks=[int(x)for v in owners for x in v['verified_mask'].split(',')];assert len(masks)==len(set(masks))==32
print(json.dumps({'passed':True,'actual_handle':q['step'],'normal_owners':8,'quality_passed':q['quality_passed'],'performance_qualified':q['performance_qualified'],'adopted':False,'source_report_SHA256':sha(p),'trials':[{k:v for k,v in c.items()if k not in ['generated_ids','CAPI_windows','request_completion_offsets_seconds','request_metrics','per_step_completion_seconds']}for c in trials],'scope':q['scope']}))
