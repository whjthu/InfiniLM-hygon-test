"""Astra43 finite occupancy comparison. Same M32; B64 then B32, never M64."""
import ctypes, hashlib, json, time
import numpy as np

REF_SHA='ba577de61c056f6e701005f9b1622cfac284d38e1671bb480087187544f35797'

def run(g):
 r,lib,ctx,cfg,check,report,path=[g[k]for k in ['r','lib','ctx','cfg','check','report','path']]
 refpath=r/'incoming/Deep-B32-samegeom-reference-r7269.json'
 assert hashlib.sha256(refpath.read_bytes()).hexdigest()==REF_SHA
 ref=json.loads(refpath.read_text());assert ref['DSO_SHA256']==report['library_sha256']=='0e0512551483457af01944e9c9298442f499d959836a1c5b66dfa438a3c6c4ea'
 assert len(ref['input_ids'])==32 and all(len(x)==512 for x in ref['input_ids'])
 report.update(config_trials_r8650=[],outcome='RUNNING',performance_adopted=False,performance_qualified=False,
  reference_SHA256=REF_SHA,input_scope='Exactly two frozen I512 prompts duplicated across B32/B64; not independent32/64prompt traffic.',
  scope='Astra43 B64M32 versus B32M32; same0e051/precision5/projection16/C4=16/layout0HC0tail0/cache128x1024/prefillchunk4/TP8PP1. Different resident workloads, notsameworkloadcodegain.',
  negative_protocol='Only completedCAPI code0 + finite/CPUargmax + verified8owner/BLASpolicy + successfulsafe reset allows freshcache B32M32 short32tokencontrol. Otherwise close; never rebaseline.',
  timing_scope='CAPI includes mandatory full-logits/IDs D2H; driver includes CPUvalidation/inputpreparation. Metadata/reset/memory observation recorded separately. Coldload separate. No HTTP/profile.',
  reverse_threshold=1.03,profiling_enabled=False)
 def save():
  temp=path.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2));temp.replace(path)
 def memory(label):
  tick=time.perf_counter();value={'label':label,'free_total_bytes':[list(g['torch'].cuda.mem_get_info(d))for d in range(8)]}
  value['observation_seconds']=time.perf_counter()-tick;report.setdefault('device_memory_snapshots',[]).append(value);save()
 def policy(tile):
  g['layout_mode']=g['HC_mode']=g['tail_SUM_mode']=0;g['projection_tile_mode']=g['c4_tile_mode']=tile;g['precision_mode']=5
  for fn,value in [('modern_deep_precision_policy_r3443',5),('modern_deep_layout_policy_r7052',0),('deep_hc_canonical_policy_r7130',0),('deep_moe_tail_set_r7211',0),('modern_deep_c4_tile_policy_r5920',tile)]:check(getattr(lib,fn)(value))
  g['os'].environ['INFINILM_DEEP_OUTPUT_TILE_R4808']=str(tile)
 def one(batch,steps,label):
  assert batch in [32,64]and steps in [32,128]
  entry={'label':label,'batch':batch,'micro':32,'requested_outputs':steps,'complete':False,'full_IDs_EOS_exact':False,'CAPI_windows':[]}
  report['config_trials_r8650'].append(entry)
  tick=time.perf_counter();check(lib.modern_native_engine_reset(ctx));entry['fresh_cache_reset_seconds']=time.perf_counter()-tick
  memory(label+'-before')
  meta=time.perf_counter();prompts=[ref['input_ids'][i%32]for i in range(batch)];gold=np.array([ref['generated_ids'][i%32][:steps]for i in range(batch)],np.int64)
  assert gold.shape==(batch,steps)and not np.any(gold==cfg['eos_token_id'])
  outbuf=np.full((32,cfg['vocab_size']),0x7fc0,np.uint16);emitted=np.full((steps,batch),-999,np.int64);stamps=np.full((steps,batch),np.nan,np.float64)
  before=[int(lib.deep_output_tile_calls_r4808(d))for d in range(8)]
  c4before=[[int(lib.modern_deep_c4_tile_calls_r5920(d,k))for k in range(2)]for d in range(8)]
  entry['metadata_and_fixedbuffer_seconds']=time.perf_counter()-meta
  validation=0.;step_times=[];began=time.perf_counter()
  for step in range(steps):
   width=4 if step==0 else 32
   for a in range(0,batch,width):
    rows=list(range(a,min(batch,a+width)));inputs=[prompts[i]if step==0 else[int(emitted[step-1,i])]for i in rows]
    past=[0 if step==0 else 512+step-1]*len(rows);sizes=list(map(len,inputs));positions=sum([list(range(p,p+n))for p,n in zip(past,sizes)],[])
    arrays=[np.ascontiguousarray(v,dtype=np.int64 if i>=4 else np.int32)for i,v in enumerate([rows,past,[p+n for p,n in zip(past,sizes)],[0]+np.cumsum(sizes).tolist(),sum(inputs,[]),positions])]
    out=outbuf[:len(rows)];ids=emitted[step,a:a+len(rows)]
    tick=time.perf_counter_ns();code=lib.modern_native_engine_forward(ctx,sum(sizes),len(rows),*[ctypes.c_void_p(v.ctypes.data)for v in arrays],ctypes.c_void_p(out.ctypes.data),ctypes.c_void_p(ids.ctypes.data));ended=time.perf_counter_ns()
    stamps[step,rows]=ended/1e9-began;check(code)
    entry['CAPI_windows'].append(dict(step=step,chunk=a,rows=len(rows),seconds=(ended-tick)/1e9))
    vt=time.perf_counter();values=(out.astype(np.uint32)<<16).view(np.float32);finite=bool(np.isfinite(values).all());argmax=values.argmax(-1).tolist()==ids.tolist();exact=ids.tolist()==gold[rows,step].tolist();validation+=time.perf_counter()-vt
    if not(finite and argmax and exact):
     failure=dict(step=step,rows=rows,finite=finite,CPUargmax_equal=argmax,CAPI_code=code,actual_IDs=ids.tolist(),reference_IDs=gold[rows,step].tolist(),top2=[])
     if finite and argmax:
      for j,row in enumerate(rows):
       if int(ids[j])==int(gold[row,step]):continue
       best=int(ids[j]);vec=values[j].copy();first=float(vec[best]);vec[best]=-np.inf;second=int(vec.argmax())
       failure['top2'].append(dict(row=row,top1_ID=best,top2_ID=second,top1_logit=first,top2_logit=float(vec[second]),margin=first-float(vec[second]),reference_ID=int(gold[row,step]),reference_logit=float(values[j,gold[row,step]])))
     entry.update(failure=failure,elapsed_wall_seconds=time.perf_counter()-began,generated_ids=[[int(v)for v in emitted[:,i]if v!=-999]for i in range(batch)],healthy_token_negative=bool(finite and argmax and not exact))
     del values;memory(label+'-negative');save();return entry
    del values
   step_times.append(time.perf_counter()-began)
  driver_wall=time.perf_counter()-began;pre=sum(x['seconds']for x in entry['CAPI_windows']if x['step']==0);dec=sum(x['seconds']for x in entry['CAPI_windows']if x['step']>0)
  assert np.array_equal(emitted.T,gold)
  calls=(steps-1)*(batch//32);assert [int(lib.deep_output_tile_calls_r4808(d))-before[d]for d in range(8)]==[43*calls]*8
  csa=sum(int(v)==4 for v in cfg['compress_ratios'][:43]);assert [[int(lib.modern_deep_c4_tile_calls_r5920(d,k))-c4before[d][k]for k in range(2)]for d in range(8)]==[[2*csa*calls]*2 for _ in range(8)]
  gaps=np.diff(stamps,axis=0);assert np.isfinite(stamps).all()and np.all(gaps>0)
  entry.update(complete=True,full_IDs_EOS_exact=steps==128,short_control_only=steps==32,generated_ids=emitted.T.tolist(),request_completion_offsets_seconds=stamps.T.tolist(),native_prefill_seconds=pre,native_decode_seconds=dec,driver_wall_seconds=driver_wall,driver_decode_seconds=step_times[-1]-step_times[0],CPU_validation_seconds=validation,per_step_completion_seconds=step_times)
  entry['request_metrics']=[dict(row=i,output_tokens=steps,TTFT_seconds=float(stamps[0,i]),decode_tok_s=float((steps-1)/(stamps[-1,i]-stamps[0,i])),ITL_p95_seconds=float(np.percentile(gaps[:,i],95)),max_gap_seconds=float(gaps[:,i].max()))for i in range(batch)]
  if steps==128:
   entry.update(native_decode_tok_s=batch*127/dec,native_decode_tok_s_per_card=batch*127/dec/8,native_prefill_input_tok_s=batch*512/pre,native_E2E_tok_s=batch*128/(pre+dec),driver_decode_tok_s=batch*127/entry['driver_decode_seconds'],driver_E2E_tok_s=batch*128/driver_wall,driver_E2E_tok_s_per_card=batch*128/driver_wall/8)
  entry['request_latency_scope']='All benchmarkrequests arrive at trajectory start after reset/metadata. ExistingchunkCAPIreturn beforeCPUvalidation. Includes waiting for laterprefillgroups; notamortizedperuser or HTTP.'
  memory(label+'-after');save();return entry
 memory('afterload');policy(0);g['run_chat']('before');assert report['finite_before_passed'];policy(16)
 candidate=one(64,128,'B64M32-first')
 if not candidate['complete']:
  report['outcome']='B64_TOKEN_NEGATIVE_CONTROL_SKIPPED_HEALTH_UNKNOWN';report['quality_passed']=False
  prewindows=[v['seconds']for v in candidate['CAPI_windows']if v['step']==0]
  estimate=(float(np.median(prewindows))*8 if prewindows else 180)+120
  report['negative_control_budget']={'remaining_process_seconds':2020-(time.perf_counter()-g['start']),'required_from_observed_prefill_plus_120s_reserve':estimate}
  if candidate.get('healthy_token_negative')and report['negative_control_budget']['remaining_process_seconds']>estimate:
   try:
    owner=g['verify_owner_affinity'](lib);policy_state=g['policy_snapshot']();assert owner['verified']and owner['width']==4 and policy_state['failures']==0
    report['negative_precontrol_health']={'owner':owner,'BLAS':policy_state,'last_CAPI_code':0,'safe_reset_contract':'DeepNative completedforward sync/poison contract plusserializedEngine reset waitingallworkers; resetreturn0 required beforecontrol.'}
    control=one(32,32,'B32M32-freshcache-short32-negative-control')
    report['outcome']='B64_TOKEN_NEGATIVE_B32_SHORT_CONTROL_'+('PASS'if control['complete']else'FAIL')
   except BaseException as error:report['negative_control_health_or_reset_error']=repr(error);report['outcome']='B64_NEGATIVE_HEALTH_CONTROL_ABORTED'
  policy(0);report['after_QA_skipped']='candidate_tokennegative; shortcontrol notperformance';save();return
 report['baseline_budget']={'remaining_process_seconds':2020-(time.perf_counter()-g['start']),'required_candidate_half_wall_plus_QA_reserve':candidate['driver_wall_seconds']/2+120}
 if report['baseline_budget']['remaining_process_seconds']<=report['baseline_budget']['required_candidate_half_wall_plus_QA_reserve']:
  report.update(outcome='B64_FULL_IDS_PASS_BASELINE_SKIPPED_HARD_BUDGET',quality_passed=True,after_QA_skipped='hardbudgetreserve');policy(0);save();return
 baseline=one(32,128,'B32M32-first-control')
 if not baseline['complete']:
  report.update(outcome='B32_FULL_CONTROL_TOKEN_NEGATIVE_STOP',quality_passed=False,after_QA_skipped='controlnegative');policy(0);save();return
 report['firstpair_capacity_decode_ratio']=candidate['native_decode_tok_s']/baseline['native_decode_tok_s']
 report['firstpair_fullcost_E2E_ratio']=candidate['driver_E2E_tok_s']/baseline['driver_E2E_tok_s']
 remaining=2020-(time.perf_counter()-g['start']);needed=candidate['driver_wall_seconds']+baseline['driver_wall_seconds']+120
 report['reverse_budget']={'remaining_process_seconds':remaining,'firstpair_measured_pair_wall_plus_QA_reserve_seconds':needed}
 if min(report['firstpair_capacity_decode_ratio'],report['firstpair_fullcost_E2E_ratio'])>1.03 and remaining>needed:
  for batch in [32,64]:
   result=one(batch,128,f'B{batch}M32-reverse')
   if not result['complete']:
    report.update(outcome='REVERSE_TOKEN_NEGATIVE_STOP',quality_passed=False,after_QA_skipped='reversenegative');policy(0);save();return
  report['outcome']='FULL_PAIRED_MATRIX_COMPLETE_PENDING_AUDIT'
 else:report['outcome']='FIRSTPAIR_ONLY_NO_MATERIAL_GAIN_OR_BUDGET_STOP'
 policy(0);g['run_chat']('after_restore');assert report['finite_after_restore_passed'];report['quality_passed']=True;report['performance_qualified']=True;report['performance_repeat_qualified']=len(report['config_trials_r8650'])==4;report['config_recommended']=False;save()
