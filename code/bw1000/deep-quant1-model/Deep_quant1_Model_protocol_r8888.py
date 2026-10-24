"""Single current-r194 metadata factor, one load, fixed B32M32 O/N/N/O."""
import ctypes,hashlib,json,time
import numpy as np
REF_SHA='ba577de61c056f6e701005f9b1622cfac284d38e1671bb480087187544f35797'
def run(g):
 r,lib,ctx,cfg,check,report,path=[g[k]for k in ['r','lib','ctx','cfg','check','report','path']]
 p=r/'incoming/Deep-B32-samegeom-reference-r7269.json';assert hashlib.sha256(p.read_bytes()).hexdigest()==REF_SHA;ref=json.loads(p.read_text());assert ref['DSO_SHA256']=='0e0512551483457af01944e9c9298442f499d959836a1c5b66dfa438a3c6c4ea'
 assert cfg['num_hidden_layers']==43 and len(ref['input_ids'])==32 and all(len(v)==512 for v in ref['input_ids'])
 report.update(quant1_trials_r8888=[],performance_adopted=False,performance_qualified=False,quality_passed=False,outcome='RUNNING',reference_SHA256=REF_SHA,scope='ActualTP8PP1 B32I512O128 Model one load,acceptedPACKON A versus quant1once+tinyfusedcopy Q only. Originalquant2/GemmEx/Runtime/stream/drain;noactivation/scatter/bankleasechange. Fourprimaryfull128 A/Q/Q/A,firstA/Q qualifyfreshandfrozengold,finalAendpoint+restoreQA. Noextraindependentwarm;firstarmcold/endpointsreported',input_scope='Original699/679 two frozen natural512 prompts duplicated across32rows, not32unique traffic.',timing_scope='CAPI includes mandatory full-logits/IDsD2H;driver includes fullvocabCPUfinite/argmax,inputprepare. Requeststamps actualCAPIreturn, notHTTP/client. Allrequests arrivalatphaseclockafterreset/bufferprepare, includes laterprefillcohortwaiting.')
 def save():
  q=path.with_suffix('.tmp');q.write_text(json.dumps(report,indent=2));q.replace(path)
 def memory(label):
  tick=time.perf_counter();free=[list(g['torch'].cuda.mem_get_info(d))for d in range(8)];assert min(v[0]for v in free)>=6*1024**3;report.setdefault('device_memory_snapshots',[]).append({'label':label,'free_total_bytes':free,'observation_seconds':time.perf_counter()-tick});save()
 def runtime_floor():
  free=min(g['torch'].cuda.mem_get_info(d)[0]for d in range(8));assert free>=512*1024**2;report['minimum_runtime_boundary_free_bytes']=min(report.get('minimum_runtime_boundary_free_bytes',free),free)
 lib.deep_prefill_pack_counts_r8767.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t];lib.deep_prefill_pack_counts_r8767.restype=ctypes.c_int
 def counts():
  result=[]
  for d in range(8):
   q=(ctypes.c_uint64*12)();check(lib.deep_prefill_pack_counts_r8767(d,q,12));result.append(list(q))
  return result
 lib.deep_model_quant1_mode_r8886.argtypes=[ctypes.c_int];lib.deep_model_quant1_mode_r8886.restype=ctypes.c_int
 lib.deep_model_quant1_counts_r8886.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t];lib.deep_model_quant1_counts_r8886.restype=ctypes.c_int
 def modelcounts():
  ans=[]
  for d in range(8):
   q=(ctypes.c_uint64*8)();check(lib.deep_model_quant1_counts_r8886(d,q,8));ans.append(list(q))
  return ans
 def select(arm):check(lib.deep_model_quant1_mode_r8886(arm));g['os'].environ['INFINILM_DEEP_PREFILL_PACKUPLOAD_R8750']='1'
 def policy(tile):
  g['layout_mode']=g['HC_mode']=g['tail_SUM_mode']=0;g['projection_tile_mode']=g['c4_tile_mode']=tile;g['precision_mode']=5
  for fn,value in [('modern_deep_precision_policy_r3443',5),('modern_deep_layout_policy_r7052',0),('deep_hc_canonical_policy_r7130',0),('deep_moe_tail_set_r7211',0),('modern_deep_c4_tile_policy_r5920',tile)]:check(getattr(lib,fn)(value))
  g['os'].environ['INFINILM_DEEP_OUTPUT_TILE_R4808']=str(tile)
 fresh_A=None
 def one(arm,label):
  nonlocal fresh_A
  flag=1;select(arm);model_before=modelcounts()
  batch,steps=32,128
  assert batch in [32,64]and steps in [32,128]
  entry={'label':label,'batch':batch,'micro':32,'requested_outputs':steps,'complete':False,'full_IDs_EOS_exact':False,'CAPI_windows':[]}
  report['quant1_trials_r8888'].append(entry);entry['pack_mode']=flag;entry['model_arm']=arm
  tick=time.perf_counter();check(lib.modern_native_engine_reset(ctx));entry['fresh_cache_reset_seconds']=time.perf_counter()-tick;g['os'].environ['INFINILM_DEEP_PREFILL_PACKUPLOAD_R8750']=str(flag);pack_before=counts()
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
    runtime_floor();tick=time.perf_counter_ns();code=lib.modern_native_engine_forward(ctx,sum(sizes),len(rows),*[ctypes.c_void_p(v.ctypes.data)for v in arrays],ctypes.c_void_p(out.ctypes.data),ctypes.c_void_p(ids.ctypes.data));ended=time.perf_counter_ns()
    stamps[step,rows]=ended/1e9-began;check(code);runtime_floor()
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
  if fresh_A is None:assert arm==0;fresh_A=emitted.T.copy()
  assert np.array_equal(emitted.T,fresh_A)
  calls=(steps-1)*(batch//32);assert [int(lib.deep_output_tile_calls_r4808(d))-before[d]for d in range(8)]==[43*calls]*8
  csa=sum(int(v)==4 for v in cfg['compress_ratios'][:43]);assert [[int(lib.modern_deep_c4_tile_calls_r5920(d,k))-c4before[d][k]for k in range(2)]for d in range(8)]==[[2*csa*calls]*2 for _ in range(8)]
  gaps=np.diff(stamps,axis=0);assert np.isfinite(stamps).all()and np.all(gaps>0)
  entry.update(complete=True,frozen814_and_fresh_A_full_IDs_exact=True,full_IDs_EOS_exact=steps==128,short_control_only=steps==32,generated_ids=emitted.T.tolist(),request_completion_offsets_seconds=stamps.T.tolist(),native_prefill_seconds=pre,native_decode_seconds=dec,driver_wall_seconds=driver_wall,driver_decode_seconds=step_times[-1]-step_times[0],CPU_validation_seconds=validation,per_step_completion_seconds=step_times)
  entry['request_metrics']=[dict(row=i,output_tokens=steps,TTFT_seconds=float(stamps[0,i]),decode_tok_s=float((steps-1)/(stamps[-1,i]-stamps[0,i])),ITL_p95_seconds=float(np.percentile(gaps[:,i],95)),ITL_p99_seconds=float(np.percentile(gaps[:,i],99)),max_gap_seconds=float(gaps[:,i].max()))for i in range(batch)]
  if steps==128:
   entry.update(native_decode_tok_s=batch*127/dec,native_decode_tok_s_per_card=batch*127/dec/8,native_prefill_input_tok_s=batch*512/pre,native_E2E_tok_s=batch*128/(pre+dec),driver_decode_tok_s=batch*127/entry['driver_decode_seconds'],driver_E2E_tok_s=batch*128/driver_wall,driver_E2E_tok_s_per_card=batch*128/driver_wall/8)
  entry['request_latency_scope']='All benchmarkrequests arrive at trajectory start after reset/metadata. ExistingchunkCAPIreturn beforeCPUvalidation. Includes waiting for laterprefillgroups; notamortizedperuser or HTTP.'
  delta=[[b-a for a,b in zip(x,y)]for x,y in zip(pack_before,counts())];entry['pack_counter_delta']=delta
  for q in delta:
   assert q[flag]==344 and q[1-flag]==0 and q[2+flag]>0 and q[3-flag]==0
   assert q[4]==2*q[2]and q[5]==2*q[1]and q[6]==2*q[2]and q[7]==q[1]
   assert q[8+flag]==43*32*512*6*16 and q[9-flag]==0 and q[10]==43*127 and q[11]==0
  md=[[b-a for a,b in zip(x,y)]for x,y in zip(model_before,modelcounts())];entry['model_counter_delta']=md
  for v,pc in zip(md,delta):
   assert v[arm]==344 and sum(v[:2])==344 and v[2+arm]==pc[3] and sum(v[2:4])==pc[3]
   assert v[4]==(344 if arm else 0) and v[5]==(43*32*512 if arm else 0) and v[6]==(pc[3] if arm else 0) and v[7]==(43*32*512*6 if arm else 0)
  assert min(g['torch'].cuda.mem_get_info(d)[0]for d in range(8))>=6*1024**3
  memory(label+'-after');save();print('DEEP_PACK_ARM_COMPLETE',label,entry['native_prefill_input_tok_s'],entry['native_decode_tok_s'],entry['driver_E2E_tok_s'],flush=True);return entry
 memory('afterload');select(0);policy(0);g['run_chat']('before');assert report['finite_before_passed']
 for i,arm in enumerate([0,1,1,0]):
  policy(16);entry=one(arm,('A'if arm==0 else'Q')+'-'+str(i))
  if not entry['complete']:save();raise RuntimeError('firstbaseline_or_candidate_full128_quality_failed_STOP')
  if i==1:
   policy(0);select(1);g['run_chat']('candidate_pack');assert report['finite_candidate_pack_passed']
 policy(0);select(0);g['run_chat']('after_restore');assert report['finite_after_restore_passed']
 report.update(outcome='FULL_AQQA_COMPLETE_PENDING_TERMINAL',quality_passed=True,performance_qualified=True,performance_adopted=False,qualification_scope='firstA/firstQ full128 fresh+frozenIDs;finalAendpoint+restoreQA',performance_scope='Quant1-only AtoQ; noactivationbundle. Noindependentwarm arm;firstarmcold/endpointsreported.');save()
