"""One load accepted A/C/C/A; only default-OFF reference-prefill scratch reuse."""
import ctypes,hashlib,json,time
import numpy as np
REF_SHA='ba577de61c056f6e701005f9b1622cfac284d38e1671bb480087187544f35797'
def run(g):
 r,lib,ctx,cfg,check,report,path=[g[k]for k in ['r','lib','ctx','cfg','check','report','path']]
 p=r/'incoming/Deep-B32-samegeom-reference-r7269.json';assert hashlib.sha256(p.read_bytes()).hexdigest()==REF_SHA;ref=json.loads(p.read_text());assert ref['DSO_SHA256']=='0e0512551483457af01944e9c9298442f499d959836a1c5b66dfa438a3c6c4ea'
 assert cfg['num_hidden_layers']==43 and len(ref['input_ids'])==32 and all(len(v)==512 for v in ref['input_ids'])
 report.update(owner_trials_r8956=[],performance_adopted=False,performance_qualified=False,quality_passed=False,outcome='RUNNING',reference_SHA256=REF_SHA,scope='ActualTP8PP1 B32I512O128 one load,A/C/C full128 +optionalA3onlyifremaining>=400s. AcceptedPACKON q1/activation/scatter/bankleaseOFF; onlyprefill FP32reference scratchOFF/ON,8933tilesOFF. AcceptedT32decodefused5461calls/ownerunchanged; frozen814 andfreshA finalIDs/EOS;originalbefore/restoreQA;noownerextraeventcapture.',input_scope='Original699/679 two frozen natural512 prompts duplicated across32rows, not32unique traffic.',timing_scope='CAPI includes mandatory full-logits/IDsD2H;driver includes fullvocabCPUfinite/argmax,inputprepare. Requeststamps actualCAPIreturn, notHTTP/client. Allrequests arrivalatphaseclockafterreset/bufferprepare, includes laterprefillcohortwaiting.')
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
 def select(arm):
  check(lib.deep_model_quant1_mode_r8886(0));check(lib.deep_latent_prefill_mode_r8933(0));check(lib.deep_reference_scratch_mode_r8953(arm));g['os'].environ['INFINILM_DEEP_PREFILL_PACKUPLOAD_R8750']='1'
 def policy(tile):
  g['layout_mode']=g['HC_mode']=g['tail_SUM_mode']=0;g['projection_tile_mode']=g['c4_tile_mode']=tile;g['precision_mode']=5
  for fn,value in [('modern_deep_precision_policy_r3443',5),('modern_deep_layout_policy_r7052',0),('deep_hc_canonical_policy_r7130',0),('deep_moe_tail_set_r7211',0),('modern_deep_c4_tile_policy_r5920',tile)]:check(getattr(lib,fn)(value))
  g['os'].environ['INFINILM_DEEP_OUTPUT_TILE_R4808']=str(tile)
 lib.deep_latent_prefill_mode_r8933.argtypes=[ctypes.c_int];lib.deep_latent_prefill_mode_r8933.restype=ctypes.c_int
 lib.deep_reference_scratch_mode_r8953.argtypes=[ctypes.c_int];lib.deep_reference_scratch_mode_r8953.restype=ctypes.c_int
 lib.deep_reference_scratch_counts_r8954.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t];lib.deep_reference_scratch_counts_r8954.restype=ctypes.c_int
 def scratchcounts():
  ans=[]
  for d in range(8):
   q=(ctypes.c_uint64*4)();check(lib.deep_reference_scratch_counts_r8954(d,q,4));ans.append(list(q))
  return ans
 report['owner_event_windows']=[]
 report['owner_event_capture_enabled']=False
 fresh_A=None
 def one(arm,label):
  nonlocal fresh_A
  flag=1;select(arm);model_before=modelcounts();scratch_before=scratchcounts();fused_before=[int(lib.modern_deep_latent_fused_calls_r681(d))for d in range(8)];g['checkpoint']('arm_enter_'+label)
  batch,steps=32,128
  assert batch in [32,64]and steps in [32,128]
  entry={'label':label,'batch':batch,'micro':32,'requested_outputs':steps,'complete':False,'full_IDs_EOS_exact':False,'CAPI_windows':[]}
  report['owner_trials_r8956'].append(entry);entry['pack_mode']=flag;entry['model_arm']=0;entry['scratch_arm']=arm;entry['scratch_prefill_chunk_checks']=[]
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
    runtime_floor()
    lb=scratchcounts()if step==0 else None;fb=[int(lib.modern_deep_latent_fused_calls_r681(d))for d in range(8)]if step==0 else None
    if step==0:g['checkpoint']('prefill_CAPI_enter_'+label,a)
    tick=time.perf_counter_ns();code=lib.modern_native_engine_forward(ctx,sum(sizes),len(rows),*[ctypes.c_void_p(v.ctypes.data)for v in arrays],ctypes.c_void_p(out.ctypes.data),ctypes.c_void_p(ids.ctypes.data));ended=time.perf_counter_ns()
    if step==0:report['selected_native_returncode']=code;g['checkpoint']('prefill_CAPI_return_'+label,a)
    stamps[step,rows]=ended/1e9-began;check(code);runtime_floor()
    if step==0:
     ld=[[y-x for x,y in zip(p,q)]for p,q in zip(lb,scratchcounts())];expected=[0,43,0,129]if arm else[43,0,43*3*sum(sizes),0]
     assert ld==[expected]*8,('scratch_actual_Tblock_path',arm,sum(sizes),ld,expected)
     assert [int(lib.modern_deep_latent_fused_calls_r681(d))-v for d,v in enumerate(fb)]==[0]*8,('allprefillreference_fused0',arm)
     entry['scratch_prefill_chunk_checks'].append(dict(chunk=a,total_tokens=sum(sizes),counter_delta=ld))
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
   assert v[0]==344 and v[1]==0 and v[2]==pc[3] and v[3:]==[0]*5
  entry['scratch_counter_delta']=[[y-x for x,y in zip(p,q)]for p,q in zip(scratch_before,scratchcounts())]
  assert entry['scratch_counter_delta']==[[0,344,0,1032]if arm else[344,0,2113536,0]]*8,('prefill_only_no_decode_hits',entry['scratch_counter_delta'])
  entry['accepted_decode_fused_counter_delta']=[int(lib.modern_deep_latent_fused_calls_r681(d))-v for d,v in enumerate(fused_before)];assert entry['accepted_decode_fused_counter_delta']==[5461]*8
  assert min(g['torch'].cuda.mem_get_info(d)[0]for d in range(8))>=6*1024**3
  memory(label+'-after');g['checkpoint']('arm_return_'+label);save();print('DEEP_PACK_ARM_COMPLETE',label,entry['native_prefill_input_tok_s'],entry['native_decode_tok_s'],entry['driver_E2E_tok_s'],flush=True);return entry
 g['checkpoint']('load_return');memory('afterload');select(0);policy(0);g['run_chat']('before');assert report['finite_before_passed']
 policy(16)
 for arm,label in [(0,'A0'),(1,'C1'),(1,'C2')]:
  entry=one(arm,label)
  if not entry['complete']:save();raise RuntimeError(label+'_full128_quality_failed_STOP')
 remaining=1740-(time.perf_counter()-g['packet_started']);report['optional_final_A_budget']={'remaining_inner_seconds':remaining,'threshold_seconds':400,'status':'RUNNING'if remaining>=400 else'SKIPPED_BUDGET'};save()
 if remaining>=400:
  entry=one(0,'A3')
  if not entry['complete']:save();raise RuntimeError('A3_full128_quality_failed_STOP')
  report['optional_final_A_budget']['status']='COMPLETED'
 policy(0);select(0);g['run_chat']('after_restore');assert report['finite_after_restore_passed']
 assert report['owner_event_windows']==[]
 report.update(outcome='FULL_ACC_OPTIONAL_A_COMPLETE_PENDING_TERMINAL',quality_passed=True,performance_qualified=True,performance_adopted=False,qualification_scope='threeorfourfull128 frozen814/freshA finalIDs/EOS;before/restore12QA;all8scratchTblockcounterchecks+accepteddecodefused5461',performance_scope='Ordinary unmarkedModel A/C/C plusoptionalA3; noendpointdriftclaimifskipped; nativeprefill/nativeDecode vs driverdecode/E2E andrequestTTFT/ITL separate. Decode sourceunchanged; drift andpairing mustbe reported. Noautomaticadoption.');save()
