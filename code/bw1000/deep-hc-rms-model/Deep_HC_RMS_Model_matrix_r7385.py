"""Same-load ordinary O/N/N/O, complete final IDs/EOS/reference and QA, no profiling."""
import ctypes,json,hashlib,time,os
import numpy as np
def run(g):
 r,tag,lib,ctx,cfg,check,report,path=[g[k]for k in ['r','tag','lib','ctx','cfg','check','report','path']]
 rp=r/'incoming/Deep-B32-samegeom-reference-r7269.json';assert hashlib.sha256(rp.read_bytes()).hexdigest()=='ba577de61c056f6e701005f9b1622cfac284d38e1671bb480087187544f35797';ref=json.loads(rp.read_text());assert ref['DSO_SHA256']=='0e0512551483457af01944e9c9298442f499d959836a1c5b66dfa438a3c6c4ea'
 fresh=json.loads((r/'results'/f'deep-serving-cases-r2130-{tag}.json').read_text());prompts=ref['input_ids'];assert prompts==[fresh['cases'][i%2]['input_ids']for i in range(32)]and all(len(v)==512 for v in prompts);gold=np.array(ref['generated_ids'],np.int64);assert gold.shape==(32,128)
 lib.deep_hc_rms_calls_r7382.argtypes=[ctypes.c_int]*2;lib.deep_hc_rms_calls_r7382.restype=ctypes.c_uint64
 report.update(HC_RMS_Model_trials=[],performance_adopted=False,formal_Model_complete=False,ordinary_environment='No hipprof/profiling runtime/interposition deliberately enabled; nativeCAPI mandatorylogits/IDsD2H retained. CPUfullvocab validation aftereachCAPI timer, includeddriverwall.',scope='Same-load narrowHC-RMS mathonly; fixedB32M32/I512/maxO128. Baseline0e051 samegeometry679/699 frozenfullIDs retained; candidate also comparedtosame-loadOFF. No crossB gold orGPUF64. OneModelweightload, O/N/N/O, freshQA andrestore, no newprofilematrix.')
 def save():
  tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2));tmp.replace(path)
 def policy(tile,flag):
  g['layout_mode']=g['HC_mode']=g['tail_SUM_mode']=0;g['projection_tile_mode']=tile;g['c4_tile_mode']=tile;g['precision_mode']=5
  check(lib.modern_deep_precision_policy_r3443(5));check(lib.modern_deep_layout_policy_r7052(0));check(lib.deep_hc_canonical_policy_r7130(0));check(lib.deep_moe_tail_set_r7211(0));check(lib.modern_deep_c4_tile_policy_r5920(tile));os.environ['INFINILM_DEEP_OUTPUT_TILE_R4808']=str(tile);os.environ['INFINILM_DEEP_HC_RMS_R7382']=str(flag)
 def counts():return [[int(lib.deep_hc_rms_calls_r7382(d,k))for k in range(3)]for d in range(8)]
 policy(0,0);before=counts();g['run_chat']('before');after=counts();assert report['finite_before_passed']and all(after[d][1]==before[d][1]for d in range(8))
 buffer=np.full((32,cfg['vocab_size']),0x7fc0,dtype=np.uint16);emitted=np.full((128,32),-999,np.int64);report['host_buffer_count']=1;report['host_buffer_payload_bytes']=int(buffer.nbytes)
 baseline=None
 try:
  for label,flag in [('OFF0',0),('ON0',1),('ON1',1),('OFF1',0)]:
   policy(16,flag);check(lib.modern_native_engine_reset(ctx));emitted.fill(-999);capi=[];timeline=[];checked=0;validation_seconds=0.;start=time.perf_counter();cb=counts();reject=None
   for step in range(128):
    for a in range(0,32,4 if step==0 else 32):
     rows=list(range(a,min(32,a+(4 if step==0 else 32))));inputs=[prompts[i]if not step else[int(emitted[step-1,i])]for i in rows];past=[0 if not step else 512+step-1 for _ in rows];widths=list(map(len,inputs));positions=sum([list(range(p,p+w))for p,w in zip(past,widths)],[]);arrays=[np.ascontiguousarray(v,dtype=np.int64 if i>=4 else np.int32)for i,v in enumerate([rows,past,[p+w for p,w in zip(past,widths)],[0]+np.cumsum(widths).tolist(),sum(inputs,[]),positions])];out=buffer[:len(rows)];idsout=emitted[step,a:a+len(rows)]
     tick=time.perf_counter_ns();code=lib.modern_native_engine_forward(ctx,sum(widths),len(rows),*[ctypes.c_void_p(v.ctypes.data)for v in arrays],ctypes.c_void_p(out.ctypes.data),ctypes.c_void_p(idsout.ctypes.data));end=time.perf_counter_ns();check(code);capi.append({'step':step,'chunk':a,'rows':len(rows),'seconds':(end-tick)/1e9})
     vs=time.perf_counter();values=(out.astype(np.uint32)<<16).view(np.float32);finite=bool(np.isfinite(values).all());argmax_ok=bool(np.array_equal(values.argmax(-1),idsout));validation_seconds+=time.perf_counter()-vs;checked+=1;del values
     dif=[{'row':i,'step':step,'actual':int(idsout[k]),'reference':int(gold[i,step]),'same_load_OFF':None if baseline is None else int(baseline[i,step])}for k,i in enumerate(rows)if int(idsout[k])!=int(gold[i,step])or baseline is not None and int(idsout[k])!=int(baseline[i,step])]
     eos_bad=any(int(i)==cfg['eos_token_id']for i in idsout);badids=any(int(i)<0 or int(i)>=cfg['vocab_size']for i in idsout)
     if not finite or not argmax_ok or dif or eos_bad or badids:
      reject={'label':label,'flag':flag,'step':step,'rows':rows,'finite':finite,'CPUargmax_equal':argmax_ok,'first_differences':dif,'actual_IDs':idsout.tolist(),'unexpectedEOS':eos_bad,'badID':badids};report['first_quality_failure']=reject;save();break
    timeline.append(time.perf_counter()-start)
    if reject:break
   ca=counts();delta=[[ca[d][k]-cb[d][k]for k in range(3)]for d in range(8)];decode_windows=sum(w['step']>0 for w in capi);expect=86*decode_windows if flag else 0;assert [v[1]for v in delta]==[expect]*8,('actualHC-RMSeligiblehit',label,delta,expect)
   pre=sum(w['seconds']for w in capi if w['step']==0);dec=sum(w['seconds']for w in capi if w['step']>0);generated=emitted.T.copy();complete=reject is None
   result={'trial':label,'flag':flag,'completed128':complete,'full4096_final_IDs_exact_reference':complete and bool(np.array_equal(generated,gold)),'same_load_OFF_exact':None if baseline is None else complete and bool(np.array_equal(generated,baseline)),'generated_ids':generated.tolist(),'EOS_pattern_matches_reference':complete,'CPUargmax_all_windows':reject is None,'fullvocab_checked_CAPI_windows':checked,'CPU_validation_seconds_outside_CAPI':validation_seconds,'CAPI_windows':capi,'native_prefill_seconds':pre,'native_decode_seconds':dec,'native_CAPI_prefill_input_tok_s':16384/pre if complete else None,'native_CAPI_decode_output_tok_s':4064/dec if complete else None,'driver_loop_wall_seconds':time.perf_counter()-start,'driver_prefill_wall_seconds':timeline[0],'driver_decode_wall_seconds':timeline[-1]-timeline[0]if complete else None,'per_step_completion_seconds':timeline,'HC_RMS_counter_delta':delta,'first_quality_failure':reject};report['HC_RMS_Model_trials'].append(result);save();print('HC_RMS_MODEL_TRIAL '+json.dumps({'label':label,'complete':complete,'prefill_tok_s':result['native_CAPI_prefill_input_tok_s'],'decode_tok_s':result['native_CAPI_decode_output_tok_s'],'first_quality_failure':reject}),flush=True)
   if reject:
    if not flag:raise RuntimeError('OFF_baseline_or_restore_regression')
    report['candidate_rejected']=True;break
   assert checked==135 and np.array_equal(generated,gold)
   if baseline is None:baseline=generated.copy()
   assert min(g['torch'].cuda.mem_get_info(d)[0]for d in range(8))>=6*1024**3
  if not report.get('candidate_rejected'):
   assert len(report['HC_RMS_Model_trials'])==4;report['formal_Model_complete']=True
   policy(0,1);cb=counts();g['run_chat']('candidate_ON_QA');ca=counts();assert report['finite_candidate_ON_QA_passed']and all(ca[d][1]==cb[d][1]for d in range(8))
 finally:
  policy(0,0);save()
 cb=counts();g['run_chat']('after_restore');ca=counts();assert report['finite_after_restore_passed']and all(ca[d][1]==cb[d][1]for d in range(8));report['QA_restore_counter_hits_zero']=True;save()
