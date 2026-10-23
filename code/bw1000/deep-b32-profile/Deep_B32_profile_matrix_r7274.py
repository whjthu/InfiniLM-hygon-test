"""Same-math B32 baseline: isolated CAPI windows plus one bounded HIP trace."""
import ctypes,json,pathlib,hashlib,time,os
import numpy as np

def run(g):
 r,tag,lib,ctx,cfg,check,report,path=[g[k]for k in ['r','tag','lib','ctx','cfg','check','report','path']]
 refp=r/'incoming/Deep-B32-samegeom-reference-r7269.json';assert hashlib.sha256(refp.read_bytes()).hexdigest()=='ba577de61c056f6e701005f9b1622cfac284d38e1671bb480087187544f35797';ref=json.loads(refp.read_text());assert report['library_sha256']==ref['DSO_SHA256']=='0e0512551483457af01944e9c9298442f499d959836a1c5b66dfa438a3c6c4ea'
 current=json.loads((r/'results'/f'deep-serving-cases-r2130-{tag}.json').read_text());prompts=ref['input_ids'];assert prompts==[current['cases'][i%2]['input_ids']for i in range(32)]and all(len(v)==512 for v in prompts);gold=np.array(ref['generated_ids'],np.int64);assert gold.shape==(32,128)
 report.update(profile_B32_r7274=[],performance_adopted=False,formal_profile_complete=False,scope='Existingqualifiedsamegeometry0e051B32/M32/I512/maxO128. Oneweightload, warmup+oneHIPtraceframe plus two trace-off baseline repeatwindows with profiling runtime present. NativeCAPI necessarily includescompletedfull-logits/IDsD2H; Fullvocab finite/CPUargmax validation immediately after each CAPI clock, included in driverloop wall. No newmath/copy/B64, no claimedcodegain.')
 def save():
  tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2));tmp.replace(path)
 def policy(c4):
  g['layout_mode']=g['HC_mode']=g['tail_SUM_mode']=0;g['projection_tile_mode']=16 if c4 else 0;g['c4_tile_mode']=c4;g['precision_mode']=5
  lib.modern_deep_precision_policy_r3443.argtypes=[ctypes.c_int];check(lib.modern_deep_precision_policy_r3443(5));check(lib.modern_deep_layout_policy_r7052(0));check(lib.deep_hc_canonical_policy_r7130(0));check(lib.deep_moe_tail_set_r7211(0));check(lib.modern_deep_c4_tile_policy_r5920(c4));os.environ['INFINILM_DEEP_OUTPUT_TILE_R4808']=str(16 if c4 else 0)
 policy(0);g['run_chat']('before')
 # Pre-touch fixed output memory before any timedCAPI: no perforward allocations/fills.
 report['host_buffer_count']=1;report['host_buffer_payload_bytes']=32*cfg['vocab_size']*2
 report['host_buffer_scope']='One main emitter reuses one fixed BF16 array, not eight owners or128 retained frames; no new host admission gate or weightbank. CPUvalidation transient one F32 frame outside CAPI clock.'
 buffer=np.full((32,cfg['vocab_size']),0x7fc0,dtype=np.uint16);emitted=np.full((128,32),-999,np.int64)
 ctypes.CDLL('/opt/dtk/hip/lib/libgalaxyhip.so.5',mode=ctypes.RTLD_GLOBAL);gpu=ctypes.CDLL(None)
 for n in ['hipProfilerStart','hipProfilerStop']:getattr(gpu,n).argtypes=[];getattr(gpu,n).restype=ctypes.c_int
 report['critical_path_trace']={'enabled_trial':'warmup','step':64,'active':False,'frames':[],'purpose':'New GPU/API/stack criticalpath evidence. Traceframe excludedfromformalperformance. Layer/componentattribution onlyifstack+sequence supports verified43anchors, otherwiseUNASSIGNED.'};save()
 trace_active=False
 try:
  for trial in ['warmup','baseline-r1','baseline-r2']:
   policy(16);check(lib.modern_native_engine_reset(ctx));emitted.fill(-999);capi=[];step_times=[];trace_this=trial=='warmup';validation_seconds=0.;checked_windows=0;start=time.perf_counter();phase_prefill_end=None
   counters_before={'projection':[int(lib.deep_output_tile_calls_r4808(d))for d in range(8)],'C4':[[int(lib.modern_deep_c4_tile_calls_r5920(d,k))for k in range(2)]for d in range(8)]}
   for step in range(128):
    for a in range(0,32,4 if step==0 else 32):
     rows=list(range(a,min(32,a+(4 if step==0 else 32))));inputs=[prompts[i]if step==0 else[int(emitted[step-1,i])]for i in rows];past=[0 if step==0 else 512+step-1 for _ in rows];widths=list(map(len,inputs));positions=sum([list(range(p,p+w))for p,w in zip(past,widths)],[]);arrays=[np.ascontiguousarray(v,dtype=np.int64 if i>=4 else np.int32)for i,v in enumerate([rows,past,[p+w for p,w in zip(past,widths)],[0]+np.cumsum(widths).tolist(),sum(inputs,[]),positions])];out=buffer[:len(rows)];idsout=emitted[step,a:a+len(rows)]
     selected=trace_this and step==64
     if selected:assert gpu.hipProfilerStart()==0;trace_active=True;report['critical_path_trace']['active']=True;frame_start=time.monotonic_ns()
     tick=time.perf_counter_ns();code=lib.modern_native_engine_forward(ctx,sum(widths),len(rows),*[ctypes.c_void_p(v.ctypes.data)for v in arrays],ctypes.c_void_p(out.ctypes.data),ctypes.c_void_p(idsout.ctypes.data));end=time.perf_counter_ns()
     if selected:
      frame_end=time.monotonic_ns();assert gpu.hipProfilerStop()==0;trace_active=False;report['critical_path_trace']['active']=False;report['critical_path_trace']['frames'].append({'step':step,'begin_monotonic_ns':frame_start,'end_monotonic_ns':frame_end})
     check(code);capi.append({'step':step,'chunk':a,'rows':len(rows),'seconds':(end-tick)/1e9,'traced':selected})
     validation_begin=time.perf_counter();values=(out.astype(np.uint32)<<16).view(np.float32)
     finite=bool(np.isfinite(values).all());argmax_ok=values.argmax(-1).tolist()==idsout.tolist();checked_windows+=1;validation_seconds+=time.perf_counter()-validation_begin;del values
     if not(finite and argmax_ok):
      report['early_quality_error']={'trial':trial,'step':step,'rows':rows,'finite':finite,'CPUargmax_equal':argmax_ok,'actual_IDs':idsout.tolist()};save();raise RuntimeError('fullvocab_quality_stop')
     if not(np.all((idsout>=0)&(idsout<cfg['vocab_size']))and np.all(idsout!=cfg['eos_token_id'])):
      report['early_ID_error']={'trial':trial,'step':step,'rows':rows,'actual_IDs':idsout.tolist(),'reference_IDs':[int(gold[i,step])for i in rows]};save();raise RuntimeError('unexpectedEOS_or_badID_stop')
    step_times.append(time.perf_counter()-start)
    if step==0:phase_prefill_end=step_times[-1]
   loop_wall=time.perf_counter()-start
   # Every full-vocabulary finite/argmax check ran immediately after its CAPI timer.
   assert checked_windows==135
   generated=emitted.T.copy();mismatches=[]
   for row in range(32):
    where=np.flatnonzero(generated[row]!=gold[row])
    if len(where):s=int(where[0]);mismatches.append({'row':row,'first':s,'actual':int(generated[row,s]),'expected':int(gold[row,s])})
   counter_after={'projection':[int(lib.deep_output_tile_calls_r4808(d))for d in range(8)],'C4':[[int(lib.modern_deep_c4_tile_calls_r5920(d,k))for k in range(2)]for d in range(8)]};proj_delta=[b-a for a,b in zip(counters_before['projection'],counter_after['projection'])];assert proj_delta==[127*43]*8
   csa=sum(int(x)==4 for x in cfg['compress_ratios'][:43]);c4_delta=[[counter_after['C4'][d][k]-counters_before['C4'][d][k]for k in range(2)]for d in range(8)];assert c4_delta==[[2*csa*127]*2 for d in range(8)]
   prefill_native=sum(v['seconds']for v in capi if v['step']==0);decode_native=sum(v['seconds']for v in capi if v['step']>0);entry={'trial':trial,'formal_measurement':not trace_this,'full4096_final_IDs_exact_reference':not mismatches,'mismatches':mismatches,'CPUargmax_all128':True,'fullvocab_checked_CAPI_windows':checked_windows,'CPU_validation_seconds_outside_CAPI':validation_seconds,'generated_ids':generated.tolist(),'CAPI_windows':capi,'native_prefill_seconds':prefill_native,'native_decode_seconds':decode_native,'native_CAPI_prefill_input_tok_s':16384/prefill_native,'native_CAPI_decode_output_tok_s':4064/decode_native,'driver_prefill_wall_seconds':phase_prefill_end,'driver_loop_wall_seconds':loop_wall,'driver_decode_wall_seconds':step_times[-1]-step_times[0],'driver_decode_output_tok_s':4064/(step_times[-1]-step_times[0]),'per_step_completion_seconds':step_times,'projection_counter_delta':proj_delta,'C4_counter_delta':c4_delta,'scope':'CAPIincludesownmandatorylogits/IDsD2H. Driverloop includes per-CAPI fullvocab CPUvalidation/inputprepare; notHTTP. Both baseline trials are trace-off with profiling runtime present, not an uninstrumented peak. Warmupcontainsoneprofiledframe andnotformalperformance.'};report['profile_B32_r7274'].append(entry);save();assert not mismatches,('stop_on_first_quality_failure_do_not_rebaseline',trial,mismatches)
   assert min(g['torch'].cuda.mem_get_info(d)[0]for d in range(8))>=6*1024**3
  assert len(report['critical_path_trace']['frames'])==1
  report['formal_profile_complete']=True
 finally:
  if trace_active:gpu.hipProfilerStop();report['critical_path_trace']['active']=False
  policy(0);save()
 g['run_chat']('after_restore');save()
