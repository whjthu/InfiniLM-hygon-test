"""OneGPU TP8rank0 routedtile; trainedcheckpoint+trainedrouterformula, syntheticBF16hidden. NotModel/quality/TPS."""
import pathlib,json,ctypes,os,time,statistics,hashlib,traceback,gc
import torch
from safetensors import safe_open
R=pathlib.Path('/refactor');INC=pathlib.Path(__file__).parent;MODEL=pathlib.Path('/old/models/ZhipuAI-GLM-5.3');V=ctypes.c_void_p;Z=ctypes.c_size_t
REPORT={};REPORT_PATH=None

def save():
 if REPORT_PATH is not None:
  t=REPORT_PATH.with_suffix('.tmp');t.write_text(json.dumps(REPORT,indent=2));t.replace(REPORT_PATH)
def ptr(t):return V(t.data_ptr())
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def quant(x):
 y=x.float().reshape(x.shape[0],-1,128);sc=y.abs().amax(-1).clamp_min(1e-10)/448.;packed=(y/sc[:,:,None]).clamp(-448,448).to(torch.float8_e4m3fn).view(torch.uint8).reshape(x.shape).contiguous();return packed,sc.contiguous()
def main():
 global REPORT_PATH
 step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];REPORT_PATH=R/'results'/('GLM-T16-chain-diagnostic-r9276-'+step+'.json');assert not REPORT_PATH.exists();q=REPORT;q.update(completed=False,actual_step=step,node=os.uname().nodename,scope=__doc__,trials=[],warm=[],math='original8529Wtree; originalT16 dispatch; irregular option remainsOFF; no math/candidate changes',synthetic_hidden=True,checkpoint_layer=38,TP_partition=8,TP_rank=0,router='TorchF32 samecheckpoint sigmoid+bias top8 formula; NOTnative routerqualification',timer='wholechain wall/finalsync; no stage drain; events streamspan with launchgaps; kernel activity intervalunion distinct; profile overhead independently measured',performance_gate=None);save()
 assert len(os.sched_getaffinity(0))>=8;torch.set_num_threads(1);torch.cuda.set_device(0);assert torch.cuda.device_count()==1;torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
 for n in ['allow_fp16_reduced_precision_reduction','allow_bf16_reduced_precision_reduction','allow_fp16_accumulation']:setattr(torch.backends.cuda.matmul,n,False)
 meta=json.loads((INC/'GPU-manifest.json').read_bytes());proof=json.loads((R/'candidates/GLM-irregular-vector-r9270b/CPU-build.json').read_bytes());assert proof['completed']and sha(proof['library'])==proof['library_SHA256']==meta['library_SHA256'];q['library_SHA256']=proof['library_SHA256'];lib=ctypes.CDLL(proof['library'])
 for n in ['modern_glm_vector_policy_r8540','modern_glm_irregular_vector_policy_r9270']:getattr(lib,n).argtypes=[ctypes.c_int,ctypes.c_int]
 for n in ['modern_glm_vector_get_r8540','modern_glm_irregular_vector_get_r9270']:getattr(lib,n).argtypes=[ctypes.c_int]
 lib.modern_glm_vector_counts_r8540.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64),Z];lib.glm_component_gate_r9270.argtypes=[ctypes.c_int]+[V]*9+[Z,Z,V];lib.glm_component_down_r9270.argtypes=[ctypes.c_int]+[V]*6+[Z,Z,V]
 def stats():
  a=(ctypes.c_uint64*8)();assert lib.modern_glm_vector_counts_r8540(0,a,8)==0;return list(a)
 assert lib.modern_glm_irregular_vector_get_r9270(0)==0;assert lib.modern_glm_vector_policy_r8540(0,1)==0
 # dladdr proves each actualprovider target uses ournewdispatch/originalGPUobjects or exactactualfallbackDSO.
 class DlInfo(ctypes.Structure):_fields_=[('fname',ctypes.c_char_p),('base',V),('sname',ctypes.c_char_p),('saddr',V)]
 proc=ctypes.CDLL(None);proc.dladdr.argtypes=[V,ctypes.POINTER(DlInfo)];proc.dladdr.restype=ctypes.c_int;owners={}
 for n in ['glm_component_gate_r9270','glm_vec_gate_r8528','modern_glm_vector_counts_r8540','glm_grouped_gate_r5123','glm_grouped_down_r5123']:
  a=DlInfo();assert proc.dladdr(ctypes.cast(getattr(lib,n),V),ctypes.byref(a));owner=pathlib.Path(a.fname.decode()).resolve();expect=pathlib.Path('/refactor/candidates/glm-hybrid-decode-r5386/libnative.so')if'grouped'in n else pathlib.Path(proof['library']);assert owner==expect.resolve();owners[n]={'path':str(owner),'SHA256':sha(owner)}
 q['actual_symbol_owners']=owners;save();mapping=json.loads((MODEL/'model.safetensors.index.json').read_bytes())['weight_map'];handles={};banks=[];scales=[];loaded=0;start=time.perf_counter()
 try:
  def load(k):
   nonlocal loaded
   name=mapping[k]
   if name not in handles:handles[name]=safe_open(str(MODEL/name),framework='pt',device='cpu');handles[name].__enter__()
   value=handles[name].get_tensor(k);loaded+=value.nbytes;return value
  for proj in ['gate_proj','up_proj','down_proj']:
   ww=[];ss=[]
   for expert in range(256):
    prefix=f'model.layers.38.mlp.experts.{expert}.{proj}';w=load(prefix+'.weight').view(torch.uint8);tile=w[:,:256]if proj=='down_proj'else w[:256,:];ww.append(tile.contiguous().clone());ss.append(load(prefix+'.weight_scale_inv').float().clone())
   banks.append(torch.stack(ww).cuda());scales.append(torch.stack(ss).cuda());del ww,ss;torch.cuda.synchronize();q['load_progress_projection']=proj;save()
  router=load('model.layers.38.mlp.gate.weight').float().cuda();bias=load('model.layers.38.mlp.gate.e_score_correction_bias').float().cuda()
 finally:
  for h in handles.values():h.__exit__(None,None,None)
 torch.cuda.synchronize();q.update(weight_load_seconds=time.perf_counter()-start,checkpoint_payload_referenced_bytes=loaded,weight_load_count=1,free_after_load=torch.cuda.mem_get_info()[0]);assert q['free_after_load']>6*1024**3;save()
 cfg=json.loads((MODEL/'config.json').read_bytes());factor=float(cfg.get('routed_scaling_factor',2.5));torch.manual_seed(9270);x16=torch.randn((16,6144),device='cuda',dtype=torch.bfloat16)*.1;input_original=x16.clone();weight_sums=[int(w.sum(dtype=torch.int64).item())for w in banks];scale_sums=[float(s.sum().item())for s in scales]
 def allocate(rows,cols):
  base=torch.full((rows*cols+32,),123.,dtype=torch.bfloat16,device='cuda');return base,base[16:-16].view(rows,cols)
 for T in [16]:
  x=x16[:T].contiguous();routes=T*8;scores=torch.sigmoid(x.float()@router.T);ids=(scores+bias).topk(8,dim=-1,largest=True,sorted=True).indices.contiguous();rr=ids.cpu().flatten().tolist();assert min(rr)>=0 and max(rr)<256 and all(len(set(rr[i:i+8]))==8 for i in range(0,routes,8));assert max(rr)-min(rr)+1>len(set(rr)),'noncontiguousexpertsrequired';chosen=scores.gather(1,ids);routing_weights=(chosen/(chosen.sum(-1,keepdim=True)+1e-20)*factor).reshape(routes,1).contiguous();q.setdefault('routing',[]).append({'T':T,'routes':routes,'U':len(set(rr)),'max_rows_per_expert':max(rr.count(i)for i in set(rr)),'expert_min':min(rr),'expert_max':max(rr),'IDs_SHA256':hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()})
  stage_records=[]
  def call(instrument=False):
   from contextlib import contextmanager
   @contextmanager
   def stage(name):
    if not instrument:
     yield;return
    begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);tick=time.perf_counter();begin.record()
    with torch.profiler.record_function('GLM_STAGE/'+name):yield
    end.record();stage_records.append((name,begin,end,time.perf_counter()-tick))
   with stage('input_quant'):
    packed,xs=quant(x)
   with stage('gateup_prepare'):
    gb,g=allocate(routes,256);ub,u=allocate(routes,256);st=V(torch.cuda.current_stream().cuda_stream)
   with stage('gate_up_provider'):
    assert lib.glm_component_gate_r9270(0,ptr(g),ptr(u),ptr(packed),ptr(xs),ptr(ids),ptr(banks[0]),ptr(scales[0]),ptr(banks[1]),ptr(scales[1]),routes,0,st)==0
   with stage('silu_mul'):
    silu=torch.nn.functional.silu(g).to(torch.bfloat16);act=(silu*u).to(torch.bfloat16)
   with stage('down_quant'):
    dp,ds=quant(act)
   with stage('down_prepare'):
    db,d=allocate(routes,6144)
   with stage('down_provider'):
    assert lib.glm_component_down_r9270(0,ptr(d),ptr(dp),ptr(ds),ptr(ids),ptr(banks[2]),ptr(scales[2]),routes,0,st)==0
   with stage('weighted_reduce'):
    reduced=(d.float()*routing_weights).view(T,8,6144).sum(1)
   return g,u,d,reduced,[gb,ub,db]
  def setmode(m):
   torch.cuda.synchronize();assert lib.modern_glm_irregular_vector_policy_r9270(0,m)==0;assert lib.modern_glm_irregular_vector_get_r9270(0)==m and lib.modern_glm_vector_get_r8540(0)==1
  setmode(0);before=stats()
  for _ in range(2):last=call()
  torch.cuda.synchronize();q['warm']=[{'T':16,'mode':0,'calls':2,'counts_delta':[v-u for u,v in zip(before,stats())]}];q['measurements']=[]
  reference=[v.clone()for v in last[:4]]
  def measure(kind,instrument=False,trace=False):
   torch.cuda.synchronize();start_counts=stats();stage_records.clear();start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
   def run():
    tick=time.perf_counter();start.record();z=call(instrument);end.record();end.synchronize();return z,time.perf_counter()-tick
   tracefile=None
   if trace:
    tracefile=R/'results'/('GLM-T16-chain-r9276-'+step+'.trace.json')
    assert not tracefile.exists()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=False,profile_memory=False,with_stack=False)as prof:
     with torch.profiler.record_function('GLM_CHAIN_DIAGNOSTIC'):last,wall=run()
    prof.export_chrome_trace(str(tracefile))
   else:last,wall=run()
   delta=[v-u for u,v in zip(start_counts,stats())];assert delta==[1,0,1,0,0,16,16,0],delta
   assert all(bool(torch.isfinite(a).all())for a in last[:4]);assert all(bool((torch.cat([a[:16],a[-16:]])==123).all())for a in last[4]);quality=[]
   for aa,bb in zip(last[:4],reference):
    rel=float(torch.sqrt(torch.mean((aa.float()-bb.float())**2)).item())/max(float(torch.sqrt(torch.mean(bb.float()**2)).item()),1e-6);assert rel<=.02;quality.append({'relative_RMS':rel,'max_abs':float((aa.float()-bb.float()).abs().max().item())})
   row={'kind':kind,'wall_seconds':wall,'stream_span_seconds':start.elapsed_time(end)/1000.,'actual_provider_counts_delta':delta,'finite':True,'output_canary':True,'quality':quality,'stage_spans':[{'name':n,'host_enqueue_seconds':h,'stream_span_seconds':b.elapsed_time(e)/1000.}for n,b,e,h in stage_records]}
   if tracefile:
    from trace_summary_r9276 import summarize
    row['activity']=summarize(json.loads(tracefile.read_bytes()));row['trace_path']=str(tracefile);row['trace_SHA256']=sha(tracefile)
   q['measurements'].append(row);save()
  for _ in range(6):measure('uninstrumented_before')
  for _ in range(3):measure('events_and_labels',True)
  measure('kineto_one_query',True,True)
  for _ in range(6):measure('uninstrumented_after')
  controls=[z['wall_seconds']for z in q['measurements']if z['kind'].startswith('uninstrumented')];q['uninstrumented_median_seconds']=statistics.median(controls);q['stage_profile_overhead_ratio']=statistics.median([z['wall_seconds']for z in q['measurements']if z['kind']=='events_and_labels'])/q['uninstrumented_median_seconds']-1;q['diagnostic_scope']='T16 only; no candidate/math change; perstage events are stream spans, kineto kernels interval union separately; profiled arms NOT faster-variant claims';q['activity_available']=bool([z for z in q['measurements']if z.get('activity',{}).get('kernel_count',0)>0]);save()
 assert torch.equal(x16,input_original);assert [int(w.sum(dtype=torch.int64).item())for w in banks]==weight_sums;assert [float(s.sum().item())for s in scales]==scale_sums;q.update(input_readonly=True,weight_readonly_full_byte_sum=True,scale_readonly_sum=True,actual_counter_errors=stats()[4]);assert q['actual_counter_errors']==0;assert lib.modern_glm_irregular_vector_policy_r9270(0,0)==0 and lib.modern_glm_vector_policy_r8540(0,0)==0;q.update(policy_closed=True,counts_after=stats(),completed_math=True);save();return q
if __name__=='__main__':
 try:
  q=main();gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();q.update(completed=True,normal=True,free_after_release=torch.cuda.mem_get_info()[0]);save();print('GLM_T16_CHAIN_R9276_NORMAL',flush=True)
 except BaseException as e:
  REPORT.update(error=repr(e),traceback=traceback.format_exc(),completed=False);save();raise
