import pathlib,json,os,ctypes,time,hashlib,contextlib,traceback,statistics
import torch,torch.multiprocessing as mp
from safetensors import safe_open
r=pathlib.Path('/refactor');model=pathlib.Path('/old/models/ZhipuAI-GLM-5.3');base=r/'candidates/glm-prefill-split-r8996';V=ctypes.c_void_p;Z=ctypes.c_size_t;I=ctypes.c_int;U=ctypes.c_uint64
def worker(rank):
 torch.set_num_threads(1);torch.cuda.set_device(rank);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
 for n in ['allow_fp16_reduced_precision_reduction','allow_bf16_reduced_precision_reduction','allow_fp16_accumulation']:setattr(torch.backends.cuda.matmul,n,False)
 cpus=sorted(os.sched_getaffinity(0));assert len(cpus)>=64;os.sched_setaffinity(0,cpus[rank*8:rank*8+8]);p=r/'results'/f'GLM-prefill-split-r8996-{os.environ["SLURM_JOB_ID"]}.{os.environ["SLURM_STEP_ID"]}-rank{rank}.json';assert not p.exists();q={'rank':rank,'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'completed':False,'trials':[],'Model_adoption':False,'scope':'Original5248 syntheticT512 seed5095000+rank+512, real firstsparse layer3 banks/router. Freeze current5386 upstream gate/activation/quant and routeIDs once. Profile complete down provider only; no sharedexpert/CCL/Model.'};lib=None;initialized=False
 def save():tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(q,indent=2));tmp.replace(p)
 def ck(code,label):
  if code:raise RuntimeError((label,code,lib.glm_grouped_error_r5123().decode()if lib else ''))
 def ptr(t):return V(t.data_ptr())
 try:
  proof=json.loads((base/'CPU-build.json').read_text());library=pathlib.Path(proof['library']);assert proof['completed']and not proof['GPU']and not proof['dlopen']and proof['compiled_TUs']==1 and hashlib.sha256(library.read_bytes()).hexdigest()==proof['library_SHA256'];q['DSO_SHA256']=proof['library_SHA256'];lib=ctypes.CDLL(str(library));lib.glm_grouped_error_r5123.restype=ctypes.c_char_p;lib.glm_grouped_gate_r5123.argtypes=[V]*9+[Z,Z,V];lib.glm_grouped_down_r5123.argtypes=[V]*6+[Z,Z,V];lib.glm_grouped_counts_r5123.argtypes=[I,ctypes.POINTER(U),Z];lib.glm_certified_stats_r5248.argtypes=[V,Z];lib.glm_prefill_trace_init_r8996.argtypes=[I,V,Z];lib.glm_prefill_trace_begin_r8996.argtypes=[I];lib.glm_prefill_trace_end_r8996.argtypes=[ctypes.POINTER(ctypes.c_double),ctypes.POINTER(Z),ctypes.POINTER(ctypes.c_double)];ck(lib.glm_prefill_trace_resolve_r8996(),'existingcudart resolver')
  class DlInfo(ctypes.Structure):_fields_=[('name',ctypes.c_char_p),('base',V),('symbol',ctypes.c_char_p),('addr',V)]
  dl=ctypes.CDLL('libdl.so.2');dl.dladdr.argtypes=[V,ctypes.POINTER(DlInfo)];q['symbol_origins']={}
  for n in ['glm_grouped_gate_r5123','glm_grouped_down_r5123','glm_certified_stats_r5248','glm_certified_repair_r5248','glm_prefill_trace_init_r8996','glm_prefill_trace_end_r8996','cudaLaunchKernel']:
   a=DlInfo();assert dl.dladdr(ctypes.cast(getattr(lib,n),V),ctypes.byref(a));assert pathlib.Path(a.name.decode()).resolve()==library.resolve();q['symbol_origins'][n]=a.name.decode()
  cfg=json.loads((model/'config.json').read_text());layer=cfg['mlp_layer_types'].index('sparse');assert layer==3;mapping=json.loads((model/'model.safetensors.index.json').read_text())['weight_map'];banks=[];offset=rank*256;started=time.monotonic()
  with contextlib.ExitStack()as stack:
   handles={}
   def load(key):
    name=mapping[key]
    if name not in handles:handles[name]=stack.enter_context(safe_open(str(model/name),framework='pt',device='cpu'))
    return handles[name].get_tensor(key)
   for proj in ['gate_proj','up_proj','down_proj']:
    weights=[];scales=[]
    for expert in range(256):
     prefix=f'model.layers.{layer}.mlp.experts.{expert}.{proj}';w=load(prefix+'.weight').view(torch.uint8);part=w[:,offset:offset+256]if proj=='down_proj'else w[offset:offset+256,:];weights.append(part.contiguous().clone());scales.append(load(prefix+'.weight_scale_inv').clone())
    banks.extend([torch.stack(weights).cuda(),torch.stack(scales).cuda()]);del weights,scales
   router=load(f'model.layers.{layer}.mlp.gate.weight').float().cuda();bias=load(f'model.layers.{layer}.mlp.gate.e_score_correction_bias').float().cuda()
  torch.cuda.synchronize();q.update(load_seconds=time.monotonic()-started,one_checkpoint_read_pass=True,layer=layer);save()
  def quant(x):
   v=x.float().reshape(x.shape[0],-1,128);scale=v.abs().amax(-1).clamp_min(1e-10)/448.;return(v/scale.unsqueeze(-1)).clamp(-448,448).to(torch.float8_e4m3fn).view(torch.uint8).reshape(x.shape).contiguous(),scale.contiguous()
  os.environ['GLM_GROUPED_ROUTED_R5123']='1';os.environ['GLM_DECODE_ROW_DEDUP_R5386']='0';os.environ['GLM_ROW_TILE_R5206']='8';torch.manual_seed(5095000+rank+512);x=torch.randn((512,6144),device='cuda',dtype=torch.bfloat16);scores=torch.sigmoid(x.float()@router.T);ids=(scores+bias).topk(8,dim=-1).indices;packed,xs=quant(x);g=torch.empty((4096,256),device='cuda',dtype=torch.bfloat16);u=torch.empty_like(g);current=torch.cuda.current_stream();external=torch.cuda.Stream();external.wait_stream(current);stream=V(external.cuda_stream);ck(lib.glm_grouped_gate_r5123(ptr(g),ptr(u),ptr(packed),ptr(xs),ptr(ids),*[ptr(t)for t in banks[:4]],4096,offset,stream),'upstream gate');current.wait_stream(external);activated=(torch.nn.functional.silu(g)*u).to(torch.bfloat16);down,ds=quant(activated);guard=torch.full((4096*6144+128,),123,dtype=torch.bfloat16,device='cuda');out=guard[64:-64].view(4096,6144);external.wait_stream(current);external.synchronize()
  counts=torch.bincount(ids.cpu().flatten(),minlength=256).tolist();items=sorted([min(1024,n-j)for n in counts if n for j in range(0,n,1024)],reverse=True);position=0;buckets=[]
  while position<len(items):
   m=items[position];limit=min(16,max(1,2048//m));end=position+1
   while end<len(items)and end-position<limit and items[end]>=max(1,m//2):end+=1
   buckets.append({'E':end-position,'M':m});position=end
  q.update(T=512,routes=4096,active_experts=sum(n>0 for n in counts),bucket_geometry=buckets,seed=5095000+rank+512,stream_handle=external.cuda_stream,route_ID_SHA256=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest());initialized=True;ck(lib.glm_prefill_trace_init_r8996(rank,stream,len(buckets)+1),'observer init');assert lib.glm_prefill_trace_begin_r8996(99)!=0;input_saved=[v.clone()for v in [down,ds,ids]]
  def counters():
   a=(U*5)();b=(U*4)();ck(lib.glm_grouped_counts_r5123(rank,a,5),'branchstats');ck(lib.glm_certified_stats_r5248(ctypes.cast(b,V),4),'repairstats');return list(a),list(b)
  first=None;own={}
  for index,mode in enumerate([0,1,0,1,1,0,1,0,0,1,0]):
   before,certbefore=counters();ck(lib.glm_prefill_trace_begin_r8996(mode),'observer begin');external.synchronize();started=time.perf_counter();ck(lib.glm_grouped_down_r5123(ptr(out),ptr(down),ptr(ds),ptr(ids),ptr(banks[4]),ptr(banks[5]),4096,offset,stream),'down');external.synchronize();wall=(time.perf_counter()-started)*1000;stages=(ctypes.c_double*3)();details=(ctypes.c_double*13)();used=Z();ck(lib.glm_prefill_trace_end_r8996(stages,ctypes.byref(used),details),'observer end');after,certafter=counters();delta=[b-a for a,b in zip(before,after)];cdelta=[b-a for a,b in zip(certbefore,certafter)];assert delta==[1,0,1,0,0]and cdelta[2]>0 and 0<=cdelta[3]<=cdelta[2]and used.value==(len(buckets)if mode else 0),('branch/sequence',delta,cdelta,used.value,len(buckets));actual=out.cpu();first=actual.clone()if first is None else first;own.setdefault(mode,actual.clone());diff=actual.float()-first.float();od=actual.float()-own[mode].float();finite=bool(torch.isfinite(actual.float()).all());canary=bool((guard[:64]==123).all()and(guard[-64:]==123).all());assert finite and canary
   q['trials'].append({'index':index,'mode':mode,'complete_provider_wall_ms':wall,'stage_spans_ms':dict(zip(['signedBMM','abs_materialize_BMM','bound_compact_repair'],stages))if mode else None,'split_detail':dict(zip(['certify_compact_event_ms','repair_event_ms','interlaunch_event_gap_ms','prefix_event_gap_ms','suffix_event_gap_ms','certify_driver_wall_ms','repair_driver_wall_ms','whole_certificate_CAPI_wall_ms','interlaunch_CPU_gap_ms','driver_launches','certify_driver_threadCPU_ms','repair_driver_threadCPU_ms','whole_certificate_CAPI_threadCPU_ms'],details)),'buckets_recorded':used.value,'branch_delta':delta,'certificate_delta':cdelta,'repair_fraction':cdelta[3]/cdelta[2],'finite':finite,'canary':canary,'ownrepeat_exact_diagnostic':bool(torch.equal(actual.view(torch.int16),own[mode].view(torch.int16))),'crossarm_exact_diagnostic':bool(torch.equal(actual.view(torch.int16),first.view(torch.int16))),'maxabs_vs_first':float(diff.abs().max()),'RMSE_vs_first':float(diff.square().mean().sqrt()),'relative_RMS_vs_first':float(diff.square().mean().sqrt()/first.float().square().mean().sqrt().clamp_min(1e-30)),'ownrepeat_maxabs':float(od.abs().max())});save()
   assert details[9]==2*len(buckets) and q['trials'][-1]['ownrepeat_exact_diagnostic'],('ownrepeat health',q['trials'][-1])
  q['inputs_readonly']=all(torch.equal(a.view(torch.uint8),b.view(torch.uint8))for a,b in zip([down,ds,ids],input_saved));assert q['inputs_readonly'];ck(lib.glm_prefill_trace_close_r8996(),'observer close');initialized=False;torch.cuda.synchronize();assert (torch.ones(32,device='cuda')+1).sum().item()==64;torch.cuda.synchronize();q.update(completed=True,normal_stream_drain=True,observer_closed=True,Torch_postclose=True,Model_adoption=False);save()
 except BaseException as exc:q.update(error=repr(exc),traceback=traceback.format_exc());save();raise
 finally:
  if initialized and lib:
   try:q['finally_close_code']=int(lib.glm_prefill_trace_close_r8996())
   except BaseException as exc:q['finally_close_error']=repr(exc)
   save()
if __name__=='__main__':assert torch.cuda.device_count()==8;mp.spawn(worker,nprocs=8,join=True)
