"""Finite real layer38 projection fixture; synthetic packed activations; no Model."""
import pathlib,ctypes,json,os,time,hashlib,statistics,contextlib,traceback
import torch,torch.multiprocessing as mp
from safetensors import safe_open
r=pathlib.Path('/refactor');model=pathlib.Path('/old/models/ZhipuAI-GLM-5.3');H=r/'candidates/glm-grouped-native-model-r5117/helpers/libglm_fp8_batched_r776.so';HS='e32b4dc2749b4012efbf4322d334b6460dd091190bca78bb8f41e4384f09a7e8';W=r/'candidates/glm-helper-stage-trace-r8982b/libglm_helper_stage_trace_r8982b.so';V=ctypes.c_void_p;Z=ctypes.c_size_t;I=ctypes.c_int
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def worker(rank):
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False;torch.backends.cuda.matmul.allow_fp16_accumulation=False;torch.cuda.set_device(rank);cpus=sorted(os.sched_getaffinity(0));assert len(cpus)>=64;os.sched_setaffinity(0,cpus[rank*8:rank*8+8]);stream=torch.cuda.current_stream();lib=ctypes.CDLL(str(H));wrap=ctypes.CDLL(str(W));lib.glm_fp8_create_r776.argtypes=[I,I]+[Z]*7+[V];lib.glm_fp8_create_r776.restype=V;lib.glm_fp8_run_r776.argtypes=[V,I]+[V]*5;lib.glm_fp8_destroy_r776.argtypes=[V];lib.glm_fp8_error_r776.restype=ctypes.c_char_p;wrap.glm_helper_trace_init_r8982.argtypes=[V,V];wrap.glm_helper_trace_begin_r8982.argtypes=[I]+[Z]*5;wrap.glm_helper_trace_end_r8982.argtypes=[V,V]
 p=r/'results'/f'GLM-helper-stage-trace-r8982-{os.environ["SLURM_JOB_ID"]}.{os.environ["SLURM_STEP_ID"]}-rank{rank}.json';assert not p.exists();q={'completed':False,'rank':rank,'helper_SHA256':HS,'wrapper_SHA256':sha(W),'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'node':os.uname().nodename,'scope':'Layer38 real FP8 weights; synthetic replicated seed8703 BF16 hidden/packed activation, not current540 activation capture. Full helper wall includes ownerstream drain. Events observer only, all original math/guards unchanged.','kernel_name':'UNKNOWN_NO_EXISTING_SOLUTION_LOGGING','cases':[]};plan=None;initialized=False
 def save():temp=p.with_suffix('.tmp');temp.write_text(json.dumps(q,indent=2));temp.replace(p)
 def tck(x,phase):
  if x:raise RuntimeError(('r8982 observer',phase,x))
 def ck(x):
  if x:raise RuntimeError((x,lib.glm_fp8_error_r776().decode()))
 try:
  assert sha(H)==HS;proof=json.loads((W.parent/'CPU-build.json').read_text());assert proof['completed']and proof['GPU']==False and proof['compiled_TUs']==1 and proof['library_SHA256']==sha(W)
  tck(wrap.glm_helper_trace_init_r8982(ctypes.cast(lib.glm_fp8_run_r776,V),V(stream.cuda_stream)),'init');initialized=True
  mapping=json.loads((model/'model.safetensors.index.json').read_text())['weight_map'];weights={};started=time.monotonic()
  specs=[('o_proj',6144,2048,0,rank*2048)]
  with contextlib.ExitStack()as stack:
   handles={}
   for name,n,k,ro,co in specs:
    items=[]
    for tail in ['weight','weight_scale_inv']:
     key=f'model.layers.38.self_attn.{name}.{tail}';fn=mapping[key]
     if fn not in handles:handles[fn]=stack.enter_context(safe_open(str(model/fn),framework='pt',device='cpu'))
     t=handles[fn].get_tensor(key)
     if tail=='weight':t=t.view(torch.uint8)[ro:ro+n,co:co+k].contiguous()
     items.append(t.cuda())
    weights[name]=items
  stream.synchronize();q['single_checkpoint_attention_load_seconds']=time.monotonic()-started;q['checkpoint_read_passes']=1;save()
  gen=torch.Generator(device='cpu').manual_seed(8703)
  for name,n,k,ro,co in specs:
   w,ws=weights[name];sr,sc=ws.shape;assert tuple(w.shape)==(n,k);hidden=(torch.randn(8,k,generator=gen)*.1).to(torch.bfloat16).cuda();blocks=hidden.float().view(8,k//128,128);local_scale=blocks.abs().amax(-1).clamp_min(1e-10)/448.;x=(blocks/local_scale[:,:,None]).clamp(-448,448).to(torch.float8_e4m3fn).view(torch.uint8).reshape(8,k).contiguous();xs=torch.full((8,sc),float('nan'),device='cuda');xs[:,co//128:co//128+k//128]=local_scale;guard=torch.full((8*n+128,),123,dtype=torch.bfloat16,device='cuda');out=guard[64:-64].view(8,n);stream.synchronize();plan=lib.glm_fp8_create_r776(rank,1,8,n,k,sr,sc,ro,co,V(stream.cuda_stream));assert plan;rows=[];references={};first=None
   for index,mode in enumerate([0,1,0,1,1,0,1,0,0,1,0]):
    tck(wrap.glm_helper_trace_begin_r8982(mode,n,k,sc,ro,co),'begin');stream.synchronize();tick=time.perf_counter();cpu=time.thread_time();ck(lib.glm_fp8_run_r776(plan,1,V(out.data_ptr()),V(x.data_ptr()),V(xs.data_ptr()),V(w.data_ptr()),V(ws.data_ptr())));stream.synchronize();host=(time.perf_counter()-tick)*1000;cpu=(time.thread_time()-cpu)*1000;events=(ctypes.c_double*4)();hits=(ctypes.c_uint64*4)();code=wrap.glm_helper_trace_end_r8982(events,hits);q['last_trace']={'mode':mode,'code':code,'hits':list(hits)};save();tck(code,'end');actual=out.cpu();references.setdefault(mode,actual.clone());first=actual.clone()if first is None else first;finite=bool(torch.isfinite(actual.float()).all());canary=bool((guard[:64]==123).all()and(guard[-64:]==123).all());assert finite and canary and list(hits)==([1]*4 if mode else[0]*4)
    rows.append({'index':index,'mode':mode,'host_wall_including_streamdrain_ms':host,'threadCPU_ms':cpu,'stage_ms':dict(zip(['unpackX','unpackW','batchedGEMM','combine'],events))if mode else None,'stage_hits':list(hits),'finite':finite,'canary':canary,'crossarm_exact_diagnostic':bool(torch.equal(actual.view(torch.int16),first.view(torch.int16))),'ownrepeat_exact_diagnostic':bool(torch.equal(actual.view(torch.int16),references[mode].view(torch.int16))),'maxabs_vs_first':float((actual.float()-first.float()).abs().max())})
   ck(lib.glm_fp8_destroy_r776(plan));plan=None;med={str(mode):statistics.median(t['host_wall_including_streamdrain_ms']for t in rows[2:10]if t['mode']==mode)for mode in [0,1]};q['cases'].append({'projection':name,'M':8,'N':n,'K':k,'row_offset':ro,'column_offset':co,'scale_shape':[sr,sc],'trials':rows,'wall_median_ms':med,'observer_wall_delta_percent':100*(med['1']/med['0']-1),'stage_median_ms':{s:statistics.median(t['stage_ms'][s]for t in rows[2:10]if t['mode'])for s in ['unpackX','unpackW','batchedGEMM','combine']}});save()
  stream.synchronize();tck(wrap.glm_helper_trace_close_r8982(),'close');initialized=False;check=torch.ones(32,device='cuda');assert (check+1).sum().item()==64;stream.synchronize();q.update(completed=True,plans_destroyed=True,normal_stream_drain=True,Torch_postclose=True,Model_adoption=False);save()
 except BaseException as exc:q.update(error=repr(exc),traceback=traceback.format_exc());save();raise
 finally:
  if plan:stream.synchronize();q['finally_destroy']=lib.glm_fp8_destroy_r776(plan)
  if initialized:stream.synchronize();q['finally_close']=wrap.glm_helper_trace_close_r8982()
  save()
if __name__=='__main__':assert torch.cuda.device_count()==8;mp.spawn(worker,nprocs=8,join=True)
