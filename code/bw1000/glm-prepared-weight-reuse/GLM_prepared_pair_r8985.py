"""Finite real layer38 projection fixture; synthetic packed activations; no Model."""
import pathlib,ctypes,json,os,time,hashlib,statistics,contextlib,traceback
# This is a host precheck IN THE GPU JOB, before Torch/weights, not CPU-only proof.
_PRE_H=pathlib.Path('/refactor/candidates/glm-grouped-native-model-r5117/helpers/libglm_fp8_batched_r776.so');_PRE_W=pathlib.Path('/refactor/candidates/glm-helper-stage-trace-r8982c/libglm_helper_stage_trace_r8982c.so')
_PRE_PROOF=json.loads((_PRE_W.parent/'CPU-build.json').read_text());assert _PRE_PROOF['completed'] and hashlib.sha256(_PRE_W.read_bytes()).hexdigest()==_PRE_PROOF['library_SHA256'];assert hashlib.sha256(_PRE_H.read_bytes()).hexdigest()=='e32b4dc2749b4012efbf4322d334b6460dd091190bca78bb8f41e4384f09a7e8'
_PRE_HOLD=ctypes.CDLL(str(_PRE_H));_PRE_WRAP=ctypes.CDLL(str(_PRE_W));_PRE_WRAP.glm_helper_trace_resolution_r8982.restype=ctypes.c_char_p;_PRE_CODE=int(_PRE_WRAP.glm_helper_trace_precheck_r8982());_PRE_INFO={'scope':'host symbol/loadorder precheck inside own GPUallocation; no explicitCUDA/deviceAPI call, no Torchimport/weights yet; dependencyconstructors notclaimed sideeffectfree','pid':os.getpid(),'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'code':_PRE_CODE,'resolution':_PRE_WRAP.glm_helper_trace_resolution_r8982().decode()}
_PRE_PATH=pathlib.Path('/refactor/results')/f'GLM-helper-host-precheck-r8982c-{_PRE_INFO["producer"]}-{os.getpid()}.json'
with _PRE_PATH.open('x')as _f:json.dump(_PRE_INFO,_f,indent=2)
assert _PRE_CODE==0,('host_resolve_STOP_beforeTorch',_PRE_INFO)

_P=pathlib.Path('/refactor/candidates/glm-prepared-weight-r2040/libglm_prepared_r2040.so')
_PS='a358bd83e41f0a9afda5db16eb335b20866ef71bd6f18a1d00775edd54ff0885'
assert hashlib.sha256(_P.read_bytes()).hexdigest()==_PS
_PH=ctypes.CDLL(str(_P))
class _DlInfo(ctypes.Structure):
 _fields_=[('fname',ctypes.c_char_p),('base',ctypes.c_void_p),('sname',ctypes.c_char_p),('saddr',ctypes.c_void_p)]
_dl=ctypes.CDLL('libdl.so.2');_dl.dladdr.argtypes=[ctypes.c_void_p,ctypes.POINTER(_DlInfo)];_origins={}
for _label,_lib,_path,_names in [('original',_PRE_HOLD,_PRE_H,['glm_fp8_create_r776','glm_fp8_run_r776','glm_fp8_destroy_r776','glm_fp8_error_r776']),('prepared',_PH,_P,['glm_fp8_prepared_create_r2040','glm_fp8_prepared_run_r2040','glm_fp8_destroy_r776','glm_fp8_error_r776'])]:
 _rows=[]
 for _name in _names:
  _d=_DlInfo();assert _dl.dladdr(ctypes.cast(getattr(_lib,_name),ctypes.c_void_p),ctypes.byref(_d));assert pathlib.Path(_d.fname.decode()).resolve()==_path.resolve(),(_label,_name,_d.fname)
  _rows.append({'symbol':_name,'provider':_d.fname.decode(),'base':int(_d.base),'address':int(ctypes.cast(getattr(_lib,_name),ctypes.c_void_p).value)})
 assert len(set(a['base']for a in _rows))==1
 _origins[_label]=_rows
assert _origins['original'][0]['base']!=_origins['prepared'][0]['base']
_PRE_INFO['prepared_distinct_origins']=_origins;_PRE_INFO['prepared_SHA256']=_PS
_PRE_PATH=pathlib.Path('/refactor/results')/f'GLM-prepared-host-precheck-r8985-{_PRE_INFO["producer"]}-{os.getpid()}.json'
with _PRE_PATH.open('x')as _f:json.dump(_PRE_INFO,_f,indent=2)

import torch,torch.multiprocessing as mp
from safetensors import safe_open
r=pathlib.Path('/refactor');model=pathlib.Path('/old/models/ZhipuAI-GLM-5.3');H=r/'candidates/glm-grouped-native-model-r5117/helpers/libglm_fp8_batched_r776.so';HS='e32b4dc2749b4012efbf4322d334b6460dd091190bca78bb8f41e4384f09a7e8';W=r/'candidates/glm-helper-stage-trace-r8982c/libglm_helper_stage_trace_r8982c.so';V=ctypes.c_void_p;Z=ctypes.c_size_t;I=ctypes.c_int
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def worker(rank):
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False;torch.backends.cuda.matmul.allow_fp16_accumulation=False;torch.cuda.set_device(rank);cpus=sorted(os.sched_getaffinity(0));assert len(cpus)>=64;os.sched_setaffinity(0,cpus[rank*8:rank*8+8]);stream=torch.cuda.current_stream();lib=ctypes.CDLL(str(H));wrap=ctypes.CDLL(str(W));lib.glm_fp8_create_r776.argtypes=[I,I]+[Z]*7+[V];lib.glm_fp8_create_r776.restype=V;lib.glm_fp8_run_r776.argtypes=[V,I]+[V]*5;lib.glm_fp8_destroy_r776.argtypes=[V];lib.glm_fp8_error_r776.restype=ctypes.c_char_p;wrap.glm_helper_trace_init_r8982.argtypes=[V,V];wrap.glm_helper_trace_begin_r8982.argtypes=[I]+[Z]*5;wrap.glm_helper_trace_end_r8982.argtypes=[V,V]
 p=r/'results'/f'GLM-prepared-pair-r8985-{os.environ["SLURM_JOB_ID"]}.{os.environ["SLURM_STEP_ID"]}-rank{rank}.json';assert not p.exists();q={'prepared_SHA256':_PS,'completed':False,'rank':rank,'helper_SHA256':HS,'wrapper_SHA256':sha(W),'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'node':os.uname().nodename,'scope':'Layer38 real FP8 weights; synthetic replicated seed8703 BF16 hidden/packed activation, not current540 activation capture. Full helper wall includes ownerstream drain. Both arms observerOFF; original776 mode1 versus existing2040 prepared mode2, same original BMM/scales, no Model.','kernel_name':'UNKNOWN_NO_EXISTING_SOLUTION_LOGGING','host_symbol_precheck':_PRE_INFO,'cases':[]};plan=None;prepared=None;initialized=False
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
   w,ws=weights[name];sr,sc=ws.shape;assert tuple(w.shape)==(n,k);hidden=(torch.randn(8,k,generator=gen)*.1).to(torch.bfloat16).cuda();blocks=hidden.float().view(8,k//128,128);local_scale=blocks.abs().amax(-1).clamp_min(1e-10)/448.;x=(blocks/local_scale[:,:,None]).clamp(-448,448).to(torch.float8_e4m3fn).view(torch.uint8).reshape(8,k).contiguous();xs=torch.full((8,sc),float('nan'),device='cuda');xs[:,co//128:co//128+k//128]=local_scale;guard=torch.full((8*n+128,),123,dtype=torch.bfloat16,device='cuda');out=guard[64:-64].view(8,n);stream.synchronize();plan=lib.glm_fp8_create_r776(rank,1,8,n,k,sr,sc,ro,co,V(stream.cuda_stream));assert plan
   plib=ctypes.CDLL(str(_P));plib.glm_fp8_prepared_create_r2040.argtypes=[I]+[Z]*8+[V,Z,V];plib.glm_fp8_prepared_create_r2040.restype=V;plib.glm_fp8_prepared_run_r2040.argtypes=[V,I]+[V]*5+[Z];plib.glm_fp8_destroy_r776.argtypes=[V];plib.glm_fp8_error_r776.restype=ctypes.c_char_p
   def pck(code):
    if code:raise RuntimeError(('prepared2040',code,plib.glm_fp8_error_r776().decode()))
   stream.synchronize();_started=time.perf_counter();prepared=plib.glm_fp8_prepared_create_r2040(rank,8,8,n,k,sr,sc,ro,co,V(w.data_ptr()),1,V(stream.cuda_stream));assert prepared,plib.glm_fp8_error_r776().decode();stream.synchronize();q['prepared_create_and_drain_ms']=(time.perf_counter()-_started)*1000
   wrongw=w.clone();sentinel=torch.full_like(out,float('nan'));saved=sentinel.view(torch.int16).clone();stream.synchronize()
   def prun(y=out,weight=w,generation=1):return plib.glm_fp8_prepared_run_r2040(prepared,2,V(y.data_ptr()),V(x.data_ptr()),V(xs.data_ptr()),V(weight.data_ptr()),V(ws.data_ptr()),generation)
   q['wrong_source_code']=prun(sentinel,wrongw,1);q['wrong_generation_code']=prun(sentinel,w,2);stream.synchronize();assert q['wrong_source_code']!=0 and q['wrong_generation_code']!=0 and torch.equal(sentinel.view(torch.int16),saved);q['wrong_source_generation_output_unchanged']=True
   rows=[];references={};first=None
   for index,mode in enumerate([0,1,0,1,1,0,1,0,0,1,0]):
    tck(wrap.glm_helper_trace_begin_r8982(0,n,k,sc,ro,co),'begin');stream.synchronize();tick=time.perf_counter();cpu=time.thread_time();(ck(lib.glm_fp8_run_r776(plan,1,V(out.data_ptr()),V(x.data_ptr()),V(xs.data_ptr()),V(w.data_ptr()),V(ws.data_ptr())))if mode==0 else pck(prun()));stream.synchronize();host=(time.perf_counter()-tick)*1000;cpu=(time.thread_time()-cpu)*1000;events=(ctypes.c_double*4)();hits=(ctypes.c_uint64*4)();code=wrap.glm_helper_trace_end_r8982(events,hits);q['last_trace']={'mode':mode,'code':code,'hits':list(hits)};save();tck(code,'end');actual=out.cpu();references.setdefault(mode,actual.clone());first=actual.clone()if first is None else first;finite=bool(torch.isfinite(actual.float()).all());canary=bool((guard[:64]==123).all()and(guard[-64:]==123).all());assert finite and canary and list(hits)==[0]*4
    rows.append({'index':index,'mode':mode,'host_wall_including_streamdrain_ms':host,'threadCPU_ms':cpu,'stage_ms':None,'stage_hits':list(hits),'finite':finite,'canary':canary,'crossarm_exact_diagnostic':bool(torch.equal(actual.view(torch.int16),first.view(torch.int16))),'ownrepeat_exact_diagnostic':bool(torch.equal(actual.view(torch.int16),references[mode].view(torch.int16))),'maxabs_vs_first':float((actual.float()-first.float()).abs().max()),'RMSE_vs_first':float((actual.float()-first.float()).square().mean().sqrt()),'relative_RMS_vs_first':float((actual.float()-first.float()).square().mean().sqrt()/first.float().square().mean().sqrt().clamp_min(1e-30))})
   pck(plib.glm_fp8_destroy_r776(prepared));prepared=None;ck(lib.glm_fp8_destroy_r776(plan));plan=None;med={str(mode):statistics.median(t['host_wall_including_streamdrain_ms']for t in rows[2:10]if t['mode']==mode)for mode in [0,1]};q['cases'].append({'projection':name,'M':8,'N':n,'K':k,'row_offset':ro,'column_offset':co,'scale_shape':[sr,sc],'trials':rows,'wall_median_ms':med,'candidate_wall_delta_percent':100*(med['1']/med['0']-1),'observer_enabled':False,'arm0':'original776_mode1','arm1':'prepared2040_mode2','saving_ms':med['0']-med['1'],'prepare_break_even_calls':q['prepared_create_and_drain_ms']/(med['0']-med['1'])if med['0']>med['1']else None});save()
  stream.synchronize();tck(wrap.glm_helper_trace_close_r8982(),'close');initialized=False;check=torch.ones(32,device='cuda');assert (check+1).sum().item()==64;stream.synchronize();q.update(completed=True,plans_destroyed=True,normal_stream_drain=True,Torch_postclose=True,Model_adoption=False);save()
 except BaseException as exc:q.update(error=repr(exc),traceback=traceback.format_exc());save();raise
 finally:
  if prepared:stream.synchronize();q['finally_prepared_destroy']=plib.glm_fp8_destroy_r776(prepared)
  if plan:stream.synchronize();q['finally_destroy']=lib.glm_fp8_destroy_r776(plan)
  if initialized:stream.synchronize();q['finally_close']=wrap.glm_helper_trace_close_r8982()
  save()
if __name__=='__main__':assert torch.cuda.device_count()==8;mp.spawn(worker,nprocs=8,join=True)
