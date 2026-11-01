"""Actual Runtime compressor, checkpoint compressor parameters, synthetic hidden.
Four independent shards: C4/C128 x decode boundary/prefill. No target load.
"""
import os,pathlib,json,hashlib,ctypes as C,time,traceback,torch,math
from safetensors import safe_open
S=pathlib.Path('/refactor/incoming/Deep-Runtime-boundary-fusion-r344');R=pathlib.Path('/refactor');MODEL=pathlib.Path('/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8')
rank=int(os.environ['SLURM_LOCALID']);torch.set_num_threads(1);torch.cuda.set_device(0);stream=C.c_void_p(torch.cuda.current_stream().cuda_stream)
P=C.c_void_p;Z=C.c_size_t;ptr=lambda x:P(x.data_ptr());sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
receipt=json.loads((R/'candidates/Deep344-adjacent-r1/CPU-build.json').read_text());assert receipt['completed'];assert sha(receipt['library'])==receipt['library_SHA256']
lib=C.CDLL(receipt['library']);lib.modern_real_compress_error.restype=C.c_char_p
lib.modern_real_compress_create.argtypes=[C.c_int]+[Z]*4+[C.c_char_p];lib.modern_real_compress_create.restype=P
lib.modern_real_compress_load.argtypes=[P,C.c_char_p,P,Z,P,C.c_int];lib.modern_real_compress_finish.argtypes=[P]
lib.modern_real_compress_forward.argtypes=[P,Z,Z]+[P]*6+[C.c_int,P]
lib.modern_real_compress_snapshot.argtypes=[P,C.c_int,P,P];lib.modern_real_compress_bytes.argtypes=[P,C.c_int];lib.modern_real_compress_bytes.restype=Z
lib.modern_real_compress_reset.argtypes=[P];lib.modern_real_compress_destroy.argtypes=[P];lib.modern_real_compress_release_workers.argtypes=[]
lib.modern_real_compress_timing_r8694.argtypes=[C.POINTER(C.c_double),Z];lib.modern_deep344_mode.argtypes=[C.c_int];lib.modern_deep344_calls.argtypes=[C.c_int];lib.modern_deep344_calls.restype=C.c_uint64
def ck(z):
 if z:raise RuntimeError(lib.modern_real_compress_error().decode())
ratio=4 if rank<2 else 128;kind='decode_boundary'if rank%2==0 else'prefill';past=127 if ratio==4 else 255;T=1 if kind=='decode_boundary'else 2048
cfg=json.loads((MODEL/'config.json').read_text());layer=next(i for i,n in enumerate(cfg['compress_ratios'])if n==ratio);index=json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map']
report=dict(completed=False,rank=rank,actual_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],scope='Actual Runtime full compressor prepare+commit+same bridge output copies/drain. Real checkpoint compressor weights; synthetic hidden; not Model quality/speed.',ratio=ratio,layer=layer,kind=kind,T=T,past=past,library_SHA256=receipt['library_SHA256'],records=[],weights={},GPU_F64=False)
out=R/'results'/f'Deep344adj1-{report["actual_step"]}-rank{rank}.json';assert not out.exists();context=None;start=time.monotonic()
def save():out.write_bytes((json.dumps(report,indent=2)+'\n').encode())
def digest(x):return hashlib.sha256(x.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def snap(cache):
 x=torch.empty(lib.modern_real_compress_bytes(context,cache),dtype=torch.uint8,device='cuda');ck(lib.modern_real_compress_snapshot(context,cache,ptr(x),stream));return x.cpu()
def forward(x,pos,past):
 n=len(x);meta=(Z*3)(0,past,n);score=torch.full((n,(4 if ratio==4 else 2)*512),float('nan'),dtype=torch.bfloat16,device='cuda');pooled=torch.full((n,512),float('nan'),dtype=torch.bfloat16,device='cuda');rot=torch.full_like(pooled,float('nan'));torch.cuda.synchronize()
 a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();t=time.perf_counter();ck(lib.modern_real_compress_forward(context,n,1,meta,ptr(x),ptr(pos),ptr(score),ptr(pooled),ptr(rot),0,stream));wall=time.perf_counter()-t;b.record();b.synchronize();slot=(C.c_double*3)();ck(lib.modern_real_compress_timing_r8694(slot,3));return [score.cpu(),pooled.cpu(),rot.cpu()],dict(full_CAPI_wall_s=wall,native_wall_s=slot[0],producer_stream_event_ms=a.elapsed_time(b),event_scope='Torch producer-stream envelope includes host call and cross-Runtime waits, NOT kernel busy or Runtime-stream interval')
try:
 weights={}
 for n in ['wkv.weight','wgate.weight','ape','norm.weight']:
  key=f'layers.{layer}.attn.compressor.{n}'
  with safe_open(MODEL/index[key],framework='pt',device='cpu')as f:w=f.get_tensor(key).contiguous()
  weights[n]=w;report['weights'][n]=dict(key=key,shape=list(w.shape),dtype=str(w.dtype),tensor_SHA256=digest(w))
 context=lib.modern_real_compress_create(0,layer,512,1,(past+T)//ratio+1,json.dumps(cfg).encode());assert context,lib.modern_real_compress_error()
 for n,w in weights.items():shape=(Z*w.ndim)(*w.shape);ck(lib.modern_real_compress_load(context,n.encode(),ptr(w),w.ndim,shape,int(w.dtype==torch.float32)))
 ck(lib.modern_real_compress_finish(context));torch.manual_seed(344+rank)
 seed=torch.randn((past,4096),dtype=torch.bfloat16,device='cuda')*.1;x=torch.randn((T,4096),dtype=torch.bfloat16,device='cuda')*.1
 seedpos=torch.arange(past,dtype=torch.int64,device='cuda');pos=torch.arange(past,past+T,dtype=torch.int64,device='cuda');input_before=digest(x);positions_before=digest(pos)
 def reference(outputs):
  ids=[i for i in range(T)if(past+i+1)%ratio==0];positions=[(past+i)//ratio*ratio for i in ids];assert positions and min(positions)>0
  selected=outputs[1][ids].float();normalized=(selected*torch.rsqrt(selected.square().mean(-1,keepdim=True)+float(cfg['rms_norm_eps']))*weights['norm.weight'].float()).bfloat16()
  theta=float(cfg['compress_rope_theta']);rs=cfg['rope_scaling'];factor=float(rs['factor']);original=int(rs['original_max_position_embeddings']);beta_fast=float(rs['beta_fast']);beta_slow=float(rs['beta_slow']);low=high=0.
  if original and factor!=1:
   corr=lambda rotations:64*math.log(original/(rotations*2*math.pi))/(2*math.log(theta));low=max(0.,min(63.,math.floor(corr(beta_fast))));high=max(0.,min(63.,math.ceil(corr(beta_slow))));high=high+.001 if low==high else high
  table=[]
  for position in positions:
   angles=[]
   for i in range(32):
    freq=1/math.pow(theta,2*i/64)
    if original and factor!=1:
     smooth=1-max(0.,min(1.,(i-low)/(high-low)));freq=freq/factor*(1-smooth)+freq*smooth
    angles.append(position*freq)
   table.append([math.cos(v)for v in angles]+[math.sin(v)for v in angles])
  table=torch.tensor(table,dtype=torch.float32);expected=normalized.clone();a=normalized[:,-64::2].float();b=normalized[:,-63::2].float();expected[:,-64::2]=(a*table[:,:32]-b*table[:,32:]).bfloat16();expected[:,-63::2]=(b*table[:,:32]+a*table[:,32:]).bfloat16();got=outputs[2][ids];delta=got.float()-expected.float();nr=float(delta.square().mean().sqrt()/(expected.float().square().mean().sqrt()+1e-12));assert nr<.01,nr
  return dict(nonzero_rotary_positions=[min(positions),max(positions)],reference_NRMSE=nr,reference_max_abs=float(delta.abs().max()),BF16_intermediate_unrotated448_max_abs=float((got[:,:448].float()-normalized[:,:448].float()).abs().max()),BF16_round_before_tail_rope=True,CPU_reference_F32=True)
 def trial(mode):
  ck(lib.modern_deep344_mode(0));ck(lib.modern_real_compress_reset(context));forward(seed,seedpos,0);state0=snap(0);ck(lib.modern_deep344_mode(mode));before=lib.modern_deep344_calls(mode);outputs,timing=forward(x,pos,past);assert lib.modern_deep344_calls(mode)==before+1,'actual postprocess boundary must execute';state1=snap(0)
  assert all(bool(torch.isfinite(z).all())for z in outputs),'poison or nonfinite output';assert digest(x)==input_before and digest(pos)==positions_before
  return outputs,state0,state1,timing
 baseline_first=trial(0);report['baseline_reference_before_any_candidate']=reference(baseline_first[0]);save()
 for mode in [0,1,1,0]:trial(mode)
 gold=None
 for group in range(5):
  for arm,mode in enumerate([0,1,1,0]):
   outputs,state0,state1,timing=trial(mode)
   if gold is None:gold=(outputs,state0,state1)
   assert torch.equal(state0,gold[1])and torch.equal(state1,gold[2]),'source state canary differs'
   numeric=[]
   for got,ref in zip(outputs,gold[0]):
    d=got.float()-ref.float();numeric.append(dict(max_abs=float(d.abs().max()),nrmse=float(d.square().mean().sqrt()/(ref.float().square().mean().sqrt()+1e-12))))
   assert max(z['nrmse']for z in numeric)<.01,numeric
   refcheck=reference(outputs)
   report['records'].append(dict(reference=refcheck,group=group,arm=arm,mode=mode,timing=timing,numerics_score_pooled_rotated=numeric,state_canary=True));save()
 report.update(completed=True,nonzero_rotary_position=True,intermediate_BF16='CPU F32 reference explicitly rounds normalized*weight to BF16 before exact config/position tail64 table rotation; unrotated448 intermediate channels checked separately',baseline_fused_counters=[lib.modern_deep344_calls(0),lib.modern_deep344_calls(1)])
except BaseException as e:report.update(error=repr(e),traceback=traceback.format_exc());raise
finally:
 if context:ck(lib.modern_real_compress_destroy(context))
 ck(lib.modern_real_compress_release_workers());report['elapsed_s']=time.monotonic()-start;save();print(json.dumps({k:report.get(k)for k in ['rank','completed','elapsed_s','error']}),flush=True)
