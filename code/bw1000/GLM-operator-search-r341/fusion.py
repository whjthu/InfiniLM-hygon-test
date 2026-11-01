import ctypes,hashlib,json,os,pathlib,time,statistics
import torch
V=ctypes.c_void_p
LIB=None;VARIANT=None
def setup_fusion(report):
 global LIB,VARIANT
 VARIANT=int(os.environ['SLURM_LOCALID'])+int(os.environ['GLM341_VARIANT_OFFSET']);assert 0<=VARIANT<16
 b=pathlib.Path('/refactor/candidates/GLM-operator-search-r341');p=b/'fused.so';q=json.loads((b/'CPU_RESULT.json').read_text());assert hashlib.sha256(p.read_bytes()).hexdigest()==q['library_SHA256']
 meta=json.loads((pathlib.Path(__file__).parent/'GPU-manifest.json').read_text());assert q['library_SHA256']==meta['fusion_library_SHA256']
 LIB=ctypes.CDLL(str(p));LIB.glm_fused341.argtypes=[V,V,V,V,ctypes.c_int,ctypes.c_int,V]
 report.update(fusion_variant=VARIANT,fusion_lanes=[16,32,64,128][VARIANT//4],fusion_groups=[1,2,4,8][VARIANT%4],fusion_library_SHA256=q['library_SHA256'],performance_scope='Component only; baseline retained r9270b Torch activation/quant chain, not a deployed native activation baseline',numeric_gate={'finite':True,'fullchain_relative_RMS_max':.02,'quant_dequant_relative_RMS_max':.01,'packed_mismatch_fraction_max':.005})
def activation(g,u,mode):
 if not mode:
  act=(torch.nn.functional.silu(g).to(torch.bfloat16)*u).to(torch.bfloat16);y=act.float().reshape(g.shape[0],2,128);sc=y.abs().amax(-1).clamp_min(1e-10)/448.;packed=(y/sc[:,:,None]).clamp(-448,448).to(torch.float8_e4m3fn).view(torch.uint8).reshape(g.shape).contiguous();return packed,sc.contiguous()
 out=torch.empty_like(g,dtype=torch.uint8);sc=torch.empty((g.shape[0],2),device=g.device,dtype=torch.float32)
 assert LIB.glm_fused341(V(g.data_ptr()),V(u.data_ptr()),V(out.data_ptr()),V(sc.data_ptr()),g.shape[0],VARIANT,V(torch.cuda.current_stream().cuda_stream))==0
 return out,sc
def standalone(q):
 torch.manual_seed(341);q['activation_trials']=[];start=time.monotonic()
 for rows in [8,64,128,4096]:
  g=torch.randn((rows,256),device='cuda',dtype=torch.bfloat16)*3;u=torch.randn_like(g)*2
  # Includes zero/underflow/extreme finite inputs and BF16 intermediate overflow safety.
  g[0,:8]=torch.tensor([0.,-0.,.001,-.001,30.,-30.,60.,-60.],device='cuda',dtype=torch.bfloat16)
  ref,rs=activation(g,u,0);got,gs=activation(g,u,1);torch.cuda.synchronize()
  a=ref.view(torch.float8_e4m3fn).float().view(rows,2,128)*rs[:,:,None];b=got.view(torch.float8_e4m3fn).float().view(rows,2,128)*gs[:,:,None]
  mismatch=float((ref!=got).float().mean().item());rms=float(((a-b).square().mean()/a.square().mean().clamp_min(1e-20)).sqrt().item());assert mismatch<=.005 and rms<=.01 and bool(torch.isfinite(b).all()),(rows,mismatch,rms)
  trial={'rows':rows,'synthetic_input':True,'packed_mismatch_fraction':mismatch,'dequant_relative_RMS':rms,'scale_max_abs':float((rs-gs).abs().max().item()),'finite':True,'pairs':[]}
  for _ in range(4):activation(g,u,0);activation(g,u,1)
  for pair,order in enumerate([[0,1],[1,0],[0,1],[1,0]]):
   rec={}
   for mode in order:
    wall=[];event=[]
    for _ in range(10):
     torch.cuda.synchronize();beg=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);t=time.perf_counter();beg.record();out=activation(g,u,mode);end.record();end.synchronize();wall.append(time.perf_counter()-t);event.append(beg.elapsed_time(end)/1000)
    rec[str(mode)]={'wall_seconds':wall,'stream_event_seconds':event,'mean_wall_seconds':statistics.mean(wall)}
   trial['pairs'].append(rec)
  q['activation_trials'].append(trial)
 q['activation_measurement_seconds']=time.monotonic()-start
