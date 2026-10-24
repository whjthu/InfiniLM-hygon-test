"""One GPU actual HC-pre component; correlated prefill rows, no Model TPS."""
import pathlib,json,hashlib,ctypes,os,time,traceback
import numpy as np
import torch
from safetensors import safe_open
r=pathlib.Path('/refactor');tag=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];out=r/'results'/f'Deep-HC-RMS-r7377-{tag}.json';assert not out.exists()
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
torch.set_num_threads(1);assert torch.cuda.device_count()==1;torch.cuda.set_device(0)
bp=r/'candidates/deep-HC-RMS-r7375/CPU-build.json';assert sha(bp)=='4e088f304fd172b03b3f706369b7c1c2a638d5eafa65923931278f9db698718d';build=json.loads(bp.read_text());assert build['completed']and build['producer']=='910374.726';library=pathlib.Path(build['library']);assert sha(library)==build['library_SHA256']=='4b8c5bad4b87c2d6553f46b5603a5eaabe3d033f8b70b7223238f247bce28aab'
for p,v in build['DTK_bindings'].items():assert sha(p)==v['SHA256']
lib=ctypes.CDLL(str(library));P=ctypes.c_void_p;I=ctypes.c_int;D=ctypes.c_double
lib.deep_hc_rms_create_r7374.argtypes=[I,D,D,I,P,P,P];lib.deep_hc_rms_create_r7374.restype=P
lib.deep_hc_rms_forward_r7374.argtypes=[P,I,ctypes.POINTER(P),P];lib.deep_hc_rms_forward_r7374.restype=I
lib.deep_hc_rms_destroy_r7374.argtypes=[P,ctypes.POINTER(I)];lib.deep_hc_rms_destroy_r7374.restype=I
for n in ['error','report']:getattr(lib,'deep_hc_rms_'+n+'_r7374').restype=ctypes.c_char_p
def check(v):
 if v:raise RuntimeError(lib.deep_hc_rms_error_r7374().decode())
report={'completed':False,'step':tag,'library_SHA256':sha(library),'CPU_receipt_SHA256':sha(bp),'calls':[],'scope':'SingleGPU production layer0 HC-attn-pre, actual591e0 fourprefillprompts eachlast8positions. M32 geometry, correlatedprefill not699decode. Baseline originalHC::pre vs oneF32RmsNorm replacement; sameCast/Gemm/Sinkhorn/tail. CompleteCAPI includesinputstaging/outputcopies/finaldrain; CPUchecks outsideCAPI/insideoverallwall. No Modelgain/finaltokens.'}
def save():
 p=out.with_suffix('.tmp');p.write_text(json.dumps(report,indent=2));p.replace(out)
ctx=None;start=time.perf_counter()
try:
 rp=r/'results/deep-layer0-r7005-910374.591-owner0-B32.json';assert sha(rp)=='977be1e7bdaef015744cc6d3921a73d5408d059ec99b16ebd0f7f2f64e42b37e';event=json.loads(rp.read_text())['history'][0];assert event['rows']==[0,1,2,3]
 meta=next(s for s in event['stages']if s['name']=='entry');raw=r/'results/deep-layer0-r7005-910374.591'/meta['raw_file'];assert sha(raw)==meta['whole_SHA256']=='87a8494c87cf3b61e95e9b96f391e4ef43f2e08c45e945849bbfaff56f37d040'
 rows=[j*512+t for j in range(4)for t in range(504,512)];a=np.fromfile(raw,np.uint16).reshape(meta['shape'])[rows].copy();x=torch.from_numpy(a).view(torch.bfloat16).cuda(0)
 report['input']={'source_SHA256':sha(raw),'selected_SHA256':hashlib.sha256(a.tobytes()).hexdigest(),'shape':list(a.shape),'rows':rows}
 model=pathlib.Path('/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8');cfg=json.loads((model/'config.json').read_text());mapping=json.loads((model/'model.safetensors.index.json').read_text())['weight_map'];gold={v['key']:v['CPU_tensor_SHA256']for v in json.loads((r/'results/deep-actualprefix-r6994-910374.587-CPUweights.json').read_text())};weights=[];proof=[]
 for suffix,shape in [('fn',[24,16384]),('base',[24]),('scale',[3])]:
  key='layers.0.hc_attn_'+suffix;assert key in mapping and key in gold
  with safe_open(model/mapping[key],framework='pt',device='cpu')as f:t=f.get_tensor(key).contiguous()
  assert t.dtype==torch.float32 and list(t.shape)==shape;arr=t.numpy();h=hashlib.sha256(arr.tobytes()).hexdigest();assert h==gold[key];weights.append(arr);proof.append({'key':key,'SHA256':h,'shape':shape})
 report['weights']=proof;report['CPU_input_weight_prepare_seconds']=time.perf_counter()-start
 outputs=[torch.empty((32,4096),device='cuda',dtype=torch.bfloat16),torch.empty((32,4),device='cuda',dtype=torch.float32),torch.empty((32,4,4),device='cuda',dtype=torch.float32)];tensors=[x,*outputs];ptrs=(P*4)(*[t.data_ptr()for t in tensors])
 tick=time.perf_counter();ctx=lib.deep_hc_rms_create_r7374(0,float(cfg['rms_norm_eps']),float(cfg['hc_eps']),int(cfg['hc_sinkhorn_iters']),*[P(a.ctypes.data)for a in weights]);assert ctx,lib.deep_hc_rms_error_r7374().decode();report['context_build_weights_constants_ones_seconds']=time.perf_counter()-tick
 props=torch.cuda.get_device_properties(0);report['hardware']={'node':os.environ['SLURMD_NODENAME'],'cards':1,'device':props.name,'gcnArchName':getattr(props,'gcnArchName',None)}
 plan=[('warmOFF',0),('warmON',1),('warmRestore',0)]+[(f'ordinary{i}',m)for i,m in enumerate([0,1,1,0]*4)]+[('restoreOFF',0)]
 report['plan']=plan;save();baseline=None
 for label,mode in plan:
  os.environ['INFINILM_DEEP_HC_RMS_R7374']=str(mode);tick=time.perf_counter();check(lib.deep_hc_rms_forward_r7374(ctx,mode,ptrs,P(torch.cuda.current_stream().cuda_stream)));wall=time.perf_counter()-tick;native=json.loads(lib.deep_hc_rms_report_r7374().decode());assert native['candidate']==bool(mode)
  actual=[t.cpu().view(torch.uint16).numpy().copy()if t.dtype==torch.bfloat16 else t.cpu().numpy().copy()for t in outputs]
  if baseline is None:baseline=[a.copy()for a in actual]
  metrics=[]
  for name,a,b in zip(['collapsed','post','comb'],actual,baseline):
   av=(a.astype(np.uint32)<<16).view(np.float32)if a.dtype==np.uint16 else a;bv=(b.astype(np.uint32)<<16).view(np.float32)if b.dtype==np.uint16 else b;assert np.isfinite(av).all()and np.isfinite(bv).all();diff=av.astype(np.float64)-bv.astype(np.float64)
   exact=bool(np.array_equal(a,b));numeric=bool(np.allclose(av,bv,rtol=1e-3,atol=1e-3));metrics.append({'name':name,'exact':exact,'exact_fraction':float(np.mean(a==b)),'amax':float(np.max(np.abs(diff))),'relative_RMS':float(np.sqrt(np.mean(diff**2))/(np.sqrt(np.mean(bv.astype(np.float64)**2))+1e-30)),'component_argmax_equal':bool(np.array_equal(av.reshape(32,-1).argmax(-1),bv.reshape(32,-1).argmax(-1))),'within_1e3_diagnostic':numeric,'SHA256':hashlib.sha256(a.tobytes()).hexdigest()})
   if not mode:assert exact,('OFF_restore_changed',label,name)
  report['calls'].append({'label':label,'mode':mode,'complete_CAPI_wall_seconds':wall,'native':native,'metrics':metrics});save()
 report['completed']=True
except BaseException as e:report.update(error=repr(e),traceback=traceback.format_exc());raise
finally:
 if ctx:
  e=I();report['destroy_code']=lib.deep_hc_rms_destroy_r7374(ctx,ctypes.byref(e));report['Runtime_expired']=e.value
 report['overall_seconds_includesCPUvalidation']=time.perf_counter()-start;save()
assert report['destroy_code']==0 and report['Runtime_expired']==1
print(json.dumps({'completed':True,'report':str(out),'SHA256':sha(out)}))
