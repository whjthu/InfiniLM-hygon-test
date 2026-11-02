"""New C++ pointwise/tail leaves versus independent172 NumPy HC oracles."""
import pathlib,json,hashlib,ctypes,time,os,re,numpy as np
from reference import full_pre,bf16_rn
R=pathlib.Path('/refactor');B=R/'candidates/Deep-HC-full-pre-fusion-r372-cpu-v1';M=pathlib.Path('/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8')
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();array_sha=lambda a:hashlib.sha256(a.tobytes()).hexdigest()
P=ctypes.c_void_p;I=ctypes.c_int;lib=ctypes.CDLL(str(B/'libCPU_math_leaf_r372.so'))
lib.hc372_cpu_cast_square.argtypes=[P,P,P];lib.hc372_cpu_tail.argtypes=[P]*5+[I]+[P]*3;lib.hc372_cpu_selected.argtypes=[I,ctypes.c_char_p]
for mode,flag,result in [(0,None,0),(1,None,0),(1,b'0',0),(0,b'1',0),(1,b'1',1),(2,b'1',-1),(1,b'bad',-1)]:assert lib.hc372_cpu_selected(mode,flag)==result
assert lib.hc372_cpu_cast_square(None,None,None)==1
started=time.perf_counter();import torch
from safetensors import safe_open
assert not torch.cuda.is_initialized()
prior=R/'candidates/Deep-HC-T1-Model-r366-cpu-v1/CPU-all86-oracle.json';assert sha(prior)=='d3881d67fc0a4730b873a45c903ce8c4de86c3a75c8e3f07eaa08dc97299cdec';old=json.loads(prior.read_bytes());assert old['passed']and old['oracle_cases']==172
cfg=json.loads((M/'config.json').read_bytes());mapping=json.loads((M/'model.safetensors.index.json').read_bytes())['weight_map'];rows=[];max_errors={k:0. for k in ['collapsed_BF16','post','comb']}
def guarded(n,dtype):
 a=np.full(n+32,0xa55a if dtype==np.uint16 else np.nan,dtype=dtype);return a,a[16:-16]
def guard(a):return a[:16].tobytes()+a[-16:].tobytes()
def value(a):return (a.astype(np.uint32)<<16).view(np.float32)if a.dtype==np.uint16 else a
for group in old['groups']:
 layer,kind=group['layer'],group['kind'];weights=[]
 for name,shape in [('fn',(24,16384)),('base',(24,)),('scale',(3,))]:
  key=f'layers.{layer}.hc_{kind}_{name}'
  with safe_open(M/mapping[key],framework='pt',device='cpu')as reader:t=reader.get_tensor(key).contiguous()
  assert t.dtype==torch.float32 and tuple(t.shape)==shape;weights.append(t.numpy().copy())
 assert [array_sha(a)for a in weights]==group['weight_state_initial_SHA256']
 for case in group['cases']:
  rng=np.random.default_rng(case['seed']);bits=bf16_rn(rng.standard_normal((1,4,4096),dtype=np.float32)*np.float32(case['std']));assert array_sha(bits)==case['input_BF16_SHA256']
  expected=full_pre(bits,*weights,float(cfg['rms_norm_eps']),float(cfg['hc_eps']),int(cfg['hc_sinkhorn_iters']));assert {n:array_sha(v)for n,v in expected.items()}==case['CPU_reference_output_SHA256']
  gx,xf=guarded(16384,np.float32);gs,square=guarded(16384,np.float32);guards=[guard(a)for a in [gx,gs]];input_sha=array_sha(bits)
  assert lib.hc372_cpu_cast_square(P(bits.ctypes.data),P(xf.ctypes.data),P(square.ctypes.data))==0
  independent=(bits.astype(np.uint32)<<16).view(np.float32).reshape(-1);assert np.array_equal(xf,independent)and np.array_equal(square,independent*independent)
  assert [guard(a)for a in [gx,gs]]==guards and array_sha(bits)==input_sha
  # CPU Mean is an independent oracle only. Actual GPU Mean/Add/Rsqrt/Mul
  # and Gemm calls remain unchanged in adapter.cpp; their GPU tree is not ported.
  mean=np.mean(square,dtype=np.float32);inv=np.float32(1)/np.sqrt(np.float32(mean+np.float32(cfg['rms_norm_eps'])),dtype=np.float32);normalized=np.multiply(independent.reshape(1,16384),inv,dtype=np.float32);mixes=np.matmul(normalized,weights[0].T).astype(np.float32).reshape(24)
  constants=np.array([cfg['rms_norm_eps'],cfg['hc_eps'],2],dtype=np.float32);inputs=[xf,mixes,weights[1],weights[2],constants];state=[array_sha(a)for a in inputs]
  gc,collapsed=guarded(4096,np.uint16);gp,post=guarded(4,np.float32);gb,comb=guarded(16,np.float32);canary=[guard(a)for a in [gc,gp,gb]]
  rc=lib.hc372_cpu_tail(*[P(a.ctypes.data)for a in inputs],int(cfg['hc_sinkhorn_iters']),*[P(a.ctypes.data)for a in [collapsed,post,comb]]);assert rc==0
  metrics={}
  for name,actual in [('collapsed_BF16',collapsed),('post',post),('comb',comb.reshape(1,4,4))]:
   av=value(actual).reshape(-1);ev=value(expected[name]).reshape(-1);assert np.isfinite(av).all();diff=av.astype(np.float64)-ev.astype(np.float64);nrmse=float(np.sqrt(np.mean(diff*diff))/max(float(np.sqrt(np.mean(ev.astype(np.float64)**2))),1e-30));assert nrmse<1e-3;metrics[name]=dict(NRMSE=nrmse,maxabs=float(np.max(np.abs(diff))),finite=True);max_errors[name]=max(max_errors[name],nrmse)
  assert [guard(a)for a in [gc,gp,gb]]==canary and [array_sha(a)for a in inputs]==state
  for bad in [0,129]:
   before=[a.tobytes()for a in [collapsed,post,comb]];assert lib.hc372_cpu_tail(*[P(a.ctypes.data)for a in inputs],bad,*[P(a.ctypes.data)for a in [collapsed,post,comb]])==1;assert before==[a.tobytes()for a in [collapsed,post,comb]]
  assert [array_sha(a)for a in weights]==group['weight_state_initial_SHA256'];rows.append(dict(layer=layer,kind=kind,seed=case['seed'],input_SHA256=input_sha,metrics=metrics,state_unchanged=True,output_canaries_unchanged=True,original_CPU724_reference_SHA256=case['CPU_reference_output_SHA256']))
assert len(rows)==172 and not torch.cuda.is_initialized();maps=pathlib.Path('/proc/self/maps').read_text();assert 'libDeep_'not in maps
q=dict(completed=True,passed=True,actual_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],CPU_only=True,GPU_initialized=False,Native_dlopen=False,CPU_math_leaf_dlopen=True,whole_Target_Model_loaded=False,small_HC_parameters_only=True,source_oracle_SHA256=sha(prior),groups=86,cases=172,policy_cases=7,max_NRMSE=max_errors,rows=rows,seconds=time.perf_counter()-started,scope='CPUnewleavesonly;originalGPUreduction/GEMVuntouchedandGPUcandidatequalificationNOT_RUN;noGPUF64')
(B/'CPU-oracle.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({k:q[k]for k in ['passed','groups','cases','max_NRMSE','seconds']}))
