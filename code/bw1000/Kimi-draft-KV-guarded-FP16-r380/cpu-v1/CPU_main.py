"""Four payload conversion statistics and actual guarded wrapper CPU closure."""
import pathlib,json,hashlib,os,time,traceback,numpy as np
from tiles import load
from reference import f32,bf16,reference,metric,converted_stats
R=pathlib.Path('/refactor');S=pathlib.Path(__file__).resolve().parent;B=R/'candidates/Kimi-draft-KV-guarded-FP16-r380-cpu-v1';B.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
q=dict(completed=False,passed=False,CPU_only=True,GPU_initialized=False,Model_loaded=False,target_head_loaded=False,actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],publication_commit=os.environ['KIMI380_PUBLICATION_COMMIT'],cases=[]);started=time.perf_counter()
try:
 mf=S/'SOURCE_MANIFEST.json';assert sha(mf)==os.environ['KIMI380_CPU_MANIFEST_SHA'];m=json.loads(mf.read_bytes());assert all(sha(S/n)==h for n,h in m['files_SHA256'].items());assert all(sha(p)==h for p,h in m['dependencies_SHA256'].items())
 import torch
 from guarded_linear import GuardedFP16Linear
 torch.set_num_threads(4);torch.set_num_interop_threads(1);assert not torch.cuda.is_initialized();arrays,proof=load();q['checkpoint']=proof;q['weight_stats']=[]
 def linear(bits):
  n,k=bits.shape;v=torch.from_numpy(bits.copy()).view(torch.bfloat16);a=torch.nn.Linear(k,n,bias=False,device='meta',dtype=torch.bfloat16);a.weight=torch.nn.Parameter(v,requires_grad=False);return a
 for index,(key,bits)in enumerate(arrays.items()):
  w=f32(bits);stats=converted_stats(w);q['weight_stats'].append(dict(key=key,**stats));assert stats['finite']and stats['FP16_nonfinite']==0
  rng=np.random.default_rng(380000+index);x=bf16(rng.standard_normal((3,w.shape[1]),dtype=np.float32)*np.float32(.1 if 'kv_a'in key else 1.));cols=np.unique(np.linspace(0,w.shape[0]-1,32,dtype=int));small=bits[cols].copy();base=reference(x,small);candidate=reference(x,small,True);err=metric(candidate,base);assert err['finite']and err['NRMSE']<.02
  original=linear(small);guard=GuardedFP16Linear(original);xt=torch.from_numpy(x.copy()).view(torch.bfloat16)
  with torch.inference_mode():a=guard(xt);b=guard(xt,enabled=True)
  base_error=metric(a.float().numpy(),base);on_error=metric(b.float().numpy(),candidate);assert base_error['finite']and on_error['finite']and base_error['NRMSE']<.02 and on_error['NRMSE']<.02 and guard.last_path=='hit';assert np.array_equal(original.weight.view(torch.uint16).numpy(),small)
  q['cases'].append(dict(key=key,full_K=w.shape[1],sample_M=3,sample_output_columns=cols.tolist(),NumPy_half_vs_BF16=err,actual_Torch_CPU_BF16_vs_independent=base_error,actual_guarded_Torch_CPU_vs_independent_half=on_error,actual_counts=guard.counts,weight_unchanged=True,input_converted_stats=converted_stats(f32(x))))
 tiny=linear(bf16(np.ones((2,2),np.float32)));g=GuardedFP16Linear(tiny)
 with torch.inference_mode():
  for name,v,path in [('range',65536.,'range_fallback'),('boundary',65280.,'output_fallback')]:
   z=torch.full((1,2),v,dtype=torch.bfloat16);out=g(z,enabled=True);assert g.last_path==path and bool(torch.isfinite(out).all())
  rejected=[]
  for value in [float('nan'),float('inf'),-float('inf')]:
   try:g(torch.full((1,2),value,dtype=torch.bfloat16),enabled=True)
   except ValueError:rejected.append(str(value))
  assert len(rejected)==3
  overflow=GuardedFP16Linear(linear(bf16(np.full((2,2),65536.,np.float32))));overflow(torch.ones((1,2),dtype=torch.bfloat16),enabled=True);assert overflow.last_path=='weight_range_fallback'
 assert not torch.cuda.is_initialized()
 q.update(passed=True,source_manifest_SHA256=sha(mf),Torch_version=torch.__version__,Torch_C_SHA256=sha(torch._C.__file__),guard_negative_cases=dict(input_65536='range_fallback',input_65280_weight1_sum2='output_fallback',nonfinite_rejected=3,weight_65536='weight_range_fallback'),underflow_contract=converted_stats(np.array([2**-40,2**-25,2**-24,2**-15,2**-14],np.float32)),GPU_numeric='NOT_RUN',Model_gain='UNKNOWN',sampled_CPU_oracle_only=True)
except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
q.update(completed=True,seconds=time.perf_counter()-started);(B/'CPU-result.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({k:q.get(k)for k in ['completed','passed','actual_CPU','seconds','error']}));assert q['passed']
