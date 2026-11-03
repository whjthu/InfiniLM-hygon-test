"""Record all frozen-v1 CPU comparisons without changing its math or acceptance predicates."""
from pathlib import Path
import json,hashlib,os,sys,time,traceback
import numpy as np
import ctypes as C
R=Path('/refactor');S=Path(__file__).resolve().parent;P=S.parent/'cpu-v1';B=R/'candidates/Deep-learned-router-suffix-r419-semantic-diagnostic-v2';B.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
self_mf=S/'SOURCE_MANIFEST.json';assert sha(self_mf)==os.environ['DEEP419_DIAG_MANIFEST_SHA'];self_meta=json.loads(self_mf.read_bytes());assert all(sha(S/n)==h for n,h in self_meta['files_SHA256'].items())
mf=P/'SOURCE_MANIFEST.json';assert sha(mf)=='54d2ec92d2cccdd73f6337eb763caf436adb02b0b2f35e264f400a10d2de43a6';m=json.loads(mf.read_bytes());assert all(sha(P/n)==h for n,h in m['files_SHA256'].items())
sys.path.insert(0,str(P));import semantic_reference as original
assert Path(original.__file__).resolve()==P/'semantic_reference.py'
import torch
torch.set_num_threads(1);assert not torch.cuda.is_initialized()
q=dict(completed=False,diagnostic_execution_passed=False,original_numerical_predicates_passed=False,CPU_only=True,new_compile=0,Native_dlopen=False,Runtime_created=False,GPU_initialized=False,weights_opened=False,original_source_manifest_SHA256=sha(mf),publication_commit=os.environ['DEEP419_PUBLICATION_COMMIT'],diagnostic_source_manifest_SHA256=sha(self_mf),actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],torch_version=torch.__version__,numpy_version=np.__version__,cases=[])
try:
 rounding=C.CDLL(None).fegetround;rounding.restype=C.c_int;q['CPU_fegetround_readonly']=rounding()
except Exception as exc:q['CPU_fegetround_readonly']='UNKNOWN:'+repr(exc)
tiny=np.array([1,2,0x007fffff,0x00800000],np.uint32).view(np.float32)
q['FTZ_probe_only_not_CPU_register_read']=dict(input_F32_bits=tiny.view(np.uint32).tolist(),numpy_times_one_bits=np.multiply(tiny,np.float32(1),dtype=np.float32).view(np.uint32).tolist(),Torch_times_one_bits=(torch.from_numpy(tiny.copy())*torch.tensor(1,dtype=torch.float32)).numpy().view(np.uint32).tolist(),no_flush_or_rounding_flags_changed=True,interpretation='Probes notproof ofinternalsoftplus/libdevice FTZ; noGPU arithmetic measured')
begin=time.monotonic()
def pack(a):
 a=np.asarray(a);return dict(shape=list(a.shape),dtype=str(a.dtype),raw_SHA256=hashlib.sha256(a.tobytes()).hexdigest(),F32_bits=a.view(np.uint32).tolist())if a.dtype==np.float32 else a.tolist()
for name,logits,bias in original.cases():
 reference=original.torch_reference(torch,logits,bias);candidate=original.numpy_suffix(logits,bias)
 row=dict(case=name,logits=pack(logits),bias=pack(bias),reference={k:pack(v)for k,v in reference.items()},NumPy_candidate={k:pack(v)for k,v in candidate.items()})
 try:row.update(original_validation=original.validate(torch,logits,bias,reference,candidate),original_passed=True)
 except BaseException as exc:row.update(original_passed=False,error=repr(exc),traceback=traceback.format_exc())
 delta={}
 for key in ['scores','selection','weights']:
  a,b=candidate[key],reference[key];finite=np.isfinite(a)&np.isfinite(b);diff=np.abs(a.astype(np.float64)-b.astype(np.float64));positions=np.flatnonzero(finite)
  at=int(positions[np.argmax(diff[positions])])if positions.size else None
  delta[key]=dict(finite_count=int(finite.sum()),maxabs=float(diff[at])if at is not None else None,maxabs_index=at,reference_value=float(b[at])if at is not None else None,candidate_value=float(a[at])if at is not None else None)
 ids=candidate['ids'];chosen=torch.from_numpy(reference['scores'][ids].copy()).reshape(1,6);expected=(chosen/(torch.sum(chosen,dim=1,keepdim=True)+torch.tensor(1e-20,dtype=torch.float32))*torch.tensor(1.5,dtype=torch.float32)).numpy().reshape(6)
 with np.errstate(all='ignore'):
  npsoft=np.where(logits>20,logits,np.log1p(np.exp(logits,dtype=np.float32),dtype=np.float32))
  selected=candidate['scores'][ids];tree=np.float32(np.float32(np.float32(selected[0]+selected[1])+np.float32(selected[2]+selected[3]))+np.float32(selected[4]+selected[5]))
 tsoft=torch.nn.functional.softplus(torch.from_numpy(logits.copy()),beta=1,threshold=20).numpy()
 good=np.isfinite(tsoft)&np.isfinite(npsoft);delta['softplus']=dict(maxabs=float(np.max(np.abs(tsoft[good].astype(np.float64)-npsoft[good].astype(np.float64))))if good.any()else None,finite_count=int(good.sum()))
 row['softplus_reference']=pack(tsoft);row['softplus_NumPy_candidate']=pack(npsoft)
 row['sqrt_reference']=pack(reference['scores']);row['sqrt_NumPy_candidate']=pack(candidate['scores'])
 row['selected_candidate_unbiased_scores']=pack(selected);row['selected_reference_unbiased_scores_at_candidate_IDs']=pack(reference['scores'][ids])
 refsix=torch.from_numpy(reference['scores'][reference['ids']].copy());candsix=torch.from_numpy(reference['scores'][ids].copy())
 row['sum_reference_at_reference_IDs']=pack(torch.sum(refsix).numpy().reshape(1));row['sum_reference_at_candidate_IDs']=pack(torch.sum(candsix).numpy().reshape(1));row['sum_NumPy_candidate_tree']=pack(np.array([tree],np.float32))
 row['candidate_denominator']=pack(np.array([np.float32(tree+np.float32(1e-20))],np.float32))
 ordered=np.sort(reference['selection'])[::-1];mask=np.isfinite(reference['selection'])&np.isfinite(candidate['selection']);selerr=float(np.max(np.abs(reference['selection'][mask].astype(np.float64)-candidate['selection'][mask].astype(np.float64))))if mask.any()else 0.
 margin=float(ordered[5]-ordered[6])if np.isfinite(ordered[[5,6]]).all()else None
 row['margin']=dict(baseline_6_7=margin,selection_maxabs=selerr,separated=margin is not None and margin>2*selerr,same_expert_set=set(ids.tolist())==set(reference['ids'].tolist()))
 def classes(v):
  v=np.asarray(v,np.float32);a=np.abs(v);return ['NaN'if np.isnan(x)else'Inf'if np.isinf(x)else'zero'if x==0 else'subnormal'if x<np.finfo(np.float32).tiny else'normal'for x in a]
 with np.errstate(all='ignore'):nexp=np.exp(logits,dtype=np.float32)
 texp=torch.exp(torch.from_numpy(logits.copy())).numpy()
 row['selected_logit_and_subnormal_diagnostics']=dict(candidate_IDs=ids.tolist(),reference_IDs=reference['ids'].tolist(),candidate_selected_logits=pack(logits[ids]),reference_selected_logits=pack(logits[reference['ids']]),NumPy_softplus_class_at_candidate_IDs=classes(npsoft[ids]),Torch_softplus_class_at_candidate_IDs=classes(tsoft[ids]),NumPy_sqrt_class_at_candidate_IDs=classes(candidate['scores'][ids]),Torch_sqrt_class_at_candidate_IDs=classes(reference['scores'][ids]),NumPy_standalone_exp_probe_at_candidate_IDs=pack(nexp[ids]),Torch_standalone_exp_probe_at_candidate_IDs=pack(texp[ids]),NumPy_exp_probe_classes=classes(nexp[ids]),Torch_exp_probe_classes=classes(texp[ids]),exp_probe_limit='Standaloneexp diagnostic, notclaimedexactinternalprimitive ofTorchsoftplus; noGPUlibdeviceconclusion')
 row['input_groups']=dict(nonfinite_logits=not bool(np.isfinite(logits).all()),exp_subnormal_or_underflow_domain=bool(np.any(np.isfinite(logits)&(logits<-87))),threshold20_neighbors=bool(np.any(np.isfinite(logits)&(np.abs(logits-20)<.02))),nonfinite_reference_weights=not bool(np.isfinite(reference['weights']).all()),nonfinite_candidate_weights=not bool(np.isfinite(candidate['weights']).all()))
 row['normalization_expected_at_candidate_IDs']=pack(expected);finite=np.isfinite(expected)&np.isfinite(candidate['weights']);row['maxabs_weights_at_same_candidate_IDs']=float(np.max(np.abs(expected[finite].astype(np.float64)-candidate['weights'][finite].astype(np.float64))))if finite.any()else None
 row['differences']=delta;q['cases'].append(row)
 q['seconds_so_far']=time.monotonic()-begin;(B/'diagnostic-partial.json').write_text(json.dumps(q,indent=2)+'\n')
assert len(q['cases'])==14 and not torch.cuda.is_initialized()
q.update(completed=True,diagnostic_execution_passed=True,original_numerical_predicates_passed=all(c['original_passed']for c in q['cases']),failing_cases=[c['case']for c in q['cases']if not c['original_passed']],seconds=time.monotonic()-begin,GPU_qualification=False,Model_qualification=False)
(B/'diagnostic.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({k:q[k]for k in ['completed','diagnostic_execution_passed','original_numerical_predicates_passed','actual_CPU','failing_cases','seconds']}))
