"""Untimed CPU oracle on the actual immutable SDK gate/up outputs.

Only the changed activation/quant chain is independently recomputed. Whole MoE
uses the qualified original SDK path plus same-hidden comparison, not a claimed
independent whole-model oracle. CPU F64 below is a diagnostic norm, never GPU F64.
"""
import numpy as np
def decode(field,kind,shape,dtypes):
 assert field is not None and field['dtype']==dtypes[kind]and tuple(field['shape'])==shape
 dtype={'BF16':np.uint16,'F32':np.float32,'I8':np.int8}[kind]
 a=field['raw'].view(dtype).reshape(shape).copy()
 return(a.astype(np.uint32)<<16).view(np.float32)if kind=='BF16'else a
def nr(a,b):return float(np.linalg.norm(a.astype(np.float64)-b.astype(np.float64))/(np.linalg.norm(b.astype(np.float64))+1e-30))
def activation_reference(torch,gu):
 g=torch.from_numpy(gu[:,:256].copy()).clamp(max=10).to(torch.bfloat16).float()
 u=torch.from_numpy(gu[:,256:].copy()).clamp(min=-10,max=10).to(torch.bfloat16).float()
 return((g*torch.reciprocal(torch.exp(-g)+1.))*u).to(torch.bfloat16).float().numpy()
def check_activation(actual,expected):
 assert np.isfinite(actual).all()and np.isfinite(expected).all()
 ab=(actual.view(np.uint32)>>16).astype(np.int32);eb=(expected.view(np.uint32)>>16).astype(np.int32)
 difference=np.abs(ab-eb);max_ulp=int(difference.max(initial=0));error=nr(actual,expected)
 assert max_ulp<=1 and error<=.005,('BF16 activation',max_ulp,error)
 return dict(NRMSE=error,max_BF16_ULP=max_ulp,bits_different=int(np.count_nonzero(difference)),CPU_libmath_vs_pinned_GPU_intrinsic=True)
def check_quant(a,packed,scale):
 assert np.isfinite(a).all()and np.isfinite(scale).all()and np.all(scale>0)
 expected_scale=np.maximum(np.max(np.abs(a),axis=1,keepdims=True)/np.float32(127),np.float32(1e-8)).astype(np.float32)
 assert np.allclose(scale,expected_scale,rtol=2e-6,atol=0),('F32 quant scale',nr(scale,expected_scale))
 # Quantize the actual BF16 activation, with actual returned scale. The only
 # permitted one-bin CPU/GPU discrepancy is an explicitly near-half F32 tie.
 v=a*(np.float32(1)/scale);mag=np.floor(np.abs(v)+np.float32(.5));expected=np.clip(np.where(v>=0,mag,-mag),-127,127).astype(np.int8)
 delta=np.abs(packed.astype(np.int16)-expected.astype(np.int16));different=delta!=0
 distance=np.abs(np.abs(v)-(np.floor(np.abs(v))+np.float32(.5)))
 bound=np.float32(4*np.finfo(np.float32).eps)*np.maximum(np.abs(v),np.float32(1))
 assert np.all(delta<=1)and np.all(~different|(distance<=bound)),('Quant half-away/non-tie mismatch',int(delta.max(initial=0)),int(np.count_nonzero(different)))
 return dict(scale_NRMSE=nr(scale,expected_scale),scale_max_relative_error=float(np.max(np.abs(scale-expected_scale)/expected_scale)),packed_exact=bool(not different.any()),near_half_F32_tie_one_bin_count=int(np.count_nonzero(different)),half_away=True,quant_reference_on_actual_BF16_activation=True)
def validate_original(torch,fields,dtypes):
 result=[]
 for role,rows in enumerate([6,1]):
  first=3+role*7;gu=decode(fields[first],'BF16',(rows,512),dtypes);a=decode(fields[first+1],'BF16',(rows,256),dtypes);p=decode(fields[first+2],'I8',(rows,256),dtypes);s=decode(fields[first+3],'F32',(rows,1),dtypes)
  assert np.isfinite(gu).all();wanted=activation_reference(torch,gu)
  result.append(dict(role=role,rows=rows,original_SDK_activation=check_activation(a,wanted),original_SDK_quant=check_quant(a,p,s),passed=True))
 return result
def validate_candidate(torch,fields,dtypes):
 result=[]
 for role,rows in enumerate([6,1]):
  first=3+role*7;gu=decode(fields[first],'BF16',(rows,512),dtypes);old=decode(fields[first+1],'BF16',(rows,256),dtypes)
  a=decode(fields[first+4],'BF16',(rows,256),dtypes);p=decode(fields[first+5],'I8',(rows,256),dtypes);s=decode(fields[first+6],'F32',(rows,1),dtypes)
  old_p=decode(fields[first+2],'I8',(rows,256),dtypes);old_s=decode(fields[first+3],'F32',(rows,1),dtypes)
  result.append(dict(role=role,rows=rows,fused_vs_independent_CPU_activation=check_activation(a,activation_reference(torch,gu)),fused_vs_same_GU_original_GPU_activation=check_activation(a,old),fused_quant=check_quant(a,p,s),packed_difference_from_SDK=int(np.max(np.abs(p.astype(np.int16)-old_p.astype(np.int16)))),scale_NRMSE_from_SDK=nr(s,old_s),passed=True))
 return result
