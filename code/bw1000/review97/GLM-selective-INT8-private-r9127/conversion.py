"""CPU FP8-block -> INT8/channel conversion prototype. Synthetic selftest; no model-quality claim."""
import numpy as np

def e4m3fn_table():
 u=np.arange(256,dtype=np.uint16);exponent=(u>>3)&15;fraction=u&7
 values=np.where(exponent==0,fraction.astype(np.float32)*np.float32(2.**-9),(np.float32(1)+fraction.astype(np.float32)/8)*np.exp2(exponent.astype(np.float32)-7)).astype(np.float32)
 values=np.where((u&128)!=0,-values,values);values[(exponent==15)&(fraction==7)]=np.nan
 return values.astype(np.float32)

def per_row_i8(x):
 x=np.asarray(x,dtype=np.float32);assert x.ndim==2 and np.isfinite(x).all()
 scale=np.maximum(np.max(np.abs(x),axis=1,keepdims=True)/np.float32(127),np.float32(1e-8)).astype(np.float32)
 v=x*(np.float32(1)/scale);packed=(np.sign(v)*np.floor(np.abs(v)+np.float32(.5))).clip(-127,127).astype(np.int8)
 return np.ascontiguousarray(packed),np.ascontiguousarray(scale)

def fp8_blocks_to_i8_channels(codes,global_scales,global_n_begin=0,global_k_begin=0):
 codes=np.asarray(codes,dtype=np.uint8);scales=np.asarray(global_scales,dtype=np.float32);assert codes.ndim==2 and scales.ndim==2 and global_n_begin>=0 and global_k_begin>=0
 n,k=codes.shape;ns=(np.arange(n)+global_n_begin)//128;ks=(np.arange(k)+global_k_begin)//128;assert ns.max()<scales.shape[0]and ks.max()<scales.shape[1]
 decoded=e4m3fn_table()[codes];assert np.isfinite(decoded).all()and np.isfinite(scales).all()and (scales>0).all()
 weight=(decoded*scales[ns[:,None],ks[None,:]]).astype(np.float32);packed,channel_scale=per_row_i8(weight)
 return packed,channel_scale,weight

def i8_gemm_oracle(a,ascale,w,wscale):
 a=np.asarray(a,np.int8);w=np.asarray(w,np.int8);assert a.ndim==w.ndim==2 and a.shape[1]==w.shape[1]and a.shape[1]<=np.iinfo(np.int32).max//16384
 accum=a.astype(np.int32)@w.astype(np.int32).T
 # Same source ordering as actual Hygon Post: (as[row]*bs[col])*float(accum).
 return (ascale*wscale.T)*accum.astype(np.float32),accum

def error_summary(actual,reference):
 a=np.asarray(actual,np.float32);b=np.asarray(reference,np.float32);diff=a-b;den=max(float(np.sqrt(np.mean(b.astype(np.float64)**2))),1e-30)
 return dict(finite=bool(np.isfinite(a).all()),maxabs=float(np.max(np.abs(diff))),relative_RMS=float(np.sqrt(np.mean(diff.astype(np.float64)**2)))/den)

def selftest():
 rng=np.random.default_rng(20261026);codes=rng.integers(0,256,(129,256),dtype=np.uint8);codes[codes==127]=126;codes[codes==255]=254
 scales=np.array([[.1,2.], [3.,.03]],np.float32);w,ws,truth=fp8_blocks_to_i8_channels(codes,scales);x=rng.normal(size=(5,256)).astype(np.float32);a,as_=per_row_i8(x);y,acc=i8_gemm_oracle(a,as_,w,ws);reconstructed=(a.astype(np.float32)*as_)@(w.astype(np.float32)*ws).T
 assert error_summary(y,reconstructed)['relative_RMS']<1e-6
 zero,zs=per_row_i8(np.zeros((2,4),np.float32));assert not zero.any()and np.all(zs==np.float32(1e-8))
 ones=np.zeros((1,256),np.uint8);ones[0,0]=56;ones[0,128]=56
 _,_,folded=fp8_blocks_to_i8_channels(ones,scales);assert folded[0,0]!=folded[0,128]
 try:fp8_blocks_to_i8_channels(np.array([[127]],np.uint8),np.ones((1,1),np.float32))
 except AssertionError:nonfinite_rejected=True
 else:raise AssertionError('FP8 NaN accepted')
 # Offset/TP slice uses global block index rather than local zero incorrectly.
 _,_,offsettruth=fp8_blocks_to_i8_channels(codes[:1,:128],scales,128,128);assert np.array_equal(offsettruth,e4m3fn_table()[codes[:1,:128]]*scales[1,1])
 return dict(passed=True,CPU_only=True,source='syntheticseed20261026/notcheckpoint/notactivation',single_scale_epilogue_valid_for_requantized_weights=True,original_FP8_blockscales_not_directly_representable_by_one_channel_scalar=True,global_TP_block_offsets_pass=True,zero_and_nonfinite_checks_pass=True,weights_error=error_summary(w.astype(np.float32)*ws,truth),activation_error=error_summary(a.astype(np.float32)*as_,x),output_error_vs_synthetic_unquantized_surrogate=error_summary(y,x@truth.T),integer_accumulator_max=int(np.max(np.abs(acc))),no_Model_quality_or_speed_claim=True)
if __name__=='__main__':
 import json
 print(json.dumps(selftest(),indent=2))
