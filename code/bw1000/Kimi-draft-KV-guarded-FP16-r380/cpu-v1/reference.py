"""Independent NumPy F32 dot, explicit BF16/FP16 storage roundings; CPU only."""
import numpy as np
def f32(bits):return (bits.astype(np.uint32)<<16).view(np.float32)
def bf16(a):
 a=np.asarray(a,dtype=np.float32);u=a.view(np.uint32);return ((u+np.uint32(0x7fff)+((u>>16)&1))>>16).astype(np.uint16)
def bf16_value(a):return f32(bf16(a))
def converted_stats(a):
 a=np.asarray(a,dtype=np.float32)
 with np.errstate(over='ignore',under='ignore',invalid='ignore'):h=a.astype(np.float16);b=h.astype(np.float32)
 return dict(elements=a.size,finite=bool(np.isfinite(a).all()),maxabs=float(np.max(np.abs(a))),nonzero_to_zero=int(np.count_nonzero((a!=0)&(b==0))),FP16_subnormal=int(np.count_nonzero((b!=0)&(np.abs(b)<np.float32(2**-14)))),changed=int(np.count_nonzero(a!=b)),FP16_nonfinite=int(np.count_nonzero(~np.isfinite(b))),conversion_maxabs=float(np.max(np.abs(a-b))))
def reference(x,w,half=False):
 x=f32(x);w=f32(w)
 if half:x=x.astype(np.float16).astype(np.float32);w=w.astype(np.float16).astype(np.float32)
 out=x@w.T
 if half:
  with np.errstate(over='ignore',under='ignore'):out=out.astype(np.float16).astype(np.float32)
 return bf16_value(out)
def metric(a,b):
 a=np.asarray(a,np.float32);b=np.asarray(b,np.float32);delta=a-b;den=float(np.sum(b*b,dtype=np.float64))**.5
 return dict(finite=bool(np.isfinite(a).all()and np.isfinite(b).all()),NRMSE=float(np.sum(delta*delta,dtype=np.float64))**.5/(den+1e-30),maxabs=float(np.max(np.abs(delta))))
