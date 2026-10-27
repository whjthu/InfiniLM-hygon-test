"""Future private candidate integration only; no DSO load, defaultOFF."""
import os,ctypes as C
class PoolTrimPolicy:
 def __init__(self,lib):
  self.fn=lib.modern_kimi_prefill_pool_trim_stats_r130;self.fn.argtypes=[C.c_int,C.POINTER(C.c_uint64),C.c_size_t];self.fn.restype=C.c_int
 def configure(self,enabled,control,label):
  assert type(enabled)is bool
  values=control.collect('prefill-pool-trim:'+label,{'enabled':enabled});assert all(v['enabled']==enabled for v in values)
  os.environ['INFINILM_KIMI_PREFILL_POOL_TRIM_R130']='1'if enabled else'0'
 def read_after_original_CAPI_completion(self):
  out=[]
  for device in range(8):
   a=(C.c_uint64*7)();assert self.fn(device,a,7)==0;out.append(list(a))
  return out
