"""Isolated binding for247; no load/create/forward occurs on import."""
import ctypes as C
import time
import numpy as np
from .long_native_r179 import Native as _LegacyNative
P=C.c_void_p;I=C.c_int;S=C.c_size_t
SYMBOL='modern_kimi_pair_append_true_input_r247'
def declare_append(lib):
 f=getattr(lib,SYMBOL);f.argtypes=[P,I,I,S,P,P,P,P,P];f.restype=I;return f
class Native(_LegacyNative):
 def __init__(self,g):
  super().__init__(g)
  self.append_function=declare_append(self.lib)
 def call_append(self,logical,past,tokens):
  if type(logical)is not int or logical!=1 or type(past)is not int or past<1:
   raise ValueError('r248 logical1 and positive committed past required')
  if not isinstance(tokens,(list,tuple)) or not 1<=len(tokens)<=512 or past>32768-len(tokens):
   raise ValueError('r248 actual append width1..512 inside32768 required')
  if any(type(v)is not int or not 0<=v<self.vocab for v in tokens):
   raise ValueError('r248 actual vocabulary IDs required')
  widths=[len(tokens)];flat=np.asarray(tokens,np.int64);pos=np.arange(past,past+len(tokens),dtype=np.int64)
  # Original nonSpec forward publishes only the final input position, not T logits.
  logits=np.full((1,self.vocab),0x7fc0,np.uint16);out=np.full(1,-1,np.int64);flags=np.asarray([-1],np.int32)
  # This is input prefill work; do not mislabel it as a T64 speculative profile.
  self.model_events.before_forward('prefill',[logical],[past],widths,self.serial)
  self.arm([logical],[past],widths,False)
  tick=time.perf_counter()
  self.ck(self.append_function(self.ctx,logical,past,len(tokens),*[self.ptr(v)for v in [flat,pos,logits,out,flags]]))
  result=self.finish('append512',[logical],[past],widths,False,logits,out,flags,time.perf_counter()-tick,None)
  self.model_events.after_forward('prefill',result[-1]);return result
