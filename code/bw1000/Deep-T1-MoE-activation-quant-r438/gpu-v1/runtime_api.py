"""Use the exact CPU889-compiled C declarations, never a second untyped binder."""
import ctypes as C
import numpy as np
import interface as source_api
Z=source_api.Z;P=source_api.V
def configure(lib):
 source_api.bind(lib)
 return lambda name:getattr(lib,'deep_actquant_'+name+'_r438')
def stats(f,check,ctx):
 seconds=(C.c_double*2)();counts=(C.c_uint64*7)();check(f('stats')(ctx,seconds,counts))
 routed=list(counts[:3]);shared=list(counts[3:6]);assert routed==shared,'Two actual branches must both take the requested mode/fallback'
 return dict(MoE_returned_s=seconds[0],MoE_complete_s=seconds[1],calls=routed[0],hits=routed[1],fallbacks=routed[2],routed=routed,shared=shared,owner_thread_hash=counts[6])
def proof(f,check,ctx,which):
 shape=(Z*5)();size=Z();dtype=C.c_int();check(f('proof_info')(ctx,which,shape,C.byref(size),C.byref(dtype)))
 if size.value==0:assert shape[0]==0 and dtype.value==-1;return None
 assert 1<=shape[0]<=4 and all(v>0 for v in shape[1:1+shape[0]])
 raw=np.empty(size.value,dtype=np.uint8);check(f('proof_read')(ctx,which,raw.ctypes.data,size.value))
 return dict(shape=tuple(shape[1:1+shape[0]]),dtype=dtype.value,raw=raw)
