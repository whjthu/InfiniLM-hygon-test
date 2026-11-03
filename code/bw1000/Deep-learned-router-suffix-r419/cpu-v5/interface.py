import ctypes as C
import numpy as np
P=C.c_void_p;Z=C.c_size_t;I=C.c_int;U=C.c_uint64
def configure(lib):
 f=lambda name:getattr(lib,'deep_router_'+name+'_r419');f('error').restype=C.c_char_p
 for name,args,result in [('id_size',[],Z),('id',[P],I),('create',[I,C.c_char_p,P],P),('keys',[P],C.c_char_p),('load',[P,C.c_char_p,P,Z,P,I],I),('finish',[P],I),('forward',[P,Z,P,P,I],I),('project',[P,P,P],I),('same_logits',[P,I],I),('proof_info',[P,I,P,P,P],I),('proof_read',[P,I,P,Z],I),('stats',[P,P,P],I),('dtype',[I],I),('destroy',[P],I),('shutdown',[],I)]:
  f(name).argtypes=args;f(name).restype=result
 return f
def proof(f,check,ctx,field):
 shape=(Z*5)();size=Z();dtype=I();check(f('proof_info')(ctx,field,shape,C.byref(size),C.byref(dtype)))
 if not size.value:return None
 data=np.empty(size.value,np.uint8);check(f('proof_read')(ctx,field,P(data.ctypes.data),size.value));return dict(raw=data,shape=tuple(shape[i+1]for i in range(shape[0])),dtype=dtype.value)
def stats(f,check,ctx):
 t=(C.c_double*2)();a=(U*4)();check(f('stats')(ctx,t,a));return dict(MoE_return_s=t[0],MoE_complete_s=t[1],calls=a[0],hits=a[1],fallbacks=a[2],owner_thread_hash=a[3])
