import ctypes as C
import numpy as np
P=C.c_void_p;Z=C.c_size_t;I=C.c_int;U=C.c_uint64
def configure(lib):
 f=lambda n:getattr(lib,'deep_c4_status_'+n+'_r406');f('error').restype=C.c_char_p
 for n,args,result in [('id_size',[],Z),('id',[P],I),('create',[I,C.c_char_p,P],P),('keys',[P],C.c_char_p),('load',[P,C.c_char_p,P,Z,P,I],I),('finish',[P],I),('reset',[P],I),('forward',[P,Z,Z,P,P,P,P,I,I],I),('proof_info',[P,I,P,P,P],I),('proof_read',[P,I,P,Z],I),('stats',[P]*7,I),('dtype',[I],I),('destroy',[P],I),('shutdown',[],I)]:
  f(n).argtypes=args;f(n).restype=result
 return f
def proof(f,check,ctx,which):
 shape=(Z*5)();size=Z();dtype=I();check(f('proof_info')(ctx,which,shape,C.byref(size),C.byref(dtype)))
 if not size.value:return None
 data=np.empty(size.value,np.uint8);check(f('proof_read')(ctx,which,P(data.ctypes.data),size.value))
 return dict(raw=data,shape=tuple(shape[i+1]for i in range(shape[0])),dtype=dtype.value)
def stats(f,check,ctx):
 t=(C.c_double*2)();a,b,c,d=(U*5)(),(U*5)(),(U*5)(),(U*5)();meta=(U*3)();check(f('stats')(ctx,t,a,b,c,d,meta))
 return dict(prepare_s=t[0],commit_s=t[1],counts=list(a),errors=list(b),status_wall_ns=list(c),status_thread_CPU_ns=list(d),clock_reads=meta[0],clocks_valid=bool(meta[1]),owner_thread_hash=meta[2])
