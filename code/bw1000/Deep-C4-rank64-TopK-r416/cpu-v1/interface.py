import ctypes as C
import numpy as np
P=C.c_void_p;Z=C.c_size_t;I=C.c_int;U=C.c_uint64
def configure(lib):
 f=lambda n:getattr(lib,'deep_c4_topk_'+n+'_r416');f('error').restype=C.c_char_p
 for n,args,result in [('id_size',[],Z),('id',[P],I),('create',[I,C.c_char_p,P],P),('keys',[P],C.c_char_p),('load',[P,C.c_char_p,P,Z,P,I],I),('finish',[P],I),('reset',[P],I),('forward',[P,Z,Z,P,P,P,P,I,I],I),('proof_info',[P,I,P,P,P],I),('proof_read',[P,I,P,Z],I),('stats',[P]*3,I),('proof_pair',[P,P,Z,P,Z],I),('dtype',[I],I),('destroy',[P],I),('shutdown',[],I)]:
  f(n).argtypes=args;f(n).restype=result
 f('freeze').argtypes=[P];f('freeze').restype=I;f('restore').argtypes=[P];f('restore').restype=I;f('freeze_stats').argtypes=[P,P,P];f('freeze_stats').restype=I
 return f
def proof(f,check,ctx,which):
 shape=(Z*5)();size=Z();dtype=I();check(f('proof_info')(ctx,which,shape,C.byref(size),C.byref(dtype)))
 if not size.value:return None
 data=np.empty(size.value,np.uint8);check(f('proof_read')(ctx,which,P(data.ctypes.data),size.value))
 return dict(raw=data,shape=tuple(shape[i+1]for i in range(shape[0])),dtype=dtype.value)
def stats(f,check,ctx):
 t=(C.c_double*2)();a=(U*4)();check(f('stats')(ctx,t,a))
 return dict(prepare_s=t[0],commit_s=t[1],topk_calls=a[0],topk_hits=a[1],Torch_fallbacks=a[2],owner_thread_hash=a[3],phase_clock_reads=0)

def proof_pair(f,check,ctx):
 outputs=[]
 for which in [5,11]:
  shape=(Z*5)();size=Z();dtype=I();check(f('proof_info')(ctx,which,shape,C.byref(size),C.byref(dtype)));assert size.value>0
  data=np.empty(size.value,np.uint8);outputs.append(dict(raw=data,shape=tuple(shape[i+1]for i in range(shape[0])),dtype=dtype.value))
 check(f('proof_pair')(ctx,P(outputs[0]['raw'].ctypes.data),outputs[0]['raw'].nbytes,P(outputs[1]['raw'].ctypes.data),outputs[1]['raw'].nbytes))
 return outputs

def freeze_stats(f,check,ctx):
 t=(C.c_double*2)();n=(U*8)();check(f('freeze_stats')(ctx,t,n));return dict(freeze_CPP_s=t[0],restore_CPP_s=t[1],captures=n[0],restores=n[1],valid_raw_bytes=n[2],state_bytes=n[3],D2D_copies=n[4],prefix_generation=n[5],restored_generation=n[6],realprefix_revision=n[7])
