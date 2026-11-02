"""Thin actual C ABI and Pending-proof readers, reused by CPU consumer and GPU plan."""
import ctypes as C,json,numpy as np
P=C.c_void_p;Z=C.c_size_t;I=C.c_int;U=C.c_uint64
def configure(lib):
 f=lambda name:getattr(lib,'deep_c4_lease_'+name+'_r390')
 f('error').restype=C.c_char_p
 for name,args,result in [('id_size',[],Z),('id',[P],I),('create',[I,C.c_char_p,P],P),('keys',[P],C.c_char_p),('load',[P,C.c_char_p,P,Z,C.POINTER(Z),I],I),('finish',[P],I),('reset',[P],I),('prefix',[P,Z,Z,P,P,P,P],I),('prepare',[P,Z,P,P,P,P,C.POINTER(U)],I),('score',[P,U,I],I),('commit',[P,U],I),('identity',[P,U],C.c_char_p),('proof_info',[P,U,I,C.POINTER(Z),C.POINTER(Z),C.POINTER(I)],I),('proof_read',[P,U,I,P,Z],I),('destroy',[P],I),('shutdown',[],I)]:
  f(name).argtypes=args;f(name).restype=result
 return f
def snapshot(f,check,ctx,token):
 out={}
 for which in range(17):
  shape=(Z*5)();size=Z();dtype=I();check(f('proof_info')(ctx,token,which,shape,C.byref(size),C.byref(dtype)))
  if not size.value:out[which]=None;continue
  data=np.empty(size.value,np.uint8);check(f('proof_read')(ctx,token,which,P(data.ctypes.data),size.value))
  out[which]=dict(shape=tuple(shape[i+1]for i in range(shape[0])),raw=data,dtype=dtype.value)
 return out
def identity(f,ctx,token):
 value=f('identity')(ctx,token);assert value is not None,'Lease identity callback failed'
 return json.loads(value)
def begin(f,check,ctx,past,x,q,pos,stream):
 token=U();check(f('prepare')(ctx,past,x,q,pos,stream,C.byref(token)))
 return token.value,dict(identity_before=identity(f,ctx,token.value),before=snapshot(f,check,ctx,token.value))
def score(f,check,ctx,token,mode,packet):
 check(f('score')(ctx,token,mode));name='after_sdk'if mode==0 else'after_paged'
 packet['identity_'+name]=identity(f,ctx,token);packet[name]=snapshot(f,check,ctx,token)
def commit(f,check,ctx,token,packet):
 check(f('commit')(ctx,token));packet['identity_post_commit']=identity(f,ctx,token);packet['post_commit']=snapshot(f,check,ctx,token)
