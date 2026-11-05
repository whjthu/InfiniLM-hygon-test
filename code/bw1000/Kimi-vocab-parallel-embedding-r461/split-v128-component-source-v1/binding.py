import ctypes as C
import numpy as np
P=C.c_void_p;U=C.c_uint64;Z=C.c_size_t;I=C.c_int
SPEC={
'kimi_split_probe_create':(P,[I]),'kimi_split_probe_error':(C.c_char_p,[]),'kimi_split_probe_symbol_path':(C.c_char_p,[I]),
'kimi_split_probe_begin':(I,[P,I,P,Z]),'kimi_split_probe_reset':(I,[P,P,Z]),'kimi_split_probe_seed':(I,[P,Z,P,P]),
'kimi_split_probe_query':(I,[P,Z,Z,P,P,P,P,Z,P,Z]),'kimi_split_probe_observe':(I,[P,P,Z]),'kimi_split_probe_audit_pages':(I,[P,P,Z]),'kimi_split_probe_destroy':(I,[P,P,Z])}
def declare(lib):
 for n,(ret,args)in SPEC.items():f=getattr(lib,n);f.restype=ret;f.argtypes=args
 return lib
class API:
 def __init__(self,lib):self.lib=declare(lib);self.ctx=None
 def rc(self,x):
  if x:raise RuntimeError((self.lib.kimi_split_probe_error()or b'Native RC').decode())
 def create(self):
  self.ctx=self.lib.kimi_split_probe_create(0)
  if not self.ctx:raise RuntimeError((self.lib.kimi_split_probe_error()or b'Native create').decode())
 def begin(self,mode,table):self.rc(self.lib.kimi_split_probe_begin(self.ctx,mode,table.ctypes.data,table.size))
 def reset(self,table):self.rc(self.lib.kimi_split_probe_reset(self.ctx,table.ctypes.data,table.size))
 def seed(self,past,k,v):self.rc(self.lib.kimi_split_probe_seed(self.ctx,past,k.ctypes.data,v.ctypes.data))
 def query(self,past,q,k,v):
  count=q.shape[0];guard=np.full(count*3*128+32,0x3f25,dtype=np.uint16);out=guard[16:-16].reshape(count,3,128);seconds=np.zeros(3,dtype=np.float64)
  self.rc(self.lib.kimi_split_probe_query(self.ctx,past,count,q.ctypes.data,k.ctypes.data,v.ctypes.data,out.ctypes.data,out.size,seconds.ctypes.data,3))
  if not np.all(guard[:16]==0x3f25)or not np.all(guard[-16:]==0x3f25):raise AssertionError('CPU output canary')
  return out,seconds.tolist()
 def observe(self):
  out=np.zeros(16+24*8,dtype=np.uint64);self.rc(self.lib.kimi_split_probe_observe(self.ctx,out.ctypes.data,out.size));return out.tolist()
 def audit(self):
  out=np.zeros(2,dtype=np.uint64);self.rc(self.lib.kimi_split_probe_audit_pages(self.ctx,out.ctypes.data,2));return out.tolist()
 def close(self):
  out=np.zeros(5,dtype=np.uint64);ctx=self.ctx;self.ctx=None
  if ctx:self.rc(self.lib.kimi_split_probe_destroy(ctx,out.ctypes.data,5))
  return out.tolist()
