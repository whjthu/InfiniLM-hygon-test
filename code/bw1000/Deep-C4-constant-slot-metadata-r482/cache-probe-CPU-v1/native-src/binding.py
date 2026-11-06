"""Typed high64-safe consumer of the real Native cache probe; device use only."""
import ctypes as C
import numpy as np

def declare(lib):
 specs={
 'error':([],C.c_char_p),'create':([C.c_int],C.c_void_p),
 'seed':([C.c_void_p,C.c_size_t,C.c_size_t,C.c_size_t,C.c_void_p,C.c_void_p],C.c_int),
 'gather':([C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t],C.c_int),
 'guard':([C.c_void_p,C.c_size_t,C.c_int],C.c_int),'reset':([C.c_void_p,C.c_size_t],C.c_int),
 'raw':([C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t],C.c_int),
 'observe':([C.c_void_p,C.c_void_p,C.c_size_t],C.c_int),'destroy':([C.c_void_p,C.c_void_p,C.c_size_t],C.c_int)}
 for name,(args,ret)in specs.items():f=getattr(lib,'deep_slot_probe_'+name+'_r482');f.argtypes=args;f.restype=ret
class API:
 def __init__(self,lib,mode):
  if mode not in (0,1):raise ValueError('fixed mode')
  declare(lib);self.lib=lib;self.ctx=lib.deep_slot_probe_create_r482(mode);self.closed=None
  if not self.ctx:raise RuntimeError(self.error())
 def error(self):return(self.lib.deep_slot_probe_error_r482()or b'cacheprobe error').decode()
 def rc(self,value):
  if value:raise RuntimeError(self.error())
 def seed(self,layer,begin,values,scales):
  v,s=np.asarray(values),np.asarray(scales);n=v.shape[0]
  if v.dtype!=np.uint8 or s.dtype!=np.float32 or v.shape!=(n,128)or s.shape!=(n,1)or not v.flags.c_contiguous or not s.flags.c_contiguous or not(0<n<=8192 and 0<=begin<8192 and n<=8192-begin):raise ValueError('original packed input')
  self.rc(self.lib.deep_slot_probe_seed_r482(self.ctx,layer,begin,n,v.ctypes.data,s.ctypes.data))
 def gather(self,layer,slots):
  ids=np.asarray(slots)
  if ids.dtype!=np.int64 or ids.ndim!=1 or not ids.flags.c_contiguous or not 0<ids.size<=8192:raise ValueError('actual int64 contiguous IDs')
  n=ids.size;vg=np.full(n*128+32,0xa5,np.uint8);sg=np.full(n+32,-3.25,np.float32);v=vg[16:-16].reshape(n,128);s=sg[16:-16].reshape(n,1);seconds=np.zeros(3,np.float64)
  self.rc(self.lib.deep_slot_probe_gather_r482(self.ctx,layer,ids.ctypes.data,n,v.ctypes.data,v.nbytes,s.ctypes.data,n,seconds.ctypes.data,3))
  if not(np.all(vg[:16]==0xa5)and np.all(vg[-16:]==0xa5)and np.all(sg[:16]==-3.25)and np.all(sg[-16:]==-3.25)):raise AssertionError('CPU output canaries')
  return v.copy(),s.copy(),seconds.tolist()
 def guard(self,layer,kind):self.rc(self.lib.deep_slot_probe_guard_r482(self.ctx,layer,kind))
 def reset(self,layer):self.rc(self.lib.deep_slot_probe_reset_r482(self.ctx,layer))
 def raw(self,layer):
  out=np.zeros(128*8448,np.uint8);self.rc(self.lib.deep_slot_probe_raw_r482(self.ctx,layer,out.ctypes.data,out.size));return out
 def observe(self):
  out=np.zeros(16,np.uint64);self.rc(self.lib.deep_slot_probe_observe_r482(self.ctx,out.ctypes.data,16));return out.tolist()
 def close(self):
  ctx=self.ctx;self.ctx=None
  if ctx:
   out=np.zeros(4,np.uint64);rc=self.lib.deep_slot_probe_destroy_r482(ctx,out.ctypes.data,4);self.closed={'rc':int(rc),'values':out.tolist()};self.rc(rc)
  return self.closed
