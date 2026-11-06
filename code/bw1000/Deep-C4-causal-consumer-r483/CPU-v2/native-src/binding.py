"""Typed C ABI binding only; loading a library is an explicit device-driver action."""
import ctypes as C
import json

V=C.c_void_p;Z=C.c_size_t;I=C.c_int;U=C.c_uint64;D=C.c_double;S=C.c_char_p
PREFIX='deep_c4_probe_'
SIGNATURES={
 'error':([],V),
 'create':([S,Z,I],V),
 'parameters':([V],V),
 'load':([V,S,V,Z],I),
 'postload':([V],I),
 'prepare':([V,Z,Z,V,Z,V,Z,C.POINTER(C.c_int64),Z,C.POINTER(D)],I),
 'tensor':([V,S,Z],V),
 'copy':([V,S,Z,V,Z],I),
 'commit':([V,C.POINTER(D)],I),
 'abort':([V],I),
 'reset':([V],I),
 'observe':([V],V),
 'destroy':([V,C.POINTER(U),Z],I),
}

def bind(library):
 result={}
 for name,(args,ret) in SIGNATURES.items():
  f=getattr(library,PREFIX+name+'_r483');f.argtypes=args;f.restype=ret;result[name]=f
 return result

class Trial:
 def __init__(self,library,config,layer,mode):
  self.library=library;self.api=bind(library);self.handle=None
  payload=json.dumps(config,separators=(',',':')).encode()
  value=self.api['create'](payload,layer,mode)
  if not value:raise RuntimeError(self.error())
  self.handle=V(value)
 def error(self):
  p=self.api['error']();return C.string_at(p).decode() if p else 'missing error'
 def call(self,name,*args):
  if self.handle is None:raise RuntimeError('closed owner')
  rc=self.api[name](self.handle,*args)
  if rc:raise RuntimeError(self.error())
 def read(self,name,*args):
  if self.handle is None:raise RuntimeError('closed owner')
  p=self.api[name](self.handle,*args)
  if not p:raise RuntimeError(self.error())
  return json.loads(C.string_at(p))
 def close(self):
  if self.handle is None:raise RuntimeError('already closed')
  out=(U*4)();self.call('destroy',out,4);self.handle=None
  return list(out)
