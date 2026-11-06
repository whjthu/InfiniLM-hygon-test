"""Recording verifies FFI widths/order/ownership, not C++ or device behavior."""
import ctypes as C
from types import SimpleNamespace
from binding import Trial,SIGNATURES,PREFIX,V,Z,I,U,D

high=(1<<60)+0x12345
calls=[];callbacks=[];buffers=[]

def ptr_json(payload):
 b=C.create_string_buffer(payload);buffers.append(b);return C.addressof(b)

def callback(name):
 def f(*args):
  calls.append((name,args))
  if name=='create':assert args[1:]==(0,1);return high
  if name=='error':return ptr_json(b'error')
  assert args[0]==high,(name,args[0])
  if name in ('parameters','tensor','observe'):return ptr_json(b'{}')
  if name=='prepare':assert args[1:3]==(32767,1);args[-1][0]=0.125
  if name=='commit':args[1][0]=0.25
  if name=='destroy':assert args[2]==4;args[1][2]=1
  return 0
 return f

lib=SimpleNamespace()
for name,(args,ret) in SIGNATURES.items():
 cb=C.CFUNCTYPE(ret,*args)(callback(name));callbacks.append(cb);setattr(lib,PREFIX+name+'_r483',cb)
t=Trial(lib,{'max_position_embeddings':32768},0,1)
assert t.handle.value==high
assert t.read('parameters')=={}
b=C.create_string_buffer(b'x'*8);t.call('load',b'w',b,8);t.call('postload')
p=(C.c_int64*1)(32767);seconds=D()
t.call('prepare',32767,1,b,8,b,8,p,1,C.byref(seconds));assert seconds.value==0.125
assert t.read('tensor',b'indices',0)=={}
t.call('copy',b'indices',0,b,8);t.call('commit',C.byref(seconds));assert seconds.value==0.25
t.call('abort');t.call('reset');assert t.read('observe')=={}
assert t.close()[2]==1 and t.handle is None
before=len(calls)
try:t.read('observe');raise AssertionError('closed read accepted')
except RuntimeError:pass
try:t.close();raise AssertionError('double close accepted')
except RuntimeError:pass
assert len(calls)==before
assert len(SIGNATURES)==13 and len({n for n,_ in calls})==12 # error is only called on a Native failure
print('PASS: 13 exact signatures; high64 outer handle, buffer/clock/close paths and closed-owner refusal; no Native/SDK/GPU')
