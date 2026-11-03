"""Actual typed driver/libffi high-pointer glue; all device actions are leaves."""
import ctypes as C,json,threading
from types import SimpleNamespace
from interface import SPECS,bind
from driver_core import forward,proof,owner_session
HIGH=0x123456789abcdef0
def main():
 keep=[];error=C.create_string_buffer(b'injected original failure');local=threading.local();lib=SimpleNamespace();callback_errors=[]
 for name,(result,args)in SPECS.items():
  if name=='error':callback=C.CFUNCTYPE(C.c_void_p)(lambda:C.addressof(error))
  else:
   def fn(*values,_name=name):
    if _name=='create':return HIGH
    if values and _name not in ['id','dtype']:assert values[0]==HIGH
    if _name in ['destroy','shutdown']:
     local.cleanup=getattr(local,'cleanup',[])+[_name];return int(_name=='destroy'and getattr(local,'destroy_fail',False))
    if _name=='forward':
     ctx,tokens,x,stream,mode,capture=values;assert x>2**32 and stream>2**32;local.tokens=tokens;local.mode=mode
     return 1 if getattr(local,'fail',False)else 0
    if _name=='stats':
     values[1][0]=.001;values[1][1]=.002
     for r in range(2):values[2][r*3]=1;values[2][r*3+1]=int(local.tokens==1 and local.mode==1);values[2][r*3+2]=1-values[2][r*3+1]
     values[2][6]=threading.get_ident();return 0
    if _name=='proof_info':values[2][0]=2;values[2][1]=1;values[2][2]=4;values[3][0]=16;values[4][0]=99;return 0
    if _name=='proof_read':C.memset(values[2],7,values[3]);return 0
    if _name=='keys':return None
    return 0
   def checked(*values,_fn=fn,_result=result):
    try:return _fn(*values)
    except BaseException as exc:callback_errors.append(repr(exc));return None if _result in (C.c_void_p,C.c_char_p)else -1
   callback=C.CFUNCTYPE(result,*args)(checked)
  keep.append(callback);setattr(lib,'deep_actquant_'+name+'_r438',callback)
 bind(lib);assert lib.deep_actquant_create_r438(0,b'{}',None)==HIGH
 out=[];failures=[];done=[]
 def owner_body():
  for mode in [0,1,1,0]:out.append(forward(lib,HIGH,HIGH-1,HIGH-2,tokens=1,mode=mode))
  forward(lib,HIGH,HIGH-1,HIGH-2,tokens=2,mode=1)
  p=proof(lib,HIGH,0);assert p['shape']==(1,4)and p['dtype']==99 and p['bytes']==16 and p['raw']==bytes([7])*16
  counts={'attempted':0,'completed':0};local.fail=True
  try:forward(lib,HIGH,HIGH-1,HIGH-2,tokens=1,mode=1,counts=counts)
  except RuntimeError:pass
  else:raise AssertionError('Original nonzero forward was accepted')
  finally:local.fail=False
  assert counts=={'attempted':1,'completed':0}
 def owner():
  try:
   owner_session(lib,0,b'{}',None,lambda lib,ctx:None,lambda lib,ctx:owner_body())
   assert local.cleanup==['destroy','shutdown'];done.append(threading.get_ident())
  except BaseException as exc:failures.append(repr(exc))
 threads=[threading.Thread(target=owner)for _ in range(8)]
 for t in threads:t.start()
 for t in threads:t.join()
 assert len(out)==32 and len(done)==8 and not failures and not callback_errors,(len(out),len(done),failures,callback_errors)
 local.destroy_fail=True;local.cleanup=[]
 def injected(lib,ctx):raise ValueError('primary body failure')
 try:owner_session(lib,0,b'{}',None,lambda lib,ctx:None,injected)
 except ValueError as exc:assert str(exc)=='primary body failure'
 else:raise AssertionError('primary error lost')
 assert local.cleanup==['destroy','shutdown']and not callback_errors
 print(json.dumps(dict(passed=True,actual_libffi_high64ctx=True,actual_owner_threads=8,actual_driver_forward_calls=40,nonzero_forward_not_completed=True,proof_shape_dtype_bytes_consumed=True,actual_create_load_finish_body_destroy_shutdown_handoff=True,primary_failure_retained_and_shutdown_attempted_after_destroy_error=True,Runtime_or_Native_loaded=False,GPU=False)))
if __name__=='__main__':main()
