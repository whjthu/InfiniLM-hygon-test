from pathlib import Path
import ctypes as C,json,sys,types
import numpy as np
from binding import API,SPEC
from schedule import run_schedule,CASES,ORDER
CTX=0x1234567887654321;trace=[];state=dict(mode=-1,queries=0,resets=0);keep=[];lib=types.SimpleNamespace()
def callback(name):
 def f(*args):
  if name=='kimi_split_probe_error':return None
  if name=='kimi_split_probe_create':trace.append(['create',args[0]]);return CTX
  if name=='kimi_split_probe_symbol_path':return None
  assert args[0]==CTX;trace.append([name,*[int(v or 0)for v in args[1:]if isinstance(v,int)]])
  if name=='kimi_split_probe_begin':state['mode']=args[1]
  elif name=='kimi_split_probe_reset':state['resets']+=1
  elif name=='kimi_split_probe_query':
   _,past,count,q,k,v,out,words,sec,ns=args;assert count in [1,512]and words==count*3*128 and ns==3
   C.memset(out,0,words*2);state['queries']+=1
  elif name=='kimi_split_probe_observe':
   out=C.cast(args[1],C.POINTER(C.c_uint64));out[4]=state['mode'];out[5]=24;out[6]=state['queries'];out[8]=state['resets']
  elif name=='kimi_split_probe_audit_pages':C.cast(args[1],C.POINTER(C.c_uint64))[0]=4
  elif name=='kimi_split_probe_destroy':C.cast(args[1],C.POINTER(C.c_uint64))[2]=1
  return 0
 return f
for name,(ret,args)in SPEC.items():
 fn=C.CFUNCTYPE(ret,*args)(callback(name));setattr(lib,name,fn);keep.append(fn)
a=np.zeros((32768,3,192),np.uint16);v=np.zeros((32768,3,128),np.uint16);events=[];records,closed=run_schedule(API(lib),(a,a,v),lambda *x:events.append(x))
assert len(records)==20 and sum(r['count']for r in records)==8196 and closed[2]==1
assert [r['mode']for r in records[::5]]==ORDER and sum(t[0]=='kimi_split_probe_seed'for t in trace)==16
assert sum(t[0]=='kimi_split_probe_reset'for t in trace)==12 and sum(t[0]=='kimi_split_probe_query'for t in trace)==20
assert trace[-1][0]=='kimi_split_probe_destroy'and events[-1][0]=='Runtime.closed'
# This is the actual schedule/binding with ctypes Native leaf callbacks; no output agreement qualification.
print(json.dumps(dict(passed=True,actual_entry_schedule=True,actual_high64_CAPI=True,recorded_query_calls=20,recorded_tokens=8196,actual_Native_leaf='recording',SDK=False,Runtime=False,GPU=0,numeric_agreement_qualified=False)))
