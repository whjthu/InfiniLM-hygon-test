from pathlib import Path
import ctypes as C,json,sys,threading,time
from types import SimpleNamespace
import numpy as np
from public_deep_model.native_context import NativeContext
from public_deep_model.request_policy import PROMPT_SHA
from public_deep_model.fullpre_control import LEGACY,FULL

class Fn:
 def __init__(self,f):self.f=f;self.argtypes=None;self.restype=None
 def __call__(self,*a):return self.f(*a)


class Library:
 def __init__(self,context):
  self.context=context;self.mode=0;self.old_mode=0;self.events=[];self.reset_count=0;self.pointer_values=[];self.callbacks=[]
  self.old=[dict.fromkeys(LEGACY,0)for _ in range(8)]
  for row in self.old:row.update(created=86,ready=86)
  self.full=[dict.fromkeys(FULL,0)for _ in range(8)]
  def callback(name,restype,args,fun):
   cb=C.CFUNCTYPE(restype,*args)(fun);cb.argtypes=args;cb.restype=restype;self.callbacks.append(cb);setattr(self,name,cb)
  callback('deep_hc_fullpre_enable_r377',C.c_int,[C.c_int],self.set)
  callback('deep_hc_fullpre_enabled_r377',C.c_int,[],lambda:self.mode)
  callback('deep_hc_fullpre_clear_r377',C.c_int,[],self.clear)
  callback('deep_hc_fullpre_stats_r377',C.c_int,[C.POINTER(C.c_uint64),C.c_int],lambda p,n:self.read(p,n,self.full,FULL))
  callback('deep_hc_model_epilogue_set_r366',C.c_int,[C.c_int],self.oldset)
  callback('deep_hc_model_epilogue_get_r366',C.c_int,[],lambda:self.old_mode)
  callback('deep_hc_model_epilogue_counters_r366',C.c_int,[C.POINTER(C.c_uint64),C.c_int],lambda p,n:self.read(p,n,self.old,LEGACY))
  callback('modern_native_engine_reset',C.c_int,[C.c_void_p],self.reset)
  callback('modern_native_engine_destroy',C.c_int,[C.c_void_p],lambda ctx:0)
  self.modern_native_engine_forward=Fn(lambda *args:0)
  self.modern_native_engine_forward.argtypes=[C.c_void_p];self.modern_native_engine_forward.restype=C.c_int
 def set(self,mode):
  if mode not in[0,1]or not self.context.drained:return 9
  self.mode=mode;self.events.append(('mode',mode,self.reset_count));return 0
 def oldset(self,mode):
  if mode!=0 or not self.context.drained:return 9
  self.old_mode=mode;self.events.append(('r366',mode,self.reset_count));return 0
 def clear(self):
  if self.mode or not self.context.drained:return 9
  self.full=[dict.fromkeys(FULL,0)for _ in range(8)];self.events.append(('clear',self.reset_count));return 0
 def read(self,p,n,rows,names):
  if n!=8*len(names)or not self.context.drained:return 9
  self.pointer_values.append(C.cast(p,C.c_void_p).value)
  for d,row in enumerate(rows):
   for k,name in enumerate(names):p[d*len(names)+k]=row[name]
  if self.context.scenario=='missing-hit'and names is FULL and self.mode==1:p[3]=0
  return 0
 def reset(self,ctx):
  self.events.append(('reset',ctx));self.reset_count+=1
  if ctx!=self.context.ctx or ctx<=2**32:return 9
  if self.context.scenario=='reset-drain-failure'and self.reset_count==2:return 9
  self.context.drained=True;self.context.current_forward=0;return 0
 def forward(self,part,total):
  self.context.current_forward+=1;self.context.drained=False;one=len(part)==1
  self.events.append(('forward',len(part),self.mode,self.old_mode))
  for old,full in zip(self.old,self.full):
   old['pre']+=86;old['eligible_bf16_t1']+=86*one;old['other_t']+=86*(not one)
   if self.mode:
    full['observed']+=86;full['eligible']+=86*one;full['selected']+=86*one;full['completed']+=86*one;full['prefill']+=86*(not one)
  self.context.drained=True
  if self.context.scenario=='forward-failure':return 7,np.zeros((1,128),dtype=np.uint16),[0]
  token=1 if total>=8491 else 9
  return 0,np.zeros((1,128),dtype=np.uint16),[token]


class Context(NativeContext):
 def __init__(self,scenario):
  self.scenario=scenario;self.cfg={'num_hidden_layers':43,'vocab_size':128};self.thread=threading.get_ident();self.closed=False;self.ctx=0x1234567800000001;self.current_forward=0;self.calls=0;self.drained=True
  lib=Library(self)
  def check(code):
   if code:raise RuntimeError('recording Native code '+str(code))
  self.binding=SimpleNamespace(lib=lib,check=check);self.forward=self.invoke
 def floor(self):self.check_thread();return [[20*1024**3,64*1024**3]for _ in range(8)]
 def invoke(self,part,positions,rows,past,total,offsets):
  self.check_thread()
  assert rows==[0]and offsets==[0,len(part)]and total==[past[0]+len(part)]
  assert positions==list(range(past[0],total[0]))and 1<=len(part)<=4096 and total[0]<=32768
  self.calls+=1;return self.binding.lib.forward(part,total[0])


class Request:
 def __init__(self,index,scenario,context):
  self.request_id='recording-'+str(index);self.max_new_tokens=128;self.prompt_token_ids=list(range(8489));self.generated_token_ids=[];self.cancel_event=threading.Event();self.abort_reason=None;self.finish_reason=None;self.failure=None;self.context=context;self.scenario=scenario
  self._admission_metadata_ready=True
  self.metadata={'history_render_audit':{'Native_prompt_token_ids_SHA256':PROMPT_SHA,'pending_thinking_prefix_open':False},'text_delivery':'P1','public_mode_does_not_arm':1}
 def emit_token(self,token):
  self.generated_token_ids.append(token)
  if self.scenario in ['disconnect','timeout']and len(self.generated_token_ids)==2:
   self.abort_reason=self.scenario;self.cancel_event.set()
 def fail(self,error):self.failure=error
 def finish(self,reason):
  assert self.context.drained and self.context.binding.lib.mode==0,'Owner finish must follow reset/drain and fullpre clear'
  self.finish_reason=reason;self.context.binding.lib.events.append(('finish',reason))
 def is_finished(self):return self.failure is not None or self.finish_reason is not None


class Engine:
 def __init__(self,requests):self.queue=requests[:];self._requests={r.request_id:r for r in requests};self._lock=threading.Lock();self._admission_lock=threading.RLock()
 def take(self,timeout):return self.queue.pop(0)if self.queue else None


class Run:
 def deadline_check(self,*args):pass
 def stop_requested(self):return False

