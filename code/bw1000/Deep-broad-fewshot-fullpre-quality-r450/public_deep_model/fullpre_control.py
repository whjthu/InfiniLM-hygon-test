"""Owner-thread whole-engine reset/drain around one private HC-fullpre mode."""
import ctypes as C,threading
from contextlib import contextmanager
LEGACY=('created','ready','pre','eligible_bf16_t1','completed','other_t','f32_t1')
FULL=('observed','eligible','selected','completed','prefill','f32_t1','layout','selected_exceptions','fallback_exceptions','active')
FULL_NAMES=['deep_hc_fullpre_'+n+'_r377'for n in ['enable','enabled','clear','stats']]
LEGACY_NAMES=['deep_hc_model_epilogue_'+n+'_r366'for n in ['set','get','counters']]
_ADMISSION=threading.RLock()
def bind(lib):
 for name,args,result in [
 ('deep_hc_fullpre_enable_r377',[C.c_int],C.c_int),('deep_hc_fullpre_enabled_r377',[],C.c_int),('deep_hc_fullpre_clear_r377',[],C.c_int),('deep_hc_fullpre_stats_r377',[C.POINTER(C.c_uint64),C.c_int],C.c_int),
 ('deep_hc_model_epilogue_set_r366',[C.c_int],C.c_int),('deep_hc_model_epilogue_get_r366',[],C.c_int),('deep_hc_model_epilogue_counters_r366',[C.POINTER(C.c_uint64),C.c_int],C.c_int)]:
  f=getattr(lib,name);f.argtypes=args;f.restype=result
 return lib
def read(lib,name,names):
 values=(C.c_uint64*(8*len(names)))();rc=getattr(lib,name)(values,len(values))
 if rc:raise RuntimeError('HC typed counter read returned '+str(rc))
 return [dict(zip(names,list(values)[d*len(names):(d+1)*len(names)]))for d in range(8)]
def read_legacy(lib):return read(lib,'deep_hc_model_epilogue_counters_r366',LEGACY)
def read_full(lib):return read(lib,'deep_hc_fullpre_stats_r377',FULL)
class QuiescentMode:
 def __init__(self,context):
  self.native=context;self.lib=bind(context.binding.lib);self.original_forward=context.forward
  self.active=False;self.poisoned=False;self.owner_thread=None;self.quiescent=False;self.last_counts=None;self.normal_reset_after=False
  for name in ['reset','destroy']:
   f=getattr(self.lib,'modern_native_engine_'+name);assert f.argtypes==[C.c_void_p]and f.restype is C.c_int
  assert self.lib.modern_native_engine_forward.argtypes[0]is C.c_void_p and self.lib.modern_native_engine_forward.restype is C.c_int
 def check(self,rc):
  if rc:raise RuntimeError('HC fullpre policy returned '+str(rc))
 def reset_and_drain(self):
  self.native.check_thread();self.native.reset();self.quiescent=True
  # Original CAPI reset holds InferEngine submission mutex, waits all8 owners,
  # and each NativeSparseModel reset syncs owner stream. No atomic substitutes.
 def forward(self,*args):
  if not self.active or self.owner_thread!=threading.get_ident()or self.poisoned:raise RuntimeError('HC forward outside the admitted owner request')
  self.quiescent=False;result=self.original_forward(*args)
  if not isinstance(result,tuple)or len(result)!=3 or result[0]!=0:raise RuntimeError('Original Native forward failed')
  # Qualified CAPI completed-output path includes each owner sync and CPU copy.
  self.quiescent=True;return result
 @contextmanager
 def request(self,mode):
  if type(mode)is not int or mode not in[0,1]:raise ValueError('Frozen fullpre mode must be integer0/1')
  with _ADMISSION:
   if self.poisoned or self.active:raise RuntimeError('HC mode owner poisoned or active')
   primary=None;cleanup=[];entered=False;self.last_counts=None;self.normal_reset_after=False;self.quiescent=False
   try:
    self.reset_and_drain();self.check(self.lib.deep_hc_fullpre_enable_r377(0));self.check(self.lib.deep_hc_model_epilogue_set_r366(0));self.check(self.lib.deep_hc_fullpre_clear_r377())
    before=read_legacy(self.lib);assert all(r['created']==r['ready']==86 for r in before)
    self.check(self.lib.deep_hc_fullpre_enable_r377(mode));self.active=True;self.owner_thread=threading.get_ident();entered=True;yield self
   except BaseException as error:primary=error
   finally:
    self.active=False;self.owner_thread=None
    if entered:
     try:
      self.quiescent=False;self.reset_and_drain();self.normal_reset_after=True
      after=read_legacy(self.lib);full=read_full(self.lib)
      self.last_counts=[dict(owner=d,legacy={k:after[d][k]-before[d][k]for k in LEGACY},fullpre=full[d],created=after[d]['created'],ready=after[d]['ready'])for d in range(8)]
      assert self.lib.deep_hc_model_epilogue_get_r366()==0
      self.check(self.lib.deep_hc_fullpre_enable_r377(0));self.check(self.lib.deep_hc_fullpre_clear_r377())
     except BaseException as error:self.poisoned=True;cleanup.append(repr(error))
   if primary is not None:
    if cleanup:setattr(primary,'cleanup_errors',cleanup)
    self.poisoned=True;raise primary
   if cleanup:raise RuntimeError('HC mode cleanup failed: '+repr(cleanup))
