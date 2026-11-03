"""Experimental single Engine mode consumer, using qualified NativeContext reset.

All request calls must use this entry in the private frozen owner. No public
request parameter can arm this policy. The shared admission lock covers reset,
publication, all synchronous original forwards, readback and clear. MoE's local
active flag is only an additional rejection; never GPU-completion evidence.
"""
import ctypes as C
from contextlib import contextmanager
import threading
from model_api import bind,COUNTERS
_ADMISSION=threading.RLock()

class QuiescentMode:
 def __init__(self,native_context):
  self.native=native_context;self.lib=bind(native_context.binding.lib)
  self.original_forward=native_context.forward;self.active=False;self.poisoned=False;self.owner_thread=None;self.quiescent=False;self.last_counts=None
  # Existing full qualified binder owns all old declarations, especially ctx.
  for name in ['reset','destroy']:
   f=getattr(self.lib,'modern_native_engine_'+name);assert f.argtypes==[C.c_void_p]and f.restype is C.c_int
  assert self.lib.modern_native_engine_forward.argtypes[0]is C.c_void_p and self.lib.modern_native_engine_forward.restype is C.c_int
 def check(self,code):
  if code:raise RuntimeError('Model opt-in policy returned '+str(code))
 def reset_and_drain(self):
  self.native.check_thread();self.native.reset()
  # Actual CAPI reset -> InferEngine::reset_cache submission_mutex and wait all
  # eight owners -> NativeSparseModel::reset_cache owner r.syncStream at both
  # boundaries. Source/ELF pins in ACTUAL_MODEL_REACHABILITY.json. Not a new sync.
  self.quiescent=True
 def forward(self,*args):
  if not self.active or self.owner_thread!=threading.get_ident()or self.poisoned:raise RuntimeError('Model forward outside the admitted mode request')
  self.quiescent=False
  result=self.original_forward(*args)
  # Exact d206/public427 forward returns code, CPU copied BF16 logits and IDs.
  # Code0 means the original InferEngine/RankWorker completed-output path,
  # NativeSparseModel r.syncStream and CAPI copy_cpu have returned, not merely
  # MoE HostReturned or an atomic active=false observation.
  if not isinstance(result,tuple)or len(result)!=3 or result[0]!=0:raise RuntimeError('Original qualified Native forward failed')
  self.quiescent=True;return result
 @contextmanager
 def request(self,mode):
  if type(mode)is not int or mode not in[0,1]:raise ValueError('Frozen Model mode must be integer0/1')
  with _ADMISSION:
   if self.poisoned or self.active:raise RuntimeError('Model mode context is poisoned or already active')
   primary=None;cleanup=[];self.last_counts=None;entered=False;self.quiescent=False
   try:
    self.reset_and_drain();self.check(self.lib.deep_model_actquant_clear_r438());self.check(self.lib.deep_model_actquant_set_r438(mode))
    self.active=True;self.owner_thread=threading.get_ident();entered=True;yield self
   except BaseException as exc:primary=exc
   finally:
    self.active=False;self.owner_thread=None
    if entered:
     try:
      if not self.quiescent:self.reset_and_drain()
      # read/clear only after actual owner completion/reset+drain succeeded.
      self.last_counts=[]
      for owner in range(8):
       values=(C.c_uint64*len(COUNTERS))();self.check(self.lib.deep_model_actquant_read_r438(owner,values,len(values)));self.last_counts.append(dict(zip(COUNTERS,values)))
      self.check(self.lib.deep_model_actquant_clear_r438())
     except BaseException as exc:self.poisoned=True;cleanup.append(repr(exc))
   if primary is not None:
    if cleanup:setattr(primary,'cleanup_errors',cleanup)
    self.poisoned=True;raise primary
   if cleanup:raise RuntimeError('Model mode cleanup failed: '+repr(cleanup))
