"""Exercise the actual private serve/generate/reset glue with recording CABI only."""
from pathlib import Path
import ctypes as C,json,sys,threading,time
from types import SimpleNamespace
import numpy as np
D=Path(__file__).resolve().parent
sys.path[:0]=[str(D),str(D.parent/'model-v1'),str(D.parents[1]/'public-Deep-native-provider-r427/source-v3')]
from public_deep_model import worker
from public_deep_model.native_context import NativeContext
from public_deep_model.request_policy import RequestPolicy,PROMPT_SHA,MODES
from model_api import COUNTERS

class Fn:
 def __init__(self,f):self.f=f;self.argtypes=None;self.restype=None
 def __call__(self,*a):return self.f(*a)
class Library:
 def __init__(self,context):
  self.context=context;self.armed=False;self.mode=0;self.sets=[];self.rows=[];self.clear_count=0;self.reset_count=0
  self.clear()
  self.deep_model_actquant_set_r438=Fn(self.set);self.deep_model_actquant_clear_r438=Fn(self.clear);self.deep_model_actquant_read_r438=Fn(self.read);self.deep_model_actquant_metadata_r438=Fn(lambda:b'{}')
  self.modern_native_engine_reset=Fn(self.reset);self.modern_native_engine_destroy=Fn(lambda ctx:0);self.modern_native_engine_forward=Fn(lambda *a:0)
  for name in ['reset','destroy']:
   f=getattr(self,'modern_native_engine_'+name);f.argtypes=[C.c_void_p];f.restype=C.c_int
  self.modern_native_engine_forward.argtypes=[C.c_void_p];self.modern_native_engine_forward.restype=C.c_int
 def set(self,mode):assert type(mode)is int and mode in[0,1];self.mode=mode;self.armed=True;self.sets.append(mode);return 0
 def clear(self):self.clear_count+=1;self.armed=False;self.mode=0;self.rows=[dict.fromkeys(COUNTERS,0)for _ in range(8)];return 0
 def read(self,owner,out,n):
  assert n==12 and owner in range(8)
  for i,k in enumerate(COUNTERS):out[i]=self.rows[owner][k]
  if self.context.scenario=='missing-hit'and self.mode==1:out[4]=0
  return 0
 def reset(self,ctx):
  assert ctx==self.context.ctx and ctx>2**32;self.reset_count+=1
  if self.context.scenario=='reset-entry-fail'and self.reset_count==1:return 9
  if self.context.scenario=='reset-drain-fail'and self.reset_count==2:return 9
  self.context.current_forward=0;return 0
 def forward(self,part):
  assert self.armed;self.context.current_forward+=1;one=len(part)==1
  for row in self.rows:
   row['entered_valid_MoE']+=43;row['T1']+=43*one;row['armed_mode1_T1']+=43*one*self.mode;row['host_returned_not_GPU_completed']+=43;row['last_mode']=self.mode
   for prefix in ['routed','shared']:
    row[prefix+'_calls']+=43;row[prefix+'_hits']+=43*one*self.mode;row[prefix+'_fallback']+=43*(not(one and self.mode))
  if self.context.scenario in ['forward-fail','reset-drain-fail']:
   return 7,np.zeros((1,128),dtype=np.uint16),[0]
  token=1 if self.context.current_forward==11 else 9
  return 0,np.zeros((1,128),dtype=np.uint16),[token]
class Context(NativeContext):
 def __init__(self,scenario):
  self.scenario=scenario;self.cfg={'num_hidden_layers':43,'vocab_size':128};self.thread=threading.get_ident();self.closed=False;self.ctx=0x1234567800000001;self.current_forward=0;self.calls=0
  lib=Library(self)
  def check(code):
   if code:raise RuntimeError('recording Native code '+str(code))
  self.binding=SimpleNamespace(lib=lib,check=check);self.forward=self.invoke
 def floor(self):self.check_thread();return [[20*1024**3,64*1024**3]for _ in range(8)]
 def invoke(self,part,*args):self.check_thread();self.calls+=1;return self.binding.lib.forward(part)
class Request:
 def __init__(self,index,scenario):
  self.request_id='recording-'+str(index);self.max_new_tokens=128;self.prompt_token_ids=list(range(8489));self.generated_token_ids=[];self.cancel_event=threading.Event();self.abort_reason=None;self.finish_reason=None;self.failure=None
  self._admission_metadata_ready=True
  self.metadata={'history_render_audit':{'Native_prompt_token_ids_SHA256':PROMPT_SHA,'pending_thinking_prefix_open':False},'text_delivery':'P1','user_r438_mode_ignored':1}
  if scenario=='bad-history':self.metadata['history_render_audit']['Native_prompt_token_ids_SHA256']='0'*64
  if scenario=='canceled':self.cancel_event.set()
 def emit_token(self,token):self.generated_token_ids.append(token)
 def fail(self,error):self.failure=error
 def finish(self,reason):self.finish_reason=reason
 def is_finished(self):return self.failure is not None or self.finish_reason is not None
class Engine:
 def __init__(self,requests):self.queue=requests[:];self._requests={r.request_id:r for r in requests};self._lock=threading.Lock();self._admission_lock=threading.RLock()
 def take(self,timeout):return self.queue.pop(0)if self.queue else None
class Run:
 def deadline_check(self,*a):pass
 def stop_requested(self):return False

def case(scenario):
 context=Context(scenario);requests=[Request(i,scenario)for i in range(7 if scenario=='extra-request'else 6)]
 engine=Engine(requests);reasons=SimpleNamespace(LENGTH='length',EOS_TOKEN='eos_token',CANCELED='canceled');report={'requests':[]};config={'gpu_start_epoch':time.time(),'service_seconds':2400,'r438_model_modes':list(MODES),'r438_output_budget':128}
 old=worker.RequestPolicy;policies=[]
 def factory(c,cfg):
  p=RequestPolicy(c,cfg,provider_validator=lambda lib:{'recording_CABI':True,'actual_DSO_or_GPU':False});policies.append(p);return p
 worker.RequestPolicy=factory;failure=None;saves=[]
 try:worker.serve(context,{'engine':engine,'FinishReason':reasons,'identity':{'family':'Deep','upstream_commit':'99f5c90a40f9b2546bd0202cba3397ed1d55c5a0','eos_token_ids':[1]}},config,Run(),threading.Event(),report,lambda:saves.append(len(report['requests'])))
 except BaseException as exc:failure=exc
 finally:worker.RequestPolicy=old
 assert all(r.emit_token.__self__ is r for r in requests),'Original emit function must be restored even on failure'
 lib=context.binding.lib
 if scenario=='six-pass':
  assert failure is None and len(report['requests'])==6 and lib.sets==list(MODES)and context.calls==66 and not lib.armed
  assert all(r.finish_reason=='eos_token'and r.generated_token_ids[-1]==1 for r in requests)
  for row,mode in zip(report['requests'],MODES):
   assert row['r438_mode']==mode and row['normal_reset_before']and row['normal_reset_after']and row['r438_actual_T1_coverage']['passed']
   assert row['r438_actual_T1_coverage']['T1_per_owner']==86 and len(row['r438_actual_owner_counts'])==8
 elif scenario=='extra-request':assert failure is not None and len(report['requests'])==7 and len(lib.sets)==6 and context.calls==66
 elif scenario=='bad-history':assert failure is not None and context.calls==0 and lib.sets==[]
 elif scenario=='missing-hit':assert failure is not None and len(report['requests'])==2 and lib.sets==[0,1]and context.calls==22
 elif scenario=='reset-entry-fail':assert failure is not None and context.calls==0 and lib.reset_count==1 and policies[0].poisoned
 elif scenario=='reset-drain-fail':assert failure is not None and context.calls==1 and lib.reset_count==2 and policies[0].poisoned and report['requests'][0]['r438_no_retry_of_failed_or_poisoned_reset']
 elif scenario=='forward-fail':assert failure is not None and context.calls==1 and lib.reset_count==2 and policies[0].poisoned
 elif scenario=='canceled':assert failure is not None and context.calls==0 and len(report['requests'])==1
 else:raise AssertionError(scenario)
 return {'scenario':scenario,'passed':True,'full_requests_seen':len(report['requests']),'recording_Native_calls':context.calls,'reset_calls':lib.reset_count,'mode_set_calls':lib.sets,'failure_observed':failure is not None,'Native_dlopen':False,'GPU':False}
if __name__=='__main__':
 values=[case(s)for s in ['six-pass','bad-history','extra-request','missing-hit','reset-entry-fail','forward-fail','reset-drain-fail','canceled']]
 assert not any(n in sys.modules for n in ['torch','_infinilm','flash_attn'])
 print(json.dumps({'passed':True,'actual_private_worker_serve_and_original_generate':True,'actual_original_NativeContext_reset_and_thread_method':True,'cases':values,'recording_CABI_only':True,'actual_Model_or_GPU_or_HTTP':False,'six_request_fake_call_total':66}))
