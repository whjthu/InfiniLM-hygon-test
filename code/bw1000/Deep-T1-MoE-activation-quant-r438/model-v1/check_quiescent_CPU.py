"""Actual new request consumer + exact qualified reset/check_thread methods.

Only Native CABI execution/weights are recording leaves. No installed class is
patched, no original Model constructor/forward math is claimed executed here.
"""
from pathlib import Path
import ast,ctypes as C,hashlib,json,threading,types
from quiescent_mode import QuiescentMode
HIGH=0x123456789abcdef0
def case(kind):
 S=Path(__file__).parent;p=S/'original_native_context.py';proof=json.loads((S/'ACTUAL_MODEL_REACHABILITY.json').read_bytes());assert hashlib.sha256(p.read_bytes()).hexdigest()==proof['qualified_native_context_SHA256']
 source=ast.parse(p.read_bytes());cls=next(n for n in source.body if isinstance(n,ast.ClassDef)and n.name=='NativeContext');methods=[n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name in['reset','check_thread']];assert len(methods)==2
 env={'threading':threading};exec(compile(ast.fix_missing_locations(ast.Module(body=methods,type_ignores=[])),'actual_qualified_NativeContext_methods','exec'),env)
 events=[];errors=[];callbacks=[];state=dict(resets=0,done=False,mode=0,armed=False)
 lib=types.SimpleNamespace()
 def install(name,result,args,body):
  def safe(*v):
   try:return body(*v)
   except BaseException as exc:errors.append(repr(exc));return 1
  fn=C.CFUNCTYPE(result,*args)(safe);fn.argtypes=args;fn.restype=result;setattr(lib,name,fn);callbacks.append(fn)
 def reset(ctx):
  assert ctx==HIGH;events.append('actual_typed_reset');state['resets']+=1
  if kind=='entry_reset_failure'or(kind=='cleanup_reset_failure'and state['resets']==2):return 1
  events.append('eight_owner_reset_sync_wait_done');state['done']=True;return 0
 def forward(ctx,*args):
  assert ctx==HIGH;events.append('actual_typed_forward');state['done']=False
  if kind in['forward_failure','cleanup_reset_failure']:return 1
  events.append('eight_owner_forward_completed_and_CPU_copy');state['done']=True;return 0
 def clear():
  assert state['done'];events.append('clear_after_real_boundary');state['armed']=False;state['mode']=0;return 0
 def set_mode(mode):
  assert state['done']and mode in[0,1];events.append('arm_after_real_boundary');state.update(mode=mode,armed=True);return 0
 def read(owner,out,n):
  assert state['done']and n==12;events.append('read'+str(owner))
  for k in range(n):out[k]=0
  out[11]=state['mode'];return 0
 install('modern_native_engine_reset',C.c_int,[C.c_void_p],reset)
 install('modern_native_engine_destroy',C.c_int,[C.c_void_p],lambda ctx:0)
 install('modern_native_engine_forward',C.c_int,[C.c_void_p,C.c_size_t],forward)
 install('deep_model_actquant_set_r438',C.c_int,[C.c_int],set_mode)
 install('deep_model_actquant_clear_r438',C.c_int,[],clear)
 install('deep_model_actquant_read_r438',C.c_int,[C.c_int,C.POINTER(C.c_uint64),C.c_size_t],read)
 install('deep_model_actquant_metadata_r438',C.c_char_p,[],lambda:None)
 def check(code):
  if code:raise RuntimeError('recorded original Native failure')
 context=types.SimpleNamespace(ctx=HIGH,thread=threading.get_ident(),closed=False,binding=types.SimpleNamespace(lib=lib,check=check))
 context.check_thread=types.MethodType(env['check_thread'],context);context.reset=types.MethodType(env['reset'],context)
 def original_forward(tokens):return(lib.modern_native_engine_forward(context.ctx,tokens),None,None)
 context.forward=original_forward;control=QuiescentMode(context);mode=0 if kind=='normal0'else 1
 if kind=='invalid_mode':mode=2
 try:
  with control.request(mode)as owner:
   if kind=='body_error':raise ValueError('primary body error')
   owner.forward(1)
 except BaseException as exc:
  assert kind not in['normal0','normal1'],repr(exc)
  if kind=='body_error':assert isinstance(exc,ValueError)and str(exc)=='primary body error'
  if kind=='cleanup_reset_failure':assert getattr(exc,'cleanup_errors',None)
 else:assert kind in['normal0','normal1']
 assert not errors,errors
 if kind.startswith('normal'):
  assert state['resets']==1 and events.index('eight_owner_reset_sync_wait_done')<events.index('arm_after_real_boundary')<events.index('actual_typed_forward')<events.index('eight_owner_forward_completed_and_CPU_copy')
  assert events[-1]=='clear_after_real_boundary'and len(control.last_counts)==8 and all(x['last_mode']==mode for x in control.last_counts)
  assert not control.poisoned and not state['armed']
 elif kind=='entry_reset_failure':assert state['resets']==1 and events==['actual_typed_reset']and control.poisoned
 elif kind=='cleanup_reset_failure':assert state['resets']==2 and events[-1]=='actual_typed_reset'and control.poisoned and state['armed']
 elif kind=='forward_failure':assert state['resets']==2 and events[-1]=='clear_after_real_boundary'and control.poisoned
 elif kind=='body_error':assert state['resets']==1 and events[-1]=='clear_after_real_boundary'and control.poisoned
 elif kind=='invalid_mode':assert events==[]
 return dict(case=kind,passed=True,events=events,actual_reset_calls=state['resets'],whole_engine_boundary_is_Native_leaf=True)
if __name__=='__main__':
 print(json.dumps(dict(passed=True,actual_qualified_reset_and_thread_methods_consumed=True,actual_new_mode_request_consumer=True,Native_ctx_high64_not_truncated=True,cases=[case(k)for k in ['normal0','normal1','invalid_mode','entry_reset_failure','body_error','forward_failure','cleanup_reset_failure']],Native_loaded=False,Native_or_Runtime_or_GPU_executed=False,whole_original_Model_module_imported=False)))
