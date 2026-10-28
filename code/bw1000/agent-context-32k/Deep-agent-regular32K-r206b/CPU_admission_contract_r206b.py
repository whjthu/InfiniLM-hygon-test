"""Bounded CPU exercise of the real private Bridge AST; no model/backend imports."""
import ast,copy,hashlib,json,pathlib,threading,time
from types import SimpleNamespace
from native_engine_bridge_r143c import NativeEngineBridge
from request_contract_r206b import check_sampling

D=pathlib.Path(__file__).resolve().parent

def run():
 source=(D/'worker_r206b.py').read_bytes();tree=ast.parse(source)
 bridge=next(n for n in ast.walk(tree)if isinstance(n,ast.ClassDef)and n.name=='Bridge')
 snapshot=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='admission_snapshot')
 # Compile the exact runtime class and snapshot, not a duplicate lock implementation.
 scope=dict(NativeEngineBridge=NativeEngineBridge,threading=threading,copy=copy,
            check_sampling=check_sampling,request_constraints=lambda data,tools:'',
            hashlib=hashlib,json=json)
 exec(compile(ast.Module(body=[bridge,snapshot],type_ignores=[]),str(D/'worker_r206b.py'),'exec'),scope)
 Bridge=scope['Bridge'];snapshot=scope['admission_snapshot']
 reasons=SimpleNamespace(ERROR='ERROR',TIMEOUT='TIMEOUT',CANCELED='CANCELED')
 prompt='CPU admission interleaving fixture; not an actual Model prompt'
 def make(bad=False):
  def encode(messages,data,generation,kwargs):
   data['_private_history_audit_r206b']={'qualified':True,'actual_prompt_SHA256':'0'*64 if bad else hashlib.sha256(prompt.encode()).hexdigest()}
   return prompt,[10,20,30]
  return Bridge(None,32768,reasons,encode_chat=encode,default_max_tokens=4096)
 def add(engine,name):
  return engine.add_chat_request(messages=[],sampling_params=SimpleNamespace(max_tokens=4096,temperature=0.,stop=None),request_id=name,request_data={'max_tokens':4096,'user':'CPU-only-label'})
 # Pause actual post-enqueue prompt hashing, after base _pending.put(req).
 paused=threading.Event();release=threading.Event();taken=threading.Event();attempted=threading.Event();done=threading.Event();out={};errors=[]
 class PausedHash:
  @staticmethod
  def sha256(value):
   if value==prompt.encode():
    paused.set()
    if not release.wait(3):raise TimeoutError('CPU admission fixture release deadline')
   return hashlib.sha256(value)
 scope['hashlib']=PausedHash;engine=make()
 def admit():
  try:out['request']=add(engine,'CPU-race')
  except BaseException as exc:errors.append(type(exc).__name__)
 def own():
  try:
   req=engine.take(timeout=3);taken.set();attempted.set();out['snapshot']=snapshot(engine,req);out['ready']=req._admission_metadata_ready;done.set()
  except BaseException as exc:errors.append(type(exc).__name__)
 a=threading.Thread(target=admit);o=threading.Thread(target=own)
 a.start()
 try:
  assert paused.wait(3),'Actual post-enqueue metadata pause not reached'
  o.start();assert taken.wait(3)and attempted.wait(3)
  # The owner has already taken the real queued object, but assembly owns RLock.
  assert not done.wait(.05),'Owner observed incomplete admission metadata'
 finally:
  release.set();a.join(3)
  if o.ident is not None:o.join(3)
 assert not a.is_alive()and not o.is_alive()and not errors and done.is_set()
 snap=out['snapshot'];assert out['ready']and snap['history_render_audit']['qualified']
 assert snap['diagnostic_user']=='CPU-only-label'and snap['raw_budget_contract']['selected']==4096
 assert snap['Native_prompt_token_ids_SHA256']==hashlib.sha256(b'[10,20,30]').hexdigest()
 # Exact runtime assembly assertion fails AFTER enqueue: ERROR/fatal, no owner entry.
 scope['hashlib']=hashlib;engine=make(bad=True)
 try:add(engine,'CPU-assembly-failure')
 except AssertionError:pass
 else:raise AssertionError('Expected actual post-enqueue metadata assertion')
 req=engine.take(timeout=1);native_entry_count=0
 try:
  snapshot(engine,req);native_entry_count+=1
 except RuntimeError as exc:assert str(exc)=='Admission metadata assembly failed; request must not enter Native'
 else:raise AssertionError('Failed assembly reached owner Native-entry boundary')
 assert req.is_finished()and req.finish_reason==reasons.ERROR and engine._fatal and not req._admission_metadata_ready
 assert native_entry_count==0 and not engine.is_healthy()
 # Source boundary proof: blocking take and first Native reset are outside snapshot.
 assert not any(isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr in ('take','wait','modern_native_engine_reset','modern_native_engine_forward')for n in ast.walk(next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='admission_snapshot')))
 serve=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='serve')
 take=next(n.lineno for n in ast.walk(serve)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='take')
 snapshot_line=next(n.lineno for n in ast.walk(serve)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='admission_snapshot')
 reset=min(n.lineno for n in ast.walk(serve)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='modern_native_engine_reset')
 assert take<snapshot_line<reset
 return dict(passed=True,scope='Pure CPU exact runtime Bridge/snapshot AST and real stdlib queue; no actual HTTP, Native, GPU or Model',worker_SHA256=hashlib.sha256(source).hexdigest(),post_enqueue_pause_reached=True,owner_took_before_assembly_complete=True,owner_snapshot_waited_for_assembly=True,complete_metadata_snapshot=True,assembly_failure_ERROR_and_fatal=True,failed_assembly_fake_Native_entry_count=native_entry_count,lock_excludes_queue_wait_and_Native=True)

if __name__=='__main__':print(json.dumps(run(),indent=2))
