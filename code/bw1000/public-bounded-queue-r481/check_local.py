"""Only new admission/actual worker glue; no Native or external HTTP."""
from pathlib import Path
import argparse,ast,asyncio,contextlib,copy,enum,importlib.util,json,queue,sys,threading,time,tempfile,hashlib
from types import SimpleNamespace
import numpy as np
from queue_policy import QueuePolicyMixin,QueueCapacity
from build_overlay import build
G=Path(__file__).resolve().parents[3];P=G/'code/bw1000/Deep-fixed-fullpre-experimental-profile-r454/source-v1';C=G/'code/bw1000/public-token-clock-r480'
def load(p,name):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
bridge=load(P/'public_deep_model/native_engine_bridge_r268.py','real_bridge');p1=load(P/'public_deep_model/incremental_text.py','real_P1');decode=load(P/'public_deep_model/native_decode.py','real_decode')
class R(enum.Enum):EOS_TOKEN='eos';LENGTH='length';CANCELED='canceled';ERROR='error';TIMEOUT='timeout'
class E(QueuePolicyMixin,p1.IncrementalTextMixin,bridge.NativeEngineBridge):
 def __init__(self,enabled=True):
  self._admission_lock=threading.RLock();super().__init__(SimpleNamespace(decode=lambda *a,**k:''),32768,R,encode_chat=lambda *a:('recording',[7]),eos_token_ids=[1]);self.configure_queue_policy(dict(bounded_queue_enabled=enabled,physical_pending_capacity=2,queue_wait_seconds=5))
 def add(self,i):
  req=self.add_chat_request(messages=[{'role':'user','content':'recording'}],sampling_params=SimpleNamespace(temperature=0.,max_tokens=4096,stop=None),request_id=i);req._admission_metadata_ready=True;return req
class Context:
 def __init__(self):self.ctx=(1<<48)+123;self.binding=SimpleNamespace(check=lambda rc:None);self.cfg={'vocab_size':129280};self.floors=0;self.resets=0;self.forward=0
 def floor(self):self.floors+=1;return [16*1024**3,32*1024**3]
 def reset(self):self.resets+=1
class Policy:
 poisoned=False;normal_reset_after=False
 def __init__(self,ctx,cancel=None):self.ctx=ctx;self.cancel=cancel
 @contextlib.contextmanager
 def request(self,req,record):
  self.ctx.reset();record['normal_reset_before']=True
  try:yield
  finally:self.ctx.reset();self.normal_reset_after=True;record['normal_reset_after']=True
 def forward(self,*args):
  self.ctx.forward+=1
  if self.cancel:self.cancel.mark_abort(R.CANCELED)
  return 0,np.zeros((1,2),dtype=np.uint16),[1]
async def main():
 e=E();a,b=e.add('cancel'),e.add('expire')
 stream=e.stream_request(a,timeout=1,request_timeout=1);pending=asyncio.create_task(anext(stream));await asyncio.sleep(0);pending.cancel()
 try:await pending
 except asyncio.CancelledError:pass
 await stream.aclose();assert a.cancel_event.is_set()and not a.is_finished()and'cancel'not in e._requests and e._pending.qsize()==2
 try:e.add('new')
 except QueueCapacity:pass
 else:raise AssertionError('Registry retirement bypassed physical bound')
 b.metadata['queue_expires_at']=-1 # Explicit recording clock leaf, no real wait/timer.
 e.stop();assert e._pending.qsize()==3 # Two requests plus single STOP sentinel; nonblocking.
 with tempfile.TemporaryDirectory()as tmp:
  O=Path(tmp)/'entry';build(P,C,O,hashlib.sha256((P/'SOURCE_MANIFEST.json').read_bytes()).hexdigest(),hashlib.sha256((C/'SOURCE_MANIFEST.json').read_bytes()).hexdigest())
  for p in(O/'public_deep_model').glob('*.py'):ast.parse(p.read_bytes())
  tree=ast.parse((O/'public_deep_model/worker.py').read_bytes());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='serve');ns=dict(copy=copy,queue=queue,time=time,np=np,sys=sys,generate=decode.generate,RequestPolicy=Policy)
  exec(compile(ast.Module(body=[node],type_ignores=[]),'actual_derived_worker_serve','exec'),ns);serve=ns['serve']
  prepared=lambda e:dict(engine=e,FinishReason=R,identity=dict(family='Deep',upstream_commit='99f5c90a40f9b2546bd0202cba3397ed1d55c5a0',eos_token_ids=[1]))
  config={'gpu_start_epoch':time.time(),'service_seconds':60};run=SimpleNamespace(stop_requested=lambda:False,deadline_check=lambda *a:None);ctx=Context();report={'requests':[]}
  serve(ctx,prepared(e),config,run,threading.Event(),report,lambda:None,policy=Policy(ctx));assert ctx.forward==ctx.floors==0 and ctx.resets==1 and a.is_finished()and b.is_finished()and not report['requests']
  outcomes=[r['outcome']for r in e.queue_ledger];assert 'queued_cancel'in outcomes and'queued_expired'in outcomes and'capacity_rejected'in outcomes
  stopped=E();orphan=stopped.add('STOP-dequeued');pending2=stopped.add('STOP-queued');stopped._requests.pop(orphan.request_id) # Prior actualP1 pop behavior already consumed above.
  event=threading.Event();original_take=stopped.take
  def take_and_stop(**kw):
   req=original_take(**kw);event.set();return req
  stopped.take=take_and_stop;cx=Context();rs={'requests':[]};serve(cx,prepared(stopped),config,run,event,rs,lambda:None,policy=Policy(cx));assert orphan.is_finished()and pending2.is_finished()and cx.forward==cx.floors==0 and cx.resets==1 and stopped._pending.empty()
  race=E();rr=race.add('STOP-lock-race');taken=threading.Event();intent=threading.Event();take0=race.take
  def take_race(**kw):
   req=take0(**kw);taken.set();return req
  race.take=take_race;rc=Context();fail=[]
  def real_worker_thread():
   try:serve(rc,prepared(race),config,run,intent,{'requests':[]},lambda:None,policy=Policy(rc))
   except BaseException as error:fail.append(repr(error))
  with race._admission_lock:
   owner=threading.Thread(target=real_worker_thread);owner.start();assert taken.wait(1);intent.set();race.stop()
  owner.join(2);assert not owner.is_alive()and not fail and rr.is_finished()and rc.forward==rc.floors==0 and rc.resets==1
  for cancel in [False,True]:
   e2=E();req=e2.add('active');e2.stop();ctx2=Context();r2={'requests':[]};serve(ctx2,prepared(e2),config,run,threading.Event(),r2,lambda:None,policy=Policy(ctx2,req if cancel else None));assert ctx2.forward==1 and ctx2.resets>=2 and req.is_finished()and r2['requests'][0]['normal_reset_after']
   assert req.finish_reason==(R.CANCELED if cancel else R.EOS_TOKEN)
 return dict(passed=True,scope='Real NativeRequest/P1.finally/Queue/STOP + actual derived worker.serve + original generate; Context/Policy/Native outputs and expiry clock are recording leaves',physical_bound_after_registry_pop=True,queued_cancel_expire_zero_Native=True,active_cancel_original_reset=True,STOP_nonblocking=True,dequeued_STOP_and_unregistered_queue_owned_finished_after_final_reset=True,high64_Context_pointer_recording=True,Native=False,HTTP_external=False,GPU=False)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();q=asyncio.run(main());Path(a.output).open('x',encoding='utf8').write(json.dumps(q,indent=2)+'\n');print(json.dumps(q))
