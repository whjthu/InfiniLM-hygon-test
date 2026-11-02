"""Actual335Publisher/client/364markers/388server/333Session; Model leaves recorded."""
from pathlib import Path
import copy,hashlib,importlib.util,inspect,json,shutil,socket,sys,tempfile,threading,time
from types import SimpleNamespace
from unittest import mock

D=Path(__file__).resolve().parent;B=D.parents[1]
sys.path[:0]=[str(D),str(B/'Kimi-long-target-publisher-r335'),str(B/'Kimi-long-target-bootstrap-r364'),str(D.parent),str(B/'Kimi-long-draft-RPC-CPU-r333')]
from context_policy_client_r388 import OFF,ON,WORKER_MF,bundle,client_type,load
from context_policy_target_r388 import settings_for_policy,connect_policy,bind_dependencies
from publisher import DraftClient,Publisher
from batch_wire_r9128 import send,recv

class SocketBound:
 def __init__(self,s):self.s=s
 def settimeout(self,t):self.s.settimeout(min(t,3))
 def __getattr__(self,n):return getattr(self.s,n)
class ListenerBound(SocketBound):
 def accept(self):s,a=self.s.accept();return SocketBound(s),a

def stage_packages(root):
 source=D.parent;manifest=json.loads((source/'SOURCE_MANIFEST.json').read_bytes())
 paths=[('/refactor/incoming/Kimi-draft-context-policy-r388/'+n)for n in manifest['files_SHA256']]+['/refactor/incoming/Kimi-draft-context-policy-r388/SOURCE_MANIFEST.json']+list(manifest['dependencies_SHA256'])
 for raw in paths:
  rel=Path(raw).relative_to('/refactor/incoming');src=B/rel;dst=root/'incoming'/rel
  dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)

def main():
 # Import real source modules. No Torch import occurs at their module scope.
 bootstrap=load(B/'Kimi-long-target-bootstrap-r364/bootstrap.py','real364_bootstrap_for_policyCPU')
 import lifecycle
 parent_checks=load(D.parent/'check_CPU.py','worker388_recording_leaves')
 from entry import serve_authenticated
 from runtime_factory import RuntimeFactory
 pubchecks=load(B/'Kimi-long-target-publisher-r335/check_publisher_CPU.py','publisher335_recording_native')
 checks=[];nonce='C'*64
 with tempfile.TemporaryDirectory(prefix='policy388-client-')as tmp:
  root=Path(tmp);(root/'control').mkdir();(root/'results').mkdir();stage_packages(root);bound=bundle(root)
  assert client_type(DraftClient,None,OFF)is DraftClient
  assert bind_dependencies(bootstrap,root,OFF)is bootstrap.RealDependencies
  ONDeps=bind_dependencies(bootstrap,root,ON);assert ONDeps.__init__ is bootstrap.RealDependencies.__init__
  assert all(getattr(ONDeps,n)is v for n,v in vars(bootstrap.RealDependencies).items()if inspect.isfunction(v)and n!='run_pair')
  checks.append(dict(case='OFF_exact_classes_ON_one_call_private_binding',passed=True,proof=ONDeps.context_policy_binding))
  class QueryContext(parent_checks.RecordingContext):
   def query(self,epoch,request,head,anchor,K,return_debug=False):
    self._check(epoch,request);assert self.phase=='decode';return dict(mode='draft',values=(list(range(10,10+K)),{'hidden':'CPUrecording'}))
  class Codec(parent_checks.RecordingCodec):
   raw_dim=35840
   def proposals(self,values,debug,K):assert len(values)==K;return values
  old=SimpleNamespace(**{k:getattr(bound.old,k)for k in ['PROTOCOL','RUNTIME_SHA','TARGET_NATIVE_SHA','expected_hello','serve_authenticated']},Runtime=parent_checks.RecordingRuntime,events=parent_checks.events,runtimes=parent_checks.runtimes,DraftSession=parent_checks.DraftSession,RecordingCodec=Codec)
  class Control(pubchecks.Control):
   def collect(self,label,value):
    if label=='r346-delayed-draft-connected':return [dict(value,node=n)for n in range(4)]
    return super().collect(label,value)
  for ordinal,mode in enumerate([OFF,ON]):
   run='client-policy-%02d'%ordinal;start=time.time()-28797
   env=dict(KIMI_LONG_PAIR_R346='1',INFINILM_KIMI_LONG_FEATURES_R305='1',KIMI_LONG_RUN_R347=run,KIMI_LONG_SESSION_R347=nonce,GLOBAL_RUN_R2354=run,KIMI_LONG_GPU_START_UNIX_R347=str(start),KIMI_LONG_NATIVE_DEADLINE_UNIX_R347=str(start+28800),KIMI_LONG_CANCEL_FILE_R347=str(root/'control'/('cancel-'+run)),SLURM_PROCID='0',KIMI_ENGINE_HEAD='127.0.0.1',GLOBAL_PORT_R2354='32947',KIMI_LONG_CAPTURE_PORT_R347='45947',KIMI_LONG_DRAFT_PORT_R347='46947',KIMI_LONG_DRAFT_HOST_R347='127.0.0.1')
   settings=lifecycle.Settings(env,root);gate=lifecycle.Gate(settings);chosen=settings_for_policy(settings,bound,mode)
   assert (chosen is settings)==(mode==OFF)and settings.identity['worker_manifest_SHA256']==bound.policy.OLD_MANIFEST_SHA256
   report={};errors=[];keeper_errors=[];connections=[];parent_checks.events.clear();parent_checks.runtimes.clear()
   listener=socket.socket();listener.bind(('127.0.0.1',0));listener.listen(1);address=listener.getsockname()
   def server():
    factory=RuntimeFactory(old,root,None,None,report,mode)
    hello=bound.policy.expected_hello(bound.old,nonce,mode,WORKER_MF)
    try:
     with mock.patch.dict(sys.modules,{'torch':SimpleNamespace(inference_mode=parent_checks.Scope),'long_context_r323':SimpleNamespace(LongCommittedFeatureContext=QueryContext)}),mock.patch.object(factory,'load_context',return_value=QueryContext),mock.patch.object(parent_checks,'RecordingCodec',Codec):
      serve_authenticated(old,ListenerBound(listener),hello,factory,report,lambda:None,mode)
    except BaseException as e:errors.append(repr(e))
   def keeper():
    try:
     end=time.monotonic()+3
     while not chosen.start_file.exists():
      assert time.monotonic()<end;time.sleep(.005)
     packet=json.loads(chosen.start_file.read_bytes());assert packet==chosen.identity
     lifecycle.publish_once(chosen.listening_file,dict(packet,actual_draft_step='910371.388',listener_ready=True))
    except BaseException as e:keeper_errors.append(repr(e))
   threads=[threading.Thread(target=server),threading.Thread(target=keeper)]
   for t in threads:t.start()
   def connect(address_requested,timeout):
    assert address_requested==('127.0.0.1',46947);connections.append(address_requested)
    return SocketBound(socket.create_connection(address,timeout=min(3,timeout)))
   saves=[];control=Control()
   connector=lifecycle.connect_draft if mode==OFF else ONDeps.run_pair.__globals__['connect_draft_policy_r388']
   rpc=connector(settings,gate,control,DraftClient,saves.append,connect=connect)
   assert type(rpc)is DraftClient if mode==OFF else rpc.command.__func__ is DraftClient.command
   events=[];native=pubchecks.RecordingNative(events);native.control=control
   publisher=Publisher(native,rpc,enabled=True);emitted=[]
   anchor=publisher.prefill([1,2],1,lambda x:emitted.extend(x))
   # Partial match needs replay and publishes only2acceptedinputs, not8captured.
   native.top=[10,77,20,21,22,23,24,25];decision=publisher.step(anchor,16,{99},lambda x:emitted.extend(x))
   assert decision.needs_replay and not decision.terminal and publisher.past==4
   native.top=[99]*8;decision=publisher.step(decision.pending_anchor,14,{99},lambda x:emitted.extend(x));assert decision.terminal
   publisher.close()
   for t in threads:t.join(5)
   assert not any(t.is_alive()for t in threads)and not errors and not keeper_errors and len(connections)==1
   assert report['normal_protocol_close']and report['context_policy_actual']==mode
   runtime=parent_checks.runtimes[0];assert runtime.closed and runtime.session.closed and not runtime.session.poisoned and runtime.session.context.length==4
   if mode==ON:
    wire=saves[-1]['draft_wire_source_binding'];assert wire['worker_manifest_SHA256']==WORKER_MF and wire['protocol']==bound.policy.PROTOCOL_ON and wire['context_source_SHA256']==bound.policy.CONTEXT_SHA256
   checks.append(dict(case=mode+':real364marker_binary335Publisher_333Session_388worker',passed=True,actual_private_run_pair_connector_called=(mode==ON),sole_connects=1,partial_replay_accepted_inputs=2,captured_verify_width=8,final_context_length=4,normal_close=True,marker_worker_MF=chosen.identity['worker_manifest_SHA256'],recording_Native_Torch=True))
  for scenario in ['wrong_nonce','legacy_client_to_ON_worker']:
   report={};errors=[];factories=[];parent_checks.events.clear()
   listener=socket.socket();listener.bind(('127.0.0.1',0));listener.listen(1);address=listener.getsockname()
   def reject_server():
    factory=RuntimeFactory(old,root,None,None,report,ON);factories.append(factory)
    try:serve_authenticated(old,ListenerBound(listener),bound.policy.expected_hello(bound.old,nonce,ON,WORKER_MF),factory,report,lambda:None,ON)
    except BaseException as e:errors.append(type(e).__name__)
   t=threading.Thread(target=reject_server);t.start();connection=SocketBound(socket.create_connection(address,timeout=3))
   cls=client_type(DraftClient,bound,ON)if scenario=='wrong_nonce'else DraftClient
   try:cls(connection,'X'*64 if scenario=='wrong_nonce'else nonce)
   except (ValueError,EOFError,ConnectionResetError):pass
   else:raise AssertionError('ON accepted bad nonce/legacyclient')
   t.join(3);assert not t.is_alive()and errors and not factories[0].called and not report.get('authenticated')and not parent_checks.events
   checks.append(dict(case=scenario,passed=True,server_factory_calls=0,Model_loads=0,no_fallback=True))
  bad_run='wrong-marker-01';start=time.time()-28797
  env.update(KIMI_LONG_RUN_R347=bad_run,GLOBAL_RUN_R2354=bad_run,KIMI_LONG_GPU_START_UNIX_R347=str(start),KIMI_LONG_NATIVE_DEADLINE_UNIX_R347=str(start+28800),KIMI_LONG_CANCEL_FILE_R347=str(root/'control'/('cancel-'+bad_run)))
  settings=lifecycle.Settings(env,root);gate=lifecycle.Gate(settings);chosen=settings_for_policy(settings,bound,ON)
  lifecycle.publish_once(chosen.listening_file,dict(chosen.identity,worker_manifest_SHA256=bound.policy.OLD_MANIFEST_SHA256,actual_draft_step='910371.388',listener_ready=True))
  connections=[]
  try:connect_policy(lifecycle.connect_draft,settings,gate,Control(),DraftClient,lambda v:None,bound=bound,mode=ON,connect=lambda *a,**kw:connections.append(a))
  except RuntimeError:pass
  else:raise AssertionError('Legacymarker accepted forON')
  assert not connections;checks.append(dict(case='legacy_worker_marker_rejected_before_connect',passed=True,connections=0))
  # A lying/legacyready must not trigger a secondhello or fallback.
  for field,bad in [('protocol',bound.old.PROTOCOL),('context_policy','off'),('context_source_SHA256','0'*64),('worker_entry_manifest_SHA256','0'*64)]:
   a,b=socket.socketpair();a.settimeout(3);b.settimeout(3);frames=[]
   def liar():
    try:
     hello,body=recv(b);frames.append(hello);assert not body
     expected=bound.policy.ready(bound.old,ON,hello);expected[field]=bad;send(b,expected)
    finally:b.close()
   t=threading.Thread(target=liar);t.start()
   try:client_type(DraftClient,bound,ON)(a,nonce)
   except ValueError:pass
   else:raise AssertionError('Mismatched ready accepted')
   t.join(3);assert not t.is_alive()and len(frames)==1 and a.fileno()==-1
   checks.append(dict(case='reject_ready_'+field,passed=True,hello_frames=1,no_fallback=True))
 print(json.dumps(dict(completed=True,passed=True,GPU=0,Slurm_jobs=0,Native_dlopen=False,Model_loads=0,new_Torch_math=False,checks=checks,check_count=len(checks),scope='Actual364 delayed marker/connect,335Publisher/clientcommands,388authenticatedserver/factory,333Session/binarywire. Native/capture/Model/query tensors/inference leaves recorded. CPU772/GPU713 numericqualification reused, no new device or Modelproof.')))
if __name__=='__main__':main()
