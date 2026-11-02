"""Real wire/new factory/original333 Session with recording Torch/load leaves."""
from pathlib import Path
import ast,copy,hashlib,importlib.util,inspect,json,socket,sys,tempfile,textwrap,threading
from types import SimpleNamespace
from unittest import mock

D=Path(__file__).resolve().parent;BASE=D.parent
sys.path[:0]=[str(D),str(BASE/'Kimi-long-draft-RPC-CPU-r333')]
from policy import OFF,ON,CONTEXT_SHA256,expected_hello,ready,checked
from runtime_factory import RuntimeFactory,private_runtime_type
from entry import serve_authenticated
from long_rpc import DraftSession
from batch_wire_r9128 import send,recv

events=[]
runtimes=[]
class RecordingContext:
 def __init__(self,model,weight_tag,*,enabled,discard_raw=False):
  assert enabled;self.owner=threading.get_ident();self.epoch=0;self.request=None;self.phase='closed';self.length=0;self.raw=self.projected=None;self.discard_raw=discard_raw
 def _check(self,epoch,request):assert threading.get_ident()==self.owner and epoch==self.epoch and request==self.request and self.phase in ['prompt','decode']
 def begin(self,epoch,request,prompt_tokens):
  assert threading.get_ident()==self.owner and self.projected is None and epoch>self.epoch
  self.epoch,self.request,self.prompt_tokens,self.phase,self.length=epoch,request,prompt_tokens,'prompt',0
 def append_prompt(self,e,r,p,part,fence):self.append(e,r,p,part,fence,'prompt')
 def append_committed(self,e,r,p,part,fence):self.append(e,r,p,part,fence,'decode')
 def append(self,e,r,p,part,fence,phase):
  self._check(e,r);assert self.phase==phase and self.length==p;fence();self.length+=part.rows;self.projected=True;self.raw=None if self.discard_raw else True
 def seal_prompt(self,e,r):self._check(e,r);assert self.length==self.prompt_tokens;self.phase='decode'
 def retire(self,e,r):self._check(e,r);value=self.projected is not None;self.raw=self.projected=None;self.phase='retired';return value
 def close(self):assert threading.get_ident()==self.owner;value=self.projected is not None;self.raw=self.projected=None;self.phase='closed';return value
class RecordingCodec:
 raw_dim=40;vocab=163840
 def decode(self,body,rows):events.append('decode_recording');return SimpleNamespace(rows=rows)
 def fence(self):events.append('original_codec_fence_recording')
class RecordingRuntime:
 def __init__(self,root,runtime,incoming,report):
  from long_context_r323 import LongCommittedFeatureContext
  events.append('weight_load_recording')
  if report.get('inject_constructor_error'):raise RuntimeError('recording constructor failure')
  context=LongCommittedFeatureContext(None,'recording',enabled=True)
  self.session=DraftSession(context,None,RecordingCodec(),enabled=True);self.closed=False;self.report=report;runtimes.append(self)
 def close(self):
  events.append('Runtime_close_recording');self.session.close();self.closed=True
  if self.report.get('inject_close_error'):raise RuntimeError('recording close failure')
class Scope:
 def __enter__(self):events.append('inference_scope_enter_recording')
 def __exit__(self,*args):events.append('inference_scope_exit_recording')

def main():
 oldpath=BASE/'Kimi-long-draft-memory-floor-r362b/worker.py';s=importlib.util.spec_from_file_location('old_worker_for_cpu388',oldpath);old=importlib.util.module_from_spec(s);s.loader.exec_module(old)
 oldbytes=oldpath.read_bytes();checks=[];nonce='n'*64;mf='f'*64
 assert expected_hello(old,nonce)==old.expected_hello(nonce)
 assert ready(old,OFF,expected_hello(old,nonce))==dict(ready=True,protocol=old.PROTOCOL,C=1,context=32768,block_size=8,proposals=7,prompt_append_max=512,committed_append_max=8,runtime_manifest_SHA256=old.RUNTIME_SHA,target_Native_SHA256=old.TARGET_NATIVE_SHA)
 private,proof=private_runtime_type(old,lambda *a,**k:None);assert private.close is old.Runtime.close and proof['replacements']==1
 tree=ast.parse(textwrap.dedent(inspect.getsource(old.Runtime.__init__)));original=ast.dump(tree,include_attributes=False);call=next(n for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='LongCommittedFeatureContext');call.func.id='context_factory_r388';assert hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()==proof['private_constructor_AST_SHA256'];call.func.id='LongCommittedFeatureContext';assert ast.dump(tree,include_attributes=False)==original
 checks.append(dict(case='actual362b_constructor_one_call_only_AST_and_identical_close',passed=True,proof=proof))
 for bad in ['ON','legacy',None,True]:
  try:checked(bad)
  except ValueError:pass
  else:raise AssertionError('Unsupported policy accepted')
 checks.append(dict(case='unsupported_mode_rejected',passed=True))
 # Actual384 source import needs no Torch/weights; checksum mismatch stops first.
 with tempfile.TemporaryDirectory()as tmp:
  root=Path(tmp);path=root/'incoming/Kimi-raw-history-elision-r384/context.py';path.parent.mkdir(parents=True);path.write_bytes((BASE/'Kimi-raw-history-elision-r384/context.py').read_bytes())
  factory=RuntimeFactory(old,root,None,None,{},ON);cls=factory.load_context();assert cls.query.__name__=='query'and cls.discard_raw.fset is None
  path.write_text('# wrong source')
  try:factory()
  except AssertionError:pass
  else:raise AssertionError('Bad source loaded')
  assert factory.scope is None
 checks.append(dict(case='actual384_source_pin_import_and_bad_source_zero_Torch_load',passed=True))
 recording_old=SimpleNamespace(**{k:getattr(old,k)for k in ['PROTOCOL','RUNTIME_SHA','TARGET_NATIVE_SHA','expected_hello','serve_authenticated']},Runtime=RecordingRuntime,events=events,runtimes=runtimes,DraftSession=DraftSession,RecordingCodec=RecordingCodec)
 scenarios=['normal','wrong_nonce','wrong_context_SHA','wrong_worker_MF','wrong_mode_hello','hello_body','EOF_after_ready','bad_commit','constructor_error','close_error']
 for mode in [OFF,ON]:
  for scenario in scenarios:
   events.clear();runtimes.clear();report={};errors=[];created=[]
   if scenario=='constructor_error':report['inject_constructor_error']=True
   if scenario=='close_error':report['inject_close_error']=True
   expected=expected_hello(old,nonce,mode,mf);wirehello=copy.deepcopy(expected);body=b''
   if scenario=='wrong_nonce':wirehello['nonce']='x'*64
   elif scenario=='wrong_context_SHA':wirehello['context_source_SHA256']='0'*64
   elif scenario=='wrong_worker_MF':wirehello['worker_entry_manifest_SHA256']='0'*64
   elif scenario=='wrong_mode_hello':wirehello=expected_hello(old,nonce,ON if mode==OFF else OFF,mf)
   elif scenario=='hello_body':body=b'forbidden'
   listener=socket.socket();listener.bind(('127.0.0.1',0));listener.listen(1);address=listener.getsockname()
   def server():
    factory=RuntimeFactory(recording_old,Path('/recording'),None,None,report,mode);created.append(factory)
    try:
     with mock.patch.dict(sys.modules,{'torch':SimpleNamespace(inference_mode=Scope),'long_context_r323':SimpleNamespace(LongCommittedFeatureContext=RecordingContext)}),mock.patch.object(factory,'load_context',return_value=RecordingContext):
      serve_authenticated(recording_old,listener,expected,factory,report,lambda:None,mode)
    except BaseException as e:errors.append(repr(e))
   t=threading.Thread(target=server);t.start();conn=socket.create_connection(address,timeout=3);conn.settimeout(3)
   try:
    send(conn,wirehello,body)
    invalid=scenario in ['wrong_nonce','wrong_context_SHA','wrong_worker_MF','wrong_mode_hello','hello_body']
    if invalid or scenario=='constructor_error':
     try:recv(conn)
     except (EOFError,ConnectionResetError):pass
     else:raise AssertionError('Invalid request got ready')
    else:
     try:got,extra=recv(conn)
     except EOFError as e:raise AssertionError((mode,scenario,errors))from e
     assert not extra and got==ready(old,mode,expected)
     if scenario not in ['EOF_after_ready']:
      seq=0;epoch=1
      def command(cmd,data=b'',**kw):
       nonlocal seq
       meta=dict(cmd=cmd,seq=seq)
       if cmd!='close':meta.update(epoch=epoch,request=1)
       meta.update(kw);send(conn,meta,data);answer,empty=recv(conn);assert not empty;seq+=1;return answer
      command('begin',prompt_tokens=1);command('append_prompt',bytes(80),past=0,rows=1,capture_epoch=1);command('seal',past=1)
      if scenario=='bad_commit':
       send(conn,dict(cmd='append_committed',seq=seq,epoch=1,request=1,past=1,rows=1,capture_epoch=2,target_commit=dict(capture_epoch=2,past_before=1,past_after=2,accepted_inputs=0,complete=True)),bytes(80))
       try:recv(conn)
       except (EOFError,ConnectionResetError):pass
       else:raise AssertionError('Rejected commit accepted')
      else:
       assert command('retire')['removed'];epoch=2
       command('begin',prompt_tokens=1);command('append_prompt',bytes(80),past=0,rows=1,capture_epoch=2);command('seal',past=1)
       assert command('retire')['removed'];assert command('close')['closed']
   finally:conn.close();t.join(5)
   assert not t.is_alive()and created and created[0].scope is None
   if invalid:assert not created[0].called and not events and not report.get('authenticated')and not runtimes
   else:
    assert report['authenticated']and events.index('inference_scope_enter_recording')<events.index('weight_load_recording')
    assert events[-1]=='inference_scope_exit_recording'
    assert bool(errors)==(scenario in ['EOF_after_ready','bad_commit','constructor_error','close_error'])
    if scenario!='constructor_error':
     assert 'Runtime_close_recording'in events and report.get('context_policy_actual')==mode and runtimes[0].closed and runtimes[0].session.closed
     if scenario=='bad_commit':assert runtimes[0].session.poisoned and runtimes[0].session.context.length==1
    before=list(events)
    try:created[0]()
    except AssertionError:pass
    else:raise AssertionError('Factory reloaded a connection')
    assert events==before
   checks.append(dict(case=mode+':'+scenario,passed=True,zero_load_rejection=invalid,recording_events=list(events),errors=errors))
 assert oldpath.read_bytes()==oldbytes
 print(json.dumps(dict(completed=True,passed=True,GPU=0,Slurm_jobs=0,Native_dlopen=False,Model_loads=0,new_Torch_math_executed=False,checks=checks,check_count=len(checks),old362b_bytes_unchanged=True,scope='Actual wire/listener/new policy factory/private constructor transformation/original333Session. Torch/load/model/contexttensor leaves recording; realnumerics reuseCPU772/GPU713. Target335 andModelcontrollerON client not adapted.')))
if __name__=='__main__':main()
