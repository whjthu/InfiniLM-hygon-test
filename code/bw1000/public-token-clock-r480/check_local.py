"""Only new clock glue: real frozen route/request/P1 methods, Native/encode leaves recorded."""
from pathlib import Path
import argparse,ast,asyncio,enum,hashlib,importlib.util,json,logging,sys,threading,time,tempfile
from types import SimpleNamespace
from token_clock import AdmissionMixin,ClockServerMixin,PREFIX,Session
from client_clock import ClientClock
import build_overlay
G=Path(__file__).resolve().parents[3];P=G/'code/bw1000/Deep-fixed-fullpre-experimental-profile-r454/source-v1';U=G/'code/bw1000/public-GLM-native-provider-r425/public-python/infinilm/server/inference_server.py'
class Reason(enum.Enum):EOS_TOKEN='eos';LENGTH='length';CANCELED='canceled';ERROR='error';TIMEOUT='timeout'
def load(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
bridge=load(P/'public_deep_model/native_engine_bridge_r268.py','actual_bridge');text=load(P/'public_deep_model/incremental_text.py','actual_P1')
tree=ast.parse(U.read_bytes());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='InferenceServer');method=next(n for n in cls.body if isinstance(n,ast.AsyncFunctionDef)and n.name=='_stream_chat');helpers=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in ['sse_json','chunk_json','error_json']]
ns=dict(asyncio=asyncio,json=json,time=time,FinishReason=Reason,Request=object,AgentProtocolError=ValueError,logger=logging.getLogger('recording'))
exec(compile(ast.fix_missing_locations(ast.Module(body=helpers+[ast.ClassDef(name='ActualRoute',bases=[],keywords=[],body=[method],decorator_list=[])],type_ignores=[])),'actual_public_route','exec'),ns)
class Engine(AdmissionMixin,text.IncrementalTextMixin,bridge.NativeEngineBridge):
 def __init__(self):
  self._admission_lock=threading.RLock();tokenizer=SimpleNamespace(all_special_ids=[99],decode=lambda ids,**kw:'好'if len(ids)>=2 else'\ufffd')
  super().__init__(tokenizer,64,Reason,encode_chat=lambda *a:('recording prompt',[7]),eos_token_ids=[0],decode_tokens=tokenizer.decode);self._healthy=True
class Server(ClockServerMixin,ns['ActualRoute']):
 def __init__(self,enabled=True,timeout=1):
  self.engine=Engine();self.model_id='recording';self.tool_call_parser='none';self.stream_timeout_s=timeout;self.request_timeout_s=timeout;self.token_clock_enabled=enabled;self.logs=[];self.token_clock_sink=self.logs.append
 def _build_sampling_params(self,data):return SimpleNamespace(temperature=0.,max_tokens=4,stop=None)
 def _agent_parser(self,*a):return None
 def _convert_finish_reason(self,r):return 'stop'if r==Reason.EOS_TOKEN else r.value
class HTTP:
 def __init__(self,on=True,disconnect=False):self.headers={'x-infinilm-token-clock':'1'}if on else{};self.disconnect=disconnect
 async def is_disconnected(self):return self.disconnect
async def case(header=True,timeout=False,disconnect=False):
 server=Server(timeout=.015 if timeout else 1);data=dict(messages=[{'role':'user','content':'recording'}],stream=True,stream_options={'include_usage':True})
 async def owner():
  while server.engine._pending.empty():await asyncio.sleep(0)
  req=server.engine.take()
  if timeout:return req
  for token in [99,5,0]:
   clock=getattr(req,'_token_clock',None)
   if clock:clock.returned();clock.sampled()
   req.emit_token(token)
   await asyncio.sleep(0)
  req.metadata['recording_reset_done']=True;req.finish(Reason.EOS_TOKEN);return req
 task=asyncio.create_task(owner());chunks=[v async for v in server._stream_chat('recording-id',data,HTTP(header,disconnect))];req=await task
 wire=''.join(chunks);client=ClientClock(1)
 # All fragments carry the same received timestamp: no fabricated pertoken ITL.
 b=wire.encode();[client.feed(b[i:i+1],100)for i in range(len(b))];client.feed(b'',100,True);summary=client.summary()
 if header:
  assert summary['chunk_is_token']is False
  if not(timeout or disconnect):assert summary['complete']
  if timeout:assert not summary['complete']
  if not(timeout or disconnect):assert len(client.tokens)==3 and summary['EOS_tokens']==1 and summary['special_tokens']==1 and summary['decode_tok_s']is None
  assert wire.rfind(PREFIX)<wire.rfind('data: [DONE]')
 else:assert not client.tokens and not summary['complete']and PREFIX not in wire
 if timeout:
  assert req.cancel_event.is_set()and not req.is_finished()and summary['owned_finished']is False
  req.metadata['recording_reset_done']=True;req.finish(Reason.TIMEOUT)
 if disconnect:assert req.cancel_event.is_set()or req.is_finished()
 return dict(header=header,timeout=timeout,disconnect=disconnect,summary=summary,data_frames=[f['data']for f in client.frames])
async def main():
 rows=[]
 for kw in [{},{'header':False},{'timeout':True},{'disconnect':True}]:rows.append(await case(**kw))
 # Exact standard data frame stream is identical when comments are ignored (created times normalized).
 normalize=lambda frames:[{**json.loads(s),'created':0}if s!='[DONE]'else s for s in frames]
 assert normalize(rows[0]['data_frames'])==normalize(rows[1]['data_frames'])
 with tempfile.TemporaryDirectory()as tmp:
  m=build_overlay.build(P,Path(tmp)/'overlay',hashlib.sha256((P/'SOURCE_MANIFEST.json').read_bytes()).hexdigest())
  for p in(Path(tmp)/'overlay/public_deep_model').glob('*.py'):ast.parse(p.read_bytes())
  decoded=load(Path(tmp)/'overlay/public_deep_model/native_decode.py','actual_derived_decode')
  req=bridge.NativeRequest('recording-decode','recording',[7],4,[0],Reason);clock=Session(req.request_id);clock.attach(req);sample=iter([99,5,0])
  result=decoded.generate(req,lambda *a:(0,None,[next(sample)]),lambda rc:None,lambda logits:None,Reason,context=64,eos=0,absolute_deadline=time.time()+1)
  assert result[0]==Reason.EOS_TOKEN and [r['kind']for r in clock.forward_returns]==['prefill','decode','decode']and [r['native_forward_ordinal']for r in clock.rows]==[0,1,2]
  assert all(r['native_return_ns']<=r['sampled_ns']<=r['native_emit_ns']for r in clock.rows)
 return dict(passed=True,scope='New clock actual public _stream_chat + actual NativeRequest/P1/queue; encoding/Native owner actions explicitly recording leaves',Native=False,HTTP_external=False,GPU=False,old_sixsuite_repeated=False,cases=rows,parent_manifest_SHA256=hashlib.sha256((P/'SOURCE_MANIFEST.json').read_bytes()).hexdigest(),route_SHA256=hashlib.sha256(U.read_bytes()).hexdigest())
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--output',required=True);args=a.parse_args();q=asyncio.run(main());Path(args.output).open('x',encoding='utf8').write(json.dumps(q,indent=2)+'\n');print(json.dumps({'passed':q['passed'],'cases':len(q['cases']),'GPU':False,'HTTP_external':False}))
