"""Actual FastAPI/Uvicorn HTTP routes with explicitly stubbed GPU-only imports.

Run in an environment with real FastAPI, Uvicorn and jsonschema (the cluster CPU
container has these). This is a transport contract, never real model P0 proof.
"""
import enum,hashlib,importlib.util,json,os,pathlib,queue,socket,sys,threading,time,types,urllib.error,urllib.request
D=pathlib.Path(__file__).resolve().parent
class Reasons(enum.Enum):
 EOS_TOKEN='eos_token';STOP_STRING='stop_string';STOP='stop';LENGTH='length';CANCELED='canceled';TIMEOUT='timeout';ERROR='error'
class SamplingParams:
 def __init__(self,**kwargs):self.__dict__.update(kwargs)
def gpu_forbidden(*args,**kwargs):raise AssertionError('GPU engine must never be constructed in CPU HTTP contract')
pkg=types.ModuleType('infinilm');pkg.__path__=[str(D/'overlay/python/infinilm')];sys.modules['infinilm']=pkg
serverpkg=types.ModuleType('infinilm.server');serverpkg.__path__=[str(D/'overlay/python/infinilm/server')];sys.modules['infinilm.server']=serverpkg
for name,values in [('infinilm.base_config',{'BaseConfig':object}),('infinilm.config',{'KVTransferConfig':object}),('infinilm.llm',{'AsyncLLMEngine':gpu_forbidden,'FinishReason':Reasons,'SamplingParams':SamplingParams}),('infinilm.moe_config',{'configure_moe_ep_backend':gpu_forbidden})]:
 module=types.ModuleType(name);module.__dict__.update(values);sys.modules[name]=module
s=importlib.util.spec_from_file_location('bridge_c',D/'native_engine_bridge_r143c.py');bridge=importlib.util.module_from_spec(s);s.loader.exec_module(bridge)
from infinilm.server import inference_server as real_server
real_server.DEFAULT_STREAM_TIMEOUT=.03;real_server.DEFAULT_REQUEST_TIMEOUT=.06
class Tokenizer:
 def apply_chat_template(self,messages,**kwargs):return messages[-1]['content']
 def encode(self,text,**kwargs):return [1,2,3]
 def decode(self,ids,**kwargs):return ''.join(chr(x)for x in ids)
engine=bridge.NativeEngineBridge(Tokenizer(),256,Reasons,eos_token_ids=[999],default_max_tokens=96)
api,app=bridge.existing_server_app(engine,'CPU-FAKE-WORKER-NOT-MODEL','glm5')
active=[];allow_reset=threading.Event();worker_errors=[]
def owner():
 try:
  while True:
   req=engine.take()
   if req is None:break
   active.append(req)
   if req.prompt.startswith('timeout'):
    for ch in '<tool_call>calculate<arg_key>a<arg_value>123456789':
     if req.cancel_event.is_set():break
     req.emit_token(ord(ch));time.sleep(.002)
    while not req.cancel_event.is_set():req.emit_token(ord('x'));time.sleep(.002)
    assert allow_reset.wait(5),'Test reset release timed out'
    req.metadata['original_reset_completed']=True;req.finish(Reasons.EOS_TOKEN)
   else:
    for ch in 'OK':req.emit_token(ord(ch))
    req.emit_token(999);req.metadata['original_reset_completed']=True;req.finish(Reasons.EOS_TOKEN)
 except BaseException as error:worker_errors.append(repr(error))
import uvicorn
sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
server=uvicorn.Server(uvicorn.Config(app,log_level='error'));worker=threading.Thread(target=owner,daemon=True);worker.start();http=threading.Thread(target=lambda:server.run(sockets=[sock]),daemon=True);http.start()
start=time.monotonic()
while not server.started:
 assert http.is_alive()and time.monotonic()-start<10
 time.sleep(.01)
def request(payload):
 req=urllib.request.Request('http://127.0.0.1:%d/v1/chat/completions'%port,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
 try:
  with urllib.request.urlopen(req,timeout=5)as response:return response.status,response.read().decode()
 except urllib.error.HTTPError as error:return error.code,error.read().decode()
checks=[];responses=[]
try:
 p={'model':'CPU-FAKE-WORKER','messages':[{'role':'user','content':'timeout-http'}],'max_tokens':96,'temperature':0}
 status,text=request(p);responses.append({'case':'nonstream-timeout','status':status,'body':text});assert status==504 and json.loads(text)['error']['type']=='timeout_error'and'choices'not in json.loads(text)
 req=active[-1];assert not req.is_finished()and req.finish_reason is None and req.abort_reason is Reasons.TIMEOUT;checks.append('actual_nonstream_HTTP504_no_empty_stop_native_still_unfinished')
 allow_reset.set()
 for _ in range(100):
  if req.is_finished():break
  time.sleep(.005)
 assert req.finish_reason is Reasons.TIMEOUT and req.metadata['original_reset_completed'];checks.append('TIMEOUT_survives_original_route_finally_and_owner_reset')
 status,text=request({'messages':[{'role':'user','content':'recover1'}],'max_tokens':8,'temperature':0});assert status==200 and json.loads(text)['choices'][0]['message']['content']=='OK';checks.append('real_HTTP_request_after_timeout_recovers')
 allow_reset.clear();p['stream']=True;p['messages'][0]['content']='timeout-sse';p['tools']=[{'type':'function','function':{'name':'calculate','parameters':{'type':'object','properties':{'a':{'type':'integer'}},'required':['a']}}}];p['tool_choice']='required'
 status,text=request(p);responses.append({'case':'stream-timeout','status':status,'body':text});assert status==200
 chunks=[json.loads(line[6:])for line in text.splitlines()if line.startswith('data: ')and line!='data: [DONE]'];assert any(c.get('error',{}).get('type')=='timeout_error'for c in chunks);assert any(x.get('finish_reason')=='timeout'for c in chunks for x in c.get('choices',[]));assert not any(x.get('finish_reason')in('stop','tool_calls')or x.get('delta',{}).get('tool_calls')for c in chunks for x in c.get('choices',[]));checks.append('actual_SSE_timeout_error_and_timeout_finish_never_executable_partial_tool')
 req=active[-1];assert req.abort_reason is Reasons.TIMEOUT and not req.is_finished();allow_reset.set()
 for _ in range(100):
  if req.is_finished():break
  time.sleep(.005)
 assert req.finish_reason is Reasons.TIMEOUT
 status,text=request({'messages':[{'role':'user','content':'recover2'}],'max_tokens':8,'temperature':0});assert status==200 and json.loads(text)['choices'][0]['message']['content']=='OK';checks.append('real_HTTP_recovers_after_SSE_timeout')
 assert not worker_errors
 out={'passed':True,'actual_HTTP_routes':True,'actual_FastAPI_Uvicorn':True,'GPU_imports_stubbed':True,'CPU_fake_owner_only':True,'Native':False,'model_weights':False,'checks':checks,'responses':responses,'patched_server_SHA256':hashlib.sha256((D/'overlay/python/infinilm/server/inference_server.py').read_bytes()).hexdigest()};target=pathlib.Path(os.environ.get('BRIDGE_CPU_OUTPUT_R145',str(D)));target.mkdir(parents=True,exist_ok=True);(target/'CPU-HTTP-timeout-contract-r145.json').write_bytes((json.dumps(out,indent=2)+'\n').encode());print(json.dumps(out))
finally:
 allow_reset.set();server.should_exit=True;http.join(10);engine.stop();worker.join(5);assert not http.is_alive()and not worker.is_alive()
