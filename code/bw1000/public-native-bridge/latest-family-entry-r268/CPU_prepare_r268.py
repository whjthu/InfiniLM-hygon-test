"""Latest full public app and actual Kimi metadata/tokenizer, no Model execution."""
import pathlib,sys,types,hashlib,json,os,datetime,importlib.machinery,asyncio,socket,copy,traceback
D=pathlib.Path('/refactor/incoming/latest-family-entry-r268');U=pathlib.Path('/refactor/candidates/Kimi-latest-service-r262/python/infinilm');R=pathlib.Path('/refactor');O=R/'candidates/latest-family-entry-r268';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
O.mkdir(exist_ok=False)
q={'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'completed':False,'passed':False,'CPU_only':True,'GPU':False,'scope':'Actual latest GLM/Deep prepared processor/server/queue/history CPU integration; no Native ABI or Model qualification','actual_public_server':True,'actual_processor_constructor':True,'actual_SamplingParams_and_FinishReason':True,'Model_or_weights_loaded':False,'Native_dlopen':False,'TCP_listener_started':False,'positive_Model_requests':0,'cases':[]}
def no_network(*a,**k):raise RuntimeError('CPU full-entry check forbids network')
socket.create_connection=no_network;socket.socket.connect=no_network
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_MODULES_CACHE=str(O/'hf-modules'),HF_HOME=str(O/'hf-home'),TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
def audit(event,args):
 if event=='open'and isinstance(args[0],(str,bytes,os.PathLike))and os.fsdecode(args[0]).lower().endswith(('.safetensors','.bin','.pt','.pth')):raise RuntimeError('Weight/model-binary read forbidden')
sys.addaudithook(audit)
class ForbiddenNative(types.ModuleType):
 def __getattr__(self,name):raise RuntimeError('InfiniLM Native API forbidden: '+name)
root=types.ModuleType('infinilm');root.__path__=[str(U)];root.__package__='infinilm';sys.modules['infinilm']=root
core=ForbiddenNative('infinilm._core');core.__file__='FORBIDDEN_NATIVE';core.__path__=[];core.__spec__=importlib.machinery.ModuleSpec(core.__name__,None);root._core=core;sys.modules[core.__name__]=core
llm=types.ModuleType('infinilm.llm');llm.__path__=[str(U/'llm')];llm.__package__='infinilm.llm';sys.modules[llm.__name__]=llm
for suffix,cls in [('scheduler','SchedulerOutput'),('static_scheduler','StaticSchedulerOutput')]:
 mod=types.ModuleType('infinilm.llm.'+suffix);setattr(mod,cls,type(cls,(),{}));sys.modules[mod.__name__]=mod
class ForbiddenEngine:
 def __init__(self,*a,**k):raise RuntimeError('Async Model engine construction forbidden')
llm.AsyncLLMEngine=ForbiddenEngine
sys.path.insert(0,str(D))

try:
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--manifest-sha',required=True);args=ap.parse_args()
 mf=D/'SOURCE_MANIFEST.json';assert sha(mf)==args.manifest_sha
 manifest=json.loads(mf.read_bytes())
 for name,expected in manifest['files_SHA256'].items():assert sha(D/name)==expected,name
 import torch
 torch.set_num_threads(1)
 def guard(name):
  def unique(*a,**k):raise RuntimeError('GPU API forbidden: '+name)
  return unique
 for name in ['_lazy_init','init','set_device','current_device']:setattr(torch.cuda,name,guard(name))
 torch.cuda.is_available=lambda:False;torch.cuda.device_count=lambda:0
 import janus,transformers,tokenizers
 q['dependencies']={n:{'version':getattr(v,'__version__',None),'source_SHA256':sha(pathlib.Path(v.__file__))} for n,v in [('torch',torch),('transformers',transformers),('tokenizers',tokenizers),('janus',janus)]}
 from infinilm.llm.sampling_params import SamplingParams
 from infinilm.llm.request import FinishReason
 llm.SamplingParams=SamplingParams;llm.FinishReason=FinishReason
 from infinilm.server.agent_protocol import prepare_agent_request
 from native_engine_bridge_r268 import NativeEngineBridge
 from service_queue_bridge_r268 import ServiceQueueBridge
 from types import SimpleNamespace
 import threading,queue
 def race_probe(bridge_cls,stop):
  entered=threading.Event();release=threading.Event();started_drain=threading.Event();errors=[];accepted=[]
  def encode(*a):entered.set();assert release.wait(10);return 'CPU only',[7,8]
  engine=bridge_cls(None,32768,FinishReason,encode_chat=encode,default_max_tokens=4096)
  params=SimpleNamespace(temperature=0.,max_tokens=4096,stop=None)
  def add():
   try:accepted.append(engine.add_chat_request(messages=[{'role':'user','content':'CPU only'}],sampling_params=params,request_id='race',request_data={'max_tokens':4096}))
   except BaseException as e:errors.append(type(e).__name__)
  writer=threading.Thread(target=add);writer.start();assert entered.wait(10)
  if stop:
   engine.stop();release.set();writer.join(10);assert not writer.is_alive() and errors==['RuntimeError'] and not accepted and len(engine._requests)==0 and engine.take(0) is None
   try:engine.take(0)
   except queue.Empty:pass
   else:raise AssertionError('Request published after stop sentinel')
  else:
   drain_errors=[]
   def drain():
    started_drain.set()
    try:engine.cancel_pending_after_global_drain(FinishReason.ERROR)
    except BaseException as e:drain_errors.append(type(e).__name__)
   reader=threading.Thread(target=drain);reader.start();assert started_drain.wait(10);release.set();writer.join(10);reader.join(10)
   assert not writer.is_alive() and not reader.is_alive() and not errors and not drain_errors and len(accepted)==1 and accepted[0].is_finished()
   assert engine.journal_snapshot()['requests'][0]['finished']
  return {'passed':True,'synthetic_CPU_interleaving_only':True,'actual_Native_reset_or_GPU':False}
 q['CPU_concurrency']={'stop_during_encode':race_probe(NativeEngineBridge,True),'GLM_admission_drain_journal':race_probe(ServiceQueueBridge,False)}
 from prepared_family_entry_r268 import prepare_app,raw_budget
 H=[{'role':'system','content':'Preserve the complete original instruction.'},{'role':'user','content':'Calculate 1+2.'},{'role':'assistant','content':'','reasoning_content':'Retain original historical reasoning with 中文 and literal text.','tool_calls':[{'id':'opaque-CPU268','type':'function','function':{'name':'calculate','arguments':'{"a":1,"b":2}'}}]},{'role':'tool','tool_call_id':'opaque-CPU268','content':'{"sum":3}'},{'role':'user','content':'Continue with the complete prior history.'}]
 TOOLS=[{'type':'function','function':{'name':'calculate','parameters':{'type':'object','properties':{'a':{'type':'integer'},'b':{'type':'integer'}},'required':['a','b']}}}]
 q['families']={}
 for family,model,metadata,parser in [('GLM','/old/models/ZhipuAI-GLM-5.3',None,'glm5'),('Deep','/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8','/refactor/results/latest-three-processors-r260/Deep-model-metadata','deepseek_v4')]:
  q['stage']='construct_'+family;p=prepare_app(family,model,processor_path=metadata);engine,server,app=p['engine'],p['server'],p['app'];row={'identity':p['identity'],'cases':[]};q['families'][family]=row
  assert sorted(engine.eos_ids)==([154820,154827,154829] if family=='GLM' else [1])
  assert engine.validation_timeouts_s=={'no_actual_progress':3600.0 if family=='GLM' else 1800.0,'absolute_request_including_queue':7200.0}
  for thinking,budget,generation in [(False,4096,True),(True,8192,True),(True,8192,False)]:
   q['stage']='queue_'+family+'_'+str(thinking)+'_'+str(generation)
   raw={'messages':copy.deepcopy(H),'tools':copy.deepcopy(TOOLS),'tool_choice':{'type':'function','function':{'name':'calculate'}},'parallel_tool_calls':False,'max_tokens':budget,'temperature':0.,'chat_template_kwargs':{'enable_thinking':thinking},'add_generation_prompt':generation};before=copy.deepcopy(raw)
   data,schemas,choice=prepare_agent_request(raw,parser);data.update(_agent_schemas=schemas,_agent_choice=choice);data_before=copy.deepcopy(data);params=server._build_sampling_params(data);assert isinstance(params,SamplingParams)
   req=engine.add_chat_request(messages=data['messages'],sampling_params=params,request_id=family+'-cpu268-'+str(len(row['cases'])),request_data=data,add_generation_prompt=generation,chat_template_kwargs=data.get('chat_template_kwargs'))
   assert engine.take(0) is req and req._admission_metadata_ready and req.max_new_tokens==budget and data==data_before
   h=req.metadata['history_render_audit'];assert h['qualified'] and h['input_RC_fields']==1 and h['RC_order_preserved'] and h['generation_prefix_checked']==generation and h['generation_prefix_correct']==(True if generation else None)
   if family=='GLM':engine.record_started(req);engine.record_finished(req,0,True);assert engine.journal_snapshot()['requests'][-1]['original_reset_all32_completed']
   else:
    with engine._admission_lock:assert req._admission_metadata_ready and req.metadata['real_cache_capacity']==32768
   row['cases'].append({'thinking':thinking,'budget':budget,'generation':generation,'prompt_tokens':len(req.prompt_token_ids),'history_RC_contributions':h['input_RC_fields'],'prompt_SHA256':h['actual_prompt_SHA256'],'IDs_SHA256':h['Native_prompt_token_ids_SHA256'],'atomic_metadata_before_take':True,'passed':True})
  q['stage']='negative_'+family
  count=len(engine._requests);invalid=[{'max_tokens':v}for v in [True,False,4096.0,'4096',None,0,-1,8193]]+[{'sampling_params':[]},{'sampling_params':{'max_tokens':True}},{'max_tokens':4096,'max_completion_tokens':8192}]
  for bad in invalid:
   try:server._build_sampling_params(bad)
   except ValueError:pass
   else:raise AssertionError('Malformed wire budget admitted')
  assert len(engine._requests)==count
  # Probe uses fully prepared data so a legitimate default reaches validation.
  d,schemas,choice=prepare_agent_request({'messages':[{'role':'user','content':'CPU only'}],'temperature':0},parser);d.update(_agent_schemas=schemas,_agent_choice=choice);assert server._build_sampling_params(d).max_tokens==4096
  async def check_routes():
   import httpx
   async with app.router.lifespan_context(app):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://cpu-only') as c:
     a=await c.get('/health');b=await c.get('/v1/models');bad=await c.post('/v1/chat/completions',json={'messages':[{'role':'user','content':'CPU only'}],'max_tokens':'4096'})
     assert a.status_code==200 and b.status_code==200 and bad.status_code==400
     return {'health':a.status_code,'models':b.status_code,'malformed_budget':bad.status_code,'TCP_listener_started':False,'positive_Model_requests':0}
  row['routes']=asyncio.run(check_routes());row['malformed_budget_cases_rejected']=len(invalid);row['passed']=True
 q.update(passed=True,stage='complete',source_manifest_SHA256=sha(mf));assert not torch.cuda.is_initialized();q['torch_cuda_initialized']=False
except BaseException as exc:q.update(error_type=type(exc).__name__,error=str(exc)[:1600],failure_locations=[{'file':v.filename,'line':v.lineno,'function':v.name}for v in traceback.extract_tb(exc.__traceback__)])
finally:
 q['completed']=True;(O/'CPU-prepared.json').write_text(json.dumps(q,ensure_ascii=False,indent=2)+chr(10));print(json.dumps({k:q.get(k)for k in ['completed','passed','stage','actual_step','error_type','error']}))
assert q['passed']
