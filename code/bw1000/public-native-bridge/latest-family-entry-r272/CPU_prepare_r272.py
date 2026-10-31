"""Latest full public app and actual Kimi metadata/tokenizer, no Model execution."""
import pathlib,sys,types,hashlib,json,os,datetime,importlib.machinery,asyncio,socket,copy,traceback
D=pathlib.Path('/refactor/incoming/latest-family-entry-r272');U=pathlib.Path('/refactor/candidates/latest-family-service-r270/GLM/python/infinilm');R=pathlib.Path('/refactor');O=R/'candidates/latest-family-entry-r272';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
O.mkdir(exist_ok=False)
q={'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'completed':False,'passed':False,'CPU_only':True,'GPU':False,'scope':'New complete RC/content contribution and actual prefix contract; actual public processors, guarded synthetic root; no Native ABI or Model qualification','actual_public_server':True,'actual_processor_constructor':True,'actual_SamplingParams_and_FinishReason':True,'Model_or_weights_loaded':False,'Native_dlopen':False,'TCP_listener_started':False,'positive_Model_requests':0,'cases':[]}
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
 ap=argparse.ArgumentParser();ap.add_argument('--manifest-sha',required=True);args=ap.parse_args();mf=D/'SOURCE_MANIFEST.json';assert sha(mf)==args.manifest_sha
 manifest=json.loads(mf.read_bytes())
 for name,h in manifest['files_SHA256'].items():assert sha(D/name)==h,name
 assert sha(R/'candidates/latest-family-service-r270/CPU-static-r273.json')=='4fba21a00ef308fd6d943b92a421e9e33f4ab03ff715241c2b27049781b076d7'
 import torch
 torch.set_num_threads(1)
 def guard(name):
  def unique(*a,**k):raise RuntimeError('GPU API forbidden: '+name)
  return unique
 for name in ['_lazy_init','init','set_device','current_device']:setattr(torch.cuda,name,guard(name))
 torch.cuda.is_available=lambda:False;torch.cuda.device_count=lambda:0
 import janus,transformers,tokenizers
 q['dependencies']={n:{'version':getattr(v,'__version__',None),'source_SHA256':sha(pathlib.Path(v.__file__))}for n,v in [('torch',torch),('transformers',transformers),('tokenizers',tokenizers),('janus',janus)]}
 from infinilm.llm.sampling_params import SamplingParams
 from infinilm.llm.request import FinishReason
 llm.SamplingParams=SamplingParams;llm.FinishReason=FinishReason
 from infinilm.server.agent_protocol import prepare_agent_request
 from prepared_family_entry_r268 import prepare_app,template_prefix_resolver
 from prepared_history import make_prepared_history_prequeue
 from native_engine_bridge_r268 import NativeEngineBridge
 import queue
 same='Shared repeated source content 中文.'
 H=[{'role':'system','content':same},{'role':'user','content':same},{'role':'assistant','content':'  Previous visible assistant content 中文.  ','reasoning_content':'Preserve actual historical reasoning 中文.','tool_calls':[{'id':'opaque272-a','type':'function','function':{'name':'calculate','arguments':'{"a":1,"b":2}'}}]},{'role':'tool','tool_call_id':'opaque272-a','content':'{"sum":3}'},{'role':'user','content':'Continue the complete actual history.'}]
 TOOLS=[{'type':'function','function':{'name':'calculate','parameters':{'type':'object','properties':{'a':{'type':'integer'},'b':{'type':'integer'}},'required':['a','b']}}}]
 P=copy.deepcopy(H[:3]);P[2]['tool_calls']=[{'id':'opaque272-'+name,'type':'function','function':{'name':'calculate','arguments':json.dumps({'a':i,'b':2})}}for i,name in enumerate(['a','b','c'])]
 P += [{'role':'tool','tool_call_id':'opaque272-c','content':''},{'role':'tool','tool_call_id':'opaque272-b','content':'B-result 中文.'},{'role':'tool','tool_call_id':'opaque272-a','content':'A-result 中文.'},{'role':'user','content':'Continue after the complete parallel tool round.'}]
 q['families']={}
 for family,model,metadata,parser in [('GLM','/old/models/ZhipuAI-GLM-5.3',None,'glm5'),('Deep','/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8','/refactor/results/latest-three-processors-r260/Deep-model-metadata','deepseek_v4')]:
  q['stage']=family+'-construct';prepared=prepare_app(family,model,processor_path=metadata);engine,server,processor=prepared['engine'],prepared['server'],prepared['processor'];row={'identity':prepared['identity'],'positive':[],'negative':[]};q['families'][family]=row
  def prepare(history,thinking,budget,generation,parallel=False,choice=None):
   raw={'messages':copy.deepcopy(history),'tools':copy.deepcopy(TOOLS),'tool_choice':choice or {'type':'function','function':{'name':'calculate'}},'parallel_tool_calls':parallel,'max_tokens':budget,'temperature':0.,'add_generation_prompt':generation,'chat_template_kwargs':{'enable_thinking':thinking}}
   data,schemas,selected=prepare_agent_request(raw,parser);data.update(_agent_schemas=schemas,_agent_choice=selected);return data,server._build_sampling_params(data)
  for label,history,thinking,budget,generation,parallel in [('chat',H,False,4096,True,False),('thinking',H,True,8192,True,False),('no-generation',H,True,8192,False,False),('parallel-empty-first-reverse-IDs',P,True,4096,True,True)]:
   q['stage']=family+'-'+label;data,params=prepare(history,thinking,budget,generation,parallel);before=copy.deepcopy(data)
   req=engine.add_chat_request(messages=data['messages'],sampling_params=params,request_id=family+'-272-'+label,request_data=data,add_generation_prompt=generation,chat_template_kwargs=data.get('chat_template_kwargs'));assert engine.take(0)is req and req._admission_metadata_ready and data==before
   h=req.metadata['history_render_audit'];assert h['qualified']and h['input_RC_fields']==1 and len(h['RC_spans'])==1 and all(v['opening_correct']and v['closing_correct']and v['actual_prompt_slice_equal']for v in h['RC_spans'])
   assert h['input_content_parts']==h['rendered_content_contributions']==(6 if parallel else 5)and h['content_one_contribution_each_and_ordered']and h['all_content_contribution_slices_equal']and h['generation_prefix_correct']==(True if generation else None)
   if family=='Deep'and generation:assert h['generation_thinking']==thinking and h['pending_thinking_prefix_open']==thinking and h['pending_chat_prefix_closed']==(not thinking)
   if family=='GLM'and generation:assert h['generation_thinking']is True and h['pending_thinking_prefix_open']is True
   if family=='GLM':engine.record_started(req);engine.record_finished(req,0,True)
   row['positive'].append({'label':label,'prompt_tokens':len(req.prompt_token_ids),'budget':budget,'RC_contributions':h['input_RC_fields'],'content_contributions':h['input_content_parts'],'RC_boundary_spans_verified':True,'actual_generation_thinking':h['generation_thinking'],'requested_enable_thinking':thinking,'generation_prefix_checked':h['generation_prefix_checked'],'prompt_SHA256':h['actual_prompt_SHA256'],'IDs_SHA256':h['Native_prompt_token_ids_SHA256'],'passed':True})
  data,params=prepare(H,True,4096,True,True,'auto');messages=data['messages'];kwargs=data['chat_template_kwargs'];assert messages[0]['content']==messages[1]['content'], 'Negative order control requires repeated equal source text'
  prefix=template_prefix_resolver(processor);opening='<|assistant|><think>'if family=='GLM'else'<｜Assistant｜><think>'
  def normalize(message):
   value=message.get('content');return value.strip()if family=='GLM'and message.get('role')=='assistant'and isinstance(value,str)else value
  for kind in ['missing-content','duplicate-content','cross-history-order','RC-outside-think','broken-pending-prefix']:
   q['stage']=family+'-negative-'+kind
   def corrupted(history,body,generation,options):
    prompt,ids=engine.encode_chat(history,body,generation,options)
    a,b=history[0]['content'],history[1]['content']
    if kind=='missing-content':assert a in prompt;prompt=prompt.replace(a,'',1)
    elif kind=='duplicate-content':assert a in prompt;prompt=prompt.replace(a,a+a,1)
    elif kind=='cross-history-order':
     assert a in prompt and b in prompt;prompt=prompt.replace(a,'CPU_SWAP_SENTINEL_272',1).replace(b,a,1).replace('CPU_SWAP_SENTINEL_272',b,1)
    elif kind=='RC-outside-think':
     value=history[2]['reasoning_content'];span=opening+value+'</think>';assert span in prompt;prompt=prompt.replace(span,opening+'</think>'+value,1)
    elif kind=='broken-pending-prefix':
     suffix=prefix(body,options);assert prompt.endswith(suffix);prompt=prompt[:-len(suffix)]+'BROKEN_PENDING_PREFIX_272'
    encoded=processor(prompt,return_tensors=None)['input_ids'];return prompt,list(encoded)
   audit_bad=make_prepared_history_prequeue(corrupted,prefix,reasoning_boundaries=(opening,'</think>'),normalize_content=normalize)
   bad=NativeEngineBridge(prepared['tokenizer'],32768,FinishReason,encode_chat=corrupted,prequeue_factory=audit_bad,default_max_tokens=4096)
   try:bad.add_chat_request(messages=copy.deepcopy(messages),sampling_params=params,request_id='bad-'+kind,request_data=copy.deepcopy(data),chat_template_kwargs=copy.deepcopy(kwargs))
   except ValueError as exc:failure=hashlib.sha256(str(exc).encode()).hexdigest()
   else:raise AssertionError('Invalid history reached queue: '+kind)
   assert not bad._requests
   try:bad.take(0)
   except queue.Empty:pass
   else:raise AssertionError('Rejected history published to queue')
   row['negative'].append({'kind':kind,'rejected_before_queue_publication':True,'error_SHA256':failure,'passed':True})
  row['passed']=True
 q.update(passed=True,stage='complete',source_manifest_SHA256=sha(mf),actual_root_or_ABI_qualified=False,root_core_llm_guarded=True);assert not torch.cuda.is_initialized();q['torch_cuda_initialized']=False
except BaseException as exc:q.update(error_type=type(exc).__name__,error=str(exc)[:1600],failure_locations=[{'file':v.filename,'line':v.lineno,'function':v.name}for v in traceback.extract_tb(exc.__traceback__)])
finally:
 q['completed']=True;(O/'CPU-prepared.json').write_text(json.dumps(q,indent=2)+chr(10));print(json.dumps({k:q.get(k)for k in ['actual_step','completed','passed','stage','error_type','error']}))
assert q['passed']
