"""Changed default delivery through actual public route and real Deep codec.

HTTP Request/transport and Native emit/reset/finish are recording leaves.
No server constructor, Native library, model payload or device is executed.
"""
from pathlib import Path
import ast,asyncio,copy,hashlib,importlib.util,io,json,logging,os,queue,sys,time,traceback,types,uuid
from enum import Enum
from types import SimpleNamespace
D=Path(__file__).resolve().parent;R=Path('/refactor')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

class Reasons(str,Enum):
 EOS_TOKEN='eos_token';STOP_STRING='stop_string';LENGTH='length';TIMEOUT='timeout';ERROR='error';CANCELED='canceled'

def actual_public_route(public,Parser,prepare,engine):
 # Original AST bodies are consumed, including the nested public endpoint.
 # Only @app decorators and the unused full InferenceServer constructor are
 # excluded. Nothing is installed into or monkeypatched on a server class.
 from fastapi.responses import JSONResponse,StreamingResponse
 from infinilm.server.agent_protocol import AgentProtocolError,collect_message
 path=public/'infinilm/server/inference_server.py';tree=ast.parse(path.read_bytes())
 cls=next(v for v in tree.body if isinstance(v,ast.ClassDef)and v.name=='InferenceServer')
 names={'_build_sampling_params','_agent_parser','_stream_chat','_chat','_convert_finish_reason'}
 actual=copy.deepcopy(cls);actual.bases=[ast.Name(id='object',ctx=ast.Load())];actual.keywords=[]
 actual.body=[v for v in actual.body if getattr(v,'name',None)in names]
 create=next(v for v in cls.body if getattr(v,'name',None)=='_register_routes')
 endpoint=copy.deepcopy(next(v for v in create.body if isinstance(v,ast.AsyncFunctionDef)and v.name=='chat_completions'));endpoint.decorator_list=[]
 helpers=[copy.deepcopy(v)for v in tree.body if isinstance(v,ast.FunctionDef)and v.name in {'sse_json','error_json','chunk_json','completion_json'}]
 ns=dict(__name__='Deep435_actual_public_route',SamplingParams=SimpleNamespace,ToolCallParser=Parser,FinishReason=Reasons,asyncio=asyncio,json=json,time=time,logger=logging.getLogger('Deep435CPU'),uuid=uuid,JSONResponse=JSONResponse,StreamingResponse=StreamingResponse,AgentProtocolError=AgentProtocolError,collect_message=collect_message,prepare_agent_request=prepare)
 module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*helpers,actual,endpoint],type_ignores=[])
 exec(compile(ast.fix_missing_locations(module),str(path),'exec'),ns)
 server=ns['InferenceServer']();server.engine=engine;server.max_tokens=4096;server.temperature=0.;server.top_p=1.;server.top_k=1;server.ignore_eos=False;server.tool_call_parser='deepseek_v4';server.model_id='qualified-Deep-afc6';server.stream_timeout_s=2.;server.request_timeout_s=5.
 from public_deep_default_text.prepared_factory import bind_public_sampling
 bind_public_sampling(server,engine);ns['self']=server
 return ns['chat_completions'],server,StreamingResponse

async def consume(public,Parser,prepare,processor,model,seq,*,name,stream='original',explicit='absent',control=None):
 from public_deep_default_text.prepared_factory import prepare_engine
 body=copy.deepcopy(seq['body'])
 if stream=='missing':body.pop('stream',None)
 elif stream!='original':body['stream']=stream
 if explicit!='absent':body['text_delivery']=explicit
 expected=explicit if explicit!='absent'else('P1'if body.get('stream')is True else'P0')
 engine=prepare_engine('Deep',str(model),processor=processor,finish_reasons=Reasons)['engine'];engine.start()
 route,server,Streaming=actual_public_route(public,Parser,prepare,engine)
 if control=='timeout':server.stream_timeout_s=.02;server.request_timeout_s=.04
 chunks=[];first_semantic_before_finish=None;owned=[None];request_data=copy.deepcopy(body)
 class Request:
  async def json(self):return copy.deepcopy(request_data)
  async def is_disconnected(self):return control=='disconnect'
 async def read_route():
  nonlocal first_semantic_before_finish
  response=await route(Request())
  if isinstance(response,Streaming):
   assert response.status_code==200
   async for line in response.body_iterator:
    line=line.decode()if isinstance(line,bytes)else line;chunks.append(line)
    if line.startswith('data: ')and line[6:].strip()!='[DONE]':
     value=json.loads(line[6:]);semantic=any(any(c.get('delta',{}).get(k)for k in ['content','reasoning_content','tool_calls'])for c in value.get('choices',[]))
     if semantic and first_semantic_before_finish is None:
      assert owned[0]is not None;first_semantic_before_finish=not owned[0].is_finished()
   return response,None
  return response,json.loads(response.body)
 task=asyncio.create_task(read_route());req=None
 try:
  limit=time.monotonic()+5
  while req is None and time.monotonic()<limit:
   try:req=engine.take(timeout=0)
   except queue.Empty:
    if task.done():raise AssertionError('Actual public route rejected before expected admission:'+repr(task.result()))
    await asyncio.sleep(.001)
  assert req is not None;owned[0]=req
  digest=hashlib.sha256(json.dumps(list(req.prompt_token_ids),separators=(',',':')).encode()).hexdigest()
  assert digest==seq['original_prompt_IDs_SHA256']and req.metadata['text_delivery']==expected
  assert type(engine).__module__=='public_deep_default_text.prepared_factory'
  assert not any(v.startswith('public_deep_text')for v in sys.modules)
  if control=='timeout':
   await asyncio.wait_for(task,2)
   assert req.cancel_event.is_set()and req.abort_reason==Reasons.TIMEOUT and not req.is_finished()
  else:
   for token in seq['visible_ids']:
    req.emit_token(token);await asyncio.sleep(.001)
    if control=='disconnect'and req.cancel_event.is_set():break
   if control=='disconnect':
    await asyncio.wait_for(task,2)
    assert req.cancel_event.is_set()and req.abort_reason==Reasons.CANCELED and not req.is_finished()
   else:
    if body.get('stream')is True:
     if expected=='P1':assert first_semantic_before_finish is True
     else:assert first_semantic_before_finish is None
    req.emit_token(1)
  # This is the explicitly recorded Native owner leaf, after its reset/drain.
  # Neither a route timeout nor disconnect is allowed to forge this finish.
  before_finish=req.is_finished();assert not before_finish
  req.metadata['recording_owned_reset_drain']=True;req.finish(Reasons.EOS_TOKEN)
  response,wire=await asyncio.wait_for(task,3)
  if control:
   assert req.finish_reason==req.abort_reason
   return dict(case=name,passed=True,mode=expected,intent=control,not_finished_until_recording_owner=True,owner_finish_preserves_abort_reason=True,route='actual_public_stream',prompt_tokens=len(req.prompt_token_ids))
  if body.get('stream')is True:
   content='';reasoning='';finish=None;usage=None;done=False
   for line in ''.join(chunks).splitlines():
    if not line.startswith('data: '):continue
    if line[6:]=='[DONE]':done=True;continue
    value=json.loads(line[6:]);assert'error'not in value
    if value.get('usage')is not None:usage=value['usage']
    for choice in value.get('choices',[]):
     delta=choice.get('delta',{});assert not delta.get('tool_calls');content+=delta.get('content')or'';reasoning+=delta.get('reasoning_content')or''
     finish=choice.get('finish_reason')or finish
   assert done and finish=='stop';actual=dict(content=content,reasoning_content=reasoning)
  else:
   assert response.status_code==200 and'error'not in wire;choice=wire['choices'][0];assert choice['finish_reason']=='stop';actual=choice['message'];usage=wire['usage']
  wanted=seq['expected_message'];assert actual.get('content','')==(wanted.get('content')or'')and actual.get('reasoning_content','')==(wanted.get('reasoning_content')or'')and not actual.get('tool_calls')
  assert req.generated_token_ids==seq['visible_ids']+[1]and usage['prompt_tokens']==len(req.prompt_token_ids)and usage['completion_tokens']==len(req.generated_token_ids)
  return dict(case=name,passed=True,selected=expected,original_index=seq['index'],actual_prompt_tokens=len(req.prompt_token_ids),prompt_IDs_SHA256=digest,actual_public_route='SSE'if body.get('stream')is True else'JSON',first_semantic_before_owned_finish=first_semantic_before_finish,complete_codec_parser_semantics_match=True,HTTP_or_Native_device=False)
 finally:
  if not task.done():task.cancel()
  try:await task
  except asyncio.CancelledError:pass
  if req is not None and not req.is_finished():req.mark_abort(Reasons.ERROR);req.metadata['recording_cleanup_owner_reset_drain']=True;req.finish(Reasons.ERROR)
  engine.stop()

async def rejected(public,Parser,prepare,processor,model,seq,value):
 from public_deep_default_text.prepared_factory import prepare_engine
 engine=prepare_engine('Deep',str(model),processor=processor,finish_reasons=Reasons)['engine'];engine.start()
 route,server,Streaming=actual_public_route(public,Parser,prepare,engine)
 body=copy.deepcopy(seq['body']);body['text_delivery']=value
 class Request:
  async def json(self):return body
 try:
  response=await route(Request());assert response.status_code==400 and not isinstance(response,Streaming)
  assert engine._pending.empty()and not engine._requests
  return dict(case='invalid_explicit_pre_admission',input_type=type(value).__name__,passed=True,status=400,no_request_or_owner_work=True)
 finally:engine.stop()

def main():
 out=R/'candidates/Deep-public-default-incremental-r435-CPU-v2';out.mkdir(exist_ok=False);report=out/'result.json';start=time.monotonic()
 q=dict(completed=False,passed=False,actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],GPU=False,Native_dlopen=None,Model_READY=False,cases=[],stages=[],weight_open_attempts=[],old_nine_cases_replayed=False,new_compile=False)
 def save():
  p=report.with_suffix('.tmp');p.write_text(json.dumps(q,indent=2)+'\n');p.replace(report)
 def observe(stage):
  torch=sys.modules.get('torch');initialized=None if torch is None else bool(torch.cuda.is_initialized())
  paths=sorted({v.split()[-1]for v in Path('/proc/self/maps').read_text().splitlines()if len(v.split())>=6 and v.split()[-1].startswith('/')})
  markers=['libDeep_','libGLM_','libKimi_','_infinilm.cpython','libinfinirt.so','libinfinilm_runtime.so','libinfiniops.so','libinfinicore.so','libinfiniccl.so']
  project=[p for p in paths if p.startswith('/refactor/')or any(n in p for n in markers)]
  actual={}
  for name,module in list(sys.modules.items()):
   if name.startswith('public_deep_default_text'):
    path=Path(module.__file__).resolve();assert path.is_relative_to(D.resolve());relative=path.relative_to(D.resolve()).as_posix();assert sha(path)==m['files_SHA256'][relative]
    actual[name]=dict(path=str(path),SHA256=sha(path))
  q['stages'].append(dict(stage=stage,project_maps=project,framework_modules={n:getattr(sys.modules.get(n),'__file__',None)for n in ['torch','transformers','tokenizers','numpy']},torch_cuda_initialized=initialized,hidden={n:os.environ[n]for n in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES']},actual_new_modules=actual));save()
  assert not project and initialized in[None,False]and all(os.environ[n]=='-1'for n in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
 save()
 try:
  assert sha(D/'SOURCE_MANIFEST.json')==os.environ['R435_CPU_SOURCE_SHA256'];m=json.loads((D/'SOURCE_MANIFEST.json').read_bytes())
  for n,h in m['files_SHA256'].items():assert sha(D/n)==h,n
  plan=json.loads((D/'PLAN.json').read_bytes());old=R/'incoming/Deep-public-incremental-text-r430';legacy=old/'cpu-v2'
  assert sha(legacy/'SOURCE_MANIFEST.json')==plan['legacy_CPU_helper_manifest_SHA256']
  for n,h in json.loads((legacy/'SOURCE_MANIFEST.json').read_bytes())['files_SHA256'].items():assert sha(legacy/n)==h,n
  fixture=R/'results/Deep430-original-two-outputs-private.json';assert sha(fixture)==plan['private_fixture_SHA256'];data=json.loads(fixture.read_bytes());assert len(data['sequences'])==2
  model=Path(plan['tokenizer_path'])
  for n,h in plan['tokenizer_assets_SHA256'].items():assert sha(model/n)==h,n
  def audit(event,args):
   if event=='open'and isinstance(args[0],(str,bytes,os.PathLike))and os.fsdecode(args[0]).lower().endswith(('.safetensors','.bin','.pt','.pth')):
    q['weight_open_attempts'].append(os.fsdecode(args[0]));raise RuntimeError('Weight payload forbidden')
  sys.addaudithook(audit);parent=R/'incoming/public-Deep-native-provider-r427/source-v3';public=R/'incoming/public-GLM-native-provider-r425/public-python';sys.path[:0]=[str(D),str(parent)];observe('before_imports')
  from public_deep_default_text.cli import main as cli
  config=D/'config.template.json';text=io.StringIO()
  from contextlib import redirect_stdout
  with redirect_stdout(text):cli(['--config',str(config),'--config-sha',sha(config),'--source-sha',sha(D/'SOURCE_MANIFEST.json'),'--run','CPU-source-only','--actual-step',q['actual_CPU'],'--gpu-start-epoch','0'])
  assert json.loads(text.getvalue())==dict(source_and_schema_valid=True,Model_READY=False,Native_dlopen=False,GPU=False);observe('actual_new_module_schema')
  spec=importlib.util.spec_from_file_location('pinned_430_processor_helpers',legacy/'CPU_entry.py');helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
  Parser,prepare,history=helpers.pure_public(public);processor,methods=helpers.actual_processor(public,model,history);observe('actual_Deep_processor_codec_parser')
  async def cases():
   specs=[dict(name='default_nonstream_original_body',seq=0),dict(name='default_stream_true_original_body',seq=1),dict(name='default_stream_missing',seq=0,stream='missing'),dict(name='explicit_P0_overrides_stream_true',seq=1,explicit='P0'),dict(name='explicit_P1_overrides_stream_false',seq=0,explicit='P1'),dict(name='default_P1_actual_route_timeout',seq=1,control='timeout'),dict(name='default_P1_actual_route_disconnect',seq=1,control='disconnect')]
   for spec in specs:
    spec=dict(spec);seq=data['sequences'][spec.pop('seq')];q['active_case']=spec['name'];save();q['cases'].append(await consume(public,Parser,prepare,processor,model,seq,**spec));save()
   for value in [None,'P2',1]:
    q['active_case']='invalid_explicit_'+type(value).__name__;save();q['cases'].append(await rejected(public,Parser,prepare,processor,model,data['sequences'][1],value));save()
  asyncio.run(cases());observe('actual_routes_default_override_rejection_and_cleanup')
  assert not q['weight_open_attempts']and all(v['passed']for v in q['cases'])and len(q['cases'])==10
  q.update(passed=True,Native_dlopen=False,source_manifest_SHA256=sha(D/'SOURCE_MANIFEST.json'),actual_public_source_SHA256=sha(public/'infinilm/server/inference_server.py'),actual_Deep_method_sources=methods,private_default_effect=True,Native_owner_is_recording_leaf=True,real_HTTP_or_Model=False,new_default_device_qualified=False,scope='Actual public endpoint/sampling/JSON+SSE/parser methods and actual prepared factory/Deep codec. Only package init/unused server constructor, HTTP Request/transport and Native owner are recording leaves; no old all-prefix/nine-case suite.')
 except BaseException as error:q.update(passed=False,error=repr(error),traceback=traceback.format_exc());raise
 finally:q.update(completed=True,wall_s=time.monotonic()-start);save()

if __name__=='__main__':main()
