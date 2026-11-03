"""Deep-specific real tokenizer/public-parser/new-selector consumer, zero Native/GPU."""
from pathlib import Path
import ast,asyncio,contextlib,copy,hashlib,importlib,importlib.util,io,json,os,sys,time,traceback,types
from types import SimpleNamespace
D=Path(__file__).resolve().parent;SOURCE=D.parent;R=Path('/refactor');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pure_public(root):
 # Only package __init__ registration imports are bypassed to exclude Native.
 # Actual parser/protocol/history modules below execute their unmodified files.
 for name,relative in [('infinilm','infinilm'),('infinilm.server','infinilm/server'),('infinilm.processors','infinilm/processors')]:
  m=types.ModuleType(name);m.__path__=[str(root/relative)];sys.modules[name]=m
 from infinilm.server.tool_parsers import ToolCallParser
 from infinilm.server.agent_protocol import prepare_agent_request
 from infinilm.processors.history_encoding import deepseek_history_encoder
 return ToolCallParser,prepare_agent_request,deepseek_history_encoder

def actual_processor(public,model,history_encoder):
 from transformers import PreTrainedTokenizerFast
 paths=[public/'infinilm/processors/basic_llm_processor.py',public/'infinilm/processors/deepseek_v4_processor.py']
 classes=[]
 for path,name,methods in [(paths[0],'BasicLLMProcessor',{'__call__','get_tokenizer'}),(paths[1],'DeepSeekV4Processor',{'__init__','apply_chat_template'})]:
  original=next(v for v in ast.parse(path.read_bytes()).body if isinstance(v,ast.ClassDef)and v.name==name)
  actual=copy.deepcopy(original);actual.body=[v for v in actual.body if getattr(v,'name',None)in methods]
  if name=='BasicLLMProcessor':actual.bases=[ast.Name(id='object',ctx=ast.Load())]
  classes.append(actual)
 ns=dict(__name__='Deep430_actual_public_CPU_methods',PreTrainedTokenizerFast=PreTrainedTokenizerFast,Path=Path,importlib=importlib,json=json,copy=copy,deepseek_history_encoder=history_encoder,override=lambda f:f)
 exec(compile(ast.fix_missing_locations(ast.Module(body=classes,type_ignores=[])),'actual_public_Deep_CPU_methods','exec'),ns)
 # Actual initializer, official history encoder and Basic __call__/get_tokenizer;
 # excluded scheduling methods would import Native and are not used here.
 return ns['DeepSeekV4Processor'](str(model)),{str(p.relative_to(public)):sha(p)for p in paths}

def public_consumer(public,Parser,engine):
 # Execute the actual two public method bodies; only unused server/FastAPI init
 # and the sampling data container are leaves. No second route implementation.
 path=public/'infinilm/server/inference_server.py';tree=ast.parse(path.read_bytes())
 original=next(v for v in tree.body if isinstance(v,ast.ClassDef)and v.name=='InferenceServer')
 actual=copy.deepcopy(original);actual.bases=[ast.Name(id='object',ctx=ast.Load())];actual.keywords=[]
 actual.body=[v for v in actual.body if getattr(v,'name',None)in ['_build_sampling_params','_agent_parser']]
 ns=dict(__name__='Deep430_actual_public_CPU_consumer',SamplingParams=SimpleNamespace,ToolCallParser=Parser)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[actual],type_ignores=[])),str(path),'exec'),ns)
 obj=ns['InferenceServer']();obj.max_tokens=4096;obj.temperature=0.;obj.top_p=1.;obj.top_k=1;obj.ignore_eos=False;obj.tool_call_parser='deepseek_v4'
 from public_deep_text.prepared_factory import bind_public_sampling
 bind_public_sampling(obj,engine);return obj

def aggregation(parser,chunks):
 result=dict(content='',reasoning_content='',tool_calls=[])
 for text,final in chunks:
  events=parser.feed(text)if text else[]
  if final:events+=parser.feed('',final=True,finish_reason='stop')
  for event in events:
   for k in ['content','reasoning_content']:result[k]+=event.get(k)or''
   result['tool_calls']+=event.get('tool_calls',[])
 return result

def main():
 out=R/'candidates/Deep-public-incremental-text-r430-CPU-v2';out.mkdir(exist_ok=False);report=out/'result.json';start=time.monotonic()
 q=dict(completed=False,passed=False,actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],GPU=False,Native_dlopen=False,weight_payload_read=False,Model_READY=False,cases=[])
 def save():
  t=report.with_suffix('.tmp');t.write_text(json.dumps(q,indent=2)+'\n');t.replace(report)
 save()
 try:
  assert all(os.environ[k]=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
  assert sha(D/'SOURCE_MANIFEST.json')==os.environ['R430_CPU_SOURCE_SHA256'];m=json.loads((D/'SOURCE_MANIFEST.json').read_bytes())
  for name,h in m['files_SHA256'].items():assert sha(D/name)==h,name
  assert sha(SOURCE/'SOURCE_MANIFEST.json')==m['production_source_manifest_SHA256']
  original=json.loads((SOURCE/'SOURCE_MANIFEST.json').read_bytes())
  for name,h in original['files_SHA256'].items():assert sha(SOURCE/name)==h,name
  plan=json.loads((SOURCE/'PLAN.json').read_bytes());fixture=R/'results/Deep430-original-two-outputs-private.json';assert sha(fixture)==plan['private_fixture_SHA256'];data=json.loads(fixture.read_bytes());assert len(data['sequences'])==2
  model=Path(plan['tokenizer_path'])
  for name,h in plan['tokenizer_assets_SHA256'].items():assert sha(model/name)==h,name
  parent=R/'incoming/public-Deep-native-provider-r427/source-v3';public=R/'incoming/public-GLM-native-provider-r425/public-python'
  assert sha(parent/'SOURCE_MANIFEST.json')==plan['qualified_427_source_SHA256']
  for name,h in json.loads((parent/'SOURCE_MANIFEST.json').read_bytes())['files_SHA256'].items():assert sha(parent/name)==h,name
  assert sha(public.parent/'SOURCE_MANIFEST.json')==plan['qualified_public_source_SHA256']
  for name,h in json.loads((public.parent/'SOURCE_MANIFEST.json').read_bytes())['files_SHA256'].items():assert sha(public.parent/name)==h,name
  def audit(event,args):
   if event=='open'and isinstance(args[0],(str,bytes,os.PathLike))and os.fsdecode(args[0]).lower().endswith(('.safetensors','.bin','.pt','.pth')):raise RuntimeError('Weight/binary payload forbidden')
  sys.addaudithook(audit);sys.path[:0]=[str(SOURCE),str(parent)]
  from public_deep_text.cli import main as source_cli
  config=SOURCE/'config.template.json';output=io.StringIO()
  with contextlib.redirect_stdout(output):source_cli(['--config',str(config),'--config-sha',sha(config),'--source-sha',sha(SOURCE/'SOURCE_MANIFEST.json'),'--run','CPU-source-only','--actual-step',q['actual_CPU'],'--gpu-start-epoch','0'])
  assert json.loads(output.getvalue())==dict(source_and_schema_valid=True,Model_READY=False,Native_dlopen=False,GPU=False)
  Parser,prepare,history=pure_public(public);processor,method_sources=actual_processor(public,model,history);tokenizer=processor.get_tokenizer()
  from public_deep_text.prepared_factory import prepare_engine
  from public_deep_text.incremental_text import TextPrefix
  class Reasons:
   TIMEOUT='timeout';ERROR='error';CANCELED='canceled';EOS_TOKEN='eos_token';LENGTH='length'
  async def consume(seq,mode=None,timeout=False):
   body=copy.deepcopy(seq['body'])
   if mode is not None:body['text_delivery']=mode
   prepared,schemas,choice=prepare(body,parser='deepseek_v4')
   prepared['_agent_schemas']=schemas
   prepared['_agent_choice']=choice
   engine=prepare_engine('Deep',str(model),processor=processor,finish_reasons=Reasons)['engine'];engine.start()
   server=public_consumer(public,Parser,engine)
   server._build_sampling_params(prepared)  # actual route pre-SSE validation
   params=server._build_sampling_params(prepared)  # actual _chat/_stream_chat sampling
   req=engine.add_chat_request(messages=prepared['messages'],sampling_params=params,request_id='CPU-'+str(seq['index'])+'-'+str(mode)+'-'+str(timeout),request_data=prepared,add_generation_prompt=bool(prepared.get('add_generation_prompt',True)),chat_template_kwargs=prepared.get('chat_template_kwargs')or{})
   assert engine.take(timeout=1)is req
   ids=list(req.prompt_token_ids);digest=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest();assert digest==seq['original_prompt_IDs_SHA256']
   actual_mode=mode or'P0';assert req.metadata['text_delivery']==actual_mode
   parser=server._agent_parser(prepared,req)
   stream=engine.stream_request(req,timeout=.02 if timeout else 2.,request_timeout=.04 if timeout else 5.)
   if timeout:
    result=await anext(stream);assert result.finish_reason==Reasons.TIMEOUT and req.cancel_event.is_set()and not req.is_finished()
    await stream.aclose();req.metadata['recording_owner_reset_completed']=True;req.finish(Reasons.EOS_TOKEN);assert req.finish_reason==Reasons.TIMEOUT
    return dict(timeout_abort_intent=True,Native_finish_remains_owner=True)
   waiting=asyncio.create_task(anext(stream));visible=seq['visible_ids'];first=None;used=0
   for token in visible:
    req.emit_token(token);used+=1;await asyncio.sleep(.001)
    if waiting.done():first=waiting.result();break
   if actual_mode=='P1':
    assert first is not None and first.token_text and not first.finished and not req.is_finished();chunks=[(first.token_text,first.finished)]
   else:assert first is None;chunks=[]
   for token in visible[used:]:req.emit_token(token)
   req.emit_token(1);req.metadata['recording_owner_reset_completed']=True;req.finish(Reasons.EOS_TOKEN)
   if first is None:
    first=await waiting;chunks.append((first.token_text,first.finished))
   chunks +=[(v.token_text,v.finished)async for v in stream]
   assert chunks[-1][1]and req.generated_token_ids==visible+[1]and not req.cancel_event.is_set()
   merged=aggregation(parser,chunks);expected=seq['expected_message']
   assert merged['content']==(expected.get('content')or'')and merged['reasoning_content']==(expected.get('reasoning_content')or'')and not merged['tool_calls']
   return dict(mode=actual_mode,original_index=seq['index'],actual_prompt_tokens=len(ids),actual_prompt_IDs_SHA256=digest,visible_tokens=len(visible),text_chunks=len(chunks),first_before_owned_finish=actual_mode=='P1',final_semantics_match_original_wire=True,final_content_SHA256=hashlib.sha256(merged['content'].encode()).hexdigest(),final_reasoning_SHA256=hashlib.sha256(merged['reasoning_content'].encode()).hexdigest())
  async def all_cases():
   for seq in data['sequences']:
    q['active_case']=dict(name='actual_Deep_allprefix_decoder_public_parser',index=seq['index']);save()
    # Every prefix independently goes through the actual decoder + scanner.
    prefix=TextPrefix(lambda ids:tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False));chunks=[]
    for n in range(1,len(seq['visible_ids'])+1):chunks.append((prefix.advance(seq['visible_ids'][:n],final=False),False))
    chunks.append((prefix.advance(seq['visible_ids'],final=True),True));parsed=aggregation(Parser({},thinking=seq['thinking'],choice='none',parallel=False,parser='deepseek_v4'),chunks)
    assert parsed['content']==(seq['expected_message'].get('content')or'')and parsed['reasoning_content']==(seq['expected_message'].get('reasoning_content')or'')
    q['cases'].append(dict(case='actual_Deep_allprefix_decoder_and_actual_public_parser',index=seq['index'],prefixes=len(seq['visible_ids']),passed=True));save()
    for mode in [None,'P1']:
     q['active_case']=dict(name='actual_private_factory_selector_public_parser',index=seq['index'],mode=mode or'P0');save()
     q['cases'].append(dict(case='actual_private_factory_selector_public_parser',**await consume(seq,mode),passed=True));save()
   q['active_case']=dict(name='actual_new_P1_selector_timeout_abort');save()
   q['cases'].append(dict(case='actual_new_P1_selector_timeout_abort',**await consume(data['sequences'][0],'P1',timeout=True),passed=True));save()
  asyncio.run(all_cases())
  q['active_case']=dict(name='actual_Deep_tokenizer_DSML_Unicode_control');save()
  control='<｜DSML｜tool_calls>\n<｜DSML｜invoke name="read_file">\n<｜DSML｜parameter name="path" string="true">中文🙂.json</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>'
  ids=tokenizer.encode(control,add_special_tokens=False);prefix=TextPrefix(lambda ids:tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False));parts=[]
  for n in range(1,len(ids)+1):parts.append((prefix.advance(ids[:n],final=False),False))
  parts.append((prefix.advance(ids,final=True),True));parsed=aggregation(Parser({'read_file':{'type':'object','properties':{'path':{'type':'string'}},'required':['path'],'additionalProperties':False}},thinking=False,choice='required',parallel=False,parser='deepseek_v4'),parts)
  assert len(parsed['tool_calls'])==1 and parsed['tool_calls'][0]['function']['name']=='read_file'and json.loads(parsed['tool_calls'][0]['function']['arguments'])=={'path':'中文🙂.json'}
  q['cases'].append(dict(case='Deep_actual_tokenizer_DSML_Unicode_complete_typed_tool',synthetic_protocol_control=True,passed=True))
  bad=copy.deepcopy(data['sequences'][0]['body']);bad['text_delivery']='P2';bad,schemas,choice=prepare(bad,parser='deepseek_v4');bad['_agent_schemas']=schemas;bad['_agent_choice']=choice
  negative=prepare_engine('Deep',str(model),processor=processor,finish_reasons=Reasons)['engine']
  try:public_consumer(public,Parser,negative)._build_sampling_params(bad)
  except ValueError:pass
  else:raise AssertionError('Invalid text delivery accepted')
  assert negative._pending.empty()and not negative._requests
  q['cases'].append(dict(case='invalid_mode_rejected_by_actual_prepared_validate_before_queue',passed=True))
  q.update(passed=True,tokenizer_class=type(tokenizer).__module__+'.'+type(tokenizer).__qualname__,public_processor_method_sources=method_sources,source_manifest_SHA256=sha(D/'SOURCE_MANIFEST.json'),production_source_manifest_SHA256=sha(SOURCE/'SOURCE_MANIFEST.json'),actual_checkpoint_tokenizer=True,actual_public_parser=True,actual_private_prepared_factory=True,original_Model_sequences=2,real_HTTP_or_Native_Model=False,CPU_scaffolding='Only package init and unused Native scheduling methods excluded; original public Deep initializer/template/Basic.__call__/get_tokenizer bodies executed. SamplingParams is a data-container leaf; actual public sampling/parser method bodies execute. Native owner emit/reset/finish actions are recording leaves.',prefix_deltas_are_not_per_token_client_timestamps=True)
  maps=Path('/proc/self/maps').read_text();assert not any(v in maps for v in ['libDeep_','libGLM_','libKimi_','_infinilm.cpython','libinfinirt.so']);assert 'torch'not in sys.modules
 except BaseException as error:q.update(error=repr(error),traceback=traceback.format_exc());raise
 finally:q.update(completed=True,wall_s=time.monotonic()-start);save()
if __name__=='__main__':main()
