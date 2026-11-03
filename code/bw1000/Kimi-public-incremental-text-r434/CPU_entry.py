"""New Kimi text consumer on exact two original572 outputs; no Native/weights/device."""
from pathlib import Path
import ast,asyncio,contextlib,copy,hashlib,importlib,importlib.util,io,json,os,re,sys,time,traceback,types
from enum import Enum
from types import SimpleNamespace
from unittest.mock import patch
D=Path(__file__).resolve().parent;R=Path('/refactor');B=R/'candidates/Kimi-public-incremental-text-r434-CPU-v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();ids_sha=lambda ids:hashlib.sha256(json.dumps(list(ids),separators=(',',':')).encode()).hexdigest()
class Reasons(str,Enum):
 EOS_TOKEN='eos_token';LENGTH='length';TIMEOUT='timeout';CANCELED='canceled';ERROR='error';STOP_STRING='stop_string'
def metadata():
 maps=Path('/proc/self/maps').read_text();tokens=['_infinilm.cpython','libKimi_','libDeep_','libGLM_','libinfinilm_runtime','libinfiniops','libinfinirt']
 native=[line for line in maps.splitlines()if any(v in line for v in tokens)];torch=sys.modules.get('torch');cuda=getattr(torch,'cuda',None)if torch is not None else None
 return dict(project_Native_maps=native,Torch_imported=torch is not None,Transformers_imported='transformers'in sys.modules,Torch_cuda_initialized=getattr(cuda,'_initialized',None)if cuda is not None else None,visible_env={k:os.environ.get(k)for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES']})
def pure_public(public):
 # A private package prefix only; no fake infinilm classes or import blockers.
 # Both actual protocol/scanner files execute unchanged with relative imports.
 prefix='kimi434_actual_public_leaf';m=types.ModuleType(prefix);m.__path__=[str(public/'infinilm/server')];sys.modules[prefix]=m
 protocol=importlib.import_module(prefix+'.agent_protocol');parsers=importlib.import_module(prefix+'.tool_parsers')
 assert Path(protocol.__file__).resolve()==(public/'infinilm/server/agent_protocol.py').resolve()
 assert Path(parsers.__file__).resolve()==(public/'infinilm/server/tool_parsers/__init__.py').resolve()
 source=public/'infinilm/server/inference_server.py';tree=ast.parse(source.read_bytes());cls=next(v for v in tree.body if isinstance(v,ast.ClassDef)and v.name=='InferenceServer');fn=next(v for v in cls.body if isinstance(v,ast.FunctionDef)and v.name=='_agent_parser');ns={'ToolCallParser':parsers.ToolCallParser};exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(fn)],type_ignores=[])),str(source),'exec'),ns)
 return protocol,parsers.ToolCallParser,ns['_agent_parser'],dict(parser_method_AST_SHA256=hashlib.sha256(ast.dump(fn,include_attributes=False).encode()).hexdigest(),source_SHA256=sha(source),only_excluded='Public package __init__ Native registration and HTTP server construction; actual protocol/parser/history/newprepared run unchanged.')
def comparable(message):
 calls=[]
 for c in message.get('tool_calls',[]):
  assert re.fullmatch(r'call_[0-9a-f]{32}',c['id']),'Actual parser must issue an opaque fresh call ID'
  calls.append(dict(name=c['function']['name'],arguments=json.loads(c['function']['arguments'])))
 return dict(content=message.get('content')or'',reasoning_content=message.get('reasoning_content')or'',tool_calls=calls)
def aggregate(protocol,events):return protocol.collect_message(events)
def main():
 assert not B.exists(),'Immutable CPU result exists; reconcile only';B.mkdir();out=B/'result.json';start=time.monotonic()
 q=dict(completed=False,passed=False,actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],CPU_only=True,GPU=0,Native_dlopen=False,weight_payload_read=False,Model_READY=False,cases=[],source_scope='Actual K3 codec/public protocol+parser/new prepared queue; Native emit/reset/finish owner is recorded, not device/HTTP qualification.')
 def private(name,value):
  p=B/(name+'-private.json');tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(p);return dict(path=str(p),SHA256=sha(p),bytes=p.stat().st_size)
 def save():p=out.with_suffix('.tmp');p.write_text(json.dumps(q,indent=2,ensure_ascii=False,allow_nan=False)+'\n');p.replace(out)
 save()
 try:
  assert all(os.environ.get(k)=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
  mf=D/'SOURCE_MANIFEST.json';assert sha(mf)==os.environ['KIMI_R434_CPU_SOURCE_SHA'];m=json.loads(mf.read_bytes());q['source_manifest_SHA256']=sha(mf)
  for name,h in m['files_SHA256'].items():assert sha(D/name)==h,name
  plan=json.loads((D/'CPU_PLAN.json').read_bytes());parent=Path(plan['qualified_parent_root']);public=Path(plan['public_root'])
  from public_kimi_text.cli import verify_tree,main as actual_cli
  verify_tree(parent,plan['parent_manifest_SHA256']);verify_tree(public.parent,plan['public_manifest_SHA256'])
  q['initial_module_maps']=metadata();assert not q['initial_module_maps']['project_Native_maps']
  def no_weights(event,args):
   if event=='open'and isinstance(args[0],(str,bytes,os.PathLike))and os.fsdecode(args[0]).lower().endswith(('.safetensors','.pt','.pth','.bin')):raise RuntimeError('Checkpoint weight payload outside this CPU scope')
  sys.addaudithook(no_weights)
  fixture=Path(plan['private_fixture_path']);assert sha(fixture)==plan['private_fixture_SHA256'];data=json.loads(fixture.read_bytes());assert len(data['sequences'])==2 and[len(x['visible_ids'])for x in data['sequences']]==[58,121]
  # Actual final self/parent import route and actual configure function. Only the
  # 4 model-node hostname+Slurm identity are recording leaves on this CPU node.
  cfg=json.loads((D/'config.template.json').read_bytes());cli_rows=[]
  for role in range(4):
   env=dict(os.environ,SLURM_PROCID=str(role),SLURM_JOB_ID='CPU',SLURM_STEP_ID='434source',GLOBAL_OFFSET_R2354='0');output=io.StringIO()
   with patch.dict(os.environ,env,clear=True),patch('socket.gethostname',return_value=cfg['nodes'][role]),contextlib.redirect_stdout(output):
    actual_cli(['--config',str(D/'config.template.json'),'--config-sha',sha(D/'config.template.json'),'--source-sha',sha(mf),'--node',str(role),'--run','CPU434source','--nonce','CPU434-source-only-not-Model-auth-identity','--actual-step','CPU.434source','--gpu-start-epoch','0'])
   row=json.loads(output.getvalue());assert row['source_and_config_valid']and row['topology']['node']==role and not row['GPU']and not row['Native_dlopen'];cli_rows.append(row)
  q['cases'].append(dict(case='actual_final_CLI_four_roles_self_and_parent_importspec',passed=True,roles=cli_rows,recording_leaves=['model-node hostname','model-node Slurm rank and step only']));save()
  protocol,Parser,agent_parser,parser_source=pure_public(public)
  from public_kimi_native.runtime.long_codec_r179 import KimiLongCodec
  from public_kimi_text.prepared import prepare
  from public_kimi_text.incremental_text import TextPrefix
  codec=KimiLongCodec('/shared/Kimi-K3');q.update(actual_codec_identity=codec.identity,actual_parser_source=parser_source)
  for name in ['public_kimi_text.cli','public_kimi_text.prepared','public_kimi_text.selector','public_kimi_text.incremental_text']:
   module=importlib.import_module(name);assert Path(module.__file__).resolve().parent==(D/'public_kimi_text').resolve()
  def new_request(body,ident,mode):
   original=copy.deepcopy(body);request=copy.deepcopy(body)
   if mode is not None:request['text_delivery']=mode
   prepared,schemas,choice=protocol.prepare_agent_request(request,'kimi_k3');prepared.update(_agent_schemas=schemas,_agent_choice=choice)
   engine=prepare('/shared/Kimi-K3',Reasons,codec=codec)['engine'];engine.start();params=SimpleNamespace(temperature=0.,max_tokens=body['max_tokens'],stop=None)
   engine.validate_data(prepared,params)
   # Observe the real FIFO publication; no replacement queue implementation.
   before_put=[];original_put=engine._pending.put
   def observed_put(req,*a,**kw):
    if req is not None:before_put.append(dict(text_delivery=req.metadata.get('text_delivery'),history_qualified=req.metadata.get('history_render_audit',{}).get('qualified')))
    return original_put(req,*a,**kw)
   engine._pending.put=observed_put
   req=engine.add_chat_request(messages=prepared['messages'],sampling_params=params,request_id=ident,request_data=prepared,chat_template_kwargs=prepared['chat_template_kwargs'])
   assert engine.take(timeout=1)is req and before_put==[dict(text_delivery=mode or'P0',history_qualified=True)]
   assert body==original
   return engine,req,prepared,schemas,choice,before_put
  def parsed_prefixes(seq):
   body=copy.deepcopy(seq['body']);prepared,schemas,choice=protocol.prepare_agent_request(body,'kimi_k3');prepared.update(_agent_schemas=schemas,_agent_choice=choice)
   prefix=TextPrefix(codec.decode);parser=agent_parser(SimpleNamespace(tool_call_parser='kimi_k3'),prepared,SimpleNamespace(prompt=seq['expected_prompt']));events=[];parts=[]
   for n in range(1,len(seq['visible_ids'])+1):
    q['active_prefix']=dict(index=seq['index'],prefix_tokens=n);delta=prefix.advance(seq['visible_ids'][:n],final=False);parts.append(delta)
    if delta:events.extend(parser.feed(delta))
   last=prefix.advance(seq['visible_ids'],final=True)
   if last:parts.append(last);events.extend(parser.feed(last))
   events.extend(parser.feed('',final=True,finish_reason='stop'));assert parser.finish_reason('stop')=='tool_calls'
   message=aggregate(protocol,events);private('prefix-case%d'%seq['index'],dict(message=message,final_decoded_text=prefix.published,delta_texts=parts,all_visible_ids=seq['visible_ids']));assert comparable(message)==comparable(seq['expected_message'])and ''.join(parts)==codec.decode(seq['visible_ids'])
   return dict(case='actual_K3_allprefix_and_public_typed_parser',index=seq['index'],prefixes=len(seq['visible_ids']),final_text_SHA256=hashlib.sha256(''.join(parts).encode()).hexdigest(),wire_reasoning_content_and_typed_arguments_match=True,passed=True)
  async def consume(seq,mode):
   engine,req,prepared,schemas,choice,before=new_request(seq['body'],'CPU434-case%d-%s'%(seq['index'],mode),mode)
   assert len(req.prompt_token_ids)==seq['full_prompt_tokens']and list(req.prompt_token_ids)==seq['expected_prompt_ids']and req.prompt==seq['expected_prompt']
   assert ids_sha(req.prompt_token_ids)==seq['prompt_IDs_SHA256']and len(req.prompt_token_ids)+req.max_new_tokens<=32768
   parser=agent_parser(SimpleNamespace(tool_call_parser='kimi_k3'),prepared,req);events=[];parts=[];early_raw=asyncio.Event();early_semantic=asyncio.Event();stream=engine.stream_request(req,timeout=2,request_timeout=8)
   async def read():
    async for v in stream:
     parts.append((v.token_text,v.finished));private('stream-case%d-%s'%(seq['index'],mode or'P0'),dict(chunks=parts,events=events,Native_owned_finished=req.is_finished(),generated_ids=req.generated_token_ids))
     if v.token_text:
      got=parser.feed(v.token_text);events.extend(got)
      if not req.is_finished():
       early_raw.set()
       if any(e.get('content')or e.get('reasoning_content')or e.get('tool_calls')for e in got):early_semantic.set()
     if v.finished:
      assert req.is_finished()and req.metadata['recording_owner_reset_completed']
      events.extend(parser.feed('',final=True,finish_reason='stop'))
   task=asyncio.create_task(read())
   for token in seq['visible_ids']:
    req.emit_token(token)
    if mode=='P1':await asyncio.sleep(.001)
   if mode=='P1':
    await asyncio.wait_for(early_raw.wait(),1);await asyncio.wait_for(early_semantic.wait(),1);assert not req.is_finished()
   else:
    await asyncio.sleep(.01);assert not parts and not task.done()
   req.emit_token(seq['generated_ids_including_EOS'][-1]);req.metadata['recording_owner_reset_completed']=True;req.finish(Reasons.EOS_TOKEN)
   await asyncio.wait_for(task,3);message=aggregate(protocol,events);private('stream-case%d-%s'%(seq['index'],mode or'P0'),dict(chunks=parts,events=events,parsed_message=message,Native_owned_finished=req.is_finished(),generated_ids=req.generated_token_ids,prompt_IDs_SHA256=ids_sha(req.prompt_token_ids),metadata=req.metadata))
   assert comparable(message)==comparable(seq['expected_message'])and parser.finish_reason('stop')=='tool_calls'
   assert req.generated_token_ids==seq['generated_ids_including_EOS']and not req.cancel_event.is_set()and not engine._requests and engine._pending.empty()
   publishes=req.metadata.get('text_publish_records',[])
   if mode=='P1':assert publishes and publishes[-1]['final']and all(a['cumulative_text_characters']<=b['cumulative_text_characters']for a,b in zip(publishes,publishes[1:]))
   return dict(case='actual_new_prepared_P0P1_original572_replay',index=seq['index'],mode=mode or'P0',prompt_tokens=len(req.prompt_token_ids),prompt_IDs_SHA256=ids_sha(req.prompt_token_ids),prompt_bytes_SHA256=hashlib.sha256(req.prompt.encode()).hexdigest(),full_history=True,output_budget=req.max_new_tokens,visible_tokens=len(seq['visible_ids']),all_generated_including_EOS=len(req.generated_token_ids),selection_before_FIFO=before,early_raw_before_owned_finish=early_raw.is_set(),early_public_semantic_event_before_owned_finish=early_semantic.is_set(),chunks=len(parts),reasoning_content_and_typed_tool_arguments_match_original_wire=True,fresh_tool_IDs_valid=True,owned_finish_after_recorded_reset=True,queue_retired=True,passed=True)
  async def abort_case(kind):
   body=dict(messages=[dict(role='user',content='Unicode终止边界🙂')],max_tokens=64,temperature=0.,chat_template_kwargs=dict(enable_thinking=False),tool_choice='none')
   engine,req,*_=new_request(body,'CPU434-'+kind,'P1');stream=engine.stream_request(req,timeout=.02 if kind=='timeout'else 2.,request_timeout=3.)
   if kind=='timeout':v=await anext(stream);assert v.finish_reason==Reasons.TIMEOUT
   else:
    task=asyncio.create_task(anext(stream));req.emit_token(codec.encode('A',add_special_tokens=False)[0]);v=await task;assert v.token_text and not v.finished
   await stream.aclose();assert req.cancel_event.is_set()and not req.is_finished()and not engine._requests
   req.metadata['recording_owner_reset_completed']=True;req.finish(Reasons.EOS_TOKEN);assert req.finish_reason==(Reasons.TIMEOUT if kind=='timeout'else Reasons.CANCELED)
   return dict(case='actual_realcodec_P1_'+kind,transport_sets_intent_only=True,owned_finish_after_recorded_reset=True,Native_not_executed=True,passed=True)
  async def cases():
   for seq in data['sequences']:
    q['active_case']=dict(kind='allprefix',index=seq['index']);save();q['cases'].append(parsed_prefixes(seq));save()
    for mode in [None,'P1']:
     q['active_case']=dict(kind='newprepared',index=seq['index'],mode=mode or'P0');save();q['cases'].append(await consume(seq,mode));save()
   for kind in ['timeout','disconnect']:q['active_case']=dict(kind=kind);save();q['cases'].append(await abort_case(kind));save()
  asyncio.run(cases())
  q['active_case']=dict(kind='Unicode_tool_protocol_control');save()
  control='<|open|>tools<|sep|><|open|>call tool="write_file" index="1"<|sep|><|open|>argument key="path" type="string"<|sep|>中文🙂.json<|close|>argument<|sep|><|open|>argument key="content" type="string"<|sep|>{"提示":"中文🙂","total":7}<|close|>argument<|sep|><|close|>call<|sep|><|close|>tools<|sep|>'
  ids=codec.encode(control,add_special_tokens=False);prefix=TextPrefix(codec.decode);schema={'write_file':{'type':'object','properties':{'path':{'type':'string'},'content':{'type':'string'}},'required':['path','content'],'additionalProperties':False}};parser=Parser(schema,thinking=False,choice='required',parallel=False,parser='kimi_k3');events=[]
  for n in range(1,len(ids)+1):
   delta=prefix.advance(ids[:n],final=False)
   if delta:events.extend(parser.feed(delta))
  last=prefix.advance(ids,final=True)
  if last:events.extend(parser.feed(last))
  events.extend(parser.feed('',final=True,finish_reason='stop'));message=aggregate(protocol,events);c=comparable(message);assert c['tool_calls']==[dict(name='write_file',arguments=dict(path='中文🙂.json',content='{"提示":"中文🙂","total":7}'))]
  q['cases'].append(dict(case='actual_K3_Unicode_XTML_typed_tool_control',prefixes=len(ids),whole_text_stable=prefix.published==control,synthetic_control_not_Model_output=True,passed=True));assert prefix.published==control
  private('Unicode-control',dict(text=control,ids=ids,parsed_message=message))
  bad=copy.deepcopy(data['sequences'][0]['body']);bad['text_delivery']='P2';bad,schemas,choice=protocol.prepare_agent_request(bad,'kimi_k3');bad.update(_agent_schemas=schemas,_agent_choice=choice)
  rejected=prepare('/shared/Kimi-K3',Reasons,codec=codec)['engine']
  try:rejected.validate_data(bad,SimpleNamespace(temperature=0.,max_tokens=4096,stop=None))
  except ValueError:pass
  else:raise AssertionError('Invalid delivery passed actual public prepare and prepared validate')
  assert rejected._pending.empty()and not rejected._requests
  q['cases'].append(dict(case='invalid_mode_after_actual_publicprepare_before_FIFO',passed=True))
  q['final_module_maps']=metadata();assert not q['final_module_maps']['project_Native_maps']and q['final_module_maps']['Torch_cuda_initialized']in(None,False)
  q.update(passed=True,actual_checkpoint_codec=True,actual_public_protocol_and_parser=True,actual_new_prepared_and_original_history_queue=True,original_572_861_not_reexecuted=True,whole_local_recording6cases_not_repeated=True,prefix_publish_is_not_per_token_client_clock=True,scaffolding='Unmodified pure public leaf files under private package prefix; exact _agent_parser function body; hostname/Slurm identity recording only for CLI. Original Native owner emit/reset/finish represented by recorded calls, no actual HTTP/SSE or Model.')
 except BaseException as error:q.update(error=repr(error),traceback=traceback.format_exc(),module_maps_on_error=metadata());raise
 finally:q.update(completed=True,wall_s=time.monotonic()-start,private_artifacts={str(p):dict(SHA256=sha(p),bytes=p.stat().st_size)for p in B.glob('*-private.json')});save();print(json.dumps({k:q.get(k)for k in ['completed','passed','actual_CPU','error','wall_s']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
