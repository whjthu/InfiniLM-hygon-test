"""Real Deep codec/public prepared FIFO for fixed444256 questions; zero Native."""
from pathlib import Path
from enum import Enum
import ast,asyncio,copy,hashlib,importlib,importlib.util,json,os,sys,time,traceback,types,unicodedata
D=Path(__file__).resolve().parent;OUT=Path('/refactor/candidates/Deep-broad-fewshot-fullpre-quality-r450-CPU-v1')
sha=lambda b:hashlib.sha256(b).hexdigest()
compact=lambda q:json.dumps(q,ensure_ascii=False,separators=(',',':')).encode()

class Reasons(str,Enum):
 EOS_TOKEN='eos_token';LENGTH='length';TIMEOUT='timeout';CANCELED='canceled';ERROR='error';STOP_STRING='stop_string'

def file_module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

def verify_tree(pin):
 root=Path(pin['path']);raw=(root/'SOURCE_MANIFEST.json').read_bytes();assert sha(raw)==pin['SHA256']
 for name,h in json.loads(raw)['files_SHA256'].items():
  p=root/name;assert p.resolve().is_relative_to(root.resolve())and sha(p.read_bytes())==h,name

def observed():
 maps=Path('/proc/self/maps').read_text();matched=[line for line in maps.splitlines()if any(x in line for x in ['libDeep_','libGLM_','libKimi_','_infinilm.cpython','libinfinilm_runtime','libinfiniops','libinfiniccl','libinfinirt'])]
 t=sys.modules.get('torch');cuda=getattr(t,'cuda',None)if t else None
 return dict(project_or_SDK_maps=matched,Torch_imported=t is not None,Torch_CUDA_initialized=getattr(cuda,'_initialized',False)if cuda else False,hidden={k:os.environ.get(k)for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES']})

def questions(value):
 if isinstance(value,list):return [s for q in value for s in questions(q)]
 if not isinstance(value,dict):return []
 return ([value['question']]if isinstance(value.get('question'),str)else[])+[s for k,q in value.items()if k not in ['source','metadata','sources']for s in questions(q)]

def question_key(s):return sha(' '.join(unicodedata.normalize('NFC',s).split()).encode())

def public_consumer(public,Parser,SamplingParams,engine):
 path=public/'infinilm/server/inference_server.py';original=next(n for n in ast.parse(path.read_bytes()).body if isinstance(n,ast.ClassDef)and n.name=='InferenceServer')
 actual=copy.deepcopy(original);actual.bases=[ast.Name(id='object',ctx=ast.Load())];actual.keywords=[];actual.body=[n for n in actual.body if getattr(n,'name',None)in ['_build_sampling_params','_agent_parser']]
 ns=dict(__name__='Deep450_actual_public_sampling',SamplingParams=SamplingParams,ToolCallParser=Parser)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[actual],type_ignores=[])),str(path),'exec'),ns)
 server=ns['InferenceServer']();server.max_tokens=4096;server.temperature=0.;server.top_p=1.;server.top_k=1;server.ignore_eos=False;server.tool_call_parser='deepseek_v4'
 from public_deep_model.prepared_factory import bind_public_sampling
 bind_public_sampling(server,engine);return server

def body_for(original,case):
 body=copy.deepcopy(original);before=compact(body['messages'])
 body.update(model='hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8',stream=True,text_delivery='P1',stream_options={'include_usage':True},user='Deep-r450-'+case['id'],chat_template_kwargs={'enable_thinking':False})
 assert compact(body['messages'])==before and body['max_tokens']==4096 and body['temperature']==0 and body['tool_choice']=='none'
 assert case['question']in body['messages'][1]['content']and case['shot_count']==(8 if case['suite']=='gsm8k'else 5)
 return body

def main():
 assert not OUT.exists(),'Immutable CPU output exists; no replay'
 OUT.mkdir();start=time.monotonic();report=OUT/'encoding.json'
 q=dict(completed=False,passed=False,actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],GPU=0,Native_dlopen=False,weights_loaded=False,Model_READY=False,HTTP=0,questions=256,logical_requests=512,actual_encodings=0,cases=[],recording_leaves=['FinishReason enum; administrative empty CPU FIFO retirement'],old_suites_replayed=False,actual_Deep_codec=False,actual_public_protocol=False,actual_prepared_FIFO=False)
 def save():
  # Each private case is already immutable. Live metadata is O(1); complete
  # aggregate is written once at terminal, not all history on every encoding.
  snapshot=q if q['completed']else dict(q,cases=[],partial_cases=q['actual_encodings'])
  t=report.with_suffix('.tmp');t.write_bytes((json.dumps(snapshot,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode());t.replace(report)
 save()
 try:
  assert all(os.environ.get(k)=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
  raw=(D/'SOURCE_MANIFEST.json').read_bytes();assert sha(raw)==os.environ['DEEP450_SOURCE_SHA256'];m=json.loads(raw)
  for name,h in m['files_SHA256'].items():assert sha((D/name).read_bytes())==h,name
  pins=json.loads((D/'PINS.json').read_bytes());plan=json.loads((D/'PLAN.json').read_bytes());assert plan['questions']==256 and plan['logical_requests']==512 and not plan['GPU_GO']
  for key in ['provider','public']:verify_tree(pins[key])
  helper=Path(pins['processor_CPU_helper']['path']);assert sha(helper.read_bytes())==pins['processor_CPU_helper']['SHA256']
  model=Path(pins['processor_path'])
  for name,h in pins['tokenizer_assets_SHA256'].items():assert sha((model/name).read_bytes())==h,name
  q.update(source_manifest_SHA256=sha(raw),plan_SHA256=sha((D/'PLAN.json').read_bytes()),source_observation=[dict(stage='initial',**observed())])
  assert not observed()['project_or_SDK_maps']
  def guard(event,args):
   if event=='open'and isinstance(args[0],(str,bytes,os.PathLike))and os.fsdecode(args[0]).lower().endswith(('.safetensors','.bin','.pth','.pt')):raise RuntimeError('CPU encoding forbids weight payload')
   if event=='ctypes.dlopen'and args and args[0]and any(x in str(args[0])for x in ['libDeep_','libGLM_','libKimi_','_infinilm','libinfiniops','libinfinirt','libinfiniccl']):raise RuntimeError('Project/SDK dlopen forbidden')
  sys.addaudithook(guard)
  inputs={}
  for name,pin in pins['inputs'].items():
   b=Path(pin['path']).read_bytes();assert sha(b)==pin['SHA256'];inputs[name]=json.loads(b)
  cases=inputs['cases']['cases'];old=inputs['requests']['requests'];metadata=inputs['metadata']
  assert len(cases)==len(old)==256 and [c['id']for c in cases]==[r['case_id']for r in old]and sum(c['suite']=='gsm8k'for c in cases)==128 and sum(c['suite']=='mmlu'for c in cases)==128
  assert metadata['files_SHA256']['cases.json']==pins['inputs']['cases']['SHA256']and metadata['files_SHA256']['requests-and-pairs.json']==pins['inputs']['requests']['SHA256']
  overlaps=[]
  for pin in pins['known_prior_question_files']:
   b=Path(pin['path']).read_bytes();assert sha(b)==pin['SHA256'];known={question_key(s)for s in questions(json.loads(b))};matches=[c['id']for c in cases if question_key(c['question'])in known]
   overlaps.append(dict(SHA256=pin['SHA256'],aliases=pin['aliases'],known_unique_questions=len(known),matching_case_ids=matches,overlap=len(matches)))
  q.update(known_prior_overlap=overlaps,overlap_scope='Exact normalized original444 prior inventory, deduplicated by byteSHA; no universal unseen claim, no reselection by Deep answers/tokens')
  sys.path[:0]=[str(D),pins['qualified427']['path'],'/old/offline-deps']
  helper_module=file_module('qualified_430_actual_processor_CPU',helper);public=Path(pins['public']['path'])/'public-python'
  Parser,prepare,history=helper_module.pure_public(public);processor,methods=helper_module.actual_processor(public,model,history)
  sampling=file_module('Deep450_actual_sampling',public/'infinilm/llm/sampling_params.py')
  assert sha(Path(sampling.__file__).read_bytes())==pins['actual_sampling_SHA256']
  assert sha(Path(sys.modules['infinilm.server.agent_protocol'].__file__).read_bytes())==pins['actual_protocol_SHA256']
  from public_deep_model.prepared_factory import prepare_engine
  prepared=prepare_engine('Deep',pins['model_path'],processor=processor,finish_reasons=Reasons,processor_path=str(model));engine=prepared['engine'];engine.start()
  server=public_consumer(public,Parser,sampling.SamplingParams,engine)
  q.update(actual_Deep_codec=True,actual_public_protocol=True,actual_prepared_FIFO=True,processor_method_sources=methods,EOS_ids=prepared['identity']['eos_token_ids'],prepared_identity=prepared['identity'],source_observation=q['source_observation']+[dict(stage='actual_processor_and_factory',**observed())])
  assert q['EOS_ids']==[1]
  bodies=[];logical=[]
  for ordinal,(case,old_req)in enumerate(zip(cases,old)):
   q['active_case']=dict(ordinal=ordinal,case_id=case['id']);save()
   assert old_req['case_id']==case['id']and sha(compact(old_req['body']))==old_req['compact_body_SHA256']
   body=body_for(old_req['body'],case);original=copy.deepcopy(body)
   data,schemas,choice=prepare(copy.deepcopy(body),parser='deepseek_v4');data.update(_agent_schemas=schemas,_agent_choice=choice)
   assert choice=='none'and data['chat_template_kwargs']['enable_thinking']is False and data['stream']is True and data['text_delivery']=='P1'
   params=server._build_sampling_params(data)
   req=engine.add_chat_request(messages=data['messages'],sampling_params=params,request_id='Deep450-CPU-'+str(ordinal),request_data=data,add_generation_prompt=bool(data.get('add_generation_prompt',True)),chat_template_kwargs=data.get('chat_template_kwargs')or{})
   assert engine.take(timeout=1)is req and req.max_new_tokens==4096
   parser=server._agent_parser(data,req);assert parser is not None
   ids=list(req.prompt_token_ids);h=req.metadata['history_render_audit'];id_hash=sha(compact(ids));assert ids and all(type(v)is int and 0<=v<129280 for v in ids)
   assert h['qualified']and h['prequeue_bound']and not h['markers_sent_to_Native']and not h['pending_thinking_prefix_open']and h['pending_chat_prefix_closed']and h['Native_prompt_token_ids_SHA256']==id_hash and req.metadata['text_delivery']=='P1'
   assert h['same_prepared_encoder_reproduces_prompt_and_IDs']and h['all_content_contribution_slices_equal']and h['content_one_contribution_each_and_ordered']and h['output_budget']==4096
   text,again=engine.encode_chat(data['messages'],data,True,data['chat_template_kwargs']);assert text==req.prompt and list(again)==ids and body==original and len(ids)+4096<=32768
   private=OUT/('case-%03d-private.json'%ordinal);private.write_bytes((json.dumps(dict(case_id=case['id'],body=body,prompt=req.prompt,prompt_token_ids=ids),ensure_ascii=False)+'\n').encode())
   req.finish(Reasons.CANCELED)
   async def retire():
    async for token in engine.stream_request(req,timeout=1,request_timeout=2):pass
   asyncio.run(retire());assert req.request_id not in engine._requests and not req.generated_token_ids
   chunks=[min(1024,len(ids)-s)for s in range(0,len(ids),1024)]
   row=dict(case_id=case['id'],suite=case['suite'],source_index=case['source_index'],shot_count=case['shot_count'],selected_ordinal=ordinal,source_case_SHA256=sha(compact(case)),original_fewshot_body_SHA256=old_req['compact_body_SHA256'],request_body_SHA256=sha(compact(body)),prompt_SHA256=sha(req.prompt.encode()),prompt_IDs_SHA256=id_hash,prompt_tokens=len(ids),output_budget=4096,input_plus_output=len(ids)+4096,prefill_chunk=1024,prefill_chunks=chunks,T1_prefill_calls=sum(v==1 for v in chunks),actual_thinking=False,text_delivery='P1',history_render_audit=h,CPU_queue_retired=True,Native_generation_or_reset=False,private_path=str(private),private_SHA256=sha(private.read_bytes()))
   q['cases'].append(row);bodies.append(dict(case_id=case['id'],body=body,request_body_SHA256=row['request_body_SHA256']))
   q['actual_encodings']=len(q['cases']);save()
  from paired_plan import make
  logical=make(cases,q['cases'])
  assert len(q['cases'])==256 and len(logical)==512 and not engine._requests;engine.stop()
  requests=OUT/'requests-private.json';requests.write_bytes((json.dumps(dict(requests=bodies,logical_plan=logical,questions=256,logical_requests=512),ensure_ascii=False,indent=2)+'\n').encode())
  from check_policy_CPU import run as changed_policy
  q['changed_finite_plan_policy_consumer']=changed_policy(cases,q['cases'],logical)
  q.update(passed=True,active_case=None,matching_original_fewshot_messages=256,no_answer_or_length_reselection=True,all_input_plus4096_within32768=True,prompt_tokens_min=min(c['prompt_tokens']for c in q['cases']),prompt_tokens_max=max(c['prompt_tokens']for c in q['cases']),requests_file=str(requests),requests_file_SHA256=sha(requests.read_bytes()),quality_result_created=False,actual_loaded_modules={n:dict(path=str(v.__file__),SHA256=sha(Path(v.__file__).read_bytes()))for n,v in list(sys.modules.items())if n.startswith(('public_deep_model','infinilm.server.agent','infinilm.processors.history'))and getattr(v,'__file__',None)})
 except BaseException as error:q.update(passed=False,error=repr(error),traceback=traceback.format_exc())
 finally:
  observation=observed();q.update(completed=True,elapsed_s=time.monotonic()-start,final_modules=observation)
  if observation['project_or_SDK_maps']or observation['Torch_CUDA_initialized']:q.update(passed=False,unexpected_device_mapping=True)
  save();print(json.dumps(dict(path=str(report),SHA256=sha(report.read_bytes()),passed=q['passed'],actual_CPU=q['actual_CPU'],actual_encodings=q['actual_encodings'],error=q.get('error'),elapsed_s=q['elapsed_s'])),flush=True)
 return 0 if q['passed']else 1
if __name__=='__main__':raise SystemExit(main())
