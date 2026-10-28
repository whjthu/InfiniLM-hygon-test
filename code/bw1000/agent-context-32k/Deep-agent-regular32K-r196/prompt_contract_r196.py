"""Same live prepare entry on exact saved requests, no Model or Native import."""
import pathlib,json,hashlib,copy,importlib.util,sys
from tokenizer_cpu_r173b import OfflineTokenizer
import chat_prepare_r173 as current
D=pathlib.Path(__file__).resolve().parent;R=pathlib.Path('/refactor');OLD=R/'incoming/Deep-agent-regular32K-r182';P=R/'results/Deep-agent-regular32K-r182-910374.1203.json';ROOT=P.with_suffix('');sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def run(model):
 assert sha(P)=='341efba6013186615cc139b8320841fa69b17c08f7f0b01a5895496eafc52571';assert sha(OLD/'manifest-r182.json')=='2fbc2b355d0c6342f33c163da3542077d1a7160ca9268c3f2cd29092e5534f22'
 manifest=json.loads((OLD/'manifest-r182.json').read_bytes())
 for rel in ['chat_prepare_r173.py','encoding/encoding_dsv4.py','overlay/python/infinilm/server/agent_protocol.py']:assert sha(OLD/rel)==manifest['source_SHA256'][rel]
 assert sha(D/'overlay/python/infinilm/server/agent_protocol.py')==sha(OLD/'overlay/python/infinilm/server/agent_protocol.py')
 protocol=load('frozen_protocol_r196_contract',D/'overlay/python/infinilm/server/agent_protocol.py');previous=load('old_prepare_r182_contract',OLD/'chat_prepare_r173.py');tokenizer=OfflineTokenizer(model);assert sha(pathlib.Path(model)/'tokenizer.json')=='8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf'
 captured=[];encode=current.encoder.encode_messages
 def capture(messages,**kwargs):captured.append(copy.deepcopy(messages));return encode(messages,**kwargs)
 current.encoder.encode_messages=capture;oldcaptured=[];oldencode=previous.encoder.encode_messages
 def oldcapture(messages,**kwargs):oldcaptured.append(copy.deepcopy(messages));return oldencode(messages,**kwargs)
 previous.encoder.encode_messages=oldcapture
 def prepare(module,body):
  data,schemas,choice=protocol.prepare_agent_request(body,'deepseek_v4');data['_agent_choice']=choice;data['_agent_schemas']=schemas;messages=copy.deepcopy(data['messages']);before=copy.deepcopy(messages);kw=copy.deepcopy(data['chat_template_kwargs']);limit=data.get('max_tokens',4096);assert type(limit)is int
  prompt,ids=module.prepare(messages,data,bool(data.get('add_generation_prompt',True)),kw,tokenizer,32768,limit);assert messages==before
  return data,schemas,choice,prompt,ids
 result=[];all_files=[];history_fields=0;reasoning_fields=0
 for directory in [ROOT/'regular'/x for x in ['HTTP','SSE','thinking','negative']]+[ROOT/'long-output']:
  if directory.is_dir():all_files.extend(sorted(directory.glob('request-*.request.json')))
 try:
  for p in all_files:
   body=json.loads(p.read_bytes());captured.clear();oldcaptured.clear()
   try:old,schemas,choice,oldprompt,oldids=prepare(previous,body)
   except (ValueError,TypeError)as exc:
    try:prepare(current,body)
    except type(exc):result.append({'request_SHA256':sha(p),'intentional_preparation_rejection_preserved':True});continue
    raise AssertionError('Previously rejected request became accepted')
   data,ss,cc,prompt,ids=prepare(current,body);assert ss==schemas and cc==choice and len(captured)==1 and len(oldcaptured)==1
   before=oldcaptured[0];after=captured[0];rules=current.request_constraints(data,data['chat_template_kwargs'].get('tools'));expected=copy.deepcopy(before)
   if rules:
    if not expected or expected[0].get('role')!='system':expected.insert(0,{'role':'system','content':''})
    expected[0]['content']=expected[0].get('content','')+'\n\n'+rules
   assert after==expected,'Only appended API constraint may differ in prepared messages'
   history_fields+=len(body['messages']);reasoning=[m.get('reasoning_content')for m in body['messages']if m.get('reasoning_content')];reasoning_fields+=len(reasoning)
   assert all(x in oldprompt and x in prompt for x in reasoning),'Complete existing thinking must remain in official encoding'
   tools=data['chat_template_kwargs'].get('tools');parallel=data.get('parallel_tool_calls',True);row={'request_SHA256':sha(p),'old_prompt_tokens':len(oldids),'new_prompt_tokens':len(ids),'output_budget':body.get('max_tokens',4096),'old_prompt_SHA256':hashlib.sha256(oldprompt.encode()).hexdigest(),'new_prompt_SHA256':hashlib.sha256(prompt.encode()).hexdigest(),'history_exact_except_appended_system_constraint':True,'reasoning_fields_retained':len(reasoning),'schema_filter_unchanged':ss==schemas,'raw_budget_unchanged':data.get('max_tokens',4096)==body.get('max_tokens',4096)}
   if tools and choice!='none':
    alt=copy.deepcopy(body);alt['parallel_tool_calls']=not parallel;_,_,_,other,_=prepare(current,alt);assert other!=prompt;row['parallel_flag_changes_prompt']=True
    if choice not in ['auto','required','none']:assert len(schemas)==1 and list(schemas)==[choice];row['named_schema_still_filtered']=True
    # auto+parallel=true has no new rule, so may still emit zero calls.
    auto=copy.deepcopy(body);auto['tool_choice']='auto';auto['parallel_tool_calls']=True;_,_,_,ap,_=prepare(current,auto);_,_,_,op,_=prepare(previous,auto);assert ap==op;row['auto_parallel_true_unchanged']=True
    required=copy.deepcopy(body);required['tool_choice']='required';required['parallel_tool_calls']=parallel;_,_,_,rp,_=prepare(current,required);auto['parallel_tool_calls']=parallel;_,_,_,aa,_=prepare(current,auto);assert rp!=aa;row['required_changes_prompt']=True
   if choice=='none':assert not schemas and rules and 'Do not invoke any tool.'in prompt;row['none_has_no_schema_and_no_call_constraint']=True
   assert len(ids)+row['output_budget']<=32768;result.append(row)
 finally:current.encoder.encode_messages=encode;previous.encoder.encode_messages=oldencode
 assert len(result)==12 and reasoning_fields>0 and all(x not in sys.modules for x in ['torch','transformers','_infinilm'])
 return {'passed':True,'scope':'Actual saved-request official-Rust prompt contract; no Model generation, HTTP or Native qualification','old_final1203_SHA256':sha(P),'source_files':len(result),'historical_messages_checked':history_fields,'existing_reasoning_fields_checked':reasoning_fields,'requests':result,'Torch_imported':False,'Native_loaded':False}
