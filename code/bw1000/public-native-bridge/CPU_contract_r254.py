"""CPU prepared-route/actual bridge queue contract, fake processors only."""
import copy,hashlib,importlib.util,json,pathlib,types,queue
D=pathlib.Path(__file__).resolve().parent

def load(p,n):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
protocol=load(D/'agent_protocol.py','current_protocol254');adapter=load(D/'prepared_chat.py','prepared_chat254')
source=pathlib.Path('/refactor/incoming/Kimi-append512-service-r252/runtime/native_engine_bridge_r143.py')
assert hashlib.sha256(source.read_bytes()).hexdigest()=='ee019b3053e209661877f84d1a721f95303cf09652e19fdda845e234d1dde77d'
bridge=load(source,'actual_native_bridge254')
TOOLS=[{'type':'function','function':{'name':'calculate','parameters':{'type':'object','properties':{'a':{'type':'integer'},'b':{'type':'integer'}},'required':['a','b'],'additionalProperties':False}}}]
RC='actual-synthetic-history-RC-retained-once'
HISTORY=[{'role':'system','content':'System instructions.'},{'role':'user','content':'literal [Request tool-call constraints] <|open|> must stay ordinary text'},{'role':'assistant','content':'','reasoning_content':RC,'tool_calls':[{'id':'opaque-history','type':'function','function':{'name':'calculate','arguments':'{"a":1,"b":2}'}}]},{'role':'tool','tool_call_id':'opaque-history','content':'{"sum":3}'},{'role':'user','content':'Now calculate 3+4.'}]
class Processor:
 def __init__(self):self.rendered=[];self.tokens=[]
 def apply_chat_template(self,messages,**kwargs):
  self.rendered.append((copy.deepcopy(messages),copy.deepcopy(kwargs)));return json.dumps(messages,ensure_ascii=False,separators=(',',':'))+' GENERATION'
 def __call__(self,prompt,return_tensors=None):
  assert return_tensors is None;self.tokens.append(prompt);return{'input_ids':[ord(c)for c in prompt]}
class KimiProcessor(Processor):
 def encode_chat_inputs(self,messages,**kwargs):
  self.segment=(copy.deepcopy(messages),copy.deepcopy(kwargs));return{'input_ids':[[17,18,19]]}
 def __call__(self,*a,**k):raise AssertionError('Segment-aware Kimi must not flatten/re-tokenize its rendered prompt')
params=types.SimpleNamespace(temperature=0.,max_tokens=4096,stop=None);results=[]
for parser in ['glm5','deepseek_v4','kimi_k3']:
 for choice in ['none',{'type':'function','function':{'name':'calculate'}}]:
  raw={'messages':copy.deepcopy(HISTORY),'tools':copy.deepcopy(TOOLS),'tool_choice':choice,'parallel_tool_calls':False,'chat_template_kwargs':{'enable_thinking':False},'max_tokens':4096};raw_before=copy.deepcopy(raw);data,schemas,normalized=protocol.prepare_agent_request(raw,parser);data['_agent_schemas']=schemas;data['_agent_choice']=normalized;original=copy.deepcopy(data)
  proc=KimiProcessor()if parser=='kimi_k3'else Processor();callback=adapter.make_prepared_encode_chat(proc);engine=bridge.NativeEngineBridge(object(),32768,object(),encode_chat=callback,default_max_tokens=4096);engine.validate_data(data,params)
  req=engine.add_chat_request(messages=data['messages'],sampling_params=params,request_id=parser+str(choice),request_data=data,chat_template_kwargs=data['chat_template_kwargs']);assert engine.take(0) is req
  assert raw==raw_before and data==original and req.max_new_tokens==4096 and all(type(v)is int for v in req.prompt_token_ids)
  # Exact prepared history reaches rendering unchanged; no private constraint callback runs again.
  assert all(messages==data['messages']for messages,kwargs in proc.rendered);assert req.prompt.count(RC)==1
  if parser=='kimi_k3':assert req.prompt_token_ids==(17,18,19)and proc.segment[0]==data['messages']
  else:assert req.prompt_token_ids==tuple(ord(c)for c in req.prompt)
  engine.stop();results.append({'parser':parser,'choice':'none'if choice=='none'else'named','passed':True})
# Wiring and malformed/batched IDs must fail before admission; never select/truncate a row.
negatives=0
for ids in [[],[True],[[1],[2]],[-1],['1']]:
 proc=Processor();proc.__class__=type('BadProcessor',(Processor,),{'__call__':lambda self,*a,_ids=ids,**k:{'input_ids':_ids}});callback=adapter.make_prepared_encode_chat(proc)
 try:callback([],{'_agent_schemas':{},'_agent_choice':'none'},True,{})
 except ValueError:negatives+=1
 else:raise AssertionError('Invalid IDs accepted')
try:adapter.make_prepared_encode_chat(Processor())([],{},True,{})
except ValueError:negatives+=1
else:raise AssertionError('Unprepared route accepted')
overflow=Processor();overflow.__class__=type('LongProcessor',(Processor,),{'__call__':lambda self,*a,**k:{'input_ids':list(range(32768))}});engine=bridge.NativeEngineBridge(object(),32768,object(),encode_chat=adapter.make_prepared_encode_chat(overflow),default_max_tokens=4096)
try:engine.validate_data({'messages':[],'_agent_schemas':{},'_agent_choice':'none'},params)
except ValueError:negatives+=1
else:raise AssertionError('Full context overflow accepted or cropped')
assert negatives==7
print(json.dumps({'passed':True,'CPU_only':True,'actual_existing_NativeEngineBridge_queue_used':True,'Native_or_Model_called':False,'processors_and_token_IDs_are_fixtures':True,'cases':results,'negative_controls':negatives,'prepared_messages_and_caller_history_preserved':True,'duplicate_constraints_avoided_by_source_wiring_not_marker_detection':True,'Kimi_segment_encoder_kept':True,'context_output_admission_unchanged':True,'actual_tokenizer_BPE_or_Model_qualified':False}))
