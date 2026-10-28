"""One official encoding/context contract for CPU fixtures and live requests."""
import copy,hashlib,json,pathlib,importlib.util
D=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('official_dsv4_r173',D/'encoding/encoding_dsv4.py');encoder=importlib.util.module_from_spec(spec);spec.loader.exec_module(encoder)
def request_constraints(data,tools):
 choice=data.get('_agent_choice',data.get('tool_choice','auto'))
 if choice is None:choice='auto'
 if isinstance(choice,dict):choice=choice['function']['name']
 parallel=data.get('parallel_tool_calls',True)
 if type(parallel)is not bool:raise ValueError('parallel_tool_calls must be boolean')
 if choice=='none':return '[Request tool-call constraints]\nTools are disabled for this assistant turn. Do not invoke any tool. Respond without a tool call.'
 if not tools:return ''
 rules=[]
 if not parallel:rules.append('In this assistant turn, issue at most one real tool call. After issuing that call, end this assistant turn and wait for its tool result before deciding or calling any next tool. Do not combine later steps or additional invocations in this turn.')
 if choice=='required':rules.append('This request requires at least one real tool call in this assistant turn. Do not substitute a text answer for the required tool call.')
 elif choice!='auto':
  available=[t['function']['name']for t in tools]
  if choice not in available:raise ValueError('Requested tool is unavailable after protocol filtering')
  rules.append('This request requires a real call to the function '+json.dumps(choice,ensure_ascii=False)+' in this assistant turn, following its provided schema. Do not substitute a text answer for the required call.')
 return '[Request tool-call constraints]\n'+'\n'.join(rules)if rules else''
def prepare(messages,data,add_generation_prompt,kwargs,tokenizer,context=None,limit=None):
 caller=messages;original=copy.deepcopy(messages);messages=copy.deepcopy(messages);tools=kwargs.get('tools')
 if tools:
  if not messages or messages[0].get('role')!='system':messages.insert(0,{'role':'system','content':''})
  messages[0]['tools']=tools
 for message in messages:
  if isinstance(message.get('content'),list):message['content']=''.join(p['text']for p in message['content'])
  for call in message.get('tool_calls')or[]:
   if isinstance(call['function'].get('arguments'),dict):call['function']['arguments']=json.dumps(call['function']['arguments'],ensure_ascii=False,allow_nan=False)
 constraints=request_constraints(data,tools)
 if constraints:
  if not messages or messages[0].get('role')!='system':messages.insert(0,{'role':'system','content':''})
  content=messages[0].get('content','')
  if not isinstance(content,str):raise ValueError('System content must be text after normal preparation')
  messages[0]['content']=content+'\n\n'+constraints
 mode=kwargs.get('thinking_mode','thinking'if kwargs.get('enable_thinking',True)else'chat')
 prompt=encoder.encode_messages(messages,thinking_mode=mode,reasoning_effort=kwargs.get('reasoning_effort'))
 if not add_generation_prompt:prompt=prompt.removesuffix('<｜Assistant｜>'+('<think>'if mode=='thinking'else'</think>'))
 ids=list(tokenizer.encode(prompt,add_special_tokens=False));assert ids and all(type(x)is int and x>=0 for x in ids)
 assert caller==original # Encoding copies; caller history is never changed.
 if context is not None:
  if type(limit)is not int or limit<1 or len(ids)+limit>context:raise ValueError('Complete prompt plus output budget exceeds actual Native context; no cropping')
 return prompt,ids
