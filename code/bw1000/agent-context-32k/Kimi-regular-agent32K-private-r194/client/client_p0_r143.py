"""Real HTTP/SSE P0 witness. Model tool calls are never constructed or replaced.

Only literal sandbox paths input.json/result.json and integer addition execute.
Independent cases continue after failure so a protocol error cannot hide recovery.
"""
import argparse, datetime, hashlib, json, pathlib, secrets, time
import urllib.request, urllib.error

def sha(b): return hashlib.sha256(b).hexdigest()
def compact(v): return json.dumps(v, ensure_ascii=False, separators=(',',':'))
TOOLS=[
 {'type':'function','function':{'name':'read_file','description':'Read file.','parameters':{'type':'object','properties':{'path':{'type':'string'}},'required':['path'],'additionalProperties':False}}},
 {'type':'function','function':{'name':'calculate','description':'Add integers.','parameters':{'type':'object','properties':{'a':{'type':'integer'},'b':{'type':'integer'}},'required':['a','b'],'additionalProperties':False}}},
 {'type':'function','function':{'name':'write_file','description':'Write file.','parameters':{'type':'object','properties':{'path':{'type':'string'},'content':{'type':'string'}},'required':['path','content'],'additionalProperties':False}}},
]

def sse_message(raw):
 message={'role':'assistant','content':'','reasoning_content':''};calls={};finish=None;usage=None;done=False
 for line in raw.decode().splitlines():
  if not line or line.startswith(':'):continue
  if not line.startswith('data: '):raise AssertionError('Invalid SSE frame')
  value=line[6:]
  if value=='[DONE]':done=True;break
  q=json.loads(value)
  if 'error'in q:raise RuntimeError(compact(q))
  if q.get('usage')is not None:usage=q['usage']
  for c in q.get('choices',[]):
   assert c.get('index',0)==0;delta=c.get('delta',{})
   for key in ['content','reasoning_content']:
    if delta.get(key):message[key]+=delta[key]
   for part in delta.get('tool_calls',[]):
    entry=calls.setdefault(part.get('index',0),{'id':'','type':'function','function':{'name':'','arguments':''}})
    if part.get('id'):entry['id']+=part['id']
    if part.get('type'):assert part['type']=='function'
    for key in ['name','arguments']:
     if part.get('function',{}).get(key):entry['function'][key]+=part['function'][key]
   if c.get('finish_reason')is not None:finish=c['finish_reason']
 assert done and finish is not None,'Incomplete SSE; no tools may execute'
 if calls:message['tool_calls']=[calls[k]for k in sorted(calls)]
 if not message['reasoning_content']:message.pop('reasoning_content')
 return message,finish,usage

class WireClient:
 def __init__(self,url,model,path,max_tokens=96,timeout=1000):
  self.url=url.rstrip('/')+'/chat/completions';self.model=model;self.path=path;self.max_tokens=max_tokens;self.timeout=timeout;self.serial=0
 def request(self,messages,choice='none',stream=False,tools=None,max_tokens=None,thinking=False,parallel=False):
  self.serial+=1;stem=self.path/('request-%03d'%self.serial)
  body={'model':self.model,'messages':messages,'temperature':0,'max_tokens':self.max_tokens if max_tokens is None else max_tokens,'stream':stream,'tool_choice':choice,'parallel_tool_calls':parallel,'chat_template_kwargs':{'enable_thinking':thinking}}
  if hasattr(self,'session')or hasattr(self,'label'):body['user']=str(getattr(self,'session',getattr(self,'label','unlabelled')))  # journal only, excluded from official prompt
  if tools is not None:body['tools']=tools
  if stream:body['stream_options']={'include_usage':True}
  stem.with_suffix('.request.json').write_bytes((compact(body)+'\n').encode());started=time.monotonic()
  req=urllib.request.Request(self.url,data=compact(body).encode(),headers={'Content-Type':'application/json'})
  try:
   with urllib.request.urlopen(req,timeout=self.timeout)as response:status=response.status;raw=response.read();headers=dict(response.headers)
  except urllib.error.HTTPError as e:status=e.code;raw=e.read();headers=dict(e.headers)
  stem.with_suffix('.wire').write_bytes(raw)
  ev={'request':stem.with_suffix('.request.json').name,'wire':stem.with_suffix('.wire').name,'wire_SHA256':sha(raw),'HTTP_status':status,'elapsed_s':time.monotonic()-started,'stream':stream}
  if status!=200:return {'error':json.loads(raw)if raw.startswith(b'{')else raw.decode(errors='replace')},None,None,ev
  if stream:
   content_type=next((value for key,value in headers.items()if key.lower()=='content-type'),'')
   assert 'text/event-stream'in content_type.lower(),'Unexpected SSE Content-Type: '+content_type
   message,finish,usage=sse_message(raw)
  else:
   q=json.loads(raw);assert 'error'not in q;message=q['choices'][0]['message'];finish=q['choices'][0]['finish_reason'];usage=q.get('usage')
  assert isinstance(usage,dict),'Missing actual token usage'
  assert all(type(usage[k])is int and usage[k]>=0 for k in ['prompt_tokens','completion_tokens','total_tokens'])
  assert usage['total_tokens']==usage['prompt_tokens']+usage['completion_tokens']
  ev['finish_reason']=finish;ev['usage']=usage
  stem.with_suffix('.decoded.json').write_bytes((compact({'message':message,'finish_reason':finish,'usage':usage})+'\n').encode())
  return message,finish,usage,ev

def execute(call,sandbox,actions):
 assert call.get('type')=='function'and isinstance(call.get('id'),str)and call['id']
 fn=call['function'];name=fn['name'];args=json.loads(fn['arguments']);assert isinstance(args,dict)
 if name=='read_file':
  assert set(args)=={'path'}and args['path']in['input.json','result.json'];result=json.loads((sandbox/args['path']).read_bytes())
 elif name=='calculate':
  assert set(args)=={'a','b'}and all(type(args[k])is int and abs(args[k])<10**9 for k in ['a','b']);result={'sum':args['a']+args['b']}
 elif name=='write_file':
  assert set(args)=={'path','content'}and args['path']=='result.json'and isinstance(args['content'],str)and len(args['content'])<2048
  value=json.loads(args['content']);assert isinstance(value,dict);p=sandbox/'result.json'
  with p.open('xb')as f:f.write((compact(value)+'\n').encode())
  result={'ok':True}
 else:raise AssertionError('Undeclared model tool')
 actions.append({'generated_call':call,'arguments':args,'actual_result':result})
 return {'role':'tool','tool_call_id':call['id'],'content':compact(result)}

def workflow(client,root,stream):
 sandbox=root/('sse-four-tool'if stream else'http-four-tool');sandbox.mkdir();values={'a':10+secrets.randbelow(90),'b':10+secrets.randbelow(90),'nonce':secrets.token_hex(4)};(sandbox/'input.json').write_bytes(compact(values).encode())
 messages=[{'role':'system','content':'Use tools; never invent their results.'},{'role':'user','content':'Read input.json, calculate a+b, write result.json with nonce and sum, read result.json, then report nonce and sum.'}]
 expected=['read_file','calculate','write_file','read_file'];actions=[];wire=[];seen=set()
 for name in expected:
  message,finish,usage,e=client.request(messages,{'type':'function','function':{'name':name}},stream,TOOLS);wire.append(e)
  assert e['HTTP_status']==200 and finish=='tool_calls','Model did not finish a complete tool call'
  calls=message.get('tool_calls',[]);assert len(calls)==1 and calls[0]['function']['name']==name
  call=calls[0];assert call['id']not in seen;seen.add(call['id']);messages.append(message);messages.append(execute(call,sandbox,actions))
 message,finish,usage,e=client.request(messages,'none',stream,TOOLS);wire.append(e)
 assert e['HTTP_status']==200 and finish=='stop'and not message.get('tool_calls')
 correct={'nonce':values['nonce'],'sum':values['a']+values['b']};saved=json.loads((sandbox/'result.json').read_bytes())
 assert saved==correct and actions[1]['arguments']=={'a':values['a'],'b':values['b']}
 assert actions[3]['arguments']=={'path':'result.json'}and actions[3]['actual_result']==correct
 assert values['nonce']in(message.get('content')or'')and str(correct['sum'])in(message.get('content')or'')
 return {'passed':True,'stream':stream,'actions':actions,'wire':wire,'final':message,'result_SHA256':sha((sandbox/'result.json').read_bytes()),'real_file_readback':True}

def ordinary(client,thinking=False):
 message,finish,usage,e=client.request([{'role':'user','content':'What is 1+1? Give a short answer.'}],'none',False,None,256 if thinking else 24,thinking)
 assert e['HTTP_status']==200 and finish=='stop'and not message.get('tool_calls')and '2'in(message.get('content')or'')
 if thinking:assert message.get('reasoning_content'),'Thinking requested but not observed separately'
 assert '<think>'not in(message.get('content')or'')and '</think>'not in(message.get('content')or'')
 return {'passed':True,'wire':e,'message':message}

def single_choice(client,choice):
 calc=[TOOLS[1]];messages=[{'role':'user','content':'Use calculate to add 12 and 34.'}]
 message,finish,usage,e=client.request(messages,choice,False,calc)
 assert e['HTTP_status']==200 and finish=='tool_calls';calls=message.get('tool_calls',[]);assert len(calls)==1
 call=calls[0];assert call['type']=='function'and call['function']['name']=='calculate';args=json.loads(call['function']['arguments']);assert args=={'a':12,'b':34}
 return {'passed':True,'wire':e,'generated_call':call,'actual_result':{'sum':args['a']+args['b']}}

def negative(client,kind):
 if kind=='invalid_history':
  messages=[{'role':'tool','tool_call_id':'no-pending-id','content':'{}'}];tools=TOOLS;limit=24;choice='auto';expected=400
 elif kind=='invalid_schema':
  messages=[{'role':'user','content':'Hi'}];tools=[{'type':'function','function':{'name':'bad','parameters':{'type':'not-a-type'}}}];limit=24;choice='auto';expected=400
 else:
  messages=[{'role':'user','content':'Use calculate to add 12 and 34.'}];tools=[TOOLS[1]];limit=1;choice='required';expected=None
 message,finish,usage,e=client.request(messages,choice,False,tools,limit)
 if expected is not None:assert e['HTTP_status']==expected
 else:assert e['HTTP_status']!=200 or(finish!='tool_calls'and not message.get('tool_calls'))
 return {'passed':True,'wire':e,'response':message,'tools_executed':0}

def parallel_history(client,root):
 messages=[{'role':'user','content':'Call calculate twice: 12+34 and 56+78. Then report both results.'}]
 message,finish,usage,e=client.request(messages,'auto',False,[TOOLS[1]],parallel=True)
 calls=message.get('tool_calls',[])
 if e['HTTP_status']!=200 or len(calls)!=2:
  return {'status':'MULTICALL_NOT_PRODUCED_BY_MODEL','observed_multicall':False,'passed':None,'wire':e,'message':message}
 assert finish=='tool_calls'and len({x['id']for x in calls})==2
 pairs=sorted((json.loads(x['function']['arguments'])['a'],json.loads(x['function']['arguments'])['b'])for x in calls)
 assert pairs==[(12,34),(56,78)]and all(x['function']['name']=='calculate'for x in calls)
 actions=[];results=[execute(call,root,actions)for call in calls];messages.append(message);messages.extend(reversed(results))
 final,reason,usage,last=client.request(messages,'none',False,[TOOLS[1]])
 assert last['HTTP_status']==200 and reason=='stop'and not final.get('tool_calls')
 assert '46'in(final.get('content')or'')and '134'in(final.get('content')or'')
 return {'status':'REVERSED_MATCHING_ID_RESULTS_PASS','observed_multicall':True,'passed':True,'wire':[e,last],'actions':actions,'reversed_tool_result_ids':[x['tool_call_id']for x in reversed(results)],'final':final}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--base-url',required=True);ap.add_argument('--model',required=True);ap.add_argument('--output-dir',required=True);ap.add_argument('--identity',required=True);ap.add_argument('--max-tokens',type=int,default=96);ap.add_argument('--timeout',type=float,default=1000);ap.add_argument('--thinking-case',action='store_true');args=ap.parse_args()
 out=pathlib.Path(args.output_dir);out.mkdir(parents=True,exist_ok=False);identity=pathlib.Path(args.identity);json.loads(identity.read_bytes());c=WireClient(args.base_url,args.model,out,args.max_tokens,args.timeout)
 report={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':False,'passed':False,'real_model_qualification':'UNPROVEN_UNTIL_BACKEND_IDENTITY_AUDIT','identity':str(identity),'identity_SHA256':sha(identity.read_bytes()),'cases':[],'conditional_cases':[],'limitations':['P0 response text may be buffered; no incremental token latency claim','Parallel multi-call reverse-result check is not fabricated if model emits one call']}
 def save():(out/'result.json').write_bytes((json.dumps(report,ensure_ascii=False,indent=2)+'\n').encode())
 plan=[('baseline',lambda:ordinary(c)),('HTTP_four_tool_named',lambda:workflow(c,out,False)),('SSE_four_tool_named',lambda:workflow(c,out,True)),('choice_auto',lambda:single_choice(c,'auto')),('choice_required',lambda:single_choice(c,'required')),('invalid_history',lambda:negative(c,'invalid_history')),('invalid_schema',lambda:negative(c,'invalid_schema')),('truncated_no_execution',lambda:negative(c,'truncated')),('recovery',lambda:ordinary(c))]
 if args.thinking_case:plan.append(('thinking_separation',lambda:ordinary(c,True)))
 for name,fn in plan:
  row={'name':name,'started_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':False};report['cases'].append(row);save()
  try:row.update(fn())
  except BaseException as e:row.update(error=repr(e),status='ACTUAL_FAILURE_REVIEW_RAW_WIRE',request_serial=c.serial)
  save()
 row={'name':'parallel_calls_reversed_matching_IDs','passed':False};report['conditional_cases'].append(row);save()
 try:row.update(parallel_history(c,out))
 except BaseException as e:row.update(error=repr(e),status='ACTUAL_MULTICALL_OR_PROTOCOL_FAILURE_REVIEW_WIRE')
 report.update(completed=True,passed=all(x['passed']for x in report['cases'])and all(x.get('passed')is not False for x in report['conditional_cases']),actual_HTTP_requests=c.serial);save();print(compact({'completed':True,'passed':report['passed'],'cases':len(report['cases']),'requests':c.serial}));return 0 if report['passed']else 1

if __name__=='__main__':raise SystemExit(main())
