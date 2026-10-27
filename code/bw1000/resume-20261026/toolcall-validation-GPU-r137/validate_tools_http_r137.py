"""Real HTTP/SSE client and local sandbox. Never constructs model outputs.

Run only against an independently qualified native service; this client does not
certify the service's backend identity or performance. Retain its startup receipt.
"""
import argparse,datetime,hashlib,json,pathlib,secrets,time,urllib.request,urllib.error

TOOLS=[
 {'type':'function','function':{'name':'read_file','description':'Read the sandbox input JSON.','parameters':{'type':'object','properties':{'path':{'type':'string'}},'required':['path'],'additionalProperties':False}}},
 {'type':'function','function':{'name':'calculate','description':'Add two integers.','parameters':{'type':'object','properties':{'a':{'type':'integer'},'b':{'type':'integer'}},'required':['a','b'],'additionalProperties':False}}},
 {'type':'function','function':{'name':'write_file','description':'Write the sandbox result JSON as a string.','parameters':{'type':'object','properties':{'path':{'type':'string'},'content':{'type':'string'}},'required':['path','content'],'additionalProperties':False}}},
]

def digest(data):return hashlib.sha256(data).hexdigest()

def decode_sse(lines):
 message={'role':'assistant','content':'','reasoning_content':''};calls={};finish=None;done=False;events=[]
 for line in lines:
  line=line.strip()
  if not line or line.startswith(':'):continue
  assert line.startswith('data: '),'Unexpected SSE framing'
  payload=line[6:]
  if payload=='[DONE]':done=True;break
  q=json.loads(payload);events.append(q)
  assert 'error'not in q,('SSE error',q.get('error'))
  for choice in q.get('choices',[]):
   assert choice.get('index',0)==0
   delta=choice.get('delta',{})
   for key in ['content','reasoning_content']:
    if delta.get(key):message[key]+=delta[key]
   for part in delta.get('tool_calls',[]):
    i=part.get('index',0);c=calls.setdefault(i,{'id':'','type':'function','function':{'name':'','arguments':''}})
    if part.get('id'):c['id']+=part['id']
    if part.get('type'):assert part['type']=='function'
    for key in ['name','arguments']:
     if part.get('function',{}).get(key):c['function'][key]+=part['function'][key]
   if choice.get('finish_reason')is not None:finish=choice['finish_reason']
 assert done and finish is not None,'SSE incomplete: never execute buffered tools'
 if calls:message['tool_calls']=[calls[i]for i in sorted(calls)]
 if not message['reasoning_content']:message.pop('reasoning_content')
 return message,finish,events

class Client:
 def __init__(self,url,model,directory,timeout,max_tokens):self.url=url.rstrip('/')+'/chat/completions';self.model=model;self.directory=directory;self.timeout=timeout;self.max_tokens=max_tokens;self.turn=0
 def request(self,messages,choice,stream,tools=TOOLS):
  self.turn+=1;p=self.directory/('turn-%02d'%self.turn)
  body={'model':self.model,'messages':messages,'tools':tools,'tool_choice':choice,'parallel_tool_calls':False,'stream':stream,'temperature':0,'max_tokens':self.max_tokens}
  body['chat_template_kwargs']={'enable_thinking':False}
  if stream:body['stream_options']={'include_usage':True}
  p.with_suffix('.request.json').write_text(json.dumps(body,ensure_ascii=False,indent=2),encoding='utf8')
  req=urllib.request.Request(self.url,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
  start=time.monotonic()
  try:
   with urllib.request.urlopen(req,timeout=self.timeout)as response:
    raw=response.read();status=response.status;headers=dict(response.headers)
  except urllib.error.HTTPError as e:
   raw=e.read();p.with_suffix('.response.json').write_bytes(raw);raise
  p.with_suffix('.wire').write_bytes(raw)
  if stream:
   assert 'text/event-stream'in headers.get('Content-Type','')
   message,finish,events=decode_sse(raw.decode().splitlines())
   q={'message':message,'finish_reason':finish,'events':events}
  else:
   q=json.loads(raw);assert 'error'not in q;q=q['choices'][0];message=q['message'];finish=q['finish_reason']
  assert status==200
  p.with_suffix('.response.json').write_text(json.dumps(q,ensure_ascii=False,indent=2),encoding='utf8')
  return message,finish,{'request':p.with_suffix('.request.json').name,'wire':p.with_suffix('.wire').name,'wire_SHA256':digest(raw),'elapsed_s':time.monotonic()-start,'finish_reason':finish}

def execute_tool(call,sandbox,audit):
 assert call['type']=='function'and isinstance(call['id'],str)and call['id']
 name=call['function']['name'];args=json.loads(call['function']['arguments']);assert isinstance(args,dict)
 if name=='read_file':
  assert args=={'path':'input.json'}
  result=json.loads((sandbox/'input.json').read_text())
 elif name=='calculate':
  assert set(args)=={'a','b'}and all(type(args[k])is int and abs(args[k])<10**9 for k in ['a','b'])
  result={'result':args['a']+args['b']}
 elif name=='write_file':
  assert set(args)=={'path','content'}and args['path']=='result.json'and isinstance(args['content'],str)and len(args['content'])<4096
  content=json.loads(args['content']);assert isinstance(content,dict)
  p=sandbox/'result.json'
  with p.open('x',encoding='utf8')as f:json.dump(content,f)
  result={'written':'result.json','bytes':p.stat().st_size,'SHA256':digest(p.read_bytes())}
 else:raise ValueError('Undeclared tool')
 audit.append({'tool_call_id':call['id'],'function':name,'arguments':args,'result':result})
 return {'role':'tool','tool_call_id':call['id'],'content':json.dumps(result,ensure_ascii=False)}

def run_case(client,sandbox,mode,stream):
 sandbox.mkdir();a=secrets.randbelow(9000)+1000;b=secrets.randbelow(9000)+1000;nonce=secrets.token_hex(10)
 (sandbox/'input.json').write_text(json.dumps({'a':a,'b':b,'nonce':nonce}),encoding='utf8')
 messages=[{'role':'system','content':'Use tools to complete the task. Never invent file contents or tool results. Follow the requested paths and JSON schema.'},{'role':'user','content':'Read input.json with read_file, use calculate to add its a and b, write result.json with write_file containing exactly JSON keys nonce and sum with the read nonce and calculated integer. Finally tell me both the nonce and sum.'}]
 calls=[];turns=[];seen=set();expected=['read_file','calculate','write_file'];final=None
 for turn in range(8):
  choice=({'type':'function','function':{'name':expected[len(calls)]}}if mode=='named'and len(calls)<3 else ('none'if len(calls)==3 else mode))
  message,finish,evidence=client.request(messages,choice,stream);turns.append(evidence);generated=message.get('tool_calls',[])
  assert finish not in ['length','timeout','error','canceled'],'Truncated/error turns are not tool completion'
  if generated:
   assert finish=='tool_calls'and len(generated)==1 and len(calls)<3
   call=generated[0];assert call['id']not in seen;seen.add(call['id']);assert call['function']['name']==expected[len(calls)]
   messages.append(message);messages.append(execute_tool(call,sandbox,calls))
  else:
   assert len(calls)==3 and finish=='stop';final=message;break
 assert final is not None,'No completed final answer'
 saved=json.loads((sandbox/'result.json').read_text());assert saved=={'nonce':nonce,'sum':a+b}
 assert calls[1]['arguments']=={'a':a,'b':b},'Calculation did not use actual read values'
 assert nonce in final.get('content','')and str(a+b)in final.get('content','')
 return {'passed':True,'choice_mode':mode,'stream':stream,'calls':calls,'turns':turns,'final':final,'sandbox_result_SHA256':digest((sandbox/'result.json').read_bytes())}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--base-url',required=True,help='OpenAI v1 URL, e.g. http://127.0.0.1:8000/v1');ap.add_argument('--model',required=True);ap.add_argument('--output-dir',required=True);ap.add_argument('--identity',required=True,help='Saved backend startup identity/weights receipt JSON');ap.add_argument('--timeout',type=float,default=1000);ap.add_argument('--max-tokens',type=int,default=1024);args=ap.parse_args()
 directory=pathlib.Path(args.output_dir);directory.mkdir(parents=True,exist_ok=False);identity=pathlib.Path(args.identity);identity_sha=digest(identity.read_bytes());json.loads(identity.read_bytes())
 report={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':False,'passed':False,'scope':'Real HTTP/SSE tool execution witness; not throughput or three-model equivalence','backend_identity_path':str(identity),'backend_identity_SHA256':identity_sha,'backend_identity_independently_verified_by_client':False,'cases':[]}
 try:
  client=Client(args.base_url,args.model,directory,args.timeout,args.max_tokens)
  for stream in [False,True]:
   for mode in ['named','auto','required']:report['cases'].append(run_case(client,directory/('%s-%s'%('SSE'if stream else'HTTP',mode)),mode,stream))
  report.update(completed=True,passed=True)
 except BaseException as e:report['error']=repr(e);raise
 finally:
  (directory/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
  print(json.dumps({k:report.get(k)for k in ['completed','passed','error']}))

if __name__=='__main__':main()
