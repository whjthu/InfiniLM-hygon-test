"""One complete existing9741history, six immutable W2/W4 requests."""
import json,pathlib,hashlib
WINDOWS=(2,4,2,4,4,2)
def load_plan(path):
 card=json.loads(pathlib.Path(path).read_bytes());fixtures=card['fixtures'];assert len(fixtures)==1 and fixtures[0]['id']=='long-write'
 f=fixtures[0];assert f['current_prepared_prompt_tokens']==9741 and f['original_budget']==4096 and len(f['request_SHA256'])==len(f['current_prepared_prompt_IDs_SHA256'])==64
 plan=[dict(global_index=i,fixture_index=0,fixture_id='long-write',arm_index=i,window=w,warmup=i<2)for i,w in enumerate(WINDOWS)]
 assert len(plan)==6;return card,plan

def body_and_gold(fixture):
 source=pathlib.Path(fixture['request_external_path'])
 raw=source.read_bytes();assert hashlib.sha256(raw).hexdigest()==fixture['request_SHA256'];body=json.loads(raw)
 assert body['model']=='GLM-5.3' and body['stream'] is False and body['max_tokens']==4096
 assert body['temperature']==0 and body['parallel_tool_calls'] is False
 assert body['chat_template_kwargs']=={'clear_thinking':False} and 'reasoning_effort' not in body
 gold=fixture['gold'];assert body['tool_choice']=={'type':'function','function':{'name':gold['tool']}}
 assert [x['role'] for x in body['messages']]==fixture['message_roles']
 if fixture['id']=='short-read':
  assert 'input.json' in body['messages'][-1]['content'];expected={'path':'input.json'}
 else:
  data=json.loads(next(x['content'] for x in body['messages'] if x['role']=='tool' and 'records' in x['content']))
  values={x['id']:x['value'] for x in data['records']};a,b,c,d=[values[k] for k in data['query_ids']]
  assert all(type(x)is int for x in (a,b,c,d))
  expected={'a':a,'b':b} if fixture['id']=='anchor-calculate' else {'path':'result.json','content':{'nonce':data['nonce'],'first_sum':a+b,'second_sum':c+d}}
 assert {k:v for k,v in expected.items() if k!='content'}==gold['arguments']
 if 'content' in expected:assert expected['content']==gold['content_JSON']
 return raw,body,expected
def validate_reply(wire,fixture,expected):
 assert isinstance(wire['id'],str) and wire['id'] and 'error' not in wire
 assert len(wire['choices'])==1
 choice=wire['choices'][0];calls=choice['message'].get('tool_calls',[])
 assert choice['finish_reason']=='tool_calls' and len(calls)==1
 call=calls[0];assert call['type']=='function' and isinstance(call['id'],str) and call['id']
 assert call['function']['name']==fixture['gold']['tool']
 actual=json.loads(call['function']['arguments'])
 if fixture['id']=='long-write':
  assert set(actual)=={'path','content'} and isinstance(actual['content'],str)
  actual['content']=json.loads(actual['content'])
 assert actual==expected
 def exact_ints(obj):
  if isinstance(obj,dict):
   for value in obj.values():exact_ints(value)
  elif isinstance(obj,(int,float)):assert type(obj)is int
 exact_ints(actual)
 usage=wire['usage'];assert all(type(usage[k])is int and usage[k]>0 for k in ['prompt_tokens','completion_tokens','total_tokens'])
 assert usage['prompt_tokens']==fixture['current_prepared_prompt_tokens'] and usage['completion_tokens']<=4096
 assert usage['total_tokens']==usage['prompt_tokens']+usage['completion_tokens']
 return usage
