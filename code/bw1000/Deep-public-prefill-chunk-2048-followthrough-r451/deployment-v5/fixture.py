"""Two transports, same complete existing tool result; independent integer gold."""
import copy,hashlib,json
from pathlib import Path
from content_json import parse_final_object
def body_and_gold(card,arm):
 raw=Path(card['fixture_path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==card['fixture_SHA256']
 fixture=json.loads(raw);body=copy.deepcopy(fixture['body']);assert body['max_tokens']==4096 and body['tool_choice']=='none'
 expected={};calls={}
 for message in body['messages']:
  if message['role']=='assistant':
   for call in message.get('tool_calls',[]):
    assert call['function']['name']=='read_file'and call['id']not in calls
    calls[call['id']]=json.loads(call['function']['arguments'])['path']
  if message['role']=='tool':
   name=calls[message['tool_call_id']];data=json.loads(message['content']);values={row['id']:row['value']for row in data['records']}
   selected=[values[name]for name in data['query_ids']];assert len(selected)==4 and all(type(v)is int for v in selected)
   expected[name]=dict(nonce=data['nonce'],sum=sum(selected))
 assert set(expected)=={'left.json','right.json'}
 expected['total']=expected['left.json']['sum']+expected['right.json']['sum'];assert expected==fixture['expected_answer']
 before=json.dumps([body['messages'],body['tools']],ensure_ascii=False,separators=(',',':')).encode()
 assert arm['stream'] is True and arm['thinking'] is False and arm['budget']==128 and arm['delivery']=='P1'
 body.update(stream=True,max_tokens=128,text_delivery='P1');body['chat_template_kwargs']={'enable_thinking':False}
 if arm['stream']:body['stream_options']={'include_usage':True}
 assert json.dumps([body['messages'],body['tools']],ensure_ascii=False,separators=(',',':')).encode()==before
 return json.dumps(body,ensure_ascii=False,separators=(',',':')).encode(),body,expected,hashlib.sha256(before).hexdigest()
def validate_reply(wire,expected,budget,thinking):
 assert isinstance(wire['id'],str)and wire['id']and'error'not in wire and len(wire['choices'])==1
 choice=wire['choices'][0];assert choice['finish_reason']=='stop'and not choice['message'].get('tool_calls')
 actual,presentation=parse_final_object(choice['message'].get('content')or'')
 def integer_types(value):
  if isinstance(value,dict):
   for key,item in value.items():
    if key in ['sum','total']:assert type(item)is int
    integer_types(item)
 integer_types(actual);assert actual==expected
 if thinking:assert choice['message'].get('reasoning_content'),'Actual public thinking parser must expose reasoning'
 usage=wire['usage'];assert all(type(usage[k])is int and usage[k]>0 for k in ['prompt_tokens','completion_tokens','total_tokens'])
 assert usage['completion_tokens']<=budget and usage['total_tokens']==usage['prompt_tokens']+usage['completion_tokens']
 return usage,presentation
