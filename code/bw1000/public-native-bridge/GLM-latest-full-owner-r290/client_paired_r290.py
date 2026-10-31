"""Replay one frozen real named-tool request, 2 warmups +4 ABBA blocks.
CPU HTTP client only. No synthetic model replies, changing inputs or retries.
This records emitter observations; root must audit all32 owners and cleanup.
"""
import argparse,hashlib,json,pathlib,time,urllib.request,urllib.error,subprocess,sys,os
PLAN=(1,2)+(1,2,2,1)*4
FROZEN_REQUEST_SHA256='5ee8ea9cc99e12969d4b0505d4dd70f50e6436079206e8de5a6e908c5854c9dd'

def publish_wrapper(path,row):
    temp=path.with_name(path.name+'.tmp-'+str(os.getpid()))
    with temp.open('x')as out:json.dump(row,out,indent=2)
    temp.replace(path)


def validate_request(body):
    assert body['model']=='GLM-5.3' and body['stream'] is False
    assert body['max_tokens']==4096 and body['chat_template_kwargs']=={'clear_thinking':False}
    assert 'reasoning_effort' not in body, 'Retain actual GLM native-default reasoning policy'
    assert body['tool_choice']=={'type':'function','function':{'name':'calculate'}}
    assert body['parallel_tool_calls'] is False and body['messages'] and body['tools']
    assert body['temperature']==0
    result=body['messages'][-1];assert result['role']=='tool'
    data=json.loads(result['content']);values={row['id']:row['value'] for row in data['records']}
    query=data['query_ids'];assert len(query)==4
    expected={'a':values[query[0]],'b':values[query[1]]}
    assert all(type(value)is int and abs(value)<10**9 for value in expected.values())
    return expected

def validate_reply(wire,expected):
    assert isinstance(wire['id'],str) and wire['id'] and 'error' not in wire
    assert len(wire['choices'])==1
    choice=wire['choices'][0];message=choice['message'];calls=message.get('tool_calls',[])
    assert choice['finish_reason']=='tool_calls' and len(calls)==1
    call=calls[0];assert call['type']=='function' and isinstance(call['id'],str) and call['id']
    assert call['function']['name']=='calculate'
    arguments=json.loads(call['function']['arguments'])
    assert arguments==expected and all(type(arguments[key])is int for key in ('a','b'))
    usage=wire['usage'];assert all(type(usage[k]) is int and usage[k]>0 for k in ['prompt_tokens','completion_tokens','total_tokens'])
    assert usage['total_tokens']==usage['prompt_tokens']+usage['completion_tokens']
    assert usage['prompt_tokens']>=8192 and usage['prompt_tokens']+4096<=32768
    assert usage['completion_tokens']<=4096
    return usage


def main():
 p=argparse.ArgumentParser()
 for n in ['base-url','model','output-dir','identity','service-report','manifest-sha','expected-step']:p.add_argument('--'+n,required=True)
 p.add_argument('--deadline-unix',type=float,required=True);p.add_argument('--timeout',type=float,default=7200);a=p.parse_args();root=pathlib.Path(a.output_dir);root.mkdir(exist_ok=False)
 from source_pin_r290 import verify,D,sha
 from client_owner_r290 import bind_identity,row,backend_metrics
 q={'completed':False,'passed':False,'actual_HTTP_requests':0,'cases':[],'warmup_requests':2,'measurement_requests':16,'no_retries':True,'matched_pair_gain':None,'all32_owner_audit_required':True,'client_incremental_metrics_qualified':False}
 def save():(root/'result.json').write_text(json.dumps(q,indent=2)+chr(10))
 def remaining():
  left=a.deadline_unix-time.time()
  if left<=180:raise TimeoutError('Original residency deadline; no retries')
  return left
 try:
  remaining();verify(a.manifest_sha);identity=bind_identity(a.identity,a.manifest_sha,a.service_report);assert identity['actual_step']==a.expected_step
  source=pathlib.Path('/refactor/results/GLM-regular-live-r204-GLM-regular-service-r204-b34db8d1eb674914/HTTP/request-002.request.json');raw=source.read_bytes();assert sha(source)==FROZEN_REQUEST_SHA256;body=json.loads(raw);expected=validate_request(body);ids=set();prompt_sha=None
  for index,window in enumerate(PLAN):
   req=urllib.request.Request(a.base_url.rstrip('/')+'/chat/completions',data=raw,headers={'Content-Type':'application/json'});q['actual_HTTP_requests']+=1;save();start=time.monotonic()
   with urllib.request.urlopen(req,timeout=min(a.timeout,remaining()-120))as response:status=response.status;wire_raw=response.read()
   wall=time.monotonic()-start;(root/('response-%02d.wire'%index)).write_bytes(wire_raw);assert status==200;wire=json.loads(wire_raw);usage=validate_reply(wire,expected);assert wire['id']not in ids;ids.add(wire['id']);r=row(a.service_report,a,wire['id'],102+index,body['user'],4096,remaining);assert r['native_finish_reason']=='eos_token'and r['pair_index']==index and r['prefill_control_window']==window and r['paired_warmup']==(index<2)and r['prompt_tokens']==usage['prompt_tokens']and r['actual_generated_tokens']==usage['completion_tokens'];actual=r['actual_prompt_token_ids_SHA256']
   if prompt_sha is None:prompt_sha=actual
   assert actual==prompt_sha
   q['cases'].append({'index':index,'window':window,'warmup':index<2,'request_id':wire['id'],'HTTP_wall_s':wall,'usage':usage,'wire_SHA256':hashlib.sha256(wire_raw).hexdigest(),'backend':backend_metrics(r),'natural_EOS':True,'passed':True});save()
  q.update(passed=q['actual_HTTP_requests']==len(q['cases'])==18,actual_request_SHA256=FROZEN_REQUEST_SHA256,actual_prompt_IDs_SHA256=prompt_sha,matched_pair_conditions_validated=True)
 except BaseException as e:q['error']=repr(e)
 finally:q['completed']=True;save()
 return 0 if q['passed']else 1
if __name__=='__main__':raise SystemExit(main())
