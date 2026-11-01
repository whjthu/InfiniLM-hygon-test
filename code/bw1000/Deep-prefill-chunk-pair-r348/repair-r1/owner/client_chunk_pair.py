"""Six actual HTTP requests: two warmups then one same-prompt ABBA."""
import argparse,copy,pathlib,json,time,hashlib
from source_pin_r281 import verify,load_wire,sha,D
from client_owner_r281 import bind_identity,row,backend_metrics
from content_json import parse_final_object
from chunk_contract import ARMS
def main():
 p=argparse.ArgumentParser()
 for n in ['base-url','model','output-dir','identity','service-report','manifest-sha','expected-step']:p.add_argument('--'+n,required=True)
 p.add_argument('--deadline-unix',required=True,type=float);p.add_argument('--timeout',type=float,default=2400);a=p.parse_args();out=pathlib.Path(a.output_dir);out.mkdir(exist_ok=False);q=dict(completed=False,passed=False,requests=[],HTTP=6,Native=6,scope='Same full actual8489-token tool-final prompt; 2warm+ABBA1024/2048, output128, no widequality or onlineSLO proof',no_retries=True)
 def save():(out/'result.json').write_bytes((json.dumps(q,indent=2)+'\n').encode())
 def remaining():
  left=a.deadline_unix-time.time()
  if left<=120:raise TimeoutError('Original residency deadline exhausted')
  return left
 save()
 try:
  remaining();identity=bind_identity(a.identity,a.manifest_sha,a.service_report);assert identity['actual_step']==a.expected_step;mf=verify(a.manifest_sha);f=D/'frozen-toolprompt.json';assert sha(f)==mf['files_SHA256']['frozen-toolprompt.json'];fixture=json.loads(f.read_bytes());body=fixture['body'];wire=load_wire(a.manifest_sha)
  for i,chunk in enumerate(ARMS):
   remaining();folder=out/f'{i:02d}-{chunk}';folder.mkdir();client=wire.WireClient(a.base_url,a.model,folder,128,min(a.timeout,remaining()-60));client.session=f'Deep348-{i}-{chunk}';messages=copy.deepcopy(body['messages']);before=copy.deepcopy(messages);message,finish,usage,event=client.request(messages,body['tool_choice'],False,body['tools'],128,False,body['parallel_tool_calls']);assert messages==before and event['HTTP_status']==200 and finish=='stop'and not message.get('tool_calls')
   actual=json.loads((folder/'request-001.request.json').read_bytes());assert actual['messages']==body['messages']and actual['tools']==body['tools']and actual['max_tokens']==128
   raw=json.loads((folder/'request-001.wire').read_bytes());r=row(pathlib.Path(a.service_report),a,raw['id'],i,client.session,128,remaining);assert r['prefill_binding']['prefill_chunk']==chunk and r['prefill_binding']['arm_index']==i;assert r['prompt_tokens']==usage['prompt_tokens']==fixture['actual_prompt_tokens'];assert r['admission']['history_render_audit']['Native_prompt_token_ids_SHA256']==fixture['actual_prompt_token_ids_SHA256'];assert r['native_finish_reason']=='eos_token'and r['actual_generated_tokens']<=128
   trace=[x for x in r['Native_continuation']if x['kind']=='prefill'];assert len(trace)==(r['prompt_tokens']+chunk-1)//chunk and all(x['tokens']==chunk for x in trace[:-1])and trace[0]['past']==0 and trace[-1]['total']==r['prompt_tokens']
   answer,presentation=parse_final_object(message.get('content')or'');assert answer==fixture['expected_answer'],'Actual complete tool task answer differs'
   q['requests'].append(dict(index=i,chunk=chunk,warmup=i<2,prompt_tokens=r['prompt_tokens'],generated_tokens=r['actual_generated_tokens'],natural_EOS=True,task_correct=True,presentation=presentation,client_HTTP_wall_s=event['elapsed_s'],wire_SHA256=event['wire_SHA256'],**backend_metrics(r)));save()
  q['passed']=len(q['requests'])==6
 except BaseException as e:q['error']=repr(e)
 finally:q['completed']=True;save()
 return 0 if q['passed']else 1
if __name__=='__main__':raise SystemExit(main())
