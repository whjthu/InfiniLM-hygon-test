"""Six actual HTTP requests: two warmups then observerOFF/ON/ON/OFF, HCfixed1."""
import argparse,copy,pathlib,json,time,hashlib
from source_pin_r281 import verify,load_wire,sha,D
from client_owner_r281 import bind_identity,row,backend_metrics,bound_report
from content_json import parse_final_object
from chunk_contract import ARMS,HC_MODES
from fixture_contract import load as load_fixture
def main():
 p=argparse.ArgumentParser()
 for n in ['base-url','model','output-dir','identity','service-report','manifest-sha','expected-step']:p.add_argument('--'+n,required=True)
 p.add_argument('--deadline-unix',required=True,type=float);p.add_argument('--timeout',type=float,default=2400);a=p.parse_args();out=pathlib.Path(a.output_dir);out.mkdir(exist_ok=False);q=dict(completed=False,passed=False,requests=[],planned_HTTP=6,planned_Native=6,actual_HTTP_attempts=0,actual_HTTP_completed=0,actual_Native_observed=0,actual_Native_completed=0,scope='Same full actual8489-token tool-final prompt; 2warm+observerOFF/ON/ON/OFFHCfixed1/fixed1024, output128, no widequality or onlineSLO proof',no_retries=True)
 def save():(out/'result.json').write_bytes((json.dumps(q,indent=2)+'\n').encode())
 def remaining():
  left=a.deadline_unix-time.time()
  if left<=120:raise TimeoutError('Original residency deadline exhausted')
  return left
 def update_native_counts():
  try:
   owner=bound_report(a.service_report,a);q['actual_Native_observed']=len(owner['requests']);q['actual_Native_completed']=sum(bool(r.get('normal_reset_after')and r.get('ended_monotonic'))for r in owner['requests'])
  except BaseException as e:q['count_observation_error']=repr(e)
 save()
 try:
  remaining();identity=bind_identity(a.identity,a.manifest_sha,a.service_report);assert identity['actual_step']==a.expected_step;verify(a.manifest_sha);fixture=load_fixture();body=fixture['body'];wire=load_wire(a.manifest_sha)
  for i,chunk in enumerate(ARMS):
   remaining();mode=HC_MODES[i];folder=out/f'{i:02d}-mode{mode}-{chunk}';folder.mkdir();client=wire.WireClient(a.base_url,a.model,folder,128,min(a.timeout,remaining()-60));client.session=f'Deep393-{i}-mode{mode}';messages=copy.deepcopy(body['messages']);before=copy.deepcopy(messages);q['actual_HTTP_attempts']+=1;save();message,finish,usage,event=client.request(messages,body['tool_choice'],False,body['tools'],128,False,body['parallel_tool_calls']);q['actual_HTTP_completed']+=1;update_native_counts();save();assert messages==before and event['HTTP_status']==200 and finish=='stop'and not message.get('tool_calls')
   actual=json.loads((folder/'request-001.request.json').read_bytes());assert actual['messages']==body['messages']and actual['tools']==body['tools']and actual['max_tokens']==128
   raw=json.loads((folder/'request-001.wire').read_bytes());r=row(pathlib.Path(a.service_report),a,raw['id'],i,client.session,128,remaining);assert r['prefill_binding']['prefill_chunk']==chunk and r['prefill_binding']['arm_index']==i;assert r['prefill_binding']['hc_epilogue_mode']==r['HC_mode']==mode;assert len(r['HC_forward_coverage'])==r['Native_forward_host_summary']['prefill_calls']+r['Native_forward_host_summary']['decode_calls'];assert all(v['mode']==mode and v['not_hardware_GPU_hit_counter']for v in r['HC_forward_coverage']);assert sum(v['candidate_completed_branch']for v in r['HC_forward_coverage'])==(r['Native_forward_host_summary']['decode_calls']if mode else 0);assert r['prompt_tokens']==usage['prompt_tokens']==fixture['actual_prompt_tokens'];assert r['admission']['history_render_audit']['Native_prompt_token_ids_SHA256']==fixture['actual_prompt_token_ids_SHA256'];assert r['native_finish_reason']=='eos_token'and r['actual_generated_tokens']<=128
   trace=[x for x in r['Native_continuation']if x['kind']=='prefill'];assert len(trace)==(r['prompt_tokens']+chunk-1)//chunk and all(x['tokens']==chunk for x in trace[:-1])and trace[0]['past']==0 and trace[-1]['total']==r['prompt_tokens']
   assert mode==1 and r['prefill_binding']['observer_enabled']==(i in [3,4])
   profiles=r.get('owner_profiles',[]);assert [v['geometry']['decode_call_1based']for v in profiles]==([8,87]if i in [3,4]else[])
   assert all(v['native_window_closed']and v['owner_event_validation_passed']and v['inner_event_validation_passed']for v in profiles)
   answer,presentation=parse_final_object(message.get('content')or'');assert answer==fixture['expected_answer'],'Actual complete tool task answer differs'
   q['requests'].append(dict(index=i,chunk=chunk,hc_epilogue_mode=mode,HC_counters_before=r['HC_counters_before_request'],HC_counters_after=r['HC_counters_after_request'],HC_counter_observation_s=r['HC_counter_observation_s'],HC_qualified_forward_calls=len(r['HC_forward_coverage']),warmup=i<2,prompt_tokens=r['prompt_tokens'],generated_tokens=r['actual_generated_tokens'],natural_EOS=True,task_correct=True,presentation=presentation,client_HTTP_wall_s=event['elapsed_s'],wire_SHA256=event['wire_SHA256'],**backend_metrics(r)));save()
  q['passed']=len(q['requests'])==6
 except BaseException as e:q['error']=repr(e)
 finally:update_native_counts();q['completed']=True;save()
 return 0 if q['passed']else 1
if __name__=='__main__':raise SystemExit(main())
