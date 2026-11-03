"""Exactly two ordinary public requests, no retries; actual payload/wires stay R/L."""
import argparse,json,time,urllib.request
from pathlib import Path
from client_common import sha,save,opaque_history,wait_json,wait_listener
from fixture import body_and_gold,validate_reply
from wire_client_r290 import sse_message
def main(argv=None):
 p=argparse.ArgumentParser()
 for name in ['base-url','runtime-dir','run','expected-step','source-sha','config-sha','fixtures','output-dir']:p.add_argument('--'+name,required=True)
 p.add_argument('--deadline-unix',type=float,required=True);a=p.parse_args(argv)
 output=Path(a.output_dir);output.mkdir(exist_ok=False);result=output/'result.json'
 q=dict(completed=False,passed=False,planned_HTTP=2,attempted_HTTP=0,completed_HTTP=0,rows=[],retry=False,performance_comparison=False,fullbenchmark=False,incremental_token_SLO=False)
 def persist():save(result,q)
 def remaining():
  left=a.deadline_unix-time.time()
  if left<=120:raise TimeoutError('Original service deadline; no partial qualification')
  return left
 persist()
 try:
  runtime=Path(a.runtime_dir)/a.run;ready=wait_json(runtime/'MODEL_READY.json',a.deadline_unix);identity=ready['identity']
  assert identity['run']==a.run and identity['actual_step']==a.expected_step and identity['source_manifest_SHA256']==a.source_sha and identity['provider_config_SHA256']==a.config_sha
  assert ready['Model_READY']and ready['Native_internal_owner_count']==8 and ready['HC_mode']==0 and ready['TP']==8 and ready['PP']==1
  frontend=wait_json(runtime/'FRONTEND.json',a.deadline_unix);assert frontend['identity']==identity;wait_listener(frontend['PID'],43882,a.deadline_unix)
  card=json.loads(Path(a.fixtures).read_bytes());assert len(card['plan'])==2;seen=set()
  for arm in card['plan']:
   remaining();raw,body,expected,history_sha=body_and_gold(card,arm);history=opaque_history(body)
   (output/('request-%d.request.json'%arm['index'])).write_bytes(raw)
   request=urllib.request.Request(a.base_url.rstrip('/')+'/chat/completions',data=raw,headers={'Content-Type':'application/json'})
   q['attempted_HTTP']+=1;persist();sent=time.monotonic();ticks=[];ids=set()
   with urllib.request.urlopen(request,timeout=min(7200,remaining()-20))as response:
    header_at=time.monotonic();assert response.status==200
    if arm['stream']:
     assert'text/event-stream'in response.headers.get('Content-Type','');chunks=[]
     for line in response:
      tick=time.monotonic();chunks.append(line)
      if line.startswith(b'data: '):
       value=line[6:].strip();row=dict(arrival_monotonic=tick,bytes=len(line),nonempty_delta=False)
       if value!=b'[DONE]':
        event=json.loads(value)
        if event.get('id'):ids.add(event['id'])
        row['nonempty_delta']=any(any(c.get('delta',{}).get(k)for k in ['content','reasoning_content','tool_calls'])for c in event.get('choices',[]))
       ticks.append(row)
     wire_raw=b''.join(chunks);message,finish,usage=sse_message(wire_raw);assert len(ids)==1
     wire={'id':next(iter(ids)),'choices':[{'index':0,'message':message,'finish_reason':finish}],'usage':usage}
    else:
     first=response.read(1);ticks=[dict(arrival_monotonic=time.monotonic(),bytes=len(first),nonempty_delta=False)]
     wire_raw=first+response.read();wire=json.loads(wire_raw)
   ended=time.monotonic();(output/('response-%d.wire'%arm['index'])).write_bytes(wire_raw)
   usage,presentation=validate_reply(wire,expected,arm['budget'],arm['thinking']);request_id=wire['id'];assert request_id not in seen;seen.add(request_id)
   report=wait_json(runtime/'owner.json',a.deadline_unix,lambda r:any(v.get('request_id')==request_id and'final_metadata'in v for v in r.get('requests',[])))
   assert report['identity']==identity
   rows=[v for v in report['requests']if v.get('request_id')==request_id];assert len(rows)==1;r=rows[0]
   h=r['admission']['history_render_audit'];assert h['qualified']and h['prequeue_bound']and h['same_prepared_encoder_reproduces_prompt_and_IDs']and not h['markers_sent_to_Native']
   assert h['content_one_contribution_each_and_ordered']and h['all_content_contribution_slices_equal']and h['output_budget']==arm['budget']
   assert r['prompt_tokens']==usage['prompt_tokens']==h['prompt_tokens']and r['max_new_tokens']==arm['budget']and r['native_finish_reason']=='eos_token'
   assert bool(h['pending_thinking_prefix_open'])==arm['thinking']and r['normal_reset_before']and r['normal_reset_after']
   generated=r['generated_token_ids'];assert generated[-1]==1 and len(generated)==usage['completion_tokens']<=arm['budget']
   assert r['actual_forward_attempts']==r['actual_forward_completed']and not r.get('error')
   prefill=[v for v in r['Native_continuation']if v['kind']=='prefill'];assert prefill[0]['past']==0 and prefill[-1]['total']==r['prompt_tokens']and all(v['tokens']==1024 for v in prefill[:-1])
   assert sum(v['tokens']for v in prefill)==r['prompt_tokens']and len(prefill)==(r['prompt_tokens']+1023)//1024
   visible=[tick for token,tick in zip(generated,r['backend_token_monotonic'])if token!=1]
   q['rows'].append(dict(**arm,request_id=request_id,actual_request_SHA256=sha(raw),wire_SHA256=sha(wire_raw),complete_history_and_tools_SHA256=history_sha,
    opaque_associations=history,semantic_gold=True,natural_EOS=True,presentation=presentation,usage=usage,prompt_IDs_SHA256=h['Native_prompt_token_ids_SHA256'],
    client_clock=dict(sent=sent,headers=header_at,completed=ended,wall_s=ended-sent,events=ticks,first_payload_delta_s=next((v['arrival_monotonic']-sent for v in ticks if v['nonempty_delta']),None)),
    backend_clock=dict(TTFT_s=r['backend_token_monotonic'][0]-r['started_monotonic'],visible_tokens=len(visible),including_EOS=len(generated),visible_decode_tps=(len(visible)-1)/(visible[-1]-visible[0])if len(visible)>1 else None),
    Native_forward_host_summary=r['Native_forward_host_summary'],transport_text_buffered=True,passed=True))
   q['completed_HTTP']+=1;persist()
  assert q['attempted_HTTP']==q['completed_HTTP']==len(q['rows'])==2;q['passed']=True
 except BaseException as error:q['error']=repr(error)
 finally:q['completed']=True;persist()
 return 0 if q['passed']else 1
if __name__=='__main__':raise SystemExit(main())
