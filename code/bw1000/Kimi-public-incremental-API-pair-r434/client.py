"""Six same-input thinking SSE P0/P1 requests, public HTTP/SSE, no retries."""
import argparse,hashlib,json,time,urllib.request,os
from pathlib import Path
from workload import body_and_gold,validate_reply,opaque_history
from wire_client_r290 import sse_message
sha=lambda raw:hashlib.sha256(raw).hexdigest()

def save(path,q):
 t=path.with_suffix('.tmp');t.write_text(json.dumps(q,indent=2,allow_nan=False)+'\n');t.replace(path)

def wait_json(path,deadline,predicate=lambda q:True):
 first=None
 while time.time()<deadline:
  try:
   q=json.loads(Path(path).read_bytes())
   if predicate(q):return q
  except(FileNotFoundError,json.JSONDecodeError):pass
  first=time.monotonic()if first is None else first
  if time.monotonic()-first>90:raise TimeoutError('Bounded artifact visibility:'+str(path))
  time.sleep(.1)
 raise TimeoutError('Original service deadline')
def wait_listener(pid,port,deadline):
 while time.time()<deadline:
  inodes=set()
  for p in Path('/proc/'+str(pid)+'/fd').iterdir():
   try:target=os.readlink(p)
   except OSError:continue
   if target.startswith('socket:['):inodes.add(target[8:-1])
  for name in ['tcp','tcp6']:
   for line in Path('/proc/net/'+name).read_text().splitlines()[1:]:
    row=line.split()
    if row[3]=='0A'and int(row[1].split(':')[1],16)==port and row[9]in inodes:return
  time.sleep(.1)
 raise TimeoutError('Public frontend listener absent; no connection retry')


def main(argv=None):
 p=argparse.ArgumentParser()
 for n in ['base-url','runtime-dir','run','expected-step','source-sha','config-sha','fixtures','output-dir']:p.add_argument('--'+n,required=True)
 p.add_argument('--deadline-unix',type=float,required=True);a=p.parse_args(argv)
 output=Path(a.output_dir);output.mkdir(exist_ok=False);result=output/'result.json'
 q=dict(completed=False,passed=False,planned_HTTP=6,attempted_HTTP=0,completed_HTTP=0,rows=[],retry=False,
  HTTP=True,SSE=True,performance_comparison=False,transport_latency_comparison=True,fullbenchmark=False,incremental_token_SLO=False)
 def persist():save(result,q)
 def remaining():
  left=a.deadline_unix-time.time()
  if left<=120:raise TimeoutError('Original service deadline')
  return left
 persist()
 try:
  runtime=Path(a.runtime_dir)
  claim=wait_json(runtime/('claim-'+a.run+'-node0.json'),a.deadline_unix)
  assert claim['run']==a.run and claim['node']==0 and claim['actual_step']==a.expected_step
  assert claim['source_manifest_SHA256']==a.source_sha and claim['config_SHA256']==a.config_sha
  wait_listener(claim['PID'],43844,a.deadline_unix)  # passive PID-owned LISTEN proof; no health request
  card=json.loads(Path(a.fixtures).read_bytes());assert len(card['plan'])==6
  seen=set();source_report=runtime/('Kimi-public-r428-'+a.run+'-node0.json')
  for fixture in card['plan']:
   raw,body,expected=body_and_gold(fixture);history=opaque_history(body)
   request=urllib.request.Request(a.base_url.rstrip('/')+'/chat/completions',data=raw,headers={'Content-Type':'application/json'})
   q['attempted_HTTP']+=1;sent=time.monotonic();ticks=[];line_arrivals=[];response_ids=set();chunks=[];wire_raw=None;header_at=None
   (output/('request-%d.json'%fixture['index'])).write_bytes(raw)
   q['current_attempt']=dict(index=fixture['index'],delivery=fixture['delivery'],warm=fixture['warm'],sent=sent,actual_request_SHA256=sha(raw));persist()
   with urllib.request.urlopen(request,timeout=min(3540,remaining()-20))as response:
    header_at=time.monotonic();assert response.status==200
    if fixture['stream']:
     assert 'text/event-stream'in response.headers.get('Content-Type','')
     chunks=[]
     for line in response:
      tick=time.monotonic();chunks.append(line);line_arrivals.append(dict(arrival_monotonic=tick,bytes=len(line),is_data=line.startswith(b'data: ')))
      if line.startswith(b'data: '):
       value=line[6:].strip();event_row=dict(arrival_monotonic=tick,bytes=len(line),nonempty_delta=False)
       if value!=b'[DONE]':
        event=json.loads(value)
        if event.get('id'):response_ids.add(event['id'])
        for channel,key in [('reasoning','reasoning_content'),('content','content'),('tool','tool_calls')]:
         event_row[channel+'_nonempty']=any(bool(c.get('delta',{}).get(key))for c in event.get('choices',[]))
        event_row['nonempty_delta']=any(event_row.get(channel+'_nonempty')for channel in ['reasoning','content','tool'])
       ticks.append(event_row)
     wire_raw=b''.join(chunks);message,finish,usage=sse_message(wire_raw);assert len(response_ids)==1
     wire=dict(id=next(iter(response_ids)),choices=[dict(index=0,message=message,finish_reason=finish)],usage=usage)
    else:
     first=response.read(1);ticks=[dict(arrival_monotonic=time.monotonic(),bytes=len(first),nonempty_delta=False)]
     wire_raw=first+response.read();wire=json.loads(wire_raw)
   ended=time.monotonic();(output/('response-%d.wire'%fixture['index'])).write_bytes(wire_raw)
   (output/('request-%d.json'%fixture['index'])).write_bytes(raw)
   q['current_attempt'].update(headers=header_at,completed=ended,wire_SHA256=sha(wire_raw),events=ticks,line_arrivals=line_arrivals);persist()
   usage,call=validate_reply(wire,fixture,expected,history)
   ident=wire['id'];assert ident not in seen;seen.add(ident)
   report=wait_json(source_report,a.deadline_unix,lambda r:any(x.get('request_id')==ident and x.get('normal_reset')and'finish'in x for x in r.get('agent_requests',[])))
   rows=[r for r in report['agent_requests']if r.get('request_id')==ident];assert len(rows)==1;r=rows[0]
   assert report['public_source_manifest_SHA256']==a.source_sha
   assert report['library_sha256']=='8df87f0741b13c90f658174a793f0068efb4eb8097eddf1d5f44edb4d208e1a2'
   assert r['prefill_mode']=='W512'and r['normal_reset']and r['finish']=='eos'
   assert r['input_tokens']==usage['prompt_tokens']==fixture['current_prompt_tokens']
   assert r['output_budget']==fixture['output_budget']and r['history_render_audit']['Native_prompt_token_ids_SHA256']==fixture['current_prompt_IDs_SHA256']
   h=r['history_render_audit'];assert h['qualified']and h['input_RC_fields']==fixture['current_history_RC_count']and h['RC_one_contribution_per_source_field']and h['RC_order_preserved']and not h['markers_sent_to_Native']
   generated=r['generated_ids'];assert generated and generated[-1]in[163585,163586]and len(generated)==usage['completion_tokens']
   assert not r.get('error')and not r.get('cleanup_errors')
   observations=r['long_prefill_observations'];assert len(observations)==1 and observations[0]['full_prompt_consumed']and observations[0]['input_tokens_committed']==r['input_tokens']
   all_ticks=r['backend_token_monotonic'];assert len(all_ticks)==len(generated)and all(x<=y for x,y in zip(all_ticks,all_ticks[1:]))
   visible=[tick for token,tick in zip(generated,all_ticks)if token not in[163585,163586]]
   q['rows'].append(dict(index=fixture['index'],stream=fixture['stream'],delivery=fixture['delivery'],warm=fixture['warm'],request_id=ident,
    source_request_SHA256=fixture['source_SHA256'],actual_request_SHA256=sha(raw),wire_SHA256=sha(wire_raw),
    opaque_history=history,actual_generated_tool_call=call,independent_gold=expected,
    semantic_gold=True,natural_EOS=True,normal_reset=True,usage=usage,prompt_IDs_SHA256=fixture['current_prompt_IDs_SHA256'],
    client_clock=dict(sent=sent,headers=header_at,completed=ended,wall_s=ended-sent,events=ticks,line_arrivals=line_arrivals,
     first_reasoning_payload_s=next((v['arrival_monotonic']-sent for v in ticks if v.get('reasoning_nonempty')),None),
     first_content_payload_s=next((v['arrival_monotonic']-sent for v in ticks if v.get('content_nonempty')),None),
     first_tool_payload_s=next((v['arrival_monotonic']-sent for v in ticks if v.get('tool_nonempty')),None),
     chunks_are_not_tokens=True,
     first_payload_delta_s=next((v['arrival_monotonic']-sent for v in ticks if v['nonempty_delta']),None)),
    backend_clock=dict(scope='Node0 Native dequeue/reset-to-emitter timeline, not admission/HTTP SLO',
     TTFT_s=all_ticks[0]-r['started_monotonic'],visible_tokens=len(visible),including_EOS=len(generated),
     visible_decode_tps=(len(visible)-1)/(visible[-1]-visible[0])if len(visible)>1 else None),passed=True))
   q['completed_HTTP']+=1;q['current_attempt']=None;persist()
  assert q['attempted_HTTP']==q['completed_HTTP']==6 and len(q['rows'])==6
  q['passed']=True
 except BaseException as error:
  partial=b''.join(chunks)if 'chunks'in locals()else b''
  if 'fixture'in locals():
   (output/('response-%d.partial.wire'%fixture['index'])).write_bytes(partial)
   q['failed_attempt']=dict(index=fixture['index'],delivery=fixture['delivery'],events=ticks if 'ticks'in locals()else [],line_arrivals=line_arrivals if 'line_arrivals'in locals()else [],partial_wire_SHA256=sha(partial))
  q.update(error=repr(error),traceback=__import__('traceback').format_exc())
 finally:q['completed']=True;persist()
 return 0 if q['passed']else 1


if __name__=='__main__':raise SystemExit(main())

