"""Exactly two frozen complete tool requests through ordinary public routes."""
import argparse,hashlib,json,os,time,urllib.request
from pathlib import Path
from workload_r349 import body_and_gold,validate_reply
from wire_client_r290 import sse_message

def sha(raw):return hashlib.sha256(raw).hexdigest()
def save(path,q):
 t=path.with_suffix('.tmp');t.write_text(json.dumps(q,indent=2,allow_nan=False)+'\n');t.replace(path)
def opaque_history(body):
 calls={};pairs=[]
 for m in body['messages']:
  if m['role']=='assistant':
   for call in m.get('tool_calls',[]):
    assert call['id']and call['id']not in calls;calls[call['id']]=call
  elif m['role']=='tool':
   key=m['tool_call_id'];assert key in calls
   pairs.append(dict(id=key,function=calls[key]['function']['name'],result_SHA256=sha(m['content'].encode())))
 return pairs
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
 for name in ['base-url','runtime-dir','run','expected-step','source-sha','config-sha','fixtures','output-dir']:p.add_argument('--'+name,required=True)
 p.add_argument('--deadline-unix',type=float,required=True);a=p.parse_args(argv)
 output=Path(a.output_dir);output.mkdir(exist_ok=False);result=output/'result.json'
 q=dict(completed=False,passed=False,planned_HTTP=2,attempted_HTTP=0,completed_HTTP=0,rows=[],retry=False,
   HTTP=True,SSE=True,performance_comparison=False,fullbenchmark=False,incremental_token_SLO=False)
 def persist():save(result,q)
 def remaining():
  left=a.deadline_unix-time.time()
  if left<=120:raise TimeoutError('Original service deadline; no partial-plan qualification')
  return left
 persist()
 try:
  runtime=Path(a.runtime_dir)/a.run
  ready=wait_json(runtime/'MODEL_READY.json',a.deadline_unix)
  identity=ready['identity'];assert identity['run']==a.run and identity['actual_step']==a.expected_step
  assert identity['source_manifest_SHA256']==a.source_sha and identity['provider_config_SHA256']==a.config_sha
  assert ready['Model_READY']and ready['actual_owner_count']==16 and ready['all16_loaded_postload4GiB_and_service_agreement']
  frontend=wait_json(runtime/'FRONTEND.json',a.deadline_unix);assert frontend['identity']==identity
  assert frontend['PID']==wait_json(runtime/'leader-1.json',a.deadline_unix)['PID']
  wait_listener(frontend['PID'],43843,a.deadline_unix)
  card=json.loads(Path(a.fixtures).read_bytes());assert len(card['plan'])==2
  seen=set();source_report=runtime/'owner-1-0.json'
  for arm in card['plan']:
   f=card['fixtures'][arm['fixture_index']];raw,body,expected=body_and_gold(f);history=opaque_history(body)
   original_messages=json.dumps(body['messages'],ensure_ascii=False,separators=(',',':')).encode()
   if arm['stream']:
    body['stream']=True;body['stream_options']={'include_usage':True};request_raw=json.dumps(body,ensure_ascii=False,separators=(',',':')).encode()
   else:request_raw=raw
   assert json.dumps(body['messages'],ensure_ascii=False,separators=(',',':')).encode()==original_messages
   headers={'Content-Type':'application/json','X-InfiniLM-Native-Profile':arm['profile']}
   request=urllib.request.Request(a.base_url.rstrip('/')+'/chat/completions',data=request_raw,headers=headers)
   q['attempted_HTTP']+=1;persist();sent=time.monotonic();ticks=[];ids=set()
   with urllib.request.urlopen(request,timeout=min(7200,remaining()-20))as response:
    header_at=time.monotonic();assert response.status==200
    if arm['stream']:
     assert 'text/event-stream'in response.headers.get('Content-Type','')
     chunks=[]
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
   usage=validate_reply(wire,f,expected);request_id=wire['id'];assert request_id not in seen;seen.add(request_id)
   assert all(call['id']not in{v['id']for v in history}for call in wire['choices'][0]['message'].get('tool_calls',[]))
   report=wait_json(source_report,a.deadline_unix,lambda r:any(x.get('request_id')==request_id and'final_metadata'in x for x in r.get('service_requests',[])))
   rows=[r for r in report['service_requests']if r.get('request_id')==request_id];assert len(rows)==1;r=rows[0]
   assert r['native_profile']==arm['profile']and r['actual_prompt_token_ids_SHA256']==f['current_prepared_prompt_IDs_SHA256']
   assert r['prompt_tokens']==f['current_prepared_prompt_tokens']==usage['prompt_tokens']
   assert r['requested_output_budget']==4096 and r['generated_natural_EOS']and r['native_finish_reason']=='eos_token'
   generated=r['generated_token_ids'];eos=ready['prepared_identity']['eos_token_ids']
   assert generated and generated[-1]in eos and len(generated)==usage['completion_tokens']
   assert r['normal_reset_before']and r['normal_reset_after']and r['final_metadata']['original_reset_all16_completed']
   assert all(call.get('finite')is True and call['CAPI_RC']==0 for call in r['native_calls'])
   visible=[t for token,t in zip(generated,r['backend_token_monotonic'])if token not in eos]
   q['rows'].append(dict(**arm,fixture=f['id'],request_id=request_id,source_request_SHA256=f['request_SHA256'],
    actual_request_SHA256=sha(request_raw),wire_SHA256=sha(wire_raw),message_history_SHA256=sha(original_messages),
    opaque_associations=history,semantic_gold=True,integer_arguments=expected,natural_EOS=True,usage=usage,prompt_IDs_SHA256=r['actual_prompt_token_ids_SHA256'],
    client_clock=dict(sent=sent,headers=header_at,completed=ended,wall_s=ended-sent,events=ticks,
     first_payload_delta_s=next((t['arrival_monotonic']-sent for t in ticks if t['nonempty_delta']),None)),
    backend_clock=dict(TTFT_s=r['backend_token_monotonic'][0]-r['started_monotonic'],
     visible_tokens=len(visible),including_EOS=len(generated),
     visible_decode_tps=(len(visible)-1)/(visible[-1]-visible[0])if len(visible)>1 else None),passed=True))
   q['completed_HTTP']+=1;persist()
  assert q['attempted_HTTP']==q['completed_HTTP']==len(q['rows'])==2
  q['passed']=True
 except BaseException as error:q['error']=repr(error)
 finally:q['completed']=True;persist()
 return 0 if q['passed']else 1
if __name__=='__main__':raise SystemExit(main())
