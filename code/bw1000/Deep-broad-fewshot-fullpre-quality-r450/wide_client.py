"""Finite broad-quality client; no retry/Model creation or answer-based stopping."""
import argparse,hashlib,json,math,time,traceback,urllib.error,urllib.request
from pathlib import Path
from paired_plan import validate_subset
from coverage import validate
from scoring import grade,summarize
from wire_client_r290 import sse_message
sha=lambda b:hashlib.sha256(b).hexdigest()
compact=lambda q:json.dumps(q,ensure_ascii=False,separators=(',',':')).encode()

def save(path,q):
 p=Path(path);tmp=p.with_suffix('.tmp');tmp.write_bytes((json.dumps(q,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode());tmp.replace(p)

def wait_json(path,deadline,predicate=lambda q:True):
 while time.time()<deadline:
  try:q=json.loads(Path(path).read_bytes())
  except(FileNotFoundError,json.JSONDecodeError):time.sleep(.1);continue
  if predicate(q):return q
  time.sleep(.1)
 raise TimeoutError('Bounded original artifact visibility:'+str(path))

def run(cases,requests,encoded,frozen,plan,*,base_url,folder,identity,deadline,output,open_http=None,read_owner=None):
 validate_subset(plan,frozen);assert len(cases)==len(requests)==len(encoded)==256
 cby={c['id']:c for c in cases};rby={r['case_id']:r for r in requests};eby={e['case_id']:e for e in encoded}
 out=Path(output);out.mkdir(exist_ok=False);result=out/'result.json'
 q=dict(completed=False,execution_passed=False,global_planned_HTTP=512,assigned_HTTP=len(plan),attempted_HTTP=0,completed_HTTP=0,rows=[],pending_attempt=None,retry=False,reload=False,performance_comparison=False,quality_scope='Original128GSM+128MMLU pairedfullpre; wrong/cap/parse remain in planned denominator',Model_default_promotion=False)
 def persist():q['summary']=summarize(cases,q['rows']);save(result,q)
 def owner_for(entry):
  report=wait_json(Path(folder)/'owner.json',min(deadline,time.time()+90),lambda v:any(r.get('r450_logical_index')==entry['logical_index']and'final_metadata'in r for r in v.get('requests',[])))
  assert report['identity']==identity and report['Native_SHA256']=='a314ae6891977316189f875b793ed3f2e2eb0898110ad0cc382d2569eaf31dc1'and report['HC_mode']==0
  rows=[r for r in report['requests']if r.get('r450_logical_index')==entry['logical_index']];assert len(rows)==1;return rows[0]
 persist()
 try:
  for entry in plan:
   left=deadline-time.time()
   if left<=120:raise TimeoutError('Original service deadline reserve; no retry or denominator shrink')
   case=cby[entry['case_id']];r=rby[entry['case_id']];e=eby[entry['case_id']];body=r['body'];raw=compact(body)
   assert sha(raw)==r['request_body_SHA256']==entry['request_body_SHA256']==e['request_body_SHA256']and body['stream']is True and body['text_delivery']=='P1'and body['max_tokens']==4096
   assert e['prompt_IDs_SHA256']==entry['prompt_IDs_SHA256']and e['prompt_tokens']==entry['prompt_tokens']
   stem=out/('%04d'%entry['logical_index']);stem.with_suffix('.request.json').write_bytes(raw)
   q['attempted_HTTP']+=1;q['pending_attempt']=dict(entry,request_body_SHA256=sha(raw));persist()
   sent=time.monotonic();wire=b'';status=None
   try:
    request=urllib.request.Request(base_url.rstrip('/')+'/chat/completions',data=raw,headers={'Content-Type':'application/json'})
    try:
     with(open_http or urllib.request.urlopen)(request,timeout=min(7200,left-90))as response:
      status=response.status;assert status!=200 or'text/event-stream'in response.headers.get('Content-Type','');wire=response.read()
    except urllib.error.HTTPError as error:status=error.code;wire=error.read()
   finally:stem.with_suffix('.wire').write_bytes(wire)
   ended=time.monotonic();q['pending_attempt'].update(HTTP_status=status,wire_SHA256=sha(wire),client_wall_s=ended-sent);persist()
   # Even a parser/HTTP failure may only continue when this exact Native owner
   # request has completed its original reset/drain and actual counters.
   row=read_owner(entry)if read_owner is not None else owner_for(entry);coverage=validate(row,entry)
   h=row['admission']['history_render_audit'];assert h['qualified']and h['prequeue_bound']and h['Native_prompt_token_ids_SHA256']==e['prompt_IDs_SHA256']and not h['pending_thinking_prefix_open']
   assert row['admission']['text_delivery']=='P1'and row['final_metadata']['buffered_P0_text']is False
   ids=row['generated_token_ids'];assert all(type(v)is int and 0<=v<129280 for v in ids)
   natural=row['native_finish_reason']=='eos_token';assert(ids[-1]==1 and ids.count(1)==1)if natural else(len(ids)==4096 and 1 not in ids and row['native_finish_reason']=='length')
   ticks=row['backend_token_monotonic'];assert len(ticks)==len(ids)and all(math.isfinite(t)for t in ticks)and all(b>a for a,b in zip(ticks,ticks[1:]))
   content='';tools=False;parse_error=None;usage=None;finish=None;decoded=False
   try:
    assert status==200;message,finish,usage=sse_message(wire);decoded=True
   except(AssertionError,ValueError,KeyError,TypeError)as error:parse_error=repr(error)
   if decoded:
    assert usage==dict(prompt_tokens=e['prompt_tokens'],completion_tokens=len(ids),total_tokens=e['prompt_tokens']+len(ids))
    identifiers={part['id']for line in wire.splitlines()if line.startswith(b'data: ')and line[6:].strip()!=b'[DONE]'for part in [json.loads(line[6:])]if part.get('id')};assert identifiers=={row['request_id']},'Foreign public response identity'
    assert finish==('stop'if natural else'length')or finish=='tool_calls';content=message.get('content')or'';tools=bool(message.get('tool_calls'))
   score=grade(case,content,natural_EOS=natural,cap_hit=not natural,unexpected_tool=tools)
   if parse_error:score.update(correct=False,parse_failure=True)
   q['rows'].append(dict(logical_index=entry['logical_index'],case_id=case['id'],suite=case['suite'],mode=entry['mode'],request_body_SHA256=sha(raw),wire_SHA256=sha(wire),prompt_IDs_SHA256=e['prompt_IDs_SHA256'],prompt_tokens=e['prompt_tokens'],actual_generated_including_EOS=len(ids),generated_IDs_SHA256=sha(compact(ids)),HTTP_status=status,HTTP_finish=finish,public_parse_error=parse_error,client_wall_s=ended-sent,actual_owner_coverage=coverage,**score))
   q['completed_HTTP']+=1;q['pending_attempt']=None;persist()
  assert q['attempted_HTTP']==q['completed_HTTP']==len(plan);q['execution_passed']=True
 except BaseException as error:q.update(error=repr(error),traceback=traceback.format_exc())
 finally:q['completed']=True;persist()
 return q

def main(argv=None):
 p=argparse.ArgumentParser()
 for name in ['cases','cases-sha','requests','requests-sha','encoding','encoding-sha','plan','plan-sha','base-url','runtime-dir','run','source-sha','config-sha','actual-step','output-dir']:p.add_argument('--'+name,required=True)
 p.add_argument('--deadline-unix',type=float,required=True);a=p.parse_args(argv)
 def read(path,h):
  b=Path(path).read_bytes();assert sha(b)==h;return json.loads(b)
 cases=read(a.cases,a.cases_sha)['cases'];assets=read(a.requests,a.requests_sha);encoding=read(a.encoding,a.encoding_sha);plan=read(a.plan,a.plan_sha)
 assert encoding['completed']and encoding['passed']and encoding['actual_encodings']==256 and encoding['actual_Deep_codec']and encoding['actual_public_protocol']and not encoding['GPU']and not encoding['Native_dlopen']
 assert assets['logical_plan']==plan['frozen512']and len(plan['frozen512'])==512
 folder=Path(a.runtime_dir)/a.run;ready=wait_json(folder/'MODEL_READY.json',a.deadline_unix);identity=ready['identity']
 assert identity['run']==a.run and identity['source_manifest_SHA256']==a.source_sha and identity['provider_config_SHA256']==a.config_sha and identity['actual_step']==a.actual_step and ready['Model_READY']and ready['Native_internal_owner_count']==8 and ready['HC_mode']==0
 q=run(cases,assets['requests'],encoding['cases'],plan['frozen512'],plan['assigned'],base_url=a.base_url,folder=folder,identity=identity,deadline=a.deadline_unix,output=a.output_dir)
 return 0 if q['execution_passed']else 1
if __name__=='__main__':raise SystemExit(main())
