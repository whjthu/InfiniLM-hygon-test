"""One immutable18 HTTP plan. Complete external bodies, no retry or restart."""
import argparse,hashlib,json,pathlib,time,urllib.request
from workload_r349 import load_plan,body_and_gold,validate_reply
def main():
 p=argparse.ArgumentParser()
 for n in ['base-url','model','output-dir','identity','service-report','manifest-sha','expected-step']:p.add_argument('--'+n,required=True)
 p.add_argument('--deadline-unix',type=float,required=True);p.add_argument('--timeout',type=float,default=7200);a=p.parse_args()
 from source_pin_r290 import verify,D
 from client_owner_r290 import bind_identity,row,backend_metrics
 root=pathlib.Path(a.output_dir);root.mkdir(exist_ok=False)
 q=dict(completed=False,passed=False,planned_HTTP=2,planned_Native=2,attempted_HTTP=0,completed_HTTP=0,actual_HTTP_requests=0,cases=[],no_retries=True,client_incremental_metrics_qualified=False,all16_owner_audit_required=True)
 def save():
  temp=root/'result.tmp';temp.write_text(json.dumps(q,indent=2));temp.replace(root/'result.json')
 def remaining():
  left=a.deadline_unix-time.time()
  if left<=180:raise TimeoutError('Original residency deadline; no skipped arms/retry')
  return left
 save()
 try:
  remaining();verify(a.manifest_sha);identity=bind_identity(a.identity,a.manifest_sha,a.service_report);assert identity['actual_step']==a.expected_step
  card,plan=load_plan(D/'FIXTURE_CARD.json');assert identity['fixture_request_SHA256']==[f['request_SHA256'] for f in card['fixtures']]
  seen=set();prompt_hashes={}
  for arm in plan:
   f=card['fixtures'][arm['fixture_index']];raw,body,expected=body_and_gold(f);remaining()
   req=urllib.request.Request(a.base_url.rstrip('/')+'/chat/completions',data=raw,headers={'Content-Type':'application/json'})
   q['attempted_HTTP']+=1;q['actual_HTTP_requests']=q['attempted_HTTP'];save();start=time.monotonic()
   with urllib.request.urlopen(req,timeout=min(a.timeout,remaining()-120))as response:status=response.status;wire_raw=response.read()
   wall=time.monotonic()-start;(root/('response-%02d.wire'%arm['global_index'])).write_bytes(wire_raw)
   assert status==200;wire=json.loads(wire_raw);usage=validate_reply(wire,f,expected);assert wire['id']not in seen;seen.add(wire['id'])
   r=row(a.service_report,a,wire['id'],arm['global_index'],body['user'],4096,remaining)
   assert r['native_finish_reason']=='eos_token' and r['generated_natural_EOS']
   for key in ['global_index','fixture_id','fixture_index','arm_index']:assert r[key]==arm[key]
   assert r['pair_index']==arm['global_index'] and r['prefill_control_window']==arm['window'] and r['paired_warmup']==arm['warmup']
   assert r['prompt_tokens']==usage['prompt_tokens'] and r['actual_generated_tokens']==usage['completion_tokens']
   actual=r['actual_prompt_token_ids_SHA256'];assert actual==f['current_prepared_prompt_IDs_SHA256']
   assert prompt_hashes.setdefault(f['id'],actual)==actual
   q['completed_HTTP']+=1;q['cases'].append(dict(**arm,request_id=wire['id'],request_SHA256=f['request_SHA256'],prompt_IDs_SHA256=actual,HTTP_wall_s=wall,usage=usage,wire_SHA256=hashlib.sha256(wire_raw).hexdigest(),backend=backend_metrics(r),natural_EOS=True,semantic_gold_passed=True,passed=True));save()
  assert q['attempted_HTTP']==q['completed_HTTP']==len(q['cases'])==2
  assert all(sum(c['fixture_id']==f['id'] for c in q['cases'])==1 for f in card['fixtures'])
  q.update(passed=True,per_fixture_prompt_IDs_SHA256=prompt_hashes,matched_pair_conditions_validated=True)
 except BaseException as error:q['error']=repr(error)
 finally:q['completed']=True;save()
 return 0 if q['passed']else 1
if __name__=='__main__':raise SystemExit(main())
