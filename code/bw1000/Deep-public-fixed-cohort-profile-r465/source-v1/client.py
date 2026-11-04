"""Actual finite32-case/64logical HTTP producer; no allgold execution gate."""
from pathlib import Path
import argparse,concurrent.futures,copy,hashlib,json,threading,time,urllib.request
from scoring import grade
from public_deep_cohort.profile import NAME
H=lambda b:hashlib.sha256(b).hexdigest()
def main(argv=None):
 p=argparse.ArgumentParser()
 for name in['private-plan','base-url','output','runtime-dir','run','expected-step','source-sha','config-sha']:p.add_argument('--'+name,required=True)
 p.add_argument('--deadline-unix',required=True,type=float);a=p.parse_args(argv);out=Path(a.output);out.mkdir(exist_ok=False);plan=json.loads(Path(a.private_plan).read_bytes());records=plan['records']
 assert len(records)==32 and len({v['case_id']for v in records})==32 and plan['HTTP']==64
 result=dict(completed=False,execution_passed=False,quality_passed=None,planned_HTTP=64,planned_cases=32,rows=[],cohorts=[],retry=False,Model_READY=False)
 def save():
  t=out/'result.tmp';t.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');t.replace(out/'result.json')
 def wait_record(name,check):
  while time.time()<a.deadline_unix:
   try:q=json.loads((Path(a.runtime_dir)/a.run/name).read_bytes())
   except(FileNotFoundError,json.JSONDecodeError):time.sleep(.05);continue
   identity=q['identity'];assert(identity['run'],identity['actual_step'],identity['source_manifest_SHA256'],identity['provider_config_SHA256'])==(a.run,a.expected_step,a.source_sha,a.config_sha)
   if check(q):return q
   time.sleep(.05)
  raise TimeoutError('Original client deadline waiting '+name)
 save()
 try:
  ready=wait_record('MODEL_READY.json',lambda q:q['Model_READY']);f=ready['fixed_profile']
  assert ready['Native_internal_owner_count']==8 and ready['rows']==4 and ready['capacity_per_row']==32768 and ready['standard_public_C1']is False and f['profile']==NAME and f['fullpre']==f['precision_policy']==0 and f['C1']is False
  result['Model_READY']=True
  for group in range(8):
   chosen=records[group*4:group*4+4];modes=['serial','cohort']if group%2==0 else['cohort','serial']
   for mode in modes:
    identifier=a.run+'-g%d-'%group+mode;barrier=threading.Barrier(4);start=time.monotonic()
    def request(row):
     v=chosen[row];body=copy.deepcopy(v['body']);assert body['max_tokens']==4096 and body['chat_template_kwargs']['enable_thinking']is False
     body['native_fixed_cohort']=dict(id=identifier,mode=mode,row=row,size=4);raw=json.dumps(body,ensure_ascii=False,separators=(',',':')).encode();stem='g%d-%s-r%d'%(group,mode,row)
     (out/(stem+'.request.json')).write_bytes(raw);barrier.wait(timeout=min(30,max(.01,a.deadline_unix-time.time())));sent=time.monotonic();ticks=[];payloads=[]
     with urllib.request.urlopen(urllib.request.Request(a.base_url.rstrip('/')+'/chat/completions',raw,headers={'Content-Type':'application/json'}),timeout=max(.01,a.deadline_unix-time.time()))as response:
      assert response.status==200;headers=time.monotonic()
      if body.get('stream'):
       chunks=[];content='';reasoning='';finish=None;usage=None;rid=None
       for line in response:
        tick=time.monotonic();chunks.append(line)
        if not line.startswith(b'data: '):continue
        value=line[6:].strip()
        if value==b'[DONE]':continue
        event=json.loads(value);rid=event.get('id',rid);usage=event.get('usage')or usage
        for c in event.get('choices',[]):
         d=c.get('delta',{});content+=d.get('content')or'';reasoning+=d.get('reasoning_content')or''
         if d.get('content')or d.get('reasoning_content'):ticks.append(tick)
         assert not d.get('tool_calls');finish=c.get('finish_reason')or finish
       wire=b''.join(chunks);message=dict(content=content,reasoning_content=reasoning);assert rid and usage and finish
      else:
       wire=response.read();q=json.loads(wire);rid=q['id'];message=q['choices'][0]['message'];finish=q['choices'][0]['finish_reason'];usage=q['usage'];ticks=[]
     ended=time.monotonic();(out/(stem+'.wire')).write_bytes(wire);natural=finish=='stop';score=grade(v['gold'],message.get('content')or'',natural_EOS=natural,cap_hit=finish=='length',unexpected_tool=bool(message.get('tool_calls')))
     return dict(case_id=v['case_id'],ordinal=v['ordinal'],cohort=identifier,group=group,row=row,mode=mode,request_id=rid,request_SHA256=H(raw),original_body_SHA256=v['body_SHA256'],wire_SHA256=H(wire),usage=usage,score=score,client_clock=dict(sent=sent,headers=headers,completed=ended,wall_s=ended-sent,semantic_chunk_ticks=ticks,per_token_rate='UNKNOWN; HTTP chunks are not tokens'))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4)as pool:rows=list(pool.map(request,range(4)))
    ids={r['request_id']for r in rows};owner=wait_record('owner.json',lambda q:ids<={r['request_id']for r in q['requests']})
    for row,v in zip(rows,chosen):
     matching=[r for r in owner['requests']if r['request_id']==row['request_id']];assert len(matching)==1;r=matching[0];meta=r['final_metadata']
     assert r['normal_reset_after']and meta['fixed_cohort_id']==identifier and meta['fixed_cohort_mode']==mode and meta['real_cache_rows']==4
     assert meta['history_render_audit']['Native_prompt_token_ids_SHA256']==v['prompt_IDs_SHA256']
     generated=r['generated_token_ids'];assert 0<len(generated)<=4096 and all(type(x)is int and 0<=x<129280 for x in generated)
     assert len(generated)==row['usage']['completion_tokens']and row['usage']['prompt_tokens']==len(v['prompt_token_ids'])
     assert(generated[-1]==1)==(r['native_finish_reason']=='EOS');row['Native_generated_tokens']=len(generated)
    result['rows']+=rows;result['cohorts'].append(dict(id=identifier,mode=mode,group=group,whole_client_seconds=time.monotonic()-start));save()
  assert len(result['rows'])==64 and len({(r['case_id'],r['mode'])for r in result['rows']})==64
  result.update(completed=True,execution_passed=True,quality_summary={mode:dict(planned=32,correct=sum(r['score']['correct']for r in result['rows']if r['mode']==mode),natural_EOS=sum(r['score']['natural_EOS']for r in result['rows']if r['mode']==mode),caps=sum(r['score']['cap_hit']for r in result['rows']if r['mode']==mode),parse_failures=sum(r['score']['parse_failure']for r in result['rows']if r['mode']==mode))for mode in['serial','cohort']},quality_scope='32 fixed paired cases, not broad noninferiority; qualitywrong never stops normal cleanup');save()
 except BaseException as error:result.update(completed=True,error=repr(error));save();raise
 return result
if __name__=='__main__':main()
