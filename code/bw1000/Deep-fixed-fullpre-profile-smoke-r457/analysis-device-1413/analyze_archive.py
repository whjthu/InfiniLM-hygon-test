"""Independent stdlib terminal smoke audit. Raw inputs/wires/IDs remain R/L."""
import argparse,base64,gzip,hashlib,json,math,re
from fractions import Fraction
from pathlib import Path
PUB='b7b0487947b53c31ae4483bcd8cd18cc780bfda9'
MF='f3741395cf413fc8fd2ea5a92ccf295dcdcd1e3489dc1fb16e43e6cff8a61128'
PROVIDER='bc7541b86afd15894d374c774ca3ed5093c623134bc493467518e4b7b3a2a98a'
NATIVE='a314ae6891977316189f875b793ed3f2e2eb0898110ad0cc382d2569eaf31dc1'
LIB='/refactor/candidates/Deep-HC-fullpre-Native-r377-cpu-v1/libDeep_HC_fullpre_optin_r377.so'
PROFILE='deep-hc0-fullpre-c1';EPOCH=1793766297.782444
sha=lambda b:hashlib.sha256(b).hexdigest()
compact=lambda q:json.dumps(q,ensure_ascii=False,separators=(',',':')).encode()
def wire(raw):
 ids=set();content='';reasoning='';tools=False;usage=None;finish=None;done=False
 for line in raw.decode().splitlines():
  if not line or line.startswith(':'):continue
  assert line.startswith('data: ')
  if line[6:]=='[DONE]':done=True;break
  q=json.loads(line[6:]);assert'error'not in q
  if q.get('id'):ids.add(q['id'])
  if q.get('usage')is not None:usage=q['usage']
  for choice in q.get('choices',[]):
   assert choice.get('index',0)==0;delta=choice.get('delta',{});content+=delta.get('content')or'';reasoning+=delta.get('reasoning_content')or'';tools|=bool(delta.get('tool_calls'))
   if choice.get('finish_reason')is not None:finish=choice['finish_reason']
 assert done and len(ids)==1
 return dict(id=next(iter(ids)),content=content,reasoning=reasoning,tools=tools,usage=usage,finish=finish)
def independent_grade(case,text,natural,tools):
 if case['suite']=='gsm8k':
  found=re.findall(r'####[ *$]*([-+]?(?:[0-9][0-9,]*(?:[.][0-9]+)?|[.][0-9]+)(?:/[0-9]+)?)',text)
  try:parsed=Fraction(found[-1].replace(',',''))if found else None
  except(ValueError,ZeroDivisionError):parsed=None
  correct=parsed is not None and parsed==Fraction(case['gold_numeric'])
 else:
  assert case['suite']=='mmlu';found=re.findall(r'(?:^|\n)\s*(?:\*\*)?(?:Answer|Final answer)\s*:\s*[\[(]?([ABCD])[\])]?(?:\*\*)?(?:[.\s]|$)',text,re.I)
  bare=re.fullmatch(r'\s*[\[(]?([ABCD])[\])]?\s*[.]?\s*',text,re.I);parsed=found[-1].upper()if found else bare[1].upper()if bare else None;correct=parsed==case['gold_choice']
 return dict(parsed_answer=str(parsed)if parsed is not None else None,parse_failure=parsed is None,answer_matches_gold=bool(correct),correct=bool(correct and natural and not tools),natural_EOS=natural,cap_hit=not natural,unexpected_tool_intent=tools)
def counts(row,index):
 assert row['r454_index']==index and row['r454_mode']==1 and row['r454_profile']==PROFILE and row['r454_r366_fixed']==0 and row['r454_quiescent']is True and row['r454_poisoned']is False
 proof=row['r454_binding_proof'];assert set(proof['providers'].values())=={LIB}and len(proof['providers'])==7 and proof['HC0_fixed']and proof['fullpre_default']==0 and proof['actual_HC_groups_per_owner']==86 and proof['old366_width']==7 and proof['new377_width']==10
 assert row['normal_reset_before']and row['normal_reset_after']and not row.get('error')and not row.get('reset_error')
 f=row['actual_forward_completed'];assert type(f)is int and f>0 and row['actual_forward_attempts']==f
 p=row['prompt_tokens'];pre=[t for t in row['Native_continuation']if t['kind']=='prefill'];ds=[t for t in row['Native_continuation']if t['kind']=='decode_summary'];n=sum(t['calls']for t in ds);one=sum(t['tokens']==1 for t in pre)
 assert len(ds)<=1 and f==len(pre)+n and len(row['generated_token_ids'])==n+1
 assert[t['past']for t in pre]==list(range(0,p,1024))and[t['tokens']for t in pre]==[min(1024,p-offset)for offset in range(0,p,1024)]and all(t['total']==t['past']+t['tokens']and t['emitted']==(i==len(pre)-1)for i,t in enumerate(pre))
 if ds:assert ds==[dict(kind='decode_summary',calls=n,first_past=p,last_total=p+n)]
 expected=dict(passed=True,HC_groups=86,completed_forwards=f,one_token_prefill_forwards=one,decode_forwards=n,T1_HC_per_owner=86*(one+n),mode=1,r366_fixed=0)
 assert row['r454_actual_T1_coverage']==expected and len(row['r454_actual_owner_counts'])==8
 for i,c in enumerate(row['r454_actual_owner_counts']):
  t1=86*(one+n);assert c['owner']==i and c['created']==c['ready']==86
  assert all(type(v)is int and v>=0 for v in [*c['legacy'].values(),*c['fullpre'].values()])
  assert c['legacy']==dict(created=0,ready=0,pre=86*f,eligible_bf16_t1=t1,completed=0,other_t=86*f-t1,f32_t1=0)
  assert c['fullpre']==dict(observed=86*f,eligible=t1,selected=t1,completed=t1,prefill=86*f-t1,f32_t1=0,layout=0,selected_exceptions=0,fallback_exceptions=0,active=0)
 return expected
def analyze(archive):
 files={}
 for name,item in archive['files'].items():
  raw=base64.b64decode(item['base64'],validate=True);assert len(raw)==item['bytes']and sha(raw)==item['SHA256'];files[name]=raw
 def raw(suffix,required=True):
  values=[v for n,v in files.items()if n.endswith(suffix)];assert len(values)<=1,(suffix,len(values))
  if not values:
   if required:raise AssertionError('Missing original '+suffix)
   return None
  return values[0]
 def read(suffix,required=True):
  value=raw(suffix,required);return json.loads(value)if value is not None else None
 w=read('-watch.json');assert w['completed']and w['actual_GPU']=='910374.1413'and w['CPU_keeper']=='910407.965'and w['original_GPU_start_unix']==EPOCH
 assert w['publication_commit']==PUB and w['deployment_source_SHA256']==MF and w['provider_source_SHA256']==PROVIDER and w['HTTP_planned']==2 and w['warms']==0
 assert [w[k]for k in['service_deadline_unix','Native_deadline_unix','GPU_deadline_unix','keeper_deadline_unix']]==[EPOCH+5100,EPOCH+5280,EPOCH+5400,EPOCH+5700]
 dm=raw('/deployment-gpu-v1/SOURCE_MANIFEST.json');pm=raw('/source-v1/SOURCE_MANIFEST.json');assert sha(dm)==MF and sha(pm)==PROVIDER
 for base,manifest in[('/deployment-gpu-v1/',json.loads(dm)),('/source-v1/',json.loads(pm))]:
  for name,h in manifest['files_SHA256'].items():assert sha(raw(base+name))==h,(base,name)
 config=read('/deployment-gpu-v1/config.json');assert config['execute_authorized']and config['experimental_profile']==PROFILE and config['HC_mode']==0 and config['TP']==8 and config['PP']==config['rows']==1 and config['context']==32768 and config['Native_SHA256']==NATIVE
 card=read('/deployment-gpu-v1/FIXTURE_CARD.json');assert len(card['plan'])==2 and[entry['selected_ordinal']for entry in card['plan']]==[0,128]
 cases_raw=raw('/fixtures-r1/cases.json');assert sha(cases_raw)==card['cases_SHA256'];cases=json.loads(cases_raw)['cases']
 requests_raw=raw('/requests-private.json');encoded_raw=raw('/encoding.json');assert sha(requests_raw)==card['requests_SHA256']and sha(encoded_raw)==card['encoding_SHA256']
 original=json.loads(requests_raw)['requests'];encoded=json.loads(encoded_raw);assert encoded['actual_CPU']=='910407.945'and encoded['completed']and encoded['passed']and encoded['actual_Deep_codec']and not encoded['GPU']
 cpu=raw('/Deep457-smoke-glue-CPU-v1/result.json');assert sha(cpu)=='c66858a7460b51d5cfd9b2b3c3148b2d8972eade9f0daaf142d52d753493a71a';assert json.loads(cpu)['passed']
 for key,expected in[('Native_seconds',5280),('service_seconds',5100)]:assert config[key]==expected
 accounts={row[0]:row for row in archive['accounting']}
 def account(step,node):
  row=accounts.get(step);assert row and row[5]==node;return row[2:4]==['COMPLETED','0:0']
 post=w.get('post')or w.get('post_after_failure');assert post and len(post)==1;p=post[0]
 assert p['KFD_available']is True and p['node']=='a02r4n04'and p['CPU_only']and p['GPU_initialized']is False and not p['kfd_pids']and not p['matching_component_processes']and not p['expected_PIDs_still_exist']and p['ports_bindable']=={'43882':True}and len(set(p['physical_PCI']))==8
 assert account(p['actual_step'],'a02r4n04')
 assert set(p['physical_PCI'])==set(w['fresh_before'][0]['physical_PCI'])and w['fresh_before'][0]['KFD_available']is True
 owner=read('/owner.json',False);ready=read('/MODEL_READY.json',False);closed=read('/closed-engine.json',False);client=read('-client/result.json',False)
 model_config=raw('/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8/config.json');generation=raw('/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8/generation_config.json')
 assert sha(model_config)=='fd2ad6f5dbd77e55a49e9b2ff29c89030fbe4b538e8870c183d194648767be09'and json.loads(model_config)['vocab_size']==129280
 assert sha(generation)=='5fccff80f55a4d455bbe516bdd552edf3e9623df95e99fbf2a3c3389fdf91af0'and json.loads(generation)['eos_token_id']in[1,[1]]
 rows=[];total_calls=0;identity=owner['identity']if owner else None
 if identity:assert identity['actual_step']==w['actual_GPU']and identity['source_manifest_SHA256']==PROVIDER and identity['gpu_start_epoch']==EPOCH
 if ready:
  identity=ready['identity'];assert identity==owner['identity']and identity['actual_step']==w['actual_GPU']and identity['gpu_start_epoch']==EPOCH and identity['source_manifest_SHA256']==PROVIDER and identity['provider_config_SHA256']==sha(raw('/deployment-gpu-v1/config.json'))
  assert ready['Model_READY']and ready['Native_internal_owner_count']==8 and ready['TP']==8 and ready['PP']==1 and ready['HC_mode']==0
  fixed=ready['fixed_profile'];assert fixed==owner['fixed_profile']and fixed['profile']==PROFILE and fixed['fixed_mode']==fixed['Native_mode_readback']==1 and fixed['HC_mode']==fixed['r366_fixed']==0 and fixed['C1']is True and fixed['HC_groups']==86
  assert len(fixed['provider']['providers'])==7 and set(fixed['provider']['providers'].values())=={LIB}
  assert owner['Native_SHA256']==NATIVE and owner['loaded_key_count']==67569 and set(owner['actual_providers_before_weights'].values())=={LIB}
  aff=owner['owner_affinity_after_weight_load']['actual_owner_report']['owners'];masks=[int(n)for row in aff for n in row['verified_mask'].split(',')];assert len(aff)==8 and{r['device']for r in aff}==set(range(8))and len(masks)==len(set(masks))==32 and all(r['changes']==1 for r in aff)
  assert owner['owner_thread_observation']['status']=='OBSERVED'and len({row['TID']for row in owner['owner_thread_observation']['owners']})==8
  engine=read('/ready-engine.json');assert engine['identity']==identity and engine['fixed_profile']==fixed and len(engine['postload_free'])==8 and all(free>=16*1024**3 for free,total in engine['postload_free'])
 if owner:
  for index,row in enumerate(owner['requests']):
   assert index<2;entry=card['plan'][index];case=next(c for c in cases if c['suite']==entry['suite']);enc=next(e for e in encoded['cases']if e['suite']==entry['suite']);request=next(r for r in original if r['case_id']==case['id'])
   assert case['id']==enc['case_id']==entry['case_id']and entry['index']==index
   request_raw=raw('/request-%d.request.json'%index,False);wire_raw=raw('/response-%d.wire'%index,False)
   if request_raw is None or wire_raw is None or not row.get('normal_reset_after')or'final_metadata'not in row:
    rows.append(dict(index=index,case_id=case['id'],complete_request_proof=False,error=row.get('error'),quality='UNKNOWN; incomplete original request'));continue
   assert sha(request_raw)==entry['request_body_SHA256']==request['request_body_SHA256']==enc['request_body_SHA256']and json.loads(request_raw)==request['body']
   body=json.loads(request_raw);assert body['stream']is True and body['text_delivery']=='P1'and body['max_tokens']==4096 and body['chat_template_kwargs']=={'enable_thinking':False}
   h=row['admission']['history_render_audit'];assert h['qualified']and h['prequeue_bound']and not h['markers_sent_to_Native']and h['same_prepared_encoder_reproduces_prompt_and_IDs']and h['content_one_contribution_each_and_ordered']and h['all_content_contribution_slices_equal']and not h['pending_thinking_prefix_open']
   assert h['Native_prompt_token_ids_SHA256']==entry['prompt_IDs_SHA256']==enc['prompt_IDs_SHA256']and h['prompt_tokens']==row['prompt_tokens']==enc['prompt_tokens']and h['output_budget']==row['max_new_tokens']==4096 and row['prompt_tokens']+4096<=32768
   assert row['admission']['text_delivery']=='P1'and row['final_metadata']['buffered_P0_text']is False
   ids=row['generated_token_ids'];assert 1<=len(ids)<=4096 and all(type(n)is int and 0<=n<129280 for n in ids)
   natural=row['native_finish_reason']=='eos_token';assert(ids[-1]==1 and ids.count(1)==1)if natural else(1 not in ids and len(ids)==4096 and row['native_finish_reason']=='length')
   decoded=wire(wire_raw);assert decoded['id']==row['request_id']and decoded['usage']==dict(prompt_tokens=enc['prompt_tokens'],completion_tokens=len(ids),total_tokens=enc['prompt_tokens']+len(ids))and decoded['finish']in(['stop','tool_calls']if natural else['length'])
   coverage=counts(row,index);total_calls+=row['actual_forward_completed'];score=independent_grade(case,decoded['content'],natural,decoded['tools'])
   ticks=row['backend_token_monotonic'];assert len(ticks)==len(ids)and all(math.isfinite(t)for t in ticks)and all(b>a for a,b in zip(ticks,ticks[1:]))and row['started_monotonic']<=ticks[0]<=ticks[-1]<=row['ended_monotonic']
   cr=next((v for v in(client or{}).get('rows',[])if v['index']==index),None)
   if cr:
    for key in['parsed_answer','parse_failure','answer_matches_gold','correct','natural_EOS','cap_hit','unexpected_tool_intent']:assert cr['score'][key]==score[key]
    assert cr['request_id']==row['request_id']and cr['actual_request_SHA256']==sha(request_raw)and cr['wire_SHA256']==sha(wire_raw)and cr['r454_T1_coverage']==coverage
    clock=cr['client_clock'];assert math.isfinite(clock['wall_s'])and clock['wall_s']==clock['completed']-clock['sent']and clock['sent']<=clock['headers']<=clock['completed']and clock['client_per_token_speed'].startswith('UNKNOWN')
   visible=ticks[:-1]if natural else ticks
   rows.append(dict(index=index,case_id=case['id'],suite=case['suite'],complete_request_proof=True,**score,prompt_tokens=enc['prompt_tokens'],generated_including_EOS=len(ids),generated_IDs_SHA256=sha(compact(ids)),request_SHA256=sha(request_raw),wire_SHA256=sha(wire_raw),coverage=coverage,Native_forwards=row['actual_forward_completed'],backend_TTFT_s=ticks[0]-row['started_monotonic'],backend_visible_decode_tps=(len(visible)-1)/(visible[-1]-visible[0])if len(visible)>1 else None,client_per_token_speed='UNKNOWN; chunk is not token'))
 normal=bool(owner and owner.get('completed')and owner.get('normal_Native_shutdown')and owner.get('engine_worker_shutdown_completed')and owner.get('observed_all8_threads_exited')and closed and closed.get('normal')and closed.get('engine_destroy_returned_zero')and closed.get('observed_all8_threads_exited'))
 if normal:assert closed['identity']==identity and owner['HC_dispatch_checked_forwards']==owner['layout_dispatch_checked_forwards']==total_calls
 success=bool(w['passed']and client and client.get('passed')and normal and not w['forced_cleanup']and len(rows)==2 and all(r.get('complete_request_proof')and r.get('correct')for r in rows))
 if success:
  assert client['attempted_HTTP']==client['completed_HTTP']==client['planned_HTTP']==len(client['rows'])==2
  assert account(w['actual_GPU'],'a02r4n04')and account(w['actual_client'],'a02r4n04')and account(w['CPU_keeper'],'a14r2n20')
 return dict(integrity_and_terminal_audit_passed=True,actual_GPU=w['actual_GPU'],actual_keeper=w['CPU_keeper'],actual_client=w.get('actual_client'),original_epoch=EPOCH,verified_raw_files=len(files),planned_requests=2,observed_owner_requests=len(rows),rows=rows,correct=sum(r.get('correct',False)for r in rows),natural_EOS=sum(r.get('natural_EOS',False)for r in rows),missing_or_incomplete=2-sum(r.get('complete_request_proof',False)for r in rows),Native_forwards=total_calls,actual_startup_profile_READY=bool(ready),normal8_destroy_and_thread_exit=normal,post_KFD_PID_port_clean=True,forced_cleanup=w['forced_cleanup'],smoke_passed=success,Model_quality_scope='Only two frozen first-suite cases; not broad quality or NI',new_performance_claim=False,default_promotion=False,production=False,old_failures_393_1374_451_preserved=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('archive',type=Path);a=p.parse_args();raw=a.archive.read_bytes();value=json.loads(gzip.decompress(raw)if raw[:2]==bytes([31,139])else raw);print(json.dumps(analyze(value),indent=2,ensure_ascii=False))
if __name__=='__main__':main()
