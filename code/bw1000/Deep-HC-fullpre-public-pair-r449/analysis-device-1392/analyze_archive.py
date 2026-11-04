"""Stdlib independent original public request/wire/Native terminal audit; R/L raw only."""
import argparse,hashlib,json,math,re,statistics
from pathlib import Path
sha=lambda raw:hashlib.sha256(raw).hexdigest()
def compact(q):return json.dumps(q,ensure_ascii=False,separators=(',',':'),sort_keys=True).encode()
def unique(pairs):
 q={}
 for k,v in pairs:
  if k in q:raise ValueError('Duplicate JSON key')
  q[k]=v
 return q
def parsed_final(text):
 text=text.strip();presentation='bare_JSON'
 if text.startswith('```'):
  m=re.fullmatch(r'```json[ \t]*\r?\n([\s\S]*?)\r?\n```',text);assert m,'Require one complete JSON or entire json fence';text=m.group(1);presentation='whole_json_fence'
 def reject(v):raise ValueError('Nonfinite JSON')
 q=json.loads(text,object_pairs_hook=unique,parse_constant=reject);assert isinstance(q,dict)
 return q,presentation
def sse(raw):
 message={'content':'','reasoning_content':''};identifier=set();usage=None;finish=None;done=False
 for line in raw.decode().splitlines():
  if not line or line.startswith(':'):continue
  assert line.startswith('data: ');text=line[6:]
  if text=='[DONE]':done=True;break
  q=json.loads(text);assert'error'not in q
  if q.get('id'):identifier.add(q['id'])
  if q.get('usage')is not None:usage=q['usage']
  for c in q.get('choices',[]):
   assert c.get('index',0)==0;delta=c.get('delta',{});assert not delta.get('tool_calls')
   for k in message:message[k]+=delta.get(k)or''
   if c.get('finish_reason')is not None:finish=c['finish_reason']
 assert done and len(identifier)==1 and finish in ['stop','length']
 return dict(id=next(iter(identifier)),choices=[dict(message=message,finish_reason=finish)],usage=usage)
def gold(body):
 calls={};expected={};pairs=[]
 for m in body['messages']:
  if m['role']=='assistant':
   for c in m.get('tool_calls',[]):
    assert c['id']not in calls and c['function']['name']=='read_file';calls[c['id']]=json.loads(c['function']['arguments'])['path']
  if m['role']=='tool':
   name=calls[m['tool_call_id']];d=json.loads(m['content']);vals={v['id']:v['value']for v in d['records']};assert len(vals)==len(d['records'])
   selected=[vals[k]for k in d['query_ids']];assert len(selected)==4 and all(type(v)is int for v in selected)
   expected[name]=dict(nonce=d['nonce'],sum=sum(selected));pairs.append(m['tool_call_id'])
 assert set(expected)=={'left.json','right.json'}and pairs==list(reversed(calls))
 expected['total']=expected['left.json']['sum']+expected['right.json']['sum'];return expected
def percentile(xs,p):
 ys=sorted(xs);return ys[min(len(ys)-1,max(0,math.ceil(p*len(ys))-1))]if ys else None
def clock_metrics(clock,wire_raw):
 rows=[]
 for line in wire_raw.splitlines(keepends=True):
  if not line.startswith(b'data: '):continue
  value=line[6:].strip();flags=dict(reasoning_nonempty=False,content_nonempty=False,tool_nonempty=False)
  if value!=b'[DONE]':
   q=json.loads(value)
   for name,key in [('reasoning_nonempty','reasoning_content'),('content_nonempty','content'),('tool_nonempty','tool_calls')]:flags[name]=any(bool(v.get('delta',{}).get(key))for v in q.get('choices',[]))
  flags['nonempty_delta']=any(flags.values());flags['bytes']=len(line);rows.append(flags)
 assert len(rows)==len(clock['events'])
 sent,headers,ended=clock['sent'],clock['headers'],clock['completed'];assert all(math.isfinite(x)for x in [sent,headers,ended])and sent<=headers<=ended
 for flags,event in zip(rows,clock['events']):
  assert all(event[k]==v for k,v in flags.items());assert headers<=event['arrival_monotonic']<=ended
 ticks=[e['arrival_monotonic']for e in clock['events']];assert ticks==sorted(ticks)
 semantic=[e['arrival_monotonic']for e in clock['events']if e['nonempty_delta']]
 first=lambda key:next((e['arrival_monotonic']-sent for e in clock['events']if e[key]),None)
 gap=lambda xs:max((b-a for a,b in zip(xs,xs[1:])),default=None)
 computed=dict(wall_s=ended-sent,headers_s=headers-sent,first_payload_delta_s=first('nonempty_delta'),first_reasoning_payload_s=first('reasoning_nonempty'),first_content_payload_s=first('content_nonempty'),first_tool_payload_s=first('tool_nonempty'),max_inter_data_frame_gap_s=gap(ticks),max_inter_semantic_delta_gap_s=gap(semantic),semantic_delta_events=len(semantic))
 for k,v in computed.items():assert clock[k]==v,(k,clock[k],v)
 assert computed['first_payload_delta_s']is not None  and computed['first_content_payload_s']is not None
 assert clock['client_per_token_speed'].startswith('UNKNOWN')
 return dict(**computed,data_frames=len(rows),client_per_token_speed='UNKNOWN; delta/frame is not a token')

def owner_coverage(row,index):
 mode=[0,1,0,1,1,0][index]
 assert row['r449_index']==index and row['r449_mode']==mode and row['r449_warm']==(index<2)and row['r449_r366_fixed']==0
 assert row['r449_quiescent']is True and row['r449_poisoned']is False
 proof=row['r449_binding_proof'];names={'deep_hc_fullpre_'+n+'_r377'for n in ['enable','enabled','clear','stats']}|{'deep_hc_model_epilogue_'+n+'_r366'for n in ['set','get','counters']}
 assert set(proof['providers'])==names and set(proof['providers'].values())=={'/refactor/candidates/Deep-HC-fullpre-Native-r377-cpu-v1/libDeep_HC_fullpre_optin_r377.so'}
 assert proof['HC0_fixed']and proof['fullpre_default']==0 and proof['actual_HC_groups_per_owner']==86 and proof['old366_width']==7 and proof['new377_width']==10 and proof['HostReturned_is_not_GPU_completed']
 f=row['actual_forward_completed'];n=len(row['generated_token_ids'])-1;t1=86*n
 expected=dict(passed=True,HC_groups=86,completed_forwards=f,one_token_prefill_forwards=0,decode_forwards=n,T1_HC_per_owner=t1,mode=mode,r366_fixed=0)
 assert row['r449_actual_T1_coverage']==expected and t1>0
 assert len(row['r449_actual_owner_counts'])==8
 for owner,c in enumerate(row['r449_actual_owner_counts']):
  assert c['owner']==owner and c['created']==c['ready']==86
  a,b=c['legacy'],c['fullpre'];assert all(type(v)is int and v>=0 for v in [*a.values(),*b.values()])
  assert a==dict(created=0,ready=0,pre=86*f,eligible_bf16_t1=t1,completed=0,other_t=86*(f-n),f32_t1=0)
  assert b==dict(observed=mode*86*f,eligible=mode*t1,selected=mode*t1,completed=mode*t1,prefill=mode*86*(f-n),f32_t1=0,layout=0,selected_exceptions=0,fallback_exceptions=0,active=0)
 return expected

def analyze(archive):
 files=archive['files'];verified=0
 def raw_by(suffix):
  matches=[v for n,v in files.items()if n.endswith(suffix)];assert len(matches)==1,(suffix,len(matches));v=matches[0]
  raw=v['text'].encode()if v.get('encoding','text')=='text'else __import__('base64').b64decode(v['base64']);assert len(raw)==v['bytes']and sha(raw)==v['SHA256'];return raw
 for v in files.values():
  raw=v['text'].encode()if v.get('encoding','text')=='text'else __import__('base64').b64decode(v['base64']);assert len(raw)==v['bytes']and sha(raw)==v['SHA256'];verified+=1
 read=lambda suffix:json.loads(raw_by(suffix))
 w=read('-watch.json');assert w['completed']and w['actual_GPU']=='910374.1392'and w['CPU_keeper']=='910407.940'
 assert w['publication_commit']=='ffd70f06cab1ca342fd332347c2ea30049931443'and w['deployment_source_SHA256']=='6edc80c82fe8a279ff53417969685967b7aa7e221b89a9ff76b809a2a74d3bc9'
 assert w['provider_source_SHA256']=='03021a3669848d8e4c494679999d119bc7a3eba15d6556fdddc39c2782c262bd'and w['original_GPU_start_unix']==1793747385.9875135
 client=read('-client/result.json');owner=read('/owner.json');front=read('/FRONTEND.json');ready=read('/MODEL_READY.json')
 assert client['completed']and client['planned_HTTP']==6 and ready['Model_READY']and ready['Native_internal_owner_count']==8 and ready['HC_mode']==0 and ready['TP']==8 and ready['PP']==1
 identity=ready['identity'];assert front['identity']==owner['identity']==identity and identity['actual_step']==w['actual_GPU']and identity['source_manifest_SHA256']==w['provider_source_SHA256']
 assert owner['loaded_key_count']==67569 and owner['Native_SHA256']=='a314ae6891977316189f875b793ed3f2e2eb0898110ad0cc382d2569eaf31dc1'and owner['HC_mode']==0
 expected_provider='/refactor/candidates/Deep-HC-fullpre-Native-r377-cpu-v1/libDeep_HC_fullpre_optin_r377.so'
 assert set(owner['actual_providers_before_weights'].values())=={expected_provider}
 aff=owner['owner_affinity_after_weight_load']['actual_owner_report']['owners'];assert {v['device']for v in aff}==set(range(8))
 mask=[int(x)for v in aff for x in v['verified_mask'].split(',')];assert len(mask)==len(set(mask))==32 and all(v['changes']==1 for v in aff)
 assert owner['owner_thread_observation']['status']=='OBSERVED'and len({v['TID']for v in owner['owner_thread_observation']['owners']})==8
 raw=raw_by('/frozen-toolprompt.json');assert sha(raw)=='43377e6a70efdf8960989559c957241f9e725f8a2c2a55b9bf438c331275cb6d';fixture=json.loads(raw)
 raw=raw_by('/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8/config.json');assert sha(raw)=='fd2ad6f5dbd77e55a49e9b2ff29c89030fbe4b538e8870c183d194648767be09';vocab=json.loads(raw)['vocab_size']
 raw=raw_by('/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8/generation_config.json');assert sha(raw)=='5fccff80f55a4d455bbe516bdd552edf3e9623df95e99fbf2a3c3389fdf91af0'and json.loads(raw)['eos_token_id']in[1,[1]]
 rows=[];allids=[];requesthash=[];calls=0
 for i,r in enumerate(owner['requests']):
  assert i<6
  raw=raw_by('/request-%d.request.json'%i);body=json.loads(raw);requesthash.append(sha(raw))
  expected_body=dict(fixture['body'],stream=True,max_tokens=128,text_delivery='P1',chat_template_kwargs={'enable_thinking':False},stream_options={'include_usage':True});assert body==expected_body
  wire_raw=raw_by('/response-%d.wire'%i);wire=sse(wire_raw);expected=gold(body);assert expected==fixture['expected_answer']
  try:actual,presentation=parsed_final(wire['choices'][0]['message']['content'])
  except(ValueError,AssertionError):actual=None;presentation='invalid_final_object'
  correct=bool(actual==expected and isinstance(actual,dict)and type(actual.get('total'))is int and all(type(actual.get(k,{}).get('sum'))is int for k in ['left.json','right.json']))
  h=r['admission']['history_render_audit'];assert all(h[k]for k in ['qualified','prequeue_bound','same_prepared_encoder_reproduces_prompt_and_IDs','content_one_contribution_each_and_ordered','all_content_contribution_slices_equal'])and not h['markers_sent_to_Native']
  assert h['prompt_tokens']==r['prompt_tokens']==8489 and h['Native_prompt_token_ids_SHA256']=='681977ee292994eac340bae5c1aadf7a6b136890df725d02bb75fc8f31ecf0df'and h['output_budget']==r['max_new_tokens']==128 and not h['pending_thinking_prefix_open']
  assert r['admission']['text_delivery']=='P1'and r['final_metadata']['buffered_P0_text']is False
  ids=r['generated_token_ids'];allids.append(ids);assert 1<len(ids)<=128 and all(type(t)is int and 0<=t<vocab for t in ids)
  natural=ids[-1]==1 and ids.count(1)==1 and r['native_finish_reason']=='eos_token'
  assert wire['usage']==dict(prompt_tokens=8489,completion_tokens=len(ids),total_tokens=8489+len(ids))and wire['id']==r['request_id']and len(ids)==r['actual_generated_tokens']
  assert r['normal_reset_before']and r['normal_reset_after']and not r.get('error')and not r.get('reset_error')
  f=r['actual_forward_completed'];assert r['actual_forward_attempts']==f==9+len(ids)-1;calls+=f
  pre=[v for v in r['Native_continuation']if v['kind']=='prefill'];ds=[v for v in r['Native_continuation']if v['kind']=='decode_summary']
  assert len(pre)==9 and[v['tokens']for v in pre]==[1024]*8+[297]and all(v['past']==1024*j and v['total']==v['past']+v['tokens']and v['emitted']==(j==8)for j,v in enumerate(pre))
  assert ds==[dict(kind='decode_summary',calls=len(ids)-1,first_past=8489,last_total=8489+len(ids)-1)]
  coverage=owner_coverage(r,i);ticks=r['backend_token_monotonic'];assert len(ticks)==len(ids)and all(math.isfinite(t)for t in ticks)and all(b>a for a,b in zip(ticks,ticks[1:]))and r['started_monotonic']<=ticks[0]<=ticks[-1]<=r['ended_monotonic']
  ns=r['Native_forward_host_summary'];assert ns['prefill_calls']==9 and ns['decode_calls']==len(ids)-1 and all(math.isfinite(ns[k])and ns[k]>0 for k in ['prefill_s','decode_s'])
  clientrows=[c for c in client['rows']if c['index']==i]
  if clientrows:
   assert len(clientrows)==1;c=clientrows[0];assert c['r449_index']==i and c['mode']==r['r449_mode']and c['r449_T1_coverage']==coverage and c['actual_owner_counts']==r['r449_actual_owner_counts']and c['actual_request_SHA256']==sha(raw)and c['wire_SHA256']==sha(wire_raw)
   clock=clock_metrics(c['client_clock'],wire_raw);assert correct and natural
  else:
   c=client['current_attempt'];assert c['index']==i and c['request_SHA256']==sha(raw)and c['wire_SHA256']==sha(wire_raw)
   clock=dict(wall_s=c['completed']-c['sent'],headers_s=c['headers']-c['sent'],terminal_quality_rejected=True)
  visible=ticks[:-1]if natural else ticks
  rows.append(dict(index=i,mode=r['r449_mode'],warm=i<2,semantic_gold=correct,natural_EOS=natural,output_cap_or_nonEOS=not natural,generated_including_EOS=len(ids),Native_forwards=f,complete_Native_prefill_s=ns['prefill_s'],complete_Native_decode_s=ns['decode_s'],complete_Native_total_s=ns['prefill_s']+ns['decode_s'],Native_decode_mean_s=ns['decode_s']/(len(ids)-1),backend_TTFT_s=ticks[0]-r['started_monotonic'],backend_visible_decode_tok_s=(len(visible)-1)/(visible[-1]-visible[0])if len(visible)>1 else None,generated_IDs_SHA256=sha(compact(ids)),request_SHA256=sha(raw),wire_SHA256=sha(wire_raw),coverage=coverage,client_clock=clock,presentation=presentation,answer_differences=[k for k in expected if actual is None or actual.get(k)!=expected[k]]))
 assert len(set(requesthash))==1 and owner['HC_dispatch_checked_forwards']==owner['layout_dispatch_checked_forwards']==calls
 normal=bool(owner.get('completed')and owner.get('normal_Native_shutdown')and owner.get('engine_worker_shutdown_completed')and owner.get('observed_all8_threads_exited'))
 if normal:
  closed=read('/closed-engine.json');assert closed['identity']==identity and closed['normal']and closed['engine_destroy_returned_zero']and closed['observed_all8_threads_exited']
 post=w.get('post')or w.get('post_after_failure');assert len(post)==1 and post[0]['node']=='a02r4n04'and not post[0]['kfd_pids']and not post[0]['matching_component_processes']and not post[0]['expected_PIDs_still_exist']and post[0]['ports_bindable']=={'43882':True}and len(set(post[0]['physical_PCI']))==8
 accounts={r[0]:r for r in archive['accounting']};assert w['actual_GPU']in accounts and w['CPU_keeper']in accounts and post[0]['actual_step']in accounts
 assert accounts[post[0]['actual_step']][2:4]==['COMPLETED','0:0']
 success=bool(w['passed']and client['passed']and normal and not w['forced_cleanup']and len(rows)==6 and all(r['semantic_gold']and r['natural_EOS']for r in rows))
 if success:
  assert client['attempted_HTTP']==client['completed_HTTP']==6 and len(client['rows'])==6
  for step in [w['actual_GPU'],w['CPU_keeper'],w['actual_client']]:assert accounts[step][2:4]==['COMPLETED','0:0']
 pairs=[];drift={}
 if success:
  for key in ['complete_Native_decode_s','Native_decode_mean_s','complete_Native_prefill_s','complete_Native_total_s','backend_TTFT_s']:
   ratios=[rows[on][key]/rows[off][key]for off,on in [(2,3),(5,4)]]
   offdrift=rows[5][key]/rows[2][key]-1
   drift[key]=dict(ON_over_OFF_adjacent=ratios,median_ON_over_OFF=statistics.median(ratios),OFF_endpoint_relative_drift=offdrift,all_pairs_faster=all(v<1 for v in ratios),change_exceeds_abs_OFF_drift=abs(statistics.median(ratios)-1)>abs(offdrift))
  for off,on in [(2,3),(5,4)]:pairs.append(dict(OFF=off,ON=on,OFF_generated=rows[off]['generated_including_EOS'],ON_generated=rows[on]['generated_including_EOS'],decode_mean_ratio=rows[on]['Native_decode_mean_s']/rows[off]['Native_decode_mean_s'],client_whole_ratio=rows[on]['client_clock']['wall_s']/rows[off]['client_clock']['wall_s']))
 comparisons=[]
 for i,j in [(0,2),(1,3),(3,4),(2,5),(2,3),(4,5)]:
  if j>=len(allids):continue
  a,b=allids[i],allids[j];first=next((dict(output_index=k,IDs=[x,y])for k,(x,y)in enumerate(zip(a,b))if x!=y),None)
  comparisons.append(dict(pair=[i,j],IDs_equal=a==b,first_different=first,lengths=[len(a),len(b)]))
 return dict(passed=True,audit_scope='Original complete currentHC0 sameNative fullpre010110 public pair',execution_passed=success,quality_passed=success,Model_promotion=False,actual_GPU=w['actual_GPU'],CPU_keeper=w['CPU_keeper'],actual_client=w['actual_client'],original_GPU_epoch=w['original_GPU_start_unix'],verified_raw_files=verified,planned_HTTP=6,attempted_HTTP=client['attempted_HTTP'],completed_client_HTTP=client['completed_HTTP'],captured_Native_requests=len(rows),correct_HTTP=sum(r['semantic_gold']for r in rows),incorrect_HTTP=sum(not r['semantic_gold']for r in rows),unexecuted_HTTP=list(range(len(rows),6)),full_ABBA_completed=success,normal8owner_destroy=normal,forced_cleanup=w['forced_cleanup'],post_clean=True,Native_CAPI_forwards=calls,loading_seconds=owner['loading_seconds'],requests=rows,formal_adjacent_pairs=pairs,formal_drift=drift,generated_ID_comparisons=comparisons,all8owner_actual86HC_and_T1_fullpre_verified=True,r366_fixed=0,fullpre_T1_fusion_not_prompt_chunk_change=True,precision_or_capture_change=False,finite_scope='Actual everyforward BF16 finite check; no independently captured fullactivations',timing_scope='Perrequest completeNative host enclosure, emitter and separate clientrelative clocks; different outputlengths retained; no chunk-as-token or online/fullbenchmarkSLO',old_failures=['393qualityFAIL','1374qualityFAIL'],decision='NO_AUTOMATIC_PROMOTION; rootconsumes wholeModel quality/timing/drift'if success else'STOP_NO_PROMOTION_NO_RELOAD')

def main():
 import gzip
 p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('--output',type=Path);a=p.parse_args();blob=a.archive.read_bytes();raw=gzip.decompress(blob)if blob[:2]==b'\x1f\x8b'else blob;q=json.loads(raw);out=analyze(q);out.update(archive_SHA256=sha(blob),uncompressed_archive_SHA256=sha(raw));text=json.dumps(out,indent=2,sort_keys=True)+'\n'
 if a.output:a.output.write_bytes(text.encode())
 print(text,end='')
if __name__=='__main__':main()
