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
 assert done and len(identifier)==1 and finish=='stop'
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
 mode=[0,1,0,1,1,0][index];assert row['r438_index']==index and row['r438_mode']==mode and row['r438_warm']==(index<2)
 assert row['r438_quiescent']is True and row['r438_poisoned']is False
 binding=row['r438_binding_proof'];assert set(binding['providers'])=={'deep_model_actquant_'+name+'_r438'for name in ['set','clear','read','metadata']}
 assert set(binding['providers'].values())=={'/refactor/candidates/Deep-T1-MoE-activation-quant-r438-model-cpu-v1/libDeep_T1_MoE_actquant_Model_r438.so'}
 meta=binding['metadata'];assert meta['owners']==8 and meta['counter_width']==12 and meta['default_mode']==0 and not meta['default_armed']and meta['HostReturned_is_not_GPU_completed']and meta['whole_engine_quiescence_required_for_set_clear']
 calls=row['actual_forward_completed'];decodes=len(row['generated_token_ids'])-1;t1=43*decodes
 expected=dict(passed=True,MoE_layers=43,completed_forwards=calls,one_token_prefill_forwards=0,decode_forwards=decodes,T1_per_owner=t1,mode=mode)
 assert row['r438_actual_T1_coverage']==expected and t1>0 and len(row['r438_actual_owner_counts'])==8
 for c in row['r438_actual_owner_counts']:
  assert all(type(v)is int and v>=0 for v in c.values())
  assert c['entered_valid_MoE']==c['host_returned_not_GPU_completed']==43*calls and c['T1']==t1 and c['armed_mode1_T1']==mode*t1 and c['last_mode']==mode and c['unwound']==0
  for role in ['routed','shared']:assert c[role+'_hits']==mode*t1 and c[role+'_calls']==c[role+'_hits']+c[role+'_fallback']
 return expected

def analyze(archive):
 files=archive['files'];verified=0
 def raw_by(suffix):
  values=[v for n,v in files.items()if n.endswith(suffix)];assert len(values)==1,(suffix,len(values));v=values[0]
  raw=v['text'].encode()if v.get('encoding','text')=='text'else __import__('base64').b64decode(v['base64']);assert len(raw)==v['bytes']and sha(raw)==v['SHA256'];return raw
 for v in files.values():
  raw=v['text'].encode()if v.get('encoding','text')=='text'else __import__('base64').b64decode(v['base64']);assert len(raw)==v['bytes']and sha(raw)==v['SHA256'];verified+=1
 read=lambda suffix:json.loads(raw_by(suffix));watch=read('-watch.json');client=read('/result.json');owner=read('/owner.json')
 assert watch['completed']and watch['passed']is False and watch['forced_cleanup']is True
 assert watch['publication_commit']=='d400924076316c9bc2a6908f9eeea8cf52befcc8'and watch['deployment_source_SHA256']=='078dd5aff3e56b1bfa5c0a27d9a24c297682e06b67cc5eb88b24e50e4321589b'
 assert watch['actual_GPU']=='910374.1374'and watch['CPU_keeper']=='910407.906'and watch['actual_client']=='910374.1375'
 assert client['completed']and client['passed']is False and client['planned_HTTP']==6 and client['attempted_HTTP']==4 and client['completed_HTTP']==3 and len(client['rows'])==3
 assert 'fixture.py'in client['error_traceback']and 'actual==expected'in client['error_traceback']and client['current_attempt']['index']==3
 front,ready=read('/FRONTEND.json'),read('/MODEL_READY.json');identity=front['identity'];assert identity==owner['identity']==ready['identity']and identity['actual_step']==watch['actual_GPU']and identity['source_manifest_SHA256']=='fa63f89b81492e8f5908939d79e7f3357b2f80530656866cc3cb324ef315c827'
 assert ready['Model_READY']and ready['Native_internal_owner_count']==8 and owner['loaded_key_count']==67569 and owner['HC_mode']==0
 assert owner['Native_SHA256']=='e3e39de9524af7c19f6bf757236e3288c4bd7647909b2ad14f3e3ebf6d1426e9'
 expected_provider='/refactor/candidates/Deep-T1-MoE-activation-quant-r438-model-cpu-v1/libDeep_T1_MoE_actquant_Model_r438.so'
 assert set(owner['actual_providers_before_weights'].values())=={expected_provider}
 assert not owner['completed']and owner['normal_Native_shutdown']is False and not owner.get('engine_worker_shutdown_completed')and not owner.get('observed_all8_threads_exited')
 assert not any(n.endswith('/closed-engine.json')for n in files),'Do not infer normalclose from clean post'
 aff=owner['owner_affinity_after_weight_load']['actual_owner_report']['owners'];assert {v['device']for v in aff}==set(range(8))
 masks=[int(n)for v in aff for n in v['verified_mask'].split(',')];assert len(masks)==len(set(masks))==32 and all(v['changes']==1 for v in aff)
 assert owner['owner_thread_observation']['status']=='OBSERVED'and len({v['TID']for v in owner['owner_thread_observation']['owners']})==8
 fixture_raw=raw_by('/frozen-toolprompt.json');assert sha(fixture_raw)=='43377e6a70efdf8960989559c957241f9e725f8a2c2a55b9bf438c331275cb6d';fixture=json.loads(fixture_raw)
 cfg_raw=raw_by('/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8/config.json');assert sha(cfg_raw)=='fd2ad6f5dbd77e55a49e9b2ff29c89030fbe4b538e8870c183d194648767be09';cfg=json.loads(cfg_raw);vocab=cfg['vocab_size']
 gen_raw=raw_by('/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8/generation_config.json');assert sha(gen_raw)=='5fccff80f55a4d455bbe516bdd552edf3e9623df95e99fbf2a3c3389fdf91af0';gen=json.loads(gen_raw);assert gen['eos_token_id']in[1,[1]]
 assert len(owner['requests'])==4;rows=[];allids=[];request_hashes=[];calls=0;objects=[]
 for i,r in enumerate(owner['requests']):
  raw=raw_by('/request-%d.request.json'%i);body=json.loads(raw);request_hashes.append(sha(raw))
  expected_body=dict(fixture['body'],stream=True,max_tokens=128,text_delivery='P1',chat_template_kwargs={'enable_thinking':False},stream_options={'include_usage':True});assert body==expected_body
  expected=gold(body);assert expected==fixture['expected_answer'];wire_raw=raw_by('/response-%d.wire'%i);wire=sse(wire_raw);actual,presentation=parsed_final(wire['choices'][0]['message']['content']);objects.append(actual)
  assert set(actual)==set(expected)and type(actual['total'])is int
  assert all(set(actual[k])=={'sum','nonce'}and type(actual[k]['sum'])is int for k in ['left.json','right.json'])
  correct=actual==expected;assert correct==(i<3)
  differences=[]
  for k in ['left.json','right.json']:
   for f in ['nonce','sum']:
    if actual[k][f]!=expected[k][f]:differences.append(dict(field=k+'.'+f,expected=expected[k][f]if type(expected[k][f])is int else sha(str(expected[k][f]).encode()),actual=actual[k][f]if type(actual[k][f])is int else sha(str(actual[k][f]).encode()),value_scope='integer scalar or SHA of opaque string'))
  if actual['total']!=expected['total']:differences.append(dict(field='total',expected=expected['total'],actual=actual['total']))
  h=r['admission']['history_render_audit'];assert h['qualified']and h['prequeue_bound']and h['same_prepared_encoder_reproduces_prompt_and_IDs']and not h['markers_sent_to_Native']and h['content_one_contribution_each_and_ordered']and h['all_content_contribution_slices_equal']
  assert h['prompt_tokens']==r['prompt_tokens']==8489 and h['Native_prompt_token_ids_SHA256']=='681977ee292994eac340bae5c1aadf7a6b136890df725d02bb75fc8f31ecf0df'and h['output_budget']==r['max_new_tokens']==128 and not h['pending_thinking_prefix_open']
  assert r['admission']['text_delivery']=='P1'and r['final_metadata']['buffered_P0_text']is False
  ids=r['generated_token_ids'];assert len(ids)==r['actual_generated_tokens']==wire['usage']['completion_tokens']==98 and ids[-1]==1 and ids.count(1)==1 and all(type(t)is int and 0<=t<vocab for t in ids)
  allids.append(ids);assert r['request_id']==wire['id']and wire['usage']['prompt_tokens']==8489 and wire['usage']['total_tokens']==8489+len(ids)
  assert r['normal_reset_before']and r['normal_reset_after']and r['native_finish_reason']=='eos_token'and not r.get('error')and not r.get('reset_error')
  assert r['actual_forward_attempts']==r['actual_forward_completed']==106;calls+=106
  pre=[v for v in r['Native_continuation']if v['kind']=='prefill'];ds=[v for v in r['Native_continuation']if v['kind']=='decode_summary']
  assert len(pre)==9 and [v['tokens']for v in pre]==[1024]*8+[297]and all(v['past']==1024*j and v['total']==v['past']+v['tokens']and v['emitted']==(j==8)for j,v in enumerate(pre))
  assert ds==[dict(kind='decode_summary',calls=97,first_past=8489,last_total=8586)]
  coverage=owner_coverage(r,i)
  ticks=r['backend_token_monotonic'];assert len(ticks)==98 and all(math.isfinite(t)for t in ticks)and all(b>a for a,b in zip(ticks,ticks[1:]))and r['started_monotonic']<=ticks[0]<=ticks[-1]<=r['ended_monotonic']
  ns=r['Native_forward_host_summary'];assert ns['prefill_calls']==9 and ns['decode_calls']==97 and all(math.isfinite(ns[k])and ns[k]>0 for k in ['prefill_s','decode_s'])
  if i<3:
   c=client['rows'][i];assert c['r438_index']==i and c['mode']==r['r438_mode']and c['r438_T1_coverage']==coverage and c['actual_owner_counts']==r['r438_actual_owner_counts']and c['actual_request_SHA256']==sha(raw)and c['wire_SHA256']==sha(wire_raw);clock=clock_metrics(c['client_clock'],wire_raw)
  else:
   c=client['current_attempt'];assert c['index']==3 and c['mode']==1 and c['request_SHA256']==sha(raw)and c['wire_SHA256']==sha(wire_raw)
   events=c['events'];assert events and c['sent']<=c['headers']<=c['completed']and all(c['headers']<=v['arrival_monotonic']<=c['completed']for v in events)
   arrivals=[v['arrival_monotonic']for v in events];assert arrivals==sorted(arrivals)
   data_lines=[v for v in wire_raw.splitlines(keepends=True)if v.startswith(b'data: ')];assert len(data_lines)==len(events)
   for line,event in zip(data_lines,events):
    flags=dict(reasoning_nonempty=False,content_nonempty=False,tool_nonempty=False);value=line[6:].strip()
    if value!=b'[DONE]':
     part=json.loads(value)
     for name,key in [('reasoning_nonempty','reasoning_content'),('content_nonempty','content'),('tool_nonempty','tool_calls')]:flags[name]=any(bool(v.get('delta',{}).get(key))for v in part.get('choices',[]))
    flags['nonempty_delta']=any(flags.values());flags['bytes']=len(line);assert all(event[k]==v for k,v in flags.items())
   clock=dict(wall_s=c['completed']-c['sent'],headers_s=c['headers']-c['sent'],first_payload_delta_s=next(v['arrival_monotonic']-c['sent']for v in events if v['nonempty_delta']),terminal_quality_rejected=True)
  rows.append(dict(index=i,mode=r['r438_mode'],warm=i<2,Native_forwards=106,generated_including_EOS=98,natural_EOS=True,semantic_gold=correct,answer_differences=differences,nonce_fields_both_equal=all(actual[k]['nonce']==expected[k]['nonce']for k in ['left.json','right.json']),request_SHA256=sha(raw),wire_SHA256=sha(wire_raw),generated_IDs_SHA256=sha(compact(ids)),actual_T1_coverage=coverage,per_owner_routed_shared_hits=[[v['routed_hits'],v['shared_hits']]for v in r['r438_actual_owner_counts']],normal_request_reset=True,Native_forward_host_summary=ns,backend_visible_decode_tok_s=96/(ticks[96]-ticks[0]),client_clock=clock,presentation=presentation))
 assert len(set(request_hashes))==1 and owner['HC_dispatch_checked_forwards']==owner['layout_dispatch_checked_forwards']==calls==424
 assert not any(n.endswith('/request-%d.request.json'%i)or n.endswith('/response-%d.wire'%i)for n in files for i in [4,5])
 comparisons=[]
 for i,j in [(0,2),(1,3),(0,1),(2,3),(0,3),(1,2)]:
  a,b=allids[i],allids[j];first=next((dict(output_index=k,IDs=[x,y])for k,(x,y)in enumerate(zip(a,b))if x!=y),None)
  comparisons.append(dict(pair=[i,j],mode_pair=[rows[i]['mode'],rows[j]['mode']],IDs_equal=a==b,first_different=first))
 assert allids[0]==allids[1]==allids[2]and allids[1]!=allids[3]
 post=watch['post_after_failure'];assert len(post)==1 and post[0]['actual_step']=='910374.1376'and not post[0]['kfd_pids']and not post[0]['matching_component_processes']and not post[0]['expected_PIDs_still_exist']and post[0]['ports_bindable']=={'43882':True}and len(post[0]['physical_PCI'])==8
 accounts=archive['accounting'];by={r[0]:r for r in accounts};assert by['910374.1375'][2:4]==['FAILED','1:0']and by['910407.906'][2:4]==['FAILED','1:0']and by['910374.1374'][2].startswith('CANCELLED')and by['910374.1376'][2:4]==['COMPLETED','0:0']
 return dict(passed=True,audit_scope='Independent verification of original Model quality FAIL, not execution PASS',execution_passed=False,Model_qualified=False,performance_qualified=False,actual_GPU=watch['actual_GPU'],CPU_keeper=watch['CPU_keeper'],actual_client=watch['actual_client'],post_step='910374.1376',publication=watch['publication_commit'],source_manifest=watch['deployment_source_SHA256'],original_GPU_epoch=watch['original_GPU_start_unix'],verified_raw_files=verified,planned_HTTP=6,actual_HTTP=4,correct_HTTP=3,incorrect_HTTP=1,unexecuted_HTTP=[4,5],Native_CAPI_forwards=424,full_ABBA_completed=False,Native_speedup_claim=False,normal_owner_destroy_qualified=False,forced_cleanup=True,owner_report_original_completed=False,normal_shutdown_stamp=False,post_clean=True,loading_seconds=owner['loading_seconds'],requests=rows,generated_ID_comparisons=comparisons,same_actual_Native_input_request=True,actual_prompt_IDs_SHA256='681977ee292994eac340bae5c1aadf7a6b136890df725d02bb75fc8f31ecf0df',same_provider_source_and_Native_reported=True,SDK_binding_scope='Frozen fa63 providers() validation executed before weights; this offline reader does not independently remap the terminated process',actual_new_Native_SHA256=owner['Native_SHA256'],all_modes_and_8owner_hits_verified=True,finite_scope='Original every-forward BF16 finite checks completed; no independent actual-activation numerical oracle or logits captured',cause='UNKNOWN; same-mode ON repeat diverged at output83, without logits/margin; no fusion/state/parser attribution from this artifact alone',decision='STOP_NO_PROMOTION_NO_RELOAD',component1371='Separate synthetichidden component gain retained, not Model qualification',timing_scope='Per-completed-request Native host enclosures/emitter/client clocks retained descriptively; incomplete ABBA and quality FAIL exclude Model gain')

def main():
 p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('--output',type=Path);a=p.parse_args();raw=a.archive.read_bytes();q=json.loads(raw);wrapper_sha=sha(raw)
 if 'sealed_archive'in q:
  seal=q['sealed_archive'];raw=seal['text'].encode();assert len(raw)==seal['bytes']and sha(raw)==seal['SHA256'];q=json.loads(raw)
 out=analyze(q);out['archive_SHA256']=sha(raw);out['local_archive_wrapper_SHA256']=wrapper_sha;result=json.dumps(out,indent=2)+'\n'
 if a.output:a.output.write_text(result,encoding='utf-8')
 print(result,end='')
if __name__=='__main__':main()
