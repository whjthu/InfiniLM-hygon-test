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
 assert computed['first_payload_delta_s']is not None and computed['first_reasoning_payload_s']is not None and computed['first_content_payload_s']is not None
 assert clock['client_per_token_speed'].startswith('UNKNOWN')
 return dict(**computed,data_frames=len(rows),client_per_token_speed='UNKNOWN; delta/frame is not a token')

def paired(rows):
 formal=rows[2:];assert [r['delivery']for r in formal]==['P0','P1','P1','P0']
 result={}
 for key in ['first_payload_delta_s','first_reasoning_payload_s','first_content_payload_s','wall_s','max_inter_data_frame_gap_s','max_inter_semantic_delta_gap_s']:
  xs=[r['client_clock_independently_checked'][key]for r in rows]
  if any(x is None for x in xs[2:]):result[key]=dict(status='UNKNOWN_missing_channel_or_interval');continue
  result[key]=dict(formal_values=xs[2:],P0_endpoints=[xs[2],xs[5]],P1_middle=[xs[3],xs[4]],adjacent_P1_over_P0=[xs[3]/xs[2]if xs[2]else None,xs[4]/xs[5]if xs[5]else None],P0_endpoint_signed_drift_fraction=xs[5]/xs[2]-1 if xs[2]else None)
 result['post_first_semantic_gap_interpretation']='P0 emits two semantic bursts at completion; its tiny inter-burst gap is not token ITL or responsiveness. Preserve absolute gaps, do not infer a streaming SLO from their ratio.'
 result['same_generated_IDs_across_formal']=len({r['generated_IDs_SHA256']for r in formal})==1
 result['same_visible_reasoning_across_formal']=len({r['reasoning_text_SHA256']for r in formal})==1
 result['same_visible_content_across_formal']=len({r['content_text_SHA256']for r in formal})==1
 result['warmups_excluded']=True;result['not_Native_throughput_or_token_SLO']=True
 result['scoped_semantic_first_payload_both_adjacent_pairs_improved']=all(x is not None and x<1 for x in result['first_payload_delta_s']['adjacent_P1_over_P0'])
 result['P1_semantic_payload_before_own_completion']=all(r['client_clock_independently_checked']['first_payload_delta_s']<r['client_clock_independently_checked']['wall_s']for r in formal if r['delivery']=='P1')
 return result

def analyze(archive):
 files=archive['files'];verified=0
 def raw_by(suffix):
  values=[v for n,v in files.items()if n.endswith(suffix)];assert len(values)==1,(suffix,len(values));v=values[0]
  raw=v['text'].encode()if v.get('encoding','text')=='text'else __import__('base64').b64decode(v['base64']);assert len(raw)==v['bytes']and sha(raw)==v['SHA256'];return raw
 for v in files.values():
  raw=v['text'].encode()if v.get('encoding','text')=='text'else __import__('base64').b64decode(v['base64']);assert len(raw)==v['bytes']and sha(raw)==v['SHA256'];verified+=1
 read=lambda suffix:json.loads(raw_by(suffix));watch=read('-watch.json');assert watch['completed']and watch['passed']and not watch['forced_cleanup']
 assert watch['publication_commit']=='12bbcace64912d62c8511408b1bfcbbced883cc7'and watch['deployment_source_SHA256']=='a0c051693f966fcc5ce96b8d2b4e16293cc399b9d6fb82871f4034df475df692'
 front,ready,closed,owner=[read('/'+n)for n in ['FRONTEND.json','MODEL_READY.json','closed-engine.json','owner.json']]
 model_raw=raw_by('/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8/config.json');assert sha(model_raw)=='fd2ad6f5dbd77e55a49e9b2ff29c89030fbe4b538e8870c183d194648767be09';model_cfg=json.loads(model_raw);vocab=model_cfg['vocab_size'];assert type(vocab)is int and vocab>0
 gen_raw=raw_by('/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8/generation_config.json');assert sha(gen_raw)=='5fccff80f55a4d455bbe516bdd552edf3e9623df95e99fbf2a3c3389fdf91af0';gen_cfg=json.loads(gen_raw);eos=gen_cfg['eos_token_id'];eos=eos if type(eos)is list else[eos];assert eos==[1]
 prepared=read('/PREPARED.json');assert prepared['prepared_identity']['eos_token_ids']==eos and prepared['prepared_identity']['generation_config_SHA256']==sha(gen_raw)
 identity=front['identity'];assert all(q['identity']==identity for q in [ready,closed,owner]);assert identity['actual_step']==watch['actual_GPU']and identity['source_manifest_SHA256']=='4ec029919491a59987dce840975898727149209749f9de322da9f3af73171a91'
 assert ready['Model_READY']and ready['Native_internal_owner_count']==8 and ready['HC_mode']==0
 assert owner['loaded_key_count']==67569 and owner['Native_SHA256']=='afc6f58b71ef7a51657deae452da86a31afbfa6e79d36d442ccdd26d56c0e6f2'
 assert len(owner['GPU_memory_after_create'])==len(owner['GPU_memory_after_load'])==8 and all(v[0]>8*1024**3 for v in owner['GPU_memory_after_create'])and all(v[0]>=16*1024**3 for v in owner['GPU_memory_after_load'])
 assert owner['completed']and owner['normal_Native_shutdown']and owner['engine_worker_shutdown_completed']and owner['observed_all8_threads_exited']
 assert closed['normal']and closed['engine_destroy_returned_zero']and closed['observed_all8_threads_exited']
 assert not watch.get('cleanup_errors')and not watch.get('STOP_error')and not watch.get('owned_cleanup')and not any(n.endswith('/FAILED.json')for n in files)
 assert owner['owner_thread_observation']['status']=='OBSERVED'and len({v['TID']for v in owner['owner_thread_observation']['owners']})==8
 aff=owner['owner_affinity_after_weight_load']['actual_owner_report']['owners'];assert {v['device']for v in aff}==set(range(8))
 masks=[int(n)for v in aff for n in v['verified_mask'].split(',')];assert len(masks)==len(set(masks))==32 and all(v['changes']==1 for v in aff)
 assert set(owner['actual_providers_before_weights'].values())=={'/refactor/candidates/latest-family-native-r269/Deep/libDeep_latest_native_r269.so'}
 client=read('/result.json');assert client['completed']and client['passed']and client['planned_HTTP']==client['attempted_HTTP']==client['completed_HTTP']==6 and len(client['rows'])==6 and client['current_attempt']is None and not client.get('failed_attempt')and not client.get('error')
 fixture=read('/thinking-request.json');assert sha(raw_by('/thinking-request.json'))=='fe1cb3636cd70504f2aff286239d8d99d9add0a5dc87480c320e12e841275076';actual=[];bodies=[];calls=0;deliveries=['P0','P1','P0','P1','P1','P0']
 for i,delivery in enumerate(deliveries):
  thinking,budget=True,8192
  body=read('/request-%d.request.json'%i);bodies.append(body);expected_body=dict(fixture,text_delivery=delivery);expected_body['stream_options']={'include_usage':True};assert body==expected_body,'Only text_delivery and exact include_usage stream_options deployment assignments are allowed; all other fixture fields identical'
  assert body['text_delivery']==delivery and body['stream']==thinking and body['max_tokens']==budget and body['chat_template_kwargs']=={'enable_thinking':thinking}and body['tool_choice']=='none'
  wire_raw=raw_by('/response-%d.wire'%i);wire=sse(wire_raw)if thinking else json.loads(wire_raw);choice=wire['choices'][0];assert choice['finish_reason']=='stop'and not choice['message'].get('tool_calls')
  result,presentation=parsed_final(choice['message'].get('content')or'');expected=gold(body);assert result==expected
  assert type(result['total'])is int and all(type(result[k]['sum'])is int for k in ['left.json','right.json'])
  if thinking:assert choice['message'].get('reasoning_content')
  r=owner['requests'][i];c=client['rows'][i];assert c['index']==i and c['delivery']==delivery and c['warmup']==(i<2) and r['admission']['text_delivery']==delivery and r['final_metadata']['buffered_P0_text']==(delivery=='P0');assert wire['id']==r['request_id']==c['request_id']and sha(wire_raw)==c['wire_SHA256']and sha(raw_by('/request-%d.request.json'%i))==c['actual_request_SHA256']
  h=r['admission']['history_render_audit'];assert h['prompt_tokens']==8490 and h['Native_prompt_token_ids_SHA256']=='d956ecfc84f1c221a7768772b868a237d02fb1e6df2b682291b9a58c9ce2e287';assert h['qualified']and h['prequeue_bound']and not h['markers_sent_to_Native']and h['same_prepared_encoder_reproduces_prompt_and_IDs']and h['content_one_contribution_each_and_ordered']and h['all_content_contribution_slices_equal']
  assert h['output_budget']==r['max_new_tokens']==budget and bool(h['pending_thinking_prefix_open'])==thinking
  ids=r['generated_token_ids'];usage=wire['usage'];assert ids and all(type(token)is int and 0<=token<vocab for token in ids)and ids[-1]in eos and not any(token in eos for token in ids[:-1]);assert ids[-1]==1 and len(ids)==usage['completion_tokens']==r['actual_generated_tokens']<=budget
  assert r['prompt_tokens']==usage['prompt_tokens']==h['prompt_tokens']and usage['total_tokens']==usage['prompt_tokens']+usage['completion_tokens']
  assert r['normal_reset_before']and r['normal_reset_after']and r['native_finish_reason']=='eos_token'and not r.get('error')
  assert r['actual_forward_attempts']==r['actual_forward_completed']==9+len(ids)-1;calls+=r['actual_forward_completed']
  clocks=r['backend_token_monotonic'];assert len(clocks)==len(ids)and all(math.isfinite(v)for v in clocks)and all(b>a for a,b in zip(clocks,clocks[1:]))and r['started_monotonic']<=clocks[0]<=clocks[-1]<=r['ended_monotonic']
  pre=[v for v in r['Native_continuation']if v['kind']=='prefill'];assert pre[0]['past']==0 and pre[-1]['total']==r['prompt_tokens']and all(v['tokens']==1024 for v in pre[:-1])and sum(v['tokens']for v in pre)==r['prompt_tokens']
  assert len(pre)==9 and all(v['past']==1024*j and v['total']==v['past']+v['tokens']and v['emitted']==(j==8)for j,v in enumerate(pre))
  ds=[v for v in r['Native_continuation']if v['kind']=='decode_summary'];assert len(ds)==1 and ds[0]==dict(kind='decode_summary',calls=len(ids)-1,first_past=8490,last_total=8490+len(ids)-1)
  ns=r['Native_forward_host_summary'];assert ns['prefill_calls']==9 and ns['decode_calls']==len(ids)-1 and all(math.isfinite(ns[k])and ns[k]>0 for k in ['prefill_s','decode_s'])
  assert len(r['GPU_free_after_request'])==8 and all(v[0]>=16*1024**3 for v in r['GPU_free_after_request'])
  visible=[t for token,t in zip(ids,r['backend_token_monotonic'])if token!=1];itls=[b-a for a,b in zip(visible,visible[1:])]
  clock=clock_metrics(c['client_clock'],wire_raw);
  actual.append(dict(index=i,delivery=delivery,warmup=i<2,thinking=thinking,budget=budget,prompt_tokens=r['prompt_tokens'],generated_tokens=len(ids),visible_tokens=len(visible),natural_EOS=True,task_correct=True,presentation=presentation,independent_gold_SHA256=sha(compact(expected)),actual_prompt_IDs_SHA256=h['Native_prompt_token_ids_SHA256'],Native_forward_host_summary=r['Native_forward_host_summary'],backend_TTFT_s=r['backend_token_monotonic'][0]-r['started_monotonic'],backend_visible_decode_tok_s=(len(visible)-1)/(visible[-1]-visible[0])if len(visible)>1 else None,backend_ITL_p95_s=percentile(itls,.95),backend_ITL_p99_s=percentile(itls,.99),client_HTTP_wall_s=c['client_clock']['wall_s'],client_first_payload_delta_s=c['client_clock']['first_payload_delta_s'],transport_text_buffered=delivery=='P0',client_clock_independently_checked=clock,generated_IDs_SHA256=sha(compact(ids)),reasoning_text_SHA256=sha(choice['message'].get('reasoning_content','').encode()),content_text_SHA256=sha(choice['message'].get('content','').encode())))
 assert len(owner['requests'])==6 and owner['HC_dispatch_checked_forwards']==owner['layout_dispatch_checked_forwards']==calls
 post=watch['post'];assert len(post)==1 and post[0]['node']=='a02r4n01'and not post[0]['kfd_pids']and not post[0]['expected_PIDs_still_exist']and not post[0]['matching_component_processes']and post[0]['ports_bindable']=={'43882':True}
 accounts=archive['accounting'];required=[watch['actual_GPU'],watch['CPU_keeper'],watch['actual_client'],post[0]['actual_step']]
 for step in required:
  rows=[r for r in accounts if r[0]==step];assert len(rows)==1 and rows[0][2:4]==['COMPLETED','0:0']
 return dict(passed=True,scope='Same complete thinking SSE8490/8192, P0P1 warm then P0P1P1P0 formal; one qualified HC0 afc6 engine; transport comparison only',actual_GPU=watch['actual_GPU'],CPU_keeper=watch['CPU_keeper'],actual_client=watch['actual_client'],original_GPU_epoch=watch['original_GPU_start_unix'],publication=watch['publication_commit'],verified_raw_files=verified,actual_HTTP=6,Native_CAPI_forwards=calls,all8internalowners_normal=True,post_clean=True,no_known_pending_requests=True,actual_vocab_size=vocab,actual_EOS_ids=eos,all_generated_IDs_integer_and_in_vocab=True,backend_emitter_strict_positive_intervals=True,loading_seconds=owner['loading_seconds'],requests=actual,finite_scope='Original producer finite BF16 checks ran for every completed forward; no independent activation oracle in this public test',performance_comparison=False,fullbenchmark=False,incremental_client_token_SLO_qualified=False,formal_transport_comparison=paired(actual),client_token_speed='UNKNOWN; frame/delta is not a token',Native_speedup_claim=False,quality_scope='Six frozen task repetitions only, not fullbenchmark',cost_boundaries=dict(startup='Qualified Native create/load in one eight-card run',loading_s=owner['loading_seconds'],actualmeasurement='Serial complete HTTP/Native requests; warmups excluded from formal ABBA',GPU_allocation_elapsed='Accounting elapsed includes setup/loading/HTTP/cleanup and is not GPUbusy'))
def main():
 p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('--output',type=Path);a=p.parse_args();raw=a.archive.read_bytes();value=json.loads(raw);digest=sha(raw)
 if 'sealed_archive'in value:
  seal=value['sealed_archive'];original=seal['text'].encode();assert len(original)==seal['bytes']and sha(original)==seal['SHA256'];value=json.loads(original);digest=seal['SHA256']
 q=analyze(value);q['archive_SHA256']=digest;q['local_archive_wrapper_SHA256']=sha(raw)
 result=json.dumps(q,indent=2)+'\n'
 if a.output:a.output.write_text(result,encoding='utf8')
 print(result,end='')
if __name__=='__main__':main()
